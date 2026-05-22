"""Lambda entry point: SQS → parse EventBridge envelope → enqueue DBOS workflows.

Replaces the prior synchronous HTTP forward to Railway ``/process`` with two
``DBOSClient.enqueue`` calls (one per pipeline). Durability ownership transfers
from SQS to DBOS at the moment Lambda returns 200.

Architecture lock: D1 (Lambda + DBOSClient) per execution plan §86. Workflow ID
format per design doc §3: ``f"action-item-graph:{pipeline}:interaction-{uuid}"``.
DBOS deduplicates same-id re-enqueues silently (recovery_attempts++), so SQS
redelivery after partial-enqueue is safe: the first call no-ops on the existing
row, the second retries.

Uses AWS Lambda Powertools for structured logging, tracing, and batch processing.
"""

from typing import Any

import boto3
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.batch import (
    BatchProcessor,
    EventType,
    process_partial_response,
)
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord
from botocore.exceptions import NoCredentialsError
from dbos import DBOSClient, EnqueueOptions

from .config import LambdaConfig
from .envelope import parse_sqs_record_body
from .secrets import get_dbos_system_database_url

# Module-level singletons — survive across warm Lambda invocations
processor = BatchProcessor(event_type=EventType.SQS, raise_on_entire_batch_failure=False)
logger = Logger(service="action-item-graph-ingest", log_uncaught_exceptions=True)
tracer = Tracer(service="action-item-graph-ingest")

_config: LambdaConfig | None = None
_dbos_client: DBOSClient | None = None
_cloudwatch_client: Any | None = None

# DBOS workflow registration names — resolved via __qualname__ in the worker
# (see ``dbos/_core.py:1274``). Both workflow functions are module-level so
# __qualname__ equals the function name. If the workflow file ever moves
# inside a class or gets renamed, update both sides in lockstep.
_ACTION_ITEM_WORKFLOW_NAME = "action_item_workflow"
_DEAL_WORKFLOW_NAME = "deal_workflow"

_ACTION_ITEM_QUEUE_NAME = "action-item-pipeline"
_DEAL_QUEUE_NAME = "deal-pipeline"

# 15-minute safety net per Open #19 — matches plan's workflow_timeout=900.
_WORKFLOW_TIMEOUT_SECONDS = 900.0

# CloudWatch metric for split-brain detection per execution plan Codex #14/15.
# Emitted on first-succeeds-second-fails. Alarm threshold defined in Pulumi.
_METRIC_NAMESPACE = "ActionItemGraph/Lambda"
_METRIC_PARTIAL_ENQUEUE = "partial_enqueue_pair_count"


def _get_config() -> LambdaConfig:
    """Lazy-init config singleton.

    Fetches DBOS_SYSTEM_DATABASE_URL from Secrets Manager on first call
    (cold start). The secret is cached for subsequent warm invocations.
    """
    global _config
    if _config is None:
        _config = LambdaConfig()
        _config.DBOS_SYSTEM_DATABASE_URL = get_dbos_system_database_url()
    return _config


def _get_dbos_client() -> DBOSClient:
    """Lazy-init DBOSClient singleton.

    Constructed once per Lambda cold start. The Neon connection is opened
    lazily on first ``enqueue`` call via SQLAlchemy's engine — DBOSClient's
    ``__init__`` only stores the URL and constructs the engine; it does not
    open a TCP connection. ``pool_pre_ping=True`` (from ``_client.py:182``)
    validates the connection on first checkout.
    """
    global _dbos_client
    if _dbos_client is None:
        config = _get_config()
        _dbos_client = DBOSClient(system_database_url=config.DBOS_SYSTEM_DATABASE_URL)
    return _dbos_client


def _get_cloudwatch_client() -> Any | None:
    """Lazy-init CloudWatch boto3 client.

    Mirrors LTF's graceful-degradation pattern at
    ``services/account_provisioning/eventbridge_emit.py:158-189``: if AWS
    credentials are unavailable (local dev, CI), return None and skip
    metric emission. The dispatcher's primary path is unaffected.
    """
    global _cloudwatch_client
    if _cloudwatch_client is None:
        try:
            _cloudwatch_client = boto3.client("cloudwatch")
        except NoCredentialsError:
            logger.warning(
                "cloudwatch_client.unavailable",
                extra={"reason": "missing AWS credentials"},
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cloudwatch_client.init_failed",
                extra={"error": str(exc)},
            )
            return None
    return _cloudwatch_client


def _emit_partial_enqueue_metric() -> None:
    """Emit ``partial_enqueue_pair_count = 1`` to CloudWatch.

    Best-effort: metric emission failure does NOT propagate to the
    dispatcher. The dispatcher's downstream behavior (re-raise so SQS
    redelivers) is the load-bearing recovery mechanism; the metric is
    just observability.
    """
    client = _get_cloudwatch_client()
    if client is None:
        return
    try:
        client.put_metric_data(
            Namespace=_METRIC_NAMESPACE,
            MetricData=[
                {
                    "MetricName": _METRIC_PARTIAL_ENQUEUE,
                    "Value": 1.0,
                    "Unit": "Count",
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "cloudwatch_metric.emit_failed",
            extra={"metric": _METRIC_PARTIAL_ENQUEUE, "error": str(exc)},
        )


def _build_enqueue_options(
    *, workflow_name: str, queue_name: str, workflow_id: str
) -> EnqueueOptions:
    """Build the EnqueueOptions dict for a DBOSClient.enqueue call.

    EnqueueOptions is a TypedDict, not a class — construct as a literal.
    workflow_timeout in seconds (DBOS converts to ms internally at
    ``_client.py:249``).
    """
    return {
        "workflow_name": workflow_name,
        "queue_name": queue_name,
        "workflow_id": workflow_id,
        "workflow_timeout": _WORKFLOW_TIMEOUT_SECONDS,
    }


@tracer.capture_method
def process_record(record: SQSRecord) -> None:
    """Process a single SQS record by enqueueing both pipeline workflows.

    Per Rule 1 (validation lives outside any retryable layer): if the
    envelope is missing required fields, raise immediately. SQS handles
    retry semantics; we don't wrap validation in any retry logic.

    Per Rule 5 (downstream-produced identifier authority): the Lambda
    uses ``envelope.interaction_id`` ONLY for the coordination workflow_id.
    Authoritative data identifiers (post-extraction interaction_id, etc.)
    are derived by the workflow itself per Phase B-1/B-2 — not here.

    Per Rule 7 (verify cross-service references match deployment topology):
    LTF's ``enqueue_async`` is an in-process pattern (route inside the
    FastAPI/DBOS runtime). AIG's Lambda is external to that runtime, so
    it uses ``DBOSClient.enqueue`` per the design doc + DBOS SDK. The
    LTF reference is structurally valuable (decorator patterns, cold-start
    fail-loud) but not for the dispatcher call shape itself.
    """
    # Force the config + client through their lazy-init paths so a missing
    # DBOS_SYSTEM_DATABASE_URL surfaces as a clear cold-start failure
    # instead of a confusing failure mid-enqueue.
    _get_config()
    client = _get_dbos_client()

    # Parse EventBridge wrapper → extract EnvelopeV1 JSON
    envelope_json = parse_sqs_record_body(record.body)

    tenant_id = envelope_json.get("tenant_id", "unknown")
    interaction_type = envelope_json.get("interaction_type", "unknown")
    interaction_id = envelope_json.get("interaction_id")

    if not interaction_id:
        # Fail-fast (Rule 1): an envelope without interaction_id can't be
        # given a deterministic workflow_id, which means DBOS dedup can't
        # protect against retry duplicates. SQS handles redelivery → DLQ.
        raise ValueError(
            "Envelope missing interaction_id; cannot construct workflow_id. "
            f"Envelope keys: {list(envelope_json.keys())}"
        )

    logger.info(
        "record.processing",
        extra={
            "tenant_id": tenant_id,
            "interaction_type": interaction_type,
            "interaction_id": interaction_id,
            "message_id": record.message_id,
        },
    )

    ai_workflow_id = f"action-item-graph:action-item:interaction-{interaction_id}"
    deal_workflow_id = f"action-item-graph:deal:interaction-{interaction_id}"

    ai_options = _build_enqueue_options(
        workflow_name=_ACTION_ITEM_WORKFLOW_NAME,
        queue_name=_ACTION_ITEM_QUEUE_NAME,
        workflow_id=ai_workflow_id,
    )
    deal_options = _build_enqueue_options(
        workflow_name=_DEAL_WORKFLOW_NAME,
        queue_name=_DEAL_QUEUE_NAME,
        workflow_id=deal_workflow_id,
    )

    # First enqueue. If this fails, no partial state — SQS retries the whole
    # message. DBOS silently dedupes on redelivery (same workflow_id +
    # workflow_name → recovery_attempts++, no exception per
    # ``_sys_db.py:680-696``); any real exception (transient DB / network)
    # propagates to SQS for retry.
    client.enqueue(ai_options, envelope_json)

    # Second enqueue. If this fails after the first succeeded, we have a
    # partial-enqueue split window: action-item workflow row exists, deal
    # does not. Emit the CloudWatch metric so the operator alarm fires,
    # then re-raise so SQS redelivers. On redelivery the action-item enqueue
    # is a silent no-op (DBOS dedup), and the deal enqueue retries.
    try:
        client.enqueue(deal_options, envelope_json)
    except Exception:
        _emit_partial_enqueue_metric()
        logger.error(
            "record.partial_enqueue",
            extra={
                "tenant_id": tenant_id,
                "interaction_id": interaction_id,
                "ai_workflow_id": ai_workflow_id,
                "deal_workflow_id": deal_workflow_id,
                "message_id": record.message_id,
            },
        )
        raise

    logger.info(
        "record.enqueued",
        extra={
            "tenant_id": tenant_id,
            "interaction_type": interaction_type,
            "interaction_id": interaction_id,
            "ai_workflow_id": ai_workflow_id,
            "deal_workflow_id": deal_workflow_id,
            "message_id": record.message_id,
        },
    )


@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
def lambda_handler(event: dict, context) -> Any:
    """Lambda entry point — processes SQS batch with partial failure reporting.

    Return type is ``PartialItemFailureResponse`` (an ``aws_lambda_powertools``
    TypedDict). Annotated as ``Any`` here so pyright doesn't insist on the
    library type at this seam; the AWS Lambda runtime accepts any
    JSON-serializable mapping.
    """
    return process_partial_response(
        event=event,
        record_handler=process_record,
        processor=processor,
        context=context,
    )
