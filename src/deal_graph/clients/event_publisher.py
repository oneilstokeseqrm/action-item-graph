"""EventBridge publisher for ``deal.processed`` events.

Emits a single ``deal.processed`` event to the AWS EventBridge default bus
(us-east-1, account 211125681610) per successful deal-workflow run when
``ENABLE_DEAL_PROCESSED_EVENTS=true``. The event flows through:

    EventBridge default bus
      ─[ source=com.eq.action-item-graph, detail-type=deal.processed ]─
        → SQS eq-opportunity-queue-dev
          → Lambda eq-opportunity-themes-ingest
            → thematic-lm POST /analyze (scope_type=opportunity)
              → Neon analyses row + opportunity codebook

Failure mode: never raises. AWS errors, missing credentials, network
timeouts, etc. log a structlog warning and return None. The deal
workflow is unaffected by publish failures.

Wire contract source of truth:
    thematic-lm/src/thematic_lm/opportunity/event_models.py
    (DealProcessedEvent.from_sqs_body)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3
import structlog
from botocore.config import Config

logger = structlog.get_logger(__name__)


_SOURCE = "com.eq.action-item-graph"
_DETAIL_TYPE = "deal.processed"
_EVENT_BUS_NAME = "default"

# Tight botocore config. ``retries_allowed=False`` on the DBOS step only
# disables DBOS-layer retries — boto3/botocore still retry inside the SDK
# (legacy mode: up to 4 attempts, ~20s backoff cap). Without this Config
# a single put_events call against an unreachable endpoint could stall the
# fail-open step ~30s, undermining its purpose. Worst case here: ~7s
# (2s connect + 5s read, one attempt).
_BOTO_CONFIG = Config(
    connect_timeout=2,
    read_timeout=5,
    retries={"max_attempts": 1, "mode": "standard"},
)


def _events_enabled() -> bool:
    return os.getenv("ENABLE_DEAL_PROCESSED_EVENTS", "false").lower() == "true"


def publish_deal_processed(
    *,
    tenant_id: str,
    account_id: str,
    interaction_id: str,
    deals_created: list[str],
    deals_merged: list[str],
    source: str = "deal-pipeline",
    timestamp: str | None = None,
    workflow_id: str | None = None,
) -> None:
    """Publish a deal.processed event to EventBridge. Never raises.

    Short-circuits when the feature flag is off, when interaction_id is
    empty (consumer parser requires it), or when both deal lists are
    empty (consumer would skip via its ``all_deal_ids`` guard).
    """
    if not _events_enabled():
        return
    if not interaction_id:
        logger.warning(
            "event_publisher.skipped_no_interaction_id",
            workflow_id=workflow_id,
        )
        return
    if not deals_created and not deals_merged:
        return

    log = logger.bind(
        interaction_id=interaction_id,
        workflow_id=workflow_id,
    )

    try:
        client = boto3.client(
            "events",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            config=_BOTO_CONFIG,
        )
        detail = {
            "tenant_id": tenant_id,
            "account_id": account_id,
            "interaction_id": interaction_id,
            "deals_created": deals_created,
            "deals_merged": deals_merged,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        response = client.put_events(
            Entries=[
                {
                    "Source": _SOURCE,
                    "DetailType": _DETAIL_TYPE,
                    "EventBusName": _EVENT_BUS_NAME,
                    "Detail": json.dumps(detail),
                }
            ]
        )
        failed = response.get("FailedEntryCount", 0)
        entries = response.get("Entries", [])
        event_id = entries[0].get("EventId") if entries else None
        if failed:
            log.warning(
                "event_publisher.put_events_partial_failure",
                failed_count=failed,
                entries=entries,
            )
        else:
            log.info(
                "event_publisher.published",
                event_id=event_id,
                deals_created=len(deals_created),
                deals_merged=len(deals_merged),
            )
    except Exception as e:
        log.warning(
            "event_publisher.failed",
            error=str(e),
            error_type=type(e).__name__,
        )
