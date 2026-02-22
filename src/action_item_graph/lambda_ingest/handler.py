"""Lambda entry point: SQS → parse EventBridge envelope → POST to Railway API.

Uses AWS Lambda Powertools for structured logging, tracing, and batch processing.
"""

from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.batch import (
    BatchProcessor,
    EventType,
    process_partial_response,
)
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord

from .config import LambdaConfig
from .envelope import parse_sqs_record_body
from .api_client import submit_to_railway

# Module-level singletons — survive across warm Lambda invocations
processor = BatchProcessor(event_type=EventType.SQS, raise_on_entire_batch_failure=False)
logger = Logger(service="action-item-graph-ingest", log_uncaught_exceptions=True)
tracer = Tracer(service="action-item-graph-ingest")

_config: LambdaConfig | None = None


def _get_config() -> LambdaConfig:
    """Lazy-init config singleton."""
    global _config
    if _config is None:
        _config = LambdaConfig()
    return _config


@tracer.capture_method
def process_record(record: SQSRecord) -> None:
    """Process a single SQS record containing an EventBridge-wrapped envelope."""
    config = _get_config()

    # Parse EventBridge wrapper → extract EnvelopeV1 JSON
    envelope_json = parse_sqs_record_body(record.body)

    tenant_id = envelope_json.get("tenant_id", "unknown")
    interaction_type = envelope_json.get("interaction_type", "unknown")

    logger.info(
        "record.processing",
        extra={
            "tenant_id": tenant_id,
            "interaction_type": interaction_type,
            "message_id": record.message_id,
        },
    )

    # Forward to Railway
    result = submit_to_railway(config, envelope_json)

    if not result.success:
        logger.error(
            "record.failed",
            extra={
                "tenant_id": tenant_id,
                "status_code": result.status_code,
                "error": result.error,
                "message_id": record.message_id,
            },
        )
        raise RuntimeError(
            f"Railway API returned failure: {result.status_code} — {result.error}"
        )

    logger.info(
        "record.success",
        extra={
            "tenant_id": tenant_id,
            "interaction_type": interaction_type,
            "status_code": result.status_code,
            "message_id": record.message_id,
        },
    )


@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
def lambda_handler(event: dict, context) -> dict:
    """Lambda entry point — processes SQS batch with partial failure reporting."""
    return process_partial_response(
        event=event,
        record_handler=process_record,
        processor=processor,
        context=context,
    )
