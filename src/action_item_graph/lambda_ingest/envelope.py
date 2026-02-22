"""Parse EventBridge-wrapped SQS message body to extract EnvelopeV1 payload."""

import json
from typing import Any


def parse_sqs_record_body(body: str) -> dict[str, Any]:
    """
    Extract the EnvelopeV1 payload from an EventBridge-wrapped SQS message body.

    SQS body format:
    {
        "version": "0",
        "id": "...",
        "detail-type": "EnvelopeV1.transcript",
        "source": "com.yourapp.transcription",
        "detail": { ... EnvelopeV1 payload ... }
    }

    Returns the `detail` dict (the raw EnvelopeV1 JSON).
    """
    try:
        event = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in SQS body: {e}") from e

    if "detail" not in event:
        raise ValueError(
            f"Missing 'detail' key in EventBridge event. "
            f"Keys present: {list(event.keys())}"
        )

    return event["detail"]
