"""Tests for Lambda envelope parsing (EventBridge -> SQS unwrapping)."""

import json
import pytest

from action_item_graph.lambda_ingest.envelope import parse_sqs_record_body


EVENTBRIDGE_WRAPPED = {
    "version": "0",
    "id": "eb-event-id-123",
    "detail-type": "EnvelopeV1.transcript",
    "source": "com.yourapp.transcription",
    "detail": {
        "schema_version": "v1",
        "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
        "user_id": "auth0|test",
        "interaction_type": "transcript",
        "content": {"text": "Hello world", "format": "plain"},
        "timestamp": "2026-02-14T15:30:00Z",
        "source": "web-mic",
    },
}


class TestParseSqsRecordBody:
    def test_extracts_detail_from_eventbridge_wrapper(self):
        body = json.dumps(EVENTBRIDGE_WRAPPED)
        result = parse_sqs_record_body(body)
        assert result["schema_version"] == "v1"
        assert result["tenant_id"] == "550e8400-e29b-41d4-a716-446655440000"
        assert result["interaction_type"] == "transcript"

    def test_handles_email_event(self):
        email_event = {
            **EVENTBRIDGE_WRAPPED,
            "detail-type": "EnvelopeV1.email",
            "source": "com.eq.email-pipeline",
            "detail": {
                "schema_version": "v1",
                "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
                "user_id": "user-123",
                "interaction_type": "email",
                "content": {"text": "---\ntype: email\n---\nHi", "format": "email"},
                "timestamp": "2026-02-14T15:30:00Z",
                "source": "gmail",
            },
        }
        body = json.dumps(email_event)
        result = parse_sqs_record_body(body)
        assert result["interaction_type"] == "email"
        assert result["source"] == "gmail"

    def test_raises_on_missing_detail(self):
        body = json.dumps({"version": "0", "id": "test"})
        with pytest.raises(ValueError, match="detail"):
            parse_sqs_record_body(body)

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="JSON"):
            parse_sqs_record_body("not json")
