"""Tests for the Lambda handler entry point."""

import json
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from action_item_graph.lambda_ingest.handler import lambda_handler


# Mock SQS event
def _sqs_event(bodies: list[str]) -> dict:
    return {
        "Records": [
            {
                "messageId": f"msg-{i}",
                "body": body,
                "receiptHandle": f"handle-{i}",
                "attributes": {},
                "messageAttributes": {},
                "md5OfBody": "",
                "eventSource": "aws:sqs",
                "eventSourceARN": "arn:aws:sqs:us-east-1:123:test-queue",
                "awsRegion": "us-east-1",
            }
            for i, body in enumerate(bodies)
        ]
    }


VALID_EB_BODY = json.dumps(
    {
        "version": "0",
        "id": "test-eb-id",
        "detail-type": "EnvelopeV1.transcript",
        "source": "com.yourapp.transcription",
        "detail": {
            "schema_version": "v1",
            "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "auth0|test",
            "interaction_type": "transcript",
            "content": {"text": "A: Hello", "format": "plain"},
            "timestamp": "2026-02-14T15:30:00Z",
            "source": "web-mic",
        },
    }
)


@dataclass
class MockConfig:
    API_BASE_URL: str = "https://test.railway.app"
    WORKER_API_KEY: str = "key"
    HTTP_TIMEOUT_SECONDS: int = 10
    MAX_RETRIES: int = 0


class TestLambdaHandler:
    @patch("action_item_graph.lambda_ingest.handler._get_config", return_value=MockConfig())
    @patch("action_item_graph.lambda_ingest.handler.submit_to_railway")
    def test_successful_processing(self, mock_submit, mock_config):
        mock_submit.return_value = MagicMock(success=True, status_code=200)


        event = _sqs_event([VALID_EB_BODY])
        context = MagicMock()
        context.function_name = "test"
        context.memory_limit_in_mb = 256
        context.invoked_function_arn = "arn:aws:lambda:us-east-1:123:function:test"

        result = lambda_handler(event, context)

        mock_submit.assert_called_once()
        # BatchProcessor returns partial failure response
        assert "batchItemFailures" in result

    @patch("action_item_graph.lambda_ingest.handler._get_config", return_value=MockConfig())
    @patch("action_item_graph.lambda_ingest.handler.submit_to_railway")
    def test_failed_record_reported(self, mock_submit, mock_config):
        mock_submit.return_value = MagicMock(success=False, status_code=500, error="Server Error")


        event = _sqs_event([VALID_EB_BODY])
        context = MagicMock()
        context.function_name = "test"
        context.memory_limit_in_mb = 256
        context.invoked_function_arn = "arn:aws:lambda:us-east-1:123:function:test"

        result = lambda_handler(event, context)
        # Failed record should be in batchItemFailures
        assert len(result["batchItemFailures"]) == 1
