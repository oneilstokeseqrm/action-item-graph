"""Tests for the Lambda dispatcher entry point.

Tests the post-Phase-C handler that uses DBOSClient.enqueue instead of
HTTP-forwarding to Railway. Patches the module-level singletons so no
real Neon connection or CloudWatch API call is attempted.

Requires `uv sync --extra lambda` for boto3 + aws_lambda_powertools;
without the extra, this module skips via ``pytest.importorskip``.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("boto3")
pytest.importorskip("aws_lambda_powertools")

# These imports trigger aws_lambda_powertools.* loads, so they live after the
# importorskip gate.
from action_item_graph.lambda_ingest import handler  # noqa: E402
from action_item_graph.lambda_ingest.handler import lambda_handler  # noqa: E402


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


def _eb_body(*, interaction_id: str = "550e8400-e29b-41d4-a716-446655440000") -> str:
    return json.dumps(
        {
            "version": "0",
            "id": "test-eb-id",
            "detail-type": "EnvelopeV1.transcript",
            "source": "com.yourapp.transcription",
            "detail": {
                "schema_version": "v1",
                "tenant_id": "11111111-1111-4111-8111-111111111111",
                "user_id": "auth0|test",
                "interaction_id": interaction_id,
                "interaction_type": "transcript",
                "content": {"text": "A: Hello", "format": "plain"},
                "timestamp": "2026-02-14T15:30:00Z",
                "source": "web-mic",
            },
        }
    )


def _lambda_context() -> MagicMock:
    context = MagicMock()
    context.function_name = "test"
    context.memory_limit_in_mb = 256
    context.invoked_function_arn = "arn:aws:lambda:us-east-1:123:function:test"
    return context


@pytest.fixture(autouse=True)
def _reset_module_singletons():
    """Reset cold-start singletons between tests so each test sees a clean state."""
    handler._config = None
    handler._dbos_client = None
    handler._cloudwatch_client = None
    yield
    handler._config = None
    handler._dbos_client = None
    handler._cloudwatch_client = None


@pytest.fixture
def mock_config():
    """Patches _get_config so no Secrets Manager call is attempted."""
    config = MagicMock()
    config.DBOS_SYSTEM_DATABASE_URL = "postgresql://stub-for-tests"
    with patch.object(handler, "_get_config", return_value=config):
        yield config


@pytest.fixture
def mock_dbos_client(mock_config):
    """Patches _get_dbos_client so no real DBOSClient is constructed."""
    client = MagicMock()
    with patch.object(handler, "_get_dbos_client", return_value=client):
        yield client


@pytest.fixture
def mock_cloudwatch_client():
    """Patches _get_cloudwatch_client so put_metric_data is observable."""
    client = MagicMock()
    with patch.object(handler, "_get_cloudwatch_client", return_value=client):
        yield client


class TestSuccessfulDispatch:
    def test_both_enqueues_succeed_no_batch_failures(self, mock_dbos_client, mock_cloudwatch_client):
        event = _sqs_event([_eb_body()])
        result = lambda_handler(event, _lambda_context())

        assert result["batchItemFailures"] == []
        # Two enqueues: one action-item, one deal
        assert mock_dbos_client.enqueue.call_count == 2
        # No partial-enqueue metric on the happy path
        mock_cloudwatch_client.put_metric_data.assert_not_called()

    def test_workflow_ids_match_locked_format(self, mock_dbos_client, mock_cloudwatch_client):
        interaction_id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        event = _sqs_event([_eb_body(interaction_id=interaction_id)])
        lambda_handler(event, _lambda_context())

        assert mock_dbos_client.enqueue.call_count == 2
        ai_call, deal_call = mock_dbos_client.enqueue.call_args_list
        ai_options, _ai_envelope = ai_call.args
        deal_options, _deal_envelope = deal_call.args

        assert ai_options["workflow_id"] == f"action-item-graph:action-item:interaction-{interaction_id}"
        assert deal_options["workflow_id"] == f"action-item-graph:deal:interaction-{interaction_id}"

    def test_enqueue_options_carry_lock_invariants(self, mock_dbos_client, mock_cloudwatch_client):
        event = _sqs_event([_eb_body()])
        lambda_handler(event, _lambda_context())

        ai_call, deal_call = mock_dbos_client.enqueue.call_args_list
        ai_options, _ = ai_call.args
        deal_options, _ = deal_call.args

        assert ai_options["workflow_name"] == "action_item_workflow"
        assert ai_options["queue_name"] == "action-item-pipeline"
        assert ai_options["workflow_timeout"] == 900.0

        assert deal_options["workflow_name"] == "deal_workflow"
        assert deal_options["queue_name"] == "deal-pipeline"
        assert deal_options["workflow_timeout"] == 900.0

    def test_envelope_passed_verbatim_to_workflows(self, mock_dbos_client, mock_cloudwatch_client):
        event = _sqs_event([_eb_body()])
        lambda_handler(event, _lambda_context())

        ai_call, deal_call = mock_dbos_client.enqueue.call_args_list
        _, ai_envelope = ai_call.args
        _, deal_envelope = deal_call.args

        assert ai_envelope == deal_envelope
        # The envelope is the EnvelopeV1 detail dict from the EventBridge wrapper
        assert ai_envelope["schema_version"] == "v1"
        assert ai_envelope["interaction_type"] == "transcript"

    def test_dbos_client_reused_across_records(self, mock_dbos_client, mock_cloudwatch_client):
        """Module-level singleton: a batch of two records makes 4 enqueue
        calls (2 per record) but only one DBOSClient construction. The
        fixture patches _get_dbos_client, which is called per record."""
        event = _sqs_event([_eb_body(), _eb_body(interaction_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff")])
        lambda_handler(event, _lambda_context())

        assert mock_dbos_client.enqueue.call_count == 4


class TestFailureModes:
    def test_first_enqueue_failure_reports_batch_failure(self, mock_dbos_client, mock_cloudwatch_client):
        """First enqueue raises → SQS retries the whole record. No partial state,
        no metric — the action-item enqueue never landed."""
        mock_dbos_client.enqueue.side_effect = [RuntimeError("transient DB error"), None]

        event = _sqs_event([_eb_body()])
        result = lambda_handler(event, _lambda_context())

        assert len(result["batchItemFailures"]) == 1
        # Second enqueue never called when first raises
        assert mock_dbos_client.enqueue.call_count == 1
        mock_cloudwatch_client.put_metric_data.assert_not_called()

    def test_second_enqueue_failure_emits_partial_metric(self, mock_dbos_client, mock_cloudwatch_client):
        """First succeeds, second raises → emit partial_enqueue_pair_count
        BEFORE re-raising. On SQS redelivery the first enqueue is a silent
        no-op via DBOS dedup; second retries."""
        mock_dbos_client.enqueue.side_effect = [None, RuntimeError("transient DB error")]

        event = _sqs_event([_eb_body()])
        result = lambda_handler(event, _lambda_context())

        assert len(result["batchItemFailures"]) == 1
        assert mock_dbos_client.enqueue.call_count == 2
        # CloudWatch metric emitted once for the partial-enqueue case
        mock_cloudwatch_client.put_metric_data.assert_called_once()
        call_kwargs = mock_cloudwatch_client.put_metric_data.call_args.kwargs
        assert call_kwargs["Namespace"] == "ActionItemGraph/Lambda"
        metric_data = call_kwargs["MetricData"][0]
        assert metric_data["MetricName"] == "partial_enqueue_pair_count"
        assert metric_data["Value"] == 1.0
        assert metric_data["Unit"] == "Count"

    def test_missing_interaction_id_reports_batch_failure(self, mock_dbos_client, mock_cloudwatch_client):
        """Fail-fast (Rule 1): envelope without interaction_id can't construct
        a deterministic workflow_id, so SQS handles redelivery → DLQ."""
        body = json.dumps(
            {
                "version": "0",
                "id": "test-eb-id",
                "detail-type": "EnvelopeV1.transcript",
                "source": "com.yourapp.transcription",
                "detail": {
                    "schema_version": "v1",
                    "tenant_id": "11111111-1111-4111-8111-111111111111",
                    # interaction_id intentionally omitted
                    "interaction_type": "transcript",
                    "content": {"text": "x", "format": "plain"},
                    "timestamp": "2026-02-14T15:30:00Z",
                    "source": "web-mic",
                },
            }
        )

        event = _sqs_event([body])
        result = lambda_handler(event, _lambda_context())

        assert len(result["batchItemFailures"]) == 1
        # Validation raised before any enqueue
        mock_dbos_client.enqueue.assert_not_called()
        mock_cloudwatch_client.put_metric_data.assert_not_called()


class TestCloudWatchDegradation:
    def test_cloudwatch_client_unavailable_does_not_crash_dispatcher(self, mock_dbos_client):
        """If _get_cloudwatch_client returns None (no AWS creds in dev/CI),
        the partial-enqueue metric emission is skipped silently. The
        dispatcher's primary path — re-raising for SQS — is unaffected."""
        mock_dbos_client.enqueue.side_effect = [None, RuntimeError("transient DB error")]

        with patch.object(handler, "_get_cloudwatch_client", return_value=None):
            event = _sqs_event([_eb_body()])
            result = lambda_handler(event, _lambda_context())

        # Partial-enqueue still surfaces as a BatchItemFailure
        assert len(result["batchItemFailures"]) == 1

    def test_cloudwatch_put_metric_data_failure_does_not_crash_dispatcher(self, mock_dbos_client):
        """If put_metric_data itself raises (e.g., throttling, network),
        the metric error is logged but does NOT propagate. The primary
        re-raise path still surfaces the partial-enqueue BatchItemFailure."""
        mock_dbos_client.enqueue.side_effect = [None, RuntimeError("transient DB error")]
        flaky_cw = MagicMock()
        flaky_cw.put_metric_data.side_effect = RuntimeError("cloudwatch down")

        with patch.object(handler, "_get_cloudwatch_client", return_value=flaky_cw):
            event = _sqs_event([_eb_body()])
            result = lambda_handler(event, _lambda_context())

        assert len(result["batchItemFailures"]) == 1
        flaky_cw.put_metric_data.assert_called_once()
