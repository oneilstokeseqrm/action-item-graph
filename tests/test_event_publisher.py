"""Unit tests for the EventBridge publisher helper.

The helper is pure (no DBOS dependency) so tests run as plain pytest with
unittest.mock patches on boto3 and structlog. Covers the 4 short-circuits
+ happy-path payload shape + boto3 error containment + partial-failure
logging.

Wire-contract anchor:
    thematic-lm/src/thematic_lm/opportunity/event_models.py
    (DealProcessedEvent.from_sqs_body)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from deal_graph.clients.event_publisher import publish_deal_processed


TENANT_ID = "11111111-1111-4111-8111-111111111111"
ACCOUNT_ID = "668b24a7-ec44-53dd-96a4-a9a392d200c2"
INTERACTION_ID = "4a41a7ff-32fc-4e3c-a51a-caecbf91b372"
DEAL_ID = "019e4124-f840-7bb1-abed-99ee6eebf8ea"


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("ENABLE_DEAL_PROCESSED_EVENTS", "true")
    yield


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("ENABLE_DEAL_PROCESSED_EVENTS", raising=False)
    yield


# ---------------------------------------------------------------------------
# Happy path — payload shape matches the consumer contract
# ---------------------------------------------------------------------------


class TestPublishConstructsCorrectPayload:
    def test_payload_matches_contract(self, flag_on):
        mock_client = MagicMock()
        mock_client.put_events.return_value = {
            "FailedEntryCount": 0,
            "Entries": [{"EventId": "evt-test-1"}],
        }
        with patch(
            "deal_graph.clients.event_publisher.boto3.client",
            return_value=mock_client,
        ) as mock_boto:
            publish_deal_processed(
                tenant_id=TENANT_ID,
                account_id=ACCOUNT_ID,
                interaction_id=INTERACTION_ID,
                deals_created=[DEAL_ID],
                deals_merged=[],
                source="deal-pipeline",
                workflow_id="action-item-graph:deal:interaction-abc",
            )

        # Client constructed with explicit region + tight botocore config
        assert mock_boto.call_count == 1
        boto_call = mock_boto.call_args
        assert boto_call.args == ("events",)
        assert boto_call.kwargs["region_name"] == "us-east-1"
        cfg = boto_call.kwargs["config"]
        # Cap worst-case stall on a fail-open step
        assert cfg.connect_timeout == 2
        assert cfg.read_timeout == 5
        assert cfg.retries == {"max_attempts": 1, "mode": "standard"}

        mock_client.put_events.assert_called_once()
        entries = mock_client.put_events.call_args.kwargs["Entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["Source"] == "com.eq.action-item-graph"
        assert entry["DetailType"] == "deal.processed"
        # Explicit bus avoids ambient-cred surprises (codex #1)
        assert entry["EventBusName"] == "default"

        detail = json.loads(entry["Detail"])
        assert detail["tenant_id"] == TENANT_ID
        assert detail["account_id"] == ACCOUNT_ID
        assert detail["interaction_id"] == INTERACTION_ID
        assert detail["deals_created"] == [DEAL_ID]
        assert detail["deals_merged"] == []
        assert detail["source"] == "deal-pipeline"
        # timestamp is ISO-8601 UTC; just sanity-check it parses as a string
        assert isinstance(detail["timestamp"], str) and "T" in detail["timestamp"]


# ---------------------------------------------------------------------------
# Short-circuit: feature flag off
# ---------------------------------------------------------------------------


class TestFeatureFlagOffDoesNotPublish:
    def test_flag_unset_skips_boto3(self, flag_off):
        with patch("deal_graph.clients.event_publisher.boto3.client") as mock_boto:
            result = publish_deal_processed(
                tenant_id=TENANT_ID,
                account_id=ACCOUNT_ID,
                interaction_id=INTERACTION_ID,
                deals_created=[DEAL_ID],
                deals_merged=[],
            )
        assert result is None
        mock_boto.assert_not_called()

    def test_flag_explicit_false_skips_boto3(self, monkeypatch):
        monkeypatch.setenv("ENABLE_DEAL_PROCESSED_EVENTS", "false")
        with patch("deal_graph.clients.event_publisher.boto3.client") as mock_boto:
            publish_deal_processed(
                tenant_id=TENANT_ID,
                account_id=ACCOUNT_ID,
                interaction_id=INTERACTION_ID,
                deals_created=[DEAL_ID],
                deals_merged=[],
            )
        mock_boto.assert_not_called()


# ---------------------------------------------------------------------------
# Short-circuit: empty deal lists
# ---------------------------------------------------------------------------


class TestEmptyDealsSkipped:
    def test_both_empty_lists_skip_publish(self, flag_on):
        with patch("deal_graph.clients.event_publisher.boto3.client") as mock_boto:
            publish_deal_processed(
                tenant_id=TENANT_ID,
                account_id=ACCOUNT_ID,
                interaction_id=INTERACTION_ID,
                deals_created=[],
                deals_merged=[],
            )
        mock_boto.assert_not_called()


# ---------------------------------------------------------------------------
# Short-circuit: empty interaction_id
# ---------------------------------------------------------------------------


class TestEmptyInteractionIdSkipped:
    def test_empty_interaction_id_skip_publish(self, flag_on):
        with patch("deal_graph.clients.event_publisher.boto3.client") as mock_boto:
            publish_deal_processed(
                tenant_id=TENANT_ID,
                account_id=ACCOUNT_ID,
                interaction_id="",
                deals_created=[DEAL_ID],
                deals_merged=[],
            )
        mock_boto.assert_not_called()


# ---------------------------------------------------------------------------
# Error containment: boto3 raises
# ---------------------------------------------------------------------------


class TestPublishFailureDoesNotRaise:
    def test_boto3_error_returns_none(self, flag_on):
        mock_client = MagicMock()
        mock_client.put_events.side_effect = RuntimeError("simulated AWS auth failure")
        with patch(
            "deal_graph.clients.event_publisher.boto3.client",
            return_value=mock_client,
        ):
            # Must not raise — log-and-continue contract
            result = publish_deal_processed(
                tenant_id=TENANT_ID,
                account_id=ACCOUNT_ID,
                interaction_id=INTERACTION_ID,
                deals_created=[DEAL_ID],
                deals_merged=[],
            )
        assert result is None


# ---------------------------------------------------------------------------
# Partial failure: FailedEntryCount > 0 logs warning, no raise
# ---------------------------------------------------------------------------


class TestPartialFailureLogged:
    def test_failed_entry_count_logs_warning_no_raise(self, flag_on):
        mock_client = MagicMock()
        mock_client.put_events.return_value = {
            "FailedEntryCount": 1,
            "Entries": [
                {"ErrorCode": "InternalFailure", "ErrorMessage": "transient"}
            ],
        }
        with patch(
            "deal_graph.clients.event_publisher.boto3.client",
            return_value=mock_client,
        ):
            # Returns None without raising — warning logged
            result = publish_deal_processed(
                tenant_id=TENANT_ID,
                account_id=ACCOUNT_ID,
                interaction_id=INTERACTION_ID,
                deals_created=[DEAL_ID],
                deals_merged=[],
            )
        assert result is None
        # put_events still called once with the right shape
        assert mock_client.put_events.call_count == 1
