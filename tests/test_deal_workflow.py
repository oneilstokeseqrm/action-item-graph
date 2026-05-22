"""Tests for the deal DBOS workflow body + per-step contracts.

Mirrors the action_item_workflow test structure. Same access pattern via
``__wrapped__.__wrapped__`` to bypass DBOS's runtime guard, same step_mocks
fixture pattern around ``patch.multiple``.

Coverage scope (P1 essentials):
- D1 validation in workflow body (Rule 1)
- Authoritative interaction_id from D3 (Rule 5)
- opportunity_id-not-found surfaces in result.warnings (Codex B-2 R1 MED)
- no_deals early-return still calls D8 enrich_interaction
- Per-step fail-open + no_op contracts on D3, D4, D7, D8, D9
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from action_item_graph.workflows import _runtime
from action_item_graph.workflows._runtime import WorkflowClients
from deal_graph.errors import DealPipelineError
from deal_graph.pipeline.merger import DealMergeResult
from deal_graph.workflows import deal_steps
from deal_graph.workflows.deal_workflow import deal_workflow


TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")
TEST_INTERACTION_ID = "550e8400-e29b-41d4-a716-446655440000"
TEST_OPPORTUNITY_ID = "019c1fa0-4444-7000-8000-000000000005"


# Access the raw workflow function, bypassing both DBOS decorator layers.
_workflow_body = deal_workflow.__wrapped__.__wrapped__


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    prior = _runtime._clients
    _runtime._clients = None
    yield
    _runtime._clients = prior


@pytest.fixture
def registered_clients():
    clients = WorkflowClients(
        neo4j=MagicMock(),
        deal_neo4j=MagicMock(),
        openai=MagicMock(),
        postgres=None,
    )
    _runtime._clients = clients
    yield clients


@pytest.fixture
def step_mocks():
    """Patches all 9 @DBOS.step calls inside deal_workflow."""
    mocks = {
        "verify_account_step": AsyncMock(return_value=None),
        "ensure_interaction_step": AsyncMock(
            return_value={"status": "ok", "interaction_id": TEST_INTERACTION_ID}
        ),
        "merge_contacts_to_deal_base_step": AsyncMock(),
        "fetch_existing_deal_step": AsyncMock(return_value=None),
        "deal_extraction_step": AsyncMock(),
        "match_merge_loop_step": AsyncMock(),
        "enrich_interaction_step": AsyncMock(),
        "deal_postgres_dual_write_step": AsyncMock(),
        "publish_deal_processed_step": AsyncMock(return_value=None),
    }
    with patch.multiple("deal_graph.workflows.deal_workflow", **mocks):
        yield mocks


def _envelope(
    *,
    account_id: str | None = "acct-1",
    interaction_id: str | None = TEST_INTERACTION_ID,
    opportunity_id: str | None = None,
) -> dict:
    """Minimal valid envelope. opportunity_id lives in extras per EnvelopeV1."""
    env: dict = {
        "schema_version": "v1",
        "tenant_id": str(TEST_TENANT_ID),
        "user_id": "auth0|test",
        "interaction_type": "transcript",
        "content": {"text": "A: Hi", "format": "plain"},
        "timestamp": "2026-02-14T15:30:00Z",
        "source": "web-mic",
    }
    if account_id is not None:
        env["account_id"] = account_id
    if interaction_id is not None:
        env["interaction_id"] = interaction_id
    if opportunity_id is not None:
        env["extras"] = {"opportunity_id": opportunity_id}
    return env


def _empty_extraction_result_dict() -> dict:
    """Mimic deal_extraction_step output with no deals."""
    return {
        "result": {
            "deals": [],
            "has_deals": False,
            "extraction_notes": None,
        },
        "embeddings": [],
    }


def _populated_extraction_result_dict() -> dict:
    """Mimic deal_extraction_step output with one deal."""
    return {
        "result": {
            "deals": [{"opportunity_name": "Acme Q1 deal"}],
            "has_deals": True,
            "extraction_notes": None,
        },
        "embeddings": [[0.1, 0.2]],
    }


# ---------------------------------------------------------------------------
# Workflow body — Rule 1 validation gate
# ---------------------------------------------------------------------------


class TestDealWorkflowValidationGate:
    async def test_missing_account_id_raises_before_any_step(self, step_mocks):
        env = _envelope(account_id=None)
        with pytest.raises(DealPipelineError, match="account_id"):
            await _workflow_body(env)
        for name, mock in step_mocks.items():
            assert not mock.called, f"step {name} should not have been called"

    async def test_empty_account_id_string_raises(self, step_mocks):
        env = _envelope(account_id="")
        with pytest.raises(DealPipelineError):
            await _workflow_body(env)


# ---------------------------------------------------------------------------
# Workflow body — opportunity_id-not-found warning
# ---------------------------------------------------------------------------


class TestDealWorkflowOpportunityNotFoundWarning:
    """Codex B-2 R1 MED: when envelope carries opportunity_id but the deal
    isn't in Neo4j, surface a warning in the workflow result rather than
    silently falling through to discovery."""

    async def test_warning_surfaces_when_existing_deal_is_none_with_opp_id(self, step_mocks):
        env = _envelope(opportunity_id=TEST_OPPORTUNITY_ID)
        # D5 returns None (not found) — but env carries opportunity_id
        step_mocks["fetch_existing_deal_step"].return_value = None
        step_mocks["deal_extraction_step"].return_value = _empty_extraction_result_dict()
        result = await _workflow_body(env)

        assert len(result["warnings"]) == 1
        assert TEST_OPPORTUNITY_ID in result["warnings"][0]
        assert "not found" in result["warnings"][0].lower()

    async def test_no_warning_when_opportunity_absent(self, step_mocks):
        env = _envelope(opportunity_id=None)
        step_mocks["fetch_existing_deal_step"].return_value = None
        step_mocks["deal_extraction_step"].return_value = _empty_extraction_result_dict()
        result = await _workflow_body(env)

        assert result["warnings"] == []

    async def test_no_warning_when_existing_deal_found(self, step_mocks):
        env = _envelope(opportunity_id=TEST_OPPORTUNITY_ID)
        step_mocks["fetch_existing_deal_step"].return_value = {"opportunity_id": TEST_OPPORTUNITY_ID}
        step_mocks["deal_extraction_step"].return_value = _empty_extraction_result_dict()
        result = await _workflow_body(env)

        assert result["warnings"] == []


# ---------------------------------------------------------------------------
# Workflow body — no_deals early-return still calls D8
# ---------------------------------------------------------------------------


class TestDealWorkflowNoDealsPath:
    async def test_no_deals_still_calls_enrich_interaction(self, step_mocks):
        env = _envelope()
        step_mocks["deal_extraction_step"].return_value = _empty_extraction_result_dict()
        result = await _workflow_body(env)

        assert result["status"] == "no_deals"
        assert result["total_extracted"] == 0
        assert result["deals_created"] == []
        # D8 enrich_interaction MUST run even on no_deals to record the
        # interaction's deal-pipeline status. Legacy parity pipeline.py:366-372.
        assert step_mocks["enrich_interaction_step"].called
        # D7 + D9 must NOT have run
        assert not step_mocks["match_merge_loop_step"].called
        assert not step_mocks["deal_postgres_dual_write_step"].called


# ---------------------------------------------------------------------------
# Workflow body — happy path
# ---------------------------------------------------------------------------


class TestDealWorkflowHappyPath:
    async def test_full_pipeline_executes_d7_d8_d9(self, step_mocks):
        env = _envelope(opportunity_id=TEST_OPPORTUNITY_ID)
        step_mocks["fetch_existing_deal_step"].return_value = {"opportunity_id": TEST_OPPORTUNITY_ID}
        step_mocks["deal_extraction_step"].return_value = _populated_extraction_result_dict()
        step_mocks["match_merge_loop_step"].return_value = {
            "merge_results": [],
            "deals_created": [TEST_OPPORTUNITY_ID],
            "deals_merged": [],
            "errors": [],
        }

        result = await _workflow_body(env)

        assert result["status"] == "ok"
        assert result["total_extracted"] == 1
        assert result["deals_created"] == [TEST_OPPORTUNITY_ID]
        assert step_mocks["match_merge_loop_step"].called
        assert step_mocks["enrich_interaction_step"].called
        assert step_mocks["deal_postgres_dual_write_step"].called


# ---------------------------------------------------------------------------
# Workflow body — D10 publish_deal_processed_step integration
# ---------------------------------------------------------------------------


class TestDealWorkflowPublishStep:
    """D10: publish_deal_processed_step is called once per successful workflow
    with the aggregated deals_created/deals_merged lists, and is skipped on the
    no_deals early-return path. Feature-flag gating lives inside the publisher
    helper — the step itself always fires at this layer."""

    async def test_publish_step_called_with_aggregated_deals(self, step_mocks):
        env = _envelope(opportunity_id=TEST_OPPORTUNITY_ID)
        step_mocks["fetch_existing_deal_step"].return_value = {
            "opportunity_id": TEST_OPPORTUNITY_ID
        }
        step_mocks["deal_extraction_step"].return_value = _populated_extraction_result_dict()
        merged_id = "019e4124-f840-7bb1-abed-99ee6eebf8ea"
        step_mocks["match_merge_loop_step"].return_value = {
            "merge_results": [],
            "deals_created": [TEST_OPPORTUNITY_ID],
            "deals_merged": [merged_id],
            "errors": [],
        }

        await _workflow_body(env)

        publish = step_mocks["publish_deal_processed_step"]
        assert publish.call_count == 1
        kwargs = publish.call_args.kwargs
        assert kwargs["tenant_id"] == str(TEST_TENANT_ID)
        assert kwargs["account_id"] == "acct-1"
        assert kwargs["interaction_id"] == TEST_INTERACTION_ID
        assert kwargs["deals_created"] == [TEST_OPPORTUNITY_ID]
        assert kwargs["deals_merged"] == [merged_id]

    async def test_publish_step_not_called_when_no_deals(self, step_mocks):
        """no_deals early-return path returns before the D10 hook."""
        env = _envelope()
        step_mocks["deal_extraction_step"].return_value = _empty_extraction_result_dict()

        result = await _workflow_body(env)

        assert result["status"] == "no_deals"
        assert not step_mocks["publish_deal_processed_step"].called


# ---------------------------------------------------------------------------
# Workflow body — Rule 5: D8 receives downstream-produced interaction_id
# ---------------------------------------------------------------------------


class TestDealWorkflowEnrichInteractionRule5:
    async def test_d8_called_with_d3_returned_interaction_id_not_envelope(self, step_mocks):
        """Rule 5: enrich_interaction_step's interaction_id arg is the value
        D3 returned, not envelope.interaction_id directly."""
        env = _envelope(interaction_id=TEST_INTERACTION_ID)
        # D3 returns a different interaction_id (simulating MERGE'd value)
        d3_resolved_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        step_mocks["ensure_interaction_step"].return_value = {
            "status": "ok",
            "interaction_id": d3_resolved_id,
        }
        step_mocks["deal_extraction_step"].return_value = _empty_extraction_result_dict()
        await _workflow_body(env)

        call = step_mocks["enrich_interaction_step"].call_args
        # Positional args: envelope_dict, interaction_id, deals_processed
        _, interaction_id_arg, _ = call.args
        assert interaction_id_arg == d3_resolved_id
        assert interaction_id_arg != TEST_INTERACTION_ID


# ---------------------------------------------------------------------------
# Per-step contract tests
# ---------------------------------------------------------------------------


class TestEnsureInteractionStep:
    """D3: no_op when missing interaction_id."""

    async def test_no_op_when_envelope_missing_interaction_id(self, registered_clients):
        env = _envelope(interaction_id=None)
        result = await deal_steps.ensure_interaction_step(env)
        assert result == {"status": "no_op", "interaction_id": None}


class TestMergeContactsToDealBaseStep:
    """D4: side-branch with conditional execution + fail-open."""

    async def test_no_op_when_opportunity_id_absent(self, registered_clients):
        env = _envelope()
        env["extras"] = {"contacts": [{"contact_id": "c-1", "email": "c@e.com"}]}
        result = await deal_steps.merge_contacts_to_deal_base_step(env)
        assert result["status"] == "no_op"

    async def test_no_op_when_no_contact_ids(self, registered_clients):
        env = _envelope(opportunity_id=TEST_OPPORTUNITY_ID)
        env["extras"]["contacts"] = []
        result = await deal_steps.merge_contacts_to_deal_base_step(env)
        assert result["status"] == "no_op"

    async def test_fail_open_when_helper_raises(self, registered_clients):
        env = _envelope(opportunity_id=TEST_OPPORTUNITY_ID)
        env["extras"]["contacts"] = [{"contact_id": "c-1", "email": "c@e.com"}]
        with patch(
            "deal_graph.workflows.deal_steps.merge_contacts_to_deal",
            AsyncMock(side_effect=RuntimeError("neo4j down")),
        ):
            result = await deal_steps.merge_contacts_to_deal_base_step(env)

        assert result["status"] == "skipped"
        assert "neo4j down" in result["error"]


class TestMatchMergeLoopStep:
    """D7: deals/embeddings length mismatch raises (Rule 4)."""

    async def test_raises_on_deals_embeddings_length_mismatch(self, registered_clients):
        env = _envelope()
        # 2 deals but 1 embedding — length skew is a real upstream bug
        extraction_dict = {
            "result": {
                "deals": [
                    {"opportunity_name": "Deal A"},
                    {"opportunity_name": "Deal B"},
                ],
                "has_deals": True,
                "extraction_notes": None,
            },
            "embeddings": [[0.1]],  # only 1 embedding
        }

        # Patch deal_extraction_result_from_dict to return the misaligned data
        # directly without going through DealExtractionResult model validation
        # (which would itself catch this).
        mock_extraction_result = MagicMock()
        mock_extraction_result.has_deals = True
        mock_extraction_result.deals = [MagicMock(), MagicMock()]
        with patch(
            "deal_graph.workflows.deal_steps.deal_extraction_result_from_dict",
            return_value=(mock_extraction_result, [[0.1]]),  # 1 embedding
        ):
            with pytest.raises(RuntimeError, match="length mismatch"):
                await deal_steps.match_merge_loop_step(env, extraction_dict)

    async def test_empty_deals_returns_empty_results(self, registered_clients):
        env = _envelope()
        extraction_dict = {
            "result": {
                "deals": [],
                "has_deals": False,
                "extraction_notes": None,
            },
            "embeddings": [],
        }
        result = await deal_steps.match_merge_loop_step(env, extraction_dict)
        assert result == {
            "merge_results": [],
            "deals_created": [],
            "deals_merged": [],
            "errors": [],
        }


class TestEnrichInteractionStepFailOpen:
    """D8: no_op when interaction_id is None, fail-open when raises."""

    async def test_no_op_when_interaction_id_is_none(self, registered_clients):
        env = _envelope()
        result = await deal_steps.enrich_interaction_step(env, None, 0)
        assert result == {"status": "no_op"}

    async def test_fail_open_when_enrich_raises(self):
        clients = WorkflowClients(
            neo4j=MagicMock(),
            deal_neo4j=MagicMock(),
            openai=MagicMock(),
            postgres=None,
        )
        _runtime._clients = clients

        mock_pipeline = MagicMock()
        mock_pipeline._enrich_interaction = AsyncMock(side_effect=RuntimeError("neo4j down"))
        with patch.object(deal_steps, "_build_deal_pipeline", return_value=mock_pipeline):
            result = await deal_steps.enrich_interaction_step(
                _envelope(), TEST_INTERACTION_ID, 1
            )

        assert result["status"] == "skipped"
        assert "neo4j down" in result["error"]


class TestDealPostgresDualWriteStep:
    """D9: no_op when postgres is None, fail-open when raises."""

    async def test_no_op_when_postgres_client_is_none(self, registered_clients):
        # registered_clients fixture already sets postgres=None
        result = await deal_steps.deal_postgres_dual_write_step(
            _envelope(), [], TEST_INTERACTION_ID
        )
        assert result["status"] == "no_op"

    async def test_fail_open_when_dual_write_raises(self):
        clients = WorkflowClients(
            neo4j=MagicMock(),
            deal_neo4j=MagicMock(),
            openai=MagicMock(),
            postgres=MagicMock(),
        )
        _runtime._clients = clients

        mock_pipeline = MagicMock()
        mock_pipeline._dual_write_postgres = AsyncMock(side_effect=RuntimeError("pg down"))
        with patch.object(deal_steps, "_build_deal_pipeline", return_value=mock_pipeline):
            result = await deal_steps.deal_postgres_dual_write_step(
                _envelope(), [], TEST_INTERACTION_ID
            )

        assert result["status"] == "skipped"
        assert "pg down" in result["error"]
