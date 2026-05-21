"""Tests for the action-item DBOS workflow body + per-step contracts.

These tests target the highest-stakes correctness invariants surfaced in
Phase A/B codex review and codified in
``memory/pattern_dbos_workflow_parity_rules.md``:

  Rule 1 — validation lives in workflow body, NOT in a retryable step
  Rule 2 — LLM-prompt source-of-truth is the upstream object
  Rule 5 — pass downstream-produced identifiers explicitly (no re-derive
           from upstream envelope)

The workflow function is accessed via ``__wrapped__.__wrapped__`` to bypass
DBOS's "Function invoked before DBOS initialized" guard at
``dbos/_core.py:1107``. Step functions are call-through transparent —
``@DBOS.step`` decorator wraps body in retry logic but the underlying
function is still directly callable.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from action_item_graph.errors import ValidationError
from action_item_graph.models.action_item import ActionItem
from action_item_graph.models.entities import Interaction, InteractionType
from action_item_graph.pipeline.extractor import ExtractionOutput
from action_item_graph.pipeline.matcher import MatchCandidate, MatchResult
from action_item_graph.pipeline.merger import MergeResult
from action_item_graph.prompts.extract_action_items import (
    DeduplicationDecision,
    ExtractedActionItem,
    ExtractedTopic,
)
from action_item_graph.workflows import _runtime, action_item_steps
from action_item_graph.workflows._runtime import WorkflowClients
from action_item_graph.workflows.action_item_workflow import action_item_workflow
from action_item_graph.workflows._serialization import (
    extraction_to_dict,
    match_result_to_dict,
    merge_result_to_dict,
)


TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")
TEST_INTERACTION_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
EXTRACTED_INTERACTION_ID = UUID("99999999-9999-4999-8999-999999999999")


# Helper: the raw workflow function, bypassing DBOS's runtime guards.
_workflow_body = action_item_workflow.__wrapped__.__wrapped__


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
    """Register a mocked WorkflowClients bundle so step bodies that call
    ``get_clients()`` see test doubles."""
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
    """AsyncMock instances for all 14 action_item_workflow step calls,
    patched into the workflow module so the body sees test doubles.

    `patch.multiple(..., kw=AsyncMock()) as mocks` quirk: the dict
    returned by the context manager is empty when kwargs are explicit
    mocks (the `as` binding only populates when DEFAULT sentinel values
    are used). So we create the mocks externally and yield them as a
    dict the test can mutate + assert against.
    """
    mocks = {
        "ensure_account_step": AsyncMock(return_value=None),
        "extraction_step": AsyncMock(),
        "consolidation_step": AsyncMock(),
        "verification_step": AsyncMock(),
        "owner_resolution_step": AsyncMock(),
        "create_interaction_step": AsyncMock(),
        "merge_contacts_to_deal_step": AsyncMock(),
        "matching_step": AsyncMock(),
        "merging_llm_step": AsyncMock(),
        "merging_persist_step": AsyncMock(),
        "topic_resolution_llm_step": AsyncMock(),
        "topic_resolution_persist_step": AsyncMock(),
        "postgres_dual_write_step": AsyncMock(),
        "agent_outbox_step": AsyncMock(),
    }
    with patch.multiple("action_item_graph.workflows.action_item_workflow", **mocks):
        yield mocks


_ENV_INTERACTION_ID = "550e8400-e29b-41d4-a716-446655440000"


def _envelope(*, account_id: str | None = "acct-1", interaction_id: str | None = _ENV_INTERACTION_ID) -> dict:
    """Minimal envelope dict for workflow body input.

    interaction_id defaults to a valid UUID so step-level tests that
    construct EnvelopeV1 don't fail UUID validation. Workflow body tests
    that need to verify the "envelope had no interaction_id" path pass
    ``interaction_id=None`` explicitly.
    """
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
    return env


def _empty_extraction_dict(*, interaction_id: UUID = EXTRACTED_INTERACTION_ID) -> dict:
    """Mimic an extraction_step output with zero items."""
    return {
        "interaction": {
            "interaction_id": str(interaction_id),
            "tenant_id": str(TEST_TENANT_ID),
            "account_id": "acct-1",
            "interaction_type": "transcript",
            "content_text": "A: Hi",
            "timestamp": "2026-02-14T15:30:00Z",
        },
        "action_items": [],
        "raw_extractions": [],
        "extraction_notes": None,
    }


def _populated_extraction_dict(*, interaction_id: UUID = EXTRACTED_INTERACTION_ID) -> dict:
    """Extraction with one action item, used for end-to-end orchestration."""
    interaction = Interaction(
        interaction_id=interaction_id,
        tenant_id=TEST_TENANT_ID,
        account_id="acct-1",
        interaction_type=InteractionType.TRANSCRIPT,
        content_text="A: I'll send the deck by Friday.",
        timestamp=datetime(2026, 2, 14, 15, 30, tzinfo=timezone.utc),
    )
    action_item = ActionItem(
        tenant_id=TEST_TENANT_ID,
        account_id="acct-1",
        action_item_text="Send the deck by Friday",
        summary="Send sales deck",
        owner="Sarah",
        owner_type="named",
        conversation_context="Sarah committed to deliver",
    )
    raw = ExtractedActionItem(
        action_item_text="Send the deck by Friday",
        owner="Sarah",
        summary="Send sales deck",
        conversation_context="Sarah committed to deliver",
        topic=ExtractedTopic(name="Sales Deck Delivery", context="Q1 sales"),
    )
    extraction = ExtractionOutput(
        interaction=interaction,
        action_items=[action_item],
        raw_extractions=[raw],
        extraction_notes=None,
    )
    return extraction_to_dict(extraction)


# ---------------------------------------------------------------------------
# Workflow body — Rule 1 (validation lives in workflow body)
# ---------------------------------------------------------------------------


class TestWorkflowValidationGate:
    async def test_missing_account_id_raises_validation_error_before_any_step(self, step_mocks):
        """Rule 1: ValidationError must propagate immediately (no
        DBOSMaxStepRetriesExceeded wrapping). Verified by checking that
        no step was called when validation raises.
        """
        env = _envelope(account_id=None)
        with pytest.raises(ValidationError, match="account_id"):
            await _workflow_body(env)
        # No step was called — validation raised before S1
        for name, mock in step_mocks.items():
            assert not mock.called, f"step {name} should not have been called"

    async def test_empty_account_id_string_raises(self, step_mocks):
        env = _envelope(account_id="")
        with pytest.raises(ValidationError):
            await _workflow_body(env)


# ---------------------------------------------------------------------------
# Workflow body — Rule 5 (authoritative downstream-produced identifier)
# ---------------------------------------------------------------------------


class TestWorkflowAuthoritativeInteractionId:
    async def test_returns_extraction_interaction_id_not_envelope(self, step_mocks):
        """Rule 5: result['interaction_id'] is the value extraction settled
        on, not whatever the envelope carried. Codex Round 2 MEDIUM #1."""
        # Use a valid UUID for envelope (the workflow itself doesn't validate,
        # but using a real UUID keeps the test future-proof).
        env = _envelope(interaction_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
        step_mocks["extraction_step"].return_value = _empty_extraction_dict()
        result = await _workflow_body(env)

        # Extracted ID wins over envelope ID
        assert result["interaction_id"] == str(EXTRACTED_INTERACTION_ID)
        assert result["interaction_id"] != "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    async def test_envelope_without_interaction_id_still_returns_extracted(self, step_mocks):
        """When the envelope has no interaction_id at all, the workflow falls
        back to '?' for the trace tag but the final result still uses
        extraction's value."""
        env = _envelope(interaction_id=None)
        step_mocks["extraction_step"].return_value = _empty_extraction_dict()
        result = await _workflow_body(env)

        assert result["interaction_id"] == str(EXTRACTED_INTERACTION_ID)


# ---------------------------------------------------------------------------
# Workflow body — early-return paths
# ---------------------------------------------------------------------------


class TestWorkflowEarlyReturns:
    async def test_no_items_after_extraction_short_circuits(self, step_mocks):
        env = _envelope()
        step_mocks["extraction_step"].return_value = _empty_extraction_dict()
        result = await _workflow_body(env)

        assert result["status"] == "no_items"
        assert result["created"] == 0
        assert result["updated"] == 0
        # Steps S3-S14 not called
        assert not step_mocks["consolidation_step"].called
        assert not step_mocks["agent_outbox_step"].called

    async def test_all_filtered_after_verification_short_circuits(self, step_mocks):
        env = _envelope()
        populated = _populated_extraction_dict()
        step_mocks["extraction_step"].return_value = populated
        step_mocks["consolidation_step"].return_value = {
            "extraction": populated,
            "items_consolidated": 0,
        }
        step_mocks["verification_step"].return_value = {
            "extraction": _empty_extraction_dict(),
            "items_rejected": 1,
            "status": "ok",
        }
        result = await _workflow_body(env)

        assert result["status"] == "all_filtered"
        assert result["items_rejected"] == 1
        assert not step_mocks["owner_resolution_step"].called
        assert not step_mocks["agent_outbox_step"].called


# ---------------------------------------------------------------------------
# Workflow body — enable_topics toggle
# ---------------------------------------------------------------------------


def _wire_full_pipeline(step_mocks: dict, populated: dict, merge_result_dict: dict) -> None:
    """Configure step_mocks for a successful end-to-end run (1 action item)."""
    step_mocks["extraction_step"].return_value = populated
    step_mocks["consolidation_step"].return_value = {"extraction": populated, "items_consolidated": 0}
    step_mocks["verification_step"].return_value = {
        "extraction": populated,
        "items_rejected": 0,
        "status": "ok",
    }
    step_mocks["owner_resolution_step"].return_value = {
        "extraction": populated,
        "contact_map": {},
    }
    step_mocks["matching_step"].return_value = {
        "match_results": [],
        "filtered_action_items": [],
    }
    step_mocks["merging_llm_step"].return_value = []
    step_mocks["merging_persist_step"].return_value = [merge_result_dict]
    step_mocks["topic_resolution_llm_step"].return_value = []
    step_mocks["topic_resolution_persist_step"].return_value = []


class TestWorkflowTopicsToggle:
    async def test_enable_topics_false_skips_s10a_and_s10b(self, step_mocks):
        env = _envelope()
        populated = _populated_extraction_dict()
        merge_result_dict = merge_result_to_dict(
            MergeResult(
                action_item_id="ai-1",
                action="created",
                was_new=True,
                version_created=False,
                linked_interaction_id=str(EXTRACTED_INTERACTION_ID),
                details={},
            )
        )
        _wire_full_pipeline(step_mocks, populated, merge_result_dict)
        result = await _workflow_body(env, enable_topics=False)

        assert result["status"] == "ok"
        assert result["topics_processed"] == 0
        assert not step_mocks["topic_resolution_llm_step"].called
        assert not step_mocks["topic_resolution_persist_step"].called

    async def test_enable_topics_true_calls_s10a_and_s10b(self, step_mocks):
        env = _envelope()
        populated = _populated_extraction_dict()
        merge_result_dict = merge_result_to_dict(
            MergeResult(
                action_item_id="ai-1",
                action="created",
                was_new=True,
                version_created=False,
                linked_interaction_id=str(EXTRACTED_INTERACTION_ID),
                details={},
            )
        )
        _wire_full_pipeline(step_mocks, populated, merge_result_dict)
        await _workflow_body(env, enable_topics=True)

        assert step_mocks["topic_resolution_llm_step"].called
        assert step_mocks["topic_resolution_persist_step"].called


# ---------------------------------------------------------------------------
# Workflow body — S14 agent_outbox gets interaction_dict (Rule 5)
# ---------------------------------------------------------------------------


class TestWorkflowAgentOutboxInteractionDictRule5:
    async def test_agent_outbox_receives_interaction_dict_from_extraction(self, step_mocks):
        """Rule 5 / Codex Round 1 HIGH #2: agent_outbox_step's 4th argument
        must be the interaction_dict extraction produced, NOT the envelope.
        The outbox dedup key needs the post-extraction interaction_id which
        may differ from the envelope's (envelope may not carry one)."""
        env = _envelope(interaction_id=None)
        populated = _populated_extraction_dict()
        merge_result_dict = merge_result_to_dict(
            MergeResult(
                action_item_id="ai-1",
                action="created",
                was_new=True,
                version_created=False,
                linked_interaction_id=str(EXTRACTED_INTERACTION_ID),
                details={},
            )
        )
        _wire_full_pipeline(step_mocks, populated, merge_result_dict)
        await _workflow_body(env)

        call = step_mocks["agent_outbox_step"].call_args
        # Positional args: merge_results, filtered_action_items, envelope_dict, interaction_dict
        assert len(call.args) == 4
        _, _, envelope_arg, interaction_arg = call.args
        # The 4th arg carries extraction's interaction_id, NOT envelope's
        assert interaction_arg["interaction_id"] == str(EXTRACTED_INTERACTION_ID)
        # Envelope arg is the original envelope (interaction_id absent)
        assert "interaction_id" not in envelope_arg or envelope_arg.get("interaction_id") is None


# ---------------------------------------------------------------------------
# Per-step contract tests
# ---------------------------------------------------------------------------


class TestVerificationStepFailOpen:
    """S4: verifier raise → fail-open returns extraction unchanged + status='skipped'."""

    async def test_verifier_exception_returns_skipped_with_unmodified_extraction(self, registered_clients):
        extraction_dict = _populated_extraction_dict()
        # Force pipeline.verifier.verify_batch to raise.
        mock_pipeline = MagicMock()
        mock_pipeline.verifier.verify_batch = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        with patch.object(action_item_steps, "_build_pipeline", return_value=mock_pipeline):
            result = await action_item_steps.verification_step(extraction_dict)

        assert result["status"] == "skipped"
        assert result["items_rejected"] == 0
        # Extraction is forwarded unmodified — same action_items count
        assert len(result["extraction"]["action_items"]) == len(extraction_dict["action_items"])
        assert "Verification failed" in result["rejection_reasons"][0]


class TestMergeContactsToDealStepFailOpen:
    """S7: side-branch with conditional execution + fail-open.

    EnvelopeV1 exposes opportunity_id + contacts via @property reads of
    ``extras.get(...)`` — see models/envelope.py:109-121. Tests must set
    these inside the extras dict, not as top-level envelope keys.
    """

    async def test_no_op_when_envelope_lacks_opportunity_id(self, registered_clients):
        env = _envelope()
        env["extras"] = {"contacts": [{"contact_id": "c-1", "email": "c@example.com"}]}
        # no opportunity_id in extras
        result = await action_item_steps.merge_contacts_to_deal_step(env)
        assert result["status"] == "no_op"
        assert result["count"] == 0

    async def test_no_op_when_envelope_has_no_contacts(self, registered_clients):
        env = _envelope()
        env["extras"] = {"opportunity_id": "opp-1"}  # no contacts
        result = await action_item_steps.merge_contacts_to_deal_step(env)
        assert result["status"] == "no_op"

    async def test_fail_open_when_helper_raises(self, registered_clients):
        env = _envelope()
        env["extras"] = {
            "opportunity_id": "opp-1",
            "contacts": [{"contact_id": "c-1", "email": "c@example.com"}],
        }
        with patch(
            "action_item_graph.workflows.action_item_steps.merge_contacts_to_deal",
            AsyncMock(side_effect=RuntimeError("neo4j down")),
        ):
            result = await action_item_steps.merge_contacts_to_deal_step(env)

        assert result["status"] == "skipped"
        assert result["count"] == 0
        assert "neo4j down" in result["error"]


class TestMergingPersistStepLengthMismatch:
    """S9b: parallel-list length skew is a real bug (Rule 4 / zip strict)."""

    async def test_raises_on_length_mismatch(self, registered_clients):
        # Provide 2 match_results but only 1 filtered_action_item → length skew
        match_results_list = [
            match_result_to_dict(
                MatchResult(
                    extracted_item=ExtractedActionItem(
                        action_item_text="x",
                        owner="O",
                        summary="s",
                        conversation_context="c",
                        topic=ExtractedTopic(name="T topic", context="ctx"),
                    ),
                    embedding=[0.1],
                    candidates=[],
                    decisions=[],
                    best_match=None,
                )
            ),
            match_result_to_dict(
                MatchResult(
                    extracted_item=ExtractedActionItem(
                        action_item_text="y",
                        owner="O",
                        summary="s2",
                        conversation_context="c",
                        topic=ExtractedTopic(name="T topic", context="ctx"),
                    ),
                    embedding=[0.1],
                    candidates=[],
                    decisions=[],
                    best_match=None,
                )
            ),
        ]
        filtered = [
            ActionItem(
                tenant_id=TEST_TENANT_ID,
                account_id="acct-1",
                action_item_text="x",
                summary="s",
                owner="O",
                owner_type="named",
                conversation_context="c",
            ).model_dump(mode="json"),
        ]  # only 1 item — length mismatch
        llm_results: list = [None, None]
        interaction_dict = _populated_extraction_dict()["interaction"]

        with pytest.raises(RuntimeError, match="list length mismatch"):
            await action_item_steps.merging_persist_step(
                match_results_list, filtered, llm_results, interaction_dict, {}
            )


class TestTopicResolutionLlmStepContextEnvelope:
    """S10a: action_item_text + owner in _action_item_context come from
    ExtractedActionItem (the upstream object), NOT the post-S5 ActionItem.
    Codex Round 2 HIGH — Rule 2."""

    async def test_action_item_context_sources_from_extracted_item(self, registered_clients):
        # The upstream extracted item has the raw owner mention "Sarah".
        extracted = ExtractedActionItem(
            action_item_text="Raw verbatim text from transcript",
            owner="Sarah",  # raw mention
            summary="Send sales deck",
            conversation_context="ctx",
            topic=ExtractedTopic(name="Sales Deck Delivery", context="Q1"),
        )
        match_result = MatchResult(
            extracted_item=extracted,
            embedding=[0.1],
            candidates=[],
            decisions=[],
            best_match=None,
        )
        # The post-S5 ActionItem has a normalized owner "Sarah O'Neill".
        ai = ActionItem(
            tenant_id=TEST_TENANT_ID,
            account_id="acct-1",
            action_item_text="post-resolution-text",
            summary="Send sales deck",
            owner="Sarah O'Neill",  # normalized!
            owner_type="named",
            conversation_context="ctx",
        )
        merge_result = MergeResult(
            action_item_id="ai-1",
            action="created",
            was_new=True,
            version_created=False,
            linked_interaction_id=None,
            details={},
        )

        # Stub TopicResolver.resolve_topic so we only inspect the
        # `_action_item_context` envelope construction.
        mock_resolver_class = MagicMock()
        mock_resolver = mock_resolver_class.return_value
        from action_item_graph.pipeline.topic_resolver import TopicResolutionResult
        mock_resolver.resolve_topic = AsyncMock(
            return_value=TopicResolutionResult(
                action_item_id="ai-1",
                action_item_summary="Send sales deck",
                extracted_topic=extracted.topic,
                decision=__import__(
                    "action_item_graph.pipeline.topic_resolver", fromlist=["TopicDecision"]
                ).TopicDecision.CREATE_NEW,
                topic_id=None,
                confidence=0.85,
            )
        )

        with patch.object(action_item_steps, "TopicResolver", mock_resolver_class):
            results = await action_item_steps.topic_resolution_llm_step(
                [match_result_to_dict(match_result)],
                [merge_result_to_dict(merge_result)],
                [ai.model_dump(mode="json")],
            )

        assert len(results) == 1
        ctx = results[0]["_action_item_context"]
        # The upstream raw owner wins, NOT the normalized owner from ActionItem
        assert ctx["owner"] == "Sarah"
        assert ctx["owner"] != "Sarah O'Neill"
        # action_item_text from the extracted item, not the ActionItem
        assert ctx["action_item_text"] == "Raw verbatim text from transcript"
        assert ctx["action_item_text"] != "post-resolution-text"


class TestPostgresDualWriteStep:
    """S13: no_op when no Postgres client; fail-open when raises."""

    async def test_no_op_when_postgres_client_is_none(self, registered_clients):
        # registered_clients fixture already sets postgres=None
        result = await action_item_steps.postgres_dual_write_step([], [], _populated_extraction_dict()["interaction"], [])
        assert result["status"] == "no_op"

    async def test_fail_open_when_dual_write_raises(self):
        clients = WorkflowClients(
            neo4j=MagicMock(),
            deal_neo4j=MagicMock(),
            openai=MagicMock(),
            postgres=MagicMock(),  # not None this time
        )
        _runtime._clients = clients

        mock_pipeline = MagicMock()
        mock_pipeline._dual_write_postgres = AsyncMock(side_effect=RuntimeError("pg down"))
        with patch.object(action_item_steps, "_build_pipeline", return_value=mock_pipeline):
            result = await action_item_steps.postgres_dual_write_step(
                [], [], _populated_extraction_dict()["interaction"], []
            )

        assert result["status"] == "skipped"
        assert result["rows_written"] == 0
        assert "pg down" in result["error"]


class TestAgentOutboxStepRule5:
    """S14: uses interaction.interaction_id (from the dict passed in),
    NOT envelope.interaction_id which may be absent. Codex Round 1 HIGH #2."""

    async def test_uses_interaction_dict_interaction_id_not_envelope(self):
        clients = WorkflowClients(
            neo4j=MagicMock(),
            deal_neo4j=MagicMock(),
            openai=MagicMock(),
            postgres=MagicMock(),
        )
        _runtime._clients = clients

        envelope = _envelope(interaction_id=None)  # NO envelope interaction_id
        interaction = _populated_extraction_dict()["interaction"]  # extracted's interaction_id

        mock_pipeline = MagicMock()
        mock_pipeline._write_agent_outbox = AsyncMock(return_value=None)
        with patch.object(action_item_steps, "_build_pipeline", return_value=mock_pipeline):
            await action_item_steps.agent_outbox_step([], [], envelope, interaction)

        # _write_agent_outbox was called with the post-extraction interaction_id
        kwargs = mock_pipeline._write_agent_outbox.call_args.kwargs
        assert kwargs["interaction_id"] == str(EXTRACTED_INTERACTION_ID)
