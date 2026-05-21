"""Round-trip tests for the DBOS step-boundary serialization helpers.

Rule 3 (memory/pattern_dbos_workflow_parity_rules.md): every type that
crosses a step boundary must round-trip cleanly through JSON. UUID,
datetime, tuples, and frozen dataclasses are the easy bugs.

These tests pin the contract so a future refactor that drops `mode='json'`
or flattens a wrong tuple doesn't silently corrupt DBOS checkpoints.
"""

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from action_item_graph.models.action_item import ActionItem
from action_item_graph.models.entities import Interaction, InteractionType
from action_item_graph.pipeline.extractor import ExtractionOutput
from action_item_graph.pipeline.matcher import MatchCandidate, MatchResult
from action_item_graph.pipeline.merger import MergeResult
from action_item_graph.pipeline.topic_executor import TopicExecutionResult
from action_item_graph.pipeline.topic_resolver import (
    TopicCandidate,
    TopicDecision,
    TopicResolutionResult,
)
from action_item_graph.prompts.extract_action_items import (
    DeduplicationDecision,
    ExtractedActionItem,
    ExtractedTopic,
)
from action_item_graph.workflows._serialization import (
    extraction_from_dict,
    extraction_to_dict,
    match_candidate_from_dict,
    match_candidate_to_dict,
    match_result_from_dict,
    match_result_to_dict,
    merge_result_from_dict,
    merge_result_to_dict,
    topic_candidate_from_dict,
    topic_candidate_to_dict,
    topic_execution_from_dict,
    topic_execution_to_dict,
    topic_resolution_from_dict,
    topic_resolution_to_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")
TEST_INTERACTION_ID = UUID("550e8400-e29b-41d4-a716-446655440000")


def _build_interaction() -> Interaction:
    return Interaction(
        interaction_id=TEST_INTERACTION_ID,
        tenant_id=TEST_TENANT_ID,
        account_id="acct-1",
        interaction_type=InteractionType.TRANSCRIPT,
        title="Sample meeting",
        content_text="A: I'll send the deck by Friday.\nB: Great.",
        timestamp=datetime(2026, 2, 14, 15, 30, tzinfo=timezone.utc),
        duration_seconds=30,
    )


def _build_extracted_topic() -> ExtractedTopic:
    return ExtractedTopic(name="Sales Deck Delivery", context="Q1 sales pipeline")


def _build_extracted_action_item() -> ExtractedActionItem:
    return ExtractedActionItem(
        action_item_text="Send the deck by Friday",
        owner="Sarah",
        summary="Send sales deck",
        conversation_context="Sarah committed to deliver",
        topic=_build_extracted_topic(),
    )


def _build_action_item() -> ActionItem:
    return ActionItem(
        tenant_id=TEST_TENANT_ID,
        account_id="acct-1",
        action_item_text="Send the deck by Friday",
        summary="Send sales deck",
        owner="Sarah",
        owner_type="named",
        conversation_context="Sarah committed to deliver",
    )


def _build_dedup_decision() -> DeduplicationDecision:
    return DeduplicationDecision(
        is_same_item=True,
        merge_recommendation="merge",
        reasoning="Same commitment, same person",
        confidence=0.9,
    )


def _build_match_candidate() -> MatchCandidate:
    return MatchCandidate(
        action_item_id="ai-existing-1",
        node_properties={"summary": "Send deck", "status": "pending"},
        similarity_score=0.92,
        matched_via="original",
    )


# ---------------------------------------------------------------------------
# ExtractionOutput
# ---------------------------------------------------------------------------


class TestExtractionSerialization:
    def test_round_trip_preserves_action_items_count(self):
        extraction = ExtractionOutput(
            interaction=_build_interaction(),
            action_items=[_build_action_item(), _build_action_item()],
            raw_extractions=[_build_extracted_action_item()],
            extraction_notes="from-test",
        )
        d = extraction_to_dict(extraction)
        roundtripped = extraction_from_dict(d)

        assert len(roundtripped.action_items) == 2
        assert len(roundtripped.raw_extractions) == 1
        assert roundtripped.extraction_notes == "from-test"

    def test_json_dumps_succeeds(self):
        """Rule 3: the dict must serialize via json.dumps — no UUID or datetime
        leaks past the boundary."""
        extraction = ExtractionOutput(
            interaction=_build_interaction(),
            action_items=[_build_action_item()],
            raw_extractions=[_build_extracted_action_item()],
            extraction_notes=None,
        )
        d = extraction_to_dict(extraction)
        # Should not raise.
        json.dumps(d)

    def test_double_roundtrip_is_idempotent(self):
        """extraction_to_dict(extraction_from_dict(extraction_to_dict(e))) == extraction_to_dict(e)"""
        extraction = ExtractionOutput(
            interaction=_build_interaction(),
            action_items=[_build_action_item()],
            raw_extractions=[_build_extracted_action_item()],
            extraction_notes=None,
        )
        d1 = extraction_to_dict(extraction)
        d2 = extraction_to_dict(extraction_from_dict(d1))
        assert d1 == d2

    def test_uuid_fields_serialize_as_strings(self):
        extraction = ExtractionOutput(
            interaction=_build_interaction(),
            action_items=[_build_action_item()],
            raw_extractions=[_build_extracted_action_item()],
            extraction_notes=None,
        )
        d = extraction_to_dict(extraction)
        assert isinstance(d["interaction"]["tenant_id"], str)
        assert isinstance(d["interaction"]["interaction_id"], str)


# ---------------------------------------------------------------------------
# MatchResult — tuples in decisions/best_match are the historical break point
# ---------------------------------------------------------------------------


class TestMatchResultSerialization:
    def test_round_trip_with_best_match_tuple(self):
        candidate = _build_match_candidate()
        decision = _build_dedup_decision()
        match = MatchResult(
            extracted_item=_build_extracted_action_item(),
            embedding=[0.1, 0.2, 0.3],
            candidates=[candidate],
            decisions=[(candidate, decision)],
            best_match=(candidate, decision),
        )
        d = match_result_to_dict(match)
        roundtripped = match_result_from_dict(d)

        assert roundtripped.best_match is not None
        rt_candidate, rt_decision = roundtripped.best_match
        assert rt_candidate.action_item_id == candidate.action_item_id
        assert rt_decision.merge_recommendation == decision.merge_recommendation

    def test_round_trip_without_best_match(self):
        match = MatchResult(
            extracted_item=_build_extracted_action_item(),
            embedding=[0.1],
            candidates=[],
            decisions=[],
            best_match=None,
        )
        d = match_result_to_dict(match)
        roundtripped = match_result_from_dict(d)

        assert roundtripped.best_match is None
        assert roundtripped.candidates == []
        assert roundtripped.decisions == []

    def test_decisions_tuples_round_trip_as_dicts(self):
        """Rule 3 specific: tuples in `decisions` flatten to dicts at the
        boundary and re-tuple-ize on the way back."""
        candidate = _build_match_candidate()
        decision = _build_dedup_decision()
        match = MatchResult(
            extracted_item=_build_extracted_action_item(),
            embedding=[0.1],
            candidates=[candidate],
            decisions=[(candidate, decision)],
            best_match=None,
        )
        d = match_result_to_dict(match)
        # Wire format: decisions is a list of dicts, not a list of tuples
        assert isinstance(d["decisions"], list)
        assert isinstance(d["decisions"][0], dict)
        assert "candidate" in d["decisions"][0]
        assert "decision" in d["decisions"][0]

        # After round-trip, each entry is a tuple again
        rt = match_result_from_dict(d)
        assert isinstance(rt.decisions[0], tuple)


class TestMatchCandidateSerialization:
    def test_dataclass_round_trips_via_asdict(self):
        candidate = _build_match_candidate()
        d = match_candidate_to_dict(candidate)
        roundtripped = match_candidate_from_dict(d)
        assert roundtripped == candidate


# ---------------------------------------------------------------------------
# MergeResult
# ---------------------------------------------------------------------------


class TestMergeResultSerialization:
    def test_round_trip_via_asdict(self):
        merge = MergeResult(
            action_item_id="ai-1",
            action="created",
            was_new=True,
            version_created=False,
            linked_interaction_id=str(TEST_INTERACTION_ID),
            details={"note": "first time"},
        )
        d = merge_result_to_dict(merge)
        roundtripped = merge_result_from_dict(d)
        assert roundtripped == merge


# ---------------------------------------------------------------------------
# TopicResolutionResult
# ---------------------------------------------------------------------------


class TestTopicResolutionSerialization:
    def test_round_trip_create_new(self):
        result = TopicResolutionResult(
            action_item_id="ai-1",
            action_item_summary="Send deck",
            extracted_topic=_build_extracted_topic(),
            decision=TopicDecision.CREATE_NEW,
            topic_id=None,
            confidence=0.85,
            embedding=[0.1, 0.2],
        )
        d = topic_resolution_to_dict(result)
        rt = topic_resolution_from_dict(d)

        assert rt.action_item_id == "ai-1"
        assert rt.decision == TopicDecision.CREATE_NEW
        assert rt.topic_id is None
        assert rt.embedding == [0.1, 0.2]
        assert rt.best_candidate is None

    def test_round_trip_link_existing_with_best_candidate(self):
        candidate = TopicCandidate(
            topic_id="topic-1",
            name="Sales",
            canonical_name="sales",
            summary="Sales work",
            action_item_count=3,
            similarity=0.91,
        )
        result = TopicResolutionResult(
            action_item_id="ai-1",
            action_item_summary="Send deck",
            extracted_topic=_build_extracted_topic(),
            decision=TopicDecision.LINK_EXISTING,
            topic_id="topic-1",
            confidence=0.91,
            candidates=[candidate],
            best_candidate=candidate,
            embedding=[0.1, 0.2],
        )
        d = topic_resolution_to_dict(result)
        rt = topic_resolution_from_dict(d)

        assert rt.decision == TopicDecision.LINK_EXISTING
        assert rt.topic_id == "topic-1"
        assert rt.best_candidate is not None
        assert rt.best_candidate.topic_id == "topic-1"


class TestTopicCandidateSerialization:
    def test_dataclass_round_trip(self):
        candidate = TopicCandidate(
            topic_id="t-1",
            name="N",
            canonical_name="n",
            summary="s",
            action_item_count=2,
            similarity=0.7,
        )
        d = topic_candidate_to_dict(candidate)
        rt = topic_candidate_from_dict(d)
        assert rt == candidate


# ---------------------------------------------------------------------------
# TopicExecutionResult
# ---------------------------------------------------------------------------


class TestTopicExecutionSerialization:
    def test_round_trip(self):
        execution = TopicExecutionResult(
            action_item_id="ai-1",
            topic_id="topic-1",
            topic_name="Sales",
            action="linked",
            was_new=False,
            version_created=False,
            summary_updated=False,
            embedding_updated=False,
            details={"matched": True},
        )
        d = topic_execution_to_dict(execution)
        rt = topic_execution_from_dict(d)
        assert rt == execution
