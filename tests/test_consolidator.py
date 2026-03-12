"""
Unit tests for within-batch action item consolidation.

Tests the consolidator's clustering logic (pure functions) and integration
with the LLM merging step.
"""

import math
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from action_item_graph.models.action_item import ActionItem
from action_item_graph.models.entities import Interaction, InteractionType
from action_item_graph.pipeline.consolidator import (
    INTRA_BATCH_SIMILARITY,
    ActionItemConsolidator,
    _cluster_items,
    _cosine_similarity,
)
from action_item_graph.pipeline.extractor import ExtractionOutput
from action_item_graph.prompts.consolidation_prompts import (
    ConsolidationDecision,
    ConsolidationResult,
)
from action_item_graph.prompts.extract_action_items import ExtractedActionItem, ExtractedTopic


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TENANT_ID = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')


def _make_embedding(base: list[float], dims: int = 1536) -> list[float]:
    """Pad a short vector to the full embedding dimension."""
    emb = base + [0.0] * (dims - len(base))
    norm = math.sqrt(sum(x * x for x in emb))
    if norm > 0:
        emb = [x / norm for x in emb]
    return emb


def _make_item(
    summary: str,
    owner: str = 'Sarah',
    embedding: list[float] | None = None,
) -> tuple[ActionItem, ExtractedActionItem]:
    """Create paired ActionItem + ExtractedActionItem for testing."""
    raw = ExtractedActionItem(
        action_item_text=f'Action: {summary}',
        owner=owner,
        summary=summary,
        conversation_context='Test context',
        topic=ExtractedTopic(name='Test Topic', context='Testing'),
        confidence=0.9,
    )
    ai = ActionItem(
        id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        action_item_text=raw.action_item_text,
        summary=summary,
        owner=owner,
        owner_type='named',
        conversation_context='Test context',
        embedding=embedding,
        embedding_current=embedding,
        confidence=0.9,
    )
    return ai, raw


def _make_extraction(
    items: list[tuple[ActionItem, ExtractedActionItem]],
) -> ExtractionOutput:
    """Build an ExtractionOutput from paired items."""
    interaction = Interaction(
        interaction_id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        interaction_type=InteractionType.TRANSCRIPT,
        content_text='test transcript',
        timestamp=datetime.now(),
    )
    return ExtractionOutput(
        interaction=interaction,
        action_items=[ai for ai, _ in items],
        raw_extractions=[raw for _, raw in items],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure function tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCosineSimiarity:
    """Test the cosine similarity calculation."""

    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_similar_vectors(self):
        a = [1.0, 0.1]
        b = [1.0, 0.2]
        sim = _cosine_similarity(a, b)
        assert sim > 0.99  # Very similar


class TestClustering:
    """Test the clustering algorithm."""

    def test_no_items(self):
        assert _cluster_items([]) == []

    def test_single_item(self):
        emb = _make_embedding([1.0, 0.0])
        clusters = _cluster_items([emb])
        assert len(clusters) == 1
        assert clusters[0] == [0]

    def test_identical_embeddings_cluster(self):
        emb = _make_embedding([1.0, 0.0])
        clusters = _cluster_items([emb, emb, emb])
        # All should be in one cluster
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_orthogonal_embeddings_no_cluster(self):
        emb_a = _make_embedding([1.0, 0.0])
        emb_b = _make_embedding([0.0, 1.0])
        clusters = _cluster_items([emb_a, emb_b])
        # Should be two separate singletons
        assert len(clusters) == 2

    def test_two_clusters(self):
        emb_a1 = _make_embedding([1.0, 0.0, 0.0])
        emb_a2 = _make_embedding([1.0, 0.01, 0.0])  # Very similar to a1
        emb_b1 = _make_embedding([0.0, 0.0, 1.0])
        emb_b2 = _make_embedding([0.0, 0.01, 1.0])  # Very similar to b1
        clusters = _cluster_items([emb_a1, emb_a2, emb_b1, emb_b2])
        assert len(clusters) == 2
        cluster_sets = [set(c) for c in clusters]
        assert {0, 1} in cluster_sets
        assert {2, 3} in cluster_sets

    def test_custom_threshold(self):
        emb_a = _make_embedding([1.0, 0.0])
        emb_b = _make_embedding([0.9, 0.1])
        # With high threshold, these might not cluster
        clusters_high = _cluster_items([emb_a, emb_b], threshold=0.999)
        assert len(clusters_high) == 2  # Too different for 0.999

        # With lower threshold, they should cluster
        clusters_low = _cluster_items([emb_a, emb_b], threshold=0.9)
        assert len(clusters_low) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Consolidator integration tests (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────


class TestConsolidator:
    """Test the ActionItemConsolidator with mocked OpenAI calls."""

    @pytest.mark.asyncio
    async def test_no_duplicates_no_llm_call(self):
        """When all items are unique, no LLM call should be made."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock()

        consolidator = ActionItemConsolidator(mock_client)

        emb_a = _make_embedding([1.0, 0.0, 0.0])
        emb_b = _make_embedding([0.0, 1.0, 0.0])
        emb_c = _make_embedding([0.0, 0.0, 1.0])

        items = [
            _make_item('Send pricing deck', embedding=emb_a),
            _make_item('Schedule demo', embedding=emb_b),
            _make_item('Review contract', embedding=emb_c),
        ]

        extraction = _make_extraction(items)
        result, removed = await consolidator.consolidate(extraction)

        assert result.count == 3
        assert removed == 0
        mock_client.chat_completion_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_item_passthrough(self):
        """Single item should pass through without consolidation."""
        mock_client = MagicMock()
        consolidator = ActionItemConsolidator(mock_client)

        emb = _make_embedding([1.0, 0.0])
        items = [_make_item('Send pricing deck', embedding=emb)]
        extraction = _make_extraction(items)

        result, removed = await consolidator.consolidate(extraction)
        assert result.count == 1
        assert removed == 0

    @pytest.mark.asyncio
    async def test_duplicates_merged(self):
        """Near-duplicate items should be merged via LLM."""
        mock_client = MagicMock()

        # LLM returns consolidation decision
        mock_client.chat_completion_structured = AsyncMock(
            return_value=ConsolidationResult(
                groups=[
                    ConsolidationDecision(
                        primary_index=0,
                        merged_summary='Send evaluation materials (pricing, SOC2, whitepaper)',
                        merged_context='Client needs materials for budget approval and security review',
                        reasoning='All three items are sub-deliverables of the same evaluation package',
                    )
                ],
                keep_indices=[],
            )
        )

        consolidator = ActionItemConsolidator(mock_client)

        # Three near-identical embeddings (same deliverable, different wording)
        emb = _make_embedding([1.0, 0.1, 0.0])
        items = [
            _make_item('Send pricing deck', embedding=emb),
            _make_item('Send SOC2 report', embedding=emb),
            _make_item('Send security whitepaper', embedding=emb),
        ]

        extraction = _make_extraction(items)
        result, removed = await consolidator.consolidate(extraction)

        assert result.count == 1  # All merged into one
        assert removed == 2
        assert 'evaluation materials' in result.action_items[0].summary

    @pytest.mark.asyncio
    async def test_mixed_duplicates_and_unique(self):
        """Mix of duplicates and unique items: only duplicates are merged."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            return_value=ConsolidationResult(
                groups=[
                    ConsolidationDecision(
                        primary_index=0,
                        merged_summary='Send evaluation package',
                        merged_context='Combined context',
                        reasoning='Sub-deliverables of same package',
                    )
                ],
                keep_indices=[],
            )
        )

        consolidator = ActionItemConsolidator(mock_client)

        emb_similar = _make_embedding([1.0, 0.1])
        emb_unique = _make_embedding([0.0, 1.0])

        items = [
            _make_item('Send pricing deck', embedding=emb_similar),
            _make_item('Send SOC2 report', embedding=emb_similar),
            _make_item('Schedule follow-up call', embedding=emb_unique),
        ]

        extraction = _make_extraction(items)
        result, removed = await consolidator.consolidate(extraction)

        assert result.count == 2  # 1 merged + 1 unique
        assert removed == 1

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        """If LLM call fails, fall back to keeping the first item in cluster."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            side_effect=Exception('LLM timeout')
        )

        consolidator = ActionItemConsolidator(mock_client)

        emb = _make_embedding([1.0, 0.0])
        items = [
            _make_item('Send pricing deck', embedding=emb),
            _make_item('Send pricing breakdown', embedding=emb),
        ]

        extraction = _make_extraction(items)
        result, removed = await consolidator.consolidate(extraction)

        # Should still consolidate (fallback to first item) but not crash
        assert result.count == 1
        assert removed == 1

    @pytest.mark.asyncio
    async def test_items_without_embeddings_kept(self):
        """Items missing embeddings should pass through untouched."""
        mock_client = MagicMock()
        consolidator = ActionItemConsolidator(mock_client)

        items = [
            _make_item('Send pricing deck', embedding=None),
            _make_item('Schedule demo', embedding=None),
        ]

        extraction = _make_extraction(items)
        result, removed = await consolidator.consolidate(extraction)

        assert result.count == 2
        assert removed == 0
