"""
Tests for Deal entity resolution (matcher) with mocked Neo4j and OpenAI.

Tests cover:
- No candidates → create_new (fast path)
- Auto-match: top candidate >= 0.90 → auto_match, no LLM call
- LLM match: borderline candidate 0.70-0.90 → LLM confirms → llm_match
- LLM reject: borderline candidate → LLM rejects → create_new
- Below threshold: all candidates < 0.70 → create_new
- Multiple borderline candidates: evaluates in order, stops on first match
- Deduplication prompt verification
- Custom threshold configuration

Run with: pytest tests/test_deal_matcher.py -v

No API keys or Neo4j required — all calls are mocked.
"""

from uuid import UUID

import pytest
from unittest.mock import AsyncMock

from deal_graph.models.extraction import (
    DealDeduplicationDecision,
    ExtractedDeal,
)
from deal_graph.pipeline.matcher import DealMatcher, DealMatchCandidate


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_neo4j():
    """Create a mocked DealNeo4jClient."""
    client = AsyncMock()
    client.search_deals_both_embeddings = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_openai():
    """Create a mocked OpenAI client."""
    client = AsyncMock()
    client.chat_completion_structured = AsyncMock()
    return client


@pytest.fixture
def matcher(mock_neo4j, mock_openai):
    """Create a DealMatcher with mocked clients and explicit thresholds."""
    return DealMatcher(
        neo4j_client=mock_neo4j,
        openai_client=mock_openai,
        similarity_threshold=0.70,
        auto_match_threshold=0.90,
    )


@pytest.fixture
def sample_extracted_deal():
    """A typical extracted deal for matching tests."""
    return ExtractedDeal(
        opportunity_name='Acme Corp Data Platform',
        opportunity_summary='Acme evaluating data platform for CRM consolidation.',
        stage_assessment='qualification',
        metrics='$200K annual cost in manual reconciliation',
        economic_buyer='Maria Chen, VP of Engineering',
        decision_criteria='SOC2 compliance required',
        decision_process='POC → security review → procurement',
        identified_pain='Data silos between three CRM systems',
        champion='James',
        estimated_amount=150000.0,
        confidence=0.92,
        reasoning='Strong MEDDIC signals across all dimensions.',
    )


@pytest.fixture
def sample_embedding():
    """A mock 1536-dimensional embedding vector."""
    return [0.01] * 1536


@pytest.fixture
def existing_deal_node():
    """Properties of an existing Deal node as returned by vector search."""
    return {
        'tenant_id': '550e8400-e29b-41d4-a716-446655440000',
        'opportunity_id': '019c1fa0-1111-7000-8000-000000000002',
        'name': 'Acme Data Platform Deal',
        'stage': 'qualification',
        'amount': 150000.0,
        'opportunity_summary': 'Acme Corp evaluating our data platform.',
        'meddic_metrics': '$200K annual cost',
        'meddic_economic_buyer': 'Maria Chen, VP Engineering',
        'meddic_decision_criteria': 'SOC2 compliance',
        'meddic_decision_process': 'POC then security review',
        'meddic_identified_pain': 'CRM data silos',
        'meddic_champion': 'James',
    }


TENANT_ID = UUID('550e8400-e29b-41d4-a716-446655440000')


def _make_search_result(node: dict, score: float) -> dict:
    """Build a search result in the format returned by search_deals_both_embeddings."""
    return {'node': node, 'score': score}


# =============================================================================
# No Candidates → Create New
# =============================================================================


class TestNoCandidates:
    """When vector search returns zero candidates."""

    @pytest.mark.asyncio
    async def test_create_new_when_no_candidates(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding,
    ):
        """Zero candidates → create_new, no LLM call."""
        mock_neo4j.search_deals_both_embeddings.return_value = []

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'create_new'
        assert result.matched_deal is None
        assert result.decision is None
        assert result.candidates_evaluated == 0
        assert result.all_candidates == []
        mock_openai.chat_completion_structured.assert_not_called()


# =============================================================================
# Auto-Match (>= 0.90)
# =============================================================================


class TestAutoMatch:
    """When top candidate scores >= auto_match_threshold."""

    @pytest.mark.asyncio
    async def test_auto_match_high_score(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Score >= 0.90 → auto_match without LLM."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.95),
        ]

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'auto_match'
        assert result.matched_deal is not None
        assert result.matched_deal.opportunity_id == '019c1fa0-1111-7000-8000-000000000002'
        assert result.matched_deal.similarity_score == 0.95
        assert result.decision is None  # No LLM call
        mock_openai.chat_completion_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_match_exact_threshold(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Score == 0.90 → auto_match (threshold is inclusive)."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.90),
        ]

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'auto_match'
        assert result.matched_deal is not None
        mock_openai.chat_completion_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_match_picks_top_candidate(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Multiple candidates above threshold → picks highest score."""
        second_node = {**existing_deal_node, 'opportunity_id': '019c1fa0-3333-7000-8000-000000000004'}
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.96),
            _make_search_result(second_node, 0.91),
        ]

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'auto_match'
        assert result.matched_deal.opportunity_id == '019c1fa0-1111-7000-8000-000000000002'
        assert result.matched_deal.similarity_score == 0.96
        assert len(result.all_candidates) == 2

    @pytest.mark.asyncio
    async def test_auto_match_preserves_node_properties(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Auto-matched candidate carries full node properties."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.93),
        ]

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.matched_deal.node_properties['name'] == 'Acme Data Platform Deal'
        assert result.matched_deal.node_properties['amount'] == 150000.0


# =============================================================================
# LLM Match (0.70-0.90)
# =============================================================================


class TestLLMMatch:
    """When candidate is in borderline zone and LLM confirms match."""

    @pytest.mark.asyncio
    async def test_llm_confirms_match(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Score 0.70-0.90, LLM says same deal → llm_match."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.82),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=True,
            recommendation='merge',
            confidence=0.88,
            reasoning='Same account, same pain point, same economic buyer.',
        )

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'llm_match'
        assert result.matched_deal is not None
        assert result.matched_deal.opportunity_id == '019c1fa0-1111-7000-8000-000000000002'
        assert result.decision is not None
        assert result.decision.is_same_deal is True
        assert result.decision.recommendation == 'merge'
        mock_openai.chat_completion_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_rejects_match(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Score 0.70-0.90, LLM says different deal → create_new."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.75),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=False,
            recommendation='create_new',
            confidence=0.80,
            reasoning='Different products — one is data platform, other is security audit.',
        )

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'create_new'
        assert result.matched_deal is None
        assert result.decision is None  # No winning decision stored
        mock_openai.chat_completion_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_is_same_but_recommends_create_new(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """LLM says is_same_deal=True but recommendation=create_new → create_new."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.78),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=True,
            recommendation='create_new',
            confidence=0.55,
            reasoning='Possibly same deal but too uncertain to merge safely.',
        )

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        # Both is_same_deal AND recommendation='merge' required
        assert result.match_type == 'create_new'

    @pytest.mark.asyncio
    async def test_multiple_borderline_first_rejected_second_accepted(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """First borderline candidate rejected, second accepted by LLM."""
        second_node = {
            **existing_deal_node,
            'opportunity_id': '019c1fa0-3333-7000-8000-000000000004',
            'name': 'Acme CRM Integration',
        }
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.85),
            _make_search_result(second_node, 0.76),
        ]
        mock_openai.chat_completion_structured.side_effect = [
            # First candidate rejected
            DealDeduplicationDecision(
                is_same_deal=False,
                recommendation='create_new',
                confidence=0.75,
                reasoning='Different scope.',
            ),
            # Second candidate accepted
            DealDeduplicationDecision(
                is_same_deal=True,
                recommendation='merge',
                confidence=0.82,
                reasoning='Same underlying opportunity, different name.',
            ),
        ]

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'llm_match'
        assert result.matched_deal.opportunity_id == '019c1fa0-3333-7000-8000-000000000004'
        assert mock_openai.chat_completion_structured.call_count == 2

    @pytest.mark.asyncio
    async def test_all_borderline_candidates_rejected(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """All borderline candidates rejected by LLM → create_new."""
        second_node = {**existing_deal_node, 'opportunity_id': '019c1fa0-3333-7000-8000-000000000004'}
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.84),
            _make_search_result(second_node, 0.72),
        ]
        mock_openai.chat_completion_structured.side_effect = [
            DealDeduplicationDecision(
                is_same_deal=False, recommendation='create_new',
                confidence=0.70, reasoning='Different.',
            ),
            DealDeduplicationDecision(
                is_same_deal=False, recommendation='create_new',
                confidence=0.65, reasoning='Also different.',
            ),
        ]

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'create_new'
        assert result.matched_deal is None
        assert mock_openai.chat_completion_structured.call_count == 2


# =============================================================================
# Below Threshold (< 0.70)
# =============================================================================


class TestBelowThreshold:
    """When all candidates score below similarity_threshold."""

    @pytest.mark.asyncio
    async def test_below_threshold_no_llm_call(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """All candidates < 0.70 → create_new, no LLM call."""
        # search_deals_both_embeddings already filters by min_score,
        # so this scenario means the search returned nothing above threshold.
        mock_neo4j.search_deals_both_embeddings.return_value = []

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        assert result.match_type == 'create_new'
        mock_openai.chat_completion_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_mix_of_above_and_below_threshold(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """One candidate at 0.75, one at 0.65 (filtered out by search)."""
        # The 0.65 candidate is filtered by search_deals_both_embeddings min_score=0.70
        # So only the 0.75 candidate appears
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.75),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=False, recommendation='create_new',
            confidence=0.60, reasoning='Different opportunity.',
        )

        result = await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        # 0.75 goes to LLM, LLM rejects → create_new
        assert result.match_type == 'create_new'
        mock_openai.chat_completion_structured.assert_called_once()


# =============================================================================
# Prompt Verification
# =============================================================================


class TestDedupPrompt:
    """Verify the deduplication prompt is built correctly."""

    @pytest.mark.asyncio
    async def test_prompt_contains_existing_deal_context(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Dedup prompt should include existing deal's MEDDIC fields."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.80),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=False, recommendation='create_new',
            confidence=0.50, reasoning='Test.',
        )

        await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        messages = call_args.kwargs['messages']
        user_content = messages[1]['content']

        # Existing deal context
        assert 'Acme Data Platform Deal' in user_content
        assert 'Maria Chen' in user_content
        assert 'SOC2' in user_content

        # Extracted deal context
        assert 'Acme Corp Data Platform' in user_content
        assert sample_extracted_deal.reasoning in user_content

        # Similarity score
        assert '0.800' in user_content

    @pytest.mark.asyncio
    async def test_prompt_system_message_has_bias_warning(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """System prompt should emphasize bias toward create_new."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.80),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=False, recommendation='create_new',
            confidence=0.50, reasoning='Test.',
        )

        await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        system_content = call_args.kwargs['messages'][0]['content']
        assert 'false merge' in system_content.lower()
        assert 'create_new' in system_content

    @pytest.mark.asyncio
    async def test_response_model_is_dedup_decision(
        self, matcher, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """LLM call should use DealDeduplicationDecision as response model."""
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.80),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=False, recommendation='create_new',
            confidence=0.50, reasoning='Test.',
        )

        await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        assert call_args.kwargs['response_model'] is DealDeduplicationDecision


# =============================================================================
# Search Integration
# =============================================================================


class TestSearchIntegration:
    """Verify search parameters passed to Neo4j client."""

    @pytest.mark.asyncio
    async def test_passes_tenant_and_account_to_search(
        self, matcher, mock_neo4j,
        sample_extracted_deal, sample_embedding,
    ):
        """Search should receive tenant_id, account_id, and min_score."""
        mock_neo4j.search_deals_both_embeddings.return_value = []

        await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
            account_id='acct_acme_001',
        )

        call_args = mock_neo4j.search_deals_both_embeddings.call_args
        assert call_args.kwargs['tenant_id'] == str(TENANT_ID)
        assert call_args.kwargs['account_id'] == 'acct_acme_001'
        assert call_args.kwargs['min_score'] == 0.70

    @pytest.mark.asyncio
    async def test_passes_embedding_to_search(
        self, matcher, mock_neo4j,
        sample_extracted_deal, sample_embedding,
    ):
        """Search should receive the extracted deal's embedding."""
        mock_neo4j.search_deals_both_embeddings.return_value = []

        await matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        call_args = mock_neo4j.search_deals_both_embeddings.call_args
        assert call_args.kwargs['embedding'] is sample_embedding


# =============================================================================
# Custom Thresholds
# =============================================================================


class TestCustomThresholds:
    """Verify configurable thresholds change behavior."""

    @pytest.mark.asyncio
    async def test_stricter_auto_match_threshold(
        self, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Raising auto_match to 0.95 sends 0.92 score to LLM instead of auto-matching."""
        strict_matcher = DealMatcher(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
            similarity_threshold=0.70,
            auto_match_threshold=0.95,
        )
        mock_neo4j.search_deals_both_embeddings.return_value = [
            _make_search_result(existing_deal_node, 0.92),
        ]
        mock_openai.chat_completion_structured.return_value = DealDeduplicationDecision(
            is_same_deal=True, recommendation='merge',
            confidence=0.90, reasoning='Same deal.',
        )

        result = await strict_matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        # 0.92 is below 0.95 threshold, so goes to LLM
        assert result.match_type == 'llm_match'
        mock_openai.chat_completion_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_lower_similarity_threshold(
        self, mock_neo4j, mock_openai,
        sample_extracted_deal, sample_embedding, existing_deal_node,
    ):
        """Lowering similarity_threshold to 0.60 passes min_score=0.60 to search."""
        loose_matcher = DealMatcher(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
            similarity_threshold=0.60,
            auto_match_threshold=0.90,
        )
        mock_neo4j.search_deals_both_embeddings.return_value = []

        await loose_matcher.find_matches(
            extracted_deal=sample_extracted_deal,
            embedding=sample_embedding,
            tenant_id=TENANT_ID,
        )

        call_args = mock_neo4j.search_deals_both_embeddings.call_args
        assert call_args.kwargs['min_score'] == 0.60
