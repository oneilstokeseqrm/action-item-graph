"""
Tests for Deal merger service with mocked Neo4j and OpenAI.

Tests cover:
- Create new path: builds Deal model, calls repo.create_deal
- Merge path: LLM synthesis → version snapshot → update deal (in that order)
- Evolution summary accumulation
- MEDDIC field flattening (None = keep existing, populated = update)
- meddic_completeness recomputation
- Stage mapping from LLM implied_stage
- Conditional embedding update
- DealMergeResult structure

Run with: pytest tests/test_deal_merger.py -v

No API keys or Neo4j required — all calls are mocked.
"""

from unittest.mock import AsyncMock, call, patch
from uuid import UUID, uuid4

import pytest

from deal_graph.models.extraction import (
    ExtractedDeal,
    MergedDeal,
)
from deal_graph.pipeline.matcher import DealMatchCandidate, DealMatchResult
from deal_graph.pipeline.merger import DealMerger


# =============================================================================
# Fixtures
# =============================================================================


TENANT_ID = UUID('550e8400-e29b-41d4-a716-446655440000')
INTERACTION_ID = UUID('660e8400-e29b-41d4-a716-446655440001')


@pytest.fixture
def mock_neo4j():
    """Create a mocked DealNeo4jClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_openai():
    """Create a mocked OpenAI client."""
    client = AsyncMock()
    client.chat_completion_structured = AsyncMock()
    client.create_embedding = AsyncMock(return_value=[0.05] * 1536)
    return client


@pytest.fixture
def merger(mock_neo4j, mock_openai):
    """Create a DealMerger with mocked clients."""
    m = DealMerger(neo4j_client=mock_neo4j, openai_client=mock_openai)
    # Mock the repository methods directly
    m.repository = AsyncMock()
    m.repository.create_deal = AsyncMock(return_value={
        'tenant_id': str(TENANT_ID),
        'opportunity_id': '019c1fa0-0000-7000-8000-000000000001',
        'name': 'Test Deal',
        'stage': 'qualification',
    })
    m.repository.update_deal = AsyncMock(return_value={
        'tenant_id': str(TENANT_ID),
        'opportunity_id': '019c1fa0-1111-7000-8000-000000000002',
        'name': 'Acme Data Platform',
        'stage': 'proposal',
        'version': 2,
    })
    m.repository.create_version_snapshot = AsyncMock(return_value={
        'version_id': str(uuid4()),
        'version': 1,
    })
    return m


@pytest.fixture
def sample_extracted_deal():
    """A typical extracted deal for merger tests."""
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
        reasoning='Strong MEDDIC signals.',
    )


@pytest.fixture
def sample_embedding():
    """A mock 1536-dimensional embedding vector."""
    return [0.01] * 1536


@pytest.fixture
def existing_deal_node():
    """Existing Deal node properties from the graph."""
    return {
        'tenant_id': str(TENANT_ID),
        'opportunity_id': '019c1fa0-1111-7000-8000-000000000002',
        'name': 'Acme Data Platform Deal',
        'stage': 'qualification',
        'amount': 150000.0,
        'opportunity_summary': 'Acme Corp evaluating our data platform.',
        'evolution_summary': 'Initial discovery call revealed data silo pain points (Jan 15).',
        'meddic_metrics': '$200K annual cost',
        'meddic_economic_buyer': 'Maria Chen, VP Engineering',
        'meddic_decision_criteria': 'SOC2 compliance',
        'meddic_decision_process': 'POC then security review',
        'meddic_identified_pain': 'CRM data silos',
        'meddic_champion': 'James',
        'version': 1,
    }


@pytest.fixture
def create_new_match_result(sample_extracted_deal, sample_embedding):
    """A DealMatchResult indicating create_new."""
    return DealMatchResult(
        extracted_deal=sample_extracted_deal,
        embedding=sample_embedding,
        match_type='create_new',
        matched_deal=None,
        decision=None,
        candidates_evaluated=0,
    )


@pytest.fixture
def auto_match_result(sample_extracted_deal, sample_embedding, existing_deal_node):
    """A DealMatchResult indicating auto_match."""
    return DealMatchResult(
        extracted_deal=sample_extracted_deal,
        embedding=sample_embedding,
        match_type='auto_match',
        matched_deal=DealMatchCandidate(
            opportunity_id='019c1fa0-1111-7000-8000-000000000002',
            node_properties=existing_deal_node,
            similarity_score=0.95,
        ),
        decision=None,
        candidates_evaluated=1,
    )


@pytest.fixture
def sample_merged_deal():
    """A typical MergedDeal LLM output."""
    return MergedDeal(
        opportunity_summary='Acme evaluating data platform for CRM consolidation. Budget confirmed at $150K.',
        evolution_summary=(
            'Initial discovery call revealed data silo pain points (Jan 15). '
            'Follow-up call confirmed budget range and identified Maria Chen as economic buyer (Jan 22).'
        ),
        change_narrative=(
            'Follow-up confirmed $150K budget and Maria Chen as final decision maker. '
            'SOC2 compliance added as hard requirement. Deal remains in qualification.'
        ),
        changed_fields=['meddic_decision_criteria', 'opportunity_summary'],
        metrics=None,  # Keep existing
        economic_buyer=None,  # Keep existing
        decision_criteria='SOC2 compliance required; must integrate with existing ERP',  # Updated (additive)
        decision_process=None,  # Keep existing
        identified_pain=None,  # Keep existing
        champion=None,  # Keep existing
        implied_stage=None,  # No change
        stage_reasoning='Deal remains in qualification — no signals of progression.',
        amount=None,  # Keep existing
        should_update_embedding=False,
    )


# =============================================================================
# Create New Path
# =============================================================================


class TestCreateNew:
    """Test the create_new path — no existing deal to merge into."""

    @pytest.mark.asyncio
    async def test_creates_deal_via_repository(
        self, merger, create_new_match_result,
    ):
        """Should call repo.create_deal with a Deal model."""
        result = await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
            account_id='acct_acme_001',
            source_interaction_id=INTERACTION_ID,
        )

        assert result.action == 'created'
        assert result.was_new is True
        assert result.version_created is False
        merger.repository.create_deal.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_deal_with_correct_properties(
        self, merger, create_new_match_result,
    ):
        """Created Deal should have properties from extracted deal."""
        result = await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
            account_id='acct_acme_001',
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]  # First positional arg is the Deal model

        assert deal.tenant_id == TENANT_ID
        assert deal.name == 'Acme Corp Data Platform'
        assert deal.amount == 150000.0
        assert deal.account_id == 'acct_acme_001'
        assert deal.confidence == 0.92

    @pytest.mark.asyncio
    async def test_creates_deal_with_meddic_profile(
        self, merger, create_new_match_result,
    ):
        """Created Deal should have MEDDIC profile from extraction."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]

        assert deal.meddic.metrics == '$200K annual cost in manual reconciliation'
        assert deal.meddic.economic_buyer == 'Maria Chen, VP of Engineering'
        assert deal.meddic.champion == 'James'
        assert deal.meddic.completeness_score == 1.0  # All 6 fields populated

    @pytest.mark.asyncio
    async def test_creates_deal_with_embeddings(
        self, merger, create_new_match_result, sample_embedding,
    ):
        """Created Deal should have both embedding and embedding_current set."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]

        assert deal.embedding == sample_embedding
        assert deal.embedding_current == sample_embedding

    @pytest.mark.asyncio
    async def test_creates_deal_with_initial_evolution_summary(
        self, merger, create_new_match_result,
    ):
        """New deal should have an initial evolution_summary."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]

        assert 'Initial extraction' in deal.evolution_summary
        assert deal.opportunity_summary in deal.evolution_summary

    @pytest.mark.asyncio
    async def test_creates_deal_with_source_interaction(
        self, merger, create_new_match_result,
    ):
        """Created Deal should track source_interaction_id."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
            source_interaction_id=INTERACTION_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]

        assert deal.source_interaction_id == INTERACTION_ID

    @pytest.mark.asyncio
    async def test_creates_deal_with_stage_from_assessment(
        self, merger, sample_embedding,
    ):
        """Stage should be mapped from extracted stage_assessment."""
        extracted = ExtractedDeal(
            opportunity_name='Test',
            opportunity_summary='Test deal.',
            stage_assessment='proposal',
            confidence=0.8,
            reasoning='Test.',
        )
        match_result = DealMatchResult(
            extracted_deal=extracted,
            embedding=sample_embedding,
            match_type='create_new',
            matched_deal=None,
            decision=None,
            candidates_evaluated=0,
        )

        await merger.merge_deal(
            match_result=match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]
        assert deal.stage == 'proposal'

    @pytest.mark.asyncio
    async def test_no_llm_call_for_create_new(
        self, merger, mock_openai, create_new_match_result,
    ):
        """Create new path should not call LLM for synthesis."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        mock_openai.chat_completion_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_version_snapshot_for_create_new(
        self, merger, create_new_match_result,
    ):
        """Create new path should not create a version snapshot."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        merger.repository.create_version_snapshot.assert_not_called()


# =============================================================================
# Merge Path
# =============================================================================


class TestMergeExisting:
    """Test the merge path — updating an existing deal."""

    @pytest.mark.asyncio
    async def test_calls_llm_for_synthesis(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Merge path should call LLM with merge prompt."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        mock_openai.chat_completion_structured.assert_called_once()
        call_args = mock_openai.chat_completion_structured.call_args
        assert call_args.kwargs['response_model'] is MergedDeal

    @pytest.mark.asyncio
    async def test_version_snapshot_before_update(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Version snapshot MUST be created BEFORE update_deal."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        # Use a call recorder to verify ordering
        call_order = []
        original_snapshot = merger.repository.create_version_snapshot
        original_update = merger.repository.update_deal

        async def record_snapshot(*args, **kwargs):
            call_order.append('snapshot')
            return await original_snapshot(*args, **kwargs)

        async def record_update(*args, **kwargs):
            call_order.append('update')
            return await original_update(*args, **kwargs)

        merger.repository.create_version_snapshot = record_snapshot
        merger.repository.update_deal = record_update

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        assert call_order == ['snapshot', 'update']

    @pytest.mark.asyncio
    async def test_snapshot_receives_change_narrative(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Version snapshot should use change_narrative as change_summary."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
            source_interaction_id=INTERACTION_ID,
        )

        call_args = merger.repository.create_version_snapshot.call_args
        assert call_args.kwargs['change_summary'] == sample_merged_deal.change_narrative
        assert call_args.kwargs['changed_fields'] == sample_merged_deal.changed_fields
        assert call_args.kwargs['change_source_interaction_id'] == INTERACTION_ID

    @pytest.mark.asyncio
    async def test_merge_result_structure(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Merge result should have correct fields."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        result = await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        assert result.action == 'merged'
        assert result.was_new is False
        assert result.version_created is True
        assert result.opportunity_id == '019c1fa0-1111-7000-8000-000000000002'
        assert result.details['change_narrative'] == sample_merged_deal.change_narrative
        assert result.details['evolution_summary'] == sample_merged_deal.evolution_summary

    @pytest.mark.asyncio
    async def test_merge_prompt_contains_existing_and_extracted(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Merge prompt should include both existing deal and new extraction."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        messages = call_args.kwargs['messages']
        user_content = messages[1]['content']

        # Existing deal context
        assert 'Acme Data Platform Deal' in user_content
        assert 'Initial discovery call' in user_content  # evolution_summary
        # Extracted deal context
        assert 'Acme Corp Data Platform' in user_content

    @pytest.mark.asyncio
    async def test_auto_match_follows_merge_path(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """auto_match should follow the same merge path as llm_match."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        result = await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        assert result.action == 'merged'
        mock_openai.chat_completion_structured.assert_called_once()


# =============================================================================
# MEDDIC Flattening
# =============================================================================


class TestMEDDICFlattening:
    """Test MEDDIC field handling in the updates dict."""

    @pytest.mark.asyncio
    async def test_none_meddic_fields_not_in_updates(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """MEDDIC fields set to None should NOT appear in updates."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']

        # These were None in sample_merged_deal → should NOT be in updates
        assert 'meddic_metrics' not in updates
        assert 'meddic_economic_buyer' not in updates
        assert 'meddic_champion' not in updates

    @pytest.mark.asyncio
    async def test_populated_meddic_fields_in_updates(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """MEDDIC fields with values should appear in updates with meddic_ prefix."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']

        # decision_criteria was updated in sample_merged_deal
        assert 'meddic_decision_criteria' in updates
        assert 'SOC2' in updates['meddic_decision_criteria']
        assert 'existing ERP' in updates['meddic_decision_criteria']

    @pytest.mark.asyncio
    async def test_meddic_completeness_recomputed(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """meddic_completeness should be recomputed from merged state."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']

        # existing_deal_node has all 6 MEDDIC fields populated
        # merged deal keeps 5 existing (None) + updates 1 (decision_criteria)
        # All 6 remain populated → completeness = 1.0
        assert updates['meddic_completeness'] == 1.0

    @pytest.mark.asyncio
    async def test_completeness_decreases_when_fields_empty(
        self, merger, mock_openai, sample_embedding,
    ):
        """Completeness should reflect actual populated count."""
        sparse_existing = {
            'opportunity_id': '019c1fa0-2222-7000-8000-000000000003',
            'name': 'Sparse Deal',
            'stage': 'prospecting',
            'meddic_metrics': '$100K cost',
            # Only 1 of 6 MEDDIC fields populated
        }
        match_result = DealMatchResult(
            extracted_deal=ExtractedDeal(
                opportunity_name='Sparse', opportunity_summary='Test.',
                stage_assessment='prospecting', confidence=0.7, reasoning='Test.',
            ),
            embedding=sample_embedding,
            match_type='auto_match',
            matched_deal=DealMatchCandidate(
                opportunity_id='019c1fa0-2222-7000-8000-000000000003',
                node_properties=sparse_existing,
                similarity_score=0.95,
            ),
            decision=None,
            candidates_evaluated=1,
        )
        merged = MergedDeal(
            opportunity_summary='Updated.',
            evolution_summary='History.',
            change_narrative='Minor update.',
            changed_fields=['opportunity_summary'],
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']

        # Only 'metrics' is populated across existing + merged
        assert updates['meddic_completeness'] == pytest.approx(1 / 6)


# =============================================================================
# Stage Mapping
# =============================================================================


class TestStageMapping:
    """Test stage mapping from LLM implied_stage."""

    @pytest.mark.asyncio
    async def test_stage_progression(
        self, merger, mock_openai, auto_match_result,
    ):
        """implied_stage should map to DealStage enum value."""
        merged = MergedDeal(
            opportunity_summary='Deal advancing.',
            evolution_summary='History.',
            change_narrative='Moved to proposal.',
            changed_fields=['stage'],
            implied_stage='proposal',
            stage_reasoning='Pricing discussed.',
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert updates['stage'] == 'proposal'

    @pytest.mark.asyncio
    async def test_stage_regression(
        self, merger, mock_openai, auto_match_result,
    ):
        """Stage regression should be allowed (shadow forecast)."""
        merged = MergedDeal(
            opportunity_summary='Deal slipping.',
            evolution_summary='History.',
            change_narrative='Regressed to prospecting.',
            changed_fields=['stage'],
            implied_stage='prospecting',
            stage_reasoning='New stakeholder raised objections.',
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert updates['stage'] == 'prospecting'

    @pytest.mark.asyncio
    async def test_no_stage_change(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """No stage update when implied_stage is None."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert 'stage' not in updates


# =============================================================================
# Embedding Updates
# =============================================================================


class TestEmbeddingUpdate:
    """Test conditional embedding_current updates."""

    @pytest.mark.asyncio
    async def test_embedding_updated_when_should_update(
        self, merger, mock_openai, auto_match_result,
    ):
        """Should re-embed when should_update_embedding is True."""
        merged = MergedDeal(
            opportunity_summary='Completely different scope now.',
            evolution_summary='History.',
            change_narrative='Scope redefined.',
            changed_fields=['opportunity_summary'],
            should_update_embedding=True,
        )
        mock_openai.chat_completion_structured.return_value = merged
        mock_openai.create_embedding.return_value = [0.99] * 1536

        result = await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        assert result.embedding_updated is True
        mock_openai.create_embedding.assert_called_once()

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert 'embedding_current' in updates
        assert updates['embedding_current'] == [0.99] * 1536

    @pytest.mark.asyncio
    async def test_embedding_not_updated_when_not_needed(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Should NOT re-embed when should_update_embedding is False."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        result = await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        assert result.embedding_updated is False
        mock_openai.create_embedding.assert_not_called()

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert 'embedding_current' not in updates

    @pytest.mark.asyncio
    async def test_embedding_text_uses_name_summary_pattern(
        self, merger, mock_openai, auto_match_result,
    ):
        """Embedding text should follow '{name}: {summary}' pattern."""
        merged = MergedDeal(
            opportunity_summary='New summary for embedding.',
            evolution_summary='History.',
            change_narrative='Summary changed.',
            changed_fields=['opportunity_summary'],
            should_update_embedding=True,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = mock_openai.create_embedding.call_args
        text = call_args.args[0]
        assert text == 'Acme Corp Data Platform: New summary for embedding.'


# =============================================================================
# Evolution Summary Accumulation
# =============================================================================


class TestEvolutionSummary:
    """Test that evolution_summary accumulates across updates."""

    @pytest.mark.asyncio
    async def test_evolution_summary_in_updates(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """evolution_summary from LLM should be in the update dict."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert 'evolution_summary' in updates
        assert 'Jan 15' in updates['evolution_summary']
        assert 'Jan 22' in updates['evolution_summary']

    @pytest.mark.asyncio
    async def test_existing_evolution_passed_to_prompt(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Existing evolution_summary should be in the merge prompt for LLM to append to."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        messages = call_args.kwargs['messages']
        user_content = messages[1]['content']

        # The existing deal's evolution_summary should be in the prompt
        assert 'Initial discovery call revealed data silo pain points (Jan 15)' in user_content

    @pytest.mark.asyncio
    async def test_new_deal_has_initial_evolution(
        self, merger, create_new_match_result,
    ):
        """Newly created deals should have a seed evolution_summary."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]
        assert deal.evolution_summary != ''
        assert 'Initial extraction' in deal.evolution_summary

    @pytest.mark.asyncio
    async def test_system_prompt_requires_cumulative_narrative(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """System prompt should instruct LLM to produce cumulative evolution_summary."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        system_content = call_args.kwargs['messages'][0]['content']
        assert 'cumulative' in system_content.lower()
        assert 'evolution_summary' in system_content


# =============================================================================
# Amount Updates
# =============================================================================


class TestAmountUpdate:
    """Test deal amount updates."""

    @pytest.mark.asyncio
    async def test_amount_updated_when_provided(
        self, merger, mock_openai, auto_match_result,
    ):
        """Amount should be in updates when LLM provides it."""
        merged = MergedDeal(
            opportunity_summary='Deal summary.',
            evolution_summary='History.',
            change_narrative='Budget expanded.',
            changed_fields=['amount'],
            amount=500000.0,
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert updates['amount'] == 500000.0

    @pytest.mark.asyncio
    async def test_amount_not_updated_when_none(
        self, merger, mock_openai, auto_match_result, sample_merged_deal,
    ):
        """Amount should NOT be in updates when LLM returns None."""
        mock_openai.chat_completion_structured.return_value = sample_merged_deal

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.update_deal.call_args
        updates = call_args.kwargs['updates']
        assert 'amount' not in updates


# =============================================================================
# Opportunity ID Format
# =============================================================================


class TestOpportunityIdFormat:
    """Verify that created deals use UUIDv7 opportunity_id."""

    @pytest.mark.asyncio
    async def test_created_deal_has_uuid7_opportunity_id(
        self, merger, create_new_match_result,
    ):
        """opportunity_id on a newly created Deal must be a parseable UUID with version 7."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]

        assert isinstance(deal.opportunity_id, UUID)
        assert deal.opportunity_id.version == 7

    @pytest.mark.asyncio
    async def test_created_deal_has_deal_ref(
        self, merger, create_new_match_result,
    ):
        """Newly created Deal should have a deal_ref derived from last 16 hex of opportunity_id."""
        await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_deal.call_args
        deal = call_args.args[0]

        assert deal.deal_ref is not None
        assert deal.deal_ref.startswith('deal_')
        assert len(deal.deal_ref) == 21  # 'deal_' + 16 hex chars
        # Verify it uses the random-heavy tail of the UUID (not timestamp prefix)
        assert deal.deal_ref == f'deal_{deal.opportunity_id.hex[-16:]}'

    @pytest.mark.asyncio
    async def test_merge_result_opportunity_id_is_string(
        self, merger, create_new_match_result,
    ):
        """DealMergeResult.opportunity_id should be a string (boundary serialization)."""
        result = await merger.merge_deal(
            match_result=create_new_match_result,
            tenant_id=TENANT_ID,
        )

        assert isinstance(result.opportunity_id, str)
        # Should be parseable back to UUID
        parsed = UUID(result.opportunity_id)
        assert parsed.version == 7


# =============================================================================
# Changed Fields Normalization
# =============================================================================


class TestChangedFieldsNormalization:
    """Verify changed_fields normalization and whitelist filtering."""

    @pytest.mark.asyncio
    async def test_bare_meddic_name_mapped_to_prefixed(
        self, merger, mock_openai, auto_match_result,
    ):
        """Bare 'champion' should become 'meddic_champion' in changed_fields."""
        merged = MergedDeal(
            opportunity_summary='Updated.',
            evolution_summary='History.',
            change_narrative='Champion changed.',
            changed_fields=['champion', 'opportunity_summary'],
            champion='New Champion',
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_version_snapshot.call_args
        changed = call_args.kwargs['changed_fields']
        assert 'meddic_champion' in changed
        assert 'champion' not in changed
        assert 'opportunity_summary' in changed

    @pytest.mark.asyncio
    async def test_meta_fields_filtered_out(
        self, merger, mock_openai, auto_match_result,
    ):
        """Meta-fields like 'change_narrative' and 'stage_reasoning' should be filtered."""
        merged = MergedDeal(
            opportunity_summary='Updated.',
            evolution_summary='History.',
            change_narrative='Something changed.',
            changed_fields=['change_narrative', 'stage_reasoning', 'opportunity_summary'],
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_version_snapshot.call_args
        changed = call_args.kwargs['changed_fields']
        assert 'change_narrative' not in changed
        assert 'stage_reasoning' not in changed
        assert 'opportunity_summary' in changed

    @pytest.mark.asyncio
    async def test_unknown_fields_filtered_out(
        self, merger, mock_openai, auto_match_result,
    ):
        """LLM-invented field names should be filtered out."""
        merged = MergedDeal(
            opportunity_summary='Updated.',
            evolution_summary='History.',
            change_narrative='Something changed.',
            changed_fields=['opportunity_summary', 'invented_field', 'foo_bar'],
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_version_snapshot.call_args
        changed = call_args.kwargs['changed_fields']
        assert changed == ['opportunity_summary']

    @pytest.mark.asyncio
    async def test_qualification_dim_fields_whitelisted(
        self, merger, mock_openai, auto_match_result,
    ):
        """Qualification dim_* fields should pass the whitelist filter."""
        merged = MergedDeal(
            opportunity_summary='Updated.',
            evolution_summary='History.',
            change_narrative='Champion strength updated.',
            changed_fields=[
                'opportunity_summary',
                'dim_champion_strength',
                'dim_identified_pain',
                'dim_economic_buyer_access',
            ],
            should_update_embedding=False,
        )
        mock_openai.chat_completion_structured.return_value = merged

        await merger.merge_deal(
            match_result=auto_match_result,
            tenant_id=TENANT_ID,
        )

        call_args = merger.repository.create_version_snapshot.call_args
        changed = call_args.kwargs['changed_fields']
        assert 'dim_champion_strength' in changed
        assert 'dim_identified_pain' in changed
        assert 'dim_economic_buyer_access' in changed


# =============================================================================
# Transcript Extracted Dimensions Count
# =============================================================================


class TestTranscriptExtractedDimensions:
    """Verify TRANSCRIPT_EXTRACTED_DIMENSIONS contains all 15 dims."""

    def test_count_is_15(self):
        from deal_graph.pipeline.merger import TRANSCRIPT_EXTRACTED_DIMENSIONS
        assert len(TRANSCRIPT_EXTRACTED_DIMENSIONS) == 15

    def test_qualification_dims_present(self):
        from deal_graph.pipeline.merger import TRANSCRIPT_EXTRACTED_DIMENSIONS
        qual_dims = {
            'champion_strength', 'economic_buyer_access', 'identified_pain',
            'metrics_business_case', 'decision_criteria_alignment', 'decision_process_clarity',
        }
        assert qual_dims.issubset(TRANSCRIPT_EXTRACTED_DIMENSIONS)

    def test_original_9_dims_still_present(self):
        from deal_graph.pipeline.merger import TRANSCRIPT_EXTRACTED_DIMENSIONS
        original_dims = {
            'competitive_position', 'incumbent_displacement_risk',
            'pricing_alignment', 'procurement_legal_progress',
            'responsiveness', 'close_date_credibility',
            'technical_fit', 'integration_security_risk', 'change_readiness',
        }
        assert original_dims.issubset(TRANSCRIPT_EXTRACTED_DIMENSIONS)

    def test_whitelist_includes_all_dim_properties(self):
        from deal_graph.pipeline.merger import (
            TRANSCRIPT_EXTRACTED_DIMENSIONS,
            DEAL_PROPERTY_WHITELIST,
        )
        for dim_id in TRANSCRIPT_EXTRACTED_DIMENSIONS:
            assert f'dim_{dim_id}' in DEAL_PROPERTY_WHITELIST
            assert f'dim_{dim_id}_confidence' in DEAL_PROPERTY_WHITELIST
