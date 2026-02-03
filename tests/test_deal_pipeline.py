"""
Tests for Deal pipeline orchestrator with mocked components.

Tests cover:
- Case A (Targeted): opportunity_id present → fetches existing deal, targeted extraction
- Case B (Discovery): no opportunity_id → discovery extraction, multi-deal loop
- Validation: account_id required
- Error handling: per-deal errors accumulated, extraction failures raised
- Interaction enrichment: deal_count metadata written to Interaction node
- Stage timings: all stages tracked
- Edge cases: no deals extracted, existing deal not found (falls back to discovery)

Run with: pytest tests/test_deal_pipeline.py -v

No API keys or Neo4j required — Extractor, Matcher, and Merger are fully mocked.
"""

from datetime import datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from action_item_graph.models.envelope import (
    ContentFormat,
    ContentPayload,
    EnvelopeV1,
    InteractionType,
    SourceType,
)
from deal_graph.errors import DealExtractionError, DealPipelineError
from deal_graph.models.extraction import (
    DealExtractionResult,
    ExtractedDeal,
)
from deal_graph.pipeline.matcher import DealMatchCandidate, DealMatchResult
from deal_graph.pipeline.merger import DealMergeResult
from deal_graph.pipeline.pipeline import DealPipeline, DealPipelineResult


# =============================================================================
# Constants
# =============================================================================


TENANT_ID = UUID('550e8400-e29b-41d4-a716-446655440000')
INTERACTION_ID = UUID('660e8400-e29b-41d4-a716-446655440001')
ACCOUNT_ID = 'acct_acme_001'


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_neo4j():
    """Create a mocked DealNeo4jClient."""
    return AsyncMock()


@pytest.fixture
def mock_openai():
    """Create a mocked OpenAI client."""
    client = AsyncMock()
    client.chat_completion_structured = AsyncMock()
    client.create_embeddings_batch = AsyncMock(return_value=[[0.1] * 1536])
    client.create_embedding = AsyncMock(return_value=[0.1] * 1536)
    return client


@pytest.fixture
def pipeline(mock_neo4j, mock_openai):
    """Create a DealPipeline with all internal components mocked."""
    p = DealPipeline(neo4j_client=mock_neo4j, openai_client=mock_openai)

    # Mock all internal components at the method level
    p.repository = AsyncMock()
    p.repository.verify_account = AsyncMock(return_value={'account_id': ACCOUNT_ID})
    p.repository.ensure_interaction = AsyncMock(return_value={'interaction_id': str(INTERACTION_ID)})
    p.repository.get_deal = AsyncMock(return_value=None)
    p.repository.enrich_interaction = AsyncMock(return_value=None)

    p.extractor = AsyncMock()
    p.matcher = AsyncMock()
    p.merger = AsyncMock()

    return p


def _make_envelope(
    opportunity_id: str | None = None,
    account_id: str = ACCOUNT_ID,
) -> EnvelopeV1:
    """Build a test envelope with optional opportunity_id."""
    extras = {}
    if opportunity_id:
        extras['opportunity_id'] = opportunity_id

    return EnvelopeV1(
        tenant_id=TENANT_ID,
        user_id='auth0|user_test_001',
        interaction_type=InteractionType.TRANSCRIPT,
        content=ContentPayload(
            text='Sarah: We should discuss the data platform pricing. Budget is 500K.',
            format=ContentFormat.DIARIZED,
        ),
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        source=SourceType.WEB_MIC,
        account_id=account_id,
        interaction_id=INTERACTION_ID,
        extras=extras,
    )


def _make_extracted_deal(name: str = 'Acme Data Platform', confidence: float = 0.85) -> ExtractedDeal:
    """Build a sample extracted deal."""
    return ExtractedDeal(
        opportunity_name=name,
        opportunity_summary=f'{name} for enterprise data consolidation',
        stage_assessment='qualification',
        metrics='Expected 40% reduction in data silos',
        economic_buyer='CFO Jane Smith',
        decision_criteria='SOC2 compliance, Salesforce integration',
        decision_process=None,
        identified_pain='Data fragmentation across 12 systems',
        champion='VP of Engineering Tom',
        estimated_amount=500000.0,
        currency='USD',
        expected_close_timeframe='Q2 2026',
        confidence=confidence,
        reasoning='Clear budget signal and decision criteria discussed',
    )


def _make_match_result(
    extracted_deal: ExtractedDeal,
    match_type: str = 'create_new',
    opportunity_id: str | None = None,
) -> DealMatchResult:
    """Build a DealMatchResult."""
    matched_deal = None
    if match_type != 'create_new' and opportunity_id:
        matched_deal = DealMatchCandidate(
            opportunity_id=opportunity_id,
            node_properties={
                'opportunity_id': opportunity_id,
                'name': 'Existing Deal',
                'stage': 'prospecting',
            },
            similarity_score=0.92 if match_type == 'auto_match' else 0.82,
        )

    return DealMatchResult(
        extracted_deal=extracted_deal,
        embedding=[0.1] * 1536,
        match_type=match_type,
        matched_deal=matched_deal,
        decision=None,
        candidates_evaluated=1 if matched_deal else 0,
    )


def _make_merge_result(
    opportunity_id: str = '019c1fa0-5555-7000-8000-000000000006',
    action: str = 'created',
) -> DealMergeResult:
    """Build a DealMergeResult."""
    return DealMergeResult(
        opportunity_id=opportunity_id,
        action=action,
        was_new=(action == 'created'),
        version_created=(action == 'merged'),
        source_interaction_id=str(INTERACTION_ID),
        embedding_updated=False,
        details={'name': 'Test Deal'},
    )


# =============================================================================
# Test: Validation
# =============================================================================


class TestValidation:
    """Ensure required fields are validated before processing."""

    @pytest.mark.asyncio
    async def test_account_id_required(self, pipeline):
        """Pipeline raises DealPipelineError when account_id is missing."""
        envelope = _make_envelope()
        # Remove account_id by constructing without it
        envelope_no_account = EnvelopeV1(
            tenant_id=TENANT_ID,
            user_id='auth0|user_test_001',
            interaction_type=InteractionType.TRANSCRIPT,
            content=ContentPayload(text='Hello', format=ContentFormat.PLAIN),
            timestamp=datetime(2026, 1, 15, 10, 30, 0),
            source=SourceType.WEB_MIC,
            account_id=None,
        )

        with pytest.raises(DealPipelineError, match='account_id is required'):
            await pipeline.process_envelope(envelope_no_account)

    @pytest.mark.asyncio
    async def test_verify_account_called(self, pipeline):
        """Pipeline calls verify_account before extraction."""
        envelope = _make_envelope()

        # Set up empty extraction
        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        await pipeline.process_envelope(envelope)

        pipeline.repository.verify_account.assert_awaited_once_with(
            TENANT_ID, ACCOUNT_ID
        )


# =============================================================================
# Test: Case B — Discovery Flow
# =============================================================================


class TestDiscoveryFlow:
    """Case B: No opportunity_id → discovery extraction, per-deal loop."""

    @pytest.mark.asyncio
    async def test_discovery_single_deal_created(self, pipeline):
        """Discovery extracts one deal, creates it (no match found)."""
        envelope = _make_envelope()  # No opportunity_id
        deal = _make_extracted_deal()
        embedding = [0.1] * 1536

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [embedding],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'create_new')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-6666-7000-8000-000000000007', 'created')
        )

        result = await pipeline.process_envelope(envelope)

        assert result.total_extracted == 1
        assert result.deals_created == ['019c1fa0-6666-7000-8000-000000000007']
        assert result.deals_merged == []
        assert result.success is True
        assert result.total_processed == 1

    @pytest.mark.asyncio
    async def test_discovery_multi_deal_flow(self, pipeline):
        """Discovery extracts multiple deals, each goes through match → merge."""
        envelope = _make_envelope()
        deal_a = _make_extracted_deal('Deal Alpha')
        deal_b = _make_extracted_deal('Deal Beta', confidence=0.70)
        embeddings = [[0.1] * 1536, [0.2] * 1536]

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal_a, deal_b], has_deals=True),
                embeddings,
            )
        )

        # Deal Alpha → create new, Deal Beta → auto_match → merge
        pipeline.matcher.find_matches = AsyncMock(
            side_effect=[
                _make_match_result(deal_a, 'create_new'),
                _make_match_result(deal_b, 'auto_match', '019c1fa0-1111-7000-8000-000000000002'),
            ]
        )
        pipeline.merger.merge_deal = AsyncMock(
            side_effect=[
                _make_merge_result('019c1fa0-7777-7000-8000-000000000008', 'created'),
                _make_merge_result('019c1fa0-1111-7000-8000-000000000002', 'merged'),
            ]
        )

        result = await pipeline.process_envelope(envelope)

        assert result.total_extracted == 2
        assert result.deals_created == ['019c1fa0-7777-7000-8000-000000000008']
        assert result.deals_merged == ['019c1fa0-1111-7000-8000-000000000002']
        assert result.total_processed == 2
        assert len(result.merge_results) == 2

    @pytest.mark.asyncio
    async def test_discovery_no_deals_extracted(self, pipeline):
        """Discovery finds no deals — result is empty but successful."""
        envelope = _make_envelope()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        result = await pipeline.process_envelope(envelope)

        assert result.total_extracted == 0
        assert result.deals_created == []
        assert result.deals_merged == []
        assert result.success is True
        # Matcher and merger should NOT be called
        pipeline.matcher.find_matches.assert_not_awaited()
        pipeline.merger.merge_deal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_discovery_no_opportunity_id_in_result(self, pipeline):
        """Discovery flow has opportunity_id=None on the result."""
        envelope = _make_envelope()  # No opportunity_id

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        result = await pipeline.process_envelope(envelope)

        assert result.opportunity_id is None

    @pytest.mark.asyncio
    async def test_discovery_passes_correct_embedding_per_deal(self, pipeline):
        """Each deal gets its corresponding embedding from the extraction."""
        envelope = _make_envelope()
        deal_a = _make_extracted_deal('Deal Alpha')
        deal_b = _make_extracted_deal('Deal Beta')
        emb_a = [0.1] * 1536
        emb_b = [0.9] * 1536

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal_a, deal_b], has_deals=True),
                [emb_a, emb_b],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            side_effect=[
                _make_match_result(deal_a, 'create_new'),
                _make_match_result(deal_b, 'create_new'),
            ]
        )
        pipeline.merger.merge_deal = AsyncMock(
            side_effect=[
                _make_merge_result('019c1fa0-dddd-7000-8000-00000000000e', 'created'),
                _make_merge_result('019c1fa0-eeee-7000-8000-00000000000f', 'created'),
            ]
        )

        await pipeline.process_envelope(envelope)

        # Verify each matcher call got the correct embedding
        calls = pipeline.matcher.find_matches.call_args_list
        assert calls[0].kwargs['embedding'] == emb_a
        assert calls[1].kwargs['embedding'] == emb_b


# =============================================================================
# Test: Case A — Targeted Flow
# =============================================================================


class TestTargetedFlow:
    """Case A: opportunity_id present → fetch existing deal, targeted extraction."""

    @pytest.mark.asyncio
    async def test_targeted_fetches_existing_deal(self, pipeline):
        """Pipeline fetches existing deal from repo when opportunity_id is present."""
        envelope = _make_envelope(opportunity_id='019c1fa0-4444-7000-8000-000000000005')
        existing_deal_props = {
            'opportunity_id': '019c1fa0-4444-7000-8000-000000000005',
            'name': 'Acme Platform',
            'stage': 'qualification',
            'meddic_champion': 'VP Tom',
        }
        pipeline.repository.get_deal = AsyncMock(return_value=existing_deal_props)

        deal = _make_extracted_deal()
        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'auto_match', '019c1fa0-4444-7000-8000-000000000005')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-4444-7000-8000-000000000005', 'merged')
        )

        result = await pipeline.process_envelope(envelope)

        # Verify get_deal was called with correct args
        pipeline.repository.get_deal.assert_awaited_once_with(
            tenant_id=TENANT_ID,
            opportunity_id='019c1fa0-4444-7000-8000-000000000005',
        )
        # Verify extractor got the existing deal context
        extract_call = pipeline.extractor.extract_from_envelope.call_args
        assert extract_call.kwargs['existing_deal'] == existing_deal_props
        assert result.opportunity_id == '019c1fa0-4444-7000-8000-000000000005'

    @pytest.mark.asyncio
    async def test_targeted_merges_existing_deal(self, pipeline):
        """Targeted flow results in a merge (not create) when deal exists."""
        envelope = _make_envelope(opportunity_id='019c1fa0-4444-7000-8000-000000000005')
        pipeline.repository.get_deal = AsyncMock(return_value={
            'opportunity_id': '019c1fa0-4444-7000-8000-000000000005',
            'name': 'Existing Deal',
        })

        deal = _make_extracted_deal()
        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'auto_match', '019c1fa0-4444-7000-8000-000000000005')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-4444-7000-8000-000000000005', 'merged')
        )

        result = await pipeline.process_envelope(envelope)

        assert result.deals_merged == ['019c1fa0-4444-7000-8000-000000000005']
        assert result.deals_created == []
        assert result.total_processed == 1

    @pytest.mark.asyncio
    async def test_targeted_fallback_when_deal_not_found(self, pipeline):
        """When opportunity_id is present but deal not in graph, falls back to discovery."""
        envelope = _make_envelope(opportunity_id='019c1fa0-8888-7000-8000-000000000009')
        pipeline.repository.get_deal = AsyncMock(return_value=None)

        deal = _make_extracted_deal()
        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'create_new')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-9999-7000-8000-00000000000a', 'created')
        )

        result = await pipeline.process_envelope(envelope)

        # Verify extractor was called with existing_deal=None (discovery mode)
        extract_call = pipeline.extractor.extract_from_envelope.call_args
        assert extract_call.kwargs['existing_deal'] is None
        # Verify warning was recorded
        assert any('not found' in w for w in result.warnings)
        assert result.deals_created == ['019c1fa0-9999-7000-8000-00000000000a']

    @pytest.mark.asyncio
    async def test_targeted_no_updates_extracted(self, pipeline):
        """Targeted flow finds no new info in transcript — zero deals extracted."""
        envelope = _make_envelope(opportunity_id='019c1fa0-4444-7000-8000-000000000005')
        pipeline.repository.get_deal = AsyncMock(return_value={
            'opportunity_id': '019c1fa0-4444-7000-8000-000000000005',
            'name': 'Existing',
        })

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        result = await pipeline.process_envelope(envelope)

        assert result.total_extracted == 0
        assert result.success is True
        pipeline.matcher.find_matches.assert_not_awaited()


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestErrorHandling:
    """Per-deal errors are accumulated; extraction failures are fatal."""

    @pytest.mark.asyncio
    async def test_extraction_failure_raises(self, pipeline):
        """If extraction itself fails, DealExtractionError is raised."""
        envelope = _make_envelope()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            side_effect=RuntimeError('LLM quota exceeded')
        )

        with pytest.raises(DealExtractionError, match='LLM quota exceeded'):
            await pipeline.process_envelope(envelope)

    @pytest.mark.asyncio
    async def test_per_deal_error_accumulated(self, pipeline):
        """If one deal fails during match/merge, the error is captured but others proceed."""
        envelope = _make_envelope()
        deal_ok = _make_extracted_deal('Good Deal')
        deal_bad = _make_extracted_deal('Bad Deal')

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal_ok, deal_bad], has_deals=True),
                [[0.1] * 1536, [0.2] * 1536],
            )
        )

        # First deal succeeds, second deal's matcher throws
        pipeline.matcher.find_matches = AsyncMock(
            side_effect=[
                _make_match_result(deal_ok, 'create_new'),
                RuntimeError('Vector index unavailable'),
            ]
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-aaaa-7000-8000-00000000000b', 'created')
        )

        result = await pipeline.process_envelope(envelope)

        # First deal succeeded
        assert result.deals_created == ['019c1fa0-aaaa-7000-8000-00000000000b']
        # Second deal's error was captured
        assert len(result.errors) == 1
        assert 'Bad Deal' in result.errors[0]
        assert 'Vector index unavailable' in result.errors[0]
        # Overall success is False because there are errors
        assert result.success is False

    @pytest.mark.asyncio
    async def test_merger_error_accumulated(self, pipeline):
        """If merger fails for a deal, the error is captured."""
        envelope = _make_envelope()
        deal = _make_extracted_deal()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'create_new')
        )
        pipeline.merger.merge_deal = AsyncMock(
            side_effect=RuntimeError('Neo4j write timeout')
        )

        result = await pipeline.process_envelope(envelope)

        assert len(result.errors) == 1
        assert 'Neo4j write timeout' in result.errors[0]
        assert result.total_processed == 0

    @pytest.mark.asyncio
    async def test_enrich_interaction_failure_swallowed(self, pipeline):
        """Interaction enrichment failure is logged but does not fail the pipeline."""
        envelope = _make_envelope()
        deal = _make_extracted_deal()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'create_new')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-bbbb-7000-8000-00000000000c', 'created')
        )
        pipeline.repository.enrich_interaction = AsyncMock(
            side_effect=RuntimeError('Connection reset')
        )

        result = await pipeline.process_envelope(envelope)

        # Pipeline still succeeds despite enrichment failure
        assert result.success is True
        assert result.deals_created == ['019c1fa0-bbbb-7000-8000-00000000000c']


# =============================================================================
# Test: Interaction Enrichment
# =============================================================================


class TestInteractionEnrichment:
    """Interaction node is enriched with deal_count after processing."""

    @pytest.mark.asyncio
    async def test_enrichment_with_deal_count(self, pipeline):
        """Interaction is enriched with the number of successfully processed deals."""
        envelope = _make_envelope()
        deal = _make_extracted_deal()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'create_new')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-bbbb-7000-8000-00000000000c', 'created')
        )

        await pipeline.process_envelope(envelope)

        pipeline.repository.enrich_interaction.assert_awaited_once_with(
            tenant_id=TENANT_ID,
            interaction_id=str(INTERACTION_ID),
            deal_count=1,
        )

    @pytest.mark.asyncio
    async def test_enrichment_zero_deals(self, pipeline):
        """Interaction is enriched with deal_count=0 when no deals extracted."""
        envelope = _make_envelope()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        await pipeline.process_envelope(envelope)

        pipeline.repository.enrich_interaction.assert_awaited_once_with(
            tenant_id=TENANT_ID,
            interaction_id=str(INTERACTION_ID),
            deal_count=0,
        )

    @pytest.mark.asyncio
    async def test_no_enrichment_without_interaction_id(self, pipeline):
        """If envelope has no interaction_id, enrichment is skipped."""
        envelope = EnvelopeV1(
            tenant_id=TENANT_ID,
            user_id='auth0|user_test_001',
            interaction_type=InteractionType.TRANSCRIPT,
            content=ContentPayload(text='Hello', format=ContentFormat.PLAIN),
            timestamp=datetime(2026, 1, 15, 10, 30, 0),
            source=SourceType.WEB_MIC,
            account_id=ACCOUNT_ID,
            interaction_id=None,
        )

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        await pipeline.process_envelope(envelope)

        pipeline.repository.enrich_interaction.assert_not_awaited()


# =============================================================================
# Test: Component Wiring
# =============================================================================


class TestComponentWiring:
    """Verify the orchestrator passes correct arguments between stages."""

    @pytest.mark.asyncio
    async def test_matcher_receives_tenant_and_account(self, pipeline):
        """Matcher is called with tenant_id and account_id from envelope."""
        envelope = _make_envelope()
        deal = _make_extracted_deal()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'create_new')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-bbbb-7000-8000-00000000000c', 'created')
        )

        await pipeline.process_envelope(envelope)

        match_call = pipeline.matcher.find_matches.call_args
        assert match_call.kwargs['tenant_id'] == TENANT_ID
        assert match_call.kwargs['account_id'] == ACCOUNT_ID

    @pytest.mark.asyncio
    async def test_merger_receives_match_result_and_interaction_id(self, pipeline):
        """Merger receives the match result, tenant_id, and source_interaction_id."""
        envelope = _make_envelope()
        deal = _make_extracted_deal()
        match_result = _make_match_result(deal, 'create_new')

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(return_value=match_result)
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-bbbb-7000-8000-00000000000c', 'created')
        )

        await pipeline.process_envelope(envelope)

        merge_call = pipeline.merger.merge_deal.call_args
        assert merge_call.kwargs['match_result'] == match_result
        assert merge_call.kwargs['tenant_id'] == TENANT_ID
        assert merge_call.kwargs['account_id'] == ACCOUNT_ID
        assert merge_call.kwargs['source_interaction_id'] == INTERACTION_ID

    @pytest.mark.asyncio
    async def test_ensure_interaction_called_with_envelope_data(self, pipeline):
        """Interaction node is ensured with envelope metadata."""
        envelope = _make_envelope()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        await pipeline.process_envelope(envelope)

        pipeline.repository.ensure_interaction.assert_awaited_once()
        call_kwargs = pipeline.repository.ensure_interaction.call_args.kwargs
        assert call_kwargs['tenant_id'] == TENANT_ID
        assert call_kwargs['interaction_id'] == str(INTERACTION_ID)
        assert 'data platform pricing' in call_kwargs['content_text']
        assert call_kwargs['interaction_type'] == 'transcript'
        assert call_kwargs['source'] == 'web-mic'


# =============================================================================
# Test: Result Structure
# =============================================================================


class TestResultStructure:
    """Verify the DealPipelineResult is populated correctly."""

    @pytest.mark.asyncio
    async def test_timing_fields_populated(self, pipeline):
        """Result includes started_at, completed_at, and processing_time_ms."""
        envelope = _make_envelope()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        result = await pipeline.process_envelope(envelope)

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.processing_time_ms is not None
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_stage_timings_tracked(self, pipeline):
        """Stage timings dict is populated for each pipeline stage."""
        envelope = _make_envelope()
        deal = _make_extracted_deal()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal], has_deals=True),
                [[0.1] * 1536],
            )
        )
        pipeline.matcher.find_matches = AsyncMock(
            return_value=_make_match_result(deal, 'create_new')
        )
        pipeline.merger.merge_deal = AsyncMock(
            return_value=_make_merge_result('019c1fa0-bbbb-7000-8000-00000000000c', 'created')
        )

        result = await pipeline.process_envelope(envelope)

        assert 'verify_account' in result.stage_timings
        assert 'ensure_interaction' in result.stage_timings
        assert 'extraction' in result.stage_timings
        assert 'match_merge' in result.stage_timings

    @pytest.mark.asyncio
    async def test_extraction_notes_captured(self, pipeline):
        """extraction_notes from the extractor are passed through to the result."""
        envelope = _make_envelope()

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(
                    deals=[],
                    has_deals=False,
                    extraction_notes='Transcript too short for deal signals',
                ),
                [],
            )
        )

        result = await pipeline.process_envelope(envelope)

        assert result.extraction_notes == 'Transcript too short for deal signals'

    @pytest.mark.asyncio
    async def test_result_context_from_envelope(self, pipeline):
        """Result captures tenant_id, account_id, interaction_id from envelope."""
        envelope = _make_envelope(opportunity_id='019c1fa0-cccc-7000-8000-00000000000d')
        pipeline.repository.get_deal = AsyncMock(return_value=None)

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[], has_deals=False),
                [],
            )
        )

        result = await pipeline.process_envelope(envelope)

        assert result.tenant_id == str(TENANT_ID)
        assert result.account_id == ACCOUNT_ID
        assert result.interaction_id == str(INTERACTION_ID)
        assert result.opportunity_id == '019c1fa0-cccc-7000-8000-00000000000d'

    @pytest.mark.asyncio
    async def test_merge_results_list(self, pipeline):
        """merge_results captures every DealMergeResult from the loop."""
        envelope = _make_envelope()
        deal_a = _make_extracted_deal('Alpha')
        deal_b = _make_extracted_deal('Beta')

        pipeline.extractor.extract_from_envelope = AsyncMock(
            return_value=(
                DealExtractionResult(deals=[deal_a, deal_b], has_deals=True),
                [[0.1] * 1536, [0.2] * 1536],
            )
        )

        merge_a = _make_merge_result('019c1fa0-dddd-7000-8000-00000000000e', 'created')
        merge_b = _make_merge_result('019c1fa0-eeee-7000-8000-00000000000f', 'merged')

        pipeline.matcher.find_matches = AsyncMock(
            side_effect=[
                _make_match_result(deal_a, 'create_new'),
                _make_match_result(deal_b, 'auto_match', '019c1fa0-eeee-7000-8000-00000000000f'),
            ]
        )
        pipeline.merger.merge_deal = AsyncMock(
            side_effect=[merge_a, merge_b]
        )

        result = await pipeline.process_envelope(envelope)

        assert result.merge_results == [merge_a, merge_b]


# =============================================================================
# Test: DealPipelineResult Properties
# =============================================================================


class TestResultProperties:
    """Unit tests for DealPipelineResult computed properties."""

    def test_success_true_no_errors(self):
        """success is True when errors list is empty."""
        result = DealPipelineResult(
            tenant_id='t', account_id='a',
            interaction_id='i', opportunity_id=None,
            total_extracted=1,
        )
        assert result.success is True

    def test_success_false_with_errors(self):
        """success is False when errors list is non-empty."""
        result = DealPipelineResult(
            tenant_id='t', account_id='a',
            interaction_id='i', opportunity_id=None,
            total_extracted=1,
            errors=['something failed'],
        )
        assert result.success is False

    def test_total_processed(self):
        """total_processed is sum of created + merged."""
        result = DealPipelineResult(
            tenant_id='t', account_id='a',
            interaction_id='i', opportunity_id=None,
            total_extracted=5,
            deals_created=['d1', 'd2'],
            deals_merged=['d3'],
        )
        assert result.total_processed == 3
