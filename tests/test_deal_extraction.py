"""
Tests for Deal extraction engine with mocked OpenAI calls.

Tests cover:
- Discovery mode (Case B): find all deals, no deals, multiple deals, context injection
- Targeted mode (Case A): find updates, no updates, single-deal enforcement
- Embedding generation: text pattern, empty input, batch handling
- Envelope routing: Case A vs B dispatch logic

Run with: pytest tests/test_deal_extraction.py -v

No API keys required — all OpenAI calls are mocked.
"""

from datetime import datetime

import pytest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

from action_item_graph.models.envelope import (
    ContentFormat,
    ContentPayload,
    EnvelopeV1,
    InteractionType,
    SourceType,
)
from deal_graph.models.extraction import (
    DealExtractionResult,
    ExtractedDeal,
)
from deal_graph.pipeline.extractor import DealExtractor


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_openai():
    """Create a mocked OpenAI client."""
    client = AsyncMock()
    client.chat_completion_structured = AsyncMock()
    client.create_embeddings_batch = AsyncMock(return_value=[])
    return client


@pytest.fixture
def extractor(mock_openai):
    """Create a DealExtractor with mocked OpenAI client."""
    return DealExtractor(openai_client=mock_openai)


@pytest.fixture
def sample_envelope():
    """Create a sample envelope without opportunity_id (Case B)."""
    return EnvelopeV1(
        tenant_id=UUID('550e8400-e29b-41d4-a716-446655440000'),
        user_id='auth0|test123',
        interaction_type=InteractionType.TRANSCRIPT,
        content=ContentPayload(
            text=(
                'Sarah: Thanks for taking the time to meet with us today. '
                'I wanted to walk you through how our platform can help with the '
                'data silo issues you mentioned in our last call.\n\n'
                'James: Yes, that\'s been a major pain point. We\'re spending about '
                '40 hours a week just reconciling data between our three CRM systems. '
                'It\'s costing us roughly $200K annually in lost productivity.\n\n'
                'Sarah: That\'s significant. Our platform typically reduces that '
                'reconciliation time by about 80%. Based on what you\'ve described, '
                'we\'d be looking at a deal in the range of $150K for the enterprise license.\n\n'
                'James: That\'s within our budget range. I should mention that our VP of '
                'Engineering, Maria Chen, will need to sign off on any technology purchase '
                'over $100K. She\'s the final decision maker on the budget side.'
            ),
            format=ContentFormat.DIARIZED,
        ),
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        source=SourceType.WEB_MIC,
        account_id='acct_acme_corp_001',
        interaction_id=uuid4(),
    )


@pytest.fixture
def sample_envelope_with_opportunity(sample_envelope):
    """Create a sample envelope WITH opportunity_id (Case A)."""
    sample_envelope.extras = {'opportunity_id': '019c1fa0-4444-7000-8000-000000000005'}
    return sample_envelope


@pytest.fixture
def sample_extracted_deal():
    """A typical extracted deal for testing."""
    return ExtractedDeal(
        opportunity_name='Acme Corp Data Platform',
        opportunity_summary=(
            'Acme is evaluating our data platform to solve CRM data silo issues '
            'costing $200K/year in manual reconciliation.'
        ),
        stage_assessment='qualification',
        metrics='$200K annual cost from manual reconciliation of 3 CRM systems, 40 hours/week spent',
        economic_buyer='Maria Chen, VP of Engineering',
        decision_criteria='SOC2 compliance, technical POC required',
        decision_process='POC → security review → Maria + procurement review → close by Q2',
        identified_pain='40 hours/week reconciling data between three CRM systems',
        champion='James (main contact, pushing internally)',
        estimated_amount=150000.0,
        currency='USD',
        expected_close_timeframe='End of Q2',
        confidence=0.92,
        reasoning=(
            'Clear MEDDIC signals: quantified pain ($200K/year), identified economic '
            'buyer (Maria Chen), specific decision process, and a champion (James).'
        ),
    )


@pytest.fixture
def sample_existing_deal():
    """Existing deal properties from Neo4j for Case A tests."""
    return {
        'tenant_id': '550e8400-e29b-41d4-a716-446655440000',
        'opportunity_id': '019c1fa0-4444-7000-8000-000000000005',
        'name': 'Acme Corp Data Platform',
        'stage': 'qualification',
        'amount': 150000.0,
        'opportunity_summary': 'Acme evaluating data platform for CRM consolidation.',
        'meddic_metrics': '$200K annual cost in manual reconciliation',
        'meddic_economic_buyer': 'Maria Chen, VP of Engineering',
        'meddic_decision_criteria': 'SOC2 compliance required',
        'meddic_decision_process': 'POC → security review → procurement',
        'meddic_identified_pain': 'Data silos between three CRM systems',
        'meddic_champion': 'James',
    }


@pytest.fixture
def sample_embedding():
    """A mock 1536-dimensional embedding vector."""
    return [0.01] * 1536


# =============================================================================
# Discovery Mode Tests (Case B)
# =============================================================================


class TestDiscoveryExtraction:
    """Test Case B: discovery mode — find all deals."""

    @pytest.mark.asyncio
    async def test_extract_discovery_finds_deals(
        self, extractor, mock_openai, sample_extracted_deal,
    ):
        """Discovery mode should return extracted deals."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal],
            has_deals=True,
        )

        result = await extractor._extract_discovery(
            content_text='Sarah: Thanks for taking the time...',
        )

        assert result.has_deals is True
        assert len(result.deals) == 1
        assert result.deals[0].opportunity_name == 'Acme Corp Data Platform'
        assert result.deals[0].metrics is not None
        assert result.deals[0].economic_buyer is not None

        # Verify the structured call was made with correct response model
        call_args = mock_openai.chat_completion_structured.call_args
        assert call_args.kwargs['response_model'] is DealExtractionResult
        messages = call_args.kwargs['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert 'MEDDIC' in messages[0]['content']

    @pytest.mark.asyncio
    async def test_extract_discovery_no_deals(self, extractor, mock_openai):
        """Discovery mode should handle no deals found."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[],
            has_deals=False,
            extraction_notes='Casual conversation with no sales context.',
        )

        result = await extractor._extract_discovery(
            content_text='Hey, how was your weekend?',
        )

        assert result.has_deals is False
        assert len(result.deals) == 0
        assert result.extraction_notes is not None

    @pytest.mark.asyncio
    async def test_extract_discovery_multiple_deals(
        self, extractor, mock_openai, sample_extracted_deal,
    ):
        """Discovery mode can find multiple deals in one transcript."""
        deal_2 = ExtractedDeal(
            opportunity_name='Acme Security Audit',
            opportunity_summary='Secondary opportunity for compliance consulting.',
            stage_assessment='prospecting',
            confidence=0.65,
            reasoning='Brief mention of needing compliance help.',
        )
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal, deal_2],
            has_deals=True,
        )

        result = await extractor._extract_discovery(
            content_text='Multiple deals discussed...',
        )

        assert result.has_deals is True
        assert len(result.deals) == 2

    @pytest.mark.asyncio
    async def test_extract_discovery_with_context(
        self, extractor, mock_openai, sample_extracted_deal,
    ):
        """Discovery mode includes optional context in prompt."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal],
            has_deals=True,
        )

        await extractor._extract_discovery(
            content_text='Transcript text...',
            account_name='Acme Corp',
            meeting_title='Q1 Review',
            participants=['Sarah', 'James'],
        )

        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        assert 'Acme Corp' in user_content
        assert 'Q1 Review' in user_content
        assert 'Sarah' in user_content

    @pytest.mark.asyncio
    async def test_discovery_prompt_contains_content_text(
        self, extractor, mock_openai, sample_extracted_deal,
    ):
        """Verify the transcript text is injected into the prompt."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[], has_deals=False,
        )

        transcript = 'James: We need SOC2 compliance for this deal.'
        await extractor._extract_discovery(content_text=transcript)

        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        assert transcript in user_content
        assert '<transcript>' in user_content

    @pytest.mark.asyncio
    async def test_system_prompt_handles_content_formats(
        self, extractor, mock_openai,
    ):
        """System prompt should mention all content_text formats."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[], has_deals=False,
        )

        await extractor._extract_discovery(content_text='test')

        call_args = mock_openai.chat_completion_structured.call_args
        system_content = call_args.kwargs['messages'][0]['content']
        assert 'Diarized' in system_content
        assert 'Plain text' in system_content
        assert 'Markdown' in system_content

    @pytest.mark.asyncio
    async def test_system_prompt_contains_qualification_dimensions(
        self, extractor, mock_openai,
    ):
        """System prompt should include all 6 qualification dimensions."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[], has_deals=False,
        )

        await extractor._extract_discovery(content_text='test')

        call_args = mock_openai.chat_completion_structured.call_args
        system_content = call_args.kwargs['messages'][0]['content']
        for dim_id in [
            'champion_strength', 'economic_buyer_access', 'identified_pain',
            'metrics_business_case', 'decision_criteria_alignment', 'decision_process_clarity',
        ]:
            assert dim_id in system_content, f'{dim_id} missing from system prompt'

    @pytest.mark.asyncio
    async def test_system_prompt_contains_all_15_dimensions(
        self, extractor, mock_openai,
    ):
        """System prompt should include all 15 transcript-extracted dimensions."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[], has_deals=False,
        )

        await extractor._extract_discovery(content_text='test')

        call_args = mock_openai.chat_completion_structured.call_args
        system_content = call_args.kwargs['messages'][0]['content']
        all_dims = [
            'champion_strength', 'economic_buyer_access', 'identified_pain',
            'metrics_business_case', 'decision_criteria_alignment', 'decision_process_clarity',
            'competitive_position', 'incumbent_displacement_risk',
            'pricing_alignment', 'procurement_legal_progress',
            'responsiveness', 'close_date_credibility',
            'technical_fit', 'integration_security_risk', 'change_readiness',
        ]
        for dim_id in all_dims:
            assert dim_id in system_content, f'{dim_id} missing from system prompt'


# =============================================================================
# Targeted Mode Tests (Case A)
# =============================================================================


class TestTargetedExtraction:
    """Test Case A: targeted mode — update specific deal."""

    @pytest.mark.asyncio
    async def test_extract_targeted_finds_update(
        self, extractor, mock_openai, sample_extracted_deal, sample_existing_deal,
    ):
        """Targeted extraction should find updates for a known deal."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal],
            has_deals=True,
        )

        result = await extractor._extract_targeted(
            content_text='James: We got the SOC2 cert...',
            existing_deal=sample_existing_deal,
        )

        assert result.has_deals is True
        assert len(result.deals) == 1

        # Verify existing deal context was included in prompt
        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        assert '<existing_deal>' in user_content
        assert 'Acme Corp Data Platform' in user_content
        assert 'Maria Chen' in user_content
        assert 'qualification' in user_content

    @pytest.mark.asyncio
    async def test_extract_targeted_no_update(
        self, extractor, mock_openai, sample_existing_deal,
    ):
        """Targeted extraction should handle no updates found."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[],
            has_deals=False,
            extraction_notes='Transcript does not contain updates for this deal.',
        )

        result = await extractor._extract_targeted(
            content_text='Unrelated conversation...',
            existing_deal=sample_existing_deal,
        )

        assert result.has_deals is False
        assert len(result.deals) == 0

    @pytest.mark.asyncio
    async def test_extract_targeted_enforces_single_deal(
        self, extractor, mock_openai, sample_extracted_deal, sample_existing_deal,
    ):
        """Targeted extraction should enforce the single-deal constraint."""
        deal_2 = ExtractedDeal(
            opportunity_name='Unexpected Second Deal',
            opportunity_summary='Should be filtered out.',
            stage_assessment='prospecting',
            confidence=0.5,
            reasoning='Extra deal.',
        )
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal, deal_2],
            has_deals=True,
        )

        result = await extractor._extract_targeted(
            content_text='Some transcript...',
            existing_deal=sample_existing_deal,
        )

        # Should keep only the highest-confidence deal
        assert len(result.deals) == 1
        assert result.deals[0].opportunity_name == 'Acme Corp Data Platform'
        assert result.deals[0].confidence == 0.92
        assert result.extraction_notes is not None
        assert 'highest-confidence' in result.extraction_notes

    @pytest.mark.asyncio
    async def test_targeted_prompt_includes_meddic_context(
        self, extractor, mock_openai, sample_existing_deal,
    ):
        """Targeted prompt should include all MEDDIC fields from existing deal."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[], has_deals=False,
        )

        await extractor._extract_targeted(
            content_text='Test transcript.',
            existing_deal=sample_existing_deal,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        # All existing MEDDIC fields should appear in the prompt
        assert '$200K' in user_content
        assert 'Maria Chen' in user_content
        assert 'SOC2' in user_content
        assert 'POC' in user_content
        assert 'Data silos' in user_content
        assert 'James' in user_content

    @pytest.mark.asyncio
    async def test_targeted_prompt_includes_qualification_dim_scores(
        self, extractor, mock_openai,
    ):
        """Targeted prompt should show existing qualification dimension scores."""
        deal_with_qual_dims = {
            'name': 'Test Deal',
            'stage': 'qualification',
            'dim_champion_strength': 2,
            'dim_identified_pain': 3,
            'dim_economic_buyer_access': 1,
        }
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[], has_deals=False,
        )

        await extractor._extract_targeted(
            content_text='Transcript...',
            existing_deal=deal_with_qual_dims,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        assert 'champion_strength: 2/3' in user_content
        assert 'identified_pain: 3/3' in user_content
        assert 'economic_buyer_access: 1/3' in user_content

    @pytest.mark.asyncio
    async def test_targeted_handles_sparse_existing_deal(
        self, extractor, mock_openai,
    ):
        """Targeted prompt should handle missing fields gracefully."""
        sparse_deal = {
            'opportunity_id': 'opp_123',
            'name': 'New Deal',
            'stage': 'prospecting',
        }
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[], has_deals=False,
        )

        await extractor._extract_targeted(
            content_text='Transcript...',
            existing_deal=sparse_deal,
        )

        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        # Missing MEDDIC fields should show fallback text
        assert 'Not yet identified' in user_content
        assert 'New Deal' in user_content


# =============================================================================
# Embedding Generation Tests
# =============================================================================


class TestEmbeddingGeneration:
    """Test embedding generation for extracted deals."""

    @pytest.mark.asyncio
    async def test_generate_embeddings(
        self, extractor, mock_openai, sample_extracted_deal, sample_embedding,
    ):
        """Should generate embeddings using 'name: summary' pattern."""
        mock_openai.create_embeddings_batch.return_value = [sample_embedding]

        embeddings = await extractor._generate_embeddings([sample_extracted_deal])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536

        # Verify the text pattern used for embedding
        call_args = mock_openai.create_embeddings_batch.call_args
        texts = call_args.args[0]
        assert len(texts) == 1
        assert texts[0].startswith('Acme Corp Data Platform:')
        assert 'CRM data silo' in texts[0]

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_list(self, extractor, mock_openai):
        """Should handle empty deal list gracefully."""
        embeddings = await extractor._generate_embeddings([])

        assert embeddings == []
        mock_openai.create_embeddings_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_multiple_deals(
        self, extractor, mock_openai, sample_extracted_deal, sample_embedding,
    ):
        """Should generate embeddings for multiple deals in one batch."""
        deal_2 = ExtractedDeal(
            opportunity_name='Second Deal',
            opportunity_summary='Another opportunity.',
            stage_assessment='prospecting',
            confidence=0.7,
            reasoning='Test deal.',
        )
        mock_openai.create_embeddings_batch.return_value = [
            sample_embedding,
            [0.02] * 1536,
        ]

        embeddings = await extractor._generate_embeddings(
            [sample_extracted_deal, deal_2],
        )

        assert len(embeddings) == 2
        call_args = mock_openai.create_embeddings_batch.call_args
        texts = call_args.args[0]
        assert len(texts) == 2
        assert 'Second Deal:' in texts[1]


# =============================================================================
# Envelope Routing Tests
# =============================================================================


class TestExtractFromEnvelope:
    """Test extract_from_envelope routing between Case A and Case B."""

    @pytest.mark.asyncio
    async def test_routes_to_discovery_without_opportunity_id(
        self, extractor, mock_openai, sample_envelope,
        sample_extracted_deal, sample_embedding,
    ):
        """Should route to discovery when no opportunity_id in envelope."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal],
            has_deals=True,
        )
        mock_openai.create_embeddings_batch.return_value = [sample_embedding]

        result, embeddings = await extractor.extract_from_envelope(sample_envelope)

        assert result.has_deals is True
        assert len(result.deals) == 1
        assert len(embeddings) == 1

        # Verify discovery prompt was used (no existing deal context)
        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        assert '<existing_deal>' not in user_content
        assert '<transcript>' in user_content

    @pytest.mark.asyncio
    async def test_routes_to_targeted_with_opportunity_id(
        self, extractor, mock_openai,
        sample_envelope_with_opportunity,
        sample_extracted_deal, sample_existing_deal, sample_embedding,
    ):
        """Should route to targeted when opportunity_id present and existing deal provided."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal],
            has_deals=True,
        )
        mock_openai.create_embeddings_batch.return_value = [sample_embedding]

        result, embeddings = await extractor.extract_from_envelope(
            sample_envelope_with_opportunity,
            existing_deal=sample_existing_deal,
        )

        assert result.has_deals is True
        assert len(result.deals) == 1
        assert len(embeddings) == 1

        # Verify targeted prompt was used (existing deal context present)
        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        assert '<existing_deal>' in user_content

    @pytest.mark.asyncio
    async def test_falls_back_to_discovery_when_no_existing_deal(
        self, extractor, mock_openai,
        sample_envelope_with_opportunity,
        sample_extracted_deal, sample_embedding,
    ):
        """Should fall back to discovery if opportunity_id present but no existing deal."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal],
            has_deals=True,
        )
        mock_openai.create_embeddings_batch.return_value = [sample_embedding]

        # opportunity_id is present in envelope but existing_deal is None
        result, embeddings = await extractor.extract_from_envelope(
            sample_envelope_with_opportunity,
            existing_deal=None,
        )

        # Should use discovery prompt (no existing deal to inject)
        call_args = mock_openai.chat_completion_structured.call_args
        user_content = call_args.kwargs['messages'][1]['content']
        assert '<existing_deal>' not in user_content

    @pytest.mark.asyncio
    async def test_no_embeddings_when_no_deals(
        self, extractor, mock_openai, sample_envelope,
    ):
        """Should not generate embeddings when no deals found."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[],
            has_deals=False,
        )

        result, embeddings = await extractor.extract_from_envelope(sample_envelope)

        assert result.has_deals is False
        assert embeddings == []
        mock_openai.create_embeddings_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_tuple_of_result_and_embeddings(
        self, extractor, mock_openai, sample_envelope,
        sample_extracted_deal, sample_embedding,
    ):
        """Return type should be (DealExtractionResult, list[list[float]])."""
        mock_openai.chat_completion_structured.return_value = DealExtractionResult(
            deals=[sample_extracted_deal],
            has_deals=True,
        )
        mock_openai.create_embeddings_batch.return_value = [sample_embedding]

        output = await extractor.extract_from_envelope(sample_envelope)

        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[0], DealExtractionResult)
        assert isinstance(output[1], list)
