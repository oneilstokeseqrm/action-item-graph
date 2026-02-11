"""
End-to-end integration tests for the full Envelope → Dispatcher → Pipelines flow.

Strategy:
- Real objects for EnvelopeDispatcher, ActionItemPipeline, DealPipeline,
  and all their internal components (extractors, matchers, mergers, repositories).
- Mock only the IO edges: OpenAIClient (shared) and Neo4j clients (separate per pipeline).
- Verify the full dependency wiring works when the real orchestration runs.

Scenarios:
1. "The Double Play": A transcript triggers both an action item and a deal.
   Both pipelines succeed; verify both repositories were called.
2. "Partial System Failure": Deal Neo4j fails during deal creation.
   Action items still succeed; dispatcher captures the deal error.

Run with: pytest tests/test_integration_e2e.py -v

No API keys or Neo4j required — only IO edges are mocked.
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
from action_item_graph.pipeline.pipeline import ActionItemPipeline, PipelineResult
from action_item_graph.prompts.extract_action_items import (
    ExtractedActionItem,
    ExtractedTopic,
    ExtractionResult,
)

from deal_graph.models.extraction import (
    DealExtractionResult,
    ExtractedDeal,
)
from deal_graph.pipeline.pipeline import DealPipeline, DealPipelineResult
from dispatcher.dispatcher import EnvelopeDispatcher


# =============================================================================
# Constants
# =============================================================================


TENANT_ID = UUID('550e8400-e29b-41d4-a716-446655440000')
INTERACTION_ID = UUID('660e8400-e29b-41d4-a716-446655440001')
ACCOUNT_ID = 'acct_acme_001'


# =============================================================================
# Test Envelope
# =============================================================================


def _make_transcript_envelope() -> EnvelopeV1:
    """
    Build an envelope containing a transcript with BOTH an action item
    and a deal signal — the "double play" scenario.
    """
    return EnvelopeV1(
        tenant_id=TENANT_ID,
        user_id='auth0|user_test_001',
        interaction_type=InteractionType.TRANSCRIPT,
        content=ContentPayload(
            text=(
                'Sarah: I will email the contract to legal for review by Friday.\n'
                'John: Sounds good. Also, the budget is confirmed at $50k for the '
                'data platform. Jane Smith from finance signed off.'
            ),
            format=ContentFormat.DIARIZED,
        ),
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        source=SourceType.WEB_MIC,
        account_id=ACCOUNT_ID,
        interaction_id=INTERACTION_ID,
    )


# =============================================================================
# LLM Response Fixtures
# =============================================================================


def _action_item_extraction() -> ExtractionResult:
    """The action item LLM would extract from the transcript."""
    return ExtractionResult(
        action_items=[
            ExtractedActionItem(
                action_item_text='Email the contract to legal for review by Friday',
                owner='Sarah',
                summary='Sarah to send contract to legal team',
                conversation_context='Discussed finalizing the Acme deal contract',
                topic=ExtractedTopic(
                    name='Contract Review',
                    context='Legal review needed for Acme deal contract before signing',
                ),
                confidence=0.92,
                is_status_update=False,
            ),
        ],
        has_action_items=True,
    )


def _deal_extraction() -> DealExtractionResult:
    """The deal LLM would extract from the transcript."""
    return DealExtractionResult(
        deals=[
            ExtractedDeal(
                opportunity_name='Acme Data Platform',
                opportunity_summary='Enterprise data platform with confirmed $50k budget',
                stage_assessment='qualification',
                economic_buyer='Jane Smith from finance',
                identified_pain='Data fragmentation across systems',
                estimated_amount=50000.0,
                confidence=0.88,
                reasoning='Budget confirmed at $50k, CFO signed off',
            ),
        ],
        has_deals=True,
    )


# =============================================================================
# Mock Builders
# =============================================================================


def _build_openai_mock() -> AsyncMock:
    """
    Shared OpenAI client mock.

    Routes chat_completion_structured calls based on response_model
    to return the correct Pydantic type — mirrors how the real client
    dispatches to different parsers.
    """
    mock = AsyncMock()

    async def _route_structured(messages, response_model):
        if response_model is ExtractionResult:
            return _action_item_extraction()
        elif response_model is DealExtractionResult:
            return _deal_extraction()
        # Matchers use DeduplicationDecision — but with empty vector search
        # results, no dedup calls will be made. If reached, fail loudly.
        raise ValueError(f'Unexpected response_model in test: {response_model}')

    mock.chat_completion_structured = AsyncMock(side_effect=_route_structured)
    mock.create_embeddings_batch = AsyncMock(return_value=[[0.1] * 1536])
    mock.create_embedding = AsyncMock(return_value=[0.1] * 1536)

    return mock


def _build_ai_neo4j_mock() -> AsyncMock:
    """
    Action Item Neo4j client mock.

    Returns a generic result dict containing all possible keys so that
    every repository method can extract the key it needs.
    """
    mock = AsyncMock()

    generic_result = [{
        'account': {
            'account_id': ACCOUNT_ID,
            'tenant_id': str(TENANT_ID),
            'name': 'Acme Corp',
        },
        'interaction': {
            'interaction_id': str(INTERACTION_ID),
            'tenant_id': str(TENANT_ID),
        },
        'action_item': {
            'action_item_id': 'ai_integration_001',
            'tenant_id': str(TENANT_ID),
            'action_item_text': 'Email the contract to legal for review by Friday',
        },
        'owner': {
            'owner_id': 'owner_sarah_001',
            'canonical_name': 'Sarah',
        },
        'created': True,
    }]

    mock.execute_write = AsyncMock(return_value=generic_result)
    mock.execute_query = AsyncMock(return_value=[])
    # No existing action items → matcher produces create_new for all
    mock.search_both_embeddings = AsyncMock(return_value=[])

    return mock


def _build_deal_neo4j_mock() -> AsyncMock:
    """
    Deal Neo4j client mock.

    Returns a generic result dict for all write operations.
    Vector search returns empty (no existing deals).
    """
    mock = AsyncMock()

    generic_result = [{
        'account': {
            'account_id': ACCOUNT_ID,
            'tenant_id': str(TENANT_ID),
        },
        'interaction': {
            'interaction_id': str(INTERACTION_ID),
            'tenant_id': str(TENANT_ID),
        },
        'deal': {
            'opportunity_id': '019c1fa0-ff00-7000-8000-000000000010',
            'tenant_id': str(TENANT_ID),
            'name': 'Acme Data Platform',
            'stage': 'qualification',
        },
    }]

    mock.execute_write = AsyncMock(return_value=generic_result)
    mock.execute_query = AsyncMock(return_value=[])
    # No existing deals → matcher produces create_new
    mock.search_deals_both_embeddings = AsyncMock(return_value=[])

    return mock


# =============================================================================
# Scenario 1: "The Double Play"
# =============================================================================


class TestDoublePlay:
    """
    A single transcript triggers both an action item and a deal.

    Both pipelines run concurrently, both succeed, and we verify the
    full wiring from dispatcher → pipeline → merger → repository.
    """

    @pytest.fixture
    def wired_system(self):
        """Wire up real objects with mocked IO edges."""
        mock_openai = _build_openai_mock()
        mock_ai_neo4j = _build_ai_neo4j_mock()
        mock_deal_neo4j = _build_deal_neo4j_mock()

        # Real pipelines (topics disabled to simplify mock chain)
        ai_pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_ai_neo4j,
            enable_topics=False,
        )
        deal_pipeline = DealPipeline(
            neo4j_client=mock_deal_neo4j,
            openai_client=mock_openai,
        )

        # Spy on the repository methods we want to verify
        # (wraps the original so the real method still executes)
        original_create_ai = ai_pipeline.merger.repository.create_action_item
        ai_create_spy = AsyncMock(side_effect=original_create_ai)
        ai_pipeline.merger.repository.create_action_item = ai_create_spy

        original_create_deal = deal_pipeline.merger.repository.create_deal
        deal_create_spy = AsyncMock(side_effect=original_create_deal)
        deal_pipeline.merger.repository.create_deal = deal_create_spy

        # Real dispatcher
        dispatcher = EnvelopeDispatcher(ai_pipeline, deal_pipeline)

        return {
            'dispatcher': dispatcher,
            'ai_create_spy': ai_create_spy,
            'deal_create_spy': deal_create_spy,
            'mock_openai': mock_openai,
            'mock_ai_neo4j': mock_ai_neo4j,
            'mock_deal_neo4j': mock_deal_neo4j,
        }

    @pytest.mark.asyncio
    async def test_both_pipelines_succeed(self, wired_system):
        """Dispatcher returns overall_success=True with both results."""
        result = await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        assert result.overall_success is True
        assert result.both_succeeded is True
        assert isinstance(result.action_item_result, PipelineResult)
        assert isinstance(result.deal_result, DealPipelineResult)

    @pytest.mark.asyncio
    async def test_action_item_created(self, wired_system):
        """ActionItemRepository.create_action_item was awaited."""
        await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        wired_system['ai_create_spy'].assert_awaited()

    @pytest.mark.asyncio
    async def test_deal_created(self, wired_system):
        """DealRepository.create_deal was awaited."""
        await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        wired_system['deal_create_spy'].assert_awaited()

    @pytest.mark.asyncio
    async def test_action_item_result_has_created_ids(self, wired_system):
        """The action item pipeline reports at least one created ID."""
        result = await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        ai_result = result.action_item_result
        assert isinstance(ai_result, PipelineResult)
        assert ai_result.total_extracted >= 1
        assert len(ai_result.created_ids) >= 1

    @pytest.mark.asyncio
    async def test_deal_result_has_created_ids(self, wired_system):
        """The deal pipeline reports at least one created deal."""
        result = await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        deal_result = result.deal_result
        assert isinstance(deal_result, DealPipelineResult)
        assert deal_result.total_extracted >= 1
        assert len(deal_result.deals_created) >= 1

    @pytest.mark.asyncio
    async def test_openai_called_for_both_extractions(self, wired_system):
        """Shared OpenAI client received extraction calls from both pipelines."""
        await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        calls = wired_system['mock_openai'].chat_completion_structured.call_args_list
        response_models = [c.kwargs['response_model'] for c in calls]

        assert ExtractionResult in response_models, 'Action item extraction not called'
        assert DealExtractionResult in response_models, 'Deal extraction not called'

    @pytest.mark.asyncio
    async def test_embeddings_generated_for_both(self, wired_system):
        """Embedding batch generation was called for both pipelines."""
        await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        # Both pipelines call create_embeddings_batch (once each, 1 item each)
        assert wired_system['mock_openai'].create_embeddings_batch.await_count >= 2

    @pytest.mark.asyncio
    async def test_separate_neo4j_clients_used(self, wired_system):
        """Each pipeline writes to its own Neo4j instance."""
        await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        # Action item Neo4j received writes
        assert wired_system['mock_ai_neo4j'].execute_write.await_count >= 1
        # Deal Neo4j received writes
        assert wired_system['mock_deal_neo4j'].execute_write.await_count >= 1

    @pytest.mark.asyncio
    async def test_no_errors_or_warnings_in_results(self, wired_system):
        """Clean run produces no errors."""
        result = await wired_system['dispatcher'].dispatch(_make_transcript_envelope())

        ai_result = result.action_item_result
        deal_result = result.deal_result
        assert isinstance(ai_result, PipelineResult)
        assert isinstance(deal_result, DealPipelineResult)
        assert ai_result.errors == []
        assert deal_result.errors == []
        assert result.errors == []


# =============================================================================
# Scenario 2: "Partial System Failure"
# =============================================================================


class TestPartialSystemFailure:
    """
    Deal Neo4j fails when writing the Deal node, but Action Item
    Neo4j works fine.  The action item should be persisted; the deal
    error should be captured in the DealPipelineResult.
    """

    @pytest.fixture
    def wired_system_with_deal_failure(self):
        """Wire up real objects; deal Neo4j fails on Deal writes only."""
        mock_openai = _build_openai_mock()
        mock_ai_neo4j = _build_ai_neo4j_mock()
        mock_deal_neo4j = _build_deal_neo4j_mock()

        # Make deal Neo4j fail specifically on Deal MERGE/CREATE queries
        # but succeed for Account and Interaction queries.
        generic_success = [{
            'account': {'account_id': ACCOUNT_ID, 'tenant_id': str(TENANT_ID)},
            'interaction': {'interaction_id': str(INTERACTION_ID), 'tenant_id': str(TENANT_ID)},
        }]

        async def _deal_write_with_failure(query, params=None):
            if 'Deal {' in query or 'Deal{' in query:
                raise RuntimeError('Neo4j connection timeout on Deal write')
            return generic_success

        mock_deal_neo4j.execute_write = AsyncMock(side_effect=_deal_write_with_failure)

        # Real pipelines
        ai_pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_ai_neo4j,
            enable_topics=False,
        )
        deal_pipeline = DealPipeline(
            neo4j_client=mock_deal_neo4j,
            openai_client=mock_openai,
        )

        # Spy on action item creation to verify it succeeded
        original_create_ai = ai_pipeline.merger.repository.create_action_item
        ai_create_spy = AsyncMock(side_effect=original_create_ai)
        ai_pipeline.merger.repository.create_action_item = ai_create_spy

        dispatcher = EnvelopeDispatcher(ai_pipeline, deal_pipeline)

        return {
            'dispatcher': dispatcher,
            'ai_create_spy': ai_create_spy,
        }

    @pytest.mark.asyncio
    async def test_overall_success_true(self, wired_system_with_deal_failure):
        """
        Dispatcher overall_success is True because at least one pipeline
        returned a result (action items succeeded).
        """
        sys = wired_system_with_deal_failure
        result = await sys['dispatcher'].dispatch(_make_transcript_envelope())

        assert result.overall_success is True

    @pytest.mark.asyncio
    async def test_action_item_persisted(self, wired_system_with_deal_failure):
        """Action item was created despite the deal failure."""
        sys = wired_system_with_deal_failure
        result = await sys['dispatcher'].dispatch(_make_transcript_envelope())

        # Action item pipeline succeeded
        assert result.action_item_success is True
        ai_result = result.action_item_result
        assert isinstance(ai_result, PipelineResult)
        assert len(ai_result.created_ids) >= 1

        # The repository method was actually called
        sys['ai_create_spy'].assert_awaited()

    @pytest.mark.asyncio
    async def test_deal_error_captured(self, wired_system_with_deal_failure):
        """
        The deal pipeline returns a result with the error captured
        in its errors list (per-deal error, not a pipeline crash).
        """
        sys = wired_system_with_deal_failure
        result = await sys['dispatcher'].dispatch(_make_transcript_envelope())

        # Deal pipeline returned a result (didn't crash the dispatcher)
        deal_result = result.deal_result
        assert isinstance(deal_result, DealPipelineResult)

        # But the result has errors from the failed Deal write
        assert deal_result.success is False
        assert len(deal_result.errors) >= 1
        assert any('Neo4j connection timeout' in e for e in deal_result.errors)

    @pytest.mark.asyncio
    async def test_deal_not_created(self, wired_system_with_deal_failure):
        """No deals were created (the write failed)."""
        sys = wired_system_with_deal_failure
        result = await sys['dispatcher'].dispatch(_make_transcript_envelope())

        deal_result = result.deal_result
        assert isinstance(deal_result, DealPipelineResult)
        assert deal_result.deals_created == []
        assert deal_result.total_processed == 0

    @pytest.mark.asyncio
    async def test_both_pipelines_received_envelope(self, wired_system_with_deal_failure):
        """Both pipelines attempted processing despite one failing mid-flight."""
        sys = wired_system_with_deal_failure
        result = await sys['dispatcher'].dispatch(_make_transcript_envelope())

        # Both pipeline slots are populated (neither is None)
        assert result.action_item_result is not None
        assert result.deal_result is not None
        # One succeeded, one had errors
        assert result.action_item_success is True
        assert result.deal_success is True  # DealPipelineResult was returned (with errors)
