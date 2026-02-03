"""
Tests for EnvelopeDispatcher with mocked pipelines.

Tests cover:
- Both pipelines succeed: results captured, overall_success=True
- Partial failure: one succeeds, one raises — survivor result intact
- Total failure: both raise — overall_success=False, exceptions captured
- Concurrent execution: both pipelines run in parallel (timing verification)
- Envelope context forwarded to both pipelines
- DispatcherResult properties: action_item_success, deal_success, both_succeeded,
  any_succeeded, overall_success, to_dict serialization

Run with: pytest tests/test_dispatcher.py -v

No API keys or Neo4j required — both pipelines are fully mocked.
"""

import asyncio
import time
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
from action_item_graph.pipeline.pipeline import PipelineResult

from deal_graph.errors import DealExtractionError, DealPipelineError
from deal_graph.pipeline.pipeline import DealPipelineResult
from dispatcher.dispatcher import DispatcherResult, EnvelopeDispatcher


# =============================================================================
# Constants
# =============================================================================


TENANT_ID = UUID('550e8400-e29b-41d4-a716-446655440000')
INTERACTION_ID = UUID('660e8400-e29b-41d4-a716-446655440001')
ACCOUNT_ID = 'acct_acme_001'


# =============================================================================
# Fixtures
# =============================================================================


def _make_envelope() -> EnvelopeV1:
    """Build a standard test envelope."""
    return EnvelopeV1(
        tenant_id=TENANT_ID,
        user_id='auth0|user_test_001',
        interaction_type=InteractionType.TRANSCRIPT,
        content=ContentPayload(
            text='Sarah: Let us discuss the platform pricing.',
            format=ContentFormat.DIARIZED,
        ),
        timestamp=datetime(2026, 1, 15, 10, 30, 0),
        source=SourceType.WEB_MIC,
        account_id=ACCOUNT_ID,
        interaction_id=INTERACTION_ID,
    )


def _make_action_item_result() -> PipelineResult:
    """Build a successful ActionItemPipeline result."""
    return PipelineResult(
        envelope_id=None,
        interaction_id=str(INTERACTION_ID),
        account_id=ACCOUNT_ID,
        tenant_id=str(TENANT_ID),
        created_ids=['ai_001', 'ai_002'],
        updated_ids=['ai_003'],
        total_extracted=3,
    )


def _make_deal_result() -> DealPipelineResult:
    """Build a successful DealPipeline result."""
    return DealPipelineResult(
        tenant_id=str(TENANT_ID),
        account_id=ACCOUNT_ID,
        interaction_id=str(INTERACTION_ID),
        opportunity_id=None,
        total_extracted=1,
        deals_created=['deal_001'],
        deals_merged=[],
    )


@pytest.fixture
def mock_ai_pipeline():
    """Mocked ActionItemPipeline."""
    pipeline = AsyncMock()
    pipeline.process_envelope = AsyncMock(return_value=_make_action_item_result())
    return pipeline


@pytest.fixture
def mock_deal_pipeline():
    """Mocked DealPipeline."""
    pipeline = AsyncMock()
    pipeline.process_envelope = AsyncMock(return_value=_make_deal_result())
    return pipeline


@pytest.fixture
def dispatcher(mock_ai_pipeline, mock_deal_pipeline):
    """EnvelopeDispatcher with mocked pipelines."""
    return EnvelopeDispatcher(
        action_item_pipeline=mock_ai_pipeline,
        deal_pipeline=mock_deal_pipeline,
    )


# =============================================================================
# Test: Both Succeed
# =============================================================================


class TestBothSucceed:
    """Happy path: both pipelines complete successfully."""

    @pytest.mark.asyncio
    async def test_both_results_captured(self, dispatcher):
        """Both pipeline results are captured in the DispatcherResult."""
        envelope = _make_envelope()
        result = await dispatcher.dispatch(envelope)

        assert isinstance(result.action_item_result, PipelineResult)
        assert isinstance(result.deal_result, DealPipelineResult)

    @pytest.mark.asyncio
    async def test_overall_success_true(self, dispatcher):
        """overall_success is True when both pipelines succeed."""
        result = await dispatcher.dispatch(_make_envelope())

        assert result.overall_success is True
        assert result.both_succeeded is True
        assert result.any_succeeded is True
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_action_item_result_contents(self, dispatcher):
        """Action item result reflects what the pipeline returned."""
        result = await dispatcher.dispatch(_make_envelope())

        ai_result = result.action_item_result
        assert isinstance(ai_result, PipelineResult)
        assert ai_result.created_ids == ['ai_001', 'ai_002']
        assert ai_result.updated_ids == ['ai_003']

    @pytest.mark.asyncio
    async def test_deal_result_contents(self, dispatcher):
        """Deal result reflects what the pipeline returned."""
        result = await dispatcher.dispatch(_make_envelope())

        deal_result = result.deal_result
        assert isinstance(deal_result, DealPipelineResult)
        assert deal_result.deals_created == ['deal_001']
        assert deal_result.total_extracted == 1


# =============================================================================
# Test: Partial Failure
# =============================================================================


class TestPartialFailure:
    """One pipeline fails, the other succeeds — survivor is unaffected."""

    @pytest.mark.asyncio
    async def test_deal_fails_action_item_succeeds(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """When deal pipeline raises, action item result is still captured."""
        mock_deal_pipeline.process_envelope = AsyncMock(
            side_effect=DealExtractionError('LLM quota exceeded')
        )
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        result = await disp.dispatch(_make_envelope())

        # Action item succeeded
        assert result.action_item_success is True
        assert isinstance(result.action_item_result, PipelineResult)

        # Deal failed — exception captured
        assert result.deal_success is False
        assert isinstance(result.deal_result, DealExtractionError)

        # Overall: at least one succeeded
        assert result.overall_success is True
        assert result.both_succeeded is False
        assert result.any_succeeded is True

    @pytest.mark.asyncio
    async def test_action_item_fails_deal_succeeds(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """When action item pipeline raises, deal result is still captured."""
        mock_ai_pipeline.process_envelope = AsyncMock(
            side_effect=RuntimeError('Neo4j connection lost')
        )
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        result = await disp.dispatch(_make_envelope())

        # Deal succeeded
        assert result.deal_success is True
        assert isinstance(result.deal_result, DealPipelineResult)

        # Action item failed — exception captured
        assert result.action_item_success is False
        assert isinstance(result.action_item_result, RuntimeError)

        # Overall: at least one succeeded
        assert result.overall_success is True

    @pytest.mark.asyncio
    async def test_failed_pipeline_error_in_errors_list(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """The exception message from a failed pipeline appears in errors."""
        mock_deal_pipeline.process_envelope = AsyncMock(
            side_effect=DealPipelineError('account_id is required')
        )
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        result = await disp.dispatch(_make_envelope())

        assert len(result.errors) == 1
        assert 'DealPipelineError' in result.errors[0]
        assert 'account_id is required' in result.errors[0]


# =============================================================================
# Test: Total Failure
# =============================================================================


class TestTotalFailure:
    """Both pipelines fail — overall_success is False."""

    @pytest.mark.asyncio
    async def test_both_fail(self, mock_ai_pipeline, mock_deal_pipeline):
        """When both pipelines raise, overall_success is False."""
        mock_ai_pipeline.process_envelope = AsyncMock(
            side_effect=RuntimeError('AI pipeline down')
        )
        mock_deal_pipeline.process_envelope = AsyncMock(
            side_effect=DealExtractionError('Deal pipeline down')
        )
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        result = await disp.dispatch(_make_envelope())

        assert result.overall_success is False
        assert result.both_succeeded is False
        assert result.any_succeeded is False
        assert result.action_item_success is False
        assert result.deal_success is False

    @pytest.mark.asyncio
    async def test_both_exceptions_captured(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """Both exception objects are preserved in the result."""
        ai_error = RuntimeError('AI exploded')
        deal_error = DealExtractionError('Deal exploded')

        mock_ai_pipeline.process_envelope = AsyncMock(side_effect=ai_error)
        mock_deal_pipeline.process_envelope = AsyncMock(side_effect=deal_error)
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        result = await disp.dispatch(_make_envelope())

        assert isinstance(result.action_item_result, RuntimeError)
        assert str(result.action_item_result) == 'AI exploded'
        assert isinstance(result.deal_result, DealExtractionError)
        assert str(result.deal_result) == 'Deal exploded'

    @pytest.mark.asyncio
    async def test_both_errors_in_errors_list(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """Both error messages appear in the errors list."""
        mock_ai_pipeline.process_envelope = AsyncMock(
            side_effect=RuntimeError('AI down')
        )
        mock_deal_pipeline.process_envelope = AsyncMock(
            side_effect=DealExtractionError('Deal down')
        )
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        result = await disp.dispatch(_make_envelope())

        assert len(result.errors) == 2
        assert any('ActionItemPipeline' in e for e in result.errors)
        assert any('DealPipeline' in e for e in result.errors)


# =============================================================================
# Test: Concurrent Execution
# =============================================================================


class TestConcurrentExecution:
    """Verify both pipelines run in parallel, not sequentially."""

    @pytest.mark.asyncio
    async def test_pipelines_run_concurrently(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """
        Both pipelines sleep 0.1s each. If run concurrently, total time
        should be ~0.1s, not ~0.2s.
        """
        async def slow_ai(envelope):
            await asyncio.sleep(0.1)
            return _make_action_item_result()

        async def slow_deal(envelope):
            await asyncio.sleep(0.1)
            return _make_deal_result()

        mock_ai_pipeline.process_envelope = AsyncMock(side_effect=slow_ai)
        mock_deal_pipeline.process_envelope = AsyncMock(side_effect=slow_deal)

        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)

        t0 = time.monotonic()
        result = await disp.dispatch(_make_envelope())
        elapsed = time.monotonic() - t0

        # Concurrent: should be ~0.1s, not ~0.2s
        # Use 0.18s as threshold to avoid flaky test while still detecting sequential
        assert elapsed < 0.18, f'Expected concurrent execution but took {elapsed:.3f}s'
        assert result.both_succeeded is True

    @pytest.mark.asyncio
    async def test_both_receive_same_envelope(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """Both pipelines receive the exact same envelope object."""
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        envelope = _make_envelope()

        await disp.dispatch(envelope)

        ai_call_envelope = mock_ai_pipeline.process_envelope.call_args[0][0]
        deal_call_envelope = mock_deal_pipeline.process_envelope.call_args[0][0]

        assert ai_call_envelope is envelope
        assert deal_call_envelope is envelope


# =============================================================================
# Test: Envelope Context
# =============================================================================


class TestEnvelopeContext:
    """DispatcherResult captures envelope context."""

    @pytest.mark.asyncio
    async def test_result_context_from_envelope(self, dispatcher):
        """Result captures tenant_id, account_id, interaction_id."""
        result = await dispatcher.dispatch(_make_envelope())

        assert result.tenant_id == str(TENANT_ID)
        assert result.account_id == ACCOUNT_ID
        assert result.interaction_id == str(INTERACTION_ID)

    @pytest.mark.asyncio
    async def test_result_context_without_interaction_id(
        self, mock_ai_pipeline, mock_deal_pipeline
    ):
        """Handles envelopes with no interaction_id."""
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
        disp = EnvelopeDispatcher(mock_ai_pipeline, mock_deal_pipeline)
        result = await disp.dispatch(envelope)

        assert result.interaction_id is None
        assert result.overall_success is True


# =============================================================================
# Test: Timing
# =============================================================================


class TestTiming:
    """Timing fields are populated correctly."""

    @pytest.mark.asyncio
    async def test_timing_fields_populated(self, dispatcher):
        """started_at, completed_at, and dispatch_time_ms are set."""
        result = await dispatcher.dispatch(_make_envelope())

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.dispatch_time_ms is not None
        assert result.dispatch_time_ms >= 0
        assert result.completed_at >= result.started_at


# =============================================================================
# Test: DispatcherResult Properties
# =============================================================================


class TestResultProperties:
    """Unit tests for DispatcherResult computed properties."""

    def test_both_succeeded(self):
        """both_succeeded is True only when both slots hold result objects."""
        result = DispatcherResult(
            tenant_id='t', account_id='a', interaction_id='i',
            action_item_result=_make_action_item_result(),
            deal_result=_make_deal_result(),
        )
        assert result.both_succeeded is True

    def test_both_succeeded_false_when_one_failed(self):
        """both_succeeded is False when one slot holds an exception."""
        result = DispatcherResult(
            tenant_id='t', account_id='a', interaction_id='i',
            action_item_result=_make_action_item_result(),
            deal_result=RuntimeError('oops'),
        )
        assert result.both_succeeded is False

    def test_any_succeeded_with_partial(self):
        """any_succeeded is True when at least one pipeline succeeded."""
        result = DispatcherResult(
            tenant_id='t', account_id='a', interaction_id='i',
            action_item_result=RuntimeError('oops'),
            deal_result=_make_deal_result(),
        )
        assert result.any_succeeded is True

    def test_any_succeeded_false_when_both_failed(self):
        """any_succeeded is False when both slots hold exceptions."""
        result = DispatcherResult(
            tenant_id='t', account_id='a', interaction_id='i',
            action_item_result=RuntimeError('a'),
            deal_result=RuntimeError('b'),
        )
        assert result.any_succeeded is False

    def test_overall_success_is_any_succeeded(self):
        """overall_success equals any_succeeded (errors list is observability only)."""
        result = DispatcherResult(
            tenant_id='t', account_id='a', interaction_id='i',
            action_item_result=_make_action_item_result(),
            deal_result=RuntimeError('oops'),
            errors=['DealPipeline: RuntimeError: oops'],
        )
        assert result.any_succeeded is True
        assert result.overall_success is True

    def test_to_dict_serialization(self):
        """to_dict produces a serializable dict with expected keys."""
        result = DispatcherResult(
            tenant_id='t_123', account_id='a_456', interaction_id='i_789',
            action_item_result=_make_action_item_result(),
            deal_result=RuntimeError('failed'),
            errors=['DealPipeline: RuntimeError: failed'],
        )
        d = result.to_dict()

        assert d['tenant_id'] == 't_123'
        assert d['action_item_success'] is True
        assert d['deal_success'] is False
        assert d['overall_success'] is True
        assert d['errors'] == ['DealPipeline: RuntimeError: failed']

    def test_none_results_not_success(self):
        """None result slots (not yet populated) are not considered success."""
        result = DispatcherResult(
            tenant_id='t', account_id='a', interaction_id='i',
        )
        assert result.action_item_success is False
        assert result.deal_success is False
        assert result.both_succeeded is False
        assert result.any_succeeded is False
        assert result.overall_success is False
