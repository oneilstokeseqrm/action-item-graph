"""
Envelope dispatcher for parallel pipeline routing.

Routes each EnvelopeV1 to both the Action Item pipeline and the Deal
pipeline concurrently using asyncio.gather(return_exceptions=True).

Fault isolation guarantee: one pipeline failing never stops the other.
The DispatcherResult captures either a successful result or the exception
from each pipeline independently.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from action_item_graph.models.envelope import EnvelopeV1
from action_item_graph.pipeline.pipeline import ActionItemPipeline, PipelineResult

from deal_graph.pipeline.pipeline import DealPipeline, DealPipelineResult

logger = structlog.get_logger(__name__)


# =============================================================================
# Result Model
# =============================================================================


@dataclass
class DispatcherResult:
    """
    Aggregate result from dispatching an envelope to both pipelines.

    Each pipeline slot holds either a successful result object or the
    Exception that was raised.  The caller can inspect each independently.
    """

    # Envelope context
    tenant_id: str
    account_id: str | None
    interaction_id: str | None

    # Pipeline results (result OR exception, never both)
    action_item_result: PipelineResult | BaseException | None = None
    deal_result: DealPipelineResult | BaseException | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    dispatch_time_ms: int | None = None

    # Errors captured at the dispatcher level (not pipeline-internal errors)
    errors: list[str] = field(default_factory=list)

    @property
    def action_item_success(self) -> bool:
        """True when the action item pipeline returned a result (not an exception)."""
        return isinstance(self.action_item_result, PipelineResult)

    @property
    def deal_success(self) -> bool:
        """True when the deal pipeline returned a result (not an exception)."""
        return isinstance(self.deal_result, DealPipelineResult)

    @property
    def both_succeeded(self) -> bool:
        """True when both pipelines completed without raising."""
        return self.action_item_success and self.deal_success

    @property
    def any_succeeded(self) -> bool:
        """True when at least one pipeline completed without raising."""
        return self.action_item_success or self.deal_success

    @property
    def overall_success(self) -> bool:
        """
        True when no critical infrastructure failures occurred.

        Defined as: at least one pipeline returned a result (not an exception).
        A single pipeline failing is a pipeline-level issue, not a dispatcher
        infrastructure failure — the dispatcher still delivered partial results.

        Use ``both_succeeded`` if you need to confirm zero exceptions.
        """
        return self.any_succeeded

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging / API responses."""
        return {
            'tenant_id': self.tenant_id,
            'account_id': self.account_id,
            'interaction_id': self.interaction_id,
            'action_item_success': self.action_item_success,
            'deal_success': self.deal_success,
            'overall_success': self.overall_success,
            'dispatch_time_ms': self.dispatch_time_ms,
            'errors': self.errors,
        }


# =============================================================================
# EnvelopeDispatcher
# =============================================================================


class EnvelopeDispatcher:
    """
    Routes envelopes to the Action Item and Deal pipelines concurrently.

    Uses asyncio.gather(return_exceptions=True) so that one pipeline
    raising an exception never cancels or blocks the other.
    """

    def __init__(
        self,
        action_item_pipeline: ActionItemPipeline,
        deal_pipeline: DealPipeline,
    ):
        """
        Initialize with pre-configured pipeline instances.

        Args:
            action_item_pipeline: Fully initialized ActionItemPipeline
            deal_pipeline: Fully initialized DealPipeline
        """
        self.action_item_pipeline = action_item_pipeline
        self.deal_pipeline = deal_pipeline

    async def dispatch(
        self,
        envelope: EnvelopeV1,
    ) -> DispatcherResult:
        """
        Dispatch an envelope to both pipelines concurrently.

        Flow:
        1. Launch both process_envelope() calls via asyncio.gather
        2. Inspect each result — either a pipeline result or an exception
        3. Build and return DispatcherResult

        Args:
            envelope: EnvelopeV1 payload to process

        Returns:
            DispatcherResult with results or exceptions from each pipeline
        """
        started_at = datetime.now()
        t0 = time.monotonic()

        tenant_id = str(envelope.tenant_id)
        account_id = envelope.account_id
        interaction_id = str(envelope.interaction_id) if envelope.interaction_id else None

        log = logger.bind(
            tenant_id=tenant_id,
            account_id=account_id,
            interaction_id=interaction_id,
        )

        log.info('dispatcher.started')

        result = DispatcherResult(
            tenant_id=tenant_id,
            account_id=account_id,
            interaction_id=interaction_id,
            started_at=started_at,
        )

        # ------------------------------------------------------------------
        # Run both pipelines concurrently with fault isolation
        # ------------------------------------------------------------------
        outcomes = await asyncio.gather(
            self.action_item_pipeline.process_envelope(envelope),
            self.deal_pipeline.process_envelope(envelope),
            return_exceptions=True,
        )

        ai_outcome, deal_outcome = outcomes

        # ------------------------------------------------------------------
        # Classify each outcome
        # ------------------------------------------------------------------
        if isinstance(ai_outcome, BaseException):
            log.error(
                'dispatcher.action_item_failed',
                error=str(ai_outcome),
                error_type=type(ai_outcome).__name__,
            )
            result.action_item_result = ai_outcome
            result.errors.append(
                f'ActionItemPipeline: {type(ai_outcome).__name__}: {ai_outcome}'
            )
        else:
            result.action_item_result = ai_outcome
            log.info(
                'dispatcher.action_item_complete',
                created=len(ai_outcome.created_ids),
                updated=len(ai_outcome.updated_ids),
            )

        if isinstance(deal_outcome, BaseException):
            log.error(
                'dispatcher.deal_failed',
                error=str(deal_outcome),
                error_type=type(deal_outcome).__name__,
            )
            result.deal_result = deal_outcome
            result.errors.append(
                f'DealPipeline: {type(deal_outcome).__name__}: {deal_outcome}'
            )
        else:
            result.deal_result = deal_outcome
            log.info(
                'dispatcher.deal_complete',
                created=len(deal_outcome.deals_created),
                merged=len(deal_outcome.deals_merged),
            )

        # ------------------------------------------------------------------
        # Finalize
        # ------------------------------------------------------------------
        result.completed_at = datetime.now()
        result.dispatch_time_ms = int((time.monotonic() - t0) * 1000)

        log.info(
            'dispatcher.complete',
            overall_success=result.overall_success,
            action_item_success=result.action_item_success,
            deal_success=result.deal_success,
            dispatch_time_ms=result.dispatch_time_ms,
        )

        return result
