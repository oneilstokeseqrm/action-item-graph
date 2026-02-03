"""
Deal pipeline orchestrator.

Wires the three pipeline stages — extraction, matching, merging — into a
single process_envelope() call that takes an EnvelopeV1 and returns a
DealPipelineResult describing every action taken.

Handles both routing modes:
- Case A (Targeted): envelope.opportunity_id present → fetch existing deal,
  targeted extraction, match, merge
- Case B (Discovery): no opportunity_id → discovery extraction, per-deal
  match + merge loop
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog

from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.models.envelope import EnvelopeV1

from ..clients.neo4j_client import DealNeo4jClient
from ..errors import DealExtractionError, DealPipelineError
from ..repository import DealRepository
from .extractor import DealExtractor
from .matcher import DealMatcher
from .merger import DealMergeResult, DealMerger

logger = structlog.get_logger(__name__)


# =============================================================================
# Result Model
# =============================================================================


@dataclass
class DealPipelineResult:
    """
    Aggregate result of processing a single envelope through the deal pipeline.

    Captures what happened at each stage: how many deals were extracted,
    which were created vs merged, and any per-deal errors.
    """

    # Envelope context
    tenant_id: str
    account_id: str | None
    interaction_id: str | None
    opportunity_id: str | None  # Non-None only for Case A

    # Counts
    total_extracted: int
    deals_created: list[str] = field(default_factory=list)
    deals_merged: list[str] = field(default_factory=list)

    # Detailed results
    merge_results: list[DealMergeResult] = field(default_factory=list)
    extraction_notes: str | None = None

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    processing_time_ms: int | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)

    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when no errors occurred."""
        return len(self.errors) == 0

    @property
    def total_processed(self) -> int:
        """Total deals that completed the pipeline (created + merged)."""
        return len(self.deals_created) + len(self.deals_merged)


# =============================================================================
# DealPipeline
# =============================================================================


class DealPipeline:
    """
    Orchestrates the full deal extraction → matching → merging flow.

    Responsibilities:
    - Route to Case A (targeted) or Case B (discovery) based on envelope
    - Fetch existing deal context for Case A
    - Loop through extracted deals, matching and merging each
    - Enrich the Interaction node with processing metadata
    - Collect results and per-deal errors without halting the pipeline
    """

    def __init__(
        self,
        neo4j_client: DealNeo4jClient,
        openai_client: OpenAIClient,
    ):
        """
        Initialize the pipeline with all required clients.

        Internally creates the Extractor, Matcher, Merger, and Repository.

        Args:
            neo4j_client: Connected DealNeo4jClient for graph operations
            openai_client: Connected OpenAI client for LLM calls + embeddings
        """
        self.extractor = DealExtractor(openai_client)
        self.matcher = DealMatcher(neo4j_client, openai_client)
        self.merger = DealMerger(neo4j_client, openai_client)
        self.repository = DealRepository(neo4j_client)

    async def process_envelope(
        self,
        envelope: EnvelopeV1,
    ) -> DealPipelineResult:
        """
        Process a single envelope through the deal pipeline.

        Full flow:
        1. Validate envelope (account_id required)
        2. Ensure account + interaction exist in graph
        3. For Case A: fetch existing deal from repository
        4. Extract deals via LLM
        5. For each extracted deal: match → merge
        6. Enrich interaction with processing metadata
        7. Return aggregate result

        Args:
            envelope: EnvelopeV1 with transcript content

        Returns:
            DealPipelineResult with all actions taken

        Raises:
            DealPipelineError: On validation or fatal infrastructure errors
            DealExtractionError: When extraction stage fails entirely
        """
        started_at = datetime.now(tz=timezone.utc)
        t0 = time.monotonic()

        tenant_id = envelope.tenant_id
        account_id = envelope.account_id
        interaction_id = str(envelope.interaction_id) if envelope.interaction_id else None
        opportunity_id = envelope.opportunity_id

        log = logger.bind(
            tenant_id=str(tenant_id),
            account_id=account_id,
            interaction_id=interaction_id,
            opportunity_id=opportunity_id,
        )

        result = DealPipelineResult(
            tenant_id=str(tenant_id),
            account_id=account_id,
            interaction_id=interaction_id,
            opportunity_id=opportunity_id,
            total_extracted=0,
            started_at=started_at,
        )

        # ------------------------------------------------------------------
        # Step 0: Validate
        # ------------------------------------------------------------------
        if not account_id:
            raise DealPipelineError('account_id is required on envelope')

        # ------------------------------------------------------------------
        # Step 1: Ensure account exists in graph
        # ------------------------------------------------------------------
        t_stage = time.monotonic()
        await self.repository.verify_account(tenant_id, account_id)
        result.stage_timings['verify_account'] = time.monotonic() - t_stage

        # ------------------------------------------------------------------
        # Step 2: Ensure interaction exists in graph
        # ------------------------------------------------------------------
        t_stage = time.monotonic()
        if interaction_id:
            await self.repository.ensure_interaction(
                tenant_id=tenant_id,
                interaction_id=interaction_id,
                content_text=envelope.content.text,
                interaction_type=envelope.interaction_type.value,
                timestamp=envelope.timestamp,
                source=envelope.source.value,
                trace_id=envelope.trace_id,
            )
        result.stage_timings['ensure_interaction'] = time.monotonic() - t_stage

        # ------------------------------------------------------------------
        # Step 3: For Case A, fetch existing deal context
        # ------------------------------------------------------------------
        existing_deal: dict[str, Any] | None = None
        if opportunity_id:
            t_stage = time.monotonic()
            existing_deal = await self.repository.get_deal(
                tenant_id=tenant_id,
                opportunity_id=opportunity_id,
            )
            result.stage_timings['fetch_existing_deal'] = time.monotonic() - t_stage

            if not existing_deal:
                log.warning(
                    'deal_pipeline.existing_deal_not_found',
                    opportunity_id=opportunity_id,
                )
                result.warnings.append(
                    f'opportunity_id={opportunity_id} not found in graph; '
                    'falling back to discovery extraction'
                )
                # Fall through to discovery mode (existing_deal=None)

        # ------------------------------------------------------------------
        # Step 4: Extract deals from transcript
        # ------------------------------------------------------------------
        t_stage = time.monotonic()
        try:
            extraction_result, embeddings = await self.extractor.extract_from_envelope(
                envelope=envelope,
                existing_deal=existing_deal,
            )
        except Exception as exc:
            log.error('deal_pipeline.extraction_failed', error=str(exc))
            raise DealExtractionError(f'Extraction failed: {exc}') from exc
        result.stage_timings['extraction'] = time.monotonic() - t_stage

        result.total_extracted = len(extraction_result.deals)
        result.extraction_notes = extraction_result.extraction_notes

        log.info(
            'deal_pipeline.extraction_complete',
            deal_count=result.total_extracted,
            has_deals=extraction_result.has_deals,
        )

        if not extraction_result.has_deals or not extraction_result.deals:
            log.info('deal_pipeline.no_deals_extracted')
            result.completed_at = datetime.now(tz=timezone.utc)
            result.processing_time_ms = int((time.monotonic() - t0) * 1000)
            # Still enrich the interaction even with 0 deals
            await self._enrich_interaction(tenant_id, interaction_id, 0)
            return result

        # ------------------------------------------------------------------
        # Step 5: Match + Merge loop (per extracted deal)
        # ------------------------------------------------------------------
        t_stage = time.monotonic()

        for i, extracted_deal in enumerate(extraction_result.deals):
            embedding = embeddings[i] if i < len(embeddings) else []
            deal_log = log.bind(
                deal_name=extracted_deal.opportunity_name,
                deal_index=i,
            )

            try:
                # Step 5a: Match
                match_result = await self.matcher.find_matches(
                    extracted_deal=extracted_deal,
                    embedding=embedding,
                    tenant_id=tenant_id,
                    account_id=account_id,
                )

                deal_log.info(
                    'deal_pipeline.matched',
                    match_type=match_result.match_type,
                    candidates=match_result.candidates_evaluated,
                )

                # Step 5b: Merge (create or update)
                merge_result = await self.merger.merge_deal(
                    match_result=match_result,
                    tenant_id=tenant_id,
                    account_id=account_id,
                    source_interaction_id=envelope.interaction_id,
                )

                # Step 5c: Collect results
                result.merge_results.append(merge_result)
                if merge_result.action == 'created':
                    result.deals_created.append(merge_result.opportunity_id)
                elif merge_result.action == 'merged':
                    result.deals_merged.append(merge_result.opportunity_id)

                deal_log.info(
                    'deal_pipeline.deal_processed',
                    action=merge_result.action,
                    opportunity_id=merge_result.opportunity_id,
                )

            except Exception as exc:
                # Per-deal errors are accumulated, not fatal
                error_msg = (
                    f'Deal "{extracted_deal.opportunity_name}" '
                    f'(index {i}): {exc}'
                )
                deal_log.error('deal_pipeline.deal_failed', error=str(exc))
                result.errors.append(error_msg)

        result.stage_timings['match_merge'] = time.monotonic() - t_stage

        # ------------------------------------------------------------------
        # Step 6: Enrich interaction with processing metadata
        # ------------------------------------------------------------------
        await self._enrich_interaction(
            tenant_id, interaction_id, result.total_processed
        )

        # ------------------------------------------------------------------
        # Step 7: Finalize
        # ------------------------------------------------------------------
        result.completed_at = datetime.now(tz=timezone.utc)
        result.processing_time_ms = int((time.monotonic() - t0) * 1000)

        log.info(
            'deal_pipeline.complete',
            total_extracted=result.total_extracted,
            deals_created=len(result.deals_created),
            deals_merged=len(result.deals_merged),
            errors=len(result.errors),
            processing_time_ms=result.processing_time_ms,
        )

        return result

    async def _enrich_interaction(
        self,
        tenant_id: UUID,
        interaction_id: str | None,
        deal_count: int,
    ) -> None:
        """
        Add pipeline processing metadata to the Interaction node.

        Best-effort — failures are logged but not raised.

        Args:
            tenant_id: Tenant UUID
            interaction_id: Interaction identifier (may be None)
            deal_count: Number of deals successfully processed
        """
        if not interaction_id:
            return

        try:
            await self.repository.enrich_interaction(
                tenant_id=tenant_id,
                interaction_id=interaction_id,
                deal_count=deal_count,
            )
        except Exception as exc:
            logger.warning(
                'deal_pipeline.enrich_interaction_failed',
                interaction_id=interaction_id,
                error=str(exc),
            )
