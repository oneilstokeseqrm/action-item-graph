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
from action_item_graph.clients.postgres_client import PostgresClient
from action_item_graph.models.envelope import EnvelopeV1

from ..clients.neo4j_client import DealNeo4jClient
from ..errors import DealExtractionError, DealPipelineError
from ..models.deal import Deal, DealVersion
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
# Neo4j → Model Helpers
# =============================================================================


def _neo4j_node_to_deal(node: dict[str, Any]) -> Deal:
    """Convert a Neo4j Deal node dict to a Deal model instance."""
    from ..models.deal import MEDDICProfile, OntologyScores
    from .merger import TRANSCRIPT_EXTRACTED_DIMENSIONS

    meddic = MEDDICProfile(
        metrics=node.get('meddic_metrics'),
        metrics_confidence=node.get('meddic_metrics_confidence', 0.0),
        economic_buyer=node.get('meddic_economic_buyer'),
        economic_buyer_confidence=node.get('meddic_economic_buyer_confidence', 0.0),
        decision_criteria=node.get('meddic_decision_criteria'),
        decision_criteria_confidence=node.get('meddic_decision_criteria_confidence', 0.0),
        decision_process=node.get('meddic_decision_process'),
        decision_process_confidence=node.get('meddic_decision_process_confidence', 0.0),
        identified_pain=node.get('meddic_identified_pain'),
        identified_pain_confidence=node.get('meddic_identified_pain_confidence', 0.0),
        champion=node.get('meddic_champion'),
        champion_confidence=node.get('meddic_champion_confidence', 0.0),
        paper_process=node.get('meddic_paper_process'),
        competition=node.get('meddic_competition'),
    )

    scores = {}
    confidences = {}
    for dim_id in TRANSCRIPT_EXTRACTED_DIMENSIONS:
        score = node.get(f'dim_{dim_id}')
        if score is not None:
            scores[dim_id] = score
        conf = node.get(f'dim_{dim_id}_confidence')
        if conf is not None:
            confidences[dim_id] = conf

    ontology = OntologyScores(scores=scores, confidences=confidences)

    return Deal(
        tenant_id=UUID(node['tenant_id']),
        opportunity_id=UUID(node['opportunity_id']),
        deal_ref=node.get('deal_ref'),
        name=node.get('name', ''),
        stage=node.get('stage', 'prospecting'),
        amount=node.get('amount'),
        account_id=node.get('account_id'),
        currency=node.get('currency', 'USD'),
        meddic=meddic,
        ontology_scores=ontology,
        ontology_version=node.get('ontology_version'),
        opportunity_summary=node.get('opportunity_summary', ''),
        evolution_summary=node.get('evolution_summary', ''),
        embedding=node.get('embedding'),
        embedding_current=node.get('embedding_current'),
        version=node.get('version', 1),
        confidence=node.get('confidence', 1.0),
        source_interaction_id=UUID(node['source_interaction_id']) if node.get('source_interaction_id') else None,
    )


def _neo4j_node_to_deal_version(node: dict[str, Any]) -> DealVersion:
    """Convert a Neo4j DealVersion node dict to a DealVersion model instance."""
    from ..models.deal import DealVersion as DV

    return DV(
        version_id=UUID(node['version_id']),
        deal_opportunity_id=UUID(node['deal_opportunity_id']),
        tenant_id=UUID(node['tenant_id']),
        version=node.get('version', 1),
        name=node.get('name', ''),
        stage=node.get('stage', 'prospecting'),
        amount=node.get('amount'),
        opportunity_summary=node.get('opportunity_summary', ''),
        evolution_summary=node.get('evolution_summary', ''),
        meddic_metrics=node.get('meddic_metrics'),
        meddic_economic_buyer=node.get('meddic_economic_buyer'),
        meddic_decision_criteria=node.get('meddic_decision_criteria'),
        meddic_decision_process=node.get('meddic_decision_process'),
        meddic_identified_pain=node.get('meddic_identified_pain'),
        meddic_champion=node.get('meddic_champion'),
        meddic_completeness=node.get('meddic_completeness'),
        change_summary=node.get('change_summary', ''),
        changed_fields=node.get('changed_fields', []),
        change_source_interaction_id=(
            UUID(node['change_source_interaction_id'])
            if node.get('change_source_interaction_id')
            else None
        ),
    )


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
        postgres_client: PostgresClient | None = None,
    ):
        """
        Initialize the pipeline with all required clients.

        Internally creates the Extractor, Matcher, Merger, and Repository.

        Args:
            neo4j_client: Connected DealNeo4jClient for graph operations
            openai_client: Connected OpenAI client for LLM calls + embeddings
            postgres_client: Optional PostgresClient for dual-write projection
        """
        self.extractor = DealExtractor(openai_client)
        self.matcher = DealMatcher(neo4j_client, openai_client)
        self.merger = DealMerger(neo4j_client, openai_client)
        self.repository = DealRepository(neo4j_client)
        self.postgres = postgres_client

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
        # Step 6b: Dual-write to Postgres (failure-isolated)
        # ------------------------------------------------------------------
        try:
            await self._dual_write_postgres(
                merge_results=result.merge_results,
                tenant_id=tenant_id,
                interaction_id=interaction_id,
                source_user_id=getattr(envelope, 'user_id', None),
            )
        except Exception:
            logger.exception('deal_pipeline.postgres_dual_write_failed')

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

    async def _dual_write_postgres(
        self,
        merge_results: list,
        tenant_id: UUID,
        interaction_id: str | None,
        source_user_id: str | None = None,
    ) -> None:
        """Write Deals to Postgres after Neo4j merge completes.

        Failure-isolated: any exception is logged and swallowed.
        """
        if self.postgres is None:
            return

        for merge_result in merge_results:
            try:
                deal_node = await self.repository.get_deal(
                    tenant_id=tenant_id,
                    opportunity_id=merge_result.opportunity_id,
                )
                if deal_node is None:
                    logger.warning(
                        'deal_pipeline.postgres_dual_write.deal_not_found',
                        opportunity_id=merge_result.opportunity_id,
                    )
                    continue

                deal = _neo4j_node_to_deal(deal_node)

                version = None
                if merge_result.version_created:
                    version = await self._fetch_latest_version(
                        tenant_id=deal.tenant_id,
                        opportunity_id=merge_result.opportunity_id,
                    )

                await self.postgres.persist_deal_full(
                    deal=deal,
                    version=version,
                    interaction_id=UUID(interaction_id) if interaction_id else None,
                    source_user_id=source_user_id,
                )
            except Exception:
                logger.exception(
                    'deal_pipeline.postgres_dual_write.deal_failed',
                    opportunity_id=merge_result.opportunity_id,
                )

    async def _fetch_latest_version(
        self,
        tenant_id: UUID,
        opportunity_id: str,
    ) -> DealVersion | None:
        """Fetch the most recent DealVersion from Neo4j."""
        if not hasattr(self.repository, 'get_latest_version'):
            return None
        version_node = await self.repository.get_latest_version(
            tenant_id=tenant_id,
            opportunity_id=opportunity_id,
        )
        if version_node is None:
            return None
        return _neo4j_node_to_deal_version(version_node)
