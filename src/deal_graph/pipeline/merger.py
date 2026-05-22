"""
Deal merger service.

Executes match decisions from the DealMatcher:
- create_new: Build a Deal model and persist via repo.create_deal
- auto_match / llm_match: LLM synthesis → version snapshot → update deal

Implements the full merge flow including MEDDIC field flattening,
evolution narrative generation, and conditional embedding updates.
"""

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import structlog

from action_item_graph.clients.openai_client import OpenAIClient

from ..clients.neo4j_client import DealNeo4jClient
from ..models.deal import Deal, DealStage, MEDDICProfile, OntologyScores
from ..models.extraction import DimensionExtraction, ExtractedDeal, MergedDeal
from ..prompts.merge_deals import build_deal_merge_prompt
from ..repository import DealRepository
from .matcher import DealMatchResult

logger = structlog.get_logger(__name__)


def _deal_extraction_content_hash(extracted: ExtractedDeal) -> str:
    """Stable content hash for two-extractions-to-same-deal disambiguation.

    Hashes the full serialized ExtractedDeal so two extractions sharing
    ``opportunity_name`` but differing on stage, MEDDIC fields, amount,
    etc. produce distinct snapshot version_ids. Byte-identical
    extractions collapse — and that's an upstream consolidation concern,
    not a snapshot concern. Mirrors
    ``action_item_graph.pipeline.merger._extraction_content_hash``
    (Codex B-2 R3 absorption).
    """
    canonical = extracted.model_dump_json(round_trip=True)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


# Map stage strings to DealStage enum
STAGE_MAP = {s.value: s for s in DealStage}

# Map bare MEDDIC names → persisted property names
BARE_TO_PREFIXED = {
    'metrics': 'meddic_metrics',
    'economic_buyer': 'meddic_economic_buyer',
    'decision_criteria': 'meddic_decision_criteria',
    'decision_process': 'meddic_decision_process',
    'identified_pain': 'meddic_identified_pain',
    'champion': 'meddic_champion',
}

# Ontology dimension IDs that are LLM-extracted from transcripts
# (excludes computed dimensions like activity_velocity, time_in_stage, etc.)
TRANSCRIPT_EXTRACTED_DIMENSIONS = frozenset({
    # Qualification
    'champion_strength',
    'economic_buyer_access',
    'identified_pain',
    'metrics_business_case',
    'decision_criteria_alignment',
    'decision_process_clarity',
    # Competitive
    'competitive_position',
    'incumbent_displacement_risk',
    # Commercial
    'pricing_alignment',
    'procurement_legal_progress',
    # Engagement
    'responsiveness',
    # Timeline
    'close_date_credibility',
    # Technical
    'technical_fit',
    'integration_security_risk',
    # Organizational
    'change_readiness',
})

# Whitelist: only actual persisted Deal node properties that can change during merge.
# Derived from _build_updates_dict() keys in this file.
DEAL_PROPERTY_WHITELIST = frozenset({
    'opportunity_summary',
    'evolution_summary',
    'meddic_metrics',
    'meddic_economic_buyer',
    'meddic_decision_criteria',
    'meddic_decision_process',
    'meddic_identified_pain',
    'meddic_champion',
    'meddic_completeness',
    'ontology_completeness',
    'stage',
    'amount',
    'embedding_current',
    # Ontology dim_* properties (generated from TRANSCRIPT_EXTRACTED_DIMENSIONS)
    *(f'dim_{d}' for d in TRANSCRIPT_EXTRACTED_DIMENSIONS),
    *(f'dim_{d}_confidence' for d in TRANSCRIPT_EXTRACTED_DIMENSIONS),
})


# =============================================================================
# Helpers
# =============================================================================


def _build_ontology_scores(
    dimensions: list[DimensionExtraction],
) -> OntologyScores:
    """Build OntologyScores from a list of DimensionExtraction results."""
    scores: dict[str, int | None] = {}
    confidences: dict[str, float] = {}
    evidence: dict[str, str | None] = {}
    for dim in dimensions:
        scores[dim.dimension_id] = dim.score
        confidences[dim.dimension_id] = dim.confidence
        evidence[dim.dimension_id] = dim.evidence
    return OntologyScores(scores=scores, confidences=confidences, evidence=evidence)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DealMergeResult:
    """
    Result of executing a merge decision.

    Captures what action was taken, provenance, and whether versioning occurred.
    """

    opportunity_id: str
    action: str  # 'created' | 'merged'
    was_new: bool
    version_created: bool
    source_interaction_id: str | None
    embedding_updated: bool
    details: dict[str, Any]


# =============================================================================
# DealMerger
# =============================================================================


class DealMerger:
    """
    Executes match decisions to create or update Deals in the graph.

    Responsibilities:
    - Create new Deal nodes for unmatched extractions
    - Merge matched extractions via LLM synthesis
    - Create DealVersion snapshots before updates
    - Flatten MEDDIC fields for Neo4j storage
    - Conditionally update embedding_current
    """

    def __init__(
        self,
        neo4j_client: DealNeo4jClient,
        openai_client: OpenAIClient,
    ):
        """
        Initialize the merger.

        Args:
            neo4j_client: Connected DealNeo4jClient
            openai_client: Connected OpenAI client for LLM synthesis + embeddings
        """
        self.neo4j = neo4j_client
        self.openai = openai_client
        self.repository = DealRepository(neo4j_client)

    async def merge_deal(
        self,
        match_result: DealMatchResult,
        tenant_id: UUID,
        account_id: str | None = None,
        source_interaction_id: UUID | None = None,
    ) -> DealMergeResult:
        """
        Execute the merge decision for a matched or unmatched extraction.

        Args:
            match_result: Result from DealMatcher.find_matches()
            tenant_id: Tenant UUID
            account_id: Account ID for scoping
            source_interaction_id: Interaction that triggered this extraction

        Returns:
            DealMergeResult describing what was done
        """
        if match_result.match_type == 'create_new':
            return await self._create_new(
                extracted_deal=match_result.extracted_deal,
                embedding=match_result.embedding,
                tenant_id=tenant_id,
                account_id=account_id,
                source_interaction_id=source_interaction_id,
            )
        else:
            # auto_match or llm_match — both follow the merge path
            return await self._merge_existing(
                match_result=match_result,
                tenant_id=tenant_id,
                source_interaction_id=source_interaction_id,
            )

    async def _create_new(
        self,
        extracted_deal: ExtractedDeal,
        embedding: list[float],
        tenant_id: UUID,
        account_id: str | None,
        source_interaction_id: UUID | None,
    ) -> DealMergeResult:
        """
        Create a new Deal from an unmatched extraction.

        Builds a Deal model from the ExtractedDeal and persists via
        repo.create_deal (MERGE on skeleton key).

        Args:
            extracted_deal: The extracted deal to persist
            embedding: Pre-computed embedding vector
            tenant_id: Tenant UUID
            account_id: Account ID for scoping
            source_interaction_id: Interaction that created this deal

        Returns:
            DealMergeResult for the creation
        """
        # Deterministic opportunity_id for retry-safe MERGE in
        # ``repository.create_deal`` (already MERGE-keyed on
        # ``(tenant_id, opportunity_id)``). D7
        # (``match_merge_loop_step``) is ``retries_allowed=False``, but
        # workflow crash-recovery still re-executes the step from scratch
        # — the prior ``uuid7()`` generation would produce a NEW
        # opportunity_id on every recovery and create a duplicate Deal
        # node. UUID5 over ``(tenant_id, source_interaction_id,
        # _deal_extraction_content_hash(extracted_deal))`` is stable
        # across recoveries: two retries of the same extracted_deal
        # produce the same opportunity_id; two distinct extracted_deals
        # sharing opportunity_name but differing in MEDDIC/stage produce
        # different opportunity_ids (the content hash is wide enough to
        # disambiguate). Per
        # ``memory/pattern_dbos_workflow_parity_rules.md`` Rule 6 —
        # Phase F /review absorption mirroring the Phase B-2 fix to
        # ``create_version_snapshot``.
        opportunity_id = uuid5(
            NAMESPACE_URL,
            f'aig-deal:{tenant_id}:{source_interaction_id or "none"}:'
            f'{_deal_extraction_content_hash(extracted_deal)}',
        )

        # Map stage assessment to DealStage enum
        stage = STAGE_MAP.get(
            extracted_deal.stage_assessment,
            DealStage.PROSPECTING,
        )

        # Build MEDDIC profile from extraction
        meddic = MEDDICProfile(
            metrics=extracted_deal.metrics,
            metrics_confidence=extracted_deal.confidence if extracted_deal.metrics else 0.0,
            economic_buyer=extracted_deal.economic_buyer,
            economic_buyer_confidence=extracted_deal.confidence if extracted_deal.economic_buyer else 0.0,
            decision_criteria=extracted_deal.decision_criteria,
            decision_criteria_confidence=extracted_deal.confidence if extracted_deal.decision_criteria else 0.0,
            decision_process=extracted_deal.decision_process,
            decision_process_confidence=extracted_deal.confidence if extracted_deal.decision_process else 0.0,
            identified_pain=extracted_deal.identified_pain,
            identified_pain_confidence=extracted_deal.confidence if extracted_deal.identified_pain else 0.0,
            champion=extracted_deal.champion,
            champion_confidence=extracted_deal.confidence if extracted_deal.champion else 0.0,
        )

        # Build ontology scores from extraction dimensions
        ontology = _build_ontology_scores(extracted_deal.ontology_dimensions)

        deal = Deal(
            tenant_id=tenant_id,
            opportunity_id=opportunity_id,
            deal_ref=f'deal_{opportunity_id.hex[-16:]}',
            name=extracted_deal.opportunity_name,
            stage=stage,
            amount=extracted_deal.estimated_amount,
            account_id=account_id,
            currency=extracted_deal.currency,
            meddic=meddic,
            ontology_scores=ontology,
            opportunity_summary=extracted_deal.opportunity_summary,
            evolution_summary=f'Initial extraction: {extracted_deal.opportunity_summary}',
            embedding=embedding,
            embedding_current=embedding,
            confidence=extracted_deal.confidence,
            source_interaction_id=source_interaction_id,
        )

        created = await self.repository.create_deal(deal)

        logger.info(
            'deal_merger.created',
            opportunity_id=str(opportunity_id),
            stage=stage.value if isinstance(stage, DealStage) else stage,
            meddic_completeness=meddic.completeness_score,
        )

        return DealMergeResult(
            opportunity_id=str(opportunity_id),
            action='created',
            was_new=True,
            version_created=False,
            source_interaction_id=str(source_interaction_id) if source_interaction_id else None,
            embedding_updated=False,
            details={
                'name': extracted_deal.opportunity_name,
                'stage': stage.value if isinstance(stage, DealStage) else stage,
                'meddic_completeness': meddic.completeness_score,
                'deal_properties': created,
            },
        )

    async def construct_merged_deal_llm(
        self,
        *,
        extracted_deal: ExtractedDeal,
        existing_props: dict[str, Any],
    ) -> dict[str, Any]:
        """Pure-LLM phase of deal merging (no Neo4j writes, no embedding call).

        Calls the LLM with the MEDDIC merge prompt to produce a
        ``MergedDeal`` and normalizes ``changed_fields`` against the
        whitelist. Returns a JSON-safe dict so DBOS can checkpoint it
        and ``persist_merged_deal_neo4j`` can consume it as a separate
        step.

        The conditional embedding call is intentionally NOT in this
        step. Legacy ordering (Codex B-2 absorption) is
        ``LLM-synthesis → snapshot → embedding → update`` — moving the
        embedding into this LLM step would change failure-path write
        semantics. Keep the embedding call inside the persist method
        between snapshot and update.

        D7 inner refactor: splitting LLM from Neo4j write at the
        function level keeps the DBOS step boundary single (per-deal
        failures are fail-open inside the match_merge_loop).
        """
        messages = build_deal_merge_prompt(
            existing_deal=existing_props,
            extracted_deal=extracted_deal,
        )

        merged: MergedDeal = await self.openai.chat_completion_structured(
            messages=messages,
            response_model=MergedDeal,
        )

        # Normalize changed_fields: map bare MEDDIC names and filter to whitelist
        merged.changed_fields = [
            f for f in (BARE_TO_PREFIXED.get(f, f) for f in merged.changed_fields)
            if f in DEAL_PROPERTY_WHITELIST
        ]

        return {
            'merged': merged.model_dump(),
        }

    async def persist_merged_deal_neo4j(
        self,
        *,
        llm_result: dict[str, Any],
        existing_props: dict[str, Any],
        opportunity_id: str,
        tenant_id: UUID,
        source_interaction_id: UUID | None,
        extracted_deal: ExtractedDeal,
    ) -> DealMergeResult:
        """Mostly-Neo4j-write phase of deal merging.

        Consumes the dict produced by :meth:`construct_merged_deal_llm`
        and applies: version snapshot, then optionally an embedding
        refresh (LLM call ordered BETWEEN snapshot and update for
        legacy parity per Codex B-2 absorption), then update_deal.

        ``extracted_deal`` is the upstream ExtractedDeal. Its
        ``opportunity_name`` feeds the embedding text composition
        (legacy: ``f'{extracted.opportunity_name}: {merged.opportunity_summary}'``)
        and its full content hash feeds the snapshot disambiguator
        (Codex B-2 R3 absorption).
        """
        merged = MergedDeal.model_validate(llm_result['merged'])

        log = logger.bind(opportunity_id=opportunity_id)

        log.info(
            'deal_merger.synthesis_complete',
            changed_fields=merged.changed_fields,
            should_update_embedding=merged.should_update_embedding,
            implied_stage=merged.implied_stage,
        )

        # Step 2: Version snapshot (BEFORE applying updates).
        # The repository uses a deterministic version_id derived from
        # (opportunity_id, source_interaction_id, extraction_disambiguator)
        # so this MERGE is idempotent under DBOS retry AND distinguishes
        # two separate extracted deals that resolve to the same existing
        # opportunity_id within one envelope (Codex B-2 R3 absorption).
        # Hash the FULL ExtractedDeal so two extractions with identical
        # opportunity_name but different stage/MEDDIC fields still
        # produce distinct snapshots.
        await self.repository.create_version_snapshot(
            tenant_id=tenant_id,
            opportunity_id=opportunity_id,
            change_summary=merged.change_narrative,
            changed_fields=merged.changed_fields,
            change_source_interaction_id=source_interaction_id,
            extraction_disambiguator=_deal_extraction_content_hash(extracted_deal),
        )

        # Step 3: Build updates dict from merged output
        updates = self._build_updates_dict(merged, existing_props)

        # Step 4: Embedding refresh between snapshot and update —
        # legacy ordering per Codex B-2 absorption. The embedding call
        # is the only LLM operation in this otherwise pure-Neo4j-write
        # phase; placing it here matches the legacy
        # ``snapshot → embedding → update`` order.
        embedding_updated = False
        if merged.should_update_embedding:
            embedding_text = (
                f'{extracted_deal.opportunity_name}: {merged.opportunity_summary}'
            )
            new_embedding = await self.openai.create_embedding(embedding_text)
            updates['embedding_current'] = new_embedding
            embedding_updated = True

        # Step 5: Persist updates
        updated = await self.repository.update_deal(
            tenant_id=tenant_id,
            opportunity_id=opportunity_id,
            updates=updates,
        )

        log.info(
            'deal_merger.merged',
            version_created=True,
            embedding_updated=embedding_updated,
            fields_updated=list(updates.keys()),
        )

        return DealMergeResult(
            opportunity_id=opportunity_id,
            action='merged',
            was_new=False,
            version_created=True,
            source_interaction_id=str(source_interaction_id) if source_interaction_id else None,
            embedding_updated=embedding_updated,
            details={
                'change_narrative': merged.change_narrative,
                'changed_fields': merged.changed_fields,
                'implied_stage': merged.implied_stage,
                'stage_reasoning': merged.stage_reasoning,
                'evolution_summary': merged.evolution_summary,
                'deal_properties': updated,
            },
        )

    async def _merge_existing(
        self,
        match_result: DealMatchResult,
        tenant_id: UUID,
        source_interaction_id: UUID | None,
    ) -> DealMergeResult:
        """
        Merge an extraction into an existing Deal.

        Implemented as a sequential call to
        :meth:`construct_merged_deal_llm` followed by
        :meth:`persist_merged_deal_neo4j`. The legacy ``/process`` HTTP
        route enters here; the DBOS workflow path's D7 step also calls
        this method (single DBOS step boundary preserved because
        per-deal failures are fail-open inside match_merge_loop), but
        the inner function refactor surfaces the split for clarity and
        for any future move to per-deal step granularity.
        """
        matched = match_result.matched_deal
        if matched is None:  # defensive — _merge_existing is only called when matched is set
            raise RuntimeError(
                'deal_merger._merge_existing called with matched_deal=None'
            )
        llm_result = await self.construct_merged_deal_llm(
            extracted_deal=match_result.extracted_deal,
            existing_props=matched.node_properties,
        )
        return await self.persist_merged_deal_neo4j(
            llm_result=llm_result,
            existing_props=matched.node_properties,
            opportunity_id=matched.opportunity_id,
            tenant_id=tenant_id,
            source_interaction_id=source_interaction_id,
            extracted_deal=match_result.extracted_deal,
        )

    def _build_updates_dict(
        self,
        merged: MergedDeal,
        existing_props: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build the flat property dict for repo.update_deal().

        Translates MergedDeal LLM output into Neo4j properties:
        - Always includes summaries and evolution_summary
        - Only includes MEDDIC fields when non-None (None = keep existing)
        - Flattens MEDDIC fields with meddic_ prefix
        - Maps implied_stage to DealStage enum value

        Args:
            merged: LLM synthesis output
            existing_props: Current deal properties (for fallback)

        Returns:
            Dict of properties to SET on the Deal node
        """
        updates: dict[str, Any] = {
            'opportunity_summary': merged.opportunity_summary,
            'evolution_summary': merged.evolution_summary,
        }

        # MEDDIC fields — only include when LLM provided a value (None = keep existing)
        meddic_fields = {
            'meddic_metrics': merged.metrics,
            'meddic_economic_buyer': merged.economic_buyer,
            'meddic_decision_criteria': merged.decision_criteria,
            'meddic_decision_process': merged.decision_process,
            'meddic_identified_pain': merged.identified_pain,
            'meddic_champion': merged.champion,
        }
        for key, value in meddic_fields.items():
            if value is not None:
                updates[key] = value

        # Recompute meddic_completeness from the merged state
        final_meddic = {
            'metrics': merged.metrics or existing_props.get('meddic_metrics'),
            'economic_buyer': merged.economic_buyer or existing_props.get('meddic_economic_buyer'),
            'decision_criteria': merged.decision_criteria or existing_props.get('meddic_decision_criteria'),
            'decision_process': merged.decision_process or existing_props.get('meddic_decision_process'),
            'identified_pain': merged.identified_pain or existing_props.get('meddic_identified_pain'),
            'champion': merged.champion or existing_props.get('meddic_champion'),
        }
        populated = sum(1 for v in final_meddic.values() if v)
        updates['meddic_completeness'] = populated / 6

        # Stage — map to enum value if provided
        if merged.implied_stage:
            stage = STAGE_MAP.get(merged.implied_stage)
            if stage:
                updates['stage'] = stage.value

        # Amount
        if merged.amount is not None:
            updates['amount'] = merged.amount

        # Ontology dimension scores — update dim_* properties from merged dimensions
        for dim in merged.ontology_dimensions:
            if dim.score is not None:
                updates[f'dim_{dim.dimension_id}'] = dim.score
                updates[f'dim_{dim.dimension_id}_confidence'] = dim.confidence

        # Recompute ontology_completeness from merged state
        all_dim_ids = list(TRANSCRIPT_EXTRACTED_DIMENSIONS)
        scored_count = 0
        total_count = len(all_dim_ids)
        for dim_id in all_dim_ids:
            # Check merged value first, then existing
            merged_score = next(
                (d.score for d in merged.ontology_dimensions if d.dimension_id == dim_id),
                None,
            )
            existing_score = existing_props.get(f'dim_{dim_id}')
            if merged_score is not None or existing_score is not None:
                scored_count += 1
        if total_count > 0:
            updates['ontology_completeness'] = scored_count / total_count

        return updates
