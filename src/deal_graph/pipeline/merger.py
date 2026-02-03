"""
Deal merger service.

Executes match decisions from the DealMatcher:
- create_new: Build a Deal model and persist via repo.create_deal
- auto_match / llm_match: LLM synthesis → version snapshot → update deal

Implements the full merge flow including MEDDIC field flattening,
evolution narrative generation, and conditional embedding updates.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from action_item_graph.clients.openai_client import OpenAIClient

from ..clients.neo4j_client import DealNeo4jClient
from ..models.deal import Deal, DealStage, MEDDICProfile
from ..models.extraction import ExtractedDeal, MergedDeal
from ..prompts.merge_deals import build_deal_merge_prompt
from ..repository import DealRepository
from ..utils import uuid7
from .matcher import DealMatchResult

logger = structlog.get_logger(__name__)


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
    'stage',
    'amount',
    'embedding_current',
})


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
        opportunity_id = uuid7()

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

    async def _merge_existing(
        self,
        match_result: DealMatchResult,
        tenant_id: UUID,
        source_interaction_id: UUID | None,
    ) -> DealMergeResult:
        """
        Merge an extraction into an existing Deal.

        Flow:
        1. LLM synthesis — apply MEDDIC merge rules
        2. Version snapshot — capture current state before changes
        3. Update deal — write merged state + increment version
        4. Optionally re-embed — if summary changed substantially

        Args:
            match_result: Match result with matched_deal set
            tenant_id: Tenant UUID
            source_interaction_id: Interaction that triggered this merge

        Returns:
            DealMergeResult for the merge
        """
        matched = match_result.matched_deal
        existing_props = matched.node_properties
        extracted = match_result.extracted_deal
        opportunity_id = matched.opportunity_id

        log = logger.bind(
            opportunity_id=opportunity_id,
            match_type=match_result.match_type,
        )

        # Step 1: LLM synthesis
        messages = build_deal_merge_prompt(
            existing_deal=existing_props,
            extracted_deal=extracted,
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

        log.info(
            'deal_merger.synthesis_complete',
            changed_fields=merged.changed_fields,
            should_update_embedding=merged.should_update_embedding,
            implied_stage=merged.implied_stage,
        )

        # Step 2: Version snapshot (BEFORE applying updates)
        await self.repository.create_version_snapshot(
            tenant_id=tenant_id,
            opportunity_id=opportunity_id,
            change_summary=merged.change_narrative,
            changed_fields=merged.changed_fields,
            change_source_interaction_id=source_interaction_id,
        )

        # Step 3: Build updates dict from merged output
        updates = self._build_updates_dict(merged, existing_props)

        # Step 4: Conditionally update embedding_current
        embedding_updated = False
        if merged.should_update_embedding:
            embedding_text = f'{extracted.opportunity_name}: {merged.opportunity_summary}'
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

        return updates
