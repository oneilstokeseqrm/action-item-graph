"""
Deal matching and entity resolution service.

Implements the graduated threshold strategy for Deal entity resolution:
- >= 0.90: Auto-match (high confidence, skip LLM)
- 0.70-0.90: LLM decides (borderline, semantic judgment needed)
- < 0.70: Create new (too dissimilar)

Uses the dual embedding search strategy from DealNeo4jClient:
- deal_embedding_idx (original, immutable) catches new deals similar to initial state
- deal_embedding_current_idx (current, mutable) catches updates to evolved deals
Results are deduplicated by opportunity_id before threshold evaluation.
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog

from action_item_graph.clients.openai_client import OpenAIClient

from ..clients.neo4j_client import DealNeo4jClient
from ..config import DealConfig
from ..models.extraction import DealDeduplicationDecision, ExtractedDeal
from ..prompts.dedup_deals import build_deal_deduplication_prompt

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DealMatchCandidate:
    """A candidate match from dual vector search."""

    opportunity_id: str
    node_properties: dict[str, Any]
    similarity_score: float


@dataclass
class DealMatchResult:
    """
    Result of matching a single extracted deal against existing deals.

    match_type values:
    - 'auto_match': Top candidate scored >= auto_match_threshold (no LLM call)
    - 'llm_match': LLM confirmed a borderline candidate is the same deal
    - 'create_new': No match found (zero candidates, below threshold, or LLM rejected)
    """

    extracted_deal: ExtractedDeal
    embedding: list[float]
    match_type: str  # 'auto_match' | 'llm_match' | 'create_new'
    matched_deal: DealMatchCandidate | None
    decision: DealDeduplicationDecision | None
    candidates_evaluated: int
    all_candidates: list[DealMatchCandidate] = field(default_factory=list)


# =============================================================================
# DealMatcher
# =============================================================================


class DealMatcher:
    """
    Matches extracted deals against existing Deal nodes in the graph.

    Pipeline:
    1. Dual vector search (original + current embeddings)
    2. Dedup by opportunity_id (handled by DealNeo4jClient)
    3. Graduated threshold evaluation
    4. Optional LLM deduplication for borderline candidates
    """

    def __init__(
        self,
        neo4j_client: DealNeo4jClient,
        openai_client: OpenAIClient,
        similarity_threshold: float | None = None,
        auto_match_threshold: float | None = None,
    ):
        """
        Initialize the matcher.

        Args:
            neo4j_client: Connected DealNeo4jClient
            openai_client: Connected OpenAI client for LLM deduplication
            similarity_threshold: Minimum score to consider (default: 0.70)
            auto_match_threshold: Score for auto-match without LLM (default: 0.90)
        """
        self.neo4j = neo4j_client
        self.openai = openai_client
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else DealConfig.SIMILARITY_THRESHOLD
        )
        self.auto_match_threshold = (
            auto_match_threshold
            if auto_match_threshold is not None
            else DealConfig.AUTO_MATCH_THRESHOLD
        )

    async def find_matches(
        self,
        extracted_deal: ExtractedDeal,
        embedding: list[float],
        tenant_id: UUID,
        account_id: str | None = None,
        max_candidates: int = 10,
    ) -> DealMatchResult:
        """
        Find potential matches for an extracted deal.

        Searches both embedding indexes, deduplicates by opportunity_id,
        then applies graduated thresholds:
        - >= auto_match_threshold: auto-match (skip LLM)
        - >= similarity_threshold: LLM decides
        - below: create new

        Args:
            extracted_deal: The deal extracted from transcript
            embedding: Embedding vector for the extracted deal
            tenant_id: Tenant UUID for filtering
            account_id: Account ID for scoping (optional)
            max_candidates: Maximum candidates to retrieve

        Returns:
            DealMatchResult indicating match_type and matched deal (if any)
        """
        log = logger.bind(
            deal_name=extracted_deal.opportunity_name,
            tenant_id=str(tenant_id),
        )

        # Step 1: Find candidates via dual vector search
        candidates = await self._find_candidates(
            embedding=embedding,
            tenant_id=str(tenant_id),
            account_id=account_id,
            limit=max_candidates,
        )

        # Step 2: No candidates → create new (fast path)
        if not candidates:
            log.info('deal_matcher.no_candidates')
            return DealMatchResult(
                extracted_deal=extracted_deal,
                embedding=embedding,
                match_type='create_new',
                matched_deal=None,
                decision=None,
                candidates_evaluated=0,
            )

        top = candidates[0]
        log.info(
            'deal_matcher.candidates_found',
            count=len(candidates),
            top_score=top.similarity_score,
            top_opportunity_id=top.opportunity_id,
        )

        # Step 3: Auto-match if top candidate is high confidence
        if top.similarity_score >= self.auto_match_threshold:
            log.info(
                'deal_matcher.auto_match',
                opportunity_id=top.opportunity_id,
                score=top.similarity_score,
            )
            return DealMatchResult(
                extracted_deal=extracted_deal,
                embedding=embedding,
                match_type='auto_match',
                matched_deal=top,
                decision=None,
                candidates_evaluated=len(candidates),
                all_candidates=candidates,
            )

        # Step 4: Evaluate borderline candidates via LLM
        for candidate in candidates:
            if candidate.similarity_score < self.similarity_threshold:
                break  # Remaining candidates are below threshold

            decision = await self._deduplicate_llm(
                candidate=candidate,
                extracted_deal=extracted_deal,
            )

            if decision.is_same_deal and decision.recommendation == 'merge':
                log.info(
                    'deal_matcher.llm_match',
                    opportunity_id=candidate.opportunity_id,
                    score=candidate.similarity_score,
                    llm_confidence=decision.confidence,
                )
                return DealMatchResult(
                    extracted_deal=extracted_deal,
                    embedding=embedding,
                    match_type='llm_match',
                    matched_deal=candidate,
                    decision=decision,
                    candidates_evaluated=len(candidates),
                    all_candidates=candidates,
                )

            log.info(
                'deal_matcher.llm_rejected',
                opportunity_id=candidate.opportunity_id,
                score=candidate.similarity_score,
                reasoning=decision.reasoning,
            )

        # Step 5: No match survived → create new
        log.info('deal_matcher.create_new', candidates_checked=len(candidates))
        return DealMatchResult(
            extracted_deal=extracted_deal,
            embedding=embedding,
            match_type='create_new',
            matched_deal=None,
            decision=None,
            candidates_evaluated=len(candidates),
            all_candidates=candidates,
        )

    async def _find_candidates(
        self,
        embedding: list[float],
        tenant_id: str,
        account_id: str | None,
        limit: int,
    ) -> list[DealMatchCandidate]:
        """
        Find candidate matches using dual embedding search.

        Uses DealNeo4jClient.search_deals_both_embeddings which:
        - Searches both deal_embedding_idx and deal_embedding_current_idx
        - Deduplicates by opportunity_id (keeps higher score)
        - Returns results sorted by score DESC

        Args:
            embedding: Query embedding vector
            tenant_id: Tenant ID for filtering
            account_id: Account ID for filtering (optional)
            limit: Maximum results

        Returns:
            List of DealMatchCandidate, sorted by score descending
        """
        raw_results = await self.neo4j.search_deals_both_embeddings(
            embedding=embedding,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=self.similarity_threshold,
        )

        candidates = []
        for result in raw_results:
            node = result['node']
            opp_id = node.get('opportunity_id', '')
            candidates.append(
                DealMatchCandidate(
                    opportunity_id=opp_id,
                    node_properties=node,
                    similarity_score=result['score'],
                )
            )

        return candidates

    async def _deduplicate_llm(
        self,
        candidate: DealMatchCandidate,
        extracted_deal: ExtractedDeal,
    ) -> DealDeduplicationDecision:
        """
        Use LLM to decide if an extracted deal matches a candidate.

        Only called for borderline candidates (similarity_threshold <= score < auto_match_threshold).

        Args:
            candidate: Existing deal candidate from vector search
            extracted_deal: Newly extracted deal from transcript

        Returns:
            DealDeduplicationDecision from LLM
        """
        messages = build_deal_deduplication_prompt(
            existing_deal=candidate.node_properties,
            extracted_deal=extracted_deal,
            similarity_score=candidate.similarity_score,
        )

        decision = await self.openai.chat_completion_structured(
            messages=messages,
            response_model=DealDeduplicationDecision,
        )

        return decision
