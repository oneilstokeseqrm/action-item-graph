"""
Action item matching and deduplication service.

Uses vector similarity search to find candidate matches, then LLM-based
deduplication to decide whether items are the same real-world task.

This implements the dual embedding strategy:
- embedding (original): catches semantically similar new items
- embedding_current: catches status updates to evolved items
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from ..clients.neo4j_client import Neo4jClient
from ..clients.openai_client import OpenAIClient
from ..models.action_item import ActionItem
from ..prompts.extract_action_items import (
    DeduplicationDecision,
    ExtractedActionItem,
    build_deduplication_prompt,
)


@dataclass
class MatchCandidate:
    """A candidate match from vector search."""

    action_item_id: str
    node_properties: dict[str, Any]
    similarity_score: float
    matched_via: str  # 'original' or 'current' embedding


@dataclass
class MatchResult:
    """Result of matching a single extracted item against candidates."""

    extracted_item: ExtractedActionItem
    embedding: list[float]
    candidates: list[MatchCandidate]
    decisions: list[tuple[MatchCandidate, DeduplicationDecision]]
    best_match: tuple[MatchCandidate, DeduplicationDecision] | None


class ActionItemMatcher:
    """
    Matches extracted action items against existing items in the graph.

    Pipeline:
    1. Vector search (dual embedding) to find candidates
    2. LLM-based deduplication to decide if items match
    3. Return match decisions for merger to act on
    """

    # Similarity thresholds
    MIN_SIMILARITY_SCORE = 0.65  # Below this, don't consider as candidate
    HIGH_CONFIDENCE_THRESHOLD = 0.85  # Above this, high confidence match

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        openai_client: OpenAIClient,
        min_similarity: float | None = None,
    ):
        """
        Initialize the matcher.

        Args:
            neo4j_client: Connected Neo4j client
            openai_client: Connected OpenAI client
            min_similarity: Override default minimum similarity threshold
        """
        self.neo4j_client = neo4j_client
        self.openai_client = openai_client
        self.min_similarity = min_similarity or self.MIN_SIMILARITY_SCORE

    async def find_matches(
        self,
        extracted_items: list[tuple[ExtractedActionItem, list[float]]],
        tenant_id: UUID,
        account_id: str,
        max_candidates: int = 5,
    ) -> list[MatchResult]:
        """
        Find potential matches for a batch of extracted action items.

        Args:
            extracted_items: List of (ExtractedActionItem, embedding) tuples
            tenant_id: Tenant UUID for filtering
            account_id: Account ID for scoping (required to prevent cross-account bleeding)
            max_candidates: Maximum candidates to consider per item

        Returns:
            List of MatchResult objects, one per extracted item
        """
        results = []

        for extracted, embedding in extracted_items:
            # Find candidates via vector search
            candidates = await self._find_candidates(
                embedding=embedding,
                tenant_id=str(tenant_id),
                account_id=account_id,
                limit=max_candidates,
            )

            # If no candidates, create result with empty matches
            if not candidates:
                results.append(
                    MatchResult(
                        extracted_item=extracted,
                        embedding=embedding,
                        candidates=[],
                        decisions=[],
                        best_match=None,
                    )
                )
                continue

            # Run deduplication on each candidate
            decisions = []
            for candidate in candidates:
                decision = await self._deduplicate(
                    existing=candidate.node_properties,
                    new_extraction=extracted,
                    similarity_score=candidate.similarity_score,
                )
                decisions.append((candidate, decision))

            # Find best match (if any)
            best_match = self._select_best_match(decisions)

            results.append(
                MatchResult(
                    extracted_item=extracted,
                    embedding=embedding,
                    candidates=candidates,
                    decisions=decisions,
                    best_match=best_match,
                )
            )

        return results

    async def _find_candidates(
        self,
        embedding: list[float],
        tenant_id: str,
        account_id: str,
        limit: int,
    ) -> list[MatchCandidate]:
        """
        Find candidate matches using dual embedding search.

        Args:
            embedding: Query embedding
            tenant_id: Tenant ID for filtering
            account_id: Account ID for filtering (required to prevent cross-account bleeding)
            limit: Maximum results

        Returns:
            List of MatchCandidate objects
        """
        # Search both embedding indexes
        raw_results = await self.neo4j_client.search_both_embeddings(
            embedding=embedding,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=self.min_similarity,
        )

        candidates = []
        for result in raw_results:
            candidates.append(
                MatchCandidate(
                    action_item_id=result['node']['id'],
                    node_properties=result['node'],
                    similarity_score=result['score'],
                    # Note: search_both_embeddings combines results, so we can't
                    # distinguish which index returned it. That's fine for matching.
                    matched_via='combined',
                )
            )

        return candidates

    async def _deduplicate(
        self,
        existing: dict[str, Any],
        new_extraction: ExtractedActionItem,
        similarity_score: float,
    ) -> DeduplicationDecision:
        """
        Use LLM to decide if two items are the same task.

        Args:
            existing: Properties of existing ActionItem node
            new_extraction: Newly extracted item
            similarity_score: Cosine similarity between embeddings

        Returns:
            DeduplicationDecision from LLM
        """
        # Build the prompt
        messages = build_deduplication_prompt(
            existing_text=existing.get('action_item_text', ''),
            existing_owner=existing.get('owner', ''),
            existing_summary=existing.get('summary', ''),
            existing_status=existing.get('status', 'open'),
            existing_created=existing.get('created_at', ''),
            new_text=new_extraction.action_item_text,
            new_owner=new_extraction.owner,
            new_summary=new_extraction.summary,
            new_is_status_update=new_extraction.is_status_update,
            new_context=new_extraction.conversation_context,
            similarity_score=similarity_score,
        )

        # Get structured decision from LLM
        decision = await self.openai_client.chat_completion_structured(
            messages=messages,
            response_model=DeduplicationDecision,
        )

        return decision

    def _select_best_match(
        self,
        decisions: list[tuple[MatchCandidate, DeduplicationDecision]],
    ) -> tuple[MatchCandidate, DeduplicationDecision] | None:
        """
        Select the best match from deduplication decisions.

        Prioritizes:
        1. is_same_item=True with highest confidence
        2. merge or update_status recommendations
        3. Higher similarity scores as tiebreaker

        Args:
            decisions: List of (candidate, decision) tuples

        Returns:
            Best (candidate, decision) tuple or None if no good match
        """
        # Filter to same-item decisions
        same_items = [
            (c, d)
            for c, d in decisions
            if d.is_same_item
            and d.merge_recommendation in ('merge', 'update_status')
        ]

        if not same_items:
            return None

        # Sort by confidence (desc), then similarity (desc)
        same_items.sort(
            key=lambda x: (x[1].confidence, x[0].similarity_score),
            reverse=True,
        )

        return same_items[0]


@dataclass
class BatchMatchResult:
    """Result of matching a batch of items."""

    # Items that matched existing action items
    matched: list[MatchResult]
    # Items with no matches (should create new)
    unmatched: list[MatchResult]
    # Statistics
    total_items: int
    total_matched: int
    total_unmatched: int


async def match_batch(
    matcher: ActionItemMatcher,
    action_items: list[ActionItem],
    raw_extractions: list[ExtractedActionItem],
    tenant_id: UUID,
    account_id: str,
) -> BatchMatchResult:
    """
    Convenience function to match a batch of ActionItems.

    Args:
        matcher: Configured ActionItemMatcher
        action_items: ActionItem objects (have embeddings)
        raw_extractions: Corresponding ExtractedActionItem objects
        tenant_id: Tenant UUID
        account_id: Account ID (required to prevent cross-account bleeding)

    Returns:
        BatchMatchResult with matched and unmatched items
    """
    # Pair items with their embeddings
    items_with_embeddings = [
        (raw, ai.embedding)
        for ai, raw in zip(action_items, raw_extractions)
        if ai.embedding is not None
    ]

    # Run matching
    results = await matcher.find_matches(
        extracted_items=items_with_embeddings,
        tenant_id=tenant_id,
        account_id=account_id,
    )

    # Separate matched from unmatched
    matched = [r for r in results if r.best_match is not None]
    unmatched = [r for r in results if r.best_match is None]

    return BatchMatchResult(
        matched=matched,
        unmatched=unmatched,
        total_items=len(results),
        total_matched=len(matched),
        total_unmatched=len(unmatched),
    )
