"""
Within-batch action item consolidator.

Clusters near-duplicate items extracted from the same transcript using
embedding cosine similarity, then asks the LLM to merge clusters into
single consolidated items. Runs AFTER extraction and BEFORE verification.
"""

from __future__ import annotations

import math

from ..clients.openai_client import OpenAIClient
from ..logging import get_logger
from ..models.action_item import ActionItem
from ..prompts.consolidation_prompts import (
    CONSOLIDATION_SYSTEM_PROMPT,
    CONSOLIDATION_USER_PROMPT_TEMPLATE,
    ConsolidationResult,
)
from ..prompts.extract_action_items import ExtractedActionItem
from .extractor import ExtractionOutput

logger = get_logger(__name__)

# Intra-batch similarity threshold — higher than cross-interaction (0.65)
# because items from the same transcript are more likely to share vocabulary.
INTRA_BATCH_SIMILARITY = 0.80


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cluster_items(
    embeddings: list[list[float]],
    threshold: float = INTRA_BATCH_SIMILARITY,
) -> list[list[int]]:
    """
    Cluster item indices by pairwise cosine similarity.

    Uses complete-linkage clustering: an item joins a cluster only if it meets
    the similarity threshold with ALL existing members. This prevents chaining
    (where A~B and B~C would pull A and C together even if A≁C).

    Returns:
        List of clusters, where each cluster is a list of indices.
        Singletons are included (clusters of size 1).
    """
    n = len(embeddings)
    if n == 0:
        return []

    # Pre-compute pairwise similarity matrix
    sim_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # Greedy complete-linkage: each item joins the first cluster where it
    # meets the threshold with ALL existing members, or starts a new cluster.
    clusters: list[list[int]] = []
    for i in range(n):
        placed = False
        for cluster in clusters:
            if all(sim_matrix[i][j] >= threshold for j in cluster):
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    return clusters


class ActionItemConsolidator:
    """
    Consolidates near-duplicate action items within a single extraction batch.

    Algorithm:
    1. Compute pairwise cosine similarity from pre-existing embeddings
    2. Cluster items above INTRA_BATCH_SIMILARITY threshold
    3. For clusters of 2+ items, ask the LLM to select the best representative
       and merge context
    4. Return a filtered ExtractionOutput with consolidated items
    """

    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client

    async def consolidate(
        self,
        extraction: ExtractionOutput,
    ) -> tuple[ExtractionOutput, int]:
        """
        Consolidate near-duplicate items in the extraction batch.

        Args:
            extraction: The raw ExtractionOutput from the extractor

        Returns:
            Tuple of (consolidated ExtractionOutput, number of items removed)
        """
        if extraction.count <= 1:
            return extraction, 0

        # Collect embeddings (skip items without embeddings)
        embeddings = []
        valid_indices = []
        for i, ai in enumerate(extraction.action_items):
            if ai.embedding is not None:
                embeddings.append(ai.embedding)
                valid_indices.append(i)

        if len(embeddings) <= 1:
            return extraction, 0

        # Cluster by similarity
        clusters = _cluster_items(embeddings)

        # Separate multi-item clusters from singletons
        merge_clusters = [c for c in clusters if len(c) > 1]

        if not merge_clusters:
            logger.info(
                'consolidation_no_duplicates',
                item_count=extraction.count,
            )
            return extraction, 0

        # Map cluster indices back to original extraction indices
        original_merge_clusters = [
            [valid_indices[i] for i in cluster] for cluster in merge_clusters
        ]

        # Ask LLM to merge each multi-item cluster
        keep_indices = set()
        for cluster in clusters:
            if len(cluster) == 1:
                keep_indices.add(valid_indices[cluster[0]])

        # Also keep items that had no embedding
        no_embedding_indices = set(range(extraction.count)) - set(valid_indices)
        keep_indices.update(no_embedding_indices)

        merged_items: list[tuple[int, ActionItem, ExtractedActionItem]] = []

        for cluster_indices in original_merge_clusters:
            primary_idx, merged_ai, merged_raw = await self._merge_cluster(
                cluster_indices=cluster_indices,
                action_items=extraction.action_items,
                raw_extractions=extraction.raw_extractions,
            )
            merged_items.append((primary_idx, merged_ai, merged_raw))

        # Build consolidated output
        new_action_items = []
        new_raw_extractions = []

        # Add singletons (in original order)
        for idx in sorted(keep_indices):
            new_action_items.append(extraction.action_items[idx])
            new_raw_extractions.append(extraction.raw_extractions[idx])

        # Add merged items (in order of their primary index)
        for primary_idx, merged_ai, merged_raw in sorted(merged_items, key=lambda x: x[0]):
            new_action_items.append(merged_ai)
            new_raw_extractions.append(merged_raw)

        items_removed = extraction.count - len(new_action_items)

        logger.info(
            'consolidation_complete',
            original_count=extraction.count,
            consolidated_count=len(new_action_items),
            clusters_merged=len(merge_clusters),
            items_removed=items_removed,
        )

        consolidated = ExtractionOutput(
            interaction=extraction.interaction,
            action_items=new_action_items,
            raw_extractions=new_raw_extractions,
            extraction_notes=extraction.extraction_notes,
        )

        return consolidated, items_removed

    async def _merge_cluster(
        self,
        cluster_indices: list[int],
        action_items: list[ActionItem],
        raw_extractions: list[ExtractedActionItem],
    ) -> tuple[int, ActionItem, ExtractedActionItem]:
        """
        Ask the LLM to merge a cluster of similar items.

        Returns:
            Tuple of (primary_index, merged_ActionItem, merged_ExtractedActionItem)
        """
        # Build items text for the LLM
        items_text_parts = []
        for i, idx in enumerate(cluster_indices):
            ai = action_items[idx]
            raw = raw_extractions[idx]
            items_text_parts.append(
                f'[{i}] Summary: {ai.summary}\n'
                f'    Owner: {ai.owner} ({ai.owner_type})\n'
                f'    Text: {ai.action_item_text}\n'
                f'    Context: {ai.conversation_context}\n'
                f'    Due: {raw.due_date_text or "Not specified"}'
            )

        items_text = '\n\n'.join(items_text_parts)

        messages = [
            {'role': 'system', 'content': CONSOLIDATION_SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': CONSOLIDATION_USER_PROMPT_TEMPLATE.format(items_text=items_text),
            },
        ]

        try:
            result = await self.openai_client.chat_completion_structured(
                messages=messages,
                response_model=ConsolidationResult,
            )

            if result.groups:
                # Use the first group's decision (there should be exactly one for this cluster)
                decision = result.groups[0]
                # Map the LLM's local index back to the original index
                if 0 <= decision.primary_index < len(cluster_indices):
                    primary_original_idx = cluster_indices[decision.primary_index]
                else:
                    primary_original_idx = cluster_indices[0]

                # Create merged item based on the primary
                primary_ai = action_items[primary_original_idx]
                primary_raw = raw_extractions[primary_original_idx]

                # Apply LLM's merged text
                merged_ai = primary_ai.model_copy()
                merged_ai.summary = decision.merged_summary
                merged_ai.conversation_context = decision.merged_context

                merged_raw = primary_raw.model_copy()
                merged_raw.summary = decision.merged_summary
                merged_raw.conversation_context = decision.merged_context

                return primary_original_idx, merged_ai, merged_raw
        except Exception:
            logger.exception('consolidation_merge_failed', cluster_size=len(cluster_indices))

        # Fallback: keep the first item in the cluster
        fallback_idx = cluster_indices[0]
        return fallback_idx, action_items[fallback_idx], raw_extractions[fallback_idx]
