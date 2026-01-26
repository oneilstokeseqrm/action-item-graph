"""
Action item merger service.

Executes match decisions from the matcher:
- merge: Synthesize content via LLM, update existing item
- update_status: Update status field only (no LLM needed)
- create_new: Create new ActionItem node
- link_related: Create new but link via relationship
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from ..clients.neo4j_client import Neo4jClient
from ..clients.openai_client import OpenAIClient
from ..models.action_item import ActionItem, ActionItemStatus
from ..models.entities import Interaction
from ..prompts.extract_action_items import ExtractedActionItem
from ..prompts.merge_action_items import MergedActionItem, build_merge_prompt
from ..repository import ActionItemRepository
from .matcher import MatchResult


# Map implied status strings to ActionItemStatus enum
STATUS_MAP = {
    'completed': ActionItemStatus.COMPLETED,
    'in_progress': ActionItemStatus.IN_PROGRESS,
    'cancelled': ActionItemStatus.CANCELLED,
    'deferred': ActionItemStatus.DEFERRED,
}


@dataclass
class MergeResult:
    """Result of executing a merge decision."""

    action_item_id: str
    action: str  # 'created', 'merged', 'status_updated', 'linked'
    was_new: bool
    version_created: bool
    linked_interaction_id: str | None
    details: dict[str, Any]


class ActionItemMerger:
    """
    Executes match decisions to create or update ActionItems in the graph.

    Responsibilities:
    - Execute merge/create/update operations based on match results
    - Create version snapshots before updates
    - Manage relationships (EXTRACTED_FROM, OWNED_BY, RELATED_TO)
    - Handle embedding updates when content changes significantly
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        openai_client: OpenAIClient,
    ):
        """
        Initialize the merger.

        Args:
            neo4j_client: Connected Neo4j client
            openai_client: Connected OpenAI client
        """
        self.neo4j = neo4j_client
        self.openai = openai_client
        self.repository = ActionItemRepository(neo4j_client)

    async def execute_decision(
        self,
        match_result: MatchResult,
        interaction: Interaction,
        action_item: ActionItem,
    ) -> MergeResult:
        """
        Execute the merge decision for a matched/unmatched extraction.

        Args:
            match_result: Result from ActionItemMatcher.find_matches()
            interaction: The Interaction this extraction came from
            action_item: The ActionItem model (with embeddings) from extraction

        Returns:
            MergeResult describing what was done
        """
        if match_result.best_match is None:
            # No match found - create new ActionItem
            return await self._create_new(
                action_item=action_item,
                interaction=interaction,
            )

        candidate, decision = match_result.best_match

        if decision.merge_recommendation == 'update_status':
            # Status-only update (no LLM synthesis needed)
            return await self._update_status(
                existing_id=candidate.action_item_id,
                extraction=match_result.extracted_item,
                interaction=interaction,
                action_item=action_item,
            )

        elif decision.merge_recommendation == 'merge':
            # Full merge with LLM synthesis
            return await self._merge_items(
                existing_id=candidate.action_item_id,
                existing_props=candidate.node_properties,
                extraction=match_result.extracted_item,
                interaction=interaction,
                action_item=action_item,
            )

        elif decision.merge_recommendation == 'link_related':
            # Create new but link as related
            return await self._create_and_link(
                related_to_id=candidate.action_item_id,
                action_item=action_item,
                interaction=interaction,
            )

        else:
            # Fallback: treat as create_new
            return await self._create_new(
                action_item=action_item,
                interaction=interaction,
            )

    async def execute_batch(
        self,
        match_results: list[MatchResult],
        interaction: Interaction,
        action_items: list[ActionItem],
    ) -> list[MergeResult]:
        """
        Execute decisions for a batch of match results.

        Args:
            match_results: Results from ActionItemMatcher.find_matches()
            interaction: The Interaction these extractions came from
            action_items: ActionItem models (with embeddings) from extraction

        Returns:
            List of MergeResult objects
        """
        results = []

        # Build a mapping from extraction to action_item by matching text
        # (since both lists should be in the same order)
        for match_result, action_item in zip(match_results, action_items):
            result = await self.execute_decision(
                match_result=match_result,
                interaction=interaction,
                action_item=action_item,
            )
            results.append(result)

        return results

    async def _create_new(
        self,
        action_item: ActionItem,
        interaction: Interaction,
    ) -> MergeResult:
        """
        Create a new ActionItem node.

        Args:
            action_item: ActionItem model to create
            interaction: Source interaction

        Returns:
            MergeResult for the creation
        """
        # Create the ActionItem node
        created = await self.repository.create_action_item(action_item)

        # Link to interaction
        await self.repository.link_to_interaction(
            action_item_id=str(action_item.id),
            interaction_id=str(interaction.id),
            tenant_id=action_item.tenant_id,
        )

        # Resolve and link owner
        owner = await self.repository.resolve_or_create_owner(
            tenant_id=action_item.tenant_id,
            owner_name=action_item.owner,
        )
        await self.repository.link_to_owner(
            action_item_id=str(action_item.id),
            owner_id=owner['id'],
            tenant_id=action_item.tenant_id,
        )

        return MergeResult(
            action_item_id=str(action_item.id),
            action='created',
            was_new=True,
            version_created=False,
            linked_interaction_id=str(interaction.id),
            details={
                'owner_id': owner['id'],
                'owner_name': owner.get('canonical_name', action_item.owner),
            },
        )

    async def _update_status(
        self,
        existing_id: str,
        extraction: ExtractedActionItem,
        interaction: Interaction,
        action_item: ActionItem,
    ) -> MergeResult:
        """
        Update only the status of an existing ActionItem.

        This is used when the new extraction is purely a status update
        (e.g., "I finished that" or "Done!").

        Args:
            existing_id: ID of existing ActionItem to update
            extraction: The extracted status update
            interaction: Source interaction
            action_item: ActionItem model (not persisted, used for tenant_id)

        Returns:
            MergeResult for the update
        """
        tenant_id = action_item.tenant_id

        # Determine new status
        new_status = ActionItemStatus.OPEN
        if extraction.implied_status:
            new_status = STATUS_MAP.get(extraction.implied_status, ActionItemStatus.OPEN)

        # Create version snapshot before update
        await self.repository.create_version_snapshot(
            action_item_id=existing_id,
            tenant_id=tenant_id,
            change_summary=f"Status updated to {new_status.value} based on: {extraction.summary}",
            source_interaction_id=interaction.id,
        )

        # Update status
        updated = await self.repository.update_action_item_status(
            action_item_id=existing_id,
            tenant_id=tenant_id,
            status=new_status,
        )

        # Link to interaction (this extraction came from this interaction)
        await self.repository.link_to_interaction(
            action_item_id=existing_id,
            interaction_id=str(interaction.id),
            tenant_id=tenant_id,
        )

        return MergeResult(
            action_item_id=existing_id,
            action='status_updated',
            was_new=False,
            version_created=True,
            linked_interaction_id=str(interaction.id),
            details={
                'previous_status': updated.get('status'),
                'new_status': new_status.value,
            },
        )

    async def _merge_items(
        self,
        existing_id: str,
        existing_props: dict[str, Any],
        extraction: ExtractedActionItem,
        interaction: Interaction,
        action_item: ActionItem,
    ) -> MergeResult:
        """
        Merge a new extraction into an existing ActionItem using LLM synthesis.

        Args:
            existing_id: ID of existing ActionItem to update
            existing_props: Current properties of the existing item
            extraction: The new extraction to merge in
            interaction: Source interaction
            action_item: ActionItem model (has embeddings)

        Returns:
            MergeResult for the merge
        """
        tenant_id = action_item.tenant_id

        # Build merge prompt and get synthesized result
        messages = build_merge_prompt(
            existing_text=existing_props.get('action_item_text', ''),
            existing_summary=existing_props.get('summary', ''),
            existing_owner=existing_props.get('owner', ''),
            existing_status=existing_props.get('status', 'open'),
            existing_due_date=existing_props.get('due_date'),
            existing_context=existing_props.get('conversation_context', ''),
            existing_created=existing_props.get('created_at', ''),
            new_text=extraction.action_item_text,
            new_summary=extraction.summary,
            new_owner=extraction.owner,
            new_context=extraction.conversation_context,
            new_is_status_update=extraction.is_status_update,
            new_implied_status=extraction.implied_status,
            new_due_date=extraction.due_date_text,
            merge_recommendation='merge',
        )

        merged = await self.openai.chat_completion_structured(
            messages=messages,
            response_model=MergedActionItem,
        )

        # Create version snapshot before update
        await self.repository.create_version_snapshot(
            action_item_id=existing_id,
            tenant_id=tenant_id,
            change_summary=merged.evolution_summary,
            source_interaction_id=interaction.id,
        )

        # Build updates dict
        updates: dict[str, Any] = {
            'action_item_text': merged.action_item_text,
            'summary': merged.summary,
            'owner': merged.owner,
            'evolution_summary': merged.evolution_summary,
        }

        # Update status if implied
        if merged.implied_status:
            updates['status'] = STATUS_MAP.get(
                merged.implied_status,
                ActionItemStatus.OPEN,
            ).value

        # Update embedding_current if content changed significantly
        if merged.should_update_embedding:
            new_embedding = await self.openai.create_embedding(merged.action_item_text)
            updates['embedding_current'] = new_embedding

        # Apply updates
        updated = await self.repository.update_action_item(
            action_item_id=existing_id,
            tenant_id=tenant_id,
            updates=updates,
        )

        # Link to interaction
        await self.repository.link_to_interaction(
            action_item_id=existing_id,
            interaction_id=str(interaction.id),
            tenant_id=tenant_id,
        )

        # Update owner link if changed
        if merged.owner != existing_props.get('owner'):
            owner = await self.repository.resolve_or_create_owner(
                tenant_id=tenant_id,
                owner_name=merged.owner,
            )
            await self.repository.link_to_owner(
                action_item_id=existing_id,
                owner_id=owner['id'],
                tenant_id=tenant_id,
            )

        return MergeResult(
            action_item_id=existing_id,
            action='merged',
            was_new=False,
            version_created=True,
            linked_interaction_id=str(interaction.id),
            details={
                'evolution_summary': merged.evolution_summary,
                'embedding_updated': merged.should_update_embedding,
                'status_changed': merged.implied_status is not None,
            },
        )

    async def _create_and_link(
        self,
        related_to_id: str,
        action_item: ActionItem,
        interaction: Interaction,
    ) -> MergeResult:
        """
        Create a new ActionItem and link it to an existing related item.

        Used when items are distinct but clearly related (e.g., "send proposal"
        and "review proposal" are different tasks but related).

        Args:
            related_to_id: ID of the related ActionItem
            action_item: ActionItem model to create
            interaction: Source interaction

        Returns:
            MergeResult for the creation with link
        """
        # First create the new item
        create_result = await self._create_new(
            action_item=action_item,
            interaction=interaction,
        )

        # Then link to the related item
        await self.repository.link_related_items(
            action_item_id=str(action_item.id),
            related_item_id=related_to_id,
            tenant_id=action_item.tenant_id,
            relationship_type='RELATED_TO',
        )

        return MergeResult(
            action_item_id=str(action_item.id),
            action='linked',
            was_new=True,
            version_created=False,
            linked_interaction_id=str(interaction.id),
            details={
                **create_result.details,
                'related_to': related_to_id,
            },
        )
