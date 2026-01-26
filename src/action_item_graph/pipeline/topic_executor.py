"""
Topic execution for creating topics and linking action items.

TopicExecutor handles the persistence operations for topic assignment:
1. Create new Topic nodes with initial summary
2. Link action items to existing topics
3. Update topic summaries when new items are linked
4. Manage version history for topic changes
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from ..clients.openai_client import OpenAIClient
from ..logging import get_logger
from ..models.topic import Topic
from ..prompts.topic_prompts import (
    TopicSummary,
    build_topic_summary_create_prompt,
    build_topic_summary_update_prompt,
)
from ..repository import ActionItemRepository
from .topic_resolver import TopicDecision, TopicResolutionResult

logger = get_logger(__name__)


@dataclass
class TopicExecutionResult:
    """Result of executing a topic resolution decision."""

    action_item_id: str
    topic_id: str
    topic_name: str
    action: str  # 'created', 'linked', 'linked_and_updated'
    was_new: bool
    version_created: bool = False
    summary_updated: bool = False
    embedding_updated: bool = False
    details: dict[str, Any] = field(default_factory=dict)


class TopicExecutor:
    """
    Executes topic resolution decisions.

    Handles:
    - Creating new Topic nodes with LLM-generated summaries
    - Linking action items to existing topics
    - Updating topic summaries when scope evolves
    - Creating version snapshots for audit trails
    """

    def __init__(
        self,
        repository: ActionItemRepository,
        openai_client: OpenAIClient,
        update_summary_on_link: bool = True,
    ):
        """
        Initialize the topic executor.

        Args:
            repository: ActionItemRepository for graph operations
            openai_client: OpenAI client for summary generation
            update_summary_on_link: Whether to update topic summary when linking new items
        """
        self.repository = repository
        self.openai = openai_client
        self.update_summary_on_link = update_summary_on_link

    async def execute_resolution(
        self,
        resolution: TopicResolutionResult,
        tenant_id: UUID,
        account_id: str,
        action_item_text: str,
        owner: str,
    ) -> TopicExecutionResult:
        """
        Execute a topic resolution decision.

        Args:
            resolution: TopicResolutionResult from TopicResolver
            tenant_id: Tenant UUID
            account_id: Account identifier
            action_item_text: Text of the action item being linked
            owner: Owner of the action item

        Returns:
            TopicExecutionResult with execution details
        """
        if resolution.decision == TopicDecision.CREATE_NEW:
            return await self._create_topic(
                resolution=resolution,
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text=action_item_text,
                owner=owner,
            )
        else:
            return await self._link_to_existing(
                resolution=resolution,
                tenant_id=tenant_id,
                action_item_text=action_item_text,
                owner=owner,
            )

    async def execute_batch(
        self,
        resolutions: list[tuple[TopicResolutionResult, str, str]],
        tenant_id: UUID,
        account_id: str,
    ) -> list[TopicExecutionResult]:
        """
        Execute multiple topic resolutions.

        Args:
            resolutions: List of (resolution, action_item_text, owner) tuples
            tenant_id: Tenant UUID
            account_id: Account identifier

        Returns:
            List of TopicExecutionResult objects
        """
        results = []
        for resolution, action_item_text, owner in resolutions:
            result = await self.execute_resolution(
                resolution=resolution,
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text=action_item_text,
                owner=owner,
            )
            results.append(result)
        return results

    async def _create_topic(
        self,
        resolution: TopicResolutionResult,
        tenant_id: UUID,
        account_id: str,
        action_item_text: str,
        owner: str,
    ) -> TopicExecutionResult:
        """
        Create a new Topic node and link the action item.

        Args:
            resolution: TopicResolutionResult with CREATE_NEW decision
            tenant_id: Tenant UUID
            account_id: Account identifier
            action_item_text: Text of the triggering action item
            owner: Owner of the action item

        Returns:
            TopicExecutionResult for the created topic
        """
        extracted = resolution.extracted_topic

        # Generate initial summary via LLM
        summary_result = await self._generate_initial_summary(
            topic_name=extracted.name,
            action_item_text=action_item_text,
            action_item_summary=resolution.action_item_summary,
            owner=owner,
            topic_context=extracted.context,
        )

        # Create the Topic model
        topic = Topic(
            id=uuid4(),
            tenant_id=tenant_id,
            account_id=account_id,
            name=extracted.name,
            canonical_name=Topic.canonicalize_name(extracted.name),
            summary=summary_result.summary,
            embedding=resolution.embedding,
            embedding_current=resolution.embedding,  # Same initially
            action_item_count=1,  # First action item
            created_from_action_item_id=UUID(resolution.action_item_id),
        )

        # Persist to graph
        await self.repository.create_topic(topic)

        # Link action item to topic
        await self.repository.link_action_item_to_topic(
            action_item_id=resolution.action_item_id,
            topic_id=str(topic.id),
            tenant_id=tenant_id,
            confidence=resolution.confidence,
            method='extracted',
        )

        # Create initial version
        await self.repository.create_topic_version(
            topic_id=str(topic.id),
            tenant_id=tenant_id,
            version_number=1,
            name=topic.name,
            summary=topic.summary,
            embedding_snapshot=topic.embedding,
            changed_by_action_item_id=UUID(resolution.action_item_id),
        )

        logger.info(
            "topic_created",
            topic_id=str(topic.id),
            topic_name=topic.name,
            action_item_id=resolution.action_item_id,
        )

        return TopicExecutionResult(
            action_item_id=resolution.action_item_id,
            topic_id=str(topic.id),
            topic_name=topic.name,
            action='created',
            was_new=True,
            version_created=True,
            details={
                'summary': topic.summary,
                'canonical_name': topic.canonical_name,
            },
        )

    async def _link_to_existing(
        self,
        resolution: TopicResolutionResult,
        tenant_id: UUID,
        action_item_text: str,
        owner: str,
    ) -> TopicExecutionResult:
        """
        Link an action item to an existing Topic, optionally updating summary.

        Args:
            resolution: TopicResolutionResult with LINK_EXISTING decision
            tenant_id: Tenant UUID
            action_item_text: Text of the action item being linked
            owner: Owner of the action item

        Returns:
            TopicExecutionResult for the linked topic
        """
        # topic_id is guaranteed to be set for LINK_EXISTING decisions
        topic_id = resolution.topic_id
        if topic_id is None:
            raise ValueError("topic_id cannot be None for LINK_EXISTING decision")

        best = resolution.best_candidate

        # Link action item to topic
        await self.repository.link_action_item_to_topic(
            action_item_id=resolution.action_item_id,
            topic_id=topic_id,
            tenant_id=tenant_id,
            confidence=resolution.confidence,
            method='resolved' if resolution.llm_decision else 'extracted',
        )

        # Increment action item count
        new_count = await self.repository.increment_topic_action_count(
            topic_id=topic_id,
            tenant_id=tenant_id,
        )

        result = TopicExecutionResult(
            action_item_id=resolution.action_item_id,
            topic_id=topic_id,
            topic_name=best.name if best else 'Unknown',
            action='linked',
            was_new=False,
            details={
                'similarity': best.similarity if best else 0,
                'new_action_item_count': new_count,
            },
        )

        # Optionally update summary
        if self.update_summary_on_link and best:
            updated = await self._update_topic_summary(
                topic_id=topic_id,
                tenant_id=tenant_id,
                current_summary=best.summary,
                current_name=best.name,
                new_count=new_count,
                action_item_text=action_item_text,
                action_item_summary=resolution.action_item_summary,
                owner=owner,
                changed_by_action_item_id=resolution.action_item_id,
            )

            if updated:
                result.action = 'linked_and_updated'
                result.summary_updated = updated.get('summary_updated', False)
                result.embedding_updated = updated.get('embedding_updated', False)
                result.version_created = updated.get('version_created', False)
                result.details.update(updated)

        logger.info(
            "topic_linked",
            topic_id=topic_id,
            topic_name=best.name if best else 'Unknown',
            action_item_id=resolution.action_item_id,
            action=result.action,
        )

        return result

    async def _generate_initial_summary(
        self,
        topic_name: str,
        action_item_text: str,
        action_item_summary: str,
        owner: str,
        topic_context: str,
    ) -> TopicSummary:
        """
        Generate the initial summary for a new topic.

        Args:
            topic_name: Name of the topic
            action_item_text: Text of the first action item
            action_item_summary: Summary of the action item
            owner: Owner of the action item
            topic_context: Context from extracted topic

        Returns:
            TopicSummary with generated summary
        """
        messages = build_topic_summary_create_prompt(
            topic_name=topic_name,
            action_item_text=action_item_text,
            action_item_summary=action_item_summary,
            owner=owner,
            topic_context=topic_context,
        )

        return await self.openai.chat_completion_structured(
            messages=messages,
            response_model=TopicSummary,
            temperature=0.0,
        )

    async def _update_topic_summary(
        self,
        topic_id: str,
        tenant_id: UUID,
        current_summary: str,
        current_name: str,
        new_count: int,
        action_item_text: str,
        action_item_summary: str,
        owner: str,
        changed_by_action_item_id: str,
    ) -> dict[str, Any] | None:
        """
        Update a topic's summary when a new action item is linked.

        Args:
            topic_id: Topic UUID string
            tenant_id: Tenant UUID
            current_summary: Current topic summary
            current_name: Current topic name
            new_count: New total action item count
            action_item_text: Text of the newly linked action item
            action_item_summary: Summary of the action item
            owner: Owner of the action item
            changed_by_action_item_id: Action item that triggered update

        Returns:
            Dict with update details, or None if no update needed
        """
        # Generate updated summary
        messages = build_topic_summary_update_prompt(
            topic_name=current_name,
            current_summary=current_summary,
            total_count=new_count,
            action_item_text=action_item_text,
            action_item_summary=action_item_summary,
            owner=owner,
        )

        summary_result = await self.openai.chat_completion_structured(
            messages=messages,
            response_model=TopicSummary,
            temperature=0.0,
        )

        # Check if summary actually changed
        if summary_result.summary == current_summary:
            return None

        updates = {
            'summary': summary_result.summary,
            'updated_at': datetime.now().isoformat(),
        }

        # Update embedding if significant change
        new_embedding = None
        if summary_result.should_update_embedding:
            topic_text = f"{current_name}: {summary_result.summary}"
            new_embedding = await self.openai.create_embedding(topic_text)
            updates['embedding_current'] = new_embedding

        # Get current version for incrementing
        topic = await self.repository.get_topic(topic_id, tenant_id)
        current_version = topic.get('version', 1) if topic else 1

        # Create version snapshot before update
        await self.repository.create_topic_version(
            topic_id=topic_id,
            tenant_id=tenant_id,
            version_number=current_version,
            name=current_name,
            summary=current_summary,
            embedding_snapshot=new_embedding,  # Snapshot new embedding if updated
            changed_by_action_item_id=UUID(changed_by_action_item_id),
        )

        # Update the topic
        updates['version'] = current_version + 1
        await self.repository.update_topic(
            topic_id=topic_id,
            tenant_id=tenant_id,
            updates=updates,
        )

        return {
            'summary_updated': True,
            'embedding_updated': summary_result.should_update_embedding,
            'version_created': True,
            'new_summary': summary_result.summary,
            'previous_summary': current_summary,
        }
