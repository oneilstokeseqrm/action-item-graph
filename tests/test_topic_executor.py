"""
Tests for TopicExecutor - topic creation and linking execution.
"""

import uuid
from datetime import datetime

import pytest

# Add project paths for imports
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from action_item_graph.pipeline.topic_executor import (
    TopicExecutor,
    TopicExecutionResult,
)
from action_item_graph.pipeline.topic_resolver import (
    TopicDecision,
    TopicResolutionResult,
    TopicCandidate,
)
from action_item_graph.repository import ActionItemRepository
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.prompts.extract_action_items import ExtractedTopic


class TestTopicExecutorInit:
    """Test TopicExecutor initialization."""

    @pytest.mark.asyncio
    async def test_initialization(self, openai_api_key, neo4j_credentials):
        """Test executor initializes correctly."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            repository = ActionItemRepository(neo4j)
            executor = TopicExecutor(repository, openai)

            assert executor.repository == repository
            assert executor.openai == openai
            assert executor.update_summary_on_link is True

        finally:
            await neo4j.close()
            await openai.close()


class TestTopicCreation:
    """Test topic creation via executor."""

    @pytest.mark.asyncio
    async def test_create_topic_from_resolution(
        self, openai_api_key, neo4j_credentials, sample_tenant_id
    ):
        """Test creating a new topic from a CREATE_NEW resolution."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repository = ActionItemRepository(neo4j)
            executor = TopicExecutor(repository, openai)

            tenant_id = uuid.UUID(sample_tenant_id)
            account_id = f"test_account_{uuid.uuid4().hex[:8]}"
            action_item_id = str(uuid.uuid4())

            # Ensure account exists
            await repository.ensure_account(tenant_id, account_id, "Test Account")

            # Create a mock action item first
            from action_item_graph.models.action_item import ActionItem, ActionItemStatus

            action_item = ActionItem(
                id=uuid.UUID(action_item_id),
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text="Hire 3 SDRs by March",
                summary="Hire SDRs for Q1",
                owner="Sarah",
                status=ActionItemStatus.OPEN,
            )
            await repository.create_action_item(action_item)

            # Create resolution for new topic
            topic = ExtractedTopic(
                name=f"Q1 Sales Expansion {uuid.uuid4().hex[:6]}",
                context="Strategic initiative to grow the sales team in Q1.",
            )

            embedding = await openai.create_embedding(f"{topic.name}: {topic.context}")

            resolution = TopicResolutionResult(
                action_item_id=action_item_id,
                action_item_summary="Hire SDRs for Q1",
                extracted_topic=topic,
                decision=TopicDecision.CREATE_NEW,
                topic_id=None,
                confidence=1.0,
                embedding=embedding,
            )

            # Execute the resolution
            result = await executor.execute_resolution(
                resolution=resolution,
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text="Hire 3 SDRs by March",
                owner="Sarah",
            )

            # Verify result
            assert result.was_new is True
            assert result.action == 'created'
            assert result.topic_id is not None
            assert result.topic_name == topic.name
            assert result.version_created is True

            # Verify topic exists in graph
            created_topic = await repository.get_topic(result.topic_id, tenant_id)
            assert created_topic is not None
            assert created_topic['name'] == topic.name
            assert created_topic['action_item_count'] == 1

        finally:
            await neo4j.close()
            await openai.close()


class TestTopicLinking:
    """Test linking action items to existing topics."""

    @pytest.mark.asyncio
    async def test_link_to_existing_topic(
        self, openai_api_key, neo4j_credentials, sample_tenant_id
    ):
        """Test linking an action item to an existing topic."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repository = ActionItemRepository(neo4j)
            executor = TopicExecutor(repository, openai, update_summary_on_link=False)

            tenant_id = uuid.UUID(sample_tenant_id)
            account_id = f"test_account_{uuid.uuid4().hex[:8]}"

            # Ensure account exists
            await repository.ensure_account(tenant_id, account_id, "Test Account")

            # Create an existing topic
            from action_item_graph.models.topic import Topic

            existing_topic = Topic(
                id=uuid.uuid4(),
                tenant_id=tenant_id,
                account_id=account_id,
                name="Q1 Sales Expansion",
                canonical_name="q1 sales expansion",
                summary="Strategic initiative for Q1 sales growth",
                action_item_count=2,
            )
            await repository.create_topic(existing_topic)

            # Create action item to link
            action_item_id = str(uuid.uuid4())
            from action_item_graph.models.action_item import ActionItem, ActionItemStatus

            action_item = ActionItem(
                id=uuid.UUID(action_item_id),
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text="Review SDR candidates",
                summary="Review candidates for SDR position",
                owner="John",
                status=ActionItemStatus.OPEN,
            )
            await repository.create_action_item(action_item)

            # Create resolution for linking
            topic = ExtractedTopic(
                name="Q1 Sales Expansion",
                context="Part of the Q1 hiring initiative.",
            )

            resolution = TopicResolutionResult(
                action_item_id=action_item_id,
                action_item_summary="Review candidates for SDR position",
                extracted_topic=topic,
                decision=TopicDecision.LINK_EXISTING,
                topic_id=str(existing_topic.id),
                confidence=0.92,
                best_candidate=TopicCandidate(
                    topic_id=str(existing_topic.id),
                    name=existing_topic.name,
                    canonical_name=existing_topic.canonical_name,
                    summary=existing_topic.summary,
                    action_item_count=existing_topic.action_item_count,
                    similarity=0.92,
                ),
            )

            # Execute the resolution
            result = await executor.execute_resolution(
                resolution=resolution,
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text="Review SDR candidates",
                owner="John",
            )

            # Verify result
            assert result.was_new is False
            assert result.action == 'linked'
            assert result.topic_id == str(existing_topic.id)
            assert result.details['new_action_item_count'] == 3  # Was 2, now 3

            # Verify topic count was incremented
            updated_topic = await repository.get_topic(str(existing_topic.id), tenant_id)
            assert updated_topic['action_item_count'] == 3

        finally:
            await neo4j.close()
            await openai.close()


class TestTopicExecutionResult:
    """Test TopicExecutionResult dataclass."""

    def test_execution_result_created(self):
        """Test execution result for created topic."""
        result = TopicExecutionResult(
            action_item_id="ai-123",
            topic_id="topic-456",
            topic_name="Q1 Sales Expansion",
            action='created',
            was_new=True,
            version_created=True,
        )

        assert result.was_new is True
        assert result.action == 'created'
        assert result.version_created is True

    def test_execution_result_linked(self):
        """Test execution result for linked topic."""
        result = TopicExecutionResult(
            action_item_id="ai-123",
            topic_id="topic-456",
            topic_name="Q1 Sales Expansion",
            action='linked',
            was_new=False,
            details={'similarity': 0.92, 'new_action_item_count': 5},
        )

        assert result.was_new is False
        assert result.action == 'linked'
        assert result.details['similarity'] == 0.92
