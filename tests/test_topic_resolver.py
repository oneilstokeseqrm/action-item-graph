"""
Tests for TopicResolver - topic matching and resolution logic.
"""

import uuid

import pytest

# Add project paths for imports
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from action_item_graph.pipeline.topic_resolver import (
    TopicResolver,
    TopicDecision,
    TopicCandidate,
    TopicResolutionResult,
)
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.prompts.extract_action_items import ExtractedTopic


class TestTopicResolverInit:
    """Test TopicResolver initialization."""

    @pytest.mark.asyncio
    async def test_initialization_with_defaults(self, openai_api_key, neo4j_credentials):
        """Test resolver initializes with default thresholds."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            resolver = TopicResolver(neo4j, openai)

            assert resolver.auto_link_threshold == 0.85
            assert resolver.auto_create_threshold == 0.70
        finally:
            await neo4j.close()
            await openai.close()

    @pytest.mark.asyncio
    async def test_initialization_with_custom_thresholds(self, openai_api_key, neo4j_credentials):
        """Test resolver accepts custom thresholds."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            resolver = TopicResolver(
                neo4j,
                openai,
                similarity_auto_link=0.90,
                similarity_auto_create=0.75,
            )

            assert resolver.auto_link_threshold == 0.90
            assert resolver.auto_create_threshold == 0.75
        finally:
            await neo4j.close()
            await openai.close()


class TestTopicResolution:
    """Test topic resolution logic."""

    @pytest.mark.asyncio
    async def test_resolve_creates_new_topic_when_no_candidates(
        self, openai_api_key, neo4j_credentials, sample_tenant_id
    ):
        """Test that resolution creates new topic when no matches found."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            resolver = TopicResolver(neo4j, openai)

            # Create a unique topic that won't match anything
            unique_topic = ExtractedTopic(
                name=f"Unique Test Project {uuid.uuid4().hex[:8]}",
                context="This is a completely unique topic for testing purposes.",
            )

            result = await resolver.resolve_topic(
                extracted_topic=unique_topic,
                action_item_id=str(uuid.uuid4()),
                action_item_summary="Test action item summary",
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id="test_account_001",
            )

            assert result.decision == TopicDecision.CREATE_NEW
            assert result.topic_id is None
            assert result.embedding is not None
            assert len(result.embedding) == 1536

        finally:
            await neo4j.close()
            await openai.close()

    @pytest.mark.asyncio
    async def test_resolve_returns_embedding(
        self, openai_api_key, neo4j_credentials, sample_tenant_id
    ):
        """Test that resolution generates embedding for the topic."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()

            resolver = TopicResolver(neo4j, openai)

            topic = ExtractedTopic(
                name="Q1 Sales Expansion",
                context="Strategic initiative to grow sales team in Q1.",
            )

            result = await resolver.resolve_topic(
                extracted_topic=topic,
                action_item_id=str(uuid.uuid4()),
                action_item_summary="Hire 3 SDRs by end of March",
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id="test_account_001",
            )

            # Verify embedding was generated
            assert result.embedding is not None
            assert isinstance(result.embedding, list)
            assert len(result.embedding) == 1536  # OpenAI embedding dimension
            assert all(isinstance(x, float) for x in result.embedding)

        finally:
            await neo4j.close()
            await openai.close()


class TestTopicCandidate:
    """Test TopicCandidate dataclass."""

    def test_topic_candidate_creation(self):
        """Test creating a TopicCandidate."""
        candidate = TopicCandidate(
            topic_id="topic-123",
            name="Q1 Sales Expansion",
            canonical_name="q1 sales expansion",
            summary="Strategic initiative for Q1 sales growth",
            action_item_count=5,
            similarity=0.87,
        )

        assert candidate.topic_id == "topic-123"
        assert candidate.name == "Q1 Sales Expansion"
        assert candidate.similarity == 0.87
        assert candidate.action_item_count == 5


class TestTopicDecisionThresholds:
    """Test topic decision threshold logic."""

    def test_decision_enum_values(self):
        """Test TopicDecision enum has expected values."""
        assert TopicDecision.CREATE_NEW.value == 'create_new'
        assert TopicDecision.LINK_EXISTING.value == 'link_existing'

    def test_resolution_result_needs_creation(self):
        """Test needs_creation property."""
        topic = ExtractedTopic(name="Test Topic", context="Test context")

        # CREATE_NEW should need creation
        result_create = TopicResolutionResult(
            action_item_id="ai-123",
            action_item_summary="Test summary",
            extracted_topic=topic,
            decision=TopicDecision.CREATE_NEW,
            topic_id=None,
            confidence=1.0,
        )
        assert result_create.needs_creation is True

        # LINK_EXISTING should not need creation
        result_link = TopicResolutionResult(
            action_item_id="ai-123",
            action_item_summary="Test summary",
            extracted_topic=topic,
            decision=TopicDecision.LINK_EXISTING,
            topic_id="topic-456",
            confidence=0.9,
        )
        assert result_link.needs_creation is False
