"""
Tests for the pipeline dual-write integration.

Tests cover:
- Pipeline accepts optional PostgresClient
- Dual-write is called after merges and topics
- Postgres failures don't propagate to pipeline result
- Neo4j-to-ActionItem/Topic model reconstruction helpers
- Pipeline works correctly without PostgresClient (no-op)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from action_item_graph.models.action_item import ActionItem, ActionItemStatus
from action_item_graph.pipeline.pipeline import (
    ActionItemPipeline,
    _neo4j_node_to_action_item,
    _neo4j_node_to_topic,
    _parse_iso,
    _parse_uuid,
)


TENANT_ID = UUID('550e8400-e29b-41d4-a716-446655440000')


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestParseIso:
    """Test ISO datetime string parsing."""

    def test_parse_valid_iso(self):
        result = _parse_iso('2026-02-25T12:30:00')
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 25

    def test_parse_datetime_passthrough(self):
        dt = datetime(2026, 2, 25)
        assert _parse_iso(dt) is dt

    def test_parse_none(self):
        assert _parse_iso(None) is None

    def test_parse_invalid(self):
        assert _parse_iso('not-a-date') is None


class TestParseUuid:
    """Test UUID string parsing."""

    def test_parse_valid_uuid(self):
        uid = uuid4()
        result = _parse_uuid(str(uid))
        assert result == uid

    def test_parse_uuid_passthrough(self):
        uid = uuid4()
        assert _parse_uuid(uid) is uid

    def test_parse_none(self):
        assert _parse_uuid(None) is None

    def test_parse_invalid(self):
        assert _parse_uuid('not-a-uuid') is None


class TestNeo4jNodeToActionItem:
    """Test converting Neo4j node dict to ActionItem model."""

    def test_basic_conversion(self):
        item_id = str(uuid4())
        node = {
            'action_item_id': item_id,
            'tenant_id': str(TENANT_ID),
            'account_id': 'acct_test',
            'action_item_text': 'Send the proposal',
            'summary': 'Send proposal',
            'owner': 'John',
            'owner_type': 'named',
            'is_user_owned': False,
            'conversation_context': 'From the planning call',
            'status': 'open',
            'version': 2,
            'evolution_summary': 'Updated from second call',
            'created_at': '2026-02-25T10:00:00',
            'last_updated_at': '2026-02-25T11:00:00',
            'confidence': 0.9,
        }

        item = _neo4j_node_to_action_item(node)

        assert str(item.id) == item_id
        assert item.tenant_id == TENANT_ID
        assert item.account_id == 'acct_test'
        assert item.action_item_text == 'Send the proposal'
        assert item.summary == 'Send proposal'
        assert item.owner == 'John'
        assert item.version == 2
        assert item.confidence == 0.9

    def test_handles_missing_optional_fields(self):
        """Node with only required fields should still convert."""
        node = {
            'action_item_id': str(uuid4()),
            'tenant_id': str(TENANT_ID),
            'action_item_text': 'Minimal item',
            'summary': 'Minimal',
            'owner': 'Someone',
        }

        item = _neo4j_node_to_action_item(node)

        assert item.account_id is None
        assert item.due_date is None
        assert item.source_interaction_id is None
        assert item.embedding is None

    def test_preserves_embeddings(self):
        emb = [0.1, 0.2, 0.3]
        node = {
            'action_item_id': str(uuid4()),
            'tenant_id': str(TENANT_ID),
            'action_item_text': 'Test',
            'summary': 'Test',
            'owner': 'Test',
            'embedding': emb,
            'embedding_current': emb,
        }

        item = _neo4j_node_to_action_item(node)
        assert item.embedding == emb
        assert item.embedding_current == emb


class TestNeo4jNodeToTopic:
    """Test converting Neo4j topic node to ActionItemTopic model."""

    def test_basic_conversion(self):
        topic_id = str(uuid4())
        node = {
            'action_item_topic_id': topic_id,
            'tenant_id': str(TENANT_ID),
            'account_id': 'acct_test',
            'name': 'Q1 Planning',
            'canonical_name': 'q1 planning',
            'summary': 'Quarterly planning tasks',
            'action_item_count': 5,
            'version': 3,
            'created_at': '2026-02-20T08:00:00',
            'updated_at': '2026-02-25T10:00:00',
        }

        topic = _neo4j_node_to_topic(node)

        assert str(topic.id) == topic_id
        assert topic.tenant_id == TENANT_ID
        assert topic.name == 'Q1 Planning'
        assert topic.canonical_name == 'q1 planning'
        assert topic.action_item_count == 5
        assert topic.version == 3

    def test_handles_missing_fields(self):
        node = {
            'action_item_topic_id': str(uuid4()),
            'tenant_id': str(TENANT_ID),
            'name': 'Min',
            'canonical_name': 'min',
        }

        topic = _neo4j_node_to_topic(node)
        assert topic.account_id == ''
        assert topic.action_item_count == 0
        assert topic.embedding is None


# =============================================================================
# Pipeline Integration Tests
# =============================================================================


class TestPipelineWithPostgres:
    """Test that the pipeline correctly wires in PostgresClient."""

    def test_pipeline_accepts_postgres_client(self):
        """Pipeline __init__ accepts optional postgres_client."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()
        mock_postgres = AsyncMock()

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
            postgres_client=mock_postgres,
        )

        assert pipeline.postgres is mock_postgres

    def test_pipeline_works_without_postgres(self):
        """Pipeline initializes correctly without postgres_client."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
        )

        assert pipeline.postgres is None

    @pytest.mark.asyncio
    async def test_close_closes_postgres(self):
        """Pipeline.close() closes the PostgresClient."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()
        mock_postgres = AsyncMock()
        mock_postgres.close = AsyncMock()

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
            postgres_client=mock_postgres,
        )

        await pipeline.close()
        mock_postgres.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_postgres(self):
        """Pipeline.close() works fine without postgres."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
        )

        await pipeline.close()  # Should not raise


class TestDualWriteFailureIsolation:
    """Verify that Postgres failures never bubble up to pipeline results."""

    @pytest.mark.asyncio
    async def test_pg_write_action_items_catches_exceptions(self):
        """Individual action item Postgres write failure is logged, not raised."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()
        mock_postgres = AsyncMock()

        # Make upsert_action_item fail
        mock_postgres.upsert_action_item = AsyncMock(
            side_effect=Exception('DB connection lost')
        )

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
            postgres_client=mock_postgres,
        )

        # Build a fake merge result + action item
        from action_item_graph.pipeline.merger import MergeResult

        merge_result = MergeResult(
            action_item_id=str(uuid4()),
            action='created',
            was_new=True,
            version_created=False,
            linked_interaction_id=str(uuid4()),
            details={'owner_name': 'Test'},
        )

        action_item = ActionItem(
            tenant_id=TENANT_ID,
            account_id='acct_test',
            action_item_text='Test item',
            summary='Test',
            owner='Test',
        )

        from action_item_graph.models.entities import Interaction, InteractionType

        interaction = Interaction(
            tenant_id=TENANT_ID,
            interaction_type=InteractionType.TRANSCRIPT,
            content_text='Test transcript',
            timestamp=datetime.now(),
        )

        # Should NOT raise despite Postgres failure
        await pipeline._pg_write_action_items(
            merge_results=[merge_result],
            action_items=[action_item],
            interaction=interaction,
        )

    @pytest.mark.asyncio
    async def test_pg_write_topics_catches_exceptions(self):
        """Topic Postgres write failure is logged, not raised."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()
        mock_postgres = AsyncMock()

        # Make upsert_topic fail
        mock_postgres.upsert_topic = AsyncMock(
            side_effect=Exception('DB connection lost')
        )

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
            postgres_client=mock_postgres,
        )

        # Mock repository.get_topic to return a valid node
        pipeline.repository.get_topic = AsyncMock(return_value={
            'action_item_topic_id': str(uuid4()),
            'tenant_id': str(TENANT_ID),
            'account_id': 'acct_test',
            'name': 'Test Topic',
            'canonical_name': 'test topic',
            'summary': 'Test summary',
        })

        from action_item_graph.pipeline.topic_executor import TopicExecutionResult

        topic_result = TopicExecutionResult(
            action_item_id=str(uuid4()),
            topic_id=str(uuid4()),
            topic_name='Test Topic',
            action='created',
            was_new=True,
            details={'confidence': 0.9, 'method': 'extracted'},
        )

        action_item = ActionItem(
            tenant_id=TENANT_ID,
            action_item_text='Test',
            summary='Test',
            owner='Test',
        )

        # Should NOT raise despite Postgres failure
        await pipeline._pg_write_topics(
            topic_results=[topic_result],
            action_items=[action_item],
        )

    @pytest.mark.asyncio
    async def test_dual_write_postgres_wraps_all_failures(self):
        """The top-level _dual_write_postgres catches ALL exceptions."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()
        mock_postgres = AsyncMock()

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
            postgres_client=mock_postgres,
        )

        # Make both sub-methods fail
        pipeline._pg_write_action_items = AsyncMock(
            side_effect=Exception('Action items failed')
        )
        pipeline._pg_write_topics = AsyncMock(
            side_effect=Exception('Topics failed')
        )

        from action_item_graph.models.entities import Interaction, InteractionType

        interaction = Interaction(
            tenant_id=TENANT_ID,
            interaction_type=InteractionType.TRANSCRIPT,
            content_text='Test',
            timestamp=datetime.now(),
        )

        # Should NOT raise
        await pipeline._dual_write_postgres(
            merge_results=[],
            action_items=[],
            interaction=interaction,
            topic_results=[],
        )

    @pytest.mark.asyncio
    async def test_dual_write_noop_without_postgres(self):
        """Dual-write is a no-op when postgres_client is None."""
        mock_openai = AsyncMock()
        mock_neo4j = AsyncMock()

        pipeline = ActionItemPipeline(
            openai_client=mock_openai,
            neo4j_client=mock_neo4j,
            postgres_client=None,
        )

        from action_item_graph.models.entities import Interaction, InteractionType

        interaction = Interaction(
            tenant_id=TENANT_ID,
            interaction_type=InteractionType.TRANSCRIPT,
            content_text='Test',
            timestamp=datetime.now(),
        )

        # Should do nothing and not raise
        await pipeline._dual_write_postgres(
            merge_results=[],
            action_items=[],
            interaction=interaction,
            topic_results=[],
        )
