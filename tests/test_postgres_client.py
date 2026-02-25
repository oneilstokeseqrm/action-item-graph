"""
Tests for the PostgresClient dual-write module.

Tests cover:
- Connection management (connect, close, verify_connectivity)
- ActionItem UPSERT with field mapping
- ActionItemVersion INSERT
- ActionItemTopic UPSERT
- ActionItemTopicVersion INSERT
- Topic membership UPSERT
- Owner UPSERT
- Entity linking (action_item_links)
- Batch persist_action_item_full convenience method
- Status mapping (Neo4j open → Postgres pending)
- Embedding-to-pgvector conversion
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from action_item_graph.clients.postgres_client import (
    PostgresClient,
    _embedding_to_pgvector,
    _map_status,
    _to_pg_ts,
    _to_pg_uuid,
)
from action_item_graph.models.action_item import ActionItem, ActionItemStatus, ActionItemVersion
from action_item_graph.models.entities import Owner
from action_item_graph.models.topic import ActionItemTopic, ActionItemTopicVersion


# =============================================================================
# Fixtures
# =============================================================================

TENANT_ID = UUID('550e8400-e29b-41d4-a716-446655440000')
ACCOUNT_ID = 'acct_test_001'


@pytest.fixture
def mock_engine():
    """Create a mock AsyncEngine with a mock connection context manager."""
    engine = AsyncMock()

    # Mock the begin() context manager to yield a mock connection
    conn = AsyncMock()
    conn.execute = AsyncMock()

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)

    engine.begin = MagicMock(return_value=ctx)
    engine.dispose = AsyncMock()

    return engine, conn


@pytest.fixture
def client(mock_engine):
    """Create a PostgresClient with a pre-injected mock engine."""
    engine, _ = mock_engine
    pg = PostgresClient()
    pg._engine = engine
    return pg


@pytest.fixture
def sample_action_item() -> ActionItem:
    """ActionItem with representative field values."""
    return ActionItem(
        id=uuid4(),
        tenant_id=TENANT_ID,
        account_id=ACCOUNT_ID,
        action_item_text='Send the proposal to the client by Friday',
        summary='Send proposal by Friday',
        owner='John Smith',
        owner_type='named',
        is_user_owned=True,
        conversation_context='Discussed during Q1 planning meeting',
        due_date=datetime(2026, 3, 1),
        status=ActionItemStatus.OPEN,
        version=1,
        evolution_summary='',
        source_interaction_id=uuid4(),
        user_id='auth0|user123',
        pg_user_id=uuid4(),
        embedding=[0.1] * 1536,
        embedding_current=[0.2] * 1536,
        confidence=0.95,
        valid_at=datetime(2026, 2, 25),
    )


@pytest.fixture
def sample_version(sample_action_item) -> ActionItemVersion:
    return ActionItemVersion(
        id=uuid4(),
        action_item_id=sample_action_item.id,
        tenant_id=TENANT_ID,
        version=1,
        action_item_text=sample_action_item.action_item_text,
        summary=sample_action_item.summary,
        owner=sample_action_item.owner,
        status=ActionItemStatus.OPEN,
        due_date=sample_action_item.due_date,
        change_summary='Initial extraction',
        change_source_interaction_id=sample_action_item.source_interaction_id,
        valid_from=datetime(2026, 2, 25),
    )


@pytest.fixture
def sample_topic() -> ActionItemTopic:
    return ActionItemTopic(
        id=uuid4(),
        tenant_id=TENANT_ID,
        account_id=ACCOUNT_ID,
        name='Q1 Planning',
        canonical_name='q1 planning',
        summary='Quarterly planning activities and tasks',
        embedding=[0.3] * 1536,
        embedding_current=[0.4] * 1536,
        action_item_count=3,
        version=2,
    )


@pytest.fixture
def sample_topic_version(sample_topic) -> ActionItemTopicVersion:
    return ActionItemTopicVersion(
        id=uuid4(),
        topic_id=sample_topic.id,
        tenant_id=TENANT_ID,
        version_number=1,
        name=sample_topic.name,
        summary=sample_topic.summary,
        embedding_snapshot=[0.3] * 1536,
        changed_by_action_item_id=uuid4(),
    )


@pytest.fixture
def sample_owner() -> Owner:
    return Owner(
        id=uuid4(),
        tenant_id=TENANT_ID,
        canonical_name='John Smith',
        aliases=['John', 'J. Smith'],
        contact_id=None,
        user_id='auth0|user123',
    )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestStatusMapping:
    """Test Neo4j → Postgres status mapping."""

    def test_open_maps_to_pending(self):
        assert _map_status('open') == 'pending'

    def test_in_progress_maps_directly(self):
        assert _map_status('in_progress') == 'in_progress'

    def test_completed_maps_directly(self):
        assert _map_status('completed') == 'completed'

    def test_cancelled_maps_directly(self):
        assert _map_status('cancelled') == 'cancelled'

    def test_deferred_maps_directly(self):
        assert _map_status('deferred') == 'deferred'

    def test_unknown_status_defaults_to_pending(self):
        assert _map_status('bogus') == 'pending'


class TestHelpers:
    """Test conversion helper functions."""

    def test_to_pg_uuid_with_uuid(self):
        uid = uuid4()
        assert _to_pg_uuid(uid) == str(uid)

    def test_to_pg_uuid_with_string(self):
        s = 'abc-123'
        assert _to_pg_uuid(s) == 'abc-123'

    def test_to_pg_uuid_with_none(self):
        assert _to_pg_uuid(None) is None

    def test_to_pg_ts_with_datetime(self):
        dt = datetime(2026, 2, 25, 12, 30)
        assert _to_pg_ts(dt) == dt.isoformat()

    def test_to_pg_ts_with_string(self):
        s = '2026-02-25T12:30:00'
        assert _to_pg_ts(s) == s

    def test_to_pg_ts_with_none(self):
        assert _to_pg_ts(None) is None

    def test_embedding_to_pgvector(self):
        emb = [0.1, 0.2, 0.3]
        result = _embedding_to_pgvector(emb)
        assert result == '[0.1,0.2,0.3]'

    def test_embedding_to_pgvector_none(self):
        assert _embedding_to_pgvector(None) is None


# =============================================================================
# Connection Management Tests
# =============================================================================


class TestConnectionManagement:
    """Test connect, close, verify_connectivity."""

    @pytest.mark.asyncio
    async def test_connect_creates_engine(self):
        pg = PostgresClient()
        with patch(
            'action_item_graph.clients.postgres_client.create_async_engine'
        ) as mock_create:
            mock_create.return_value = AsyncMock()
            await pg.connect('postgresql://localhost/test')
            mock_create.assert_called_once()
            assert pg._engine is not None

    @pytest.mark.asyncio
    async def test_connect_normalises_postgres_url(self):
        pg = PostgresClient()
        with patch(
            'action_item_graph.clients.postgres_client.create_async_engine'
        ) as mock_create:
            mock_create.return_value = AsyncMock()
            await pg.connect('postgres://host/db')

            # Should have been converted to postgresql+asyncpg://
            call_args = mock_create.call_args
            url = call_args[0][0]
            assert url.startswith('postgresql+asyncpg://')

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        pg = PostgresClient()
        with patch(
            'action_item_graph.clients.postgres_client.create_async_engine'
        ) as mock_create:
            mock_create.return_value = AsyncMock()
            await pg.connect('postgresql://localhost/test')
            await pg.connect('postgresql://localhost/test')  # second call is no-op
            assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_connect_raises_without_url(self):
        pg = PostgresClient()
        with pytest.raises(ValueError, match='database_url is required'):
            await pg.connect()

    @pytest.mark.asyncio
    async def test_close_disposes_engine(self, client, mock_engine):
        engine, _ = mock_engine
        await client.close()
        engine.dispose.assert_awaited_once()
        assert client._engine is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        pg = PostgresClient()
        pg._engine = None
        await pg.close()  # should not raise

    @pytest.mark.asyncio
    async def test_verify_connectivity_success(self, client, mock_engine):
        _, conn = mock_engine
        conn.execute = AsyncMock()
        result = await client.verify_connectivity()
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_connectivity_failure(self, client, mock_engine):
        engine, _ = mock_engine
        engine.begin.side_effect = Exception('Connection refused')
        # Re-inject since begin is now failing
        client._engine = engine
        result = await client.verify_connectivity()
        assert result is False

    def test_engine_property_raises_when_not_connected(self):
        pg = PostgresClient()
        with pytest.raises(RuntimeError, match='not connected'):
            _ = pg.engine


# =============================================================================
# UPSERT / INSERT Tests
# =============================================================================


class TestUpsertActionItem:
    """Test action item UPSERT SQL and parameter mapping."""

    @pytest.mark.asyncio
    async def test_upsert_executes_sql(self, client, mock_engine, sample_action_item):
        _, conn = mock_engine
        await client.upsert_action_item(sample_action_item)
        conn.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upsert_maps_status_open_to_pending(self, client, mock_engine, sample_action_item):
        _, conn = mock_engine
        sample_action_item.status = ActionItemStatus.OPEN
        await client.upsert_action_item(sample_action_item)

        # Extract params from the execute call
        call_args = conn.execute.call_args
        params = call_args[0][1]  # second positional arg
        assert params['status'] == 'pending'

    @pytest.mark.asyncio
    async def test_upsert_maps_field_names(self, client, mock_engine, sample_action_item):
        _, conn = mock_engine
        await client.upsert_action_item(sample_action_item)

        params = conn.execute.call_args[0][1]

        # Neo4j summary → Postgres title
        assert params['title'] == sample_action_item.summary
        # Neo4j action_item_text → Postgres description
        assert params['description'] == sample_action_item.action_item_text
        # graph_action_item_id == id (same UUID)
        assert params['graph_action_item_id'] == str(sample_action_item.id)
        assert params['id'] == str(sample_action_item.id)
        # generated_by_ai is hardcoded true in SQL
        assert params['owner_name'] == 'John Smith'
        assert params['owner_type'] == 'named'
        assert params['is_user_owned'] is True
        assert params['version_number'] == 1
        assert params['source_user_id'] == 'auth0|user123'

    @pytest.mark.asyncio
    async def test_upsert_includes_embeddings_as_pgvector(self, client, mock_engine, sample_action_item):
        _, conn = mock_engine
        await client.upsert_action_item(sample_action_item)

        params = conn.execute.call_args[0][1]
        assert params['embedding'] is not None
        assert params['embedding'].startswith('[')
        assert params['embedding_current'] is not None

    @pytest.mark.asyncio
    async def test_upsert_handles_null_optional_fields(self, client, mock_engine):
        _, conn = mock_engine
        item = ActionItem(
            tenant_id=TENANT_ID,
            action_item_text='Minimal item',
            summary='Minimal',
            owner='Someone',
        )
        await client.upsert_action_item(item)

        params = conn.execute.call_args[0][1]
        assert params['due_date'] is None
        assert params['embedding'] is None
        assert params['source_interaction_id'] is None
        assert params['valid_at'] is None

    @pytest.mark.asyncio
    async def test_upsert_ai_suggestion_details_json(self, client, mock_engine, sample_action_item):
        _, conn = mock_engine
        await client.upsert_action_item(sample_action_item)

        params = conn.execute.call_args[0][1]
        details = json.loads(params['ai_suggestion_details'])
        assert details['source'] == 'action_item_graph'
        assert details['confidence'] == 0.95


class TestInsertActionItemVersion:
    """Test action item version INSERT."""

    @pytest.mark.asyncio
    async def test_insert_version(self, client, mock_engine, sample_version):
        _, conn = mock_engine
        await client.insert_action_item_version(sample_version)
        conn.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_version_params(self, client, mock_engine, sample_version):
        _, conn = mock_engine
        await client.insert_action_item_version(sample_version)

        params = conn.execute.call_args[0][1]
        assert params['action_item_id'] == str(sample_version.action_item_id)
        assert params['version_number'] == 1
        assert params['summary'] == sample_version.summary
        assert params['change_summary'] == 'Initial extraction'


class TestUpsertTopic:
    """Test topic UPSERT."""

    @pytest.mark.asyncio
    async def test_upsert_topic(self, client, mock_engine, sample_topic):
        _, conn = mock_engine
        await client.upsert_topic(sample_topic)
        conn.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_topic_params(self, client, mock_engine, sample_topic):
        _, conn = mock_engine
        await client.upsert_topic(sample_topic)

        params = conn.execute.call_args[0][1]
        assert params['name'] == 'Q1 Planning'
        assert params['canonical_name'] == 'q1 planning'
        assert params['action_item_count'] == 3
        assert params['version_number'] == 2
        assert params['embedding'] is not None


class TestInsertTopicVersion:
    """Test topic version INSERT."""

    @pytest.mark.asyncio
    async def test_insert_topic_version(self, client, mock_engine, sample_topic_version):
        _, conn = mock_engine
        await client.insert_topic_version(sample_topic_version)
        conn.execute.assert_awaited_once()


class TestUpsertTopicMembership:
    """Test topic membership UPSERT."""

    @pytest.mark.asyncio
    async def test_upsert_membership(self, client, mock_engine):
        _, conn = mock_engine
        await client.upsert_topic_membership(
            tenant_id=TENANT_ID,
            action_item_id=uuid4(),
            topic_id=uuid4(),
            confidence=0.85,
            method='resolved',
        )
        conn.execute.assert_awaited_once()

        params = conn.execute.call_args[0][1]
        assert params['confidence'] == 0.85
        assert params['method'] == 'resolved'


class TestUpsertOwner:
    """Test owner UPSERT."""

    @pytest.mark.asyncio
    async def test_upsert_owner(self, client, mock_engine, sample_owner):
        _, conn = mock_engine
        await client.upsert_owner(sample_owner)
        conn.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_owner_params(self, client, mock_engine, sample_owner):
        _, conn = mock_engine
        await client.upsert_owner(sample_owner)

        params = conn.execute.call_args[0][1]
        assert params['canonical_name'] == 'John Smith'
        aliases = json.loads(params['aliases'])
        assert 'John' in aliases
        assert 'J. Smith' in aliases


class TestLinkActionItemToEntity:
    """Test entity linking INSERT."""

    @pytest.mark.asyncio
    async def test_link_account(self, client, mock_engine):
        _, conn = mock_engine
        await client.link_action_item_to_entity(
            tenant_id=TENANT_ID,
            action_item_id=uuid4(),
            entity_type='account',
            entity_id=uuid4(),
        )
        conn.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_link_interaction(self, client, mock_engine):
        _, conn = mock_engine
        await client.link_action_item_to_entity(
            tenant_id=TENANT_ID,
            action_item_id=uuid4(),
            entity_type='interaction',
            entity_id=uuid4(),
        )
        conn.execute.assert_awaited_once()


# =============================================================================
# Batch Convenience Method Tests
# =============================================================================


class TestPersistActionItemFull:
    """Test the batch persist_action_item_full convenience method."""

    @pytest.mark.asyncio
    async def test_persist_writes_action_item(self, client, mock_engine, sample_action_item):
        _, conn = mock_engine
        await client.persist_action_item_full(item=sample_action_item)
        # At minimum, action item UPSERT + account link + no interaction link
        assert conn.execute.await_count >= 1

    @pytest.mark.asyncio
    async def test_persist_with_owner(self, client, mock_engine, sample_action_item, sample_owner):
        _, conn = mock_engine
        await client.persist_action_item_full(
            item=sample_action_item,
            owner=sample_owner,
        )
        # action item + owner + account link
        assert conn.execute.await_count >= 2

    @pytest.mark.asyncio
    async def test_persist_with_topic(
        self, client, mock_engine, sample_action_item, sample_topic
    ):
        _, conn = mock_engine
        await client.persist_action_item_full(
            item=sample_action_item,
            topic=sample_topic,
            topic_confidence=0.9,
        )
        # action item + topic + membership + account link
        assert conn.execute.await_count >= 3

    @pytest.mark.asyncio
    async def test_persist_with_interaction_link(self, client, mock_engine, sample_action_item):
        _, conn = mock_engine
        interaction_id = uuid4()
        await client.persist_action_item_full(
            item=sample_action_item,
            interaction_id=interaction_id,
        )
        # action item + account link + interaction link
        assert conn.execute.await_count >= 2

    @pytest.mark.asyncio
    async def test_persist_owner_failure_does_not_block(
        self, client, mock_engine, sample_action_item, sample_owner
    ):
        """Owner write failure should not prevent action item write."""
        _, conn = mock_engine

        call_count = 0
        original_execute = conn.execute

        async def _failing_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception('Owner write failed')
            return await original_execute(*args, **kwargs)

        conn.execute = AsyncMock(side_effect=_failing_on_second)

        # Should not raise despite owner failure
        await client.persist_action_item_full(
            item=sample_action_item,
            owner=sample_owner,
        )

    @pytest.mark.asyncio
    async def test_persist_topic_failure_does_not_block(
        self, client, mock_engine, sample_action_item, sample_topic
    ):
        """Topic write failure should not prevent action item write."""
        _, conn = mock_engine

        call_count = 0
        original_execute = conn.execute

        async def _failing_on_second(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception('Topic write failed')
            return await original_execute(*args, **kwargs)

        conn.execute = AsyncMock(side_effect=_failing_on_second)

        # Should not raise despite topic failure
        await client.persist_action_item_full(
            item=sample_action_item,
            topic=sample_topic,
        )
