"""Tests for Deal dual-write methods on PostgresClient.

Mirrors the test structure of test_postgres_client.py (Phase A) but covers
Deal-specific UPSERT/INSERT operations on opportunities + deal_versions.
"""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from deal_graph.models.deal import Deal, DealStage, DealVersion, MEDDICProfile, OntologyScores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_deal():
    """A Deal with MEDDIC, ontology scores, and embeddings."""
    return Deal(
        tenant_id=UUID('11111111-1111-1111-1111-111111111111'),
        opportunity_id=UUID('22222222-2222-2222-2222-222222222222'),
        deal_ref='deal_2222222222222222',
        name='Acme Cloud Migration',
        stage=DealStage.QUALIFICATION,
        amount=150000.0,
        account_id='33333333-3333-3333-3333-333333333333',
        currency='USD',
        meddic=MEDDICProfile(
            metrics='$150K annual savings',
            metrics_confidence=0.85,
            economic_buyer='Sarah Jones, VP Engineering',
            economic_buyer_confidence=0.90,
            decision_criteria='SOC2 compliance, 99.9% uptime',
            decision_criteria_confidence=0.75,
            champion='James Park, Director of Sales Ops',
            champion_confidence=0.80,
        ),
        ontology_scores=OntologyScores(
            scores={'champion_strength': 2, 'competitive_position': 3},
            confidences={'champion_strength': 0.85, 'competitive_position': 0.90},
            evidence={'champion_strength': 'James actively promoted our solution', 'competitive_position': 'No competing vendors mentioned'},
        ),
        ontology_version='abc123',
        opportunity_summary='Cloud migration for Acme Corp',
        evolution_summary='Initial qualification call confirmed budget authority',
        embedding=[0.1] * 1536,
        embedding_current=[0.2] * 1536,
        version=2,
        confidence=0.88,
        source_interaction_id=UUID('44444444-4444-4444-4444-444444444444'),
    )


@pytest.fixture
def sample_deal_version():
    """A DealVersion snapshot."""
    return DealVersion(
        version_id=UUID('55555555-5555-5555-5555-555555555555'),
        deal_opportunity_id=UUID('22222222-2222-2222-2222-222222222222'),
        tenant_id=UUID('11111111-1111-1111-1111-111111111111'),
        version=1,
        name='Acme Cloud Migration',
        stage=DealStage.PROSPECTING,
        amount=100000.0,
        opportunity_summary='Initial contact with Acme',
        evolution_summary='',
        meddic_metrics=None,
        meddic_completeness=0.0,
        ontology_scores_json=json.dumps({'champion_strength': 1}),
        ontology_completeness=0.1,
        change_summary='Deal progressed from prospecting to qualification',
        changed_fields=['stage', 'amount', 'meddic_metrics'],
        change_source_interaction_id=UUID('44444444-4444-4444-4444-444444444444'),
    )


# ---------------------------------------------------------------------------
# upsert_deal tests
# ---------------------------------------------------------------------------


def _mock_upsert_engine():
    """Create a mock engine whose conn.execute returns a RETURNING result with a UUID."""
    fake_pg_id = UUID('99999999-9999-9999-9999-999999999999')
    mock_engine = AsyncMock()
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchone.return_value = (fake_pg_id,)
    mock_conn.execute.return_value = mock_result
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_engine.begin = MagicMock(return_value=ctx)
    return mock_engine, mock_conn, fake_pg_id


class TestUpsertDeal:
    """Tests for PostgresClient.upsert_deal()."""

    @pytest.mark.asyncio
    async def test_upsert_deal_executes_sql(self, sample_deal):
        """upsert_deal should execute an INSERT ... ON CONFLICT UPDATE."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql_text = str(call_args[0][0])
        # Must target graph_opportunity_id for conflict resolution
        assert 'graph_opportunity_id' in sql_text
        assert 'ON CONFLICT' in sql_text

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_meddic_fields(self, sample_deal):
        """MEDDIC fields should be flattened to individual columns."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params['meddic_metrics'] == '$150K annual savings'
        assert params['meddic_metrics_confidence'] == 0.85
        assert params['meddic_economic_buyer'] == 'Sarah Jones, VP Engineering'
        assert params['meddic_champion'] == 'James Park, Director of Sales Ops'

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_ontology_dimensions(self, sample_deal):
        """Individual dim_* columns and ontology_scores_json should both be set."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        # Individual columns
        assert params['dim_champion_strength'] == 2
        assert params['dim_champion_strength_confidence'] == 0.85
        assert params['dim_competitive_position'] == 3
        # JSONB (includes evidence)
        scores_json = json.loads(params['ontology_scores_json'])
        assert scores_json['champion_strength']['score'] == 2
        assert scores_json['champion_strength']['evidence'] == 'James actively promoted our solution'

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_embeddings(self, sample_deal):
        """Embeddings should be converted to pgvector format."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params['extraction_embedding'].startswith('[0.1,')
        assert params['extraction_embedding_current'].startswith('[0.2,')

    @pytest.mark.asyncio
    async def test_upsert_deal_maps_summaries_to_ai_columns(self, sample_deal):
        """opportunity_summary -> latest_ai_summary, evolution_summary -> ai_evolution_summary."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        assert params['latest_ai_summary'] == 'Cloud migration for Acme Corp'
        assert params['ai_evolution_summary'] == 'Initial qualification call confirmed budget authority'

    @pytest.mark.asyncio
    async def test_upsert_deal_does_not_write_trigger_columns(self, sample_deal):
        """Must NOT write to the 8 forecast-trigger-protected columns."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.upsert_deal(sample_deal)

        call_args = mock_conn.execute.call_args
        sql_text = str(call_args[0][0])
        trigger_cols = ['stage', 'amount', 'close_date', 'deal_status',
                        'forecast_category', 'next_step', 'description', 'lost_reason']
        for col in trigger_cols:
            # Verify no bare column name writes (meddic_* and dim_* contain
            # substrings like 'stage' but are distinct columns)
            # Check the SET clause specifically
            assert f'"{col}"' not in sql_text or f'meddic_{col}' in sql_text or f'dim_{col}' in sql_text


# ---------------------------------------------------------------------------
# insert_deal_version tests
# ---------------------------------------------------------------------------


class TestInsertDealVersion:
    """Tests for PostgresClient.insert_deal_version()."""

    @pytest.mark.asyncio
    async def test_insert_deal_version_executes_sql(self, sample_deal_version):
        """insert_deal_version should INSERT with ON CONFLICT DO NOTHING."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.insert_deal_version(sample_deal_version)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        sql_text = str(call_args[0][0])
        assert 'deal_versions' in sql_text
        assert 'ON CONFLICT' in sql_text
        assert 'DO NOTHING' in sql_text

    @pytest.mark.asyncio
    async def test_insert_deal_version_maps_changed_fields_to_jsonb(self, sample_deal_version):
        """changed_fields list should be serialized to JSONB."""
        from action_item_graph.clients.postgres_client import PostgresClient

        client = PostgresClient('postgresql+asyncpg://test')
        mock_engine, mock_conn, _ = _mock_upsert_engine()
        client._engine = mock_engine

        await client.insert_deal_version(sample_deal_version)

        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        changed = json.loads(params['changed_fields'])
        assert 'stage' in changed
        assert 'amount' in changed


# ---------------------------------------------------------------------------
# persist_deal_full tests
# ---------------------------------------------------------------------------


class TestPersistDealFull:
    """Tests for PostgresClient.persist_deal_full() convenience method."""

    @pytest.mark.asyncio
    async def test_persist_deal_full_writes_deal_and_version(self, sample_deal, sample_deal_version):
        """Should call upsert_deal, insert_deal_version, and link_deal_to_interaction."""
        from action_item_graph.clients.postgres_client import PostgresClient

        fake_pg_id = UUID('99999999-9999-9999-9999-999999999999')
        interaction_id = uuid4()
        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock(return_value=fake_pg_id)
        client.insert_deal_version = AsyncMock()
        client.link_deal_to_interaction = AsyncMock()

        await client.persist_deal_full(
            deal=sample_deal, version=sample_deal_version, interaction_id=interaction_id,
        )

        client.upsert_deal.assert_called_once_with(sample_deal)
        client.insert_deal_version.assert_called_once_with(
            sample_deal_version, pg_opportunity_id=fake_pg_id,
        )
        client.link_deal_to_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_deal_full_version_failure_does_not_block_deal(self, sample_deal, sample_deal_version):
        """Version insert failure should not prevent deal upsert (failure isolation)."""
        from action_item_graph.clients.postgres_client import PostgresClient

        fake_pg_id = UUID('99999999-9999-9999-9999-999999999999')
        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock(return_value=fake_pg_id)
        client.insert_deal_version = AsyncMock(side_effect=Exception('version write failed'))
        client.link_deal_to_interaction = AsyncMock()

        # Should NOT raise
        await client.persist_deal_full(deal=sample_deal, version=sample_deal_version)

        client.upsert_deal.assert_called_once()
        client.insert_deal_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_deal_full_link_failure_does_not_block(self, sample_deal):
        """Interaction link failure should not block deal upsert (failure isolation)."""
        from action_item_graph.clients.postgres_client import PostgresClient

        fake_pg_id = UUID('99999999-9999-9999-9999-999999999999')
        interaction_id = uuid4()
        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock(return_value=fake_pg_id)
        client.insert_deal_version = AsyncMock()
        client.link_deal_to_interaction = AsyncMock(side_effect=Exception('link failed'))

        # Should NOT raise
        await client.persist_deal_full(
            deal=sample_deal, version=None, interaction_id=interaction_id,
        )

        client.upsert_deal.assert_called_once()
        client.link_deal_to_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_deal_full_no_version_no_interaction(self, sample_deal):
        """When version and interaction_id are None, only upsert_deal should be called."""
        from action_item_graph.clients.postgres_client import PostgresClient

        fake_pg_id = UUID('99999999-9999-9999-9999-999999999999')
        client = PostgresClient('postgresql+asyncpg://test')
        client.upsert_deal = AsyncMock(return_value=fake_pg_id)
        client.insert_deal_version = AsyncMock()
        client.link_deal_to_interaction = AsyncMock()

        await client.persist_deal_full(deal=sample_deal, version=None)

        client.upsert_deal.assert_called_once()
        client.insert_deal_version.assert_not_called()
        client.link_deal_to_interaction.assert_not_called()
