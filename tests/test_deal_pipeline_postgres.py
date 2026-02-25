"""Tests for DealPipeline Postgres dual-write wiring.

Verifies that DealPipeline accepts an optional postgres_client and calls
persist_deal_full() after Neo4j merging, with failure isolation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from action_item_graph.models.envelope import (
    ContentFormat,
    ContentPayload,
    EnvelopeV1,
    InteractionType,
    SourceType,
)
from deal_graph.models.deal import Deal, DealStage, DealVersion
from deal_graph.pipeline.pipeline import DealPipeline


@pytest.fixture
def mock_neo4j():
    client = AsyncMock()
    client.driver = MagicMock()
    return client


@pytest.fixture
def mock_openai():
    return AsyncMock()


@pytest.fixture
def mock_postgres():
    pg = AsyncMock()
    pg.persist_deal_full = AsyncMock()
    return pg


class TestDealPipelinePostgresWiring:
    """DealPipeline should accept and use an optional postgres_client."""

    def test_init_accepts_postgres_client(self, mock_neo4j, mock_openai, mock_postgres):
        """DealPipeline.__init__ should accept postgres_client parameter."""
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
            postgres_client=mock_postgres,
        )
        assert pipeline.postgres is mock_postgres

    def test_init_postgres_defaults_to_none(self, mock_neo4j, mock_openai):
        """postgres_client should default to None (backward compatible)."""
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
        )
        assert pipeline.postgres is None


class TestDealPipelineDualWrite:
    """DealPipeline._dual_write_postgres() tests."""

    @pytest.mark.asyncio
    async def test_dual_write_skipped_when_no_postgres(self, mock_neo4j, mock_openai):
        """When postgres_client is None, dual write should be a no-op."""
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
        )
        # Should not raise
        await pipeline._dual_write_postgres([], str(uuid4()))

    @pytest.mark.asyncio
    async def test_dual_write_failure_does_not_raise(self, mock_neo4j, mock_openai, mock_postgres):
        """Postgres failure should be logged but not propagated."""
        mock_postgres.persist_deal_full = AsyncMock(side_effect=Exception('pg down'))
        pipeline = DealPipeline(
            neo4j_client=mock_neo4j,
            openai_client=mock_openai,
            postgres_client=mock_postgres,
        )
        # Should not raise
        await pipeline._dual_write_postgres(
            [MagicMock(opportunity_id=str(uuid4()), action='created', was_new=True, version_created=False)],
            str(uuid4()),
        )
