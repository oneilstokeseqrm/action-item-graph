"""Tests for the FastAPI app startup/shutdown and route wiring."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@asynccontextmanager
async def _noop_dbos_lifespan(app):
    """No-op replacement for ``dbos_lifespan`` in tests.

    The real ``dbos_lifespan`` calls ``DBOS.launch()`` which opens a
    Postgres connection to ``DBOS_SYSTEM_DATABASE_URL``. Tests don't want
    to depend on a live DBOS state DB; they replace the lifespan with
    this no-op.
    """
    del app
    yield


class TestAppRouteWiring:
    @patch("action_item_graph.api.main.dbos_lifespan", _noop_dbos_lifespan)
    @patch("action_item_graph.api.main.get_settings")
    @patch("action_item_graph.api.main.Neo4jClient")
    @patch("action_item_graph.api.main.DealNeo4jClient")
    @patch("action_item_graph.api.main.OpenAIClient")
    def test_health_route_registered(self, mock_openai, mock_deal_neo4j, mock_neo4j, mock_settings):
        mock_settings.return_value = MagicMock(
            NEO4J_URI="neo4j+s://test",
            NEO4J_USERNAME="neo4j",
            NEO4J_PASSWORD="pass",
            NEO4J_DATABASE="neo4j",
            OPENAI_API_KEY="sk-test",
            WORKER_API_KEY="key",
        )
        mock_neo4j_instance = AsyncMock()
        mock_neo4j.return_value = mock_neo4j_instance
        mock_deal_neo4j_instance = AsyncMock()
        mock_deal_neo4j.return_value = mock_deal_neo4j_instance
        mock_openai.return_value = MagicMock()

        from action_item_graph.api.main import app

        client = TestClient(app)
        resp = client.get("/health")
        # Lifespan sets up app.state.neo4j, health route calls verify_connectivity
        assert resp.status_code in (200, 503)

    @patch("action_item_graph.api.main.dbos_lifespan", _noop_dbos_lifespan)
    @patch("action_item_graph.api.main.get_settings")
    @patch("action_item_graph.api.main.Neo4jClient")
    @patch("action_item_graph.api.main.DealNeo4jClient")
    @patch("action_item_graph.api.main.OpenAIClient")
    def test_process_route_requires_auth(self, mock_openai, mock_deal_neo4j, mock_neo4j, mock_settings):
        mock_settings.return_value = MagicMock(
            NEO4J_URI="neo4j+s://test",
            NEO4J_USERNAME="neo4j",
            NEO4J_PASSWORD="pass",
            NEO4J_DATABASE="neo4j",
            OPENAI_API_KEY="sk-test",
            WORKER_API_KEY="key",
        )
        mock_neo4j.return_value = AsyncMock()
        mock_deal_neo4j.return_value = AsyncMock()
        mock_openai.return_value = MagicMock()

        from action_item_graph.api.main import app

        client = TestClient(app)
        resp = client.post("/process", json={})
        # Should get 401 (no auth header) or 422 (missing header)
        assert resp.status_code in (401, 422)
