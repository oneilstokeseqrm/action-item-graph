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


class TestDBOSWorkflowRegistration:
    """Regression test for Phase F /codex Round 1 absorption.

    The Lambda dispatcher enqueues both workflows by name
    (``EnqueueOptions["workflow_name"] = "action_item_workflow" |
    "deal_workflow"``). DBOS resolves the name against
    ``DBOSRegistry.workflow_info_map``, populated as a side effect of
    the ``@DBOS.workflow()`` decorator running at module import time.

    /codex Round 1 caught that the deal workflow module was never
    imported in the production runtime path (``api/main.py`` ->
    ``action_item_graph.workflows.__init__`` chain only imported
    ``action_item_workflow``, never ``deal_workflow``). The deal queue
    on Railway would have had no registered consumer in production
    even though tests passed (tests import ``deal_workflow`` directly).

    This test pins the registration contract: after importing the
    production entrypoint chain, the DBOSRegistry MUST contain both
    workflow names. If a future engineer drops the side-effect import,
    this test fails loudly.

    Per Rule 7 (cross-service reference call shape must match
    deployment topology): the DBOSRegistry API
    (``workflow_info_map`` keyed by workflow name) was verified
    against installed ``dbos==2.22.0`` source at ``dbos/_dbos.py:211``.
    """

    def test_both_workflows_registered_after_production_import_chain(self):
        # Import the entrypoint the same way api/main.py does. Forces the
        # side-effect-import chain to execute.
        import action_item_graph.workflows  # noqa: F401

        from dbos._dbos import _get_or_create_dbos_registry

        registry = _get_or_create_dbos_registry()

        # The Lambda dispatcher uses these literal workflow names in
        # EnqueueOptions["workflow_name"]. The MERGE keys here must
        # match the names hardcoded at
        # src/action_item_graph/lambda_ingest/handler.py:46-47.
        assert "action_item_workflow" in registry.workflow_info_map, (
            "action_item_workflow not registered with DBOS. The Lambda "
            "dispatcher's enqueue call would have no consumer."
        )
        assert "deal_workflow" in registry.workflow_info_map, (
            "deal_workflow not registered with DBOS. Verify that "
            "action_item_graph/workflows/__init__.py still has the "
            "`import deal_graph.workflows` side-effect import — without "
            "it, Lambda enqueues to the deal queue have no Railway "
            "consumer in production."
        )
