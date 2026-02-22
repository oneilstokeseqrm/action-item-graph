"""Tests for the /health endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from action_item_graph.api.routes.health import router


def _make_app(neo4j_client=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.neo4j = neo4j_client or AsyncMock()
    return app


class TestHealthRoute:
    def test_health_ok(self):
        mock_neo4j = AsyncMock()
        mock_neo4j.verify_connectivity = AsyncMock(return_value=None)
        app = _make_app(mock_neo4j)
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_neo4j_down(self):
        mock_neo4j = AsyncMock()
        mock_neo4j.verify_connectivity = AsyncMock(side_effect=Exception("Connection refused"))
        app = _make_app(mock_neo4j)
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 503
