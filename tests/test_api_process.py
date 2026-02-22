"""Tests for POST /process endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from action_item_graph.api.routes.process import router
from action_item_graph.api.auth import verify_worker_token


def _make_app(dispatcher_result=None) -> FastAPI:
    """Build a test app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    # Override auth dependency so it never hits real Settings
    async def _noop_auth():
        return None

    app.dependency_overrides[verify_worker_token] = _noop_auth

    # Mock clients stored in app.state
    app.state.neo4j = AsyncMock()
    app.state.deal_neo4j = AsyncMock()
    app.state.openai = MagicMock()

    return app


VALID_ENVELOPE = {
    "schema_version": "v1",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "auth0|test123",
    "interaction_type": "transcript",
    "content": {
        "text": "John: I'll send the proposal by Friday.\nSarah: Great, thanks.",
        "format": "diarized",
    },
    "timestamp": "2026-02-14T15:30:00Z",
    "source": "web-mic",
    "account_id": "acct_test_001",
}


class TestProcessRoute:
    @patch("action_item_graph.api.routes.process.EnvelopeDispatcher")
    def test_valid_envelope_returns_200(self, mock_dispatcher_cls):
        mock_result = MagicMock()
        mock_result.overall_success = True
        mock_result.to_dict.return_value = {
            "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
            "overall_success": True,
        }
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch.return_value = mock_result
        mock_dispatcher_cls.return_value = mock_dispatcher

        app = _make_app()
        client = TestClient(app)
        response = client.post(
            "/process",
            json=VALID_ENVELOPE,
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        assert response.json()["overall_success"] is True

    def test_invalid_envelope_returns_422(self):
        app = _make_app()
        client = TestClient(app)
        response = client.post(
            "/process",
            json={"not": "valid"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 422

    @patch("action_item_graph.api.routes.process.EnvelopeDispatcher")
    def test_pipeline_failure_returns_500(self, mock_dispatcher_cls):
        mock_dispatcher = AsyncMock()
        mock_dispatcher.dispatch.side_effect = Exception("OpenAI down")
        mock_dispatcher_cls.return_value = mock_dispatcher

        app = _make_app()
        client = TestClient(app)
        response = client.post(
            "/process",
            json=VALID_ENVELOPE,
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 500
