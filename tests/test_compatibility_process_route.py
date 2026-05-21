"""T32 compatibility test: /process route still works with DBOS infrastructure deployed.

Simulates the Phase 1 mid-state (per execution plan §Phase 1):
- Railway has the new code with DBOS workers + DBOS lifespan integration
- Lambda STILL uses the legacy HTTP path (POSTs to /process)

The /process route MUST keep working in this state, otherwise in-flight
Lambda invocations during the Phase 1 → Phase 2 cutover lose data. This
is the rollback safety net for the 2-week monitoring window.

**Deletion seam**: this test is intentionally scoped to the migration
window. It will be deleted in the same PR that retires /process from
production traffic (Phase D, Day 14+ post-deploy).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from action_item_graph.api.routes.process import router as process_router


TEST_TENANT_ID = "11111111-1111-4111-8111-111111111111"
TEST_INTERACTION_ID = "550e8400-e29b-41d4-a716-446655440000"


def _canonical_envelope_dict() -> dict[str, Any]:
    return {
        "schema_version": "v1",
        "tenant_id": TEST_TENANT_ID,
        "user_id": "auth0|test",
        "interaction_id": TEST_INTERACTION_ID,
        "interaction_type": "transcript",
        "content": {"text": "A: I'll send the proposal by Friday.", "format": "plain"},
        "timestamp": "2026-02-14T15:30:00Z",
        "source": "api",
        "account_id": "acct-lightbox",
        "extras": {},
    }


def _build_test_app() -> FastAPI:
    """FastAPI app with the /process route + minimal app.state for the route's dependencies."""
    app = FastAPI()
    app.include_router(process_router)
    # The route reads request.app.state.{neo4j,deal_neo4j,openai,postgres}
    app.state.neo4j = MagicMock()
    app.state.deal_neo4j = MagicMock()
    app.state.openai = MagicMock()
    app.state.postgres = None
    return app


@pytest.fixture
def test_client():
    """Test client with auth dependency overridden + dispatcher mocked."""
    app = _build_test_app()

    # Override the worker-token dependency so we don't need real credentials
    from action_item_graph.api.auth import verify_worker_token

    app.dependency_overrides[verify_worker_token] = lambda: None

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestProcessRouteCompatibility:
    """The /process HTTP path stays functional throughout Phase 1 + Phase 2."""

    def test_process_returns_200_for_valid_envelope(self, test_client):
        """The route accepts a valid envelope and returns the dispatcher result."""
        # Mock the dispatcher to return a successful result without touching
        # real pipelines or external services.
        mock_dispatcher_result = MagicMock()
        mock_dispatcher_result.overall_success = True
        mock_dispatcher_result.dispatch_time_ms = 100
        mock_dispatcher_result.to_dict.return_value = {
            "overall_success": True,
            "dispatch_time_ms": 100,
        }

        with patch(
            "action_item_graph.api.routes.process.EnvelopeDispatcher"
        ) as mock_disp_class:
            mock_disp = mock_disp_class.return_value
            mock_disp.dispatch = AsyncMock(return_value=mock_dispatcher_result)

            response = test_client.post("/process", json=_canonical_envelope_dict())

        assert response.status_code == 200, (
            f"/process MUST return 200 for a valid envelope; got "
            f"{response.status_code}. Phase 1 rollback safety net depends "
            f"on /process accepting Lambda invocations during the 2-week "
            f"monitoring window."
        )
        body = response.json()
        assert body["overall_success"] is True

    def test_process_returns_422_for_invalid_envelope(self, test_client):
        """Malformed envelopes still produce 422 (Pydantic validation error),
        not 500. Lambda's old retry policy depends on 4xx-no-retry vs
        5xx-retry semantics."""
        response = test_client.post(
            "/process",
            json={"schema_version": "v1"},  # missing required fields
        )
        assert response.status_code == 422

    def test_process_returns_500_for_pipeline_failure(self, test_client):
        """A pipeline failure surfaces as 500 so Lambda's SQS retry kicks in
        (vs 4xx which would be persistent failure → DLQ immediately)."""
        with patch(
            "action_item_graph.api.routes.process.EnvelopeDispatcher"
        ) as mock_disp_class:
            mock_disp = mock_disp_class.return_value
            mock_disp.dispatch = AsyncMock(side_effect=RuntimeError("Neo4j unreachable"))

            response = test_client.post("/process", json=_canonical_envelope_dict())

        assert response.status_code == 500
        body = response.json()
        assert body["overall_success"] is False
        assert "Neo4j unreachable" in body["error"]
