"""Tests for the DBOS workflow client registry (workflows/_runtime.py).

The registry is the bridge between FastAPI lifespan's connected clients and
the module-level @DBOS.step functions that can't reach `request.app.state`.
"""

import dataclasses

import pytest
from unittest.mock import MagicMock

from action_item_graph.workflows import _runtime
from action_item_graph.workflows._runtime import (
    WorkflowClients,
    get_clients,
    register_clients,
    reset_clients_for_testing,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test sees a fresh registry; teardown restores it."""
    prior = _runtime._clients
    _runtime._clients = None
    yield
    _runtime._clients = prior


def _build_clients() -> WorkflowClients:
    return WorkflowClients(
        neo4j=MagicMock(),
        deal_neo4j=MagicMock(),
        openai=MagicMock(),
        postgres=None,
    )


class TestRegistry:
    def test_get_clients_after_register_returns_same_instance(self):
        clients = _build_clients()
        register_clients(clients)
        assert get_clients() is clients

    def test_get_clients_without_register_raises(self):
        """Test fixture or production misconfiguration must surface a clear error."""
        with pytest.raises(RuntimeError, match="DBOS workflow clients not registered"):
            get_clients()

    def test_error_message_names_lifespan_contract(self):
        """Operator should know where to look — message points at FastAPI lifespan."""
        with pytest.raises(RuntimeError) as exc_info:
            get_clients()
        msg = str(exc_info.value)
        assert "register_clients" in msg
        assert "lifespan" in msg.lower()

    def test_reset_clients_for_testing_clears_registry(self):
        register_clients(_build_clients())
        assert get_clients() is not None
        reset_clients_for_testing()
        with pytest.raises(RuntimeError):
            get_clients()


class TestWorkflowClientsImmutability:
    def test_clients_dataclass_is_frozen(self):
        """A step accidentally rebinding clients.neo4j = other_client would
        silently corrupt cross-step state. Frozen guards against that."""
        clients = _build_clients()
        with pytest.raises(dataclasses.FrozenInstanceError):
            clients.neo4j = MagicMock()  # type: ignore[misc]
