"""Module-level client registry for DBOS workflows.

DBOS workflows and steps are module-level functions; they can't access
FastAPI's ``request.app.state``. The FastAPI lifespan in ``api/main.py``
calls :func:`register_clients` after Neo4j / Postgres / OpenAI clients
connect AND before ``DBOS.launch()``, so step functions can resolve
clients from anywhere by calling :func:`get_clients`.

This module is a thin registry. It doesn't own client lifecycle —
``api/main.py`` lifespan owns connect / close. The registry just makes
the existing client instances reachable from inside @DBOS.step bodies.

Pattern adopted from ``live-transcription-fastapi/services/database.py``'s
``get_async_session`` pattern. LTF wraps a single SQLAlchemy session
factory; we wrap a tuple of pre-connected clients because action-item
graph fans out across Neo4j / Postgres / OpenAI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from action_item_graph.clients.neo4j_client import Neo4jClient
    from action_item_graph.clients.openai_client import OpenAIClient
    from action_item_graph.clients.postgres_client import PostgresClient
    from deal_graph.clients.neo4j_client import DealNeo4jClient


@dataclass(frozen=True)
class WorkflowClients:
    """Bundle of pre-connected clients used by DBOS workflow steps.

    Frozen so a step can't accidentally mutate the registry between calls.
    Construct once during lifespan startup; pass by value when needed.
    """

    neo4j: "Neo4jClient"
    deal_neo4j: "DealNeo4jClient"
    openai: "OpenAIClient"
    postgres: "PostgresClient | None"


_clients: WorkflowClients | None = None


def register_clients(clients: WorkflowClients) -> None:
    """Register pre-connected clients for use by DBOS steps.

    Called once during FastAPI lifespan startup, BEFORE ``DBOS.launch()``,
    so that recovery threads picking up in-flight workflows can resolve
    clients immediately.
    """
    global _clients
    _clients = clients


def get_clients() -> WorkflowClients:
    """Return the registered clients bundle.

    Raises:
        RuntimeError: If :func:`register_clients` was never called.
            That indicates either a misconfigured lifespan (DBOS launched
            without client init) or a step being called outside of a
            FastAPI-managed runtime (e.g., a test without the lifespan
            fixture). Tests should call :func:`register_clients` with
            mocks before invoking step functions directly.
    """
    if _clients is None:
        raise RuntimeError(
            "DBOS workflow clients not registered. Either FastAPI lifespan "
            "did not call register_clients() before DBOS.launch(), or a "
            "step is being invoked outside the FastAPI runtime without a "
            "test fixture calling register_clients(WorkflowClients(...))."
        )
    return _clients


def reset_clients_for_testing() -> None:
    """Test-only hook to clear the registry between tests.

    Production code should never call this. Tests call it in teardown to
    avoid cross-test contamination when one test registers mocks and the
    next would otherwise see them.
    """
    global _clients
    _clients = None
