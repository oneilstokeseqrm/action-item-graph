"""DBOS workflows for action-item-graph.

Phase B of the DBOS migration (`docs/plans/2026-05-20-dbos-migration-
execution-plan.md`) introduces:

- ``queues.py`` — DBOS Queue objects for both pipelines
- ``_runtime.py`` — module-level client registry (set by FastAPI lifespan,
  consumed by step functions which can't reach request.app.state)
- ``action_item_steps.py`` — the 14 @DBOS.step functions for the action-item
  pipeline (one per stage; S9 / S10 split into LLM + Neo4j-write per Codex #10)
- ``action_item_workflow.py`` — the @DBOS.workflow orchestrator

Mirrored for deals at ``src/deal_graph/workflows/``.

Lambda dispatcher (T13/Phase C) imports the queue names + workflow IDs
from here to call ``DBOSClient.enqueue``.
"""

from action_item_graph.workflows._runtime import (
    WorkflowClients,
    get_clients,
    register_clients,
    reset_clients_for_testing,
)
from action_item_graph.workflows.action_item_workflow import action_item_workflow
from action_item_graph.workflows.queues import (
    ACTION_ITEM_QUEUE,
    DEAL_QUEUE,
)

__all__ = [
    "ACTION_ITEM_QUEUE",
    "DEAL_QUEUE",
    "WorkflowClients",
    "action_item_workflow",
    "get_clients",
    "register_clients",
    "reset_clients_for_testing",
]
