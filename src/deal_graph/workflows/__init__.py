"""DBOS workflow for the deal pipeline.

Mirrors ``src/action_item_graph/workflows/`` structure for the deal
pipeline. Both queues live in ``action_item_graph.workflows.queues`` as
the single source of truth; ``deal_workflow`` imports ``DEAL_QUEUE`` from
there.
"""

from deal_graph.workflows.deal_workflow import deal_workflow

__all__ = ["deal_workflow"]
