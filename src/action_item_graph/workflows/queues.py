"""DBOS Queue definitions for action-item-graph workflows.

Both queues live in the same module so the Lambda dispatcher (T13/Phase C)
can import names from one place, and the queue-depth observability scripts
have one source of truth.

Concurrency posture (per D4 + Codex absorption of plan Open #23):
both queues start at ``concurrency=1`` at V1. Raising the cap is gated on
the empirical criterion documented in the execution plan: 100+ successful
invocations with no DB pool / Neo4j session / OpenAI rate-limit errors,
AND queue-depth metric trending positive. The cap matters because the
real bottleneck under load is not Railway RAM (which fits ~6 concurrent
workflows × 50 MB working set) but rather DB pool exhaustion, Neo4j
session caps, and OpenAI tier rate limits — none of which scale linearly
with worker count.

Queue-depth observability: ``SELECT queue_name, COUNT(*) FROM
dbos.workflow_queue GROUP BY queue_name`` against
``DBOS_SYSTEM_DATABASE_URL``. The smoke test (Phase A) confirmed the
``dbos.workflow_queue`` table is created at first DBOS.launch().
"""

from __future__ import annotations

from dbos import Queue

# Action-item pipeline queue. Workflow name (set by the @DBOS.workflow
# decorator on ``action_item_workflow``) is what DBOSClient.enqueue
# references.
ACTION_ITEM_QUEUE = Queue("action-item-pipeline", concurrency=1)

# Deal pipeline queue. Same concurrency posture as the action-item queue.
DEAL_QUEUE = Queue("deal-pipeline", concurrency=1)
