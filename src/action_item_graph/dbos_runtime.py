"""DBOS substrate initialization and FastAPI lifespan integration.

Phase 1 of the DBOS migration (`docs/plans/2026-05-20-dbos-migration-execution-plan.md`)
uses DBOS Transact for durable execution of the action-item and deal
pipelines. The 10-stage LLM-heavy ingest path moves out of the Lambda's
120s timeout window into per-step retried, checkpointed workflows that
run on the Railway service.

Multi-replica posture: ``executor_id`` derives from Railway's auto-injected
``RAILWAY_REPLICA_ID``. V1 ships on a single Railway replica with
``uvicorn --workers 1``; raising concurrency or replica count is gated on
the empirical criterion documented in Open #23 of the execution plan
(100+ successful invocations with no DB pool / Neo4j session / OpenAI
rate-limit errors AND queue-depth trending positive).

Pattern adapted from ``live-transcription-fastapi/services/dbos_runtime.py``.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dbos import DBOS, DBOSConfig
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def build_dbos_config() -> DBOSConfig:
    """Construct ``DBOSConfig`` from environment.

    ``DBOS_SYSTEM_DATABASE_URL`` is REQUIRED. It must be a direct
    (non-pooler) Postgres connection — Neon's pooler endpoint
    (``-pooler.`` in hostname) breaks DBOS because PgBouncer's
    transaction-mode pooling drops the advisory locks DBOS uses for
    workflow coordination. See the LTF plan §3.5 and the project memory
    entry ``reference_neon_dbos_state_dbs`` for details.

    If the env var is unset, DBOS would silently fall back to a local
    SQLite file that gets blown away on every container restart —
    defeating the durability guarantee the migration is buying. Fail
    fast with a clear error instead.

    ``RAILWAY_REPLICA_ID`` is auto-injected by Railway. Locally it is
    unset and DBOS picks its own executor identity. When ``None`` is
    passed, DBOS's config translator skips the field via a
    ``"executor_id" in config and config["executor_id"] is not None``
    guard (see ``dbos/_dbos.py:445``).
    """
    system_database_url = os.environ.get("DBOS_SYSTEM_DATABASE_URL")
    if not system_database_url:
        raise RuntimeError(
            "DBOS_SYSTEM_DATABASE_URL is required for the DBOS workflow "
            "runtime but is unset. In production this env var must point "
            "at a direct (non-pooler) Postgres connection on the Neon "
            "database `eq_aig_dbos_sys` — set it in the Railway service "
            "config. For tests, set the variable in your fixture or "
            "environment before importing the FastAPI app."
        )
    return DBOSConfig(
        name="action-item-graph",
        system_database_url=system_database_url,
        executor_id=os.environ.get("RAILWAY_REPLICA_ID"),
        # No operator admin server at V1; revisit when adding tooling
        # that benefits from it.
        run_admin_server=False,
    )


@asynccontextmanager
async def dbos_lifespan(
    app: FastAPI,
    *,
    drain_timeout_sec: int = 20,
) -> AsyncIterator[None]:
    """Launch DBOS at app startup and drain workflows at shutdown.

    ``DBOS.launch()`` and ``DBOS.destroy()`` are synchronous in DBOS
    v2.x (verified against ``dbos==2.22.0`` source at
    ``dbos/_dbos.py:519`` and ``:362``). Calling them inside an async
    lifespan is the documented pattern — they are quick startup/shutdown
    bookkeeping with no blocking I/O concerns at the event-loop level.

    ``DBOS.launch()`` starts recovery and queue worker threads
    immediately. Compose this lifespan INSIDE any setup that creates
    clients on ``app.state`` (Neo4j, Postgres, OpenAI) so recovered
    workflows can reach those clients on startup; this lifespan's
    ``finally`` clause drains in-flight workflows BEFORE those clients
    are closed. See ``api/main.py`` lifespan for the correct nesting.

    Workflows are registered via ``@DBOS.workflow`` decorators at module
    import time. Whatever module imports register workflows in the DBOS
    runtime created here; the workflows run after launch() completes.

    Args:
        app: FastAPI app instance (required by lifespan signature, unused here).
        drain_timeout_sec: Seconds DBOS waits for in-flight workflows to
            checkpoint before stopping their executor. Default 20s leaves
            ~10s buffer under Railway's typical 30s SIGTERM-to-SIGKILL
            grace for the rest of ``DBOS.destroy()`` (which then joins
            each background thread up to 10s sequentially per
            ``dbos/_dbos.py:759``).

            IMPORTANT: ``DBOS.destroy(workflow_completion_timeout_sec=N)``
            does NOT kill workflows past the timeout. It calls
            ``executor.shutdown(wait=False, cancel_futures=True)`` which
            only cancels PENDING futures; RUNNING workflow threads keep
            executing until the OS SIGKILLs the container. Workflows
            still running when clients close will log "client closed"
            errors; DBOS resumes them from their last checkpoint on the
            next container start.
    """
    del app  # FastAPI lifespan signature requires this arg; DBOS init does not use it.
    config = build_dbos_config()
    try:
        DBOS(config=config)
        DBOS.launch()
    except Exception:
        # Launch can fail mid-way (DB unreachable, schema migration error)
        # after partial init — best-effort cleanup so we don't leak threads
        # or DB handles. timeout=0: don't drain, fail-fast.
        try:
            DBOS.destroy(workflow_completion_timeout_sec=0)
        except Exception:
            logger.exception(
                "DBOS.destroy() during failed-launch cleanup also raised"
            )
        raise
    logger.info(
        "DBOS launched (executor_id=%s)",
        os.environ.get("RAILWAY_REPLICA_ID") or "<unset; DBOS default>",
    )
    try:
        yield
    finally:
        DBOS.destroy(workflow_completion_timeout_sec=drain_timeout_sec)
        logger.info(
            "DBOS destroyed (drain_timeout=%ss)", drain_timeout_sec
        )
