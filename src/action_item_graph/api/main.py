"""FastAPI application for the action-item-graph Railway service."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.clients.postgres_client import PostgresClient
from action_item_graph.dbos_runtime import dbos_lifespan
from action_item_graph.workflows import WorkflowClients, register_clients
from deal_graph.clients.neo4j_client import DealNeo4jClient

from .config import get_settings
from .routes.health import router as health_router
from .routes.process import router as process_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize persistent clients at startup, clean up at shutdown.

    Composition order matters: app-state clients are connected FIRST so DBOS
    workflows (including recovered workflows that DBOS picks up at launch)
    can reach them via ``app.state``. ``dbos_lifespan`` is therefore the
    INNER context manager; its ``finally`` clause drains in-flight workflows
    BEFORE we close the clients those workflows depend on.

    Codex review of Phase A (2026-05-20) caught the original outer-DBOS
    ordering — it was a latent startup race against recovered workflows and
    a real shutdown sequencing bug where ``DBOS.destroy()`` was running
    after Neo4j/Postgres had already been closed.
    """
    settings = get_settings()

    logger.info("lifespan.startup", neo4j_uri=settings.NEO4J_URI)

    # Start each client as None so the teardown loop can skip any that never
    # opened (e.g., if Neo4j.connect() raises before DealNeo4jClient runs).
    neo4j: Neo4jClient | None = None
    deal_neo4j: DealNeo4jClient | None = None
    postgres: PostgresClient | None = None

    try:
        # Neo4j — action item pipeline
        neo4j = Neo4jClient(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
        await neo4j.connect()
        await neo4j.setup_schema()

        # Neo4j — deal pipeline (same DB, different client for schema ownership)
        deal_neo4j = DealNeo4jClient(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
        await deal_neo4j.connect()
        await deal_neo4j.setup_schema()

        # OpenAI — shared across both pipelines
        openai = OpenAIClient(api_key=settings.OPENAI_API_KEY)

        # Postgres (Neon) — dual-write projection (optional, failure-isolated)
        if settings.NEON_DATABASE_URL:
            pg = PostgresClient(settings.NEON_DATABASE_URL)
            await pg.connect()
            if await pg.verify_connectivity():
                postgres = pg
                logger.info("lifespan.postgres_ready")
            else:
                logger.warning("lifespan.postgres_connectivity_failed")
                await pg.close()

        # Store on app.state BEFORE launching DBOS so recovered workflows can reach them
        app.state.neo4j = neo4j
        app.state.deal_neo4j = deal_neo4j
        app.state.openai = openai
        app.state.postgres = postgres

        # Register clients in the DBOS workflows module-level registry BEFORE
        # DBOS.launch() so recovery threads picking up in-flight workflows can
        # resolve clients via ``get_clients()``. Step functions cannot reach
        # ``app.state`` (they're not bound to a request); the registry is the
        # equivalent of LTF's ``get_async_session`` pattern.
        register_clients(WorkflowClients(
            neo4j=neo4j,
            deal_neo4j=deal_neo4j,
            openai=openai,
            postgres=postgres,
        ))

        # DBOS launches here, AFTER app.state + workflow registry populated.
        # Recovery threads pick up any in-flight workflows from the previous
        # container and find clients ready via get_clients().
        async with dbos_lifespan(app):
            logger.info("lifespan.ready")
            try:
                yield
            finally:
                logger.info("lifespan.shutdown")
                # dbos_lifespan's finally clause runs as we exit this async with,
                # draining in-flight workflows BEFORE the outer finally closes
                # the clients those workflows depend on.
    finally:
        logger.info("lifespan.clients_closing")
        # Close each client independently so one failing close doesn't skip
        # the rest. Codex Phase A review (2026-05-20) flagged the original
        # fail-fast sequence as a cleanup robustness gap.
        for label, client in (
            ("neo4j", neo4j),
            ("deal_neo4j", deal_neo4j),
            ("postgres", postgres),
        ):
            if client is None:
                continue
            try:
                await client.close()
            except Exception:
                logger.exception("lifespan.client_close_failed", client=label)


app = FastAPI(
    title="action-item-graph",
    description="EnvelopeV1 event consumer — extracts action items and deals from transcripts and emails",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(process_router)
