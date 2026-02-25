"""FastAPI application for the action-item-graph Railway service."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.clients.postgres_client import PostgresClient
from deal_graph.clients.neo4j_client import DealNeo4jClient

from .config import get_settings
from .routes.health import router as health_router
from .routes.process import router as process_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize persistent clients at startup, clean up at shutdown."""
    settings = get_settings()

    logger.info("lifespan.startup", neo4j_uri=settings.NEO4J_URI)

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
    postgres: PostgresClient | None = None
    if settings.NEON_DATABASE_URL:
        pg = PostgresClient(settings.NEON_DATABASE_URL)
        await pg.connect()
        if await pg.verify_connectivity():
            postgres = pg
            logger.info("lifespan.postgres_ready")
        else:
            logger.warning("lifespan.postgres_connectivity_failed")
            await pg.close()

    # Store on app.state for request handlers
    app.state.neo4j = neo4j
    app.state.deal_neo4j = deal_neo4j
    app.state.openai = openai
    app.state.postgres = postgres

    logger.info("lifespan.ready")
    yield

    # Shutdown
    logger.info("lifespan.shutdown")
    await neo4j.close()
    await deal_neo4j.close()
    if postgres is not None:
        await postgres.close()


app = FastAPI(
    title="action-item-graph",
    description="EnvelopeV1 event consumer — extracts action items and deals from transcripts and emails",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(process_router)
