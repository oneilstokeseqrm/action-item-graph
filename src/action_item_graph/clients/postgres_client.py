"""
Postgres (Neon) dual-write client for the Action Item Graph and Deal pipelines.

Mirrors Neo4j writes to Postgres using SQLAlchemy 2.0 async engine + asyncpg.
Neo4j remains the source of truth; Postgres is a read-optimized projection.
All writes are failure-isolated — a Postgres failure must never block Neo4j.

Tables written:
- action_items (UPSERT on graph_action_item_id)
- action_item_versions (INSERT)
- action_item_topics (UPSERT on action_item_topic_id)
- action_item_topic_versions (INSERT)
- action_item_topic_memberships (INSERT ON CONFLICT DO NOTHING)
- action_item_owners (UPSERT on owner_id)
- action_item_links (INSERT ON CONFLICT DO NOTHING)
- opportunities (UPSERT on graph_opportunity_id)
- deal_versions (INSERT)
- interaction_links (INSERT ON CONFLICT DO NOTHING)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ..models.action_item import ActionItem, ActionItemVersion
from ..models.entities import Owner
from ..models.topic import ActionItemTopic, ActionItemTopicVersion
from deal_graph.models.deal import Deal, DealStage, DealVersion, OntologyScores

logger = structlog.get_logger(__name__)

# Neo4j status → Postgres ActionItemStatus enum mapping.
# Neo4j uses: open, in_progress, completed, cancelled, deferred
# Postgres uses: pending, in_progress, completed, cancelled, deferred
_STATUS_MAP = {
    'open': 'pending',
    'in_progress': 'in_progress',
    'completed': 'completed',
    'cancelled': 'cancelled',
    'deferred': 'deferred',
}

# Columns on opportunities table that fire notify_forecast_job() on UPDATE.
# The Deal dual-write MUST NOT write to these columns.
_DEAL_TRIGGER_PROTECTED_COLUMNS = frozenset({
    'stage', 'amount', 'close_date', 'deal_status',
    'forecast_category', 'next_step', 'description', 'lost_reason',
})


def _map_status(neo4j_status: str) -> str:
    """Map Neo4j ActionItemStatus to Postgres enum value."""
    return _STATUS_MAP.get(neo4j_status, 'pending')


def _to_pg_uuid(val: UUID | str | None) -> str | None:
    """Convert UUID or string to plain string for Postgres, or None."""
    if val is None:
        return None
    return str(val)


def _to_pg_ts(val: datetime | str | None) -> datetime | None:
    """Ensure value is a datetime for asyncpg (which needs native types, not strings).

    If already a datetime, return as-is. If an ISO string, parse it.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    # Parse ISO string back to datetime for asyncpg compatibility
    return datetime.fromisoformat(val)


def _sanitize_url(url: str) -> str:
    """Remove URL query params that asyncpg does not understand.

    Neon pooler URLs include ``channel_binding=require`` and ``sslmode=require``
    which are libpq parameters. asyncpg rejects unknown connection params.
    SQLAlchemy's asyncpg dialect handles SSL via ``connect_args`` instead.
    """
    _STRIP_PARAMS = {'channel_binding', 'sslmode'}
    parsed = urlparse(url)
    if not parsed.query:
        return url
    params = parse_qs(parsed.query)
    filtered = {k: v for k, v in params.items() if k not in _STRIP_PARAMS}
    new_query = urlencode(filtered, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def _embedding_to_pgvector(embedding: list[float] | None) -> str | None:
    """Convert embedding list to pgvector literal string, e.g. '[0.1,0.2,...]'."""
    if embedding is None:
        return None
    return '[' + ','.join(str(f) for f in embedding) + ']'


def _ontology_to_jsonb(scores: OntologyScores) -> str | None:
    """Serialize OntologyScores to JSONB string with evidence text."""
    if not scores.scores:
        return None
    result = {}
    for dim_id, score in scores.scores.items():
        result[dim_id] = {
            'score': score,
            'confidence': scores.confidences.get(dim_id, 0.0),
            'evidence': scores.evidence.get(dim_id),
        }
    return json.dumps(result)


def _ontology_dim_params(scores: OntologyScores) -> dict[str, int | float | None]:
    """Flatten OntologyScores to individual dim_* column parameters."""
    from deal_graph.pipeline.merger import TRANSCRIPT_EXTRACTED_DIMENSIONS

    params: dict[str, int | float | None] = {}
    for dim_id in TRANSCRIPT_EXTRACTED_DIMENSIONS:
        params[f'dim_{dim_id}'] = scores.scores.get(dim_id)
        params[f'dim_{dim_id}_confidence'] = scores.confidences.get(dim_id)
    return params


class PostgresClient:
    """
    Async Postgres client for dual-write from the Action Item Graph pipeline.

    Uses SQLAlchemy 2.0 async engine with asyncpg for raw SQL execution.
    All methods are fire-and-forget safe — callers should catch exceptions
    to preserve Neo4j-first write semantics.
    """

    def __init__(self, database_url: str | None = None):
        """
        Initialize with a Neon/Postgres connection URL.

        Args:
            database_url: Postgres connection URL. Should use asyncpg driver
                          (postgresql+asyncpg://...). If the URL starts with
                          'postgres://' or 'postgresql://', it will be
                          converted to use asyncpg.
        """
        self._engine: AsyncEngine | None = None
        self._database_url = database_url

    async def connect(self, database_url: str | None = None) -> None:
        """
        Create the async engine. Idempotent — no-op if already connected.

        Args:
            database_url: Override the URL from __init__.
        """
        if self._engine is not None:
            return

        url = database_url or self._database_url
        if not url:
            raise ValueError('database_url is required')

        # Strip query params unsupported by asyncpg (e.g. channel_binding)
        url = _sanitize_url(url)

        # Normalise driver prefix for asyncpg
        if url.startswith('postgres://'):
            url = url.replace('postgres://', 'postgresql+asyncpg://', 1)
        elif url.startswith('postgresql://') and '+asyncpg' not in url:
            url = url.replace('postgresql://', 'postgresql+asyncpg://', 1)

        self._engine = create_async_engine(
            url,
            pool_size=5,
            max_overflow=5,
            pool_pre_ping=True,
            # Neon serverless can be slow on first connect
            pool_timeout=30,
            # Neon pooler (PgBouncer) doesn't support prepared statements.
            # ssl='require' replaces the stripped sslmode query param.
            connect_args={
                'prepared_statement_cache_size': 0,
                'ssl': 'require',
            },
        )
        logger.info('postgres_client.connected')

    async def close(self) -> None:
        """Dispose of the engine and connection pool."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            logger.info('postgres_client.closed')

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError('PostgresClient not connected — call connect() first')
        return self._engine

    async def verify_connectivity(self) -> bool:
        """Return True if we can execute a simple query."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text('SELECT 1'))
            return True
        except Exception:
            logger.exception('postgres_client.connectivity_check_failed')
            return False

    # =========================================================================
    # Action Item UPSERT
    # =========================================================================

    async def upsert_action_item(self, item: ActionItem) -> None:
        """
        UPSERT an ActionItem into the action_items table.

        Uses graph_action_item_id as the conflict target (unique index).
        Sets generated_by_ai=true and populates ai_suggestion_details JSONB.
        """
        sql = text("""
            INSERT INTO action_items (
                id, tenant_id, title, description, status, due_date,
                owner_id, generated_by_ai, ai_suggestion_details,
                workflow_metadata, created_at, updated_at,
                graph_action_item_id, account_id, owner_name, owner_type,
                is_user_owned, conversation_context, version_number,
                evolution_summary, source_interaction_id, source_user_id,
                embedding, embedding_current,
                valid_at, invalid_at, invalidated_by
            ) VALUES (
                :id, :tenant_id, :title, :description,
                CAST(:status AS "ActionItemStatus"), :due_date,
                :owner_id, true, :ai_suggestion_details,
                :workflow_metadata, :created_at, :updated_at,
                :graph_action_item_id, :account_id, :owner_name, :owner_type,
                :is_user_owned, :conversation_context, :version_number,
                :evolution_summary, :source_interaction_id, :source_user_id,
                CAST(:embedding AS vector), CAST(:embedding_current AS vector),
                :valid_at, :invalid_at, :invalidated_by
            )
            ON CONFLICT (graph_action_item_id) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                status = EXCLUDED.status,
                due_date = EXCLUDED.due_date,
                owner_name = EXCLUDED.owner_name,
                owner_type = EXCLUDED.owner_type,
                is_user_owned = EXCLUDED.is_user_owned,
                conversation_context = EXCLUDED.conversation_context,
                version_number = EXCLUDED.version_number,
                evolution_summary = EXCLUDED.evolution_summary,
                embedding_current = EXCLUDED.embedding_current,
                valid_at = EXCLUDED.valid_at,
                invalid_at = EXCLUDED.invalid_at,
                invalidated_by = EXCLUDED.invalidated_by,
                updated_at = EXCLUDED.updated_at
        """)

        status_val = item.status.value if hasattr(item.status, 'value') else item.status
        item_id = _to_pg_uuid(item.id)

        params: dict[str, Any] = {
            'id': item_id,
            'tenant_id': _to_pg_uuid(item.tenant_id),
            'title': item.summary,
            'description': item.action_item_text,
            'status': _map_status(status_val),
            'due_date': _to_pg_ts(item.due_date),
            'owner_id': _to_pg_uuid(item.pg_user_id),
            'ai_suggestion_details': json.dumps({
                'source': 'action_item_graph',
                'confidence': item.confidence,
            }),
            'workflow_metadata': json.dumps({
                'pipeline_version': '1.0',
            }),
            'created_at': _to_pg_ts(item.created_at),
            'updated_at': _to_pg_ts(item.last_updated_at),
            'graph_action_item_id': item_id,
            'account_id': _to_pg_uuid(item.account_id),
            'owner_name': item.owner,
            'owner_type': item.owner_type,
            'is_user_owned': item.is_user_owned,
            'conversation_context': item.conversation_context or None,
            'version_number': item.version,
            'evolution_summary': item.evolution_summary or None,
            'source_interaction_id': _to_pg_uuid(item.source_interaction_id),
            'source_user_id': item.user_id,
            'embedding': _embedding_to_pgvector(item.embedding),
            'embedding_current': _embedding_to_pgvector(item.embedding_current),
            'valid_at': _to_pg_ts(item.valid_at),
            'invalid_at': _to_pg_ts(item.invalid_at),
            'invalidated_by': _to_pg_uuid(item.invalidated_by),
        }

        async with self.engine.begin() as conn:
            await conn.execute(sql, params)

        logger.debug('postgres_client.upsert_action_item', action_item_id=item_id)

    # =========================================================================
    # Action Item Version INSERT
    # =========================================================================

    async def insert_action_item_version(self, version: ActionItemVersion) -> None:
        """
        INSERT an ActionItemVersion snapshot.

        The action_item_id FK must reference an existing action_items.id row —
        call upsert_action_item first.
        """
        sql = text("""
            INSERT INTO action_item_versions (
                id, tenant_id, action_item_id, version_number,
                action_item_text, summary, owner, status,
                due_date, change_summary, change_source_interaction_id,
                valid_from, valid_until, created_at
            ) VALUES (
                :id, :tenant_id, :action_item_id, :version_number,
                :action_item_text, :summary, :owner, :status,
                :due_date, :change_summary, :change_source_interaction_id,
                :valid_from, :valid_until, :created_at
            )
            ON CONFLICT (tenant_id, action_item_id, version_number) DO NOTHING
        """)

        status_val = version.status.value if hasattr(version.status, 'value') else version.status

        params: dict[str, Any] = {
            'id': _to_pg_uuid(version.id),
            'tenant_id': _to_pg_uuid(version.tenant_id),
            'action_item_id': _to_pg_uuid(version.action_item_id),
            'version_number': version.version,
            'action_item_text': version.action_item_text,
            'summary': version.summary,
            'owner': version.owner,
            'status': status_val,
            'due_date': _to_pg_ts(version.due_date),
            'change_summary': version.change_summary,
            'change_source_interaction_id': _to_pg_uuid(version.change_source_interaction_id),
            'valid_from': _to_pg_ts(version.valid_from),
            'valid_until': _to_pg_ts(version.valid_until),
            'created_at': _to_pg_ts(version.created_at),
        }

        async with self.engine.begin() as conn:
            await conn.execute(sql, params)

        logger.debug(
            'postgres_client.insert_action_item_version',
            action_item_id=str(version.action_item_id),
            version=version.version,
        )

    # =========================================================================
    # Topic UPSERT
    # =========================================================================

    async def upsert_topic(self, topic: ActionItemTopic) -> None:
        """
        UPSERT an ActionItemTopic into the action_item_topics table.

        Uses action_item_topic_id as the conflict target (unique index).
        """
        sql = text("""
            INSERT INTO action_item_topics (
                id, tenant_id, account_id, action_item_topic_id,
                name, canonical_name, summary,
                embedding, embedding_current,
                action_item_count, version_number,
                created_from_action_item_id,
                created_at, updated_at
            ) VALUES (
                :id, :tenant_id, :account_id, :action_item_topic_id,
                :name, :canonical_name, :summary,
                CAST(:embedding AS vector), CAST(:embedding_current AS vector),
                :action_item_count, :version_number,
                :created_from_action_item_id,
                :created_at, :updated_at
            )
            ON CONFLICT (action_item_topic_id) DO UPDATE SET
                name = EXCLUDED.name,
                summary = EXCLUDED.summary,
                embedding_current = EXCLUDED.embedding_current,
                action_item_count = EXCLUDED.action_item_count,
                version_number = EXCLUDED.version_number,
                updated_at = EXCLUDED.updated_at
        """)

        topic_id = _to_pg_uuid(topic.id)

        params: dict[str, Any] = {
            'id': topic_id,
            'tenant_id': _to_pg_uuid(topic.tenant_id),
            'account_id': _to_pg_uuid(topic.account_id),
            'action_item_topic_id': topic_id,
            'name': topic.name,
            'canonical_name': topic.canonical_name,
            'summary': topic.summary,
            'embedding': _embedding_to_pgvector(topic.embedding),
            'embedding_current': _embedding_to_pgvector(topic.embedding_current),
            'action_item_count': topic.action_item_count,
            'version_number': topic.version,
            'created_from_action_item_id': _to_pg_uuid(topic.created_from_action_item_id),
            'created_at': _to_pg_ts(topic.created_at),
            'updated_at': _to_pg_ts(topic.updated_at),
        }

        async with self.engine.begin() as conn:
            await conn.execute(sql, params)

        logger.debug('postgres_client.upsert_topic', topic_id=topic_id)

    # =========================================================================
    # Topic Version INSERT
    # =========================================================================

    async def insert_topic_version(self, version: ActionItemTopicVersion) -> None:
        """INSERT an ActionItemTopicVersion snapshot."""
        sql = text("""
            INSERT INTO action_item_topic_versions (
                id, tenant_id, topic_id, version_number,
                name, summary, embedding_snapshot,
                changed_by_action_item_id, created_at
            ) VALUES (
                :id, :tenant_id, :topic_id, :version_number,
                :name, :summary, CAST(:embedding_snapshot AS vector),
                :changed_by_action_item_id, :created_at
            )
            ON CONFLICT (tenant_id, topic_id, version_number) DO NOTHING
        """)

        params: dict[str, Any] = {
            'id': _to_pg_uuid(version.id),
            'tenant_id': _to_pg_uuid(version.tenant_id),
            'topic_id': _to_pg_uuid(version.topic_id),
            'version_number': version.version_number,
            'name': version.name,
            'summary': version.summary,
            'embedding_snapshot': _embedding_to_pgvector(version.embedding_snapshot),
            'changed_by_action_item_id': _to_pg_uuid(version.changed_by_action_item_id),
            'created_at': _to_pg_ts(version.created_at),
        }

        async with self.engine.begin() as conn:
            await conn.execute(sql, params)

        logger.debug(
            'postgres_client.insert_topic_version',
            topic_id=str(version.topic_id),
            version=version.version_number,
        )

    # =========================================================================
    # Topic Membership (ActionItem ↔ Topic link)
    # =========================================================================

    async def upsert_topic_membership(
        self,
        tenant_id: UUID,
        action_item_id: UUID,
        topic_id: UUID,
        confidence: float = 1.0,
        method: str = 'extracted',
    ) -> None:
        """
        Link an ActionItem to an ActionItemTopic.

        Uses (tenant_id, action_item_id, topic_id) unique constraint.
        """
        sql = text("""
            INSERT INTO action_item_topic_memberships (
                tenant_id, action_item_id, topic_id, confidence, method
            ) VALUES (
                :tenant_id, :action_item_id, :topic_id, :confidence, :method
            )
            ON CONFLICT (tenant_id, action_item_id, topic_id) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                method = EXCLUDED.method
        """)

        params: dict[str, Any] = {
            'tenant_id': _to_pg_uuid(tenant_id),
            'action_item_id': _to_pg_uuid(action_item_id),
            'topic_id': _to_pg_uuid(topic_id),
            'confidence': confidence,
            'method': method,
        }

        async with self.engine.begin() as conn:
            await conn.execute(sql, params)

        logger.debug(
            'postgres_client.upsert_topic_membership',
            action_item_id=str(action_item_id),
            topic_id=str(topic_id),
        )

    # =========================================================================
    # Owner UPSERT
    # =========================================================================

    async def upsert_owner(self, owner: Owner) -> None:
        """
        UPSERT an Owner into the action_item_owners table.

        Uses owner_id as the conflict target (unique index).
        """
        sql = text("""
            INSERT INTO action_item_owners (
                id, tenant_id, owner_id, canonical_name,
                aliases, contact_id, user_id, created_at
            ) VALUES (
                :id, :tenant_id, :owner_id, :canonical_name,
                :aliases, :contact_id, :user_id, :created_at
            )
            ON CONFLICT (owner_id) DO UPDATE SET
                canonical_name = EXCLUDED.canonical_name,
                aliases = EXCLUDED.aliases,
                contact_id = EXCLUDED.contact_id,
                user_id = EXCLUDED.user_id
        """)

        owner_id = _to_pg_uuid(owner.id)

        params: dict[str, Any] = {
            'id': owner_id,
            'tenant_id': _to_pg_uuid(owner.tenant_id),
            'owner_id': owner_id,
            'canonical_name': owner.canonical_name,
            'aliases': json.dumps(owner.aliases),
            'contact_id': _to_pg_uuid(owner.contact_id),
            'user_id': owner.user_id,
            'created_at': _to_pg_ts(owner.created_at),
        }

        async with self.engine.begin() as conn:
            await conn.execute(sql, params)

        logger.debug('postgres_client.upsert_owner', owner_id=owner_id)

    # =========================================================================
    # Action Item Links (polymorphic entity linking)
    # =========================================================================

    async def link_action_item_to_entity(
        self,
        tenant_id: UUID,
        action_item_id: UUID,
        entity_type: str,
        entity_id: UUID | str,
    ) -> None:
        """
        Insert a row into action_item_links for provenance tracking.

        entity_type must be one of: account, contact, opportunity, meeting, interaction.
        Uses a uniqueness check to avoid duplicates.

        Args:
            tenant_id: Tenant UUID
            action_item_id: ActionItem UUID (Postgres id)
            entity_type: Polymorphic entity type
            entity_id: UUID of the linked entity
        """
        sql = text("""
            INSERT INTO action_item_links (
                id, tenant_id, action_item_id, entity_type, entity_id
            ) VALUES (
                gen_random_uuid(), :tenant_id, :action_item_id,
                CAST(:entity_type AS "ActionItemLinkEntityType"),
                :entity_id
            )
            ON CONFLICT DO NOTHING
        """)

        params: dict[str, Any] = {
            'tenant_id': _to_pg_uuid(tenant_id),
            'action_item_id': _to_pg_uuid(action_item_id),
            'entity_type': entity_type,
            'entity_id': _to_pg_uuid(entity_id),
        }

        async with self.engine.begin() as conn:
            await conn.execute(sql, params)

        logger.debug(
            'postgres_client.link_action_item_to_entity',
            action_item_id=str(action_item_id),
            entity_type=entity_type,
            entity_id=str(entity_id),
        )

    # =========================================================================
    # Batch / Convenience
    # =========================================================================

    async def persist_action_item_full(
        self,
        item: ActionItem,
        owner: Owner | None = None,
        topic: ActionItemTopic | None = None,
        topic_confidence: float = 1.0,
        topic_method: str = 'extracted',
        interaction_id: UUID | None = None,
    ) -> None:
        """
        Convenience method to persist a full action item extraction result.

        Writes the action item, optionally the owner, topic, membership link,
        and entity links (account + interaction) in one call.

        All sub-writes are independent; individual failures are logged but
        do not abort the remaining writes.
        """
        # 1. Action item
        await self.upsert_action_item(item)

        # 2. Owner
        if owner is not None:
            try:
                await self.upsert_owner(owner)
            except Exception:
                logger.exception('postgres_client.persist_owner_failed', owner_id=str(owner.id))

        # 3. Topic + membership
        if topic is not None:
            try:
                await self.upsert_topic(topic)
            except Exception:
                logger.exception('postgres_client.persist_topic_failed', topic_id=str(topic.id))
            else:
                try:
                    await self.upsert_topic_membership(
                        tenant_id=item.tenant_id,
                        action_item_id=item.id,
                        topic_id=topic.id,
                        confidence=topic_confidence,
                        method=topic_method,
                    )
                except Exception:
                    logger.exception(
                        'postgres_client.persist_membership_failed',
                        action_item_id=str(item.id),
                        topic_id=str(topic.id),
                    )

        # 4. Entity links
        if item.account_id:
            try:
                await self.link_action_item_to_entity(
                    tenant_id=item.tenant_id,
                    action_item_id=item.id,
                    entity_type='account',
                    entity_id=item.account_id,
                )
            except Exception:
                logger.exception(
                    'postgres_client.link_account_failed',
                    action_item_id=str(item.id),
                )

        if interaction_id is not None:
            try:
                await self.link_action_item_to_entity(
                    tenant_id=item.tenant_id,
                    action_item_id=item.id,
                    entity_type='interaction',
                    entity_id=interaction_id,
                )
            except Exception:
                logger.exception(
                    'postgres_client.link_interaction_failed',
                    action_item_id=str(item.id),
                )

    # =========================================================================
    # Deal UPSERT
    # =========================================================================

    async def upsert_deal(
        self,
        deal: Deal,
        source_user_id: str | None = None,
    ) -> UUID:
        """Upsert a Deal into the opportunities table.

        Conflict resolution on graph_opportunity_id (unique index).
        Does NOT write to trigger-protected columns (stage, amount, etc.).

        Returns:
            The Postgres-side ``opportunities.id`` (PK) for FK references.
        """
        dim_params = _ontology_dim_params(deal.ontology_scores)

        dim_col_names = sorted(dim_params.keys())
        dim_insert_cols = ', '.join(f'"{c}"' for c in dim_col_names)
        dim_insert_vals = ', '.join(f':{c}' for c in dim_col_names)
        dim_update_set = ', '.join(f'"{c}" = :{c}' for c in dim_col_names)

        sql = f"""
        INSERT INTO opportunities (
            id, graph_opportunity_id, tenant_id, account_id, opportunity_name, deal_ref,
            currency, actual_close_date,
            latest_ai_summary, ai_evolution_summary,
            meddic_metrics, meddic_metrics_confidence,
            meddic_economic_buyer, meddic_economic_buyer_confidence,
            meddic_decision_criteria, meddic_decision_criteria_confidence,
            meddic_decision_process, meddic_decision_process_confidence,
            meddic_identified_pain, meddic_identified_pain_confidence,
            meddic_champion, meddic_champion_confidence,
            meddic_completeness, meddic_paper_process, meddic_competition,
            {dim_insert_cols},
            ontology_scores_json, ontology_completeness, ontology_version,
            extraction_embedding, extraction_embedding_current,
            extraction_confidence, extraction_version, source_interaction_id,
            qualification_status, source_user_id,
            ai_workflow_metadata, updated_at
        ) VALUES (
            gen_random_uuid(), :graph_opportunity_id, :tenant_id, :account_id, :opportunity_name, :deal_ref,
            :currency, :actual_close_date,
            :latest_ai_summary, :ai_evolution_summary,
            :meddic_metrics, :meddic_metrics_confidence,
            :meddic_economic_buyer, :meddic_economic_buyer_confidence,
            :meddic_decision_criteria, :meddic_decision_criteria_confidence,
            :meddic_decision_process, :meddic_decision_process_confidence,
            :meddic_identified_pain, :meddic_identified_pain_confidence,
            :meddic_champion, :meddic_champion_confidence,
            :meddic_completeness, :meddic_paper_process, :meddic_competition,
            {dim_insert_vals},
            :ontology_scores_json, :ontology_completeness, :ontology_version,
            :extraction_embedding, :extraction_embedding_current,
            :extraction_confidence, :extraction_version, :source_interaction_id,
            :qualification_status, :source_user_id,
            :ai_workflow_metadata, now()
        )
        ON CONFLICT (graph_opportunity_id) DO UPDATE SET
            deal_ref = :deal_ref,
            latest_ai_summary = :latest_ai_summary,
            ai_evolution_summary = :ai_evolution_summary,
            meddic_metrics = :meddic_metrics,
            meddic_metrics_confidence = :meddic_metrics_confidence,
            meddic_economic_buyer = :meddic_economic_buyer,
            meddic_economic_buyer_confidence = :meddic_economic_buyer_confidence,
            meddic_decision_criteria = :meddic_decision_criteria,
            meddic_decision_criteria_confidence = :meddic_decision_criteria_confidence,
            meddic_decision_process = :meddic_decision_process,
            meddic_decision_process_confidence = :meddic_decision_process_confidence,
            meddic_identified_pain = :meddic_identified_pain,
            meddic_identified_pain_confidence = :meddic_identified_pain_confidence,
            meddic_champion = :meddic_champion,
            meddic_champion_confidence = :meddic_champion_confidence,
            meddic_completeness = :meddic_completeness,
            meddic_paper_process = :meddic_paper_process,
            meddic_competition = :meddic_competition,
            {dim_update_set},
            ontology_scores_json = :ontology_scores_json,
            ontology_completeness = :ontology_completeness,
            ontology_version = :ontology_version,
            extraction_embedding = :extraction_embedding,
            extraction_embedding_current = :extraction_embedding_current,
            extraction_confidence = :extraction_confidence,
            extraction_version = :extraction_version,
            source_interaction_id = :source_interaction_id,
            qualification_status = :qualification_status,
            source_user_id = :source_user_id,
            ai_workflow_metadata = :ai_workflow_metadata,
            updated_at = now()
        RETURNING id
        """

        params = {
            'graph_opportunity_id': _to_pg_uuid(deal.opportunity_id),
            'tenant_id': _to_pg_uuid(deal.tenant_id),
            'account_id': _to_pg_uuid(deal.account_id) if deal.account_id else None,
            'opportunity_name': deal.name or None,
            'currency': deal.currency or 'USD',
            'actual_close_date': deal.closed_at,
            'deal_ref': deal.deal_ref,
            'latest_ai_summary': deal.opportunity_summary or None,
            'ai_evolution_summary': deal.evolution_summary or None,
            # MEDDIC
            'meddic_metrics': deal.meddic.metrics,
            'meddic_metrics_confidence': deal.meddic.metrics_confidence,
            'meddic_economic_buyer': deal.meddic.economic_buyer,
            'meddic_economic_buyer_confidence': deal.meddic.economic_buyer_confidence,
            'meddic_decision_criteria': deal.meddic.decision_criteria,
            'meddic_decision_criteria_confidence': deal.meddic.decision_criteria_confidence,
            'meddic_decision_process': deal.meddic.decision_process,
            'meddic_decision_process_confidence': deal.meddic.decision_process_confidence,
            'meddic_identified_pain': deal.meddic.identified_pain,
            'meddic_identified_pain_confidence': deal.meddic.identified_pain_confidence,
            'meddic_champion': deal.meddic.champion,
            'meddic_champion_confidence': deal.meddic.champion_confidence,
            'meddic_completeness': deal.meddic.completeness_score,
            'meddic_paper_process': deal.meddic.paper_process,
            'meddic_competition': deal.meddic.competition,
            # Ontology
            'ontology_scores_json': _ontology_to_jsonb(deal.ontology_scores),
            'ontology_completeness': deal.ontology_scores.completeness_score,
            'ontology_version': deal.ontology_version,
            # Embeddings
            'extraction_embedding': _embedding_to_pgvector(deal.embedding),
            'extraction_embedding_current': _embedding_to_pgvector(deal.embedding_current),
            # Extraction metadata
            'extraction_confidence': deal.confidence,
            'extraction_version': deal.version,
            'source_interaction_id': _to_pg_uuid(deal.source_interaction_id),
            # Future slots
            'qualification_status': deal.qualification_status,
            'source_user_id': source_user_id,
            # Workflow metadata
            'ai_workflow_metadata': json.dumps({
                'source': 'deal_graph',
                'deal_name': deal.name,
                'stage_assessment': deal.stage.value if isinstance(deal.stage, DealStage) else deal.stage,
                'amount_estimate': deal.amount,
                'currency': deal.currency,
                'deal_ref': deal.deal_ref,
                'expected_close_date': deal.expected_close_date.isoformat() if deal.expected_close_date else None,
                'closed_at': deal.closed_at.isoformat() if deal.closed_at else None,
            }),
            # Dim columns
            **dim_params,
        }

        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql), params)
            row = result.fetchone()
            pg_id: UUID = row[0]  # type: ignore[index]

        logger.info(
            'postgres_client.upsert_deal',
            opportunity_id=str(deal.opportunity_id),
            tenant_id=str(deal.tenant_id),
            pg_id=str(pg_id),
        )
        return pg_id

    # =========================================================================
    # Deal Version INSERT
    # =========================================================================

    async def insert_deal_version(
        self,
        version: DealVersion,
        pg_opportunity_id: UUID | None = None,
    ) -> None:
        """Insert a DealVersion snapshot into deal_versions table.

        Args:
            version: DealVersion model
            pg_opportunity_id: Postgres-side opportunities.id (PK) for the FK.
                If None, falls back to version.deal_opportunity_id (graph ID).
        """
        sql = """
        INSERT INTO deal_versions (
            id, tenant_id, opportunity_id, version_number,
            name, stage, amount,
            opportunity_summary, evolution_summary,
            meddic_metrics, meddic_economic_buyer, meddic_decision_criteria,
            meddic_decision_process, meddic_identified_pain, meddic_champion,
            meddic_completeness,
            ontology_scores_json, ontology_completeness, ontology_version,
            change_summary, changed_fields, change_source_interaction_id,
            created_at, valid_from, valid_until
        ) VALUES (
            :id, :tenant_id, :opportunity_id, :version_number,
            :name, :stage, :amount,
            :opportunity_summary, :evolution_summary,
            :meddic_metrics, :meddic_economic_buyer, :meddic_decision_criteria,
            :meddic_decision_process, :meddic_identified_pain, :meddic_champion,
            :meddic_completeness,
            :ontology_scores_json, :ontology_completeness, :ontology_version,
            :change_summary, :changed_fields, :change_source_interaction_id,
            :created_at, :valid_from, :valid_until
        )
        ON CONFLICT (tenant_id, opportunity_id, version_number) DO NOTHING
        """

        params = {
            'id': _to_pg_uuid(version.version_id),
            'tenant_id': _to_pg_uuid(version.tenant_id),
            'opportunity_id': _to_pg_uuid(pg_opportunity_id) if pg_opportunity_id else _to_pg_uuid(version.deal_opportunity_id),
            'version_number': version.version,
            'name': version.name,
            'stage': version.stage.value if isinstance(version.stage, DealStage) else version.stage,
            'amount': version.amount,
            'opportunity_summary': version.opportunity_summary,
            'evolution_summary': version.evolution_summary,
            'meddic_metrics': version.meddic_metrics,
            'meddic_economic_buyer': version.meddic_economic_buyer,
            'meddic_decision_criteria': version.meddic_decision_criteria,
            'meddic_decision_process': version.meddic_decision_process,
            'meddic_identified_pain': version.meddic_identified_pain,
            'meddic_champion': version.meddic_champion,
            'meddic_completeness': version.meddic_completeness,
            'ontology_scores_json': version.ontology_scores_json,
            'ontology_completeness': version.ontology_completeness,
            'ontology_version': version.ontology_version,
            'change_summary': version.change_summary,
            'changed_fields': json.dumps(version.changed_fields),
            'change_source_interaction_id': _to_pg_uuid(version.change_source_interaction_id),
            'created_at': _to_pg_ts(version.created_at),
            'valid_from': _to_pg_ts(version.valid_from),
            'valid_until': _to_pg_ts(version.valid_until),
        }

        async with self.engine.begin() as conn:
            await conn.execute(text(sql), params)

        logger.info(
            'postgres_client.insert_deal_version',
            version_id=str(version.version_id),
            opportunity_id=str(version.deal_opportunity_id),
        )

    # =========================================================================
    # Deal Entity Links
    # =========================================================================

    async def link_deal_to_interaction(
        self,
        tenant_id: UUID,
        interaction_id: UUID,
        opportunity_id: UUID,
    ) -> None:
        """Link an interaction to an opportunity via interaction_links table."""
        sql = """
        INSERT INTO interaction_links (tenant_id, interaction_id, entity_type, entity_id)
        VALUES (:tenant_id, :interaction_id, 'opportunity', :entity_id)
        ON CONFLICT DO NOTHING
        """
        async with self.engine.begin() as conn:
            await conn.execute(text(sql), {
                'tenant_id': _to_pg_uuid(tenant_id),
                'interaction_id': _to_pg_uuid(interaction_id),
                'entity_id': _to_pg_uuid(opportunity_id),
            })

    # =========================================================================
    # Deal Batch / Convenience
    # =========================================================================

    async def persist_deal_full(
        self,
        deal: Deal,
        version: DealVersion | None = None,
        interaction_id: UUID | None = None,
        source_user_id: str | None = None,
    ) -> None:
        """Write a Deal + optional DealVersion to Postgres with failure isolation."""
        if source_user_id is not None:
            pg_id = await self.upsert_deal(deal, source_user_id=source_user_id)
        else:
            pg_id = await self.upsert_deal(deal)

        if version is not None:
            try:
                await self.insert_deal_version(version, pg_opportunity_id=pg_id)
            except Exception:
                logger.exception(
                    'postgres_client.persist_deal_version_failed',
                    opportunity_id=str(deal.opportunity_id),
                )

        if interaction_id is not None:
            try:
                await self.link_deal_to_interaction(
                    tenant_id=deal.tenant_id,
                    interaction_id=interaction_id,
                    opportunity_id=deal.opportunity_id,
                )
            except Exception:
                logger.exception(
                    'postgres_client.link_deal_interaction_failed',
                    opportunity_id=str(deal.opportunity_id),
                    interaction_id=str(interaction_id),
                )
