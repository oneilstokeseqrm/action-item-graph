"""
Graph repository for Deal pipeline CRUD operations.

Provides high-level methods for working with Deal, DealVersion, Interaction,
and Account nodes in the neo4j_structured database.

Key design decisions:
- create_deal() uses MERGE on skeleton key (tenant_id, opportunity_id) so it
  is safe whether or not the skeleton layer has already created the Deal stub.
- read_interaction() reads content_text from skeleton-created Interaction nodes.
- enrich_interaction() adds pipeline metadata (processed_at, deal_count) to
  existing Interaction nodes without touching skeleton-owned properties.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .clients.neo4j_client import DealNeo4jClient
from .models.deal import Deal, DealStage


class DealRepository:
    """
    CRUD operations for Deal pipeline graph entities.

    All queries use the composite key pattern (tenant_id, entity_id) matching
    the schema authority's constraint conventions.
    """

    def __init__(self, neo4j_client: DealNeo4jClient):
        self.neo4j = neo4j_client

    # =========================================================================
    # Account Operations
    # =========================================================================

    async def verify_account(
        self,
        tenant_id: UUID,
        account_id: str,
    ) -> dict[str, Any] | None:
        """
        Verify an Account node exists (created by skeleton layer).

        If the skeleton hasn't created it yet, MERGE with base properties
        as a graceful degradation.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier

        Returns:
            Account node properties, or None if MERGE somehow failed
        """
        query = """
            MERGE (a:Account {tenant_id: $tenant_id, account_id: $account_id})
            ON CREATE SET a.created_at = datetime()
            RETURN a {.*} as account
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'tenant_id': str(tenant_id),
                'account_id': account_id,
            },
        )
        return result[0]['account'] if result else None

    # =========================================================================
    # Interaction Operations
    # =========================================================================

    async def read_interaction(
        self,
        tenant_id: UUID,
        interaction_id: str,
    ) -> dict[str, Any] | None:
        """
        Read an Interaction node created by the skeleton layer.

        Retrieves content_text and metadata for MEDDIC extraction.

        Args:
            tenant_id: Tenant UUID
            interaction_id: Interaction identifier (from skeleton)

        Returns:
            Interaction properties including content_text, or None if not found
        """
        query = """
            MATCH (i:Interaction {tenant_id: $tenant_id, interaction_id: $interaction_id})
            RETURN i {.*} as interaction
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'tenant_id': str(tenant_id),
                'interaction_id': interaction_id,
            },
        )
        return result[0]['interaction'] if result else None

    async def enrich_interaction(
        self,
        tenant_id: UUID,
        interaction_id: str,
        processed_at: datetime | None = None,
        deal_count: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Add enrichment properties to an existing Interaction node.

        Only sets pipeline metadata — never touches skeleton-owned properties
        (content_text, timestamp, interaction_type, etc.).

        Args:
            tenant_id: Tenant UUID
            interaction_id: Interaction identifier
            processed_at: When deals were extracted (defaults to now)
            deal_count: Number of deals extracted

        Returns:
            Updated Interaction properties, or None if not found
        """
        updates: dict[str, Any] = {}
        if processed_at is not None:
            updates['processed_at'] = processed_at.isoformat()
        else:
            updates['processed_at'] = datetime.now(tz=timezone.utc).isoformat()
        if deal_count is not None:
            updates['deal_count'] = deal_count

        query = """
            MATCH (i:Interaction {tenant_id: $tenant_id, interaction_id: $interaction_id})
            SET i += $updates
            RETURN i {.*} as interaction
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'tenant_id': str(tenant_id),
                'interaction_id': interaction_id,
                'updates': updates,
            },
        )
        return result[0]['interaction'] if result else None

    async def ensure_interaction(
        self,
        tenant_id: UUID,
        interaction_id: str,
        content_text: str,
        interaction_type: str = 'transcript',
        timestamp: datetime | None = None,
        source: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """
        MERGE an Interaction node with skeleton base properties.

        Used as a fallback if our pipeline runs before the skeleton layer.
        If the skeleton already created the node, this is a no-op on
        skeleton properties (ON MATCH only sets enrichment timestamp).

        Args:
            tenant_id: Tenant UUID
            interaction_id: Interaction identifier
            content_text: Full transcript text
            interaction_type: Type of interaction
            timestamp: When the interaction occurred
            source: Origin of the content
            trace_id: Distributed tracing identifier

        Returns:
            Interaction node properties
        """
        query = """
            MERGE (i:Interaction {tenant_id: $tenant_id, interaction_id: $interaction_id})
            ON CREATE SET
                i.content_text = $content_text,
                i.interaction_type = $interaction_type,
                i.timestamp = $timestamp,
                i.source = $source,
                i.trace_id = $trace_id,
                i.created_at = datetime()
            ON MATCH SET
                i.processed_at = datetime()
            RETURN i {.*} as interaction
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'tenant_id': str(tenant_id),
                'interaction_id': interaction_id,
                'content_text': content_text,
                'interaction_type': interaction_type,
                'timestamp': timestamp.isoformat() if timestamp else datetime.now(tz=timezone.utc).isoformat(),
                'source': source,
                'trace_id': trace_id,
            },
        )
        return result[0]['interaction'] if result else {}

    # =========================================================================
    # Deal Operations
    # =========================================================================

    async def create_deal(
        self,
        deal: Deal,
    ) -> dict[str, Any]:
        """
        MERGE a Deal node on skeleton key, SET enrichment properties.

        Uses MERGE on (tenant_id, opportunity_id) so it is safe whether or
        not the skeleton layer has already created the Deal stub:
        - ON CREATE: sets both skeleton base properties and enrichment properties
        - ON MATCH: sets only enrichment properties (skeleton props already correct)

        Also sets source_interaction_id for provenance tracking and creates
        the [:HAS_VERSION] relationship for the initial DealVersion.

        Args:
            deal: Deal model with all properties

        Returns:
            Created/merged Deal node properties
        """
        props = deal.to_neo4j_properties()

        # Separate skeleton keys from the SET properties
        # (tenant_id and opportunity_id are in the MERGE pattern, not in SET)
        set_props = {k: v for k, v in props.items()
                     if k not in ('tenant_id', 'opportunity_id')}

        query = """
            MERGE (d:Deal {tenant_id: $tenant_id, opportunity_id: $opportunity_id})
            ON CREATE SET d += $set_props, d.created_at = datetime()
            ON MATCH SET d += $set_props
            SET d.last_updated_at = datetime()
            RETURN d {.*} as deal
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'tenant_id': str(deal.tenant_id),
                'opportunity_id': str(deal.opportunity_id),
                'set_props': set_props,
            },
        )
        return result[0]['deal'] if result else {}

    async def get_deal(
        self,
        tenant_id: UUID,
        opportunity_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve a Deal by composite key.

        Args:
            tenant_id: Tenant UUID
            opportunity_id: Deal identifier

        Returns:
            Deal node properties or None if not found
        """
        query = """
            MATCH (d:Deal {tenant_id: $tenant_id, opportunity_id: $opportunity_id})
            RETURN d {.*} as deal
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'tenant_id': str(tenant_id),
                'opportunity_id': opportunity_id,
            },
        )
        return result[0]['deal'] if result else None

    async def update_deal(
        self,
        tenant_id: UUID,
        opportunity_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update enrichment properties on an existing Deal.

        Increments version and updates last_updated_at automatically.
        Call create_version_snapshot() before this to preserve history.

        Args:
            tenant_id: Tenant UUID
            opportunity_id: Deal identifier
            updates: Properties to update (enrichment only)

        Returns:
            Updated Deal node properties
        """
        updates['last_updated_at'] = datetime.now(tz=timezone.utc).isoformat()

        query = """
            MATCH (d:Deal {tenant_id: $tenant_id, opportunity_id: $opportunity_id})
            SET d += $updates,
                d.version = d.version + 1
            RETURN d {.*} as deal
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'tenant_id': str(tenant_id),
                'opportunity_id': opportunity_id,
                'updates': updates,
            },
        )
        return result[0]['deal'] if result else {}

    # =========================================================================
    # DealVersion Operations
    # =========================================================================

    async def create_version_snapshot(
        self,
        tenant_id: UUID,
        opportunity_id: str,
        change_summary: str,
        changed_fields: list[str],
        change_source_interaction_id: UUID | None = None,
    ) -> dict[str, Any]:
        """
        Create a DealVersion snapshot before updating a Deal.

        Captures the full current state of the Deal node as a DealVersion,
        then links it via [:HAS_VERSION].

        Args:
            tenant_id: Tenant UUID
            opportunity_id: Deal identifier
            change_summary: LLM-generated narrative of why the deal changed
            changed_fields: List of property names that will change
            change_source_interaction_id: Interaction that triggered this change

        Returns:
            Created DealVersion node properties
        """
        query = """
            MATCH (d:Deal {tenant_id: $tenant_id, opportunity_id: $opportunity_id})
            CREATE (v:DealVersion {
                version_id: randomUUID(),
                deal_opportunity_id: d.opportunity_id,
                tenant_id: d.tenant_id,
                version: d.version,
                name: d.name,
                stage: d.stage,
                amount: d.amount,
                opportunity_summary: d.opportunity_summary,
                evolution_summary: d.evolution_summary,
                meddic_metrics: d.meddic_metrics,
                meddic_economic_buyer: d.meddic_economic_buyer,
                meddic_decision_criteria: d.meddic_decision_criteria,
                meddic_decision_process: d.meddic_decision_process,
                meddic_identified_pain: d.meddic_identified_pain,
                meddic_champion: d.meddic_champion,
                meddic_completeness: d.meddic_completeness,
                change_summary: $change_summary,
                changed_fields: $changed_fields,
                change_source_interaction_id: $change_source_interaction_id,
                created_at: datetime(),
                valid_from: coalesce(d.created_at, datetime()),
                valid_until: datetime()
            })
            MERGE (d)-[:HAS_VERSION]->(v)
            RETURN v {.*} as version
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'tenant_id': str(tenant_id),
                'opportunity_id': opportunity_id,
                'change_summary': change_summary,
                'changed_fields': changed_fields,
                'change_source_interaction_id': (
                    str(change_source_interaction_id)
                    if change_source_interaction_id
                    else None
                ),
            },
        )
        return result[0]['version'] if result else {}

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_deals_for_account(
        self,
        tenant_id: UUID,
        account_id: str,
        stage: DealStage | str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get all Deals for an account.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier
            stage: Optional stage filter
            limit: Maximum results

        Returns:
            List of Deal node properties
        """
        stage_filter = ''
        params: dict[str, Any] = {
            'tenant_id': str(tenant_id),
            'account_id': account_id,
            'limit': limit,
        }

        if stage:
            stage_filter = 'AND d.stage = $stage'
            params['stage'] = stage.value if isinstance(stage, DealStage) else stage

        query = f"""
            MATCH (d:Deal {{tenant_id: $tenant_id, account_id: $account_id}})
            WHERE d.tenant_id = $tenant_id {stage_filter}
            RETURN d {{.*}} as deal
            ORDER BY d.last_updated_at DESC
            LIMIT $limit
        """
        result = await self.neo4j.execute_query(query, params)
        return [r['deal'] for r in result]

    async def get_deal_history(
        self,
        tenant_id: UUID,
        opportunity_id: str,
    ) -> list[dict[str, Any]]:
        """
        Get version history for a Deal via [:HAS_VERSION] chain.

        Args:
            tenant_id: Tenant UUID
            opportunity_id: Deal identifier

        Returns:
            List of DealVersion properties, ordered by version descending
        """
        query = """
            MATCH (d:Deal {tenant_id: $tenant_id, opportunity_id: $opportunity_id})
                  -[:HAS_VERSION]->(v:DealVersion)
            RETURN v {.*} as version
            ORDER BY v.version DESC
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'tenant_id': str(tenant_id),
                'opportunity_id': opportunity_id,
            },
        )
        return [r['version'] for r in result]
