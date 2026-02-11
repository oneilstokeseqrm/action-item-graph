"""
Graph repository for high-level CRUD operations.

Provides:
- Entity creation (Account, Interaction, ActionItem, ActionItemVersion, Owner)
- Relationship management (HAS_ACTION_ITEM, EXTRACTED_FROM, OWNED_BY, etc.)
- Owner resolution with alias matching
- Version snapshot creation
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from .clients.neo4j_client import Neo4jClient
from .models.action_item import ActionItem, ActionItemStatus, ActionItemVersion
from .models.entities import Account, Interaction, Owner
from .models.topic import ActionItemTopic, ActionItemTopicVersion


class ActionItemRepository:
    """
    High-level graph CRUD operations for action items and related entities.

    Handles:
    - Creating and updating ActionItem nodes
    - Version snapshots before updates
    - Relationship management
    - Owner resolution with aliases
    """

    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize the repository.

        Args:
            neo4j_client: Connected Neo4j client
        """
        self.neo4j = neo4j_client

    # =========================================================================
    # Account Operations
    # =========================================================================

    async def ensure_account(
        self,
        tenant_id: UUID,
        account_id: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """
        Ensure an Account node exists, creating if necessary.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier
            name: Account name (used if creating)

        Returns:
            Account node properties
        """
        query = """
            MERGE (a:Account {account_id: $account_id, tenant_id: $tenant_id})
            ON CREATE SET
                a.name = coalesce($name, $account_id),
                a.created_at = datetime()
            RETURN a {.*} as account
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'account_id': account_id,
                'tenant_id': str(tenant_id),
                'name': name,
            },
        )
        return result[0]['account'] if result else {}

    # =========================================================================
    # Interaction Operations
    # =========================================================================

    async def create_interaction(
        self,
        interaction: Interaction,
    ) -> dict[str, Any]:
        """
        Create or merge an Interaction node and link to Account.

        Uses defensive MERGE so that if another pipeline (e.g. structured graph
        or deal pipeline) already created the Interaction node, we enrich it
        rather than failing with a constraint violation.

        Args:
            interaction: Interaction model

        Returns:
            Created/merged Interaction node properties
        """
        query = """
            MERGE (i:Interaction {tenant_id: $tenant_id, interaction_id: $interaction_id})
            ON CREATE SET
                i.content_text = $content_text,
                i.interaction_type = $interaction_type,
                i.timestamp = $timestamp,
                i.source = $source,
                i.user_id = $user_id,
                i.title = $title,
                i.duration_seconds = $duration_seconds,
                i.created_at = datetime()
            ON MATCH SET
                i.action_item_count = $action_item_count,
                i.processed_at = datetime()
            WITH i
            OPTIONAL MATCH (a:Account {account_id: $account_id, tenant_id: $tenant_id})
            FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
                MERGE (a)-[:HAS_INTERACTION]->(i)
            )
            RETURN i {.*} as interaction
        """
        # Handle interaction_type whether it's an enum or already a string
        from .models.entities import InteractionType
        interaction_type_value = (
            interaction.interaction_type.value
            if isinstance(interaction.interaction_type, InteractionType)
            else interaction.interaction_type
        )

        result = await self.neo4j.execute_write(
            query,
            {
                'tenant_id': str(interaction.tenant_id),
                'interaction_id': str(interaction.interaction_id),
                'content_text': interaction.content_text,
                'interaction_type': interaction_type_value,
                'timestamp': interaction.timestamp.isoformat(),
                'source': interaction.source,
                'user_id': interaction.user_id,
                'title': interaction.title,
                'duration_seconds': interaction.duration_seconds,
                'action_item_count': interaction.action_item_count,
                'account_id': interaction.account_id,
            },
        )
        return result[0]['interaction'] if result else {}

    async def update_interaction(
        self,
        interaction_id: UUID,
        tenant_id: UUID,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update an existing Interaction node.

        Args:
            interaction_id: Interaction UUID
            tenant_id: Tenant UUID
            updates: Properties to update

        Returns:
            Updated Interaction node properties
        """
        query = """
            MATCH (i:Interaction {interaction_id: $interaction_id, tenant_id: $tenant_id})
            SET i += $updates
            RETURN i {.*} as interaction
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'interaction_id': str(interaction_id),
                'tenant_id': str(tenant_id),
                'updates': updates,
            },
        )
        return result[0]['interaction'] if result else {}

    # =========================================================================
    # ActionItem Operations
    # =========================================================================

    async def create_action_item(
        self,
        action_item: ActionItem,
    ) -> dict[str, Any]:
        """
        Create an ActionItem node and link to Account.

        Uses MERGE on ActionItem.action_item_id so that duplicate writes
        (e.g. from dict-key collisions in the pipeline's text-based lookup)
        are idempotent instead of raising a constraint violation.

        Args:
            action_item: ActionItem model with embeddings

        Returns:
            Created ActionItem node properties
        """
        props = action_item.to_neo4j_properties()

        query = """
            MERGE (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            ON CREATE SET ai += $props
            WITH ai
            OPTIONAL MATCH (a:Account {account_id: $account_id, tenant_id: $tenant_id})
            FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
                MERGE (a)-[:HAS_ACTION_ITEM]->(ai)
            )
            RETURN ai {.*} as action_item
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': props['action_item_id'],
                'tenant_id': str(action_item.tenant_id),
                'props': props,
                'account_id': action_item.account_id,
            },
        )
        return result[0]['action_item'] if result else {}

    async def get_action_item(
        self,
        action_item_id: str,
        tenant_id: UUID,
    ) -> dict[str, Any] | None:
        """
        Retrieve an ActionItem by ID.

        Args:
            action_item_id: ActionItem UUID string
            tenant_id: Tenant UUID

        Returns:
            ActionItem node properties or None if not found
        """
        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            RETURN ai {.*} as action_item
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'action_item_id': action_item_id,
                'tenant_id': str(tenant_id),
            },
        )
        return result[0]['action_item'] if result else None

    async def update_action_item(
        self,
        action_item_id: str,
        tenant_id: UUID,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update an existing ActionItem node.

        Note: Call create_version_snapshot() before this to preserve history.

        Args:
            action_item_id: ActionItem UUID string
            tenant_id: Tenant UUID
            updates: Properties to update

        Returns:
            Updated ActionItem node properties
        """
        # Add last_updated_at and increment version
        updates['last_updated_at'] = datetime.now().isoformat()

        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            SET ai += $updates,
                ai.version = ai.version + 1
            RETURN ai {.*} as action_item
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': action_item_id,
                'tenant_id': str(tenant_id),
                'updates': updates,
            },
        )
        return result[0]['action_item'] if result else {}

    async def update_action_item_status(
        self,
        action_item_id: str,
        tenant_id: UUID,
        status: ActionItemStatus | str,
    ) -> dict[str, Any]:
        """
        Update only the status of an ActionItem.

        This is a lightweight update that doesn't require re-embedding.

        Args:
            action_item_id: ActionItem UUID string
            tenant_id: Tenant UUID
            status: New status value

        Returns:
            Updated ActionItem node properties
        """
        status_value = status.value if isinstance(status, ActionItemStatus) else status

        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            SET ai.status = $status,
                ai.last_updated_at = $now,
                ai.version = ai.version + 1
            RETURN ai {.*} as action_item
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': action_item_id,
                'tenant_id': str(tenant_id),
                'status': status_value,
                'now': datetime.now().isoformat(),
            },
        )
        return result[0]['action_item'] if result else {}

    # =========================================================================
    # ActionItemVersion Operations
    # =========================================================================

    async def create_version_snapshot(
        self,
        action_item_id: str,
        tenant_id: UUID,
        change_summary: str,
        source_interaction_id: UUID | None = None,
    ) -> dict[str, Any]:
        """
        Create a version snapshot before updating an ActionItem.

        This preserves the current state as an ActionItemVersion node.

        Args:
            action_item_id: ActionItem UUID string
            tenant_id: Tenant UUID
            change_summary: Description of what's changing
            source_interaction_id: Interaction that triggered this update

        Returns:
            Created ActionItemVersion node properties
        """
        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            CREATE (v:ActionItemVersion {
                version_id: randomUUID(),
                action_item_id: ai.action_item_id,
                tenant_id: ai.tenant_id,
                version: ai.version,
                action_item_text: ai.action_item_text,
                summary: ai.summary,
                owner: ai.owner,
                status: ai.status,
                due_date: ai.due_date,
                change_summary: $change_summary,
                change_source_interaction_id: $source_interaction_id,
                created_at: datetime(),
                valid_from: ai.created_at,
                valid_until: datetime()
            })
            MERGE (ai)-[:HAS_VERSION]->(v)
            RETURN v {.*} as version
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': action_item_id,
                'tenant_id': str(tenant_id),
                'change_summary': change_summary,
                'source_interaction_id': str(source_interaction_id)
                if source_interaction_id
                else None,
            },
        )
        return result[0]['version'] if result else {}

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    async def link_to_interaction(
        self,
        action_item_id: str,
        interaction_id: str,
        tenant_id: UUID,
    ) -> bool:
        """
        Create EXTRACTED_FROM relationship between ActionItem and Interaction.

        Args:
            action_item_id: ActionItem UUID string
            interaction_id: Interaction UUID string
            tenant_id: Tenant UUID

        Returns:
            True if relationship was created
        """
        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            MATCH (i:Interaction {interaction_id: $interaction_id, tenant_id: $tenant_id})
            MERGE (ai)-[r:EXTRACTED_FROM]->(i)
            ON CREATE SET r.created_at = datetime()
            RETURN r IS NOT NULL as created
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': action_item_id,
                'interaction_id': interaction_id,
                'tenant_id': str(tenant_id),
            },
        )
        return result[0]['created'] if result else False

    async def link_to_owner(
        self,
        action_item_id: str,
        owner_id: str,
        tenant_id: UUID,
    ) -> bool:
        """
        Create OWNED_BY relationship between ActionItem and Owner.

        Args:
            action_item_id: ActionItem UUID string
            owner_id: Owner UUID string
            tenant_id: Tenant UUID

        Returns:
            True if relationship was created
        """
        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            MATCH (o:Owner {owner_id: $owner_id, tenant_id: $tenant_id})
            MERGE (ai)-[r:OWNED_BY]->(o)
            ON CREATE SET r.created_at = datetime()
            RETURN r IS NOT NULL as created
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': action_item_id,
                'owner_id': owner_id,
                'tenant_id': str(tenant_id),
            },
        )
        return result[0]['created'] if result else False

    async def link_related_items(
        self,
        action_item_id: str,
        related_item_id: str,
        tenant_id: UUID,
        relationship_type: str = 'RELATED_TO',
    ) -> bool:
        """
        Create a relationship between two ActionItems.

        Args:
            action_item_id: Source ActionItem UUID string
            related_item_id: Target ActionItem UUID string
            tenant_id: Tenant UUID
            relationship_type: Type of relationship (default: RELATED_TO)

        Returns:
            True if relationship was created
        """
        # Use parameterized relationship type for safety
        query = f"""
            MATCH (ai1:ActionItem {{action_item_id: $action_item_id, tenant_id: $tenant_id}})
            MATCH (ai2:ActionItem {{action_item_id: $related_item_id, tenant_id: $tenant_id}})
            MERGE (ai1)-[r:{relationship_type}]->(ai2)
            ON CREATE SET r.created_at = datetime()
            RETURN r IS NOT NULL as created
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': action_item_id,
                'related_item_id': related_item_id,
                'tenant_id': str(tenant_id),
            },
        )
        return result[0]['created'] if result else False

    # =========================================================================
    # Owner Resolution
    # =========================================================================

    async def resolve_or_create_owner(
        self,
        tenant_id: UUID,
        owner_name: str,
    ) -> dict[str, Any]:
        """
        Find an existing Owner by name/alias, or create a new one.

        Matching logic:
        1. Exact match on canonical_name
        2. Match in aliases list
        3. Case-insensitive partial match (e.g., "John" matches "John Smith")
        4. Create new Owner if no match

        Args:
            tenant_id: Tenant UUID
            owner_name: Name from extraction (e.g., "John", "Sarah")

        Returns:
            Owner node properties (existing or newly created)
        """
        # Normalize the search name
        normalized_name = owner_name.strip()

        # Try to find existing owner
        query = """
            MATCH (o:Owner {tenant_id: $tenant_id})
            WHERE o.canonical_name = $name
                OR $name IN o.aliases
                OR toLower(o.canonical_name) CONTAINS toLower($name)
                OR toLower($name) CONTAINS toLower(o.canonical_name)
            RETURN o {.*} as owner
            ORDER BY
                CASE
                    WHEN o.canonical_name = $name THEN 0
                    WHEN $name IN o.aliases THEN 1
                    ELSE 2
                END
            LIMIT 1
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'tenant_id': str(tenant_id),
                'name': normalized_name,
            },
        )

        if result:
            owner = result[0]['owner']
            # Add name to aliases if it's a new variant
            if normalized_name not in owner.get('aliases', []) and normalized_name != owner.get('canonical_name'):
                await self._add_owner_alias(owner['owner_id'], str(tenant_id), normalized_name)
            return owner

        # Create new owner
        create_query = """
            CREATE (o:Owner {
                owner_id: randomUUID(),
                tenant_id: $tenant_id,
                canonical_name: $name,
                aliases: [],
                created_at: datetime()
            })
            RETURN o {.*} as owner
        """
        create_result = await self.neo4j.execute_write(
            create_query,
            {
                'tenant_id': str(tenant_id),
                'name': normalized_name,
            },
        )
        return create_result[0]['owner'] if create_result else {}

    async def _add_owner_alias(
        self,
        owner_id: str,
        tenant_id: str,
        alias: str,
    ) -> None:
        """Add an alias to an existing Owner."""
        query = """
            MATCH (o:Owner {owner_id: $owner_id, tenant_id: $tenant_id})
            WHERE NOT $alias IN o.aliases
            SET o.aliases = o.aliases + $alias
        """
        await self.neo4j.execute_write(
            query,
            {
                'owner_id': owner_id,
                'tenant_id': tenant_id,
                'alias': alias,
            },
        )

    async def get_owner_by_name(
        self,
        tenant_id: UUID,
        owner_name: str,
    ) -> dict[str, Any] | None:
        """
        Find an Owner by name without creating.

        Args:
            tenant_id: Tenant UUID
            owner_name: Name to search for

        Returns:
            Owner node properties or None if not found
        """
        normalized_name = owner_name.strip()

        query = """
            MATCH (o:Owner {tenant_id: $tenant_id})
            WHERE o.canonical_name = $name
                OR $name IN o.aliases
                OR toLower(o.canonical_name) CONTAINS toLower($name)
                OR toLower($name) CONTAINS toLower(o.canonical_name)
            RETURN o {.*} as owner
            ORDER BY
                CASE
                    WHEN o.canonical_name = $name THEN 0
                    WHEN $name IN o.aliases THEN 1
                    ELSE 2
                END
            LIMIT 1
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'tenant_id': str(tenant_id),
                'name': normalized_name,
            },
        )
        return result[0]['owner'] if result else None

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_action_items_for_account(
        self,
        tenant_id: UUID,
        account_id: str,
        status: ActionItemStatus | str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get all ActionItems for an account.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of ActionItem node properties
        """
        status_filter = ''
        params: dict[str, Any] = {
            'tenant_id': str(tenant_id),
            'account_id': account_id,
            'limit': limit,
        }

        if status:
            status_filter = 'AND ai.status = $status'
            params['status'] = status.value if isinstance(status, ActionItemStatus) else status

        query = f"""
            MATCH (a:Account {{account_id: $account_id, tenant_id: $tenant_id}})-[:HAS_ACTION_ITEM]->(ai:ActionItem)
            WHERE ai.tenant_id = $tenant_id {status_filter}
            RETURN ai {{.*}} as action_item
            ORDER BY ai.created_at DESC
            LIMIT $limit
        """
        result = await self.neo4j.execute_query(query, params)
        return [r['action_item'] for r in result]

    async def get_action_item_history(
        self,
        action_item_id: str,
        tenant_id: UUID,
    ) -> list[dict[str, Any]]:
        """
        Get version history for an ActionItem.

        Args:
            action_item_id: ActionItem UUID string
            tenant_id: Tenant UUID

        Returns:
            List of ActionItemVersion properties, ordered by version desc
        """
        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})-[:HAS_VERSION]->(v:ActionItemVersion)
            RETURN v {.*} as version
            ORDER BY v.version DESC
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'action_item_id': action_item_id,
                'tenant_id': str(tenant_id),
            },
        )
        return [r['version'] for r in result]

    # =========================================================================
    # ActionItemTopic Operations
    # =========================================================================

    async def create_topic(
        self,
        topic: ActionItemTopic,
    ) -> dict[str, Any]:
        """
        Create an ActionItemTopic node and link to Account.

        Args:
            topic: ActionItemTopic model with embeddings

        Returns:
            Created ActionItemTopic node properties
        """
        props = topic.to_neo4j_properties()

        query = """
            CREATE (t:ActionItemTopic $props)
            WITH t
            OPTIONAL MATCH (a:Account {account_id: $account_id, tenant_id: $tenant_id})
            FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END |
                MERGE (a)-[:HAS_TOPIC]->(t)
            )
            RETURN t {.*} as topic
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'props': props,
                'account_id': topic.account_id,
                'tenant_id': str(topic.tenant_id),
            },
        )
        return result[0]['topic'] if result else {}

    async def get_topic(
        self,
        topic_id: str,
        tenant_id: UUID,
    ) -> dict[str, Any] | None:
        """
        Retrieve an ActionItemTopic by ID.

        Args:
            topic_id: ActionItemTopic UUID string
            tenant_id: Tenant UUID

        Returns:
            ActionItemTopic node properties or None if not found
        """
        query = """
            MATCH (t:ActionItemTopic {action_item_topic_id: $topic_id, tenant_id: $tenant_id})
            RETURN t {.*} as topic
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'topic_id': topic_id,
                'tenant_id': str(tenant_id),
            },
        )
        return result[0]['topic'] if result else None

    async def update_topic(
        self,
        topic_id: str,
        tenant_id: UUID,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update an existing ActionItemTopic node.

        Args:
            topic_id: ActionItemTopic UUID string
            tenant_id: Tenant UUID
            updates: Properties to update

        Returns:
            Updated ActionItemTopic node properties
        """
        query = """
            MATCH (t:ActionItemTopic {action_item_topic_id: $topic_id, tenant_id: $tenant_id})
            SET t += $updates
            RETURN t {.*} as topic
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'topic_id': topic_id,
                'tenant_id': str(tenant_id),
                'updates': updates,
            },
        )
        return result[0]['topic'] if result else {}

    async def create_topic_version(
        self,
        topic_id: str,
        tenant_id: UUID,
        version_number: int,
        name: str,
        summary: str,
        embedding_snapshot: list[float] | None = None,
        changed_by_action_item_id: UUID | None = None,
    ) -> dict[str, Any]:
        """
        Create an ActionItemTopicVersion snapshot.

        Args:
            topic_id: ActionItemTopic UUID string
            tenant_id: Tenant UUID
            version_number: Version number
            name: Topic name at this version
            summary: Topic summary at this version
            embedding_snapshot: Optional embedding snapshot
            changed_by_action_item_id: Action item that triggered this version

        Returns:
            Created ActionItemTopicVersion node properties
        """
        query = """
            MATCH (t:ActionItemTopic {action_item_topic_id: $topic_id, tenant_id: $tenant_id})
            CREATE (v:ActionItemTopicVersion {
                version_id: randomUUID(),
                topic_id: $topic_id,
                tenant_id: $tenant_id,
                version_number: $version_number,
                name: $name,
                summary: $summary,
                embedding_snapshot: $embedding_snapshot,
                changed_by_action_item_id: $changed_by_action_item_id,
                created_at: datetime()
            })
            MERGE (t)-[:HAS_VERSION {version_number: $version_number}]->(v)
            RETURN v {.*} as version
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'topic_id': topic_id,
                'tenant_id': str(tenant_id),
                'version_number': version_number,
                'name': name,
                'summary': summary,
                'embedding_snapshot': embedding_snapshot,
                'changed_by_action_item_id': str(changed_by_action_item_id)
                if changed_by_action_item_id
                else None,
            },
        )
        return result[0]['version'] if result else {}

    async def link_action_item_to_topic(
        self,
        action_item_id: str,
        topic_id: str,
        tenant_id: UUID,
        confidence: float = 1.0,
        method: str = 'extracted',
    ) -> bool:
        """
        Create BELONGS_TO relationship between ActionItem and ActionItemTopic.

        Args:
            action_item_id: ActionItem UUID string
            topic_id: ActionItemTopic UUID string
            tenant_id: Tenant UUID
            confidence: Confidence score for the link
            method: How the link was created ('extracted', 'resolved', 'manual')

        Returns:
            True if relationship was created
        """
        query = """
            MATCH (ai:ActionItem {action_item_id: $action_item_id, tenant_id: $tenant_id})
            MATCH (t:ActionItemTopic {action_item_topic_id: $topic_id, tenant_id: $tenant_id})
            MERGE (ai)-[r:BELONGS_TO]->(t)
            ON CREATE SET
                r.confidence = $confidence,
                r.method = $method,
                r.created_at = datetime()
            RETURN r IS NOT NULL as created
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'action_item_id': action_item_id,
                'topic_id': topic_id,
                'tenant_id': str(tenant_id),
                'confidence': confidence,
                'method': method,
            },
        )
        return result[0]['created'] if result else False

    async def increment_topic_action_count(
        self,
        topic_id: str,
        tenant_id: UUID,
    ) -> int:
        """
        Increment the action_item_count for an ActionItemTopic.

        Args:
            topic_id: ActionItemTopic UUID string
            tenant_id: Tenant UUID

        Returns:
            New action item count
        """
        query = """
            MATCH (t:ActionItemTopic {action_item_topic_id: $topic_id, tenant_id: $tenant_id})
            SET t.action_item_count = coalesce(t.action_item_count, 0) + 1,
                t.updated_at = datetime()
            RETURN t.action_item_count as count
        """
        result = await self.neo4j.execute_write(
            query,
            {
                'topic_id': topic_id,
                'tenant_id': str(tenant_id),
            },
        )
        return result[0]['count'] if result else 0

    async def get_topics_for_account(
        self,
        tenant_id: UUID,
        account_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Get all ActionItemTopics for an account.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier
            limit: Maximum results

        Returns:
            List of ActionItemTopic node properties
        """
        query = """
            MATCH (a:Account {account_id: $account_id, tenant_id: $tenant_id})-[:HAS_TOPIC]->(t:ActionItemTopic)
            WHERE t.tenant_id = $tenant_id
            RETURN t {.*} as topic
            ORDER BY t.action_item_count DESC, t.created_at DESC
            LIMIT $limit
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'tenant_id': str(tenant_id),
                'account_id': account_id,
                'limit': limit,
            },
        )
        return [r['topic'] for r in result]

    async def get_topic_with_action_items(
        self,
        topic_id: str,
        tenant_id: UUID,
    ) -> dict[str, Any] | None:
        """
        Get an ActionItemTopic with all its linked ActionItems.

        Args:
            topic_id: ActionItemTopic UUID string
            tenant_id: Tenant UUID

        Returns:
            Topic properties with 'action_items' list, or None if not found
        """
        query = """
            MATCH (t:ActionItemTopic {action_item_topic_id: $topic_id, tenant_id: $tenant_id})
            OPTIONAL MATCH (ai:ActionItem)-[r:BELONGS_TO]->(t)
            WHERE ai.tenant_id = $tenant_id
            WITH t, collect({
                action_item: ai {.*},
                confidence: r.confidence,
                method: r.method,
                linked_at: r.created_at
            }) as items
            RETURN t {.*, action_items: items} as topic
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'topic_id': topic_id,
                'tenant_id': str(tenant_id),
            },
        )
        return result[0]['topic'] if result else None

    async def get_topic_history(
        self,
        topic_id: str,
        tenant_id: UUID,
    ) -> list[dict[str, Any]]:
        """
        Get version history for an ActionItemTopic.

        Args:
            topic_id: ActionItemTopic UUID string
            tenant_id: Tenant UUID

        Returns:
            List of ActionItemTopicVersion properties, ordered by version_number desc
        """
        query = """
            MATCH (t:ActionItemTopic {action_item_topic_id: $topic_id, tenant_id: $tenant_id})-[:HAS_VERSION]->(v:ActionItemTopicVersion)
            RETURN v {.*} as version
            ORDER BY v.version_number DESC
        """
        result = await self.neo4j.execute_query(
            query,
            {
                'topic_id': topic_id,
                'tenant_id': str(tenant_id),
            },
        )
        return [r['version'] for r in result]
