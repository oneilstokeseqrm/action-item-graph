"""
Neo4j client wrapper for the Action Item Graph pipeline.

Handles:
- Connection management with async driver
- Schema setup (constraints, indexes, vector indexes)
- CRUD operations for nodes and relationships
- Vector similarity search for action item matching
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential


class Neo4jClient:
    """
    Async Neo4j client with vector search support.

    Configuration via environment variables:
    - NEO4J_URI: Database URI (e.g., neo4j+s://xxx.databases.neo4j.io)
    - NEO4J_USERNAME: Username (default: neo4j)
    - NEO4J_PASSWORD: Password
    - NEO4J_DATABASE: Database name (default: neo4j)
    """

    # Vector index configuration
    EMBEDDING_DIMENSIONS = 1536
    VECTOR_INDEX_NAME = 'action_item_embedding_idx'
    VECTOR_CURRENT_INDEX_NAME = 'action_item_embedding_current_idx'
    TOPIC_VECTOR_INDEX_NAME = 'topic_embedding_idx'
    TOPIC_VECTOR_CURRENT_INDEX_NAME = 'topic_embedding_current_idx'

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        """
        Initialize the Neo4j client.

        Args:
            uri: Neo4j URI (defaults to NEO4J_URI env var)
            username: Username (defaults to NEO4J_USERNAME or 'neo4j')
            password: Password (defaults to NEO4J_PASSWORD env var)
            database: Database name (defaults to NEO4J_DATABASE or 'neo4j')
        """
        self.uri = uri or os.getenv('NEO4J_URI')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD')
        self.database = database or os.getenv('NEO4J_DATABASE', 'neo4j')

        if not self.uri:
            raise ValueError('NEO4J_URI environment variable is required')
        if not self.password:
            raise ValueError('NEO4J_PASSWORD environment variable is required')

        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
            )
            # Verify connectivity
            await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close the database connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    @asynccontextmanager
    async def session(self):
        """Get an async session context manager."""
        if self._driver is None:
            await self.connect()
        async with self._driver.session(database=self.database) as session:
            yield session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        async with self.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a write query within a transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        async with self.session() as session:

            async def _write_tx(tx):
                result = await tx.run(query, parameters or {})
                return await result.data()

            return await session.execute_write(_write_tx)

    async def health_check(self) -> dict[str, bool | str]:
        """
        Verify database connectivity.

        Returns:
            Dict with 'healthy' bool, database info, and optional error
        """
        try:
            if self._driver is None:
                await self.connect()
            await self._driver.verify_connectivity()

            # Get some basic info
            result = await self.execute_query('RETURN 1 as test')

            return {
                'healthy': True,
                'uri': self.uri,
                'database': self.database,
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    async def setup_schema(self) -> dict[str, list[str]]:
        """
        Create all necessary constraints and indexes.

        Constraints use tenant-scoped NODE KEY on (tenant_id, id) for
        multi-tenant isolation.  Migration from older global single-property
        constraints is handled automatically: new composite constraints are
        created first, then stale global ones are dropped.

        Returns:
            Dict with lists of created constraints and indexes
        """
        created = {'constraints': [], 'indexes': [], 'vector_indexes': []}

        # --- Step 1: Create tenant-scoped NODE KEY constraints -----------
        # NODE KEY enforces existence + uniqueness of (tenant_id, id).
        composite_constraints = [
            ('account_tenant_key', 'Account'),
            ('interaction_tenant_key', 'Interaction'),
            ('action_item_tenant_key', 'ActionItem'),
            ('action_item_version_tenant_key', 'ActionItemVersion'),
            ('owner_tenant_key', 'Owner'),
            ('topic_tenant_key', 'Topic'),
            ('topic_version_tenant_key', 'TopicVersion'),
        ]

        for name, label in composite_constraints:
            try:
                await self.execute_write(
                    f'CREATE CONSTRAINT {name} IF NOT EXISTS '
                    f'FOR (n:{label}) REQUIRE (n.tenant_id, n.id) IS NODE KEY'
                )
                created['constraints'].append(name)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    raise

        # --- Step 2: Drop stale global single-property constraints -------
        # These are superseded by the composite constraints above.
        # Only covers labels owned by the Action Item pipeline.
        legacy_constraints = [
            'account_id_unique',
            'interaction_id_unique',
            'action_item_id_unique',
            'action_item_version_id_unique',
            'owner_id_unique',
            'topic_id_unique',
            'topic_version_id_unique',
        ]

        for name in legacy_constraints:
            try:
                await self.execute_write(f'DROP CONSTRAINT {name} IF EXISTS')
            except Exception:
                pass  # already gone

        # Regular indexes for common queries
        indexes = [
            ('action_item_tenant_idx', 'ActionItem', 'tenant_id'),
            ('action_item_account_idx', 'ActionItem', 'account_id'),
            ('action_item_status_idx', 'ActionItem', 'status'),
            ('interaction_tenant_idx', 'Interaction', 'tenant_id'),
            ('interaction_account_idx', 'Interaction', 'account_id'),
            ('owner_tenant_idx', 'Owner', 'tenant_id'),
            ('topic_tenant_idx', 'Topic', 'tenant_id'),
            ('topic_account_idx', 'Topic', 'account_id'),
            ('topic_canonical_name_idx', 'Topic', 'canonical_name'),
        ]

        for name, label, prop in indexes:
            try:
                await self.execute_write(
                    f'CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})'
                )
                created['indexes'].append(name)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    raise

        # Vector indexes for similarity search
        vector_indexes = [
            (self.VECTOR_INDEX_NAME, 'ActionItem', 'embedding'),
            (self.VECTOR_CURRENT_INDEX_NAME, 'ActionItem', 'embedding_current'),
            (self.TOPIC_VECTOR_INDEX_NAME, 'Topic', 'embedding'),
            (self.TOPIC_VECTOR_CURRENT_INDEX_NAME, 'Topic', 'embedding_current'),
        ]

        for name, label, prop in vector_indexes:
            try:
                await self.execute_write(f"""
                    CREATE VECTOR INDEX {name} IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.{prop})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.EMBEDDING_DIMENSIONS},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                created['vector_indexes'].append(name)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    raise

        return created

    async def vector_search(
        self,
        embedding: list[float],
        index_name: str | None = None,
        tenant_id: str | None = None,
        account_id: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search for similar action items using vector similarity.

        Args:
            embedding: Query embedding vector
            index_name: Vector index to search (default: action_item_embedding_idx)
            tenant_id: Filter by tenant (strongly recommended for multi-tenancy)
            account_id: Filter by account (optional additional filter)
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of dicts with 'node' (ActionItem properties) and 'score'
        """
        index = index_name or self.VECTOR_INDEX_NAME

        # Build the query with optional filters
        filters = []
        params: dict[str, Any] = {
            'embedding': embedding,
            'limit': limit,
            'min_score': min_score,
        }

        # Always include min_score filter
        filters.append('score >= $min_score')

        if tenant_id:
            filters.append('node.tenant_id = $tenant_id')
            params['tenant_id'] = tenant_id
        if account_id:
            filters.append('node.account_id = $account_id')
            params['account_id'] = account_id

        where_clause = f"WHERE {' AND '.join(filters)}"

        query = f"""
            CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
            YIELD node, score
            {where_clause}
            RETURN node {{.*}} as node, score
            ORDER BY score DESC
        """
        params['index_name'] = index

        return await self.execute_query(query, params)

    async def search_both_embeddings(
        self,
        embedding: list[float],
        tenant_id: str,
        account_id: str,
        limit: int = 10,
        min_score: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Search both original and current embeddings for matches.

        This dual search helps catch:
        - New items similar to original action items (via embedding)
        - Status updates to evolved action items (via embedding_current)

        Args:
            embedding: Query embedding vector
            tenant_id: Filter by tenant (required for multi-tenancy)
            account_id: Filter by account (required to prevent cross-account bleeding)
            limit: Maximum results per index
            min_score: Minimum similarity score

        Returns:
            Deduplicated list of matches, sorted by best score
        """
        # Search both indexes
        original_results = await self.vector_search(
            embedding=embedding,
            index_name=self.VECTOR_INDEX_NAME,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=min_score,
        )

        current_results = await self.vector_search(
            embedding=embedding,
            index_name=self.VECTOR_CURRENT_INDEX_NAME,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=min_score,
        )

        # Combine and deduplicate by action item ID, keeping best score
        seen: dict[str, dict[str, Any]] = {}

        for result in original_results + current_results:
            node_id = result['node']['id']
            if node_id not in seen or result['score'] > seen[node_id]['score']:
                seen[node_id] = result

        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:limit]

    async def get_schema_info(self) -> dict[str, Any]:
        """
        Get information about the current database schema.

        Returns:
            Dict with node labels, relationship types, and indexes
        """
        labels_result = await self.execute_query('CALL db.labels()')
        rels_result = await self.execute_query('CALL db.relationshipTypes()')
        indexes_result = await self.execute_query('SHOW INDEXES')

        return {
            'labels': [r['label'] for r in labels_result],
            'relationship_types': [r['relationshipType'] for r in rels_result],
            'indexes': indexes_result,
        }

    # =========================================================================
    # Topic Vector Search
    # =========================================================================

    async def topic_vector_search(
        self,
        embedding: list[float],
        index_name: str | None = None,
        tenant_id: str | None = None,
        account_id: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search for similar topics using vector similarity.

        Args:
            embedding: Query embedding vector
            index_name: Vector index to search (default: topic_embedding_idx)
            tenant_id: Filter by tenant (strongly recommended for multi-tenancy)
            account_id: Filter by account (optional additional filter)
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of dicts with 'node' (Topic properties) and 'score'
        """
        index = index_name or self.TOPIC_VECTOR_INDEX_NAME

        # Build the query with optional filters
        filters = []
        params: dict[str, Any] = {
            'embedding': embedding,
            'limit': limit,
            'min_score': min_score,
        }

        filters.append('score >= $min_score')

        if tenant_id:
            filters.append('node.tenant_id = $tenant_id')
            params['tenant_id'] = tenant_id
        if account_id:
            filters.append('node.account_id = $account_id')
            params['account_id'] = account_id

        where_clause = f"WHERE {' AND '.join(filters)}"

        query = f"""
            CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
            YIELD node, score
            {where_clause}
            RETURN node {{.*}} as node, score
            ORDER BY score DESC
        """
        params['index_name'] = index

        return await self.execute_query(query, params)

    async def search_topics_both_embeddings(
        self,
        embedding: list[float],
        tenant_id: str,
        account_id: str,
        limit: int = 5,
        min_score: float = 0.65,
    ) -> list[dict[str, Any]]:
        """
        Search both original and current topic embeddings for matches.

        This dual search helps catch:
        - New items similar to original topic scope (via embedding)
        - Items related to evolved topic scope (via embedding_current)

        Args:
            embedding: Query embedding vector
            tenant_id: Filter by tenant (required for multi-tenancy)
            account_id: Filter by account (required to prevent cross-account bleeding)
            limit: Maximum results per index
            min_score: Minimum similarity score

        Returns:
            Deduplicated list of topic matches, sorted by best score
        """
        # Search both indexes
        original_results = await self.topic_vector_search(
            embedding=embedding,
            index_name=self.TOPIC_VECTOR_INDEX_NAME,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=min_score,
        )

        current_results = await self.topic_vector_search(
            embedding=embedding,
            index_name=self.TOPIC_VECTOR_CURRENT_INDEX_NAME,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=min_score,
        )

        # Combine and deduplicate by topic ID, keeping best score
        seen: dict[str, dict[str, Any]] = {}

        for result in original_results + current_results:
            node_id = result['node']['id']
            if node_id not in seen or result['score'] > seen[node_id]['score']:
                seen[node_id] = result

        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:limit]
