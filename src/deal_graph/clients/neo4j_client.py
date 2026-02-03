"""
Neo4j client for the Deal Graph pipeline.

Inherits from the base Neo4jClient for connection management and retry logic.
Overrides setup_schema() to create only enrichment schema elements:
- DealVersion constraint (our addition)
- Deal vector indexes (our addition)
- Deal performance indexes (our addition)

Does NOT create skeleton constraints (deal_unique, interaction_unique, etc.)
which are owned by eq-structured-graph-core.

Provides verify_skeleton_schema() to fail fast if the database was not
prepped by the schema authority.
"""

import os
from typing import Any

from action_item_graph.clients.neo4j_client import Neo4jClient


class DealNeo4jClient(Neo4jClient):
    """
    Neo4j client for the Deal pipeline.

    Connects to the existing neo4j_structured database managed by
    eq-structured-graph-core. Only creates enrichment schema elements;
    skeleton constraints are verified, not created.

    Configuration via DEAL_NEO4J_* environment variables.
    """

    # Vector index configuration for Deal nodes
    EMBEDDING_DIMENSIONS = 1536
    DEAL_VECTOR_INDEX_NAME = 'deal_embedding_idx'
    DEAL_VECTOR_CURRENT_INDEX_NAME = 'deal_embedding_current_idx'

    # Skeleton constraints we expect to exist (owned by eq-structured-graph-core)
    EXPECTED_SKELETON_CONSTRAINTS = [
        'deal_unique',
        'interaction_unique',
        'account_unique',
        'contact_unique',
    ]

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        """
        Initialize the Deal Neo4j client.

        Args:
            uri: Neo4j URI (defaults to DEAL_NEO4J_URI env var)
            username: Username (defaults to DEAL_NEO4J_USERNAME or 'neo4j')
            password: Password (defaults to DEAL_NEO4J_PASSWORD env var)
            database: Database name (defaults to DEAL_NEO4J_DATABASE or 'neo4j')
        """
        super().__init__(
            uri=uri or os.getenv('DEAL_NEO4J_URI'),
            username=username or os.getenv('DEAL_NEO4J_USERNAME', 'neo4j'),
            password=password or os.getenv('DEAL_NEO4J_PASSWORD'),
            database=database or os.getenv('DEAL_NEO4J_DATABASE', 'neo4j'),
        )

    async def verify_skeleton_schema(self) -> dict[str, Any]:
        """
        Verify that eq-structured-graph-core's skeleton schema exists.

        Checks for expected constraints and node labels. Fails fast if the
        database was not prepped by the schema authority.

        Returns:
            Dict with verification results:
            - 'verified': bool
            - 'found_constraints': list of found constraint names
            - 'missing_constraints': list of missing constraint names
            - 'found_labels': list of node labels in the database

        Raises:
            RuntimeError: If critical skeleton constraints are missing
        """
        # Get current constraints
        constraints_result = await self.execute_query('SHOW CONSTRAINTS')
        existing_names = {r.get('name', '') for r in constraints_result}

        found = []
        missing = []
        for expected in self.EXPECTED_SKELETON_CONSTRAINTS:
            if expected in existing_names:
                found.append(expected)
            else:
                missing.append(expected)

        # Get current labels
        labels_result = await self.execute_query('CALL db.labels()')
        labels = [r['label'] for r in labels_result]

        result = {
            'verified': len(missing) == 0,
            'found_constraints': found,
            'missing_constraints': missing,
            'found_labels': labels,
        }

        if missing:
            raise RuntimeError(
                f'Skeleton schema not found in database. Missing constraints: {missing}. '
                f'Run eq-structured-graph-core setup_db.py before starting the Deal pipeline.'
            )

        return result

    async def setup_schema(self) -> dict[str, list[str]]:
        """
        Create enrichment-only schema elements.

        Creates:
        - DealVersion uniqueness constraint (our addition)
        - Deal vector indexes for similarity search (our addition)
        - Deal performance indexes (our addition)

        Does NOT create skeleton constraints — those are owned by
        eq-structured-graph-core and verified separately via
        verify_skeleton_schema().

        Returns:
            Dict with lists of created constraints, indexes, and vector indexes
        """
        created: dict[str, list[str]] = {
            'constraints': [],
            'indexes': [],
            'vector_indexes': [],
        }

        # --- Our constraints (new node types we own) ---
        enrichment_constraints = [
            (
                'dealversion_unique',
                'CREATE CONSTRAINT dealversion_unique IF NOT EXISTS '
                'FOR (n:DealVersion) REQUIRE (n.tenant_id, n.version_id) IS UNIQUE',
            ),
        ]

        for name, query in enrichment_constraints:
            try:
                await self.execute_write(query)
                created['constraints'].append(name)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    raise

        # --- Our performance indexes ---
        enrichment_indexes = [
            (
                'deal_stage_idx',
                'CREATE INDEX deal_stage_idx IF NOT EXISTS '
                'FOR (n:Deal) ON (n.tenant_id, n.stage)',
            ),
            (
                'deal_account_idx',
                'CREATE INDEX deal_account_idx IF NOT EXISTS '
                'FOR (n:Deal) ON (n.tenant_id, n.account_id)',
            ),
        ]

        for name, query in enrichment_indexes:
            try:
                await self.execute_write(query)
                created['indexes'].append(name)
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    raise

        # --- Our vector indexes ---
        vector_indexes = [
            (self.DEAL_VECTOR_INDEX_NAME, 'Deal', 'embedding'),
            (self.DEAL_VECTOR_CURRENT_INDEX_NAME, 'Deal', 'embedding_current'),
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

    async def deal_vector_search(
        self,
        embedding: list[float],
        index_name: str | None = None,
        tenant_id: str | None = None,
        account_id: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search for similar Deal nodes using vector similarity.

        Args:
            embedding: Query embedding vector
            index_name: Vector index to search (default: deal_embedding_idx)
            tenant_id: Filter by tenant (required for multi-tenancy)
            account_id: Filter by account (optional)
            limit: Maximum results to return
            min_score: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of dicts with 'node' (Deal properties) and 'score'
        """
        index = index_name or self.DEAL_VECTOR_INDEX_NAME

        filters = ['score >= $min_score']
        params: dict[str, Any] = {
            'embedding': embedding,
            'limit': limit,
            'min_score': min_score,
        }

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

    async def search_deals_both_embeddings(
        self,
        embedding: list[float],
        tenant_id: str,
        account_id: str | None = None,
        limit: int = 10,
        min_score: float = 0.70,
    ) -> list[dict[str, Any]]:
        """
        Search both original and current Deal embeddings for matches.

        Dual search catches:
        - New deals similar to original state (via embedding)
        - Updates to evolved deals (via embedding_current)

        Results are deduplicated by opportunity_id, keeping the higher score.

        Args:
            embedding: Query embedding vector
            tenant_id: Filter by tenant (required)
            account_id: Filter by account (optional)
            limit: Maximum results per index
            min_score: Minimum similarity score

        Returns:
            Deduplicated list of matches, sorted by best score
        """
        original_results = await self.deal_vector_search(
            embedding=embedding,
            index_name=self.DEAL_VECTOR_INDEX_NAME,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=min_score,
        )

        current_results = await self.deal_vector_search(
            embedding=embedding,
            index_name=self.DEAL_VECTOR_CURRENT_INDEX_NAME,
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
            min_score=min_score,
        )

        # Deduplicate by opportunity_id, keeping best score
        seen: dict[str, dict[str, Any]] = {}
        for result in original_results + current_results:
            opp_id = result['node'].get('opportunity_id')
            if opp_id and (opp_id not in seen or result['score'] > seen[opp_id]['score']):
                seen[opp_id] = result

        return sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:limit]
