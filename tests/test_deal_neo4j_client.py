"""
Integration tests for DealNeo4jClient.

Tests connectivity, enrichment schema setup, and skeleton schema verification
against the neo4j_structured database.

Run with: pytest tests/test_deal_neo4j_client.py -v

Requires DEAL_NEO4J_* environment variables to be set.
"""

import pytest

from deal_graph.clients.neo4j_client import DealNeo4jClient


class TestDealNeo4jHealth:
    """Test Deal Neo4j connectivity."""

    @pytest.mark.asyncio
    async def test_health_check(self, deal_neo4j_credentials: dict[str, str]):
        """Verify we can connect to the neo4j_structured database."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            result = await client.health_check()
            assert result['healthy'] is True
            assert 'uri' in result
            print(f'\nDeal Neo4j Health Check: {result}')
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_schema_info(self, deal_neo4j_credentials: dict[str, str]):
        """Get current database schema information."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            schema = await client.get_schema_info()

            print(f'\nNode Labels: {schema["labels"]}')
            print(f'Relationship Types: {schema["relationship_types"]}')
            print(f'Indexes: {len(schema["indexes"])} total')
            for idx in schema['indexes'][:5]:
                print(f'  - {idx.get("name", "unnamed")}: {idx.get("type", "unknown")}')
        finally:
            await client.close()


class TestDealSkeletonVerification:
    """Test skeleton schema verification."""

    @pytest.mark.asyncio
    async def test_verify_skeleton_schema(self, deal_neo4j_credentials: dict[str, str]):
        """Verify skeleton constraints exist in the database."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            result = await client.verify_skeleton_schema()

            assert result['verified'] is True
            assert len(result['missing_constraints']) == 0
            assert 'Deal' in result['found_labels']

            print(f'\nSkeleton verification passed:')
            print(f'  Found constraints: {result["found_constraints"]}')
            print(f'  Found labels: {result["found_labels"]}')
        finally:
            await client.close()


class TestDealEnrichmentSchema:
    """Test enrichment schema setup."""

    @pytest.mark.asyncio
    async def test_setup_schema(self, deal_neo4j_credentials: dict[str, str]):
        """Test creating enrichment constraints and indexes."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            result = await client.setup_schema()

            print(f'\nCreated constraints: {result["constraints"]}')
            print(f'Created indexes: {result["indexes"]}')
            print(f'Created vector indexes: {result["vector_indexes"]}')

            # Verify the enrichment indexes now exist
            schema = await client.get_schema_info()
            index_names = {idx.get('name', '') for idx in schema['indexes']}

            # Our enrichment additions should be present
            assert 'dealversion_unique' in index_names or any(
                'dealversion' in name.lower() for name in index_names
            )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_setup_schema_idempotent(self, deal_neo4j_credentials: dict[str, str]):
        """Running setup_schema() twice should not raise errors."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            result1 = await client.setup_schema()
            result2 = await client.setup_schema()

            # Both should complete without errors
            assert isinstance(result1, dict)
            assert isinstance(result2, dict)
            print('\nIdempotent schema setup: both runs completed without error')
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_does_not_create_skeleton_constraints(
        self, deal_neo4j_credentials: dict[str, str]
    ):
        """Verify setup_schema() only creates enrichment items, not skeleton."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            result = await client.setup_schema()

            # None of the skeleton constraint names should appear in our created list
            skeleton_names = {'deal_unique', 'interaction_unique', 'account_unique', 'contact_unique'}
            created_names = set(result['constraints'])
            assert created_names.isdisjoint(skeleton_names), (
                f'setup_schema() should not create skeleton constraints, '
                f'but created: {created_names & skeleton_names}'
            )
            print(f'\nConfirmed: no skeleton constraints in created list: {result["constraints"]}')
        finally:
            await client.close()


class TestDealVectorSetup:
    """Test vector index configuration."""

    @pytest.mark.asyncio
    async def test_vector_indexes_exist(self, deal_neo4j_credentials: dict[str, str]):
        """Verify Deal vector indexes exist after schema setup."""
        client = DealNeo4jClient(**deal_neo4j_credentials)
        try:
            await client.connect()
            await client.setup_schema()

            schema = await client.get_schema_info()
            vector_indexes = [
                idx for idx in schema['indexes'] if idx.get('type') == 'VECTOR'
            ]
            vector_names = {idx.get('name', '') for idx in vector_indexes}

            print(f'\nVector indexes found: {vector_names}')

            assert client.DEAL_VECTOR_INDEX_NAME in vector_names
            assert client.DEAL_VECTOR_CURRENT_INDEX_NAME in vector_names
        finally:
            await client.close()
