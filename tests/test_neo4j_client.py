"""
Live integration tests for Neo4j client.

These tests hit the actual Neo4j database and require credentials to be set.
Run with: pytest tests/test_neo4j_client.py -v

WARNING: Some tests create/modify data. Use a test database if possible.
"""

import uuid

import pytest

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.models.action_item import ActionItem, ActionItemStatus


class TestNeo4jHealth:
    """Test Neo4j connectivity."""

    @pytest.mark.asyncio
    async def test_health_check(self, neo4j_credentials: dict[str, str]):
        """Verify we can connect to Neo4j."""
        client = Neo4jClient(**neo4j_credentials)
        try:
            result = await client.health_check()
            assert result['healthy'] is True
            assert 'uri' in result
            print(f"\nNeo4j Health Check: {result}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_schema_info(self, neo4j_credentials: dict[str, str]):
        """Get current database schema information."""
        client = Neo4jClient(**neo4j_credentials)
        try:
            await client.connect()
            schema = await client.get_schema_info()

            print(f"\nNode Labels: {schema['labels']}")
            print(f"Relationship Types: {schema['relationship_types']}")
            print(f"Indexes: {len(schema['indexes'])} total")
            for idx in schema['indexes'][:5]:  # Show first 5
                print(f"  - {idx.get('name', 'unnamed')}: {idx.get('type', 'unknown')}")
        finally:
            await client.close()


class TestNeo4jSchema:
    """Test schema setup."""

    @pytest.mark.asyncio
    async def test_setup_schema(self, neo4j_credentials: dict[str, str]):
        """Test creating constraints and indexes."""
        client = Neo4jClient(**neo4j_credentials)
        try:
            await client.connect()
            result = await client.setup_schema()

            print(f"\nCreated constraints: {result['constraints']}")
            print(f"Created indexes: {result['indexes']}")
            print(f"Created vector indexes: {result['vector_indexes']}")

            # Verify we can query the indexes
            schema = await client.get_schema_info()
            assert len(schema['indexes']) > 0
        finally:
            await client.close()


class TestNeo4jCRUD:
    """Test basic CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_and_query_node(
        self,
        neo4j_credentials: dict[str, str],
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test creating and querying an ActionItem node."""
        client = Neo4jClient(**neo4j_credentials)
        try:
            await client.connect()

            # Create a test action item
            test_id = str(uuid.uuid4())
            action_item = ActionItem(
                id=uuid.UUID(test_id),
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id=sample_account_id,
                action_item_text="Test action item - send proposal",
                summary="Send the proposal document",
                owner="John",
                status=ActionItemStatus.OPEN,
            )

            # Insert the node
            props = action_item.to_neo4j_properties()
            await client.execute_write(
                """
                MERGE (a:ActionItem {id: $id})
                SET a += $props
                RETURN a.id as id
                """,
                {'id': test_id, 'props': props},
            )

            # Query it back
            result = await client.execute_query(
                "MATCH (a:ActionItem {id: $id}) RETURN a",
                {'id': test_id},
            )

            assert len(result) == 1
            node = result[0]['a']
            assert node['action_item_text'] == "Test action item - send proposal"
            assert node['owner'] == "John"
            assert node['tenant_id'] == sample_tenant_id

            print(f"\nCreated and retrieved ActionItem: {node['id']}")
            print(f"  Text: {node['action_item_text']}")
            print(f"  Owner: {node['owner']}")

            # Cleanup
            await client.execute_write(
                "MATCH (a:ActionItem {id: $id}) DELETE a",
                {'id': test_id},
            )
            print("  (Cleaned up test node)")
        finally:
            await client.close()


class TestNeo4jVectorSearch:
    """Test vector similarity search."""

    @pytest.mark.asyncio
    async def test_vector_search_setup(self, neo4j_credentials: dict[str, str]):
        """Verify vector indexes exist and are queryable."""
        client = Neo4jClient(**neo4j_credentials)
        try:
            await client.connect()

            # Ensure schema is set up
            await client.setup_schema()

            # Check that vector indexes exist
            schema = await client.get_schema_info()
            vector_indexes = [
                idx for idx in schema['indexes'] if idx.get('type') == 'VECTOR'
            ]

            print(f"\nVector indexes found: {len(vector_indexes)}")
            for idx in vector_indexes:
                print(f"  - {idx.get('name')}: {idx.get('state', 'unknown state')}")

            # We should have at least our two indexes
            assert len(vector_indexes) >= 2
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_vector_search_with_data(
        self,
        neo4j_credentials: dict[str, str],
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test vector search with actual embeddings."""
        # This test requires OpenAI for embeddings
        import os

        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            pytest.skip('OPENAI_API_KEY not set - skipping vector search test')

        from action_item_graph.clients.openai_client import OpenAIClient

        neo4j_client = Neo4jClient(**neo4j_credentials)
        openai_client = OpenAIClient(api_key=openai_key)

        try:
            await neo4j_client.connect()
            await neo4j_client.setup_schema()

            # Create test action items with embeddings
            test_items = [
                {
                    'text': 'Send the updated proposal deck to Acme Corp',
                    'owner': 'Sarah',
                },
                {
                    'text': 'Schedule demo meeting with technical team',
                    'owner': 'John',
                },
                {
                    'text': 'Review contract terms with legal department',
                    'owner': 'Sarah',
                },
            ]

            test_ids = []
            for item in test_items:
                test_id = str(uuid.uuid4())
                test_ids.append(test_id)

                # Generate embedding
                embedding = await openai_client.create_embedding(item['text'])

                # Create action item
                action_item = ActionItem(
                    id=uuid.UUID(test_id),
                    tenant_id=uuid.UUID(sample_tenant_id),
                    account_id=sample_account_id,
                    action_item_text=item['text'],
                    summary=item['text'],
                    owner=item['owner'],
                    embedding=embedding,
                    embedding_current=embedding,
                )

                props = action_item.to_neo4j_properties()
                await neo4j_client.execute_write(
                    """
                    MERGE (a:ActionItem {id: $id})
                    SET a += $props
                    """,
                    {'id': test_id, 'props': props},
                )

            print(f"\nCreated {len(test_ids)} test action items with embeddings")

            # Search for similar items
            query_text = "Email the proposal document to the customer"
            query_embedding = await openai_client.create_embedding(query_text)

            results = await neo4j_client.vector_search(
                embedding=query_embedding,
                tenant_id=sample_tenant_id,
                limit=5,
                min_score=0.5,
            )

            print(f"\nQuery: '{query_text}'")
            print(f"Found {len(results)} similar action items:")
            for r in results:
                print(f"  - Score {r['score']:.4f}: {r['node'].get('action_item_text', 'N/A')}")

            # The proposal-related item should rank highest
            if results:
                assert 'proposal' in results[0]['node']['action_item_text'].lower()

            # Cleanup
            for test_id in test_ids:
                await neo4j_client.execute_write(
                    "MATCH (a:ActionItem {id: $id}) DELETE a",
                    {'id': test_id},
                )
            print("  (Cleaned up test nodes)")

        finally:
            await neo4j_client.close()
            await openai_client.close()


class TestNeo4jRelationships:
    """Test relationship operations."""

    @pytest.mark.asyncio
    async def test_create_account_with_interaction(
        self,
        neo4j_credentials: dict[str, str],
        sample_tenant_id: str,
    ):
        """Test creating Account -> Interaction relationship."""
        client = Neo4jClient(**neo4j_credentials)
        try:
            await client.connect()

            account_id = f'acct_test_{uuid.uuid4().hex[:8]}'
            interaction_id = str(uuid.uuid4())

            # Create account and interaction with relationship
            await client.execute_write(
                """
                MERGE (acc:Account {account_id: $account_id})
                SET acc.tenant_id = $tenant_id, acc.name = 'Test Account'

                CREATE (int:Interaction {
                    interaction_id: $interaction_id,
                    tenant_id: $tenant_id,
                    account_id: $account_id,
                    interaction_type: 'transcript',
                    content_text: 'Test transcript content',
                    timestamp: datetime()
                })

                MERGE (acc)-[:HAS_INTERACTION]->(int)
                RETURN acc.account_id as account_id, int.interaction_id as interaction_id
                """,
                {
                    'account_id': account_id,
                    'interaction_id': interaction_id,
                    'tenant_id': sample_tenant_id,
                },
            )

            # Query the relationship
            result = await client.execute_query(
                """
                MATCH (acc:Account {account_id: $account_id})-[:HAS_INTERACTION]->(int:Interaction)
                RETURN acc.name as account_name, int.interaction_type as type
                """,
                {'account_id': account_id},
            )

            assert len(result) == 1
            assert result[0]['account_name'] == 'Test Account'
            assert result[0]['type'] == 'transcript'

            print(f"\nCreated Account -> Interaction relationship")
            print(f"  Account: {account_id}")
            print(f"  Interaction: {interaction_id}")

            # Cleanup
            await client.execute_write(
                """
                MATCH (acc:Account {account_id: $account_id})
                OPTIONAL MATCH (acc)-[:HAS_INTERACTION]->(int:Interaction)
                DETACH DELETE acc, int
                """,
                {'account_id': account_id},
            )
            print("  (Cleaned up test nodes)")
        finally:
            await client.close()
