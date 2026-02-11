"""
Integration tests for the graph repository.

These tests verify CRUD operations and relationship management:
- Account, Interaction, ActionItem creation
- EXTRACTED_FROM, OWNED_BY relationships
- Owner resolution with aliases
- Version snapshots

Run with: pytest tests/test_repository.py -v
"""

import uuid
from datetime import datetime

import pytest

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.models.action_item import ActionItem, ActionItemStatus
from action_item_graph.models.entities import Interaction, InteractionType
from action_item_graph.repository import ActionItemRepository


class TestAccountOperations:
    """Test Account node operations."""

    @pytest.mark.asyncio
    async def test_ensure_account_creates_new(
        self, neo4j_credentials: dict, sample_tenant_id: str
    ):
        """Test that ensure_account creates a new account when none exists."""
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)
            account_id = f"acct_test_{uuid.uuid4().hex[:8]}"

            # Ensure account (should create)
            account = await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
                name="Test Company Inc",
            )

            print(f"\nCreated account: {account}")
            assert account['account_id'] == account_id
            assert account['tenant_id'] == sample_tenant_id
            assert account['name'] == "Test Company Inc"

        finally:
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_ensure_account_idempotent(
        self, neo4j_credentials: dict, sample_tenant_id: str
    ):
        """Test that ensure_account is idempotent."""
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)
            account_id = f"acct_idempotent_{uuid.uuid4().hex[:8]}"

            # Create first time
            account1 = await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
                name="Original Name",
            )

            # Call again (should return same, not change name)
            account2 = await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
                name="Different Name",
            )

            assert account1['account_id'] == account2['account_id']
            assert account2['name'] == "Original Name"  # Name not changed
            print(f"\nIdempotent ensure returned same account: {account2['account_id']}")

        finally:
            await neo4j.close()


class TestInteractionOperations:
    """Test Interaction node operations."""

    @pytest.mark.asyncio
    async def test_create_interaction(
        self, neo4j_credentials: dict, sample_tenant_id: str, sample_account_id: str
    ):
        """Test creating an Interaction node."""
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            # Ensure account exists first
            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )

            # Create interaction
            interaction_id = uuid.uuid4()
            interaction = Interaction(
                interaction_id=interaction_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                interaction_type=InteractionType.TRANSCRIPT,
                title="Q1 Review Call",
                content_text="Discussion about Q1 results...",
                timestamp=datetime.now(),
            )

            created = await repo.create_interaction(interaction)

            print(f"\nCreated interaction: {created['interaction_id']}")
            assert created['interaction_id'] == str(interaction_id)
            assert created['title'] == "Q1 Review Call"

        finally:
            await neo4j.close()


class TestActionItemOperations:
    """Test ActionItem node operations."""

    @pytest.mark.asyncio
    async def test_create_action_item(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test creating an ActionItem with embeddings."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )

            # Generate embedding
            embedding = await openai.create_embedding(
                "Schedule follow-up meeting with the team"
            )

            item_id = uuid.uuid4()
            item = ActionItem(
                id=item_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Schedule follow-up meeting with the team",
                summary="Schedule team follow-up meeting",
                owner="John",
                embedding=embedding,
                embedding_current=embedding,
            )

            created = await repo.create_action_item(item)

            print(f"\nCreated ActionItem: {created['action_item_id']}")
            assert created['action_item_id'] == str(item_id)
            assert created['summary'] == "Schedule team follow-up meeting"
            assert created['embedding'] is not None

            # Verify retrieval
            retrieved = await repo.get_action_item(str(item_id), tenant_uuid)
            assert retrieved is not None
            assert retrieved['owner'] == "John"

        finally:
            await openai.close()
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_update_action_item(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test updating an ActionItem's properties."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )

            # Create item
            embedding = await openai.create_embedding("Draft the proposal")
            item_id = uuid.uuid4()
            item = ActionItem(
                id=item_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Draft the proposal",
                summary="Draft proposal",
                owner="Sarah",
                version=1,
                embedding=embedding,
                embedding_current=embedding,
            )
            await repo.create_action_item(item)

            # Update item
            updated = await repo.update_action_item(
                action_item_id=str(item_id),
                tenant_id=tenant_uuid,
                updates={
                    'summary': 'Draft and finalize proposal',
                    'evolution_summary': 'Scope expanded to include finalization',
                },
            )

            print(f"\nUpdated ActionItem: {updated}")
            assert updated['summary'] == 'Draft and finalize proposal'
            assert updated['version'] == 2  # Incremented

        finally:
            await openai.close()
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_update_action_item_status(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test updating only the status of an ActionItem."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )

            # Create item
            embedding = await openai.create_embedding("Send email to client")
            item_id = uuid.uuid4()
            item = ActionItem(
                id=item_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Send email to client",
                summary="Send client email",
                owner="John",
                status=ActionItemStatus.OPEN,
                embedding=embedding,
                embedding_current=embedding,
            )
            await repo.create_action_item(item)

            # Update status
            updated = await repo.update_action_item_status(
                action_item_id=str(item_id),
                tenant_id=tenant_uuid,
                status=ActionItemStatus.COMPLETED,
            )

            print(f"\nStatus updated: {updated['status']}")
            assert updated['status'] == 'completed'

        finally:
            await openai.close()
            await neo4j.close()


class TestRelationships:
    """Test relationship operations."""

    @pytest.mark.asyncio
    async def test_link_to_interaction(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test EXTRACTED_FROM relationship creation."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )

            # Create interaction
            interaction_id = uuid.uuid4()
            interaction = Interaction(
                interaction_id=interaction_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                interaction_type=InteractionType.TRANSCRIPT,
                content_text="Test transcript",
                timestamp=datetime.now(),
            )
            await repo.create_interaction(interaction)

            # Create action item
            embedding = await openai.create_embedding("Follow up with client")
            item_id = uuid.uuid4()
            item = ActionItem(
                id=item_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Follow up with client",
                summary="Client follow-up",
                owner="Sarah",
                embedding=embedding,
                embedding_current=embedding,
            )
            await repo.create_action_item(item)

            # Create relationship
            linked = await repo.link_to_interaction(
                action_item_id=str(item_id),
                interaction_id=str(interaction_id),
                tenant_id=tenant_uuid,
            )

            assert linked is True
            print(f"\nLinked ActionItem to Interaction: {linked}")

        finally:
            await openai.close()
            await neo4j.close()


class TestOwnerResolution:
    """Test Owner resolution with aliases."""

    @pytest.mark.asyncio
    async def test_create_new_owner(
        self, neo4j_credentials: dict, sample_tenant_id: str
    ):
        """Test creating a new Owner when none exists."""
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            # Generate unique name to ensure new owner
            unique_name = f"TestOwner_{uuid.uuid4().hex[:8]}"

            owner = await repo.resolve_or_create_owner(
                tenant_id=tenant_uuid,
                owner_name=unique_name,
            )

            print(f"\nCreated owner: {owner}")
            assert owner['canonical_name'] == unique_name
            assert owner['aliases'] == []

        finally:
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_resolve_existing_owner_exact_match(
        self, neo4j_credentials: dict, sample_tenant_id: str
    ):
        """Test resolving an existing Owner by exact name match."""
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            unique_name = f"JohnSmith_{uuid.uuid4().hex[:8]}"

            # Create owner first
            owner1 = await repo.resolve_or_create_owner(
                tenant_id=tenant_uuid,
                owner_name=unique_name,
            )

            # Resolve again with same name
            owner2 = await repo.resolve_or_create_owner(
                tenant_id=tenant_uuid,
                owner_name=unique_name,
            )

            assert owner1['owner_id'] == owner2['owner_id']
            print(f"\nResolved to same owner: {owner2['owner_id']}")

        finally:
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_resolve_owner_adds_alias(
        self, neo4j_credentials: dict, sample_tenant_id: str
    ):
        """Test that partial name match adds alias to existing owner.

        This test verifies that searching for a partial name that is CONTAINED
        in an existing owner's canonical_name will match that owner and add
        the partial as an alias.

        Example: If owner has canonical_name "AliceJohnsonXyz123", searching
        for "AliceJohnson" should match because "alicejohnsonxyz123" CONTAINS
        "alicejohnson".
        """
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            # Use a completely unique base name that won't match any existing owners
            # We use the full UUID to guarantee uniqueness across test runs
            unique_id = uuid.uuid4().hex

            # Full name: "UniqueOwner_{uuid}_FullName"
            # Partial:   "UniqueOwner_{uuid}"
            # The partial is CONTAINED in the full name
            full_name = f"UniqueOwner{unique_id}FullName"
            partial_name = f"UniqueOwner{unique_id}"

            # First, create the owner with the FULL name
            owner1 = await repo.resolve_or_create_owner(
                tenant_id=tenant_uuid,
                owner_name=full_name,
            )
            print(f"\nCreated owner with full_name: {full_name}")
            print(f"Owner 1 canonical_name: {owner1.get('canonical_name')}")

            # Verify it was actually created (not matched to existing)
            assert owner1.get('canonical_name') == full_name, (
                f"Expected new owner with canonical_name={full_name}, "
                f"but got {owner1.get('canonical_name')}. "
                "This might mean there's test data pollution."
            )

            # Now search with the partial name
            # full_name CONTAINS partial_name, so it should match
            owner2 = await repo.resolve_or_create_owner(
                tenant_id=tenant_uuid,
                owner_name=partial_name,
            )
            print(f"Searched with partial_name: {partial_name}")
            print(f"Owner 2 canonical_name: {owner2.get('canonical_name')}")

            # Should be same owner (full_name CONTAINS partial_name)
            assert owner1['owner_id'] == owner2['owner_id'], (
                f"Expected partial name search to match existing owner. "
                f"Full name: {full_name}, Partial: {partial_name}, "
                f"Owner1 ID: {owner1['owner_id']}, Owner2 ID: {owner2['owner_id']}"
            )

            # Check that alias was added
            updated = await repo.get_owner_by_name(tenant_uuid, full_name)
            assert updated is not None, "Owner should exist after creation"
            print(f"Owner aliases after partial search: {updated.get('aliases', [])}")
            aliases = updated.get('aliases', [])
            assert aliases is not None and partial_name in aliases

        finally:
            await neo4j.close()


class TestVersionSnapshots:
    """Test ActionItemVersion creation."""

    @pytest.mark.asyncio
    async def test_create_version_snapshot(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test creating a version snapshot before update."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)

            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )

            # Create item
            embedding = await openai.create_embedding("Prepare presentation")
            item_id = uuid.uuid4()
            item = ActionItem(
                id=item_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Prepare presentation",
                summary="Prep presentation",
                owner="Sarah",
                status=ActionItemStatus.OPEN,
                embedding=embedding,
                embedding_current=embedding,
            )
            await repo.create_action_item(item)

            # Create version snapshot
            interaction_id = uuid.uuid4()
            version = await repo.create_version_snapshot(
                action_item_id=str(item_id),
                tenant_id=tenant_uuid,
                change_summary="Status being updated to in_progress",
                source_interaction_id=interaction_id,
            )

            print(f"\nCreated version snapshot: {version}")
            assert version['version'] == 1
            assert version['status'] == 'open'  # Captured old status
            assert version['change_summary'] == "Status being updated to in_progress"

            # Verify history retrieval
            history = await repo.get_action_item_history(str(item_id), tenant_uuid)
            assert len(history) == 1
            print(f"History count: {len(history)}")

        finally:
            await openai.close()
            await neo4j.close()


class TestQueryOperations:
    """Test query operations."""

    @pytest.mark.asyncio
    async def test_get_action_items_for_account(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test retrieving all action items for an account."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)
            account_id = f"acct_query_test_{uuid.uuid4().hex[:8]}"

            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
                name="Query Test Account",
            )

            # Create multiple items
            for i in range(3):
                embedding = await openai.create_embedding(f"Task {i}")
                item = ActionItem(
                    id=uuid.uuid4(),
                    tenant_id=tenant_uuid,
                    account_id=account_id,
                    action_item_text=f"Task {i} text",
                    summary=f"Task {i}",
                    owner="TestOwner",
                    embedding=embedding,
                    embedding_current=embedding,
                )
                await repo.create_action_item(item)

            # Query items
            items = await repo.get_action_items_for_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
            )

            print(f"\nFound {len(items)} items for account")
            assert len(items) == 3

        finally:
            await openai.close()
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_get_action_items_with_status_filter(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test retrieving action items with status filter."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            repo = ActionItemRepository(neo4j)
            tenant_uuid = uuid.UUID(sample_tenant_id)
            account_id = f"acct_status_test_{uuid.uuid4().hex[:8]}"

            await repo.ensure_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
            )

            # Create items with different statuses
            for i, status in enumerate([
                ActionItemStatus.OPEN,
                ActionItemStatus.OPEN,
                ActionItemStatus.COMPLETED,
            ]):
                embedding = await openai.create_embedding(f"Status task {i}")
                item = ActionItem(
                    id=uuid.uuid4(),
                    tenant_id=tenant_uuid,
                    account_id=account_id,
                    action_item_text=f"Status task {i}",
                    summary=f"Status task {i}",
                    owner="TestOwner",
                    status=status,
                    embedding=embedding,
                    embedding_current=embedding,
                )
                await repo.create_action_item(item)

            # Query only open items
            open_items = await repo.get_action_items_for_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
                status=ActionItemStatus.OPEN,
            )

            print(f"\nFound {len(open_items)} open items")
            assert len(open_items) == 2

            # Query completed items
            completed_items = await repo.get_action_items_for_account(
                tenant_id=tenant_uuid,
                account_id=account_id,
                status=ActionItemStatus.COMPLETED,
            )
            assert len(completed_items) == 1

        finally:
            await openai.close()
            await neo4j.close()
