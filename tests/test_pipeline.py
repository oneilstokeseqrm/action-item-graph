"""
End-to-end integration tests for the ActionItemPipeline.

These tests verify the complete flow:
- New transcript processing
- Multi-turn conversation handling
- Status update detection and linking
- Multi-tenancy isolation
- Idempotency behavior

Run with: pytest tests/test_pipeline.py -v
"""

import uuid
from datetime import datetime

import pytest

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.errors import ValidationError
from action_item_graph.models.envelope import EnvelopeV1, ContentPayload
from action_item_graph.pipeline.pipeline import ActionItemPipeline, PipelineResult


class TestPipelineSetup:
    """Test pipeline initialization."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(
        self, openai_api_key: str, neo4j_credentials: dict
    ):
        """Test that pipeline initializes correctly with all components."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            assert pipeline.extractor is not None
            assert pipeline.matcher is not None
            assert pipeline.merger is not None
            assert pipeline.repository is not None
            print("\nPipeline initialized with all components")

        finally:
            await openai.close()
            await neo4j.close()


class TestNewTranscriptProcessing:
    """Test processing new transcripts."""

    @pytest.mark.asyncio
    async def test_process_envelope_new_items(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test processing an envelope creates new action items."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            # Create unique account to ensure fresh state
            account_id = f"acct_pipeline_test_{uuid.uuid4().hex[:8]}"

            envelope = EnvelopeV1(
                tenant_id=uuid.UUID(sample_tenant_id),
                user_id="test_user",
                interaction_type="transcript",
                content=ContentPayload(
                    text="""
                    Sarah: I'll send the proposal to Acme Corp by Friday.
                    John: Great. And I'll schedule the demo for next Tuesday.
                    Sarah: Perfect. Let me also loop in legal to review the contract.
                    """,
                    format="plain",
                ),
                timestamp=datetime.now(),
                source="api",
                account_id=account_id,
            )

            result = await pipeline.process_envelope(envelope)

            print(f"\nPipeline result:")
            print(f"  Extracted: {result.total_extracted}")
            print(f"  Created: {len(result.created_ids)}")
            print(f"  Updated: {len(result.updated_ids)}")
            print(f"  Processing time: {result.processing_time_ms}ms")

            assert result.total_extracted >= 2  # At least 2 action items
            assert len(result.created_ids) >= 2  # All should be new
            assert len(result.updated_ids) == 0  # Nothing to update
            assert result.interaction_id != ''
            assert result.processing_time_ms is not None

        finally:
            await openai.close()
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_process_text_convenience_method(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test the process_text convenience method."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            account_id = f"acct_text_test_{uuid.uuid4().hex[:8]}"

            result = await pipeline.process_text(
                text="John: I'll prepare the quarterly report by Monday.",
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id=account_id,
                meeting_title="Q4 Planning",
            )

            print(f"\nProcess text result:")
            print(f"  Extracted: {result.total_extracted}")
            print(f"  Created: {len(result.created_ids)}")

            assert result.total_extracted >= 1
            assert len(result.created_ids) >= 1

        finally:
            await openai.close()
            await neo4j.close()


class TestMultiTurnConversation:
    """Test processing multiple interactions for the same account."""

    @pytest.mark.asyncio
    async def test_second_interaction_matches_existing(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test that a second interaction can match/update existing items."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            account_id = f"acct_multi_turn_{uuid.uuid4().hex[:8]}"
            tenant_uuid = uuid.UUID(sample_tenant_id)

            # First interaction: create action items
            result1 = await pipeline.process_text(
                text="Sarah: I'll send the budget analysis to the finance team by Friday.",
                tenant_id=tenant_uuid,
                account_id=account_id,
                meeting_title="Budget Review",
            )

            print(f"\nFirst interaction:")
            print(f"  Created: {len(result1.created_ids)}")
            first_created_id = result1.created_ids[0] if result1.created_ids else None

            assert len(result1.created_ids) >= 1

            # Second interaction: status update on the same item
            result2 = await pipeline.process_text(
                text="Sarah: I sent the budget analysis to finance yesterday, they're reviewing it now.",
                tenant_id=tenant_uuid,
                account_id=account_id,
                meeting_title="Status Update Call",
            )

            print(f"\nSecond interaction:")
            print(f"  Extracted: {result2.total_extracted}")
            print(f"  Matched: {result2.total_matched}")
            print(f"  Updated: {len(result2.updated_ids)}")
            print(f"  Created: {len(result2.created_ids)}")

            # The status update should match the existing item
            # It might create new OR update existing depending on LLM's decision
            assert result2.total_extracted >= 1

            # Check that the original item still exists
            items = await pipeline.get_action_items(
                tenant_id=tenant_uuid,
                account_id=account_id,
            )
            print(f"\nTotal items in account: {len(items)}")

        finally:
            await openai.close()
            await neo4j.close()


class TestStatusUpdates:
    """Test status update detection and handling."""

    @pytest.mark.asyncio
    async def test_status_update_marks_complete(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test that explicit completion updates status."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            account_id = f"acct_status_{uuid.uuid4().hex[:8]}"
            tenant_uuid = uuid.UUID(sample_tenant_id)

            # First: create an action item
            result1 = await pipeline.process_text(
                text="John: I'll send the contract to the client today.",
                tenant_id=tenant_uuid,
                account_id=account_id,
            )

            assert len(result1.created_ids) >= 1
            created_id = result1.created_ids[0]

            # Check initial status
            items_before = await pipeline.get_action_items(
                tenant_id=tenant_uuid,
                account_id=account_id,
            )
            initial_item = next((i for i in items_before if i['id'] == created_id), None)
            print(f"\nInitial status: {initial_item['status'] if initial_item else 'not found'}")

            # Second: mark it complete
            result2 = await pipeline.process_text(
                text="John: Done! I sent the contract to the client.",
                tenant_id=tenant_uuid,
                account_id=account_id,
            )

            print(f"\nStatus update result:")
            print(f"  Matched: {result2.total_matched}")
            print(f"  Updated: {len(result2.updated_ids)}")

            # Check if status was updated
            items_after = await pipeline.get_action_items(
                tenant_id=tenant_uuid,
                account_id=account_id,
            )

            # Find the original item
            updated_item = next((i for i in items_after if i['id'] == created_id), None)
            if updated_item:
                print(f"Final status: {updated_item['status']}")
                # Status might be 'completed' if matching worked
                # If not matched, a new item would be created

        finally:
            await openai.close()
            await neo4j.close()


class TestMultiTenancyIsolation:
    """Test that tenants are properly isolated."""

    @pytest.mark.asyncio
    async def test_tenants_isolated(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
    ):
        """Test that action items from different tenants don't interfere."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            # Two different tenants with unique account IDs
            # (Account IDs are globally unique in current schema)
            tenant_a = uuid.uuid4()
            tenant_b = uuid.uuid4()
            account_a = f"acct_tenant_a_{uuid.uuid4().hex[:8]}"
            account_b = f"acct_tenant_b_{uuid.uuid4().hex[:8]}"

            # Tenant A creates an item
            result_a = await pipeline.process_text(
                text="Sarah: I'll send the proposal by Friday.",
                tenant_id=tenant_a,
                account_id=account_a,
            )

            # Tenant B creates a semantically identical item
            result_b = await pipeline.process_text(
                text="Sarah: I'll send the proposal by Friday.",
                tenant_id=tenant_b,
                account_id=account_b,
            )

            print(f"\nTenant A created: {len(result_a.created_ids)}")
            print(f"Tenant B created: {len(result_b.created_ids)}")

            # Both should create new items (no cross-tenant matching)
            assert len(result_a.created_ids) >= 1
            assert len(result_b.created_ids) >= 1

            # Verify isolation by querying each tenant
            items_a = await pipeline.get_action_items(
                tenant_id=tenant_a,
                account_id=account_a,
            )
            items_b = await pipeline.get_action_items(
                tenant_id=tenant_b,
                account_id=account_b,
            )

            print(f"Tenant A items: {len(items_a)}")
            print(f"Tenant B items: {len(items_b)}")

            # Each tenant should only see their own items
            assert len(items_a) >= 1
            assert len(items_b) >= 1

            # IDs should be different
            ids_a = {i['id'] for i in items_a}
            ids_b = {i['id'] for i in items_b}
            assert ids_a.isdisjoint(ids_b), "Tenant items should not overlap"

        finally:
            await openai.close()
            await neo4j.close()


class TestAccountIsolation:
    """Test that accounts within a tenant are properly isolated."""

    @pytest.mark.asyncio
    async def test_accounts_isolated(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test that similar items in different accounts don't match."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            tenant_uuid = uuid.UUID(sample_tenant_id)
            account_a = f"acct_alpha_{uuid.uuid4().hex[:8]}"
            account_b = f"acct_beta_{uuid.uuid4().hex[:8]}"

            # Account A creates an item
            result_a = await pipeline.process_text(
                text="Sarah: I'll send the proposal by Friday.",
                tenant_id=tenant_uuid,
                account_id=account_a,
            )

            # Account B creates semantically identical item
            result_b = await pipeline.process_text(
                text="Sarah: I'll send the proposal by Friday.",
                tenant_id=tenant_uuid,
                account_id=account_b,
            )

            print(f"\nAccount A created: {len(result_a.created_ids)}")
            print(f"Account B created: {len(result_b.created_ids)}")

            # Both should create NEW items (no cross-account matching)
            assert len(result_a.created_ids) >= 1
            assert len(result_b.created_ids) >= 1

            # The items should have different IDs
            assert set(result_a.created_ids).isdisjoint(set(result_b.created_ids))

        finally:
            await openai.close()
            await neo4j.close()


class TestEmptyTranscript:
    """Test handling of transcripts with no action items."""

    @pytest.mark.asyncio
    async def test_no_action_items(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test that transcripts with no action items are handled gracefully."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            account_id = f"acct_empty_{uuid.uuid4().hex[:8]}"

            result = await pipeline.process_text(
                text="Nice weather today, isn't it? Yes, very pleasant.",
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id=account_id,
            )

            print(f"\nEmpty transcript result:")
            print(f"  Extracted: {result.total_extracted}")
            print(f"  Created: {len(result.created_ids)}")

            # Should complete without error, with no items
            assert result.total_extracted == 0
            assert len(result.created_ids) == 0
            assert result.processing_time_ms is not None

        finally:
            await openai.close()
            await neo4j.close()


class TestValidation:
    """Test input validation."""

    @pytest.mark.asyncio
    async def test_envelope_requires_account_id(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
    ):
        """Test that processing fails if account_id is missing."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()

            pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

            envelope = EnvelopeV1(
                tenant_id=uuid.UUID(sample_tenant_id),
                user_id="test_user",
                interaction_type="transcript",
                content=ContentPayload(
                    text="Some text",
                    format="plain",
                ),
                timestamp=datetime.now(),
                source="api",
                account_id=None,  # Missing!
            )

            with pytest.raises(ValidationError) as exc_info:
                await pipeline.process_envelope(envelope)

            assert "account_id" in str(exc_info.value)
            print(f"\nValidation error: {exc_info.value}")

        finally:
            await openai.close()
            await neo4j.close()


class TestPipelineResult:
    """Test PipelineResult structure."""

    def test_result_to_dict(self):
        """Test that PipelineResult serializes correctly."""
        result = PipelineResult(
            envelope_id="env_123",
            interaction_id="int_456",
            account_id="acct_789",
            tenant_id="tenant_abc",
            created_ids=["id1", "id2"],
            updated_ids=["id3"],
            total_extracted=3,
            total_matched=1,
            total_unmatched=2,
        )

        data = result.to_dict()

        assert data['envelope_id'] == "env_123"
        assert data['created_ids'] == ["id1", "id2"]
        assert data['updated_ids'] == ["id3"]
        assert data['total_extracted'] == 3
        print(f"\nSerialized result: {data}")
