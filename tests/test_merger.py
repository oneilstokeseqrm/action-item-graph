"""
Integration tests for action item merger service.

These tests verify the merger correctly executes match decisions:
- create_new: Creates new ActionItem with relationships
- update_status: Updates status only (no LLM)
- merge: Full LLM synthesis with content update
- link_related: Creates new and links to existing

Run with: pytest tests/test_merger.py -v
"""

import uuid
from datetime import datetime

import pytest

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.models.action_item import ActionItem, ActionItemStatus
from action_item_graph.models.entities import Interaction, InteractionType
from action_item_graph.pipeline.merger import ActionItemMerger, MergeResult


from action_item_graph.prompts.extract_action_items import ExtractedActionItem, ExtractedTopic, DeduplicationDecision
from action_item_graph.pipeline.matcher import MatchCandidate, MatchResult


# Helper to create a default topic for test extractions
def make_test_topic(name: str = "Test Topic", context: str = "Test context") -> ExtractedTopic:
    return ExtractedTopic(name=name, context=context)


class TestMergerSetup:
    """Test merger initialization."""

    @pytest.mark.asyncio
    async def test_merger_initialization(
        self, openai_api_key: str, neo4j_credentials: dict
    ):
        """Test that merger initializes correctly."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            merger = ActionItemMerger(neo4j_client=neo4j, openai_client=openai)

            assert merger.neo4j is neo4j
            assert merger.openai is openai
            assert merger.repository is not None
            print("\nMerger initialized successfully")

        finally:
            await openai.close()
            await neo4j.close()


class TestCreateNew:
    """Test creating new ActionItems."""

    @pytest.mark.asyncio
    async def test_create_new_action_item(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test creating a new ActionItem with full graph structure."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            merger = ActionItemMerger(neo4j_client=neo4j, openai_client=openai)

            tenant_uuid = uuid.UUID(sample_tenant_id)
            interaction_id = uuid.uuid4()
            action_item_id = uuid.uuid4()

            # Create test data
            interaction = Interaction(
                interaction_id=interaction_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                interaction_type=InteractionType.TRANSCRIPT,
                content_text="Test transcript",
                timestamp=datetime.now(),
            )

            # Create ActionItem model
            embedding = await openai.create_embedding("Send the proposal by Friday")
            action_item = ActionItem(
                id=action_item_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="I'll send the proposal by Friday",
                summary="Send proposal by Friday",
                owner="Sarah",
                embedding=embedding,
                embedding_current=embedding,
                source_interaction_id=interaction_id,
            )

            # Create extraction for match result
            extraction = ExtractedActionItem(
                action_item_text="I'll send the proposal by Friday",
                owner="Sarah",
                summary="Send proposal by Friday",
                conversation_context="Discussing deliverables",
                topic=make_test_topic("Sales Proposal", "Discussing deliverables"),
            )

            # Create unmatched result (no best_match)
            match_result = MatchResult(
                extracted_item=extraction,
                embedding=embedding,
                candidates=[],
                decisions=[],
                best_match=None,
            )

            # Ensure account and interaction exist
            await merger.repository.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                name="Test Account",
            )
            await merger.repository.create_interaction(interaction)

            # Execute creation
            result = await merger.execute_decision(
                match_result=match_result,
                interaction=interaction,
                action_item=action_item,
            )

            print(f"\nCreated ActionItem: {result.action_item_id}")
            print(f"Action: {result.action}")
            print(f"Was new: {result.was_new}")
            print(f"Details: {result.details}")

            assert result.action == 'created'
            assert result.was_new is True
            assert result.version_created is False
            assert result.linked_interaction_id == str(interaction_id)
            assert 'owner_id' in result.details

            # Verify in database
            stored = await merger.repository.get_action_item(
                str(action_item_id), tenant_uuid
            )
            assert stored is not None
            assert stored['summary'] == "Send proposal by Friday"

        finally:
            await openai.close()
            await neo4j.close()


class TestUpdateStatus:
    """Test status-only updates."""

    @pytest.mark.asyncio
    async def test_update_status_to_completed(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test updating status from open to completed."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            merger = ActionItemMerger(neo4j_client=neo4j, openai_client=openai)

            tenant_uuid = uuid.UUID(sample_tenant_id)

            # First create an ActionItem to update
            existing_id = uuid.uuid4()
            embedding = await openai.create_embedding("Send the contract to the client")
            existing_item = ActionItem(
                id=existing_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Send the contract to the client",
                summary="Send contract to client",
                owner="John",
                status=ActionItemStatus.OPEN,
                embedding=embedding,
                embedding_current=embedding,
            )

            # Create account and item
            await merger.repository.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )
            await merger.repository.create_action_item(existing_item)

            # Create interaction for the status update
            interaction_id = uuid.uuid4()
            interaction = Interaction(
                interaction_id=interaction_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                interaction_type=InteractionType.TRANSCRIPT,
                content_text="John: I sent the contract yesterday.",
                timestamp=datetime.now(),
            )
            await merger.repository.create_interaction(interaction)

            # Create status update extraction
            extraction = ExtractedActionItem(
                action_item_text="I sent the contract yesterday",
                owner="John",
                summary="Contract was sent",
                conversation_context="John confirming completion",
                is_status_update=True,
                implied_status='completed',
                topic=make_test_topic("Client Contract", "Contract delivery"),
            )

            # Create match result with update_status recommendation
            candidate = MatchCandidate(
                action_item_id=str(existing_id),
                node_properties=existing_item.to_neo4j_properties(),
                similarity_score=0.88,
                matched_via='combined',
            )
            decision = DeduplicationDecision(
                is_same_item=True,
                is_status_update=True,
                merge_recommendation='update_status',
                reasoning='Status update for existing task',
                confidence=0.95,
            )
            match_result = MatchResult(
                extracted_item=extraction,
                embedding=embedding,
                candidates=[candidate],
                decisions=[(candidate, decision)],
                best_match=(candidate, decision),
            )

            # Execute status update
            result = await merger.execute_decision(
                match_result=match_result,
                interaction=interaction,
                action_item=existing_item,  # Used for tenant_id
            )

            print(f"\nUpdated ActionItem: {result.action_item_id}")
            print(f"Action: {result.action}")
            print(f"Version created: {result.version_created}")
            print(f"Details: {result.details}")

            assert result.action == 'status_updated'
            assert result.was_new is False
            assert result.version_created is True

            # Verify version was created
            history = await merger.repository.get_action_item_history(
                str(existing_id), tenant_uuid
            )
            assert len(history) >= 1
            print(f"Version history: {len(history)} versions")

        finally:
            await openai.close()
            await neo4j.close()


class TestMergeItems:
    """Test full merge with LLM synthesis."""

    @pytest.mark.asyncio
    async def test_merge_with_new_context(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test merging when new extraction adds context."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            merger = ActionItemMerger(neo4j_client=neo4j, openai_client=openai)

            tenant_uuid = uuid.UUID(sample_tenant_id)

            # Create existing item with minimal info
            existing_id = uuid.uuid4()
            embedding = await openai.create_embedding("Send the proposal")
            existing_item = ActionItem(
                id=existing_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Send the proposal",
                summary="Send proposal",
                owner="Sarah",
                status=ActionItemStatus.OPEN,
                embedding=embedding,
                embedding_current=embedding,
            )

            await merger.repository.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )
            await merger.repository.create_action_item(existing_item)

            # Create interaction with more details
            interaction_id = uuid.uuid4()
            interaction = Interaction(
                interaction_id=interaction_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                interaction_type=InteractionType.TRANSCRIPT,
                content_text="Sarah: I'll send the pricing proposal to John by end of day.",
                timestamp=datetime.now(),
            )
            await merger.repository.create_interaction(interaction)

            # New extraction with more details
            extraction = ExtractedActionItem(
                action_item_text="I'll send the pricing proposal to John by end of day",
                owner="Sarah",
                summary="Send pricing proposal to John by EOD",
                conversation_context="Discussing timeline for deliverables",
                due_date_text="end of day",
                topic=make_test_topic("Pricing Proposal", "Timeline for deliverables"),
            )

            # Create match result with merge recommendation
            candidate = MatchCandidate(
                action_item_id=str(existing_id),
                node_properties=existing_item.to_neo4j_properties(),
                similarity_score=0.82,
                matched_via='combined',
            )
            decision = DeduplicationDecision(
                is_same_item=True,
                is_status_update=False,
                merge_recommendation='merge',
                reasoning='Same task, new extraction adds details (recipient, timeline)',
                confidence=0.88,
            )
            match_result = MatchResult(
                extracted_item=extraction,
                embedding=embedding,
                candidates=[candidate],
                decisions=[(candidate, decision)],
                best_match=(candidate, decision),
            )

            # Execute merge
            result = await merger.execute_decision(
                match_result=match_result,
                interaction=interaction,
                action_item=existing_item,
            )

            print(f"\nMerged ActionItem: {result.action_item_id}")
            print(f"Action: {result.action}")
            print(f"Version created: {result.version_created}")
            print(f"Details: {result.details}")

            assert result.action == 'merged'
            assert result.was_new is False
            assert result.version_created is True

            # Verify content was updated
            updated = await merger.repository.get_action_item(
                str(existing_id), tenant_uuid
            )
            assert updated is not None
            print(f"Updated summary: {updated.get('summary')}")
            print(f"Evolution summary: {updated.get('evolution_summary')}")

        finally:
            await openai.close()
            await neo4j.close()


class TestLinkRelated:
    """Test creating linked related items."""

    @pytest.mark.asyncio
    async def test_create_and_link_related(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test creating a new item linked to an existing related item."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            merger = ActionItemMerger(neo4j_client=neo4j, openai_client=openai)

            tenant_uuid = uuid.UUID(sample_tenant_id)

            # Create existing item
            existing_id = uuid.uuid4()
            embedding1 = await openai.create_embedding("Send the proposal")
            existing_item = ActionItem(
                id=existing_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Send the proposal",
                summary="Send proposal",
                owner="Sarah",
                embedding=embedding1,
                embedding_current=embedding1,
            )

            await merger.repository.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )
            await merger.repository.create_action_item(existing_item)

            # Create interaction
            interaction_id = uuid.uuid4()
            interaction = Interaction(
                interaction_id=interaction_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                interaction_type=InteractionType.TRANSCRIPT,
                content_text="John: I'll review the proposal once Sarah sends it.",
                timestamp=datetime.now(),
            )
            await merger.repository.create_interaction(interaction)

            # New related but different task
            new_id = uuid.uuid4()
            embedding2 = await openai.create_embedding("Review the proposal")
            new_item = ActionItem(
                id=new_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="I'll review the proposal once Sarah sends it",
                summary="Review proposal after Sarah sends",
                owner="John",
                embedding=embedding2,
                embedding_current=embedding2,
                source_interaction_id=interaction_id,
            )

            extraction = ExtractedActionItem(
                action_item_text="I'll review the proposal once Sarah sends it",
                owner="John",
                summary="Review proposal after Sarah sends",
                conversation_context="John committing to review",
                topic=make_test_topic("Proposal Review", "Reviewing proposal"),
            )

            # Create match result with link_related recommendation
            candidate = MatchCandidate(
                action_item_id=str(existing_id),
                node_properties=existing_item.to_neo4j_properties(),
                similarity_score=0.72,
                matched_via='combined',
            )
            decision = DeduplicationDecision(
                is_same_item=False,
                is_status_update=False,
                merge_recommendation='link_related',
                reasoning='Different tasks (send vs review) but clearly related',
                confidence=0.85,
            )
            match_result = MatchResult(
                extracted_item=extraction,
                embedding=embedding2,
                candidates=[candidate],
                decisions=[(candidate, decision)],
                best_match=(candidate, decision),
            )

            # Execute link
            result = await merger.execute_decision(
                match_result=match_result,
                interaction=interaction,
                action_item=new_item,
            )

            print(f"\nCreated and linked ActionItem: {result.action_item_id}")
            print(f"Action: {result.action}")
            print(f"Was new: {result.was_new}")
            print(f"Details: {result.details}")

            assert result.action == 'linked'
            assert result.was_new is True
            assert 'related_to' in result.details
            assert result.details['related_to'] == str(existing_id)

        finally:
            await openai.close()
            await neo4j.close()


class TestVersionSnapshots:
    """Test version snapshot creation."""

    @pytest.mark.asyncio
    async def test_version_created_before_update(
        self,
        openai_api_key: str,
        neo4j_credentials: dict,
        sample_tenant_id: str,
        sample_account_id: str,
    ):
        """Test that version snapshots are created before updates."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            merger = ActionItemMerger(neo4j_client=neo4j, openai_client=openai)

            tenant_uuid = uuid.UUID(sample_tenant_id)

            # Create item
            item_id = uuid.uuid4()
            embedding = await openai.create_embedding("Schedule the demo")
            item = ActionItem(
                id=item_id,
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                action_item_text="Schedule the demo",
                summary="Schedule demo",
                owner="John",
                status=ActionItemStatus.OPEN,
                embedding=embedding,
                embedding_current=embedding,
            )

            await merger.repository.ensure_account(
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
            )
            await merger.repository.create_action_item(item)

            # Check initial history (should be empty)
            history_before = await merger.repository.get_action_item_history(
                str(item_id), tenant_uuid
            )
            assert len(history_before) == 0

            # Create status update
            interaction = Interaction(
                interaction_id=uuid.uuid4(),
                tenant_id=tenant_uuid,
                account_id=sample_account_id,
                interaction_type=InteractionType.TRANSCRIPT,
                content_text="John: Demo is scheduled for next week.",
                timestamp=datetime.now(),
            )
            await merger.repository.create_interaction(interaction)

            extraction = ExtractedActionItem(
                action_item_text="Demo is scheduled for next week",
                owner="John",
                summary="Demo scheduled",
                conversation_context="John confirming progress",
                is_status_update=True,
                implied_status='in_progress',
                topic=make_test_topic("Product Demo", "Demo scheduling"),
            )

            candidate = MatchCandidate(
                action_item_id=str(item_id),
                node_properties=item.to_neo4j_properties(),
                similarity_score=0.9,
                matched_via='combined',
            )
            decision = DeduplicationDecision(
                is_same_item=True,
                is_status_update=True,
                merge_recommendation='update_status',
                reasoning='Progress update',
                confidence=0.95,
            )
            match_result = MatchResult(
                extracted_item=extraction,
                embedding=embedding,
                candidates=[candidate],
                decisions=[(candidate, decision)],
                best_match=(candidate, decision),
            )

            # Execute update
            await merger.execute_decision(
                match_result=match_result,
                interaction=interaction,
                action_item=item,
            )

            # Check version was created
            history_after = await merger.repository.get_action_item_history(
                str(item_id), tenant_uuid
            )
            assert len(history_after) == 1
            print(f"\nVersion history after update: {len(history_after)} versions")
            print(f"Version details: {history_after[0]}")

            # Verify version preserves old state
            version = history_after[0]
            assert version['status'] == 'open'  # Old status before update

        finally:
            await openai.close()
            await neo4j.close()
