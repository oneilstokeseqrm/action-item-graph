"""
Live integration tests for action item matching and deduplication.

These tests hit the actual OpenAI and Neo4j APIs.
Run with: pytest tests/test_matcher.py -v
"""

import uuid
from datetime import datetime

import pytest

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.models.action_item import ActionItem, ActionItemStatus
from action_item_graph.pipeline.extractor import ActionItemExtractor
from action_item_graph.pipeline.matcher import (
    ActionItemMatcher,
    MatchCandidate,
    MatchResult,
    match_batch,
)
from action_item_graph.prompts.extract_action_items import (
    ExtractedActionItem,
    ExtractedTopic,
    DeduplicationDecision,
)


# Helper to create a default topic for test extractions
def make_test_topic(name: str = "Test Topic", context: str = "Test context") -> ExtractedTopic:
    return ExtractedTopic(name=name, context=context)


class TestMatcherSetup:
    """Test matcher initialization and basic connectivity."""

    @pytest.mark.asyncio
    async def test_matcher_initialization(
        self, openai_api_key: str, neo4j_credentials: dict
    ):
        """Test that matcher initializes correctly."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            matcher = ActionItemMatcher(neo4j_client=neo4j, openai_client=openai)

            assert matcher.min_similarity == ActionItemMatcher.MIN_SIMILARITY_SCORE
            print(f"\nMatcher initialized with min_similarity={matcher.min_similarity}")

        finally:
            await openai.close()
            await neo4j.close()


class TestVectorSearch:
    """Test vector search for candidate matches."""

    @pytest.mark.asyncio
    async def test_find_candidates_empty_database(
        self, openai_api_key: str, neo4j_credentials: dict, sample_tenant_id: str
    ):
        """Test that search returns empty when no matching items exist."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            matcher = ActionItemMatcher(neo4j_client=neo4j, openai_client=openai)

            # Create embedding for a test query
            embedding = await openai.create_embedding(
                "Send the proposal document by Friday"
            )

            # Search with a random tenant ID to ensure no matches
            random_tenant = str(uuid.uuid4())
            candidates = await matcher._find_candidates(
                embedding=embedding,
                tenant_id=random_tenant,
                account_id=None,
                limit=5,
            )

            assert candidates == []
            print("\nNo candidates found (as expected for empty/isolated tenant)")

        finally:
            await openai.close()
            await neo4j.close()


class TestDeduplicationDecisions:
    """Test LLM-based deduplication decisions."""

    @pytest.mark.asyncio
    async def test_dedupe_same_item_different_wording(
        self, openai_api_key: str, neo4j_credentials: dict
    ):
        """Test that similar items with different wording are matched."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            matcher = ActionItemMatcher(neo4j_client=neo4j, openai_client=openai)

            # Existing item
            existing = {
                'action_item_text': "I'll send the proposal deck by Friday",
                'owner': 'Sarah',
                'summary': 'Send proposal deck by Friday',
                'status': 'open',
                'created_at': '2024-01-15T10:00:00',
            }

            # New extraction - same task, different wording
            new_extraction = ExtractedActionItem(
                action_item_text="Sarah will send over the proposal document by end of week",
                owner='Sarah',
                summary='Send proposal document by end of week',
                conversation_context='Discussing deliverables for the deal',
                is_status_update=False,
                topic=make_test_topic("Sales Proposal", "Discussing deliverables"),
            )

            decision = await matcher._deduplicate(
                existing=existing,
                new_extraction=new_extraction,
                similarity_score=0.85,
            )

            print(f"\nDecision: is_same_item={decision.is_same_item}")
            print(f"Recommendation: {decision.merge_recommendation}")
            print(f"Reasoning: {decision.reasoning}")
            print(f"Confidence: {decision.confidence}")

            # Should recognize these as the same task
            assert decision.is_same_item is True
            assert decision.merge_recommendation in ('merge', 'update_status')

        finally:
            await openai.close()
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_dedupe_status_update(
        self, openai_api_key: str, neo4j_credentials: dict
    ):
        """Test that status updates are correctly identified."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            matcher = ActionItemMatcher(neo4j_client=neo4j, openai_client=openai)

            # Existing open item
            existing = {
                'action_item_text': "Send the pricing breakdown to the client",
                'owner': 'John',
                'summary': 'Send pricing breakdown to client',
                'status': 'open',
                'created_at': '2024-01-10T09:00:00',
            }

            # Status update - task completed
            new_extraction = ExtractedActionItem(
                action_item_text="I sent the pricing breakdown to them yesterday",
                owner='John',
                summary='Sent pricing breakdown',
                conversation_context='John confirming he completed the task',
                is_status_update=True,
                implied_status='completed',
                topic=make_test_topic("Client Pricing", "Pricing breakdown task"),
            )

            decision = await matcher._deduplicate(
                existing=existing,
                new_extraction=new_extraction,
                similarity_score=0.78,
            )

            print(f"\nDecision: is_same_item={decision.is_same_item}")
            print(f"Is status update: {decision.is_status_update}")
            print(f"Recommendation: {decision.merge_recommendation}")
            print(f"Reasoning: {decision.reasoning}")

            # Should recognize as status update
            assert decision.is_same_item is True
            assert decision.is_status_update is True
            assert decision.merge_recommendation == 'update_status'

        finally:
            await openai.close()
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_dedupe_different_items(
        self, openai_api_key: str, neo4j_credentials: dict
    ):
        """Test that different items are not matched."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            matcher = ActionItemMatcher(neo4j_client=neo4j, openai_client=openai)

            # Existing item
            existing = {
                'action_item_text': "Schedule a demo with the technical team",
                'owner': 'John',
                'summary': 'Schedule technical demo',
                'status': 'open',
                'created_at': '2024-01-15T10:00:00',
            }

            # Different task - related but distinct
            new_extraction = ExtractedActionItem(
                action_item_text="Send the technical documentation to Sarah",
                owner='John',
                summary='Send technical docs to Sarah',
                conversation_context='Preparing for the technical review',
                is_status_update=False,
                topic=make_test_topic("Technical Review", "Preparing for review"),
            )

            decision = await matcher._deduplicate(
                existing=existing,
                new_extraction=new_extraction,
                similarity_score=0.68,  # Similar but not the same
            )

            print(f"\nDecision: is_same_item={decision.is_same_item}")
            print(f"Recommendation: {decision.merge_recommendation}")
            print(f"Reasoning: {decision.reasoning}")

            # Should recognize as different tasks
            assert decision.is_same_item is False
            assert decision.merge_recommendation in ('create_new', 'link_related')

        finally:
            await openai.close()
            await neo4j.close()

    @pytest.mark.asyncio
    async def test_dedupe_different_owners(
        self, openai_api_key: str, neo4j_credentials: dict
    ):
        """Test that items with different owners are not matched."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()
            matcher = ActionItemMatcher(neo4j_client=neo4j, openai_client=openai)

            # Existing item - Sarah's task
            existing = {
                'action_item_text': "Send the proposal to the client",
                'owner': 'Sarah',
                'summary': 'Send proposal to client',
                'status': 'open',
                'created_at': '2024-01-15T10:00:00',
            }

            # Same-sounding task but different owner
            new_extraction = ExtractedActionItem(
                action_item_text="John will send the proposal to the client",
                owner='John',
                summary='Send proposal to client',
                conversation_context='Discussing who handles the deliverable',
                is_status_update=False,
                topic=make_test_topic("Client Proposal", "Discussing deliverables"),
            )

            decision = await matcher._deduplicate(
                existing=existing,
                new_extraction=new_extraction,
                similarity_score=0.92,  # Very similar text
            )

            print(f"\nDecision: is_same_item={decision.is_same_item}")
            print(f"Recommendation: {decision.merge_recommendation}")
            print(f"Reasoning: {decision.reasoning}")

            # Should recognize as different (different owners)
            assert decision.is_same_item is False

        finally:
            await openai.close()
            await neo4j.close()


class TestBestMatchSelection:
    """Test best match selection logic."""

    def test_select_best_match_no_matches(self, openai_api_key: str, neo4j_credentials: dict):
        """Test selection with no valid matches."""
        # Create mock decisions where nothing matches

        # Manually create a matcher just to test the selection logic
        # (no need for actual clients for this unit test)
        class MockClient:
            pass

        matcher = ActionItemMatcher.__new__(ActionItemMatcher)
        matcher.neo4j_client = MockClient()
        matcher.openai_client = MockClient()
        matcher.min_similarity = 0.65

        decisions = [
            (
                MatchCandidate('id1', {}, 0.8, 'combined'),
                DeduplicationDecision(
                    is_same_item=False,
                    is_status_update=False,
                    merge_recommendation='create_new',
                    reasoning='Different tasks',
                    confidence=0.9,
                ),
            ),
            (
                MatchCandidate('id2', {}, 0.7, 'combined'),
                DeduplicationDecision(
                    is_same_item=False,
                    is_status_update=False,
                    merge_recommendation='link_related',
                    reasoning='Related but distinct',
                    confidence=0.85,
                ),
            ),
        ]

        result = matcher._select_best_match(decisions)
        assert result is None
        print("\nCorrectly returned None when no matches found")

    def test_select_best_match_with_matches(self, openai_api_key: str, neo4j_credentials: dict):
        """Test selection picks highest confidence match."""

        class MockClient:
            pass

        matcher = ActionItemMatcher.__new__(ActionItemMatcher)
        matcher.neo4j_client = MockClient()
        matcher.openai_client = MockClient()
        matcher.min_similarity = 0.65

        decisions = [
            (
                MatchCandidate('id1', {'summary': 'Item 1'}, 0.85, 'combined'),
                DeduplicationDecision(
                    is_same_item=True,
                    is_status_update=False,
                    merge_recommendation='merge',
                    reasoning='Same task',
                    confidence=0.8,  # Lower confidence
                ),
            ),
            (
                MatchCandidate('id2', {'summary': 'Item 2'}, 0.75, 'combined'),
                DeduplicationDecision(
                    is_same_item=True,
                    is_status_update=True,
                    merge_recommendation='update_status',
                    reasoning='Status update',
                    confidence=0.95,  # Higher confidence - should win
                ),
            ),
        ]

        result = matcher._select_best_match(decisions)
        assert result is not None
        candidate, decision = result
        assert candidate.action_item_id == 'id2'
        assert decision.confidence == 0.95
        print(f"\nSelected best match: {candidate.action_item_id} with confidence {decision.confidence}")


class TestEndToEndMatching:
    """End-to-end matching tests with extraction + matching pipeline."""

    @pytest.mark.asyncio
    async def test_extract_and_match_new_items(
        self, openai_api_key: str, neo4j_credentials: dict, sample_tenant_id: str, sample_account_id: str
    ):
        """Test full pipeline: extract items, then try to match them."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)

        try:
            await neo4j.connect()

            # First, extract some items
            extractor = ActionItemExtractor(openai_client=openai)
            matcher = ActionItemMatcher(neo4j_client=neo4j, openai_client=openai)

            transcript = """
            Sarah: I'll finalize the budget report by Wednesday.
            John: Great. And I'll set up the stakeholder meeting for next Monday.
            """

            extraction_result = await extractor.extract_from_text(
                text=transcript,
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id=sample_account_id,
            )

            print(f"\nExtracted {extraction_result.count} items")

            # Now try to match them (should be unmatched since DB is empty for this tenant)
            if extraction_result.count > 0:
                batch_result = await match_batch(
                    matcher=matcher,
                    action_items=extraction_result.action_items,
                    raw_extractions=extraction_result.raw_extractions,
                    tenant_id=uuid.UUID(sample_tenant_id),
                    account_id=sample_account_id,
                )

                print(f"Matched: {batch_result.total_matched}")
                print(f"Unmatched: {batch_result.total_unmatched}")

                # All should be unmatched (no existing items in DB)
                assert batch_result.total_unmatched == extraction_result.count
                assert batch_result.total_matched == 0

        finally:
            await openai.close()
            await neo4j.close()
