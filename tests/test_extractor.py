"""
Live integration tests for action item extraction.

These tests hit the actual OpenAI API.
Run with: pytest tests/test_extractor.py -v
"""

import uuid

import pytest

from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.pipeline.extractor import ActionItemExtractor
from action_item_graph.models.action_item import ActionItemStatus


class TestActionItemExtraction:
    """Test action item extraction from transcripts."""

    @pytest.mark.asyncio
    async def test_extract_basic_action_items(
        self, openai_api_key: str, sample_transcript: str, sample_tenant_id: str
    ):
        """Test extracting action items from a sample transcript."""
        client = OpenAIClient(api_key=openai_api_key)
        extractor = ActionItemExtractor(openai_client=client)

        try:
            result = await extractor.extract_from_text(
                text=sample_transcript,
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id='acct_test_001',
                meeting_title='Q1 Proposal Review',
            )

            # Should have extracted some action items
            assert result.count > 0
            print(f"\nExtracted {result.count} action items:")

            for i, (ai, raw) in enumerate(
                zip(result.action_items, result.raw_extractions), 1
            ):
                print(f"\n  {i}. {ai.summary}")
                print(f"     Owner: {ai.owner}")
                print(f"     Status Update: {raw.is_status_update}")
                print(f"     Confidence: {ai.confidence:.2f}")
                print(f"     Embedding dims: {len(ai.embedding)}")

            # Verify all action items have required fields
            for ai in result.action_items:
                assert ai.tenant_id == uuid.UUID(sample_tenant_id)
                assert ai.account_id == 'acct_test_001'
                assert ai.action_item_text
                assert ai.summary
                assert ai.owner
                assert ai.embedding is not None
                assert len(ai.embedding) == 1536
                assert ai.embedding_current is not None

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_extract_with_status_updates(self, openai_api_key: str, sample_tenant_id: str):
        """Test detecting status updates in transcript."""
        client = OpenAIClient(api_key=openai_api_key)
        extractor = ActionItemExtractor(openai_client=client)

        transcript_with_updates = """
Sarah: Hey John, just wanted to give you a quick update.
John: Sure, what's up?
Sarah: I finished that proposal deck we discussed. Sent it over to Acme Corp this morning.
John: Great! What about the legal review?
Sarah: Still working on that. I have a meeting with legal tomorrow.
John: Perfect. Also, can you schedule a demo with their technical team for next week?
Sarah: Will do. I'll send the invite today.
"""

        try:
            result = await extractor.extract_from_text(
                text=transcript_with_updates,
                tenant_id=uuid.UUID(sample_tenant_id),
            )

            print(f"\nExtracted {result.count} items:")
            print(f"  New items: {len(result.new_items)}")
            print(f"  Status updates: {len(result.status_updates)}")

            # Should detect both new items and status updates
            for ai, raw in zip(result.action_items, result.raw_extractions):
                status_type = 'STATUS UPDATE' if raw.is_status_update else 'NEW'
                print(f"\n  [{status_type}] {ai.summary}")
                print(f"     Owner: {ai.owner}")
                if raw.is_status_update:
                    print(f"     Implied status: {raw.implied_status}")

            # We expect at least one status update (the sent proposal)
            assert len(result.status_updates) >= 1

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_extract_no_action_items(self, openai_api_key: str, sample_tenant_id: str):
        """Test handling transcript with no action items."""
        client = OpenAIClient(api_key=openai_api_key)
        extractor = ActionItemExtractor(openai_client=client)

        casual_transcript = """
John: Hey, how was your weekend?
Sarah: Pretty good! Went hiking with the family.
John: Nice! Where did you go?
Sarah: Up to the mountains. The weather was perfect.
John: Sounds great. We should plan a team outing sometime.
Sarah: Yeah, that would be fun.
"""

        try:
            result = await extractor.extract_from_text(
                text=casual_transcript,
                tenant_id=uuid.UUID(sample_tenant_id),
            )

            print(f"\nExtracted {result.count} action items from casual conversation")
            if result.extraction_notes:
                print(f"Notes: {result.extraction_notes}")

            # Should have few or no action items
            # The vague "we should plan" might or might not be extracted
            assert result.count <= 1

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_extract_complex_transcript(self, openai_api_key: str, sample_tenant_id: str):
        """Test extraction from a more complex sales call transcript."""
        client = OpenAIClient(api_key=openai_api_key)
        extractor = ActionItemExtractor(openai_client=client)

        complex_transcript = """
John: Thanks for joining the call today, Sarah. I know you've been evaluating our enterprise solution.
Sarah: Yes, we're very interested. A few things we need to clarify before moving forward.
John: Absolutely, fire away.
Sarah: First, can you send us the detailed pricing breakdown? We need it for the budget approval.
John: I'll have that to you by end of day tomorrow. I'll include the volume discounts we discussed.
Sarah: Perfect. Also, our security team needs to review your SOC 2 report.
John: I'll send that along with the pricing. Actually, let me also include our security whitepaper.
Sarah: That would be helpful. Now, regarding implementation - we'd need it done by Q2.
John: That's doable. I'll work with our implementation team to draft a timeline. Can we schedule a follow-up call next week to review it?
Sarah: Yes, let's do Thursday at 2pm. I'll have our CTO join as well.
John: Sounds good. I'll send the calendar invite. One more thing - should I loop in our solutions architect for the technical discussion?
Sarah: Please do. We have some specific integration questions.
John: Will do. I'll reach out to Mike and have him prepare some integration examples.
Sarah: Great. Oh, and I almost forgot - I need to get internal sign-off from our VP. Can you prepare a one-pager executive summary?
John: Of course. I'll have that ready before our Thursday call.
"""

        try:
            result = await extractor.extract_from_text(
                text=complex_transcript,
                tenant_id=uuid.UUID(sample_tenant_id),
                account_id='acct_enterprise_deal',
                meeting_title='Enterprise Solution Evaluation',
            )

            print(f"\nExtracted {result.count} action items from complex sales call:")

            for i, (ai, raw) in enumerate(
                zip(result.action_items, result.raw_extractions), 1
            ):
                print(f"\n  {i}. {ai.summary}")
                print(f"     Owner: {ai.owner}")
                print(f"     Due: {raw.due_date_text or 'Not specified'}")
                print(f"     Confidence: {ai.confidence:.2f}")

            # Should extract multiple action items (pricing, SOC2, timeline, calendar, etc.)
            assert result.count >= 5

        finally:
            await client.close()


class TestExtractionOutput:
    """Test ExtractionOutput helper methods."""

    @pytest.mark.asyncio
    async def test_new_items_vs_status_updates(
        self, openai_api_key: str, sample_tenant_id: str
    ):
        """Test separating new items from status updates."""
        client = OpenAIClient(api_key=openai_api_key)
        extractor = ActionItemExtractor(openai_client=client)

        mixed_transcript = """
Sarah: I completed the market analysis report that was assigned last week.
John: Excellent! Now we need to schedule a presentation to the board.
Sarah: I'll set that up for next Monday.
John: Also, the client proposal I mentioned is ready - sent it yesterday.
Sarah: Great. I'll follow up with them on Friday to get their feedback.
"""

        try:
            result = await extractor.extract_from_text(
                text=mixed_transcript,
                tenant_id=uuid.UUID(sample_tenant_id),
            )

            new_items = result.new_items
            status_updates = result.status_updates

            print(f"\nNew items ({len(new_items)}):")
            for ai in new_items:
                print(f"  - {ai.summary}")

            print(f"\nStatus updates ({len(status_updates)}):")
            for ai, implied_status in status_updates:
                print(f"  - {ai.summary} (implied: {implied_status})")

            # Should have both new items and status updates
            assert len(new_items) >= 1
            # Note: status update detection may vary

        finally:
            await client.close()
