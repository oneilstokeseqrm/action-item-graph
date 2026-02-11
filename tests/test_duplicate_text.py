"""
Regression test for the duplicate-text dict-collision bug.

Previously, _execute_merges used a dict keyed by action_item_text to
correlate MatchResult → ActionItem.  When two extracted items had
identical text (e.g., repeated phrases in a transcript), the second
silently overwrote the first, causing the same UUID to be used for
both merge operations.

The fix replaces the dict lookup with 1:1 positional alignment (zip)
between match_results and filtered_action_items, which is structurally
incapable of collision.

Run with: pytest tests/test_duplicate_text.py -v
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from action_item_graph.models.action_item import ActionItem, ActionItemStatus
from action_item_graph.models.entities import Interaction, InteractionType
from action_item_graph.pipeline.matcher import MatchResult
from action_item_graph.pipeline.merger import MergeResult
from action_item_graph.pipeline.pipeline import ActionItemPipeline
from action_item_graph.prompts.extract_action_items import ExtractedActionItem, ExtractedTopic


def _make_action_item(
    *,
    item_id: uuid.UUID | None = None,
    tenant_id: uuid.UUID,
    text: str,
    owner: str = 'Alice',
) -> ActionItem:
    """Helper to build a minimal ActionItem for testing."""
    return ActionItem(
        id=item_id or uuid.uuid4(),
        tenant_id=tenant_id,
        account_id='acct_test',
        action_item_text=text,
        summary=text[:40],
        owner=owner,
        status=ActionItemStatus.OPEN,
        embedding=[0.1] * 1536,
        embedding_current=[0.1] * 1536,
        source_interaction_id=uuid.uuid4(),
    )


def _make_match_result(text: str, owner: str = 'Alice') -> MatchResult:
    """Helper to build a MatchResult with no candidates (new item)."""
    return MatchResult(
        extracted_item=ExtractedActionItem(
            action_item_text=text,
            summary=text[:40],
            owner=owner,
            conversation_context='test context',
            topic=ExtractedTopic(name='Test Topic', context='test'),
        ),
        embedding=[0.1] * 1536,
        candidates=[],
        decisions=[],
        best_match=None,
    )


class TestDuplicateTextMerges:
    """
    Verify that _execute_merges handles duplicate action_item_text correctly.

    The old code built {ai.action_item_text: ai} which silently dropped
    duplicates.  The fix uses 1:1 zip alignment.
    """

    @pytest.mark.asyncio
    async def test_duplicate_text_produces_distinct_merges(self):
        """Two items with identical text must get separate merge operations."""
        tenant_id = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')
        same_text = "Schedule follow-up meeting with the team next week"

        id_a = uuid.uuid4()
        id_b = uuid.uuid4()

        ai_a = _make_action_item(item_id=id_a, tenant_id=tenant_id, text=same_text, owner='Alice')
        ai_b = _make_action_item(item_id=id_b, tenant_id=tenant_id, text=same_text, owner='Bob')

        interaction = Interaction(
            interaction_id=uuid.uuid4(),
            tenant_id=tenant_id,
            account_id='acct_test',
            interaction_type=InteractionType.TRANSCRIPT,
            content_text='test',
            timestamp=datetime.now(tz=timezone.utc),
        )

        match_a = _make_match_result(same_text, owner='Alice')
        match_b = _make_match_result(same_text, owner='Bob')

        # Build pipeline with mocked merger
        pipeline = ActionItemPipeline.__new__(ActionItemPipeline)
        pipeline.merger = AsyncMock()
        pipeline.merger.execute_decision.side_effect = [
            MergeResult(
                action='created',
                action_item_id=str(id_a),
                was_new=True,
                version_created=False,
                linked_interaction_id=str(interaction.interaction_id),
                details={},
            ),
            MergeResult(
                action='created',
                action_item_id=str(id_b),
                was_new=True,
                version_created=False,
                linked_interaction_id=str(interaction.interaction_id),
                details={},
            ),
        ]

        # Execute
        results = await pipeline._execute_merges(
            match_results=[match_a, match_b],
            action_items=[ai_a, ai_b],
            interaction=interaction,
        )

        # Both items must produce merge results
        assert len(results) == 2
        assert results[0].action_item_id == str(id_a)
        assert results[1].action_item_id == str(id_b)

        # Merger must have been called with the correct (distinct) ActionItem each time
        calls = pipeline.merger.execute_decision.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs['action_item'].id == id_a
        assert calls[1].kwargs['action_item'].id == id_b

    @pytest.mark.asyncio
    async def test_single_item_still_works(self):
        """Sanity check: a single item still goes through correctly."""
        tenant_id = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')
        item_id = uuid.uuid4()

        ai = _make_action_item(item_id=item_id, tenant_id=tenant_id, text='Send the report')

        interaction = Interaction(
            interaction_id=uuid.uuid4(),
            tenant_id=tenant_id,
            account_id='acct_test',
            interaction_type=InteractionType.TRANSCRIPT,
            content_text='test',
            timestamp=datetime.now(tz=timezone.utc),
        )

        match = _make_match_result('Send the report')

        pipeline = ActionItemPipeline.__new__(ActionItemPipeline)
        pipeline.merger = AsyncMock()
        pipeline.merger.execute_decision.return_value = MergeResult(
            action='created',
            action_item_id=str(item_id),
            was_new=True,
            version_created=False,
            linked_interaction_id=str(interaction.interaction_id),
            details={},
        )

        results = await pipeline._execute_merges(
            match_results=[match],
            action_items=[ai],
            interaction=interaction,
        )

        assert len(results) == 1
        assert results[0].action_item_id == str(item_id)

    @pytest.mark.asyncio
    async def test_three_duplicates_all_preserved(self):
        """Three items with identical text must all get processed."""
        tenant_id = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')
        same_text = "Review the contract terms before Thursday"

        ids = [uuid.uuid4() for _ in range(3)]
        action_items = [
            _make_action_item(item_id=uid, tenant_id=tenant_id, text=same_text)
            for uid in ids
        ]
        match_results = [_make_match_result(same_text) for _ in range(3)]

        interaction = Interaction(
            interaction_id=uuid.uuid4(),
            tenant_id=tenant_id,
            account_id='acct_test',
            interaction_type=InteractionType.TRANSCRIPT,
            content_text='test',
            timestamp=datetime.now(tz=timezone.utc),
        )

        pipeline = ActionItemPipeline.__new__(ActionItemPipeline)
        pipeline.merger = AsyncMock()
        pipeline.merger.execute_decision.side_effect = [
            MergeResult(
                action='created',
                action_item_id=str(uid),
                was_new=True,
                version_created=False,
                linked_interaction_id=str(interaction.interaction_id),
                details={},
            )
            for uid in ids
        ]

        results = await pipeline._execute_merges(
            match_results=match_results,
            action_items=action_items,
            interaction=interaction,
        )

        assert len(results) == 3
        returned_ids = {r.action_item_id for r in results}
        expected_ids = {str(uid) for uid in ids}
        assert returned_ids == expected_ids, "All three distinct UUIDs must appear in results"
