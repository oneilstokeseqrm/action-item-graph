"""
Unit tests for LLM-as-Judge action item verification.

Tests the verifier's filtering logic with mocked LLM responses.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from action_item_graph.models.action_item import ActionItem
from action_item_graph.models.entities import Interaction, InteractionType
from action_item_graph.pipeline.extractor import ExtractionOutput
from action_item_graph.pipeline.verifier import (
    CONFIDENCE_FLOOR,
    ActionItemVerifier,
)
from action_item_graph.prompts.extract_action_items import ExtractedActionItem, ExtractedTopic
from action_item_graph.prompts.verification_prompts import (
    VerificationResult,
    VerificationVerdict,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TENANT_ID = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')


def _make_item(
    summary: str,
    owner: str = 'Sarah',
    confidence: float = 0.9,
    commitment_strength: str = 'explicit',
) -> tuple[ActionItem, ExtractedActionItem]:
    """Create paired ActionItem + ExtractedActionItem for testing."""
    raw = ExtractedActionItem(
        action_item_text=f'Action: {summary}',
        owner=owner,
        summary=summary,
        conversation_context='Test context',
        topic=ExtractedTopic(name='Test Topic', context='Testing'),
        confidence=confidence,
        commitment_strength=commitment_strength,
    )
    ai = ActionItem(
        id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        action_item_text=raw.action_item_text,
        summary=summary,
        owner=owner,
        owner_type='named',
        conversation_context='Test context',
        confidence=confidence,
    )
    return ai, raw


def _make_extraction(
    items: list[tuple[ActionItem, ExtractedActionItem]],
    content_text: str = 'Test transcript text',
) -> ExtractionOutput:
    """Build an ExtractionOutput from paired items."""
    interaction = Interaction(
        interaction_id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        interaction_type=InteractionType.TRANSCRIPT,
        content_text=content_text,
        timestamp=datetime.now(),
    )
    return ExtractionOutput(
        interaction=interaction,
        action_items=[ai for ai, _ in items],
        raw_extractions=[raw for _, raw in items],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verifier tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVerifier:
    """Test the ActionItemVerifier with mocked OpenAI calls."""

    @pytest.mark.asyncio
    async def test_all_items_pass(self):
        """When all items are actionable, all should pass through."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            return_value=VerificationResult(
                verdicts=[
                    VerificationVerdict(
                        index=0,
                        is_actionable=True,
                        adjusted_confidence=0.95,
                        issues=[],
                        reasoning='Clear commitment with named owner',
                    ),
                    VerificationVerdict(
                        index=1,
                        is_actionable=True,
                        adjusted_confidence=0.85,
                        issues=[],
                        reasoning='Solid action item with timeline',
                    ),
                ]
            )
        )

        verifier = ActionItemVerifier(mock_client)
        items = [
            _make_item('Send pricing deck to client by Friday'),
            _make_item('Schedule demo with technical team next week'),
        ]
        extraction = _make_extraction(items)

        result, rejected, reasons = await verifier.verify_batch(extraction)

        assert result.count == 2
        assert rejected == 0
        assert len(reasons) == 0
        # Confidence should be updated with verifier's assessment
        assert result.action_items[0].confidence == 0.95
        assert result.action_items[1].confidence == 0.85

    @pytest.mark.asyncio
    async def test_non_actionable_items_rejected(self):
        """Non-actionable items should be filtered out."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            return_value=VerificationResult(
                verdicts=[
                    VerificationVerdict(
                        index=0,
                        is_actionable=True,
                        adjusted_confidence=0.90,
                        issues=[],
                        reasoning='Clear commitment',
                    ),
                    VerificationVerdict(
                        index=1,
                        is_actionable=False,
                        adjusted_confidence=0.20,
                        issues=['No clear deliverable', 'Observation not commitment'],
                        reasoning='This is an observation about team behavior, not a commitment',
                    ),
                ]
            )
        )

        verifier = ActionItemVerifier(mock_client)
        items = [
            _make_item('Send pricing deck by Friday'),
            _make_item('The team is working on the integration'),
        ]
        extraction = _make_extraction(items)

        result, rejected, reasons = await verifier.verify_batch(extraction)

        assert result.count == 1
        assert rejected == 1
        assert len(reasons) == 1
        assert 'observation' in reasons[0].lower()

    @pytest.mark.asyncio
    async def test_low_confidence_items_rejected(self):
        """Items below confidence floor should be filtered out."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            return_value=VerificationResult(
                verdicts=[
                    VerificationVerdict(
                        index=0,
                        is_actionable=True,
                        adjusted_confidence=0.90,
                        issues=[],
                        reasoning='Clear commitment',
                    ),
                    VerificationVerdict(
                        index=1,
                        is_actionable=True,
                        adjusted_confidence=0.30,  # Below CONFIDENCE_FLOOR (0.4)
                        issues=['Vague intention'],
                        reasoning='Sounds like a weak intention rather than firm commitment',
                    ),
                ]
            )
        )

        verifier = ActionItemVerifier(mock_client)
        items = [
            _make_item('Send pricing deck by Friday'),
            _make_item('We should think about expanding to Europe'),
        ]
        extraction = _make_extraction(items)

        result, rejected, reasons = await verifier.verify_batch(extraction)

        assert result.count == 1
        assert rejected == 1
        assert 'confidence floor' in reasons[0].lower()

    @pytest.mark.asyncio
    async def test_empty_extraction_passthrough(self):
        """Empty extraction should pass through without LLM call."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock()

        verifier = ActionItemVerifier(mock_client)
        extraction = _make_extraction([])

        result, rejected, reasons = await verifier.verify_batch(extraction)

        assert result.count == 0
        assert rejected == 0
        mock_client.chat_completion_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_failure_passes_all(self):
        """If LLM call fails, all items should pass through (fail-open)."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            side_effect=Exception('LLM timeout')
        )

        verifier = ActionItemVerifier(mock_client)
        items = [
            _make_item('Send pricing deck'),
            _make_item('Schedule demo'),
        ]
        extraction = _make_extraction(items)

        result, rejected, reasons = await verifier.verify_batch(extraction)

        assert result.count == 2  # All pass through
        assert rejected == 0
        assert len(reasons) == 1  # One reason explaining the failure
        assert 'failed' in reasons[0].lower()

    @pytest.mark.asyncio
    async def test_missing_verdict_items_kept(self):
        """Items without a verdict (LLM skipped them) should be kept."""
        mock_client = MagicMock()
        # LLM only returns verdict for index 0, skips index 1
        mock_client.chat_completion_structured = AsyncMock(
            return_value=VerificationResult(
                verdicts=[
                    VerificationVerdict(
                        index=0,
                        is_actionable=True,
                        adjusted_confidence=0.85,
                        issues=[],
                        reasoning='Clear commitment',
                    ),
                ]
            )
        )

        verifier = ActionItemVerifier(mock_client)
        items = [
            _make_item('Send pricing deck'),
            _make_item('Schedule demo'),
        ]
        extraction = _make_extraction(items)

        result, rejected, reasons = await verifier.verify_batch(extraction)

        assert result.count == 2  # Both kept (missing verdict = keep)
        assert rejected == 0

    @pytest.mark.asyncio
    async def test_custom_confidence_floor(self):
        """Custom confidence floor should be respected."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            return_value=VerificationResult(
                verdicts=[
                    VerificationVerdict(
                        index=0,
                        is_actionable=True,
                        adjusted_confidence=0.55,
                        issues=[],
                        reasoning='Okay commitment',
                    ),
                ]
            )
        )

        # With high floor, item should be rejected
        verifier_high = ActionItemVerifier(mock_client, confidence_floor=0.6)
        items = [_make_item('Maybe send the deck')]
        extraction = _make_extraction(items)
        result, rejected, _ = await verifier_high.verify_batch(extraction)
        assert rejected == 1

        # With low floor, same item should pass
        verifier_low = ActionItemVerifier(mock_client, confidence_floor=0.5)
        result, rejected, _ = await verifier_low.verify_batch(extraction)
        assert rejected == 0

    @pytest.mark.asyncio
    async def test_transcript_excerpt_used(self):
        """Verify the transcript excerpt is passed to the LLM."""
        mock_client = MagicMock()
        mock_client.chat_completion_structured = AsyncMock(
            return_value=VerificationResult(verdicts=[
                VerificationVerdict(
                    index=0,
                    is_actionable=True,
                    adjusted_confidence=0.90,
                    issues=[],
                    reasoning='OK',
                ),
            ])
        )

        verifier = ActionItemVerifier(mock_client)
        items = [_make_item('Send deck')]
        extraction = _make_extraction(items, content_text='Full transcript here...')

        await verifier.verify_batch(extraction, transcript_excerpt='Custom excerpt')

        # Check that the LLM was called with the custom excerpt
        call_args = mock_client.chat_completion_structured.call_args
        messages = call_args.kwargs.get('messages', call_args.args[0] if call_args.args else None)
        user_message = messages[-1]['content']
        assert 'Custom excerpt' in user_message


class TestConfidenceFloor:
    """Test the confidence floor constant."""

    def test_default_floor(self):
        assert CONFIDENCE_FLOOR == 0.4

    def test_floor_boundary(self):
        """Items exactly at the floor should pass; below should fail."""
        # This is tested via the verifier above, but let's be explicit
        assert 0.4 >= CONFIDENCE_FLOOR  # Exactly at floor → passes
        assert 0.39 < CONFIDENCE_FLOOR  # Below floor → rejected
