"""
LLM-as-Judge action item verifier.

Adversarial quality check that runs after extraction (and consolidation)
to filter out items that don't meet the commitment quality bar. Uses a
deliberately different perspective from the extraction prompt to catch
false positives.
"""

from __future__ import annotations

from ..clients.openai_client import OpenAIClient
from ..logging import get_logger
from ..models.action_item import ActionItem
from ..prompts.extract_action_items import ExtractedActionItem
from ..prompts.verification_prompts import (
    VERIFICATION_SYSTEM_PROMPT,
    VERIFICATION_USER_PROMPT_TEMPLATE,
    VerificationResult,
)
from .extractor import ExtractionOutput

logger = get_logger(__name__)

# Items with adjusted confidence below this floor are dropped.
CONFIDENCE_FLOOR = 0.4


class ActionItemVerifier:
    """
    Adversarial verifier using the LLM-as-Judge pattern.

    Sends all extracted items in a single batch call (not N individual calls)
    to minimize latency. The verifier deliberately uses different evaluation
    criteria from the extraction prompt to catch items that slipped through.
    """

    def __init__(self, openai_client: OpenAIClient, confidence_floor: float = CONFIDENCE_FLOOR):
        self.openai_client = openai_client
        self.confidence_floor = confidence_floor

    async def verify_batch(
        self,
        extraction: ExtractionOutput,
        transcript_excerpt: str | None = None,
    ) -> tuple[ExtractionOutput, int, list[str]]:
        """
        Verify a batch of extracted action items.

        Args:
            extraction: The ExtractionOutput (possibly post-consolidation)
            transcript_excerpt: Optional transcript excerpt for context
                (first ~2000 chars recommended for token efficiency)

        Returns:
            Tuple of:
                - Filtered ExtractionOutput with only verified items
                - Number of items rejected
                - List of rejection reasons (for logging/debugging)
        """
        if extraction.count == 0:
            return extraction, 0, []

        # Build items text for the verifier
        items_text_parts = []
        for i, (ai, raw) in enumerate(
            zip(extraction.action_items, extraction.raw_extractions)
        ):
            items_text_parts.append(
                f'[{i}] Summary: {ai.summary}\n'
                f'    Owner: {ai.owner} (type: {ai.owner_type})\n'
                f'    Text: {ai.action_item_text}\n'
                f'    Context: {ai.conversation_context}\n'
                f'    Commitment: {raw.commitment_strength}\n'
                f'    Confidence: {ai.confidence:.2f}\n'
                f'    Status update: {raw.is_status_update}'
            )

        items_text = '\n\n'.join(items_text_parts)

        # Use a truncated excerpt for context (avoid token bloat)
        excerpt = ''
        if transcript_excerpt:
            excerpt = transcript_excerpt[:2000]
        elif extraction.interaction.content_text:
            excerpt = extraction.interaction.content_text[:2000]

        messages = [
            {'role': 'system', 'content': VERIFICATION_SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': VERIFICATION_USER_PROMPT_TEMPLATE.format(
                    transcript_excerpt=excerpt,
                    items_text=items_text,
                ),
            },
        ]

        try:
            result = await self.openai_client.chat_completion_structured(
                messages=messages,
                response_model=VerificationResult,
            )
        except Exception:
            logger.exception('verification_llm_call_failed')
            # On failure, pass everything through (don't block the pipeline)
            return extraction, 0, ['Verification LLM call failed — all items passed through']

        # Apply verdicts
        verified_action_items: list[ActionItem] = []
        verified_raw: list[ExtractedActionItem] = []
        rejection_reasons: list[str] = []

        # Build verdict lookup by index
        verdict_map = {v.index: v for v in result.verdicts}

        for i, (ai, raw) in enumerate(
            zip(extraction.action_items, extraction.raw_extractions)
        ):
            verdict = verdict_map.get(i)

            if verdict is None:
                # No verdict for this item — keep it (defensive)
                verified_action_items.append(ai)
                verified_raw.append(raw)
                continue

            if not verdict.is_actionable:
                reason = f'Item {i} rejected: {verdict.reasoning}'
                if verdict.issues:
                    reason += f' Issues: {", ".join(verdict.issues)}'
                rejection_reasons.append(reason)
                logger.debug(
                    'verification_item_rejected',
                    index=i,
                    summary=ai.summary,
                    reasoning=verdict.reasoning,
                    issues=verdict.issues,
                )
                continue

            if verdict.adjusted_confidence < self.confidence_floor:
                reason = (
                    f'Item {i} below confidence floor '
                    f'({verdict.adjusted_confidence:.2f} < {self.confidence_floor}): '
                    f'{verdict.reasoning}'
                )
                rejection_reasons.append(reason)
                logger.debug(
                    'verification_item_low_confidence',
                    index=i,
                    summary=ai.summary,
                    adjusted_confidence=verdict.adjusted_confidence,
                )
                continue

            # Item passed — update confidence with the verifier's assessment
            ai.confidence = verdict.adjusted_confidence
            verified_action_items.append(ai)
            verified_raw.append(raw)

        items_rejected = extraction.count - len(verified_action_items)

        logger.info(
            'verification_complete',
            original_count=extraction.count,
            verified_count=len(verified_action_items),
            rejected_count=items_rejected,
        )

        verified_extraction = ExtractionOutput(
            interaction=extraction.interaction,
            action_items=verified_action_items,
            raw_extractions=verified_raw,
            extraction_notes=extraction.extraction_notes,
        )

        return verified_extraction, items_rejected, rejection_reasons
