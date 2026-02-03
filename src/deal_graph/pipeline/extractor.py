"""
Deal extraction engine for MEDDIC analysis.

Orchestrates LLM extraction from transcripts, handling both:
- Case A (Targeted): Update a specific deal when opportunity_id is provided
- Case B (Discovery): Find all deals when no opportunity_id present

Uses OpenAI structured output with DealExtractionResult as the response model.
"""

import structlog

from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.models.envelope import EnvelopeV1

from ..models.extraction import DealExtractionResult, ExtractedDeal
from ..prompts.extract_deals import build_discovery_prompt, build_targeted_prompt

logger = structlog.get_logger(__name__)


class DealExtractor:
    """
    Extracts deal opportunities from transcripts using MEDDIC analysis.

    Branches on the presence of opportunity_id in the envelope:
    - Present (+ existing deal context) → targeted extraction (Case A)
    - Absent → discovery extraction (Case B)

    Generates embeddings for extracted deals using the pattern:
        "{opportunity_name}: {opportunity_summary}"
    """

    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize the extractor.

        Args:
            openai_client: OpenAI client for LLM calls and embeddings
        """
        self.openai = openai_client

    async def extract_from_envelope(
        self,
        envelope: EnvelopeV1,
        existing_deal: dict | None = None,
    ) -> tuple[DealExtractionResult, list[list[float]]]:
        """
        Extract deals from an envelope, branching on opportunity_id.

        Case A (opportunity_id present + existing_deal provided):
            Targeted extraction scoped to one known deal.

        Case B (no opportunity_id, or no existing_deal):
            Discovery extraction that may find 0, 1, or N deals.

        Args:
            envelope: Input envelope with content_text in content.text
            existing_deal: For Case A — current Deal node properties

        Returns:
            Tuple of (extraction result, list of embedding vectors per deal)
        """
        content_text = envelope.content.text
        opportunity_id = envelope.opportunity_id

        log = logger.bind(
            tenant_id=str(envelope.tenant_id),
            interaction_id=str(envelope.interaction_id) if envelope.interaction_id else None,
            opportunity_id=opportunity_id,
        )

        if opportunity_id and existing_deal:
            log.info('deal_extraction.targeted', mode='case_a')
            result = await self._extract_targeted(
                content_text=content_text,
                existing_deal=existing_deal,
                account_name=existing_deal.get('account_name'),
                meeting_title=envelope.meeting_title,
            )
        else:
            log.info('deal_extraction.discovery', mode='case_b')
            result = await self._extract_discovery(
                content_text=content_text,
                meeting_title=envelope.meeting_title,
            )

        # Generate embeddings for extracted deals
        embeddings: list[list[float]] = []
        if result.has_deals and result.deals:
            embeddings = await self._generate_embeddings(result.deals)
            log.info(
                'deal_extraction.complete',
                deal_count=len(result.deals),
                embeddings_count=len(embeddings),
            )
        else:
            log.info('deal_extraction.no_deals_found')

        return result, embeddings

    async def _extract_discovery(
        self,
        content_text: str,
        account_name: str | None = None,
        meeting_title: str | None = None,
        participants: list[str] | None = None,
    ) -> DealExtractionResult:
        """
        Case B: Extract ALL deal opportunities from a transcript.

        Args:
            content_text: Transcript text (from Interaction.content_text)
            account_name: Optional account name
            meeting_title: Optional meeting title
            participants: Optional participant names

        Returns:
            DealExtractionResult with 0+ extracted deals
        """
        messages = build_discovery_prompt(
            content_text=content_text,
            account_name=account_name,
            meeting_title=meeting_title,
            participants=participants,
        )

        result = await self.openai.chat_completion_structured(
            messages=messages,
            response_model=DealExtractionResult,
        )

        logger.info(
            'deal_extraction.discovery_result',
            has_deals=result.has_deals,
            deal_count=len(result.deals),
            notes=result.extraction_notes,
        )

        return result

    async def _extract_targeted(
        self,
        content_text: str,
        existing_deal: dict,
        account_name: str | None = None,
        meeting_title: str | None = None,
        participants: list[str] | None = None,
    ) -> DealExtractionResult:
        """
        Case A: Extract updates for a SPECIFIC existing deal.

        Enforces the V1 constraint: at most 1 deal returned. If the LLM
        returns multiple, keeps the highest-confidence deal.

        Args:
            content_text: Transcript text (from Interaction.content_text)
            existing_deal: Current Deal node properties from the graph
            account_name: Optional account name
            meeting_title: Optional meeting title
            participants: Optional participant names

        Returns:
            DealExtractionResult with 0 or 1 extracted deals
        """
        messages = build_targeted_prompt(
            content_text=content_text,
            existing_deal=existing_deal,
            account_name=account_name,
            meeting_title=meeting_title,
            participants=participants,
        )

        result = await self.openai.chat_completion_structured(
            messages=messages,
            response_model=DealExtractionResult,
        )

        # Enforce single-deal constraint for targeted extraction
        if len(result.deals) > 1:
            best = max(result.deals, key=lambda d: d.confidence)
            logger.warning(
                'deal_extraction.targeted_multi_deal',
                deal_count=len(result.deals),
                keeping=best.opportunity_name,
            )
            result = DealExtractionResult(
                deals=[best],
                has_deals=True,
                extraction_notes=(
                    f'Targeted extraction returned {len(result.deals)} deals; '
                    'kept highest-confidence deal only.'
                ),
            )

        logger.info(
            'deal_extraction.targeted_result',
            has_deals=result.has_deals,
            deal_count=len(result.deals),
        )

        return result

    async def _generate_embeddings(
        self,
        deals: list[ExtractedDeal],
    ) -> list[list[float]]:
        """
        Generate embeddings for extracted deals.

        Uses the embedding text pattern: "{opportunity_name}: {opportunity_summary}"
        Matches the Deal pipeline's entity resolution search strategy.

        Args:
            deals: List of extracted deals to embed

        Returns:
            List of embedding vectors (same order as input deals)
        """
        if not deals:
            return []

        texts = [
            f'{deal.opportunity_name}: {deal.opportunity_summary}'
            for deal in deals
        ]

        embeddings = await self.openai.create_embeddings_batch(texts)

        logger.info(
            'deal_extraction.embeddings_generated',
            count=len(embeddings),
            dimensions=len(embeddings[0]) if embeddings else 0,
        )

        return embeddings
