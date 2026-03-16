"""
Action item extraction service.

Extracts action items from transcripts using OpenAI structured output,
then generates embeddings for matching.
"""

from datetime import datetime
from uuid import UUID, uuid4

from ..clients.openai_client import OpenAIClient
from ..models.action_item import ActionItem, ActionItemStatus
from ..models.envelope import EnvelopeV1
from ..models.entities import Interaction, InteractionType
from ..prompts.extract_action_items import (
    ExtractionResult,
    ExtractedActionItem,
    build_extraction_prompt,
)


def compute_priority_score(
    impact: int,
    urgency: int,
    specificity: int,
    confidence: float,
) -> float:
    """
    Compute a weighted priority score (0.0-1.0).

    Weights: impact 0.40, urgency 0.35, specificity 0.15, confidence 0.10.
    These weights reflect that business impact and time sensitivity are the
    strongest signals for prioritization, while specificity and confidence
    serve as quality tiebreakers.

    Args:
        impact: Business impact score (1-5)
        urgency: Time sensitivity score (1-5)
        specificity: Actionability score (1-5)
        confidence: Extraction confidence (0.0-1.0)

    Returns:
        Weighted priority score (0.0-1.0), rounded to 3 decimal places
    """
    return round(
        0.40 * (impact / 5)
        + 0.35 * (urgency / 5)
        + 0.15 * (specificity / 5)
        + 0.10 * confidence,
        3,
    )


class ExtractionOutput:
    """Output from the extraction process."""

    def __init__(
        self,
        interaction: Interaction,
        action_items: list[ActionItem],
        raw_extractions: list[ExtractedActionItem],
        extraction_notes: str | None = None,
    ):
        self.interaction = interaction
        self.action_items = action_items
        self.raw_extractions = raw_extractions
        self.extraction_notes = extraction_notes

    @property
    def count(self) -> int:
        """Number of action items extracted."""
        return len(self.action_items)

    @property
    def new_items(self) -> list[ActionItem]:
        """Action items that are new (not status updates)."""
        return [
            ai
            for ai, raw in zip(self.action_items, self.raw_extractions)
            if not raw.is_status_update
        ]

    @property
    def status_updates(self) -> list[tuple[ActionItem, str | None]]:
        """Action items that are status updates, with their implied status."""
        return [
            (ai, raw.implied_status)
            for ai, raw in zip(self.action_items, self.raw_extractions)
            if raw.is_status_update
        ]


class ActionItemExtractor:
    """
    Extracts action items from transcripts.

    Uses OpenAI for extraction and embedding generation.
    """

    def __init__(self, openai_client: OpenAIClient):
        """
        Initialize the extractor.

        Args:
            openai_client: Configured OpenAI client
        """
        self.openai_client = openai_client

    async def extract_from_envelope(self, envelope: EnvelopeV1) -> ExtractionOutput:
        """
        Extract action items from an EnvelopeV1 payload.

        Args:
            envelope: The input envelope containing transcript and metadata

        Returns:
            ExtractionOutput with Interaction and ActionItem objects
        """
        # Create the Interaction record
        interaction = Interaction(
            interaction_id=envelope.interaction_id or uuid4(),
            tenant_id=envelope.tenant_id,
            account_id=envelope.account_id,
            interaction_type=InteractionType(envelope.interaction_type.value),
            title=envelope.meeting_title,
            content_text=envelope.content.text,
            timestamp=envelope.timestamp,
            duration_seconds=envelope.duration_seconds,
            source=envelope.source.value,
            user_id=envelope.user_id,
            pg_user_id=envelope.pg_user_id,
            extras=envelope.extras,
        )

        # Extract action items — use rich contact labels if available, fall back to contact_ids
        participants = (
            envelope.contact_labels if envelope.contacts
            else (envelope.contact_ids if envelope.contact_ids else None)
        )
        extraction_result = await self._extract_action_items(
            transcript_text=envelope.content.text,
            meeting_title=envelope.meeting_title,
            participants=participants,
            user_name=envelope.extras.get('user_name') if envelope.extras else None,
        )

        # Convert to ActionItem models with embeddings
        action_items = await self._create_action_items(
            extractions=extraction_result.action_items,
            tenant_id=envelope.tenant_id,
            account_id=envelope.account_id,
            interaction_id=interaction.interaction_id,
            user_id=envelope.user_id,
            pg_user_id=envelope.pg_user_id,
        )

        # Auto-resolve recording user's owner name to canonical form
        user_name = envelope.extras.get('user_name') if envelope.extras else None
        if user_name:
            for action_item, raw in zip(action_items, extraction_result.action_items):
                if raw.is_user_owned and action_item.owner != user_name:
                    action_item.owner = user_name
                    action_item.owner_type = 'named'

        # Update interaction with count
        interaction.action_item_count = len(action_items)
        interaction.processed_at = datetime.now(tz=None)

        return ExtractionOutput(
            interaction=interaction,
            action_items=action_items,
            raw_extractions=extraction_result.action_items,
            extraction_notes=extraction_result.extraction_notes,
        )

    async def extract_from_text(
        self,
        text: str,
        tenant_id: UUID,
        account_id: str | None = None,
        user_id: str | None = None,
        meeting_title: str | None = None,
        participants: list[str] | None = None,
        user_name: str | None = None,
    ) -> ExtractionOutput:
        """
        Extract action items from raw text (without envelope wrapper).

        Args:
            text: The transcript text
            tenant_id: Tenant UUID
            account_id: Optional account ID
            user_id: Optional user ID
            meeting_title: Optional meeting title
            participants: Optional list of participant names

        Returns:
            ExtractionOutput with ActionItem objects
        """
        # Create a minimal Interaction record
        interaction_id = uuid4()
        interaction = Interaction(
            interaction_id=interaction_id,
            tenant_id=tenant_id,
            account_id=account_id,
            interaction_type=InteractionType.TRANSCRIPT,
            title=meeting_title,
            content_text=text,
            timestamp=datetime.now(tz=None),
            user_id=user_id,
        )

        # Extract action items
        extraction_result = await self._extract_action_items(
            transcript_text=text,
            meeting_title=meeting_title,
            participants=participants,
            user_name=user_name,
        )

        # Convert to ActionItem models with embeddings
        action_items = await self._create_action_items(
            extractions=extraction_result.action_items,
            tenant_id=tenant_id,
            account_id=account_id,
            interaction_id=interaction_id,
            user_id=user_id,
        )

        # Update interaction with count
        interaction.action_item_count = len(action_items)
        interaction.processed_at = datetime.now(tz=None)

        return ExtractionOutput(
            interaction=interaction,
            action_items=action_items,
            raw_extractions=extraction_result.action_items,
            extraction_notes=extraction_result.extraction_notes,
        )

    async def _extract_action_items(
        self,
        transcript_text: str,
        meeting_title: str | None = None,
        participants: list[str] | None = None,
        user_name: str | None = None,
    ) -> ExtractionResult:
        """
        Call OpenAI to extract action items from transcript.

        Args:
            transcript_text: The transcript text
            meeting_title: Optional meeting title
            participants: Optional participant list
            user_name: Optional recording user name (for is_user_owned tagging)

        Returns:
            ExtractionResult with raw extractions
        """
        messages = build_extraction_prompt(
            transcript_text=transcript_text,
            meeting_title=meeting_title,
            participants=participants,
            user_name=user_name,
        )

        result = await self.openai_client.chat_completion_structured(
            messages=messages,
            response_model=ExtractionResult,
        )

        return result

    async def _create_action_items(
        self,
        extractions: list[ExtractedActionItem],
        tenant_id: UUID,
        account_id: str | None,
        interaction_id: UUID,
        user_id: str | None,
        pg_user_id: UUID | None = None,
    ) -> list[ActionItem]:
        """
        Convert raw extractions to ActionItem models with embeddings.

        Args:
            extractions: Raw extraction results
            tenant_id: Tenant UUID
            account_id: Account ID
            interaction_id: Source interaction ID
            user_id: User ID

        Returns:
            List of ActionItem models with embeddings
        """
        if not extractions:
            return []

        # Generate embeddings in batch for efficiency
        texts_to_embed = [e.action_item_text for e in extractions]
        embeddings = await self.openai_client.create_embeddings_batch(texts_to_embed)

        # Create ActionItem models
        action_items = []
        for extraction, embedding in zip(extractions, embeddings):
            # Determine initial status
            if extraction.is_status_update and extraction.implied_status:
                try:
                    status = ActionItemStatus(extraction.implied_status)
                except ValueError:
                    status = ActionItemStatus.OPEN
            else:
                status = ActionItemStatus.OPEN

            # Compute priority score from extraction dimensions
            priority = compute_priority_score(
                impact=extraction.score_impact,
                urgency=extraction.score_urgency,
                specificity=extraction.score_specificity,
                confidence=extraction.confidence,
            )

            action_item = ActionItem(
                id=uuid4(),
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text=extraction.action_item_text,
                summary=extraction.summary,
                owner=extraction.owner,
                owner_type=extraction.owner_type,
                is_user_owned=extraction.is_user_owned,
                conversation_context=extraction.conversation_context,
                status=status,
                source_interaction_id=interaction_id,
                user_id=user_id,
                pg_user_id=pg_user_id,
                embedding=embedding,
                embedding_current=embedding,  # Same as original initially
                confidence=extraction.confidence,
                # First-class scoring fields (Phase 4)
                commitment_strength=extraction.commitment_strength,
                score_impact=extraction.score_impact,
                score_urgency=extraction.score_urgency,
                score_specificity=extraction.score_specificity,
                score_effort=extraction.score_effort,
                priority_score=priority,
                definition_of_done=extraction.definition_of_done,
                attributes={
                    'decision_context': extraction.decision_context,
                },
            )

            # TODO: Parse due_date_text into actual datetime
            # This would require additional date parsing logic

            action_items.append(action_item)

        return action_items
