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
            id=envelope.interaction_id or uuid4(),
            tenant_id=envelope.tenant_id,
            account_id=envelope.account_id,
            interaction_type=InteractionType(envelope.interaction_type.value),
            title=envelope.meeting_title,
            transcript_text=envelope.content.text,
            occurred_at=envelope.timestamp,
            duration_seconds=envelope.duration_seconds,
            source=envelope.source.value,
            user_id=envelope.user_id,
            extras=envelope.extras,
        )

        # Extract action items
        extraction_result = await self._extract_action_items(
            transcript_text=envelope.content.text,
            meeting_title=envelope.meeting_title,
            participants=envelope.contact_ids if envelope.contact_ids else None,
        )

        # Convert to ActionItem models with embeddings
        action_items = await self._create_action_items(
            extractions=extraction_result.action_items,
            tenant_id=envelope.tenant_id,
            account_id=envelope.account_id,
            interaction_id=interaction.id,
            user_id=envelope.user_id,
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

    async def extract_from_text(
        self,
        text: str,
        tenant_id: UUID,
        account_id: str | None = None,
        user_id: str | None = None,
        meeting_title: str | None = None,
        participants: list[str] | None = None,
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
            id=interaction_id,
            tenant_id=tenant_id,
            account_id=account_id,
            interaction_type=InteractionType.TRANSCRIPT,
            title=meeting_title,
            transcript_text=text,
            occurred_at=datetime.now(tz=None),
            user_id=user_id,
        )

        # Extract action items
        extraction_result = await self._extract_action_items(
            transcript_text=text,
            meeting_title=meeting_title,
            participants=participants,
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
    ) -> ExtractionResult:
        """
        Call OpenAI to extract action items from transcript.

        Args:
            transcript_text: The transcript text
            meeting_title: Optional meeting title
            participants: Optional participant list

        Returns:
            ExtractionResult with raw extractions
        """
        messages = build_extraction_prompt(
            transcript_text=transcript_text,
            meeting_title=meeting_title,
            participants=participants,
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

            action_item = ActionItem(
                id=uuid4(),
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text=extraction.action_item_text,
                summary=extraction.summary,
                owner=extraction.owner,
                conversation_context=extraction.conversation_context,
                status=status,
                source_interaction_id=interaction_id,
                user_id=user_id,
                embedding=embedding,
                embedding_current=embedding,  # Same as original initially
                confidence=extraction.confidence,
            )

            # TODO: Parse due_date_text into actual datetime
            # This would require additional date parsing logic

            action_items.append(action_item)

        return action_items
