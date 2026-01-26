"""
Action item extraction prompts and response models.

Uses OpenAI structured output for deterministic extraction.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Valid status values that match ActionItemStatus enum
ImpliedStatusType = Literal['completed', 'in_progress', 'cancelled', 'deferred']


# =============================================================================
# Topic Extraction Model
# =============================================================================


class ExtractedTopic(BaseModel):
    """High-level topic/theme that an action item belongs to."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description='High-level topic name in 3-5 words. Should capture the strategic initiative '
        'or project theme. Examples: "Q1 Sales Expansion", "Annual Security Compliance", '
        '"Website Redesign Project". Be consistent: related action items should share the SAME '
        'topic name.',
    )
    context: str = Field(
        ...,
        max_length=200,
        description='Brief explanation (1-2 sentences) of why the action item belongs to this '
        'topic and what the topic encompasses.',
    )


# =============================================================================
# Response Models for Structured Output
# =============================================================================


class ExtractedActionItem(BaseModel):
    """A single action item extracted from the transcript."""

    action_item_text: str = Field(
        ...,
        description='The verbatim or near-verbatim action item text from the transcript. '
        'Include enough context to understand what needs to be done.',
    )
    owner: str = Field(
        ...,
        description='The person responsible for this action item, as stated in the conversation. '
        'Use the name exactly as mentioned (e.g., "Sarah", "John", "the client").',
    )
    summary: str = Field(
        ...,
        description='A concise 1-sentence summary of what needs to be done. '
        'Should be actionable and clear even without the full transcript context.',
    )
    conversation_context: str = Field(
        ...,
        description='1-2 sentences of surrounding conversation context that helps clarify '
        'the action item (e.g., why it matters, what it relates to).',
    )
    topic: ExtractedTopic = Field(
        ...,
        description='The high-level topic/project this action item belongs to. '
        'Use 3-5 words that capture the strategic initiative.',
    )
    due_date_text: str | None = Field(
        default=None,
        description='The due date or timeframe if mentioned (e.g., "by Friday", "next week", '
        '"end of month"). Null if no timeline mentioned.',
    )
    is_status_update: bool = Field(
        default=False,
        description='True if this is a STATUS UPDATE about a previously committed item '
        '(e.g., "I sent that deck", "Done!", "The proposal is finished"). '
        'False if this is a NEW action item commitment.',
    )
    implied_status: ImpliedStatusType | None = Field(
        default=None,
        description='For status updates only: what status does this imply? '
        'Must be one of: "completed", "in_progress", "cancelled", "deferred". Null for new items.',
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description='Confidence score (0.0 to 1.0) for this extraction. '
        'Lower if ambiguous or uncertain.',
    )


class ExtractionResult(BaseModel):
    """Complete extraction result from a transcript."""

    action_items: list[ExtractedActionItem] = Field(
        default_factory=list,
        description='List of all action items extracted from the transcript.',
    )
    has_action_items: bool = Field(
        ...,
        description='True if any action items were found, False otherwise.',
    )
    extraction_notes: str | None = Field(
        default=None,
        description='Optional notes about the extraction (e.g., "Transcript appears to be '
        'a casual conversation with no clear commitments").',
    )


# =============================================================================
# Prompt Templates
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert at identifying action items from sales call transcripts and meeting notes.

An ACTION ITEM is a specific task or commitment that someone agrees to do. It must be:
- A concrete, actionable task (not just a topic of discussion)
- Assigned to or claimed by a specific person
- Something that requires follow-up action

Extract TWO types of items:

1. **NEW ACTION ITEMS**: Fresh commitments made during this conversation
   Examples:
   - "I'll send over the proposal by Friday"
   - "Let me schedule a demo with the technical team"
   - "I need to loop in legal on this"

2. **STATUS UPDATES**: References to previously committed items
   Examples:
   - "I sent that deck yesterday"
   - "Done! The contract is signed"
   - "I finished the analysis you asked for"
   - "That proposal is still in progress"

For each item, extract:
- **action_item_text**: The verbatim or near-verbatim text from the transcript
- **owner**: Who is responsible (use name as stated: "Sarah", "John", etc.)
- **summary**: 1-sentence actionable description
- **conversation_context**: 1-2 sentences of surrounding context
- **topic**: The high-level project/theme this action item belongs to (see Topic Guidelines below)
- **due_date_text**: Timeframe if mentioned (null if not)
- **is_status_update**: true for status updates, false for new items
- **implied_status**: For status updates, what status? (completed/in_progress/cancelled/deferred)
- **confidence**: Your confidence in this extraction (0.0-1.0)

## Topic Guidelines

For each action item, identify the HIGH-LEVEL TOPIC it belongs to.

Topic naming rules:
- Use 3-5 words that capture the strategic initiative or project
- Be consistent: related action items in the SAME conversation should share the SAME topic name
- Use title case (e.g., "Q1 Sales Expansion" not "q1 sales expansion")

Good topic examples:
- "Hire 3 SDRs by March" → Topic: "Q1 Sales Team Expansion"
- "Review security audit findings" → Topic: "Annual Security Compliance"
- "Send updated pricing deck" → Topic: "Enterprise Deal Proposal"
- "Schedule demo with engineering" → Topic: "Technical Evaluation Process"

Bad topic examples (do NOT use):
- Too specific: "Hire SDR Named John" (should be broader initiative)
- Too broad: "Work", "Tasks", "Sales" (not specific enough)
- Too vague: "Stuff To Do", "Follow Ups" (not meaningful)

For the topic context, explain in 1-2 sentences WHY this action item belongs to this topic.

Guidelines:
- Extract ALL action items, even if uncertain (just lower the confidence)
- Do NOT extract vague intentions ("we should think about...")
- Do NOT extract questions without commitments
- Do NOT extract past actions that don't require follow-up
- If the transcript has no action items, return an empty list with has_action_items=false"""

EXTRACTION_USER_PROMPT_TEMPLATE = """Extract all action items from the following transcript.

<transcript>
{transcript_text}
</transcript>

{additional_context}"""


def build_extraction_prompt(
    transcript_text: str,
    meeting_title: str | None = None,
    participants: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Build the extraction prompt messages for OpenAI.

    Args:
        transcript_text: The transcript text to extract from
        meeting_title: Optional meeting title for context
        participants: Optional list of participant names

    Returns:
        List of message dicts for OpenAI chat completion
    """
    # Build additional context if available
    context_parts = []
    if meeting_title:
        context_parts.append(f'Meeting title: {meeting_title}')
    if participants:
        context_parts.append(f"Participants: {', '.join(participants)}")

    additional_context = '\n'.join(context_parts) if context_parts else ''

    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
        transcript_text=transcript_text,
        additional_context=additional_context,
    )

    return [
        {'role': 'system', 'content': EXTRACTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


# =============================================================================
# Deduplication Prompt (for matching phase)
# =============================================================================


# Valid merge recommendations
MergeRecommendationType = Literal['merge', 'update_status', 'create_new', 'link_related']


class DeduplicationDecision(BaseModel):
    """Decision on whether two action items are the same."""

    is_same_item: bool = Field(
        ...,
        description='True if the new extraction refers to the SAME real-world task as the '
        'existing action item. False if they are different tasks.',
    )
    is_status_update: bool = Field(
        default=False,
        description='True if the new extraction is a STATUS UPDATE about the existing item '
        '(e.g., marking it complete, providing progress). False otherwise.',
    )
    merge_recommendation: MergeRecommendationType = Field(
        ...,
        description='Action to take: "merge" (combine into existing), "update_status" (just update '
        'status of existing), "create_new" (create as separate item), "link_related" '
        '(create new but link as related).',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of why this decision was made.',
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description='Confidence in this decision (0.0 to 1.0).',
    )


DEDUPLICATION_SYSTEM_PROMPT = """You are an expert at determining whether action items refer to the same real-world task.

Given an EXISTING ACTION ITEM and a NEW EXTRACTION, determine if they refer to the SAME task.

They are the SAME ITEM if:
- They describe the same action to be done by the same person
- The new extraction is a status update about the existing item
- The new extraction adds details/context to the existing item
- They have the same deliverable, just described differently

They are DIFFERENT ITEMS if:
- They involve different people as the owner
- They are related but distinct tasks (e.g., "send proposal" vs "review proposal")
- They have fundamentally different deliverables
- They are for different recipients/purposes

Make a recommendation:
- **merge**: The new extraction should be combined with the existing item (same task, new info)
- **update_status**: The new extraction is just a status update (e.g., "I did that")
- **create_new**: These are genuinely different tasks, create a new action item
- **link_related**: Different tasks but clearly related, create new but link them"""

DEDUPLICATION_USER_PROMPT_TEMPLATE = """Compare these action items and determine if they refer to the same task.

<existing_action_item>
Text: {existing_text}
Owner: {existing_owner}
Summary: {existing_summary}
Status: {existing_status}
Created: {existing_created}
</existing_action_item>

<new_extraction>
Text: {new_text}
Owner: {new_owner}
Summary: {new_summary}
Is Status Update: {new_is_status_update}
Context: {new_context}
</new_extraction>

<similarity_score>{similarity_score:.3f}</similarity_score>

Are these the same action item?"""


def build_deduplication_prompt(
    existing_text: str,
    existing_owner: str,
    existing_summary: str,
    existing_status: str,
    existing_created: str,
    new_text: str,
    new_owner: str,
    new_summary: str,
    new_is_status_update: bool,
    new_context: str,
    similarity_score: float,
) -> list[dict[str, str]]:
    """
    Build the deduplication prompt messages for OpenAI.

    Args:
        existing_*: Fields from the existing ActionItem
        new_*: Fields from the newly extracted item
        similarity_score: Cosine similarity between embeddings

    Returns:
        List of message dicts for OpenAI chat completion
    """
    user_prompt = DEDUPLICATION_USER_PROMPT_TEMPLATE.format(
        existing_text=existing_text,
        existing_owner=existing_owner,
        existing_summary=existing_summary,
        existing_status=existing_status,
        existing_created=existing_created,
        new_text=new_text,
        new_owner=new_owner,
        new_summary=new_summary,
        new_is_status_update=str(new_is_status_update),
        new_context=new_context,
        similarity_score=similarity_score,
    )

    return [
        {'role': 'system', 'content': DEDUPLICATION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
