"""
Merge synthesis prompts and response models.

When an extracted action item matches an existing one, these prompts
guide the LLM to synthesize the merged content appropriately.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Import shared types from extraction prompts
from .extract_action_items import ImpliedStatusType


# =============================================================================
# Response Models for Structured Output
# =============================================================================


class MergedActionItem(BaseModel):
    """Result of merging an existing action item with new extraction."""

    action_item_text: str = Field(
        ...,
        description='Synthesized action item text combining original and new information. '
        'Preserve the core commitment while incorporating any new details or context.',
    )
    summary: str = Field(
        ...,
        description='Updated 1-sentence summary reflecting the current state of the action item.',
    )
    evolution_summary: str = Field(
        ...,
        description='A brief description of how this action item evolved (e.g., '
        '"Updated with additional context about timeline" or "Status changed to completed").',
    )
    implied_status: ImpliedStatusType | None = Field(
        default=None,
        description='If the merge implies a status change, what status? '
        'Must be one of: "completed", "in_progress", "cancelled", "deferred". '
        'Null if status should remain unchanged.',
    )
    owner: str = Field(
        ...,
        description='The resolved owner. Usually stays the same unless explicitly reassigned. '
        'Use the most complete/canonical form of the name.',
    )
    owner_type: str = Field(
        default="named",
        description='Owner identification type: "named", "role_inferred", or "unconfirmed"',
    )
    is_user_owned: bool = Field(
        default=False,
        description='Whether this action item belongs to the recording user',
    )
    due_date_text: str | None = Field(
        default=None,
        description='Updated due date or timeframe if mentioned in the new extraction. '
        'Null if no new timeline information.',
    )
    should_update_embedding: bool = Field(
        ...,
        description='True if the action_item_text changed significantly enough to warrant '
        're-embedding. False for minor wording changes or status-only updates.',
    )
    merge_notes: str | None = Field(
        default=None,
        description='Optional notes about the merge decision (e.g., "New extraction adds '
        'recipient information that was missing").',
    )


# =============================================================================
# Prompt Templates
# =============================================================================


MERGE_SYNTHESIS_SYSTEM_PROMPT = """You are an expert at synthesizing action item updates from sales call transcripts.

Given an EXISTING ACTION ITEM and a NEW EXTRACTION that refers to the same task, your job is to:
1. Synthesize the content appropriately
2. Determine if status should change
3. Decide if the embedding needs updating

**When to update text:**
- New extraction adds meaningful details (recipient, specific documents, etc.)
- New extraction clarifies ambiguity in the original
- DO NOT update if new text is just a rephrasing with no new info

**When to update status:**
- New extraction explicitly indicates completion ("I sent it", "Done!")
- New extraction indicates progress ("Working on it", "Started drafting")
- New extraction indicates deferral ("Let's push this to next week")
- New extraction indicates cancellation ("Actually, we don't need this anymore")

**When to update embedding:**
- The action_item_text changes substantially (new deliverable details, different scope)
- DO NOT update for minor wording changes or status updates only

**Owner resolution:**
- Keep the existing owner unless the new extraction explicitly reassigns the task
- Use the most complete name form (prefer "John Smith" over "John")

**Owner type resolution:**
- Prefer "named" owners over "role_inferred" over "unconfirmed"
- If the existing owner is "unconfirmed" and the new extraction has a name, upgrade to the name
- Preserve `owner_type` and `is_user_owned` from whichever source has higher confidence"""


MERGE_SYNTHESIS_USER_PROMPT_TEMPLATE = """Synthesize an update for this action item.

<existing_action_item>
Text: {existing_text}
Summary: {existing_summary}
Owner: {existing_owner}
Owner Type: {existing_owner_type}
Is User Owned: {existing_is_user_owned}
Status: {existing_status}
Due Date: {existing_due_date}
Context: {existing_context}
Created: {existing_created}
</existing_action_item>

<new_extraction>
Text: {new_text}
Summary: {new_summary}
Owner: {new_owner}
Owner Type: {new_owner_type}
Is User Owned: {new_is_user_owned}
Context: {new_context}
Is Status Update: {new_is_status_update}
Implied Status: {new_implied_status}
Due Date: {new_due_date}
</new_extraction>

<merge_recommendation>{merge_recommendation}</merge_recommendation>

How should these be merged?"""


def build_merge_prompt(
    existing_text: str,
    existing_summary: str,
    existing_owner: str,
    existing_status: str,
    existing_due_date: str | None,
    existing_context: str,
    existing_created: str,
    new_text: str,
    new_summary: str,
    new_owner: str,
    new_context: str,
    new_is_status_update: bool,
    new_implied_status: str | None,
    new_due_date: str | None,
    merge_recommendation: str,
    existing_owner_type: str = 'named',
    existing_is_user_owned: bool = False,
    new_owner_type: str = 'named',
    new_is_user_owned: bool = False,
) -> list[dict[str, str]]:
    """
    Build the merge synthesis prompt messages for OpenAI.

    Args:
        existing_*: Fields from the existing ActionItem
        new_*: Fields from the newly extracted item
        merge_recommendation: The recommendation from deduplication ('merge' or 'update_status')

    Returns:
        List of message dicts for OpenAI chat completion
    """
    user_prompt = MERGE_SYNTHESIS_USER_PROMPT_TEMPLATE.format(
        existing_text=existing_text,
        existing_summary=existing_summary,
        existing_owner=existing_owner,
        existing_owner_type=existing_owner_type,
        existing_is_user_owned=str(existing_is_user_owned),
        existing_status=existing_status,
        existing_due_date=existing_due_date or 'Not specified',
        existing_context=existing_context or 'No additional context',
        existing_created=existing_created,
        new_text=new_text,
        new_summary=new_summary,
        new_owner=new_owner,
        new_owner_type=new_owner_type,
        new_is_user_owned=str(new_is_user_owned),
        new_context=new_context,
        new_is_status_update=str(new_is_status_update),
        new_implied_status=new_implied_status or 'None',
        new_due_date=new_due_date or 'Not specified',
        merge_recommendation=merge_recommendation,
    )

    return [
        {'role': 'system', 'content': MERGE_SYNTHESIS_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
