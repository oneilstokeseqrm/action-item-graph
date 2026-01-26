"""
Topic-related prompts for summary generation and topic matching confirmation.

Uses OpenAI structured output for deterministic responses.
"""

from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Topic Resolution Response Models
# =============================================================================


TopicDecisionType = Literal['link_existing', 'create_new']


class TopicMatchDecision(BaseModel):
    """Decision on whether an extracted topic matches an existing topic."""

    decision: TopicDecisionType = Field(
        ...,
        description='Whether to link to the existing topic or create a new one. '
        '"link_existing" if they refer to the same project/initiative, '
        '"create_new" if they are different topics.',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation (1-2 sentences) of why this decision was made.',
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description='Confidence in this decision (0.0 to 1.0).',
    )


class TopicSummary(BaseModel):
    """Generated or updated topic summary."""

    summary: str = Field(
        ...,
        max_length=500,
        description='Concise summary of the topic scope and what action items it encompasses. '
        'Should be 2-3 sentences that capture the strategic initiative.',
    )
    should_update_embedding: bool = Field(
        default=False,
        description='True if the summary has changed significantly enough to warrant '
        're-embedding. False for minor wording changes.',
    )


# =============================================================================
# Topic Match Confirmation Prompts
# =============================================================================


TOPIC_MATCH_SYSTEM_PROMPT = """You are an expert at determining whether action item topics refer to the same project or initiative.

Given an EXISTING TOPIC and a NEW EXTRACTED TOPIC, determine if they refer to the SAME strategic initiative.

They are the SAME TOPIC if:
- They describe the same project, initiative, or strategic goal
- They are different phases of the same overarching effort
- They use different words but clearly mean the same thing
- One is a subset or component of the other

They are DIFFERENT TOPICS if:
- They are for different projects, even if related domains
- They have fundamentally different goals or outcomes
- They involve different teams, stakeholders, or departments
- They are distinct initiatives that happen to share a theme

IMPORTANT: When uncertain, LEAN TOWARD "create_new" to avoid incorrect grouping.
Incorrect topic grouping is more harmful than having separate topics that could be merged later.

Examples:
- "Q1 Sales Hiring" vs "Sales Team Expansion" → SAME (same initiative, different wording)
- "Q1 Sales Hiring" vs "Engineering Hiring" → DIFFERENT (different departments)
- "Website Redesign" vs "Homepage Update" → SAME (subset of larger project)
- "Security Audit" vs "Compliance Review" → Could be SAME or DIFFERENT depending on context
- "Customer Onboarding" vs "Enterprise Sales" → DIFFERENT (different processes)"""


TOPIC_MATCH_USER_PROMPT_TEMPLATE = """Compare these topics and determine if they refer to the same initiative.

<existing_topic>
Name: {existing_name}
Summary: {existing_summary}
Action Items Count: {existing_count}
</existing_topic>

<new_extracted_topic>
Name: {new_name}
Context: {new_context}
Action Item Summary: {action_item_summary}
</new_extracted_topic>

<similarity_score>{similarity_score:.3f}</similarity_score>

Are these the same topic/initiative?"""


def build_topic_match_prompt(
    existing_name: str,
    existing_summary: str,
    existing_count: int,
    new_name: str,
    new_context: str,
    action_item_summary: str,
    similarity_score: float,
) -> list[dict[str, str]]:
    """
    Build the topic match confirmation prompt messages for OpenAI.

    Args:
        existing_name: Name of existing topic
        existing_summary: Summary of existing topic
        existing_count: Number of action items in existing topic
        new_name: Name from extracted topic
        new_context: Context from extracted topic
        action_item_summary: Summary of the action item being assigned
        similarity_score: Cosine similarity between topic embeddings

    Returns:
        List of message dicts for OpenAI chat completion
    """
    user_prompt = TOPIC_MATCH_USER_PROMPT_TEMPLATE.format(
        existing_name=existing_name,
        existing_summary=existing_summary or '(no summary yet)',
        existing_count=existing_count,
        new_name=new_name,
        new_context=new_context,
        action_item_summary=action_item_summary,
        similarity_score=similarity_score,
    )

    return [
        {'role': 'system', 'content': TOPIC_MATCH_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


# =============================================================================
# Topic Summary Generation Prompts
# =============================================================================


TOPIC_SUMMARY_SYSTEM_PROMPT = """You are an expert at summarizing project topics and initiatives.

Generate a concise summary (2-3 sentences) that captures:
1. What the topic/initiative is about
2. The types of action items it encompasses
3. The overall goal or outcome

The summary should be:
- Clear and specific (not generic)
- Useful for understanding what belongs in this topic
- Written in present tense, describing the ongoing initiative"""


TOPIC_SUMMARY_CREATE_TEMPLATE = """Create a summary for a new topic based on the first action item assigned to it.

<topic>
Name: {topic_name}
</topic>

<first_action_item>
Text: {action_item_text}
Summary: {action_item_summary}
Owner: {owner}
Context: {topic_context}
</first_action_item>

Generate a 2-3 sentence summary describing this topic/initiative."""


TOPIC_SUMMARY_UPDATE_TEMPLATE = """Update the topic summary to reflect a newly linked action item.

<topic>
Name: {topic_name}
Current Summary: {current_summary}
Total Action Items: {total_count}
</topic>

<new_action_item>
Text: {action_item_text}
Summary: {action_item_summary}
Owner: {owner}
</new_action_item>

Update the summary to incorporate this new action item while keeping it concise (2-3 sentences).
If the new action item doesn't significantly change the topic scope, you can keep the summary mostly the same.
Set should_update_embedding to true only if the topic scope has meaningfully expanded."""


def build_topic_summary_create_prompt(
    topic_name: str,
    action_item_text: str,
    action_item_summary: str,
    owner: str,
    topic_context: str,
) -> list[dict[str, str]]:
    """
    Build prompt for creating an initial topic summary.

    Args:
        topic_name: Name of the new topic
        action_item_text: Text of the triggering action item
        action_item_summary: Summary of the action item
        owner: Owner of the action item
        topic_context: Context explaining why action item belongs to topic

    Returns:
        List of message dicts for OpenAI chat completion
    """
    user_prompt = TOPIC_SUMMARY_CREATE_TEMPLATE.format(
        topic_name=topic_name,
        action_item_text=action_item_text,
        action_item_summary=action_item_summary,
        owner=owner,
        topic_context=topic_context,
    )

    return [
        {'role': 'system', 'content': TOPIC_SUMMARY_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


def build_topic_summary_update_prompt(
    topic_name: str,
    current_summary: str,
    total_count: int,
    action_item_text: str,
    action_item_summary: str,
    owner: str,
) -> list[dict[str, str]]:
    """
    Build prompt for updating a topic summary when a new action item is linked.

    Args:
        topic_name: Name of the topic
        current_summary: Current topic summary
        total_count: Total action items after this link
        action_item_text: Text of the newly linked action item
        action_item_summary: Summary of the action item
        owner: Owner of the action item

    Returns:
        List of message dicts for OpenAI chat completion
    """
    user_prompt = TOPIC_SUMMARY_UPDATE_TEMPLATE.format(
        topic_name=topic_name,
        current_summary=current_summary or '(no summary yet)',
        total_count=total_count,
        action_item_text=action_item_text,
        action_item_summary=action_item_summary,
        owner=owner,
    )

    return [
        {'role': 'system', 'content': TOPIC_SUMMARY_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
