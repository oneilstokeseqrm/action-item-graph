"""
Prompts and response models for role-to-name owner resolution.

Used when an action item has owner_type='role_inferred' (e.g., "the account manager")
and known named owners exist for the account. The LLM tries to map the role
description to a specific person.
"""

from pydantic import BaseModel, Field


class RoleResolutionDecision(BaseModel):
    """Decision on whether a role description maps to a known person."""

    resolved_name: str | None = Field(
        ...,
        description='The canonical name of the matched person, or null if no confident match.',
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Confidence in the resolution (0.0 = no match, 1.0 = certain match).',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of why this resolution was (or was not) made.',
    )


ROLE_RESOLUTION_SYSTEM_PROMPT = """You are an expert at resolving role descriptions to specific people in a business context.

Given a role description from a meeting transcript (e.g., "the account manager", "our sales lead")
and a list of known people associated with this account, determine if the role refers to one of them.

Rules:
- Only resolve if you are CONFIDENT the role matches a specific person
- Use contextual clues from the action item text and conversation context
- If multiple people could fit the role, return null (ambiguous)
- Common role mappings: "account manager"/"AE"/"rep" often = the primary salesperson
- If the role is too generic ("someone", "the team"), return null
- Set confidence >= 0.8 only for high-confidence matches"""

ROLE_RESOLUTION_USER_PROMPT_TEMPLATE = """Resolve this role description to a specific person.

<role_description>{role_description}</role_description>

<action_item_context>
Summary: {summary}
Text: {action_item_text}
Conversation context: {conversation_context}
</action_item_context>

<known_people>
{known_people_text}
</known_people>

Does this role refer to one of the known people? If so, which one?"""
