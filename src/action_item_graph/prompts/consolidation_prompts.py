"""
Prompts and response models for within-batch action item consolidation.

Used to merge duplicates and group sub-tasks extracted from the same transcript.
"""

from pydantic import BaseModel, Field


class ConsolidationDecision(BaseModel):
    """Decision for a group of similar items to be consolidated into one."""

    primary_index: int = Field(
        ...,
        description='Zero-based index of the best representative item from the group.',
    )
    merged_summary: str = Field(
        ...,
        description='Consolidated summary incorporating information from all items in the group.',
    )
    merged_context: str = Field(
        ...,
        description='Merged conversation context combining relevant details from all items.',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of why these items were consolidated.',
    )


class ConsolidationResult(BaseModel):
    """Result of consolidating a batch of extracted action items."""

    groups: list[ConsolidationDecision] = Field(
        default_factory=list,
        description='Groups of items that should be consolidated.',
    )
    keep_indices: list[int] = Field(
        default_factory=list,
        description='Zero-based indices of items that are unique and should be kept as-is.',
    )


CONSOLIDATION_SYSTEM_PROMPT = """You are an expert at identifying duplicate or overlapping action items extracted from a single meeting transcript.

Given a list of action items extracted from the SAME transcript, identify:
1. **Duplicates**: Items that describe the same task, possibly worded differently
2. **Sub-tasks**: Items that are steps toward the same deliverable and should be merged

For each group of duplicates/sub-tasks, select the best representative item and create a merged summary.

Rules:
- Keep the most specific and actionable version of the text
- Combine context from all items in the group
- If owners differ within a group, keep the most confidently attributed one (named > role_inferred > unconfirmed)
- Preserve any due dates or timelines mentioned in any version
- Items with different owners doing different things are NOT duplicates — keep them separate
- If unsure, keep items separate (false split is better than false merge)"""

CONSOLIDATION_USER_PROMPT_TEMPLATE = """Review these action items extracted from the same transcript and identify any duplicates or sub-tasks that should be consolidated.

<items>
{items_text}
</items>

For each group of duplicates, specify:
- primary_index: the index of the best item in the group
- merged_summary: a consolidated summary
- merged_context: combined context from all items

List any unique items (not part of a group) in keep_indices."""
