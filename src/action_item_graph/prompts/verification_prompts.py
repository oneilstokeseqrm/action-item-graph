"""
Prompts and response models for LLM-as-Judge action item verification.

Adversarial second pass that validates extracted items against quality criteria.
Deliberately uses a different perspective from the extraction prompt to catch
items that slipped through the commitment framework.
"""

from pydantic import BaseModel, Field


class VerificationVerdict(BaseModel):
    """Verdict for a single action item from the LLM-as-Judge."""

    index: int = Field(
        ...,
        description='Zero-based index of the item being evaluated.',
    )
    is_actionable: bool = Field(
        ...,
        description='True if this is a genuine, actionable commitment with a clear owner '
        'and deliverable. False if it is vague, observational, or not a real commitment.',
    )
    adjusted_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description='Adjusted confidence score after adversarial review. '
        'Lower than the original if the item has issues.',
    )
    issues: list[str] = Field(
        default_factory=list,
        description='List of specific issues found (e.g., "Owner is vague", '
        '"No clear deliverable", "Observation not commitment").',
    )
    reasoning: str = Field(
        ...,
        description='Brief explanation of the verdict.',
    )


class VerificationResult(BaseModel):
    """Result of verifying a batch of extracted action items."""

    verdicts: list[VerificationVerdict] = Field(
        default_factory=list,
        description='One verdict per input item, in the same order.',
    )


VERIFICATION_SYSTEM_PROMPT = """You are a QUALITY AUDITOR reviewing action items extracted from a meeting transcript.

Your job is to be ADVERSARIAL — challenge each item and look for reasons it should NOT be included. You are deliberately skeptical because the extraction system tends to over-extract.

For each item, ask yourself:

1. **Is this a real commitment?** Did someone actually promise to do something, or is this an observation, question, or vague intention?
2. **Is the owner identifiable?** Can you tell WHO is responsible, or is it attributed to a vague group/role?
3. **Is the deliverable specific?** Could someone look at this and know exactly what "done" looks like?
4. **Is it actionable?** Can someone actually go DO this, or is it too abstract?
5. **Is it significant enough?** Is this a meaningful commitment or trivial filler?

## Confidence Scoring Guide

- **0.9-1.0**: Crystal clear commitment, named owner, specific deliverable, timeline mentioned
- **0.7-0.89**: Solid commitment with minor ambiguity (e.g., timeline unclear, deliverable could be more specific)
- **0.5-0.69**: Borderline — commitment is implied but not explicit, or owner attribution is uncertain
- **0.3-0.49**: Weak — likely an observation or vague intention masquerading as an action item
- **0.0-0.29**: Should not have been extracted — not a commitment at all

## Common Extraction Errors to Watch For

- **Observations extracted as tasks**: "The team is working on X" is NOT an action item
- **Past completions without follow-up**: "We did that last week" is NOT an action item (unless there's a new follow-up)
- **Third-party expectations**: "They'll probably send it" — who committed to what?
- **Meeting logistics mistaken for deliverables**: "Let's circle back" is NOT an action item
- **Duplicate information**: Same task extracted with slightly different wording
- **Conditional items treated as firm**: "If we get budget approval, we could..." is NOT a firm commitment

Set is_actionable=false for items that fail any of criteria 1-3 above."""

VERIFICATION_USER_PROMPT_TEMPLATE = """Review these action items that were extracted from a meeting transcript. For each one, determine if it is a genuine, actionable commitment.

<transcript_excerpt>
{transcript_excerpt}
</transcript_excerpt>

<extracted_items>
{items_text}
</extracted_items>

For EACH item (by index), provide:
- is_actionable: Does this pass the quality bar?
- adjusted_confidence: Your confidence in this item's quality (0.0-1.0)
- issues: Any problems found (empty list if clean)
- reasoning: Brief explanation"""
