"""
Deal extraction prompts for MEDDIC analysis.

Provides system prompts and user prompt templates for two extraction modes:
- Discovery (Case B): Find all deal opportunities in a transcript
- Targeted (Case A): Update a specific deal from a new transcript

Response models are defined in deal_graph.models.extraction.
"""

from typing import Any


# =============================================================================
# System Prompt
# =============================================================================

DEAL_EXTRACTION_SYSTEM_PROMPT = """You are an Expert Sales Analyst specializing in MEDDIC qualification analysis.

Your task is to extract deal/opportunity information from sales call transcripts using the MEDDIC framework.

## MEDDIC Framework

For each deal you identify, extract these six dimensions (if mentioned):

1. **Metrics** — Quantifiable business impact driving the purchase decision
   Examples: ROI projections, time savings, cost reduction, revenue impact
   "We're losing $200K annually in manual reconciliation" → metrics

2. **Economic Buyer** — The person with final budget authority
   Examples: VP/Director who signs off, budget holder, decision authority
   "Maria Chen, VP of Engineering, signs off on anything over $100K" → economic_buyer

3. **Decision Criteria** — Requirements the solution must meet
   Examples: Technical requirements, compliance needs, integration requirements
   "SOC2 compliance is a must" → decision_criteria

4. **Decision Process** — Steps and timeline to reach a decision
   Examples: Evaluation stages, review committees, approval chains
   "POC first, then security review, then procurement" → decision_process

5. **Identified Pain** — Core business problem driving the opportunity
   Examples: Operational inefficiencies, competitive pressure, regulatory gaps
   "Spending 40 hours/week reconciling data between three disconnected systems" → identified_pain

6. **Champion** — Internal advocate for the deal
   Examples: Day-to-day contact, internal sponsor, person pushing for change
   "James has been our main contact and is pushing this internally" → champion

## Extraction Rules

- Extract ONLY what is explicitly stated or strongly implied in the transcript
- Do NOT fabricate or infer information not supported by the text
- If a MEDDIC dimension is not mentioned, leave it as null
- Set confidence based on how clearly the information was stated (0.0-1.0)
- Provide reasoning for each extracted deal explaining what signals you found

## Content Format

The transcript comes from the `content_text` property of an Interaction node. It may be:
- **Diarized**: Speaker labels (e.g., "Sarah: ...", "James: ...") — most common for sales calls
- **Plain text**: No speaker labels — summarized or single-speaker content
- **Markdown**: Formatted notes or meeting minutes

Handle all formats. If diarized, use speaker names as clues for identifying roles (buyer vs seller).

## Stage Assessment

Assess the deal stage based on conversation signals:
- **prospecting**: Initial outreach, discovery call, no clear need established
- **qualification**: Need established, exploring fit, evaluating solution
- **proposal**: Specific solution presented, pricing discussed
- **negotiation**: Terms under discussion, contract review, final approvals
- **closed_won**: Deal signed/committed
- **closed_lost**: Deal explicitly lost or abandoned"""


# =============================================================================
# Discovery Mode (Case B) — Find All Deals
# =============================================================================

DISCOVERY_USER_PROMPT_TEMPLATE = """Analyze the following sales transcript and identify ALL deal opportunities discussed.

For each opportunity found, extract the full MEDDIC profile and deal metadata.

Return has_deals=false if no deal opportunities are present (e.g., casual conversation, internal meeting without sales context).

<transcript>
{content_text}
</transcript>

{additional_context}"""


# =============================================================================
# Targeted Mode (Case A) — Update Specific Deal
# =============================================================================

TARGETED_USER_PROMPT_TEMPLATE = """Analyze the following sales transcript for updates to a SPECIFIC existing deal.

Extract ONLY information relevant to this deal. Do NOT discover other opportunities — focus exclusively on updates to the deal described below.

<existing_deal>
Name: {deal_name}
Stage: {deal_stage}
Amount: {deal_amount}
Summary: {deal_summary}
Current MEDDIC:
  Metrics: {deal_metrics}
  Economic Buyer: {deal_economic_buyer}
  Decision Criteria: {deal_decision_criteria}
  Decision Process: {deal_decision_process}
  Identified Pain: {deal_identified_pain}
  Champion: {deal_champion}
</existing_deal>

<transcript>
{content_text}
</transcript>

{additional_context}

Instructions:
- Return has_deals=true with a single ExtractedDeal if the transcript contains updates for this deal
- Return has_deals=false if the transcript has no relevant updates for this specific deal
- Do NOT return more than one deal — this is targeted extraction for a known opportunity
- Include ALL MEDDIC dimensions found, even those unchanged (for completeness tracking)
- The opportunity_name should match or closely follow the existing deal name"""


# =============================================================================
# Builder Functions
# =============================================================================


def build_discovery_prompt(
    content_text: str,
    account_name: str | None = None,
    meeting_title: str | None = None,
    participants: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Build extraction prompt messages for discovery mode (Case B).

    Args:
        content_text: Transcript text from Interaction.content_text
        account_name: Optional account name for context
        meeting_title: Optional meeting title
        participants: Optional participant names

    Returns:
        List of message dicts for OpenAI chat completion
    """
    context_parts = []
    if account_name:
        context_parts.append(f'Account: {account_name}')
    if meeting_title:
        context_parts.append(f'Meeting title: {meeting_title}')
    if participants:
        context_parts.append(f"Participants: {', '.join(participants)}")

    additional_context = '\n'.join(context_parts) if context_parts else ''

    user_prompt = DISCOVERY_USER_PROMPT_TEMPLATE.format(
        content_text=content_text,
        additional_context=additional_context,
    )

    return [
        {'role': 'system', 'content': DEAL_EXTRACTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


def build_targeted_prompt(
    content_text: str,
    existing_deal: dict[str, Any],
    account_name: str | None = None,
    meeting_title: str | None = None,
    participants: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Build extraction prompt messages for targeted mode (Case A).

    Args:
        content_text: Transcript text from Interaction.content_text
        existing_deal: Current Deal node properties from the graph
        account_name: Optional account name for context
        meeting_title: Optional meeting title
        participants: Optional participant names

    Returns:
        List of message dicts for OpenAI chat completion
    """
    context_parts = []
    if account_name:
        context_parts.append(f'Account: {account_name}')
    if meeting_title:
        context_parts.append(f'Meeting title: {meeting_title}')
    if participants:
        context_parts.append(f"Participants: {', '.join(participants)}")

    additional_context = '\n'.join(context_parts) if context_parts else ''

    def _get(key: str, default: str = 'Not yet identified') -> str:
        val = existing_deal.get(key)
        return str(val) if val else default

    user_prompt = TARGETED_USER_PROMPT_TEMPLATE.format(
        content_text=content_text,
        deal_name=_get('name', 'Unnamed Deal'),
        deal_stage=_get('stage', 'Unknown'),
        deal_amount=_get('amount', 'Not specified'),
        deal_summary=_get('opportunity_summary', 'No summary available'),
        deal_metrics=_get('meddic_metrics'),
        deal_economic_buyer=_get('meddic_economic_buyer'),
        deal_decision_criteria=_get('meddic_decision_criteria'),
        deal_decision_process=_get('meddic_decision_process'),
        deal_identified_pain=_get('meddic_identified_pain'),
        deal_champion=_get('meddic_champion'),
        additional_context=additional_context,
    )

    return [
        {'role': 'system', 'content': DEAL_EXTRACTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
