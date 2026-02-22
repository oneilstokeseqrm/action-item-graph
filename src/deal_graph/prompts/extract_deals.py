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

DEAL_EXTRACTION_SYSTEM_PROMPT = """You are an Expert Sales Analyst specializing in deal qualification analysis.

Your task is to extract deal/opportunity information from sales call transcripts using the EQ Deal Intelligence framework.

## Core MEDDIC Dimensions

For each deal you identify, extract these six core dimensions as TEXT (if mentioned):

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

## Extended Ontology Dimensions

In addition to MEDDIC text fields, score these dimensions on a 0-3 scale using the `ontology_dimensions` list.
Return a DimensionExtraction for EACH dimension where evidence exists. Omit dimensions with no evidence.

### Competitive (category)
- **competitive_position**: Strength of position vs. alternatives
  0=Strong competitor advantage, 1=Competing equally, 2=Preferred vendor, 3=Sole/selected vendor
  Evidence: competitor mentions, positioning language, vendor evaluation status

- **incumbent_displacement_risk**: Likelihood of status quo winning (INVERSE — lower is riskier)
  0=Deeply entrenched incumbent, 1=Strong incumbent, 2=Weak/no incumbent, 3=Greenfield/no competition
  Evidence: current vendor mentions, switching cost references, status quo bias signals

### Commercial (category)
- **pricing_alignment**: Pricing model and level matches buyer expectations
  0=Deal-breaker objections, 1=Significant pushback, 2=Minor concerns, 3=Aligned/no issues
  Evidence: pricing discussions, budget reactions, packaging concerns

- **procurement_legal_progress**: How far through formal buying process
  0=Not started, 1=Initiated, 2=In review, 3=Approved/near-complete
  Evidence: procurement mentions, legal review, contract discussions, security review

### Engagement (category)
- **responsiveness**: Speed and quality of buyer responses
  0=Ghosting/>7 days, 1=4-7 day responses, 2=1-3 day responses, 3=Same-day/proactive
  Evidence: response time mentions, scheduling ease, engagement quality signals

### Timeline (category)
- **close_date_credibility**: Is the close date anchored to a real forcing function?
  0=Unknown/no date, 1=Aspirational target, 2=Soft target with rationale, 3=Hard deadline (event/budget/regulatory)
  Evidence: deadline mentions, budget cycle references, event-driven urgency

### Technical (category)
- **technical_fit**: Does the solution genuinely solve the problem?
  0=Poor fit/major gaps, 1=Significant gaps, 2=Minor gaps, 3=Strong/perfect fit
  Evidence: technical evaluation results, fit assessments, integration compatibility

- **integration_security_risk**: Are there technical blockers? (INVERSE — lower is riskier)
  0=Potential deal-breaker blockers, 1=Complex issues, 2=Manageable concerns, 3=No blockers
  Evidence: security review concerns, integration complexity, compliance blockers

### Qualification (category)
- **champion_strength**: Internal advocate with influence and motivation
  0=No champion identified or actively opposing, 1=Contact identified but unclear influence, 2=Champion with some influence showing advocacy, 3=Strong champion with executive access actively advocating
  Evidence: named advocate, internal coaching, lobbying for budget, scheduling internal reviews

- **economic_buyer_access**: Budget authority engaged in buying process
  0=Economic buyer unknown or not engaged, 1=EB identified but not yet engaged, 2=EB aware and partially engaged, 3=EB directly engaged with confirmed budget authority
  Evidence: budget holder mentioned, EB attended meetings, CFO sign-off

- **identified_pain**: Quantified business problem driving urgency
  0=No pain articulated, 1=Vague dissatisfaction, 2=Pain articulated clearly but not quantified, 3=Pain quantified with business impact and urgency
  Evidence: specific pain statements, cost/time impact metrics, priority rankings

- **metrics_business_case**: ROI or value proposition quantified and agreed
  0=No metrics or value discussion, 1=Metrics discussed at high level, 2=Metrics validated by buyer, 3=Compelling business case built and shared internally
  Evidence: ROI projections, cost savings validated, joint business case documents

- **decision_criteria_alignment**: Solution meets stated requirements
  0=Criteria unknown or major misfit, 1=Some criteria identified with gaps, 2=Most criteria mapped with minor gaps, 3=All criteria mapped and buyer confirmed fit
  Evidence: RFP criteria, technical evaluations, requirements mapping

- **decision_process_clarity**: Steps and timeline to purchase decision understood
  0=Process completely opaque, 1=Basic understanding with many unknowns, 2=Process mapped with milestones and stakeholders, 3=Detailed process confirmed with timeline and gates
  Evidence: approval chains described, decision timeline shared, stakeholder sign-off requirements

### Organizational (category)
- **change_readiness**: Is the buyer organization ready to adopt?
  0=Hostile/resistant to change, 1=Reluctant, 2=Willing, 3=Ready/eager
  Evidence: change management discussions, adoption concerns, organizational readiness signals

## Scoring Rules

- Score 0-3 based on the rubric above, or omit the dimension if no evidence
- A score of 0 means EVIDENCE OF WEAKNESS was found (not absence of evidence)
- Set confidence based on how clearly the signal was stated (0.0-1.0)
- Include the specific evidence text that supports the score

## Extraction Rules

- Extract ONLY what is explicitly stated or strongly implied in the transcript
- Do NOT fabricate or infer information not supported by the text
- If a MEDDIC dimension is not mentioned, leave it as null
- For ontology dimensions, only include those with actual evidence — omit the rest
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
{existing_dimensions_section}
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
- Score any ontology dimensions where the transcript provides evidence (include in ontology_dimensions list)
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

    # Build existing ontology dimensions section
    dim_lines = []
    for dim_id in [
        'champion_strength', 'economic_buyer_access', 'identified_pain',
        'metrics_business_case', 'decision_criteria_alignment', 'decision_process_clarity',
        'competitive_position', 'incumbent_displacement_risk',
        'pricing_alignment', 'procurement_legal_progress',
        'responsiveness', 'close_date_credibility',
        'technical_fit', 'integration_security_risk', 'change_readiness',
    ]:
        score = existing_deal.get(f'dim_{dim_id}')
        if score is not None:
            dim_lines.append(f'  {dim_id}: {score}/3')
    existing_dimensions_section = ''
    if dim_lines:
        existing_dimensions_section = (
            'Current Ontology Dimensions:\n' + '\n'.join(dim_lines)
        )

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
        existing_dimensions_section=existing_dimensions_section,
        additional_context=additional_context,
    )

    return [
        {'role': 'system', 'content': DEAL_EXTRACTION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
