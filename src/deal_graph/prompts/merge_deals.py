"""
Deal merge synthesis prompts.

Used during Phase 5 when an extracted deal matches an existing Deal node.
The LLM synthesizes the merged state by applying MEDDIC merge rules:
- Additive fields accumulate (decision_criteria, decision_process, identified_pain)
- Replaceable fields supersede (economic_buyer, champion, amount, stage)
- Evolution narrative explains WHY, not just WHAT changed

Response model: MergedDeal (defined in deal_graph.models.extraction).
"""

from typing import Any

from ..models.extraction import ExtractedDeal


# =============================================================================
# System Prompt
# =============================================================================

MERGE_SYNTHESIS_SYSTEM_PROMPT = """You are an expert sales analyst synthesizing deal updates from call transcripts.

Given an EXISTING DEAL and a NEW EXTRACTION from a recent transcript, merge them into an updated deal state.

## MEDDIC Merge Rules

### Additive Fields (accumulate over time)
These fields naturally grow as new information surfaces. Synthesize a COMBINED value:
- **decision_criteria** — new criteria ADD to existing (e.g., "SOC2 required" + "Must integrate with existing ERP")
- **decision_process** — new steps/stakeholders discovered (e.g., "Board review in Q2" + "Legal review added before board")
- **identified_pain** — additional pain points surface over time

When both existing and new have content, write a combined description that integrates both. Do NOT simply concatenate — synthesize into a coherent narrative.

### Replaceable Fields (new supersedes old)
These represent a single entity/value; the latest information takes precedence:
- **economic_buyer** — changes when authority shifts (e.g., VP of Sales → CFO)
- **champion** — changes when advocate shifts
- **amount** — replaces when revised
- **stage** — replaces (regression is allowed for shadow forecasting)

### Rule: Never Lose Information
If the existing deal has a MEDDIC field populated and the new extraction does NOT mention it, **keep the existing value**. Output null for that field to signal "no change". Only output a value when the transcript provides new or corrected information.

### Confidence Ratcheting
If a MEDDIC field was previously low-confidence and the new transcript explicitly confirms it, the field should be updated with the stronger evidence.

## Narrative Requirements

### evolution_summary (cumulative — on Deal node)
Append to the existing evolution_summary. This is the FULL STORY of the deal's progression across all interactions. Explain the **business context** driving each change. Connect changes to what was said in the transcript and why it matters for the deal's trajectory.

Example:
> "Initial discovery call revealed data silo pain points (Jan 15). Sarah Jones from Engineering emerged as champion during technical deep-dive (Jan 22). Budget expanded from $200K to $500K after CEO saw demo and recognized enterprise-wide applicability (Jan 29)."

### change_narrative (per-update — becomes DealVersion.change_summary)
Describe ONLY what changed in THIS specific interaction and WHY. This is NOT the cumulative history — it is the delta.

Example:
> "Budget expanded from $200K to $500K after CEO attended the demo and recognized the platform's applicability beyond the initial Sales Ops scope. Deal advanced from qualification to proposal."

## Stage Assessment

Assess whether the deal stage should change:
- Progression (qualification → proposal) requires clear signals (pricing discussed, proposal sent)
- Regression (proposal → qualification) is ALLOWED when signals indicate slippage (new stakeholder objections, timeline pushed)
- Provide stage_reasoning explaining your assessment

## Embedding Decision

Set should_update_embedding=true ONLY when the opportunity_summary changed substantially (scope redefined, fundamentally different deal identity). Minor MEDDIC updates or stage changes do NOT warrant re-embedding.

## Ontology Dimension Scores

In addition to MEDDIC text fields, update ontology dimension scores (0-3 scale) via the `ontology_dimensions` list.
Include a DimensionExtraction for each dimension where the new transcript provides evidence.
Omit dimensions where the transcript adds no new information (existing scores are preserved).

Dimension merge rules by accumulation type:
- **Additive** (identified_pain, metrics_business_case, decision_criteria_alignment, decision_process_clarity, technical_fit): new evidence enriches existing — score may increase
- **Point-in-time** (champion_strength, economic_buyer_access, competitive_position, incumbent_displacement_risk, pricing_alignment, responsiveness, close_date_credibility, integration_security_risk, change_readiness): latest evidence supersedes — score reflects current state
- **Monotonic** (procurement_legal_progress): should only increase; if evidence suggests regression, flag in change_narrative

## Output Rules

- For each MEDDIC field: output the updated value if the transcript provides new info, or null to keep existing
- For ontology_dimensions: include only dimensions with updated scores from the new transcript
- changed_fields: list EVERY property name that was modified (e.g., ["meddic_champion", "stage", "amount", "dim_competitive_position"])
- evolution_summary: MUST include the existing narrative PLUS the new update
- change_narrative: MUST explain business context, not just list field diffs"""


# =============================================================================
# User Prompt Template
# =============================================================================

MERGE_SYNTHESIS_USER_PROMPT_TEMPLATE = """Synthesize an update for this deal by merging the new extraction with the existing state.

<existing_deal>
Name: {existing_name}
Stage: {existing_stage}
Amount: {existing_amount}
Summary: {existing_summary}
Evolution History: {existing_evolution_summary}
MEDDIC:
  Metrics: {existing_metrics}
  Economic Buyer: {existing_economic_buyer}
  Decision Criteria: {existing_decision_criteria}
  Decision Process: {existing_decision_process}
  Identified Pain: {existing_identified_pain}
  Champion: {existing_champion}
{existing_dimensions_section}
</existing_deal>

<new_extraction>
Name: {extracted_name}
Stage Assessment: {extracted_stage}
Estimated Amount: {extracted_amount}
Summary: {extracted_summary}
MEDDIC:
  Metrics: {extracted_metrics}
  Economic Buyer: {extracted_economic_buyer}
  Decision Criteria: {extracted_decision_criteria}
  Decision Process: {extracted_decision_process}
  Identified Pain: {extracted_identified_pain}
  Champion: {extracted_champion}
{extracted_dimensions_section}
Confidence: {extracted_confidence}
Reasoning: {extracted_reasoning}
</new_extraction>

How should these be merged? Apply the merge rules (additive vs replaceable) for both MEDDIC text fields and ontology dimension scores, and generate the evolution narrative."""


# =============================================================================
# Builder Function
# =============================================================================


def build_deal_merge_prompt(
    existing_deal: dict[str, Any],
    extracted_deal: ExtractedDeal,
) -> list[dict[str, str]]:
    """
    Build merge synthesis prompt messages for OpenAI.

    Args:
        existing_deal: Current Deal node properties from the graph
        extracted_deal: Newly extracted deal from transcript

    Returns:
        List of message dicts for OpenAI chat completion
    """

    def _get(key: str, default: str = 'Not identified') -> str:
        val = existing_deal.get(key)
        return str(val) if val else default

    # Build existing ontology dimensions section
    dim_ids = [
        'champion_strength', 'economic_buyer_access', 'identified_pain',
        'metrics_business_case', 'decision_criteria_alignment', 'decision_process_clarity',
        'competitive_position', 'incumbent_displacement_risk',
        'pricing_alignment', 'procurement_legal_progress',
        'responsiveness', 'close_date_credibility',
        'technical_fit', 'integration_security_risk', 'change_readiness',
    ]
    existing_dim_lines = []
    for dim_id in dim_ids:
        score = existing_deal.get(f'dim_{dim_id}')
        if score is not None:
            existing_dim_lines.append(f'  {dim_id}: {score}/3')
    existing_dimensions_section = ''
    if existing_dim_lines:
        existing_dimensions_section = (
            'Ontology Dimensions:\n' + '\n'.join(existing_dim_lines)
        )

    # Build extracted dimensions section
    extracted_dim_lines = []
    for dim in extracted_deal.ontology_dimensions:
        if dim.score is not None:
            evidence_hint = f' — "{dim.evidence[:80]}..."' if dim.evidence else ''
            extracted_dim_lines.append(
                f'  {dim.dimension_id}: {dim.score}/3 (conf: {dim.confidence:.2f}){evidence_hint}'
            )
    extracted_dimensions_section = ''
    if extracted_dim_lines:
        extracted_dimensions_section = (
            'Ontology Dimensions:\n' + '\n'.join(extracted_dim_lines)
        )

    user_prompt = MERGE_SYNTHESIS_USER_PROMPT_TEMPLATE.format(
        # Existing deal fields
        existing_name=_get('name', 'Unnamed Deal'),
        existing_stage=_get('stage', 'Unknown'),
        existing_amount=_get('amount', 'Not specified'),
        existing_summary=_get('opportunity_summary', 'No summary available'),
        existing_evolution_summary=_get('evolution_summary', 'No prior history'),
        existing_metrics=_get('meddic_metrics'),
        existing_economic_buyer=_get('meddic_economic_buyer'),
        existing_decision_criteria=_get('meddic_decision_criteria'),
        existing_decision_process=_get('meddic_decision_process'),
        existing_identified_pain=_get('meddic_identified_pain'),
        existing_champion=_get('meddic_champion'),
        existing_dimensions_section=existing_dimensions_section,
        # Extracted deal fields
        extracted_name=extracted_deal.opportunity_name,
        extracted_stage=extracted_deal.stage_assessment,
        extracted_amount=extracted_deal.estimated_amount or 'Not specified',
        extracted_summary=extracted_deal.opportunity_summary,
        extracted_metrics=extracted_deal.metrics or 'Not identified',
        extracted_economic_buyer=extracted_deal.economic_buyer or 'Not identified',
        extracted_decision_criteria=extracted_deal.decision_criteria or 'Not identified',
        extracted_decision_process=extracted_deal.decision_process or 'Not identified',
        extracted_identified_pain=extracted_deal.identified_pain or 'Not identified',
        extracted_champion=extracted_deal.champion or 'Not identified',
        extracted_dimensions_section=extracted_dimensions_section,
        extracted_confidence=f'{extracted_deal.confidence:.2f}',
        extracted_reasoning=extracted_deal.reasoning,
    )

    return [
        {'role': 'system', 'content': MERGE_SYNTHESIS_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
