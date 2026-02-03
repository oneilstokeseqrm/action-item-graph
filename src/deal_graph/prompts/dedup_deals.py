"""
Deal deduplication prompts for entity resolution.

Used during the matching phase (Phase 4) when a candidate match falls in the
borderline similarity zone (0.70-0.90). The LLM compares an extracted deal
with an existing Deal node and decides whether they represent the same
business opportunity.

Response model: DealDeduplicationDecision (defined in deal_graph.models.extraction).
"""

from typing import Any

from ..models.extraction import ExtractedDeal


# =============================================================================
# System Prompt
# =============================================================================

DEAL_DEDUPLICATION_SYSTEM_PROMPT = """You are an expert at determining whether two deal opportunities refer to the same real-world business opportunity.

Given an EXISTING DEAL from the deal graph and a NEWLY EXTRACTED deal from a transcript, determine if they represent the SAME opportunity.

## Same Deal Indicators

They are the SAME DEAL if:
- They describe the same product/service being sold to the same buyer organization
- The new extraction is an update on progress for the existing deal
- They have the same pain points and stakeholders, described at different points in time
- The deal names are similar and involve the same account/parties
- Dollar amounts are in the same range (allowing for negotiation movement)
- The economic buyer or champion is the same person

## Different Deal Indicators

They are DIFFERENT DEALS if:
- They involve different products, service lines, or solution categories
- They target different buyer organizations or distinct divisions/departments
- They have fundamentally different price ranges suggesting different scopes (e.g., $50K vs $500K)
- They address different business problems even if the same parties are involved
- They are at incompatible stages (e.g., one is closed_won, other is prospecting for a new need)

## Critical Rule — Bias Toward create_new

A false merge (incorrectly combining two different deals) is FAR WORSE than a false split (creating a duplicate that can be merged later). When uncertain, always recommend 'create_new'.

## Recommendation Options

- **merge**: The deals are the same opportunity — the new extraction should update the existing deal
- **create_new**: These are genuinely different opportunities — create a separate deal"""


# =============================================================================
# User Prompt Template
# =============================================================================

DEAL_DEDUPLICATION_USER_PROMPT_TEMPLATE = """Compare these deals and determine if they represent the same business opportunity.

<existing_deal>
Name: {existing_name}
Stage: {existing_stage}
Amount: {existing_amount}
Summary: {existing_summary}
MEDDIC:
  Metrics: {existing_metrics}
  Economic Buyer: {existing_economic_buyer}
  Decision Criteria: {existing_decision_criteria}
  Decision Process: {existing_decision_process}
  Identified Pain: {existing_identified_pain}
  Champion: {existing_champion}
</existing_deal>

<extracted_deal>
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
Confidence: {extracted_confidence}
Reasoning: {extracted_reasoning}
</extracted_deal>

<similarity_score>{similarity_score:.3f}</similarity_score>

Are these the same business opportunity?"""


# =============================================================================
# Builder Function
# =============================================================================


def build_deal_deduplication_prompt(
    existing_deal: dict[str, Any],
    extracted_deal: ExtractedDeal,
    similarity_score: float,
) -> list[dict[str, str]]:
    """
    Build deduplication prompt messages for OpenAI.

    Args:
        existing_deal: Properties of existing Deal node from the graph
        extracted_deal: Newly extracted deal from transcript
        similarity_score: Cosine similarity between embeddings

    Returns:
        List of message dicts for OpenAI chat completion
    """

    def _get(key: str, default: str = 'Not identified') -> str:
        val = existing_deal.get(key)
        return str(val) if val else default

    user_prompt = DEAL_DEDUPLICATION_USER_PROMPT_TEMPLATE.format(
        # Existing deal fields
        existing_name=_get('name', 'Unnamed Deal'),
        existing_stage=_get('stage', 'Unknown'),
        existing_amount=_get('amount', 'Not specified'),
        existing_summary=_get('opportunity_summary', 'No summary available'),
        existing_metrics=_get('meddic_metrics'),
        existing_economic_buyer=_get('meddic_economic_buyer'),
        existing_decision_criteria=_get('meddic_decision_criteria'),
        existing_decision_process=_get('meddic_decision_process'),
        existing_identified_pain=_get('meddic_identified_pain'),
        existing_champion=_get('meddic_champion'),
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
        extracted_confidence=f'{extracted_deal.confidence:.2f}',
        extracted_reasoning=extracted_deal.reasoning,
        # Similarity
        similarity_score=similarity_score,
    )

    return [
        {'role': 'system', 'content': DEAL_DEDUPLICATION_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
