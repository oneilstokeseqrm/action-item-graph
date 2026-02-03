"""
LLM structured output models for Deal extraction, deduplication, and merge.

These models are used as Pydantic response_format targets for OpenAI
structured output calls. Each model corresponds to a specific pipeline stage:
- ExtractedDeal / DealExtractionResult: MEDDIC extraction from transcripts
- DealDeduplicationDecision: Entity resolution (is this the same deal?)
- MergedDeal: Synthesis of existing deal state + new extraction data
"""

from pydantic import BaseModel, Field


class ExtractedDeal(BaseModel):
    """
    A single deal extracted from a transcript via MEDDIC analysis.

    Output by the extraction LLM for each opportunity identified.
    May represent a new deal or an update to an existing one.
    """

    opportunity_name: str = Field(
        ..., description='Descriptive name for the opportunity'
    )
    opportunity_summary: str = Field(
        ..., description='2-3 sentence overview of the deal'
    )
    stage_assessment: str = Field(
        ..., description="LLM's assessment of current deal stage"
    )

    # MEDDIC dimensions (each None if not mentioned in transcript)
    metrics: str | None = Field(
        default=None,
        description='Quantifiable business impact (ROI, time savings, cost reduction)',
    )
    economic_buyer: str | None = Field(
        default=None,
        description='Person with final budget authority',
    )
    decision_criteria: str | None = Field(
        default=None,
        description='Technical/business evaluation criteria',
    )
    decision_process: str | None = Field(
        default=None,
        description='Steps/timeline to reach decision',
    )
    identified_pain: str | None = Field(
        default=None,
        description='Core business problem driving the opportunity',
    )
    champion: str | None = Field(
        default=None,
        description='Internal advocate for the deal',
    )

    # Deal metadata
    estimated_amount: float | None = Field(
        default=None, description='Estimated deal value'
    )
    currency: str = Field(default='USD', description='Currency code')
    expected_close_timeframe: str | None = Field(
        default=None, description='Freetext close timeframe (e.g., "Q2 2026")'
    )

    # Quality signals
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description='Extraction confidence (0.0-1.0)'
    )
    reasoning: str = Field(
        ..., description='Why this is a deal and what signals were present'
    )


class DealExtractionResult(BaseModel):
    """
    Wrapper for LLM extraction output, containing zero or more deals.

    Used as the structured output target for both discovery mode (Case B)
    and targeted mode (Case A, max 1 deal).
    """

    deals: list[ExtractedDeal] = Field(
        default_factory=list, description='Extracted deals (may be empty)'
    )
    has_deals: bool = Field(
        ..., description='Whether any deals were found in the transcript'
    )
    extraction_notes: str | None = Field(
        default=None,
        description='Optional notes about the extraction (e.g., ambiguous signals)',
    )


class DealDeduplicationDecision(BaseModel):
    """
    LLM decision on whether two deals are the same entity.

    Used during entity resolution when similarity is in the borderline
    zone (0.70-0.90). Bias toward create_new when uncertain.
    """

    is_same_deal: bool = Field(
        ..., description='Whether the two deals are the same opportunity'
    )
    recommendation: str = Field(
        ..., description="'merge' or 'create_new'"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description='Decision confidence (0.0-1.0)'
    )
    reasoning: str = Field(
        ..., description='Explanation of why this decision was made'
    )


class MergedDeal(BaseModel):
    """
    LLM synthesis output when merging new extraction data with an existing Deal.

    Implements the MEDDIC merge rules:
    - Additive fields (decision_criteria, decision_process, identified_pain) accumulate
    - Replaceable fields (economic_buyer, champion, amount, stage) supersede
    - Evolution narrative explains WHY, not just WHAT changed
    """

    # Summaries
    opportunity_summary: str = Field(
        ..., description='Updated deal summary'
    )
    evolution_summary: str = Field(
        ..., description='Cumulative narrative of WHY the deal evolved'
    )
    change_narrative: str = Field(
        ..., description='This-update-only narrative (becomes DealVersion.change_summary)'
    )
    changed_fields: list[str] = Field(
        ..., description='Machine-readable list of property names that changed'
    )

    # Updated MEDDIC fields (None = keep existing value, populated = replace/extend)
    metrics: str | None = Field(default=None, description='Updated metrics')
    economic_buyer: str | None = Field(default=None, description='Updated economic buyer')
    decision_criteria: str | None = Field(default=None, description='Updated decision criteria')
    decision_process: str | None = Field(default=None, description='Updated decision process')
    identified_pain: str | None = Field(default=None, description='Updated identified pain')
    champion: str | None = Field(default=None, description='Updated champion')

    # Stage assessment
    implied_stage: str | None = Field(
        default=None, description='Stage assessment (may progress or regress)'
    )
    stage_reasoning: str | None = Field(
        default=None, description='Why stage changed (or stayed the same)'
    )

    # Deal metadata
    amount: float | None = Field(default=None, description='Updated deal value')
    expected_close_date_text: str | None = Field(
        default=None, description='Updated close timeframe (freetext)'
    )

    # Embedding decision
    should_update_embedding: bool = Field(
        ...,
        description='Whether the summary changed enough to warrant re-embedding',
    )
