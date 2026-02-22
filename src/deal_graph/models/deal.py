"""
Deal, DealVersion, and MEDDICProfile models for the Deal Graph pipeline.

Deal is the central enrichment node, built on top of skeleton Deal nodes
created by eq-structured-graph-core. Properties are split into:
- Skeleton: tenant_id, opportunity_id, name, stage, amount (created by schema authority)
- Enrichment: meddic_*, embedding*, evolution_summary, version, etc. (added by our pipeline)

DealVersion captures bi-temporal snapshots before each update, enabling
full audit trails and time-travel queries.

Key design decisions:
- opportunity_id is a UUID (UUIDv7, serialized to string for Neo4j)
- Composite key: (tenant_id, opportunity_id) for multi-tenant uniqueness
- MEDDIC fields are flattened as string properties — no separate Person nodes in V1
- Dual embeddings: original (immutable) + current (evolves with updates)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DealStage(str, Enum):
    """Deal stage lifecycle. Regression is allowed (shadow forecast)."""

    PROSPECTING = 'prospecting'
    QUALIFICATION = 'qualification'
    PROPOSAL = 'proposal'
    NEGOTIATION = 'negotiation'
    CLOSED_WON = 'closed_won'
    CLOSED_LOST = 'closed_lost'


class MEDDICProfile(BaseModel):
    """
    MEDDIC qualification profile stored as flattened properties on Deal.

    Each dimension has a value field (str) and confidence field (float 0.0-1.0).
    Completeness is computed from how many of the 6 core dimensions are populated.
    """

    # Core MEDDIC dimensions
    metrics: str | None = Field(
        default=None, description='Quantifiable business impact (ROI, time savings, cost reduction)'
    )
    metrics_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Confidence in metrics extraction'
    )
    economic_buyer: str | None = Field(
        default=None,
        description='Person with final budget authority (e.g., "Sarah Jones, VP Engineering")',
    )
    economic_buyer_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Confidence in economic buyer identification'
    )
    decision_criteria: str | None = Field(
        default=None,
        description='Technical/business evaluation criteria (accumulates over time)',
    )
    decision_criteria_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Confidence in decision criteria extraction'
    )
    decision_process: str | None = Field(
        default=None,
        description='Steps/timeline to reach decision (accumulates over time)',
    )
    decision_process_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Confidence in decision process extraction'
    )
    identified_pain: str | None = Field(
        default=None,
        description='Core business problem driving the opportunity (accumulates over time)',
    )
    identified_pain_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Confidence in pain identification'
    )
    champion: str | None = Field(
        default=None,
        description='Internal advocate (e.g., "James Park, Director of Sales Ops")',
    )
    champion_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Confidence in champion identification'
    )

    # Future MEDDPICC extension slots
    paper_process: str | None = Field(
        default=None, description='MEDDPICC: procurement/legal process'
    )
    competition: str | None = Field(
        default=None, description='MEDDPICC: competitive landscape'
    )

    @property
    def completeness_score(self) -> float:
        """Fraction of the 6 core MEDDIC dimensions that are populated (0.0-1.0)."""
        fields = [
            self.metrics,
            self.economic_buyer,
            self.decision_criteria,
            self.decision_process,
            self.identified_pain,
            self.champion,
        ]
        populated = sum(1 for f in fields if f)
        return populated / 6

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Flatten to Neo4j properties with meddic_ prefix."""
        props: dict[str, Any] = {
            'meddic_metrics': self.metrics,
            'meddic_metrics_confidence': self.metrics_confidence,
            'meddic_economic_buyer': self.economic_buyer,
            'meddic_economic_buyer_confidence': self.economic_buyer_confidence,
            'meddic_decision_criteria': self.decision_criteria,
            'meddic_decision_criteria_confidence': self.decision_criteria_confidence,
            'meddic_decision_process': self.decision_process,
            'meddic_decision_process_confidence': self.decision_process_confidence,
            'meddic_identified_pain': self.identified_pain,
            'meddic_identified_pain_confidence': self.identified_pain_confidence,
            'meddic_champion': self.champion,
            'meddic_champion_confidence': self.champion_confidence,
            'meddic_completeness': self.completeness_score,
        }
        # Include MEDDPICC extensions only if populated
        if self.paper_process is not None:
            props['meddic_paper_process'] = self.paper_process
        if self.competition is not None:
            props['meddic_competition'] = self.competition
        return props


class OntologyScores(BaseModel):
    """
    Ontology dimension scores stored alongside MEDDIC on Deal nodes.

    Stores the scored (0-3) output for each ontology dimension beyond
    the core MEDDIC fields. MEDDIC dimensions are kept in MEDDICProfile
    for backward compatibility; this captures the expanded set.

    Scores are stored in dicts keyed by dimension_id. A score of None
    means "insufficient evidence" (distinct from 0 = "evidence of weakness").
    """

    scores: dict[str, int | None] = Field(
        default_factory=dict,
        description='dimension_id → 0-3 score or None (insufficient evidence)',
    )
    confidences: dict[str, float] = Field(
        default_factory=dict,
        description='dimension_id → confidence in score (0.0-1.0)',
    )
    evidence: dict[str, str | None] = Field(
        default_factory=dict,
        description='dimension_id → evidence text from transcript',
    )

    @property
    def completeness_score(self) -> float:
        """Fraction of dimensions that have a non-null score."""
        if not self.scores:
            return 0.0
        scored = sum(1 for v in self.scores.values() if v is not None)
        return scored / len(self.scores)

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Flatten to Neo4j properties with dim_ prefix."""
        props: dict[str, Any] = {}
        for dim_id, score in self.scores.items():
            props[f'dim_{dim_id}'] = score
            props[f'dim_{dim_id}_confidence'] = self.confidences.get(dim_id, 0.0)
        props['ontology_completeness'] = self.completeness_score
        return props


class Deal(BaseModel):
    """
    Deal node in the knowledge graph.

    Combines skeleton properties (created by eq-structured-graph-core) with
    enrichment properties (added by our MEDDIC pipeline). The composite key
    is (tenant_id, opportunity_id).

    Skeleton properties are replicated here for codebase isolation — we do
    not import from eq-structured-graph-core.
    """

    # --- Skeleton properties (created by eq-structured-graph-core) ---
    tenant_id: UUID = Field(..., description='Tenant/organization UUID (composite key part)')
    opportunity_id: UUID = Field(
        ..., description='Deal primary key (UUIDv7, composite with tenant_id)'
    )
    deal_ref: str | None = Field(default=None, description='Display-only alias derived from last 16 hex chars of opportunity_id (e.g. deal_e302ad7806fe1234)')
    name: str = Field(default='', description='Descriptive opportunity name')
    stage: DealStage = Field(
        default=DealStage.PROSPECTING, description='Current deal stage'
    )
    amount: float | None = Field(default=None, description='Estimated or confirmed deal value')
    account_id: str | None = Field(default=None, description='Account-level scoping')

    # --- Enrichment properties (added by our pipeline) ---
    currency: str = Field(default='USD', description='Currency code')

    # MEDDIC qualification profile
    meddic: MEDDICProfile = Field(
        default_factory=MEDDICProfile, description='MEDDIC qualification data'
    )

    # Extended ontology dimension scores (beyond MEDDIC)
    ontology_scores: OntologyScores = Field(
        default_factory=OntologyScores,
        description='Ontology dimension scores (0-3) keyed by dimension_id',
    )
    ontology_version: str | None = Field(
        default=None, description='Hash of ontology config used for extraction'
    )

    # Summaries
    opportunity_summary: str = Field(
        default='', description='LLM-generated summary, evolves over time'
    )
    evolution_summary: str = Field(
        default='',
        description='Cumulative narrative of how and why the deal evolved',
    )

    # Dual embeddings for entity resolution
    embedding: list[float] | None = Field(
        default=None,
        description='Immutable original embedding (1536-dim, from first extraction)',
    )
    embedding_current: list[float] | None = Field(
        default=None,
        description='Mutable embedding, updated on significant summary changes',
    )

    # Version tracking
    version: int = Field(default=1, description='Incremented on each update')

    # Extraction confidence
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description='Extraction confidence (0.0-1.0)'
    )

    # Provenance
    source_interaction_id: UUID | None = Field(
        default=None, description='Interaction that first created this deal'
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description='When first created',
    )
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description='Last modification timestamp',
    )
    expected_close_date: datetime | None = Field(
        default=None, description='Projected close date'
    )
    closed_at: datetime | None = Field(
        default=None, description='Actual close date'
    )

    # Future slots
    forecast_score: float | None = Field(
        default=None, description='Probabilistic close probability'
    )
    qualification_status: str | None = Field(
        default=None, description='qualified / unqualified'
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """
        Convert to Neo4j-compatible property dict.

        Includes both skeleton and enrichment properties. MEDDIC fields are
        flattened with meddic_ prefix. UUIDs become strings, datetimes become
        ISO 8601 strings. None values are filtered out.
        """
        props: dict[str, Any] = {
            # Skeleton properties
            'tenant_id': str(self.tenant_id),
            'opportunity_id': str(self.opportunity_id),
            'deal_ref': self.deal_ref,
            'name': self.name,
            'stage': self.stage.value if isinstance(self.stage, DealStage) else self.stage,
            'amount': self.amount,
            'account_id': self.account_id,
            # Enrichment properties
            'currency': self.currency,
            'opportunity_summary': self.opportunity_summary,
            'evolution_summary': self.evolution_summary,
            'embedding': self.embedding,
            'embedding_current': self.embedding_current,
            'version': self.version,
            'confidence': self.confidence,
            'source_interaction_id': (
                str(self.source_interaction_id) if self.source_interaction_id else None
            ),
            'created_at': self.created_at.isoformat(),
            'last_updated_at': self.last_updated_at.isoformat(),
            'expected_close_date': (
                self.expected_close_date.isoformat() if self.expected_close_date else None
            ),
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'forecast_score': self.forecast_score,
            'qualification_status': self.qualification_status,
        }
        # Flatten MEDDIC properties
        props.update(self.meddic.to_neo4j_properties())
        # Flatten ontology dimension scores
        props.update(self.ontology_scores.to_neo4j_properties())
        if self.ontology_version:
            props['ontology_version'] = self.ontology_version
        # Filter out None values for cleaner Neo4j storage
        return {k: v for k, v in props.items() if v is not None}

    model_config = {'use_enum_values': True}


class DealVersion(BaseModel):
    """
    Historical snapshot of a Deal at a specific version.

    Created before every Deal update, preserving the full state. Enables:
    - Full audit trail of changes (which fields changed and why)
    - Time-travel queries via bi-temporal valid_from/valid_until
    - MEDDIC progress timeline reconstruction

    Linked to parent Deal via [:HAS_VERSION] relationship.
    Composite key: (tenant_id, version_id).
    """

    # Identity
    version_id: UUID = Field(
        default_factory=uuid4,
        description='Primary key (composite with tenant_id)',
    )
    deal_opportunity_id: UUID = Field(
        ..., description="Parent Deal's opportunity_id (UUIDv7)"
    )
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')

    # Version info
    version: int = Field(..., description='Version number at snapshot time')

    # Snapshot of Deal state
    name: str = Field(..., description='Snapshot of deal name')
    stage: DealStage = Field(..., description='Snapshot of stage')
    amount: float | None = Field(default=None, description='Snapshot of amount')
    opportunity_summary: str = Field(default='', description='Snapshot of summary')
    evolution_summary: str = Field(
        default='', description='Snapshot of cumulative evolution narrative'
    )

    # MEDDIC snapshots (flattened, not a MEDDICProfile object — simpler for Neo4j)
    meddic_metrics: str | None = Field(default=None, description='Snapshot of MEDDIC metrics')
    meddic_economic_buyer: str | None = Field(
        default=None, description='Snapshot of economic buyer'
    )
    meddic_decision_criteria: str | None = Field(
        default=None, description='Snapshot of decision criteria'
    )
    meddic_decision_process: str | None = Field(
        default=None, description='Snapshot of decision process'
    )
    meddic_identified_pain: str | None = Field(
        default=None, description='Snapshot of identified pain'
    )
    meddic_champion: str | None = Field(default=None, description='Snapshot of champion')
    meddic_completeness: float | None = Field(
        default=None, description='Snapshot of completeness score'
    )

    # Ontology dimension score snapshots (JSON-serialized dicts)
    ontology_scores_json: str | None = Field(
        default=None, description='JSON snapshot of dimension_id → score dict'
    )
    ontology_completeness: float | None = Field(
        default=None, description='Snapshot of ontology completeness score'
    )
    ontology_version: str | None = Field(
        default=None, description='Ontology config version hash at snapshot time'
    )

    # Change tracking
    change_summary: str = Field(
        default='',
        description='LLM-generated narrative explaining why the deal changed',
    )
    changed_fields: list[str] = Field(
        default_factory=list,
        description='Machine-readable list of property names that changed',
    )
    change_source_interaction_id: UUID | None = Field(
        default=None, description='Which interaction triggered the change'
    )

    # Timestamps and bi-temporal validity
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description='When this snapshot was taken',
    )
    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description='Start of validity window',
    )
    valid_until: datetime | None = Field(
        default=None, description='End of validity window (None if current)'
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        props: dict[str, Any] = {
            'version_id': str(self.version_id),
            'deal_opportunity_id': str(self.deal_opportunity_id),
            'tenant_id': str(self.tenant_id),
            'version': self.version,
            'name': self.name,
            'stage': self.stage.value if isinstance(self.stage, DealStage) else self.stage,
            'amount': self.amount,
            'opportunity_summary': self.opportunity_summary,
            'evolution_summary': self.evolution_summary,
            'meddic_metrics': self.meddic_metrics,
            'meddic_economic_buyer': self.meddic_economic_buyer,
            'meddic_decision_criteria': self.meddic_decision_criteria,
            'meddic_decision_process': self.meddic_decision_process,
            'meddic_identified_pain': self.meddic_identified_pain,
            'meddic_champion': self.meddic_champion,
            'meddic_completeness': self.meddic_completeness,
            'change_summary': self.change_summary,
            'changed_fields': self.changed_fields,
            'change_source_interaction_id': (
                str(self.change_source_interaction_id)
                if self.change_source_interaction_id
                else None
            ),
            'created_at': self.created_at.isoformat(),
            'valid_from': self.valid_from.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
        }
        return {k: v for k, v in props.items() if v is not None}

    model_config = {'use_enum_values': True}
