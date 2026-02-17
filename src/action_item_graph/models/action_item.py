"""
ActionItem and ActionItemVersion models with temporal tracking.

ActionItem is the central hub node in our graph schema, connecting to:
- Account (parent relationship)
- Interaction (extracted from, many-to-many)
- Owner (assigned to)
- ActionItemVersion (version history)

Key features:
- Dual embeddings: original (immutable) + current (evolves with updates)
- Version tracking for temporal evolution
- Status lifecycle management
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ActionItemStatus(str, Enum):
    """Status lifecycle for action items."""

    OPEN = 'open'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'
    DEFERRED = 'deferred'


class ActionItem(BaseModel):
    """
    Core action item node in the knowledge graph.

    The ActionItem is the hub node that connects:
    - To Account via HAS_ACTION_ITEM relationship
    - To Interaction(s) via EXTRACTED_FROM relationship (grows over time)
    - To Owner via OWNED_BY relationship
    - To ActionItemVersion(s) via HAS_VERSION relationship
    - To other ActionItems via INVALIDATES relationship (for superseded items)

    Embeddings:
    - embedding: Original embedding (immutable, computed from first extraction)
    - embedding_current: Current state embedding (updated on significant changes)

    Both embeddings are used for deduplication matching:
    - embedding catches semantically similar new items
    - embedding_current catches status updates to existing items
    """

    # Identity
    id: UUID = Field(default_factory=uuid4, description='Unique identifier for this action item')

    # Multi-tenancy (present on ALL nodes)
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')
    account_id: str | None = Field(
        default=None, description='Account identifier for CRM context'
    )

    # Core content
    action_item_text: str = Field(
        ..., description='The verbatim or near-verbatim action item text from the transcript'
    )
    summary: str = Field(
        ..., description='Concise summary of what needs to be done (LLM-generated)'
    )
    owner: str = Field(
        ..., description='Person responsible for this action item (name or identifier)'
    )
    owner_type: str = Field(
        default='named',
        description='How owner was identified: "named", "role_inferred", or "unconfirmed"',
    )
    is_user_owned: bool = Field(
        default=False,
        description='Whether this action item belongs to the recording user',
    )
    conversation_context: str = Field(
        default='',
        description='Surrounding context from the conversation that clarifies the action item',
    )

    # Temporal tracking
    due_date: datetime | None = Field(
        default=None, description='When the action item is due (if extractable)'
    )
    status: ActionItemStatus = Field(
        default=ActionItemStatus.OPEN, description='Current status of the action item'
    )

    # Version tracking
    version: int = Field(default=1, description='Version number, incremented on updates')
    evolution_summary: str = Field(
        default='',
        description='LLM-generated summary of how this action item has evolved across interactions',
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=None), description='When the action item was first created'
    )
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=None), description='When the action item was last updated'
    )

    # Provenance
    source_interaction_id: UUID | None = Field(
        default=None, description='The interaction that first created this action item'
    )
    user_id: str | None = Field(
        default=None, description='User who initiated the interaction'
    )

    # Dual embeddings for matching
    embedding: list[float] | None = Field(
        default=None,
        description='Original embedding (immutable, from first extraction). '
        'Used to catch semantically similar new items.',
    )
    embedding_current: list[float] | None = Field(
        default=None,
        description='Current state embedding (updated on changes). '
        'Used to catch status updates to existing items.',
    )

    # Additional metadata
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description='Extraction confidence score (0.0 to 1.0)',
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict, description='Flexible attributes for extensions'
    )

    # Temporal validity (for graph edges)
    valid_at: datetime | None = Field(
        default=None, description='When this version became valid'
    )
    invalid_at: datetime | None = Field(
        default=None, description='When this version was superseded (if applicable)'
    )
    invalidated_by: UUID | None = Field(
        default=None, description='UUID of the action item that superseded this one'
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """
        Convert to Neo4j-compatible property dict.

        Note: Embeddings are stored as lists, which Neo4j handles natively for vector indexes.
        UUIDs are converted to strings for Neo4j compatibility.
        """
        props = {
            'action_item_id': str(self.id),
            'tenant_id': str(self.tenant_id),
            'account_id': self.account_id,
            'action_item_text': self.action_item_text,
            'summary': self.summary,
            'owner': self.owner,
            'owner_type': self.owner_type,
            'is_user_owned': self.is_user_owned,
            'conversation_context': self.conversation_context,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'status': self.status.value if isinstance(self.status, ActionItemStatus) else self.status,
            'version': self.version,
            'evolution_summary': self.evolution_summary,
            'created_at': self.created_at.isoformat(),
            'last_updated_at': self.last_updated_at.isoformat(),
            'source_interaction_id': str(self.source_interaction_id)
            if self.source_interaction_id
            else None,
            'user_id': self.user_id,
            'embedding': self.embedding,
            'embedding_current': self.embedding_current,
            'confidence': self.confidence,
            'valid_at': self.valid_at.isoformat() if self.valid_at else None,
            'invalid_at': self.invalid_at.isoformat() if self.invalid_at else None,
            'invalidated_by': str(self.invalidated_by) if self.invalidated_by else None,
        }
        # Filter out None values for cleaner Neo4j storage
        return {k: v for k, v in props.items() if v is not None}

    model_config = {'use_enum_values': True}


class ActionItemVersion(BaseModel):
    """
    Historical snapshot of an ActionItem at a specific point in time.

    Created whenever an ActionItem is updated, preserving the full state
    before the update. This enables:
    - Full audit trail of changes
    - Time-travel queries ("what was the status on date X?")
    - Understanding action item evolution across conversations
    """

    # Identity
    id: UUID = Field(default_factory=uuid4, description='Unique identifier for this version')
    action_item_id: UUID = Field(..., description='The parent ActionItem this version belongs to')

    # Multi-tenancy
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')

    # Version info
    version: int = Field(..., description='Version number (1, 2, 3, ...)')

    # Snapshot of ActionItem state at this version
    action_item_text: str = Field(..., description='Action item text at this version')
    summary: str = Field(..., description='Summary at this version')
    owner: str = Field(..., description='Owner at this version')
    status: ActionItemStatus = Field(..., description='Status at this version')
    due_date: datetime | None = Field(default=None, description='Due date at this version')

    # What changed
    change_summary: str = Field(
        default='', description='LLM-generated description of what changed in this version'
    )
    change_source_interaction_id: UUID | None = Field(
        default=None, description='The interaction that triggered this version'
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=None), description='When this version was created'
    )
    valid_from: datetime = Field(
        default_factory=lambda: datetime.now(tz=None), description='When this version became active'
    )
    valid_until: datetime | None = Field(
        default=None, description='When this version was superseded (None if current)'
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        props = {
            'version_id': str(self.id),
            'action_item_id': str(self.action_item_id),
            'tenant_id': str(self.tenant_id),
            'version': self.version,
            'action_item_text': self.action_item_text,
            'summary': self.summary,
            'owner': self.owner,
            'status': self.status.value if isinstance(self.status, ActionItemStatus) else self.status,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'change_summary': self.change_summary,
            'change_source_interaction_id': str(self.change_source_interaction_id)
            if self.change_source_interaction_id
            else None,
            'created_at': self.created_at.isoformat(),
            'valid_from': self.valid_from.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
        }
        return {k: v for k, v in props.items() if v is not None}

    model_config = {'use_enum_values': True}
