"""
Entity models for the Action Item Graph.

Graph hierarchy:
- Account: Root node, represents a company/organization in CRM
- Interaction: Calls, meetings, notes - contains transcript text
- Owner: Person responsible for action items (may or may not be a Contact)
- Contact: Person who participated in an Interaction
- Deal: Sales opportunity associated with an Account

All entities include tenant_id for multi-tenancy isolation.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Account(BaseModel):
    """
    Root node in the graph representing a company/organization.

    All ActionItems and Interactions belong to an Account.
    This is the primary aggregation point for CRM data.
    """

    account_id: str = Field(..., description='Account identifier (e.g., acct_acme_corp_001)')
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')

    # Account details
    name: str = Field(..., description='Company/organization name')
    domain: str | None = Field(default=None, description='Primary domain (e.g., acme.com)')

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))
    last_interaction_at: datetime | None = Field(
        default=None, description='When the last interaction occurred'
    )

    # Metadata
    attributes: dict[str, Any] = Field(
        default_factory=dict, description='Flexible attributes (industry, size, etc.)'
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        props = {
            'account_id': self.account_id,
            'tenant_id': str(self.tenant_id),
            'name': self.name,
            'domain': self.domain,
            'created_at': self.created_at.isoformat(),
            'last_interaction_at': self.last_interaction_at.isoformat()
            if self.last_interaction_at
            else None,
        }
        return {k: v for k, v in props.items() if v is not None}


class InteractionType(str, Enum):
    """Type of interaction."""

    TRANSCRIPT = 'transcript'
    NOTE = 'note'
    DOCUMENT = 'document'
    EMAIL = 'email'
    MEETING = 'meeting'


class Interaction(BaseModel):
    """
    Represents a conversation, meeting, or other interaction.

    Interactions contain the source text from which ActionItems are extracted.
    They connect to:
    - Account (via HAS_INTERACTION)
    - Contact (via PARTICIPATED_IN, incoming from Contact)
    - ActionItem (via EXTRACTED_FROM, incoming from ActionItem)
    """

    interaction_id: UUID = Field(default_factory=uuid4, description='Unique interaction identifier')
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')
    account_id: str | None = Field(default=None, description='Parent account identifier')

    # Content
    interaction_type: InteractionType = Field(..., description='Type of interaction')
    title: str | None = Field(default=None, description='Meeting title or subject')
    content_text: str = Field(..., description='Full text content of the interaction')

    # Timing
    timestamp: datetime = Field(..., description='When the interaction took place')
    duration_seconds: int | None = Field(default=None, description='Duration in seconds')

    # Provenance
    source: str | None = Field(
        default=None, description='Origin of content (web-mic, upload, api, import)'
    )
    user_id: str | None = Field(default=None, description='User who created/uploaded this (Auth0 sub)')
    pg_user_id: UUID | None = Field(
        default=None, description='Postgres user UUID from identity bridge'
    )

    # Processing state
    processed_at: datetime | None = Field(
        default=None, description='When action items were extracted'
    )
    action_item_count: int = Field(
        default=0, description='Number of action items extracted from this interaction'
    )

    # Metadata
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description='Additional metadata (opportunity_id, contact_ids, etc.)',
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        # Handle interaction_type whether it's an enum or already a string
        interaction_type_value = (
            self.interaction_type.value
            if isinstance(self.interaction_type, InteractionType)
            else self.interaction_type
        )
        props = {
            'interaction_id': str(self.interaction_id),
            'tenant_id': str(self.tenant_id),
            'account_id': self.account_id,
            'interaction_type': interaction_type_value,
            'title': self.title,
            'content_text': self.content_text,
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': self.duration_seconds,
            'source': self.source,
            'user_id': self.user_id,
            'pg_user_id': str(self.pg_user_id) if self.pg_user_id else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'action_item_count': self.action_item_count,
        }
        return {k: v for k, v in props.items() if v is not None}

    model_config = {'use_enum_values': True}


class Owner(BaseModel):
    """
    Represents a person who owns action items.

    Owners are resolved from extracted owner names/references.
    They may or may not correspond to Contacts in the system.
    Name canonicalization helps deduplicate (e.g., "John", "John Smith" -> same Owner).
    """

    id: UUID = Field(default_factory=uuid4, description='Unique owner identifier')
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')

    # Identity
    canonical_name: str = Field(
        ..., description='Resolved canonical name for this owner'
    )
    aliases: list[str] = Field(
        default_factory=list,
        description='Alternative names/references that resolve to this owner',
    )

    # Optional linking
    contact_id: str | None = Field(
        default=None, description='Linked Contact ID if this owner is a known contact'
    )
    user_id: str | None = Field(
        default=None, description='Linked User ID if this owner is an internal user'
    )

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        return {
            'owner_id': str(self.id),
            'tenant_id': str(self.tenant_id),
            'canonical_name': self.canonical_name,
            'aliases': self.aliases,
            'contact_id': self.contact_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
        }


class Contact(BaseModel):
    """
    Represents a person who participated in an Interaction.

    Contacts connect to Interactions via PARTICIPATED_IN relationship.
    They do NOT connect directly to ActionItems (that's the Owner's role).
    """

    id: str = Field(..., description='Contact identifier (e.g., contact_sarah_001)')
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')
    account_id: str | None = Field(default=None, description='Associated account')

    # Identity
    name: str = Field(..., description='Contact name')
    email: str | None = Field(default=None, description='Email address')
    title: str | None = Field(default=None, description='Job title')

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))
    attributes: dict[str, Any] = Field(
        default_factory=dict, description='Flexible attributes'
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        props = {
            'contact_id': self.id,
            'tenant_id': str(self.tenant_id),
            'account_id': self.account_id,
            'name': self.name,
            'email': self.email,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
        }
        return {k: v for k, v in props.items() if v is not None}


class DealStage(str, Enum):
    """Common deal/opportunity stages."""

    PROSPECTING = 'prospecting'
    QUALIFICATION = 'qualification'
    PROPOSAL = 'proposal'
    NEGOTIATION = 'negotiation'
    CLOSED_WON = 'closed_won'
    CLOSED_LOST = 'closed_lost'


class Deal(BaseModel):
    """
    Represents a sales opportunity/deal associated with an Account.

    Deals can be linked to ActionItems through the extras/attributes
    to provide sales context for action items.
    """

    id: str = Field(..., description='Deal/opportunity identifier')
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')
    account_id: str = Field(..., description='Parent account')

    # Deal info
    name: str = Field(..., description='Deal/opportunity name')
    stage: DealStage | str = Field(..., description='Current stage')
    value: float | None = Field(default=None, description='Deal value')
    currency: str = Field(default='USD', description='Currency code')

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=None))
    expected_close_date: datetime | None = Field(default=None)
    closed_at: datetime | None = Field(default=None)

    # Metadata
    attributes: dict[str, Any] = Field(default_factory=dict)

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        props = {
            'id': self.id,
            'tenant_id': str(self.tenant_id),
            'account_id': self.account_id,
            'name': self.name,
            'stage': self.stage.value if isinstance(self.stage, DealStage) else self.stage,
            'value': self.value,
            'currency': self.currency,
            'created_at': self.created_at.isoformat(),
            'expected_close_date': self.expected_close_date.isoformat()
            if self.expected_close_date
            else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
        }
        return {k: v for k, v in props.items() if v is not None}

    model_config = {'use_enum_values': True}
