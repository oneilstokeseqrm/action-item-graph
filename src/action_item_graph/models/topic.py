"""
Topic and TopicVersion models for grouping action items into high-level themes.

Topics enable users to retrieve the full "story" of a project across conversations
by clustering related action items into shared themes (e.g., "Q3 Audit", "Website Redesign").

Key features:
- Dual embeddings: original (immutable) + current (evolves as items are linked)
- Version tracking for summary evolution
- Multi-tenant isolation via tenant_id
- Account scoping via account_id
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Topic(BaseModel):
    """
    High-level theme/project that groups related action items.

    The Topic node connects:
    - To Account via HAS_TOPIC relationship
    - To ActionItem(s) via BELONGS_TO relationship (action item -> topic)
    - To TopicVersion(s) via HAS_VERSION relationship

    Embeddings:
    - embedding: Original embedding (immutable, from first creation)
    - embedding_current: Current state embedding (updated as summary evolves)

    Both embeddings are used for topic matching:
    - embedding catches new items conceptually similar to original topic
    - embedding_current catches items related to evolved topic scope
    """

    # Identity
    id: UUID = Field(default_factory=uuid4, description='Unique identifier for this topic')

    # Multi-tenancy (present on ALL nodes)
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')
    account_id: str = Field(..., description='Account identifier for CRM context')

    # Core content
    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description='Display name for the topic (3-5 words, e.g., "Q3 Sales Audit")',
    )
    canonical_name: str = Field(
        ...,
        description='Normalized name for matching (lowercase, trimmed)',
    )
    summary: str = Field(
        default='',
        max_length=500,
        description='LLM-generated summary of the topic scope and related action items',
    )

    # Dual embeddings for matching
    embedding: list[float] | None = Field(
        default=None,
        description='Original embedding (immutable, from first creation). '
        'Used to catch new items conceptually similar to original topic.',
    )
    embedding_current: list[float] | None = Field(
        default=None,
        description='Current state embedding (updated as summary evolves). '
        'Used to catch items related to evolved topic scope.',
    )

    # Statistics
    action_item_count: int = Field(
        default=0,
        ge=0,
        description='Number of action items linked to this topic (denormalized for performance)',
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=None),
        description='When the topic was first created',
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=None),
        description='When the topic was last updated',
    )

    # Provenance
    created_from_action_item_id: UUID | None = Field(
        default=None,
        description='The action item that triggered creation of this topic',
    )

    # Version tracking
    version: int = Field(default=1, description='Version number, incremented on updates')

    def to_neo4j_properties(self) -> dict[str, Any]:
        """
        Convert to Neo4j-compatible property dict.

        Note: Embeddings are stored as lists, which Neo4j handles natively for vector indexes.
        UUIDs are converted to strings for Neo4j compatibility.
        """
        props = {
            'id': str(self.id),
            'tenant_id': str(self.tenant_id),
            'account_id': self.account_id,
            'name': self.name,
            'canonical_name': self.canonical_name,
            'summary': self.summary,
            'embedding': self.embedding,
            'embedding_current': self.embedding_current,
            'action_item_count': self.action_item_count,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_from_action_item_id': str(self.created_from_action_item_id)
            if self.created_from_action_item_id
            else None,
            'version': self.version,
        }
        # Filter out None values for cleaner Neo4j storage
        return {k: v for k, v in props.items() if v is not None}

    @classmethod
    def canonicalize_name(cls, name: str) -> str:
        """
        Convert a topic name to its canonical form.

        Args:
            name: Raw topic name

        Returns:
            Normalized name (lowercase, trimmed, single spaces)
        """
        return ' '.join(name.lower().strip().split())


class TopicVersion(BaseModel):
    """
    Historical snapshot of a Topic at a specific point in time.

    Created whenever a Topic summary is updated, preserving the full state
    before the update. This enables:
    - Full audit trail of topic evolution
    - Understanding how topic scope has changed
    - Tracking which action items drove topic changes
    """

    # Identity
    id: UUID = Field(default_factory=uuid4, description='Unique identifier for this version')
    topic_id: UUID = Field(..., description='The parent Topic this version belongs to')

    # Multi-tenancy
    tenant_id: UUID = Field(..., description='Tenant/organization UUID')

    # Version info
    version_number: int = Field(..., description='Version number (1, 2, 3, ...)')

    # Snapshot of Topic state at this version
    name: str = Field(..., description='Topic name at this version')
    summary: str = Field(..., description='Topic summary at this version')
    embedding_snapshot: list[float] | None = Field(
        default=None,
        description='Embedding at this version (for analyzing drift)',
    )

    # What triggered this version
    changed_by_action_item_id: UUID | None = Field(
        default=None,
        description='The action item that triggered this version change',
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=None),
        description='When this version was created',
    )

    def to_neo4j_properties(self) -> dict[str, Any]:
        """Convert to Neo4j-compatible property dict."""
        props = {
            'id': str(self.id),
            'topic_id': str(self.topic_id),
            'tenant_id': str(self.tenant_id),
            'version_number': self.version_number,
            'name': self.name,
            'summary': self.summary,
            'embedding_snapshot': self.embedding_snapshot,
            'changed_by_action_item_id': str(self.changed_by_action_item_id)
            if self.changed_by_action_item_id
            else None,
            'created_at': self.created_at.isoformat(),
        }
        return {k: v for k, v in props.items() if v is not None}


class ExtractedTopic(BaseModel):
    """
    A topic extracted alongside an action item from a transcript.

    This is the intermediate representation used during extraction,
    before being resolved to an existing Topic or creating a new one.
    """

    name: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description='High-level topic name (3-5 words). Examples: "Q1 Sales Expansion", '
        '"Annual Security Compliance", "Website Redesign Project"',
    )
    context: str = Field(
        ...,
        max_length=200,
        description='Brief explanation (1-2 sentences) of why the action item belongs '
        'to this topic and what the topic encompasses.',
    )
