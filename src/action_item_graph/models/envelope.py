"""
Input envelope model matching the upstream EnvelopeV1 Pydantic schema.

This is the standardized event envelope for all ecosystem events (Version 1).
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class InteractionType(str, Enum):
    """Type of interaction content."""

    TRANSCRIPT = 'transcript'
    NOTE = 'note'
    DOCUMENT = 'document'


class ContentFormat(str, Enum):
    """Format of the content text."""

    PLAIN = 'plain'
    MARKDOWN = 'markdown'
    DIARIZED = 'diarized'


class SourceType(str, Enum):
    """Origin of the content."""

    WEB_MIC = 'web-mic'
    UPLOAD = 'upload'
    API = 'api'
    IMPORT = 'import'


class ContentPayload(BaseModel):
    """The actual content payload within an envelope."""

    text: str = Field(..., description='The actual content text')
    format: ContentFormat = Field(
        default=ContentFormat.PLAIN,
        description='Content format (plain, markdown, diarized)',
    )


class EnvelopeV1(BaseModel):
    """
    Standardized event envelope for all ecosystem events (Version 1).

    This schema matches the EnvelopeV1 Pydantic model from the source service.
    All required fields must be present for valid payloads.
    """

    # Schema version
    schema_version: str = Field(default='v1', description='Event schema version')

    # Required identifiers
    tenant_id: UUID = Field(..., description='Tenant/organization UUID (REQUIRED)')
    user_id: str = Field(
        ..., description='User identifier (supports Auth0 IDs, type-prefixed IDs) (REQUIRED)'
    )
    pg_user_id: UUID | None = Field(
        default=None,
        description='Postgres user UUID from identity bridge (optional, dual-write alongside user_id)',
    )

    # Content
    interaction_type: InteractionType = Field(..., description='Type of interaction (REQUIRED)')
    content: ContentPayload = Field(..., description='The actual content payload (REQUIRED)')

    # Timing and source
    timestamp: datetime = Field(
        ..., description='Event creation timestamp (UTC, ISO 8601) (REQUIRED)'
    )
    source: SourceType = Field(..., description='Origin of content (REQUIRED)')

    # Optional fields
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description='Flexible metadata for domain-specific extensions. Can include: '
        'opportunity_id, contact_ids, deal_stage, etc.',
    )
    interaction_id: UUID | None = Field(
        default=None,
        description='Unique identifier for this interaction (optional, will be generated if not provided)',
    )
    trace_id: str | None = Field(
        default=None, description='Distributed tracing identifier (optional)'
    )
    account_id: str | None = Field(
        default=None,
        description='Account identifier for account-level sales context (optional but recommended for action item graph)',
    )

    # Helper properties for common extras
    @property
    def opportunity_id(self) -> str | None:
        """Extract opportunity_id from extras if present."""
        return self.extras.get('opportunity_id')

    @property
    def contact_ids(self) -> list[str]:
        """Extract contact_ids from extras if present."""
        return self.extras.get('contact_ids', [])

    @property
    def meeting_title(self) -> str | None:
        """Extract meeting_title from extras if present."""
        return self.extras.get('meeting_title')

    @property
    def duration_seconds(self) -> int | None:
        """Extract duration_seconds from extras if present."""
        return self.extras.get('duration_seconds')

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'schema_version': 'v1',
                    'tenant_id': '550e8400-e29b-41d4-a716-446655440000',
                    'user_id': 'auth0|abc123def456',
                    'interaction_type': 'transcript',
                    'content': {
                        'text': 'John: Thanks for joining the call today...',
                        'format': 'diarized',
                    },
                    'timestamp': '2025-01-23T10:30:00Z',
                    'source': 'web-mic',
                    'extras': {
                        'opportunity_id': '019c1fa0-4444-7000-8000-000000000005',
                        'contact_ids': ['contact_sarah_001'],
                    },
                    'account_id': 'acct_acme_corp_001',
                }
            ]
        }
    }
