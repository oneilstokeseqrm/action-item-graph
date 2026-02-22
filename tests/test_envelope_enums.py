"""Tests for EnvelopeV1 enum expansion (email + meeting support)."""

from action_item_graph.models.envelope import (
    ContentFormat,
    EnvelopeV1,
    InteractionType,
    SourceType,
)


class TestInteractionTypeEnum:
    def test_email_type_exists(self):
        assert InteractionType.EMAIL == "email"

    def test_meeting_type_exists(self):
        assert InteractionType.MEETING == "meeting"


class TestContentFormatEnum:
    def test_email_format_exists(self):
        assert ContentFormat.EMAIL == "email"


class TestSourceTypeEnum:
    def test_gmail_source_exists(self):
        assert SourceType.GMAIL == "gmail"

    def test_outlook_source_exists(self):
        assert SourceType.OUTLOOK == "outlook"


class TestEmailEnvelopeParsing:
    def test_parse_email_envelope(self):
        raw = {
            "schema_version": "v1",
            "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "b88c4c51-b342-4f24-bd0c-cb1b8faf10d0",
            "interaction_type": "email",
            "content": {
                "text": "---\ntype: email\nsubject: Test\n---\nHello",
                "format": "email",
            },
            "timestamp": "2026-02-14T15:30:00Z",
            "source": "gmail",
            "extras": {
                "subject": "Test Subject",
                "from_email": "alice@example.com",
                "direction": "inbound",
                "thread_key": "gmail:abc123",
                "has_attachments": False,
            },
            "interaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "account_id": "acct_test_001",
        }
        envelope = EnvelopeV1.model_validate(raw)
        assert envelope.interaction_type == InteractionType.EMAIL
        assert envelope.content.format == ContentFormat.EMAIL
        assert envelope.source == SourceType.GMAIL
        assert envelope.extras["subject"] == "Test Subject"

    def test_parse_outlook_email_envelope(self):
        raw = {
            "schema_version": "v1",
            "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "user-123",
            "interaction_type": "email",
            "content": {"text": "Hello from Outlook", "format": "email"},
            "timestamp": "2026-02-14T15:30:00Z",
            "source": "outlook",
        }
        envelope = EnvelopeV1.model_validate(raw)
        assert envelope.source == SourceType.OUTLOOK

    def test_parse_meeting_envelope(self):
        raw = {
            "schema_version": "v1",
            "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "user-123",
            "interaction_type": "meeting",
            "content": {"text": "Meeting notes...", "format": "plain"},
            "timestamp": "2026-02-14T15:30:00Z",
            "source": "web-mic",
        }
        envelope = EnvelopeV1.model_validate(raw)
        assert envelope.interaction_type == InteractionType.MEETING

    def test_email_envelope_null_account_id(self):
        raw = {
            "schema_version": "v1",
            "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "user-123",
            "interaction_type": "email",
            "content": {"text": "Body", "format": "email"},
            "timestamp": "2026-02-14T15:30:00Z",
            "source": "gmail",
            "account_id": None,
        }
        envelope = EnvelopeV1.model_validate(raw)
        assert envelope.account_id is None
