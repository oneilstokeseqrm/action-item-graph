"""Tests for EnvelopeV1 enum expansion (email + meeting support) and contact enrichment."""

from action_item_graph.models.envelope import (
    ContentFormat,
    EnvelopeV1,
    InteractionType,
    SourceType,
)


# Reusable base envelope dict for tests
_BASE_ENVELOPE = {
    "schema_version": "v1",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user-123",
    "interaction_type": "transcript",
    "content": {"text": "Hello", "format": "plain"},
    "timestamp": "2026-03-16T10:00:00Z",
    "source": "web-mic",
}


class TestContactEnrichmentProperties:
    """Tests for contacts, contact_names properties on EnvelopeV1."""

    def test_contacts_returns_full_metadata(self):
        contacts = [
            {"contact_id": "c1", "email": "jane@acme.com", "name": "Jane Smith", "role": "organizer"},
            {"contact_id": "c2", "email": "bob@acme.com", "name": "Bob Jones", "role": "attendee"},
        ]
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contacts": contacts},
        })
        assert envelope.contacts == contacts
        assert len(envelope.contacts) == 2

    def test_contacts_empty_when_missing(self):
        envelope = EnvelopeV1.model_validate({**_BASE_ENVELOPE, "extras": {}})
        assert envelope.contacts == []

    def test_contacts_empty_when_no_extras(self):
        envelope = EnvelopeV1.model_validate(_BASE_ENVELOPE)
        assert envelope.contacts == []

    def test_contact_names_returns_names(self):
        contacts = [
            {"contact_id": "c1", "email": "jane@acme.com", "name": "Jane Smith", "role": "organizer"},
            {"contact_id": "c2", "email": "bob@acme.com", "name": "Bob Jones", "role": "attendee"},
        ]
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contacts": contacts},
        })
        assert envelope.contact_names == ["Jane Smith", "Bob Jones"]

    def test_contact_names_falls_back_to_email(self):
        contacts = [
            {"contact_id": "c1", "email": "unknown@acme.com", "name": None, "role": "attendee"},
            {"contact_id": "c2", "email": "bob@acme.com", "name": "Bob Jones", "role": "attendee"},
        ]
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contacts": contacts},
        })
        assert envelope.contact_names == ["unknown@acme.com", "Bob Jones"]

    def test_contact_names_falls_back_to_unknown(self):
        contacts = [{"contact_id": "c1", "role": "attendee"}]
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contacts": contacts},
        })
        assert envelope.contact_names == ["Unknown"]

    def test_contact_names_empty_when_no_contacts(self):
        envelope = EnvelopeV1.model_validate(_BASE_ENVELOPE)
        assert envelope.contact_names == []

    def test_opportunity_id_from_extras(self):
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"opportunity_id": "opp-123"},
        })
        assert envelope.opportunity_id == "opp-123"

    def test_opportunity_id_none_when_missing(self):
        envelope = EnvelopeV1.model_validate(_BASE_ENVELOPE)
        assert envelope.opportunity_id is None

    def test_contact_ids_still_works(self):
        """Backward compat: contact_ids property unchanged."""
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contact_ids": ["id1", "id2"]},
        })
        assert envelope.contact_ids == ["id1", "id2"]

    def test_contacts_and_contact_ids_coexist(self):
        """Both old and new contact formats can coexist."""
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {
                "contact_ids": ["c1", "c2"],
                "contacts": [
                    {"contact_id": "c1", "email": "jane@acme.com", "name": "Jane Smith", "role": "organizer"},
                ],
            },
        })
        assert envelope.contact_ids == ["c1", "c2"]
        assert len(envelope.contacts) == 1
        assert envelope.contact_names == ["Jane Smith"]


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
