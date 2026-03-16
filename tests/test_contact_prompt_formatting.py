"""Tests for contact-enriched prompt formatting in extraction prompts."""

from action_item_graph.models.envelope import EnvelopeV1
from action_item_graph.prompts.extract_action_items import build_extraction_prompt
from deal_graph.prompts.extract_deals import build_discovery_prompt, build_targeted_prompt


_BASE_ENVELOPE = {
    "schema_version": "v1",
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "user-123",
    "interaction_type": "transcript",
    "content": {"text": "Hello", "format": "plain"},
    "timestamp": "2026-03-16T10:00:00Z",
    "source": "web-mic",
}


class TestContactLabels:
    """Tests for EnvelopeV1.contact_labels property."""

    def test_rich_labels_with_name_email_role(self):
        contacts = [
            {"contact_id": "c1", "email": "jane@acme.com", "name": "Jane Smith", "role": "organizer"},
            {"contact_id": "c2", "email": "bob@acme.com", "name": "Bob Jones", "role": "attendee"},
        ]
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contacts": contacts},
        })
        labels = envelope.contact_labels
        assert labels == [
            "Jane Smith <jane@acme.com> (organizer)",
            "Bob Jones <bob@acme.com> (attendee)",
        ]

    def test_label_name_only_no_email(self):
        contacts = [{"contact_id": "c1", "name": "Jane Smith", "role": "organizer"}]
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contacts": contacts},
        })
        assert envelope.contact_labels == ["Jane Smith (organizer)"]

    def test_label_email_only_no_name(self):
        contacts = [{"contact_id": "c1", "email": "unknown@acme.com", "name": None, "role": "attendee"}]
        envelope = EnvelopeV1.model_validate({
            **_BASE_ENVELOPE,
            "extras": {"contacts": contacts},
        })
        assert envelope.contact_labels == ["unknown@acme.com (attendee)"]

    def test_empty_contacts_returns_empty_labels(self):
        envelope = EnvelopeV1.model_validate(_BASE_ENVELOPE)
        assert envelope.contact_labels == []


class TestActionItemPromptParticipants:
    """Tests for participant formatting in action item extraction prompt."""

    def test_prompt_includes_bullet_point_participants(self):
        participants = [
            "Jane Smith <jane@acme.com> (organizer)",
            "Bob Jones <bob@acme.com> (attendee)",
        ]
        messages = build_extraction_prompt(
            transcript_text="Hello world",
            meeting_title="Q3 Review",
            participants=participants,
        )
        user_content = messages[1]['content']
        assert "Meeting participants:" in user_content
        assert "  - Jane Smith <jane@acme.com> (organizer)" in user_content
        assert "  - Bob Jones <bob@acme.com> (attendee)" in user_content

    def test_prompt_without_participants(self):
        messages = build_extraction_prompt(
            transcript_text="Hello world",
            meeting_title="Q3 Review",
        )
        user_content = messages[1]['content']
        assert "Meeting participants:" not in user_content
        assert "Participants:" not in user_content

    def test_prompt_with_simple_name_participants(self):
        """Backward compat: simple name strings also work."""
        participants = ["Sarah Jones", "James Park"]
        messages = build_extraction_prompt(
            transcript_text="Hello",
            participants=participants,
        )
        user_content = messages[1]['content']
        assert "  - Sarah Jones" in user_content
        assert "  - James Park" in user_content


class TestDealPromptParticipants:
    """Tests for participant formatting in deal extraction prompts."""

    def test_discovery_prompt_includes_participants(self):
        participants = ["Jane Smith <jane@acme.com> (organizer)"]
        messages = build_discovery_prompt(
            content_text="Hello world",
            meeting_title="Pipeline Review",
            participants=participants,
        )
        user_content = messages[1]['content']
        assert "Meeting participants:" in user_content
        assert "  - Jane Smith <jane@acme.com> (organizer)" in user_content

    def test_targeted_prompt_includes_participants(self):
        participants = ["Jane Smith <jane@acme.com> (organizer)"]
        existing_deal = {
            "name": "Acme Deal",
            "stage": "discovery",
            "opportunity_summary": "A deal",
        }
        messages = build_targeted_prompt(
            content_text="Hello world",
            existing_deal=existing_deal,
            meeting_title="Deal Review",
            participants=participants,
        )
        user_content = messages[1]['content']
        assert "Meeting participants:" in user_content
        assert "  - Jane Smith <jane@acme.com> (organizer)" in user_content

    def test_discovery_prompt_without_participants(self):
        messages = build_discovery_prompt(
            content_text="Hello world",
        )
        user_content = messages[1]['content']
        assert "Meeting participants:" not in user_content
