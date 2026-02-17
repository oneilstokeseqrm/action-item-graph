"""
Unit tests for owner attribution features (owner_type, is_user_owned).

Tests the prompt builder, model fields, and neo4j property serialization
without hitting external services.

Run with: pytest tests/test_owner_attribution.py -v
"""

import uuid

import pytest

from action_item_graph.models.action_item import ActionItem, ActionItemStatus
from action_item_graph.prompts.extract_action_items import (
    ExtractedActionItem,
    ExtractedTopic,
    ExtractionResult,
    build_extraction_prompt,
)
from action_item_graph.prompts.merge_action_items import (
    MergedActionItem,
    build_merge_prompt,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_topic() -> ExtractedTopic:
    return ExtractedTopic(name="Test Topic", context="Test context for topic")


# =============================================================================
# ExtractedActionItem field defaults
# =============================================================================


class TestExtractedActionItemDefaults:
    """Verify new fields have correct defaults on the extraction model."""

    def test_default_owner_type_is_named(self):
        item = ExtractedActionItem(
            action_item_text="Send the report",
            owner="Sarah",
            summary="Sarah to send the report",
            conversation_context="Discussed quarterly review",
            topic=_make_topic(),
        )
        assert item.owner_type == "named"

    def test_default_is_user_owned_is_false(self):
        item = ExtractedActionItem(
            action_item_text="Send the report",
            owner="Sarah",
            summary="Sarah to send the report",
            conversation_context="Discussed quarterly review",
            topic=_make_topic(),
        )
        assert item.is_user_owned is False

    def test_explicit_owner_type_role_inferred(self):
        item = ExtractedActionItem(
            action_item_text="Send the report",
            owner="the account executive",
            summary="The account executive to send the report",
            conversation_context="Discussed quarterly review",
            topic=_make_topic(),
            owner_type="role_inferred",
        )
        assert item.owner_type == "role_inferred"

    def test_explicit_owner_type_unconfirmed(self):
        item = ExtractedActionItem(
            action_item_text="Send the report",
            owner="unconfirmed",
            summary="Send the report",
            conversation_context="Discussed quarterly review",
            topic=_make_topic(),
            owner_type="unconfirmed",
        )
        assert item.owner_type == "unconfirmed"

    def test_explicit_is_user_owned_true(self):
        item = ExtractedActionItem(
            action_item_text="Send the report",
            owner="Peter",
            summary="Peter to send the report",
            conversation_context="Discussed quarterly review",
            topic=_make_topic(),
            is_user_owned=True,
        )
        assert item.is_user_owned is True


# =============================================================================
# ActionItem model fields
# =============================================================================


class TestActionItemModelFields:
    """Verify new fields on the ActionItem persistence model."""

    def test_default_owner_type(self):
        ai = ActionItem(
            tenant_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            action_item_text="Send the report",
            summary="Send the report",
            owner="Sarah",
        )
        assert ai.owner_type == "named"
        assert ai.is_user_owned is False

    def test_custom_owner_type(self):
        ai = ActionItem(
            tenant_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            action_item_text="Send the report",
            summary="Send the report",
            owner="the project lead",
            owner_type="role_inferred",
            is_user_owned=True,
        )
        assert ai.owner_type == "role_inferred"
        assert ai.is_user_owned is True

    def test_to_neo4j_properties_includes_new_fields(self):
        ai = ActionItem(
            tenant_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            action_item_text="Send the report",
            summary="Send the report",
            owner="Sarah",
            owner_type="named",
            is_user_owned=True,
        )
        props = ai.to_neo4j_properties()
        assert props['owner_type'] == 'named'
        assert props['is_user_owned'] is True

    def test_to_neo4j_properties_defaults(self):
        ai = ActionItem(
            tenant_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            action_item_text="Send the report",
            summary="Send the report",
            owner="Sarah",
        )
        props = ai.to_neo4j_properties()
        assert props['owner_type'] == 'named'
        assert props['is_user_owned'] is False


# =============================================================================
# Extraction prompt: user_name threading
# =============================================================================


class TestBuildExtractionPrompt:
    """Verify user_name is threaded into the extraction prompt."""

    def test_user_name_included_in_context(self):
        messages = build_extraction_prompt(
            transcript_text="A: Hello\nB: Hi",
            meeting_title="Test Meeting",
            user_name="Peter O'Neil",
        )
        user_msg = messages[1]['content']
        assert "Recording user: Peter O'Neil" in user_msg

    def test_user_name_absent_when_none(self):
        messages = build_extraction_prompt(
            transcript_text="A: Hello\nB: Hi",
            meeting_title="Test Meeting",
        )
        user_msg = messages[1]['content']
        assert "Recording user" not in user_msg

    def test_user_name_absent_when_not_provided(self):
        messages = build_extraction_prompt(
            transcript_text="A: Hello\nB: Hi",
            meeting_title="Test Meeting",
            user_name=None,
        )
        user_msg = messages[1]['content']
        assert "Recording user" not in user_msg

    def test_system_prompt_contains_speaker_attribution_rules(self):
        messages = build_extraction_prompt(
            transcript_text="A: Hello\nB: Hi",
        )
        system_msg = messages[0]['content']
        assert "Speaker Attribution Rules" in system_msg
        assert "owner_type" in system_msg
        assert "is_user_owned" in system_msg
        assert "diarization labels" in system_msg

    def test_all_context_fields_present(self):
        """Verify meeting_title, participants, and user_name all appear."""
        messages = build_extraction_prompt(
            transcript_text="A: Hello\nB: Hi",
            meeting_title="Weekly Sync",
            participants=["Alice", "Bob"],
            user_name="Alice",
        )
        user_msg = messages[1]['content']
        assert "Meeting title: Weekly Sync" in user_msg
        assert "Participants: Alice, Bob" in user_msg
        assert "Recording user: Alice" in user_msg


# =============================================================================
# Merge prompt: owner_type threading
# =============================================================================


class TestBuildMergePrompt:
    """Verify new fields are threaded into the merge prompt."""

    def test_merge_prompt_includes_owner_type(self):
        messages = build_merge_prompt(
            existing_text="Send the report",
            existing_summary="Send the report",
            existing_owner="the account exec",
            existing_status="open",
            existing_due_date=None,
            existing_context="Quarterly review",
            existing_created="2026-01-15",
            new_text="Send the report to legal",
            new_summary="Send report to legal",
            new_owner="Sarah",
            new_context="Follow-up meeting",
            new_is_status_update=False,
            new_implied_status=None,
            new_due_date=None,
            merge_recommendation="merge",
            existing_owner_type="role_inferred",
            existing_is_user_owned=False,
            new_owner_type="named",
            new_is_user_owned=True,
        )
        user_msg = messages[1]['content']
        assert "Owner Type: role_inferred" in user_msg
        assert "Owner Type: named" in user_msg
        assert "Is User Owned: False" in user_msg
        assert "Is User Owned: True" in user_msg

    def test_merge_prompt_defaults(self):
        """Verify merge prompt works without explicit owner_type args (backwards compat)."""
        messages = build_merge_prompt(
            existing_text="Send the report",
            existing_summary="Send the report",
            existing_owner="Sarah",
            existing_status="open",
            existing_due_date=None,
            existing_context="Context",
            existing_created="2026-01-15",
            new_text="Send report to legal",
            new_summary="Send report to legal",
            new_owner="Sarah",
            new_context="Context",
            new_is_status_update=False,
            new_implied_status=None,
            new_due_date=None,
            merge_recommendation="merge",
        )
        user_msg = messages[1]['content']
        # Defaults should be "named" and "False"
        assert "Owner Type: named" in user_msg
        assert "Is User Owned: False" in user_msg

    def test_merge_system_prompt_contains_owner_type_rules(self):
        messages = build_merge_prompt(
            existing_text="x",
            existing_summary="x",
            existing_owner="x",
            existing_status="open",
            existing_due_date=None,
            existing_context="x",
            existing_created="2026-01-15",
            new_text="y",
            new_summary="y",
            new_owner="y",
            new_context="y",
            new_is_status_update=False,
            new_implied_status=None,
            new_due_date=None,
            merge_recommendation="merge",
        )
        system_msg = messages[0]['content']
        assert "Owner type resolution" in system_msg
        assert "unconfirmed" in system_msg


# =============================================================================
# MergedActionItem fields
# =============================================================================


class TestMergedActionItemFields:
    """Verify new fields on the MergedActionItem model."""

    def test_defaults(self):
        merged = MergedActionItem(
            action_item_text="Send the report",
            summary="Send the report",
            evolution_summary="Updated with details",
            owner="Sarah",
            should_update_embedding=False,
        )
        assert merged.owner_type == "named"
        assert merged.is_user_owned is False

    def test_explicit_values(self):
        merged = MergedActionItem(
            action_item_text="Send the report",
            summary="Send the report",
            evolution_summary="Updated with details",
            owner="Peter",
            owner_type="named",
            is_user_owned=True,
            should_update_embedding=False,
        )
        assert merged.owner_type == "named"
        assert merged.is_user_owned is True


# =============================================================================
# Field flow: ExtractedActionItem → ActionItem
# =============================================================================


class TestFieldFlow:
    """Verify new fields correctly flow from extraction to ActionItem."""

    def test_named_owner_flows_to_action_item(self):
        extraction = ExtractedActionItem(
            action_item_text="Send the report",
            owner="Sarah",
            summary="Sarah to send the report",
            conversation_context="Discussed quarterly review",
            topic=_make_topic(),
            owner_type="named",
            is_user_owned=False,
        )

        ai = ActionItem(
            tenant_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            action_item_text=extraction.action_item_text,
            summary=extraction.summary,
            owner=extraction.owner,
            owner_type=extraction.owner_type,
            is_user_owned=extraction.is_user_owned,
        )

        assert ai.owner_type == "named"
        assert ai.is_user_owned is False
        props = ai.to_neo4j_properties()
        assert props['owner_type'] == "named"
        assert props['is_user_owned'] is False

    def test_unconfirmed_owner_flows_to_action_item(self):
        extraction = ExtractedActionItem(
            action_item_text="Send the report by Friday",
            owner="unconfirmed",
            summary="Send the report by Friday",
            conversation_context="Speaker unknown",
            topic=_make_topic(),
            owner_type="unconfirmed",
            is_user_owned=False,
        )

        ai = ActionItem(
            tenant_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            action_item_text=extraction.action_item_text,
            summary=extraction.summary,
            owner=extraction.owner,
            owner_type=extraction.owner_type,
            is_user_owned=extraction.is_user_owned,
        )

        assert ai.owner_type == "unconfirmed"
        props = ai.to_neo4j_properties()
        assert props['owner_type'] == "unconfirmed"

    def test_user_owned_flows_to_action_item(self):
        extraction = ExtractedActionItem(
            action_item_text="I'll follow up with the client",
            owner="Peter",
            summary="Peter to follow up with the client",
            conversation_context="Peter commits to follow-up",
            topic=_make_topic(),
            owner_type="named",
            is_user_owned=True,
        )

        ai = ActionItem(
            tenant_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            action_item_text=extraction.action_item_text,
            summary=extraction.summary,
            owner=extraction.owner,
            owner_type=extraction.owner_type,
            is_user_owned=extraction.is_user_owned,
        )

        assert ai.is_user_owned is True
        props = ai.to_neo4j_properties()
        assert props['is_user_owned'] is True
