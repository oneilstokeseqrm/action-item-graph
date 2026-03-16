"""
Tests for shared contact operations (ENGAGED_ON + name matching).

Covers:
- merge_contacts_to_deal: Neo4j MERGE calls and counting
- enrich_engaged_on_role: SET on existing relationship
- match_name_to_contact: fuzzy name matching

Run with: pytest tests/test_contact_ops.py -v
"""

from unittest.mock import AsyncMock

import pytest

from shared.contact_ops import (
    enrich_engaged_on_role,
    match_name_to_contact,
    merge_contacts_to_deal,
)

TENANT = 'tenant-1'
OPP_ID = 'opp-123'


# =============================================================================
# merge_contacts_to_deal
# =============================================================================


class TestMergeContactsToDeal:
    """MERGE (Contact)-[:ENGAGED_ON]->(Deal) for each contact."""

    @pytest.mark.asyncio
    async def test_creates_relationships_for_each_contact(self):
        neo4j = AsyncMock()
        neo4j.execute_write.return_value = [{'created': True}]

        count = await merge_contacts_to_deal(
            neo4j, TENANT, ['c1', 'c2', 'c3'], OPP_ID,
        )

        assert count == 3
        assert neo4j.execute_write.call_count == 3

    @pytest.mark.asyncio
    async def test_returns_zero_for_empty_contact_ids(self):
        neo4j = AsyncMock()

        count = await merge_contacts_to_deal(neo4j, TENANT, [], OPP_ID)

        assert count == 0
        neo4j.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_match_returns_nothing(self):
        """When Contact or Deal node missing, MATCH returns empty → count 0."""
        neo4j = AsyncMock()
        neo4j.execute_write.return_value = []

        count = await merge_contacts_to_deal(
            neo4j, TENANT, ['c1', 'c2'], OPP_ID,
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_passes_source_parameter(self):
        neo4j = AsyncMock()
        neo4j.execute_write.return_value = [{'created': True}]

        await merge_contacts_to_deal(
            neo4j, TENANT, ['c1'], OPP_ID, source='deal_pipeline',
        )

        call_params = neo4j.execute_write.call_args[0][1]
        assert call_params['source'] == 'deal_pipeline'

    @pytest.mark.asyncio
    async def test_passes_correct_parameters(self):
        neo4j = AsyncMock()
        neo4j.execute_write.return_value = [{'created': True}]

        await merge_contacts_to_deal(
            neo4j, TENANT, ['c1'], OPP_ID,
        )

        call_params = neo4j.execute_write.call_args[0][1]
        assert call_params['tenant_id'] == TENANT
        assert call_params['contact_id'] == 'c1'
        assert call_params['opportunity_id'] == OPP_ID
        assert call_params['source'] == 'envelope'

    @pytest.mark.asyncio
    async def test_counts_only_successful_merges(self):
        """Mixed results: some contacts exist, some don't."""
        neo4j = AsyncMock()
        neo4j.execute_write.side_effect = [
            [{'created': True}],  # c1 exists
            [],                    # c2 missing
            [{'created': True}],  # c3 exists
        ]

        count = await merge_contacts_to_deal(
            neo4j, TENANT, ['c1', 'c2', 'c3'], OPP_ID,
        )

        assert count == 2


# =============================================================================
# enrich_engaged_on_role
# =============================================================================


class TestEnrichEngagedOnRole:
    """Enrich ENGAGED_ON with LLM-extracted role."""

    @pytest.mark.asyncio
    async def test_sets_role_and_confidence(self):
        neo4j = AsyncMock()
        neo4j.execute_write.return_value = [{'enriched': True}]

        result = await enrich_engaged_on_role(
            neo4j, TENANT, 'c1', OPP_ID, 'champion', 0.9,
        )

        assert result is True
        call_params = neo4j.execute_write.call_args[0][1]
        assert call_params['role'] == 'champion'
        assert call_params['confidence'] == 0.9

    @pytest.mark.asyncio
    async def test_returns_false_when_relationship_missing(self):
        neo4j = AsyncMock()
        neo4j.execute_write.return_value = []

        result = await enrich_engaged_on_role(
            neo4j, TENANT, 'c1', OPP_ID, 'champion', 0.9,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_passes_all_parameters(self):
        neo4j = AsyncMock()
        neo4j.execute_write.return_value = [{'enriched': True}]

        await enrich_engaged_on_role(
            neo4j, TENANT, 'c1', OPP_ID, 'economic_buyer', 0.85,
        )

        call_params = neo4j.execute_write.call_args[0][1]
        assert call_params['tenant_id'] == TENANT
        assert call_params['contact_id'] == 'c1'
        assert call_params['opportunity_id'] == OPP_ID
        assert call_params['role'] == 'economic_buyer'
        assert call_params['confidence'] == 0.85


# =============================================================================
# match_name_to_contact
# =============================================================================


CONTACTS = [
    {'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'},
    {'contact_id': 'c2', 'name': 'Bob Jones', 'email': 'bob@acme.com'},
    {'contact_id': 'c3', 'name': 'Sarah Chen', 'email': 'sarah@acme.com'},
]


class TestMatchNameToContact:
    """Fuzzy name matching against contact list."""

    def test_exact_match(self):
        result = match_name_to_contact('Jane Smith', CONTACTS)
        assert result is not None
        assert result['contact_id'] == 'c1'

    def test_case_insensitive_match(self):
        result = match_name_to_contact('jane smith', CONTACTS)
        assert result is not None
        assert result['contact_id'] == 'c1'

    def test_partial_match_above_threshold(self):
        result = match_name_to_contact('Jane', CONTACTS)
        # "Jane" vs "Jane Smith" — SequenceMatcher should be above 0.75
        # Let's check: len("jane")=4, len("jane smith")=10, matching="jane"=4
        # ratio = 2*4 / (4+10) = 8/14 ≈ 0.57 — below 0.75
        # So "Jane" alone won't match. Use a closer partial.
        result2 = match_name_to_contact('Jane Smit', CONTACTS)
        assert result2 is not None
        assert result2['contact_id'] == 'c1'

    def test_below_threshold_returns_none(self):
        result = match_name_to_contact('Unknown Person', CONTACTS)
        assert result is None

    def test_empty_name_returns_none(self):
        assert match_name_to_contact('', CONTACTS) is None

    def test_empty_contacts_returns_none(self):
        assert match_name_to_contact('Jane', []) is None

    def test_none_name_returns_none(self):
        assert match_name_to_contact(None, CONTACTS) is None

    def test_contacts_with_no_names(self):
        contacts = [
            {'contact_id': 'c1', 'email': 'jane@acme.com'},
            {'contact_id': 'c2', 'name': '', 'email': 'bob@acme.com'},
        ]
        assert match_name_to_contact('Jane Smith', contacts) is None

    def test_best_match_wins(self):
        """When multiple contacts match, the best ratio wins."""
        contacts = [
            {'contact_id': 'c1', 'name': 'Sarah Jones'},
            {'contact_id': 'c2', 'name': 'Sarah Chen'},
        ]
        result = match_name_to_contact('Sarah Chen', contacts)
        assert result is not None
        assert result['contact_id'] == 'c2'

    def test_custom_threshold(self):
        """Lower threshold allows more partial matches."""
        result = match_name_to_contact('Jane', CONTACTS, min_ratio=0.5)
        assert result is not None
        assert result['contact_id'] == 'c1'
