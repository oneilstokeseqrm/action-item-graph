"""
Unit tests for account-scoped owner pre-resolution.

Tests the OwnerCache matching logic (pure functions) and the
OwnerPreResolver integration with mocked repository and LLM.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from action_item_graph.models.action_item import ActionItem
from action_item_graph.pipeline.owner_resolver import (
    FUZZY_MATCH_THRESHOLD,
    OwnerCache,
    OwnerPreResolver,
    _name_similarity,
    _normalize_name,
    _word_boundary_match,
)
from action_item_graph.prompts.owner_prompts import RoleResolutionDecision


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

TENANT_ID = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')


def _make_owner_node(
    canonical_name: str,
    aliases: list[str] | None = None,
    owner_id: str | None = None,
) -> dict:
    """Create a mock owner node dict."""
    return {
        'owner_id': owner_id or str(uuid.uuid4()),
        'tenant_id': str(TENANT_ID),
        'canonical_name': canonical_name,
        'aliases': aliases or [],
    }


def _make_action_item(
    owner: str,
    owner_type: str = 'named',
    summary: str = 'Test action item',
) -> ActionItem:
    """Create a test ActionItem."""
    return ActionItem(
        id=uuid.uuid4(),
        tenant_id=TENANT_ID,
        action_item_text=f'Action: {summary}',
        summary=summary,
        owner=owner,
        owner_type=owner_type,
        conversation_context='Test context',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure function tests
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeName:
    """Test name normalization."""

    def test_basic_normalization(self):
        assert _normalize_name('  John Smith  ') == 'john smith'

    def test_apostrophe_variants(self):
        # All apostrophe variants should normalize to the same form
        assert _normalize_name("O'Neill") == _normalize_name("O\u2019Neill")
        assert _normalize_name("O'Neill") == _normalize_name("O\u2018Neill")

    def test_unicode_normalization(self):
        assert _normalize_name('José') == 'jose'
        assert _normalize_name('naïve') == 'naive'


class TestWordBoundaryMatch:
    """Test word-boundary matching."""

    def test_name_at_start(self):
        assert _word_boundary_match('Peter', 'Peter O\'Neill') is True

    def test_name_at_end(self):
        assert _word_boundary_match('Neill', "Peter O'Neill") is True

    def test_partial_word_no_match(self):
        # "Peter" should NOT match "Peterson"
        assert _word_boundary_match('Peter', 'Peterson') is False

    def test_exact_match(self):
        assert _word_boundary_match('Sarah', 'Sarah') is True

    def test_case_insensitive(self):
        assert _word_boundary_match('peter', 'Peter O\'Neill') is True

    def test_no_match(self):
        assert _word_boundary_match('John', 'Sarah Smith') is False


class TestNameSimilarity:
    """Test sequence-based name similarity."""

    def test_identical_strings(self):
        assert _name_similarity('oneill', 'oneill') == 1.0

    def test_similar_strings(self):
        # O'Neill vs O'Neil — very similar
        ratio = _name_similarity("o'neill", "o'neil")
        assert ratio >= 0.85

    def test_different_strings(self):
        ratio = _name_similarity('sarah', 'john')
        assert ratio < 0.5

    def test_empty_strings(self):
        assert _name_similarity('', 'test') == 0.0
        assert _name_similarity('test', '') == 0.0

    def test_anagram_no_false_match(self):
        """'Sarah' and 'Harsh' share the same character SET but are different names.

        The old set-based char overlap would return ~1.0 for these.
        Sequence-based similarity correctly sees they are different.
        """
        ratio = _name_similarity('sarah', 'harsh')
        assert ratio < FUZZY_MATCH_THRESHOLD, (
            f"Anagram 'sarah'/'harsh' should NOT match (ratio={ratio:.2f}, "
            f"threshold={FUZZY_MATCH_THRESHOLD})"
        )

    def test_transposition_no_false_match(self):
        """Names that are anagram-like should not falsely match."""
        ratio = _name_similarity('clara', 'carol')
        assert ratio < FUZZY_MATCH_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# OwnerCache tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOwnerCache:
    """Test the OwnerCache lookup logic."""

    def test_empty_cache(self):
        cache = OwnerCache()
        assert cache.is_empty
        name, method = cache.resolve('Anyone')
        assert name is None
        assert method == 'unresolved'

    def test_exact_match(self):
        cache = OwnerCache()
        cache.add(_make_owner_node("Peter O'Neill"))

        name, method = cache.resolve("Peter O'Neill")
        assert name == "Peter O'Neill"
        assert method == 'exact'

    def test_case_insensitive_exact(self):
        cache = OwnerCache()
        cache.add(_make_owner_node("Peter O'Neill"))

        name, method = cache.resolve("peter o'neill")
        assert name == "Peter O'Neill"
        assert method == 'exact'

    def test_alias_match(self):
        cache = OwnerCache()
        cache.add(_make_owner_node("Peter O'Neill", aliases=['Pete', 'PO']))

        name, method = cache.resolve('Pete')
        assert name == "Peter O'Neill"
        assert method == 'alias'

    def test_substring_match_short_in_long(self):
        """'Peter' should match 'Peter O'Neill' (word boundary)."""
        cache = OwnerCache()
        cache.add(_make_owner_node("Peter O'Neill"))

        name, method = cache.resolve('Peter')
        assert name == "Peter O'Neill"
        assert method == 'substring'

    def test_substring_no_match_partial_word(self):
        """'Peter' should NOT match 'Peterson' (not a word boundary)."""
        cache = OwnerCache()
        cache.add(_make_owner_node('Peterson'))

        name, method = cache.resolve('Peter')
        assert name is None
        assert method == 'unresolved'

    def test_fuzzy_apostrophe_variant(self):
        """O'Neill should fuzzy-match O'Neil."""
        cache = OwnerCache()
        cache.add(_make_owner_node("Peter O'Neill"))

        name, method = cache.resolve("Peter O'Neil")
        # Should match via substring (Peter is in both) or fuzzy
        assert name == "Peter O'Neill"
        assert method in ('substring', 'fuzzy')

    def test_different_people_no_match(self):
        cache = OwnerCache()
        cache.add(_make_owner_node('Sarah Smith'))

        name, method = cache.resolve('John Doe')
        assert name is None
        assert method == 'unresolved'

    def test_named_owners(self):
        cache = OwnerCache()
        cache.add(_make_owner_node("Peter O'Neill"))
        cache.add(_make_owner_node('Sarah Smith'))

        names = cache.named_owners
        assert len(names) == 2
        assert "Peter O'Neill" in names
        assert 'Sarah Smith' in names


# ─────────────────────────────────────────────────────────────────────────────
# OwnerPreResolver tests (mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestOwnerPreResolver:
    """Test the OwnerPreResolver with mocked repository."""

    @pytest.mark.asyncio
    async def test_cache_loading(self):
        """Should load owners from repository."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(
            return_value=[
                _make_owner_node("Peter O'Neill", aliases=['Pete']),
                _make_owner_node('Sarah Smith'),
            ]
        )

        resolver = OwnerPreResolver(mock_repo)
        count = await resolver.load_cache(TENANT_ID, 'acct_001')

        assert count == 2
        mock_repo.get_owners_for_account.assert_called_once()

    @pytest.mark.asyncio
    async def test_name_resolution_in_place(self):
        """Should modify ActionItem owner in-place when resolved."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(
            return_value=[_make_owner_node("Peter O'Neill")]
        )

        resolver = OwnerPreResolver(mock_repo)
        await resolver.load_cache(TENANT_ID, 'acct_001')

        ai = _make_action_item('Peter')  # Short name variant
        methods = await resolver.resolve_batch([ai])

        assert ai.owner == "Peter O'Neill"  # Resolved in-place
        assert 'substring' in methods

    @pytest.mark.asyncio
    async def test_empty_cache_no_resolution(self):
        """With no known owners, nothing should be resolved."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(return_value=[])

        resolver = OwnerPreResolver(mock_repo)
        await resolver.load_cache(TENANT_ID, 'acct_001')

        ai = _make_action_item('Peter')
        methods = await resolver.resolve_batch([ai])

        assert ai.owner == 'Peter'  # Unchanged
        assert methods.get('unresolved', 0) == 1

    @pytest.mark.asyncio
    async def test_role_inferred_llm_resolution(self):
        """Role-inferred owners should be resolved via LLM when cache misses."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(
            return_value=[
                _make_owner_node("Peter O'Neill"),
                _make_owner_node('Sarah Smith'),
            ]
        )

        mock_openai = MagicMock()
        mock_openai.chat_completion_structured = AsyncMock(
            return_value=RoleResolutionDecision(
                resolved_name="Peter O'Neill",
                confidence=0.95,
                reasoning='Account manager is the primary salesperson',
            )
        )

        resolver = OwnerPreResolver(mock_repo, openai_client=mock_openai)
        await resolver.load_cache(TENANT_ID, 'acct_001')

        ai = _make_action_item('the account manager', owner_type='role_inferred')
        methods = await resolver.resolve_batch([ai])

        assert ai.owner == "Peter O'Neill"
        assert ai.owner_type == 'named'
        assert methods.get('llm_role_resolved', 0) == 1

    @pytest.mark.asyncio
    async def test_role_inferred_low_confidence_no_resolution(self):
        """LLM role resolution below confidence threshold should not resolve."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(
            return_value=[_make_owner_node("Peter O'Neill")]
        )

        mock_openai = MagicMock()
        mock_openai.chat_completion_structured = AsyncMock(
            return_value=RoleResolutionDecision(
                resolved_name="Peter O'Neill",
                confidence=0.5,  # Below ROLE_RESOLUTION_CONFIDENCE (0.8)
                reasoning='Not confident enough',
            )
        )

        resolver = OwnerPreResolver(mock_repo, openai_client=mock_openai)
        await resolver.load_cache(TENANT_ID, 'acct_001')

        ai = _make_action_item('someone on the team', owner_type='role_inferred')
        methods = await resolver.resolve_batch([ai])

        assert ai.owner == 'someone on the team'  # Unchanged
        assert methods.get('unresolved', 0) == 1

    @pytest.mark.asyncio
    async def test_already_correct_no_change(self):
        """Items with owners already matching canonical name should pass through."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(
            return_value=[_make_owner_node("Peter O'Neill")]
        )

        resolver = OwnerPreResolver(mock_repo)
        await resolver.load_cache(TENANT_ID, 'acct_001')

        ai = _make_action_item("Peter O'Neill")
        methods = await resolver.resolve_batch([ai])

        assert ai.owner == "Peter O'Neill"  # Unchanged (exact match)
        assert methods.get('exact', 0) == 1

    @pytest.mark.asyncio
    async def test_llm_failure_no_crash(self):
        """LLM failure during role resolution should not crash."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(
            return_value=[_make_owner_node("Peter O'Neill")]
        )

        mock_openai = MagicMock()
        mock_openai.chat_completion_structured = AsyncMock(
            side_effect=Exception('LLM timeout')
        )

        resolver = OwnerPreResolver(mock_repo, openai_client=mock_openai)
        await resolver.load_cache(TENANT_ID, 'acct_001')

        ai = _make_action_item('the lead', owner_type='role_inferred')
        methods = await resolver.resolve_batch([ai])

        assert ai.owner == 'the lead'  # Unchanged on failure
        assert methods.get('unresolved', 0) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Contact Seeding Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOwnerCacheContactSeeding:
    """Tests for seeding OwnerCache with contact metadata from envelope."""

    def test_add_contact_makes_name_resolvable(self):
        cache = OwnerCache()
        cache.add_contact({'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'})
        resolved, method = cache.resolve('Jane Smith')
        assert resolved == 'Jane Smith'
        assert method == 'exact'

    def test_add_contact_enables_substring_match(self):
        cache = OwnerCache()
        cache.add_contact({'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'})
        resolved, method = cache.resolve('Jane')
        assert resolved == 'Jane Smith'
        assert method == 'substring'

    def test_add_contact_does_not_overwrite_existing_owner(self):
        cache = OwnerCache()
        cache.add({'canonical_name': 'Jane Smith', 'owner_id': 'o1', 'aliases': []})
        cache.add_contact({'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'})
        # Should still resolve to the original owner
        resolved, method = cache.resolve('Jane Smith')
        assert resolved == 'Jane Smith'
        # But contact_id should still be recorded
        assert cache.get_contact_id('Jane Smith') == 'c1'

    def test_get_contact_id_returns_id_for_contact(self):
        cache = OwnerCache()
        cache.add_contact({'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'})
        assert cache.get_contact_id('Jane Smith') == 'c1'

    def test_get_contact_id_returns_none_for_non_contact(self):
        cache = OwnerCache()
        cache.add({'canonical_name': 'Bob Jones', 'owner_id': 'o1', 'aliases': []})
        assert cache.get_contact_id('Bob Jones') is None

    def test_add_contact_skips_missing_name(self):
        cache = OwnerCache()
        cache.add_contact({'contact_id': 'c1', 'email': 'unknown@acme.com'})
        assert cache.is_empty

    def test_add_contact_skips_missing_contact_id(self):
        cache = OwnerCache()
        cache.add_contact({'name': 'Jane Smith', 'email': 'jane@acme.com'})
        assert cache.is_empty

    def test_contact_seeding_with_owner_priority(self):
        """Owners loaded first take priority, contacts fill gaps."""
        cache = OwnerCache()
        # Load existing owner
        cache.add({'canonical_name': 'Jane Smith', 'owner_id': 'o1', 'aliases': []})
        # Seed contacts — Jane already exists, Bob is new
        cache.add_contact({'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'})
        cache.add_contact({'contact_id': 'c2', 'name': 'Bob Jones', 'email': 'bob@acme.com'})

        # Both resolvable
        resolved_jane, _ = cache.resolve('Jane Smith')
        resolved_bob, _ = cache.resolve('Bob Jones')
        assert resolved_jane == 'Jane Smith'
        assert resolved_bob == 'Bob Jones'

        # Both have contact_ids
        assert cache.get_contact_id('Jane Smith') == 'c1'
        assert cache.get_contact_id('Bob Jones') == 'c2'


class TestOwnerPreResolverContactSeeding:
    """Tests for OwnerPreResolver.load_cache with contacts parameter."""

    @pytest.mark.asyncio
    async def test_load_cache_seeds_contacts(self):
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(return_value=[])

        resolver = OwnerPreResolver(mock_repo)
        contacts = [
            {'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'},
            {'contact_id': 'c2', 'name': 'Bob Jones', 'email': 'bob@acme.com'},
        ]
        await resolver.load_cache(TENANT_ID, 'acct_001', contacts=contacts)

        assert resolver.get_contact_id('Jane Smith') == 'c1'
        assert resolver.get_contact_id('Bob Jones') == 'c2'

    @pytest.mark.asyncio
    async def test_load_cache_without_contacts_backward_compat(self):
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(return_value=[
            {'canonical_name': 'Alice', 'owner_id': 'o1', 'aliases': []},
        ])

        resolver = OwnerPreResolver(mock_repo)
        count = await resolver.load_cache(TENANT_ID, 'acct_001')
        assert count == 1
        assert resolver.get_contact_id('Alice') is None

    @pytest.mark.asyncio
    async def test_contact_resolution_flow(self):
        """Full flow: contact seeded → name resolved → contact_id available."""
        mock_repo = MagicMock()
        mock_repo.get_owners_for_account = AsyncMock(return_value=[])

        resolver = OwnerPreResolver(mock_repo)
        await resolver.load_cache(
            TENANT_ID, 'acct_001',
            contacts=[{'contact_id': 'c1', 'name': 'Jane Smith', 'email': 'jane@acme.com'}],
        )

        ai = _make_action_item('Jane')
        methods = await resolver.resolve_batch([ai])

        assert ai.owner == 'Jane Smith'
        assert methods.get('substring', 0) == 1
        assert resolver.get_contact_id('Jane Smith') == 'c1'
