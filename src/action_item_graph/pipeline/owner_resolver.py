"""
Account-scoped owner pre-resolution.

Normalizes owner names on extracted ActionItems BEFORE they reach the matcher/merger.
This prevents duplicate Owner nodes from being created for name variants like
"Peter O'Neill" / "Peter O'Neil" / "Peter" / "the account manager".

Runs after verification and before matching in the pipeline.
"""

from __future__ import annotations

import re
import unicodedata

from ..clients.openai_client import OpenAIClient
from ..logging import get_logger
from ..models.action_item import ActionItem
from ..prompts.owner_prompts import (
    ROLE_RESOLUTION_SYSTEM_PROMPT,
    ROLE_RESOLUTION_USER_PROMPT_TEMPLATE,
    RoleResolutionDecision,
)
from ..repository import ActionItemRepository

logger = get_logger(__name__)

# Minimum confidence for role-to-name LLM resolution
ROLE_RESOLUTION_CONFIDENCE = 0.8

# Minimum character overlap ratio for fuzzy name matching
FUZZY_MATCH_THRESHOLD = 0.80


def _normalize_name(name: str) -> str:
    """
    Normalize a name for comparison.

    - Lowercase
    - Strip whitespace
    - Normalize unicode (é → e)
    - Normalize apostrophes (O'Neill → o'neill, O\u2019Neill → o'neill)
    """
    name = name.strip().lower()
    # Normalize unicode
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    # Normalize all apostrophe variants to standard
    name = name.replace('\u2019', "'").replace('\u2018', "'").replace('\u0060', "'")
    return name


def _word_boundary_match(short: str, long: str) -> bool:
    """
    Check if the short name matches a word boundary in the long name.

    Prevents "Peter" from matching "Peterson" — requires the match to be
    at a word boundary (start/end of string or surrounded by non-alphanumeric).
    """
    pattern = r'\b' + re.escape(short) + r'\b'
    return bool(re.search(pattern, long, re.IGNORECASE))


def _char_overlap_ratio(a: str, b: str) -> float:
    """
    Compute character overlap ratio between two normalized strings.

    Used for fuzzy matching of name variants (O'Neill ↔ O'Neil).
    """
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = set_a & set_b
    return len(intersection) / max(len(set_a), len(set_b))


class OwnerCache:
    """
    In-memory cache of known owners for a specific account.

    Populated once per pipeline run from the graph database.
    """

    def __init__(self) -> None:
        self._by_canonical: dict[str, dict] = {}  # normalized_name -> owner_node
        self._by_alias: dict[str, dict] = {}  # normalized_alias -> owner_node
        self._all_owners: list[dict] = []

    def add(self, owner_node: dict) -> None:
        """Add an owner to the cache."""
        canonical = owner_node.get('canonical_name', '')
        if not canonical:
            return

        self._all_owners.append(owner_node)
        self._by_canonical[_normalize_name(canonical)] = owner_node

        for alias in owner_node.get('aliases', []):
            self._by_alias[_normalize_name(alias)] = owner_node

    def resolve(self, name: str) -> tuple[str | None, str]:
        """
        Try to resolve a name against the cache.

        Returns:
            Tuple of (resolved_canonical_name, resolution_method).
            Method is one of: 'exact', 'alias', 'substring', 'fuzzy', 'unresolved'.
        """
        normalized = _normalize_name(name)

        # 1. Exact match on canonical name
        if normalized in self._by_canonical:
            return self._by_canonical[normalized]['canonical_name'], 'exact'

        # 2. Exact match on alias
        if normalized in self._by_alias:
            return self._by_alias[normalized]['canonical_name'], 'alias'

        # 3. Word-boundary substring match (in either direction)
        for canon_norm, owner_node in self._by_canonical.items():
            if len(normalized) < len(canon_norm):
                # "Peter" matching "Peter O'Neill" — short in long
                if _word_boundary_match(normalized, canon_norm):
                    return owner_node['canonical_name'], 'substring'
            elif len(normalized) > len(canon_norm):
                # "Peter O'Neill" matching "Peter" — long contains short
                if _word_boundary_match(canon_norm, normalized):
                    return owner_node['canonical_name'], 'substring'

        # 4. Fuzzy variant match (O'Neill ↔ O'Neil, 80%+ char overlap)
        for canon_norm, owner_node in self._by_canonical.items():
            ratio = _char_overlap_ratio(normalized, canon_norm)
            if ratio >= FUZZY_MATCH_THRESHOLD and len(normalized) > 3:
                return owner_node['canonical_name'], 'fuzzy'

        return None, 'unresolved'

    @property
    def named_owners(self) -> list[str]:
        """Return list of all canonical owner names."""
        return [o.get('canonical_name', '') for o in self._all_owners if o.get('canonical_name')]

    @property
    def is_empty(self) -> bool:
        return len(self._all_owners) == 0


class OwnerPreResolver:
    """
    Pre-resolves owner names on ActionItems before they reach the matcher/merger.

    Algorithm:
    1. Load all known Owner nodes for the account into a cache
    2. For each ActionItem, try cache-based resolution (exact, alias, substring, fuzzy)
    3. For role_inferred owners with no cache hit, optionally ask LLM
    4. Update the ActionItem's owner/owner_type in-place
    """

    def __init__(
        self,
        repository: ActionItemRepository,
        openai_client: OpenAIClient | None = None,
    ):
        self.repository = repository
        self.openai_client = openai_client
        self._cache = OwnerCache()

    async def load_cache(
        self,
        tenant_id,
        account_id: str,
    ) -> int:
        """
        Populate the owner cache from the graph database.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier

        Returns:
            Number of owners loaded into cache
        """
        owners = await self.repository.get_owners_for_account(
            tenant_id=tenant_id,
            account_id=account_id,
        )

        for owner_node in owners:
            self._cache.add(owner_node)

        logger.info(
            'owner_cache_loaded',
            count=len(owners),
            account_id=account_id,
        )

        return len(owners)

    async def resolve_batch(
        self,
        action_items: list[ActionItem],
    ) -> dict[str, int]:
        """
        Resolve owner names for a batch of action items in-place.

        Args:
            action_items: List of ActionItems to resolve (modified in-place)

        Returns:
            Dict of resolution method counts (e.g., {'exact': 2, 'substring': 1, 'unresolved': 1})
        """
        if self._cache.is_empty:
            return {'unresolved': len(action_items)}

        method_counts: dict[str, int] = {}

        for ai in action_items:
            method = await self._resolve_single(ai)
            method_counts[method] = method_counts.get(method, 0) + 1

        logger.info(
            'owner_resolution_complete',
            methods=method_counts,
            total_items=len(action_items),
        )

        return method_counts

    async def _resolve_single(self, action_item: ActionItem) -> str:
        """
        Resolve a single ActionItem's owner. Modifies in-place.

        Returns the resolution method used.
        """
        original_owner = action_item.owner

        # Try cache-based resolution first
        resolved, method = self._cache.resolve(original_owner)

        if resolved and method != 'unresolved':
            if resolved != original_owner:
                logger.debug(
                    'owner_resolved',
                    original=original_owner,
                    resolved=resolved,
                    method=method,
                )
                action_item.owner = resolved
                if action_item.owner_type != 'named':
                    action_item.owner_type = 'named'
            return method

        # For role_inferred owners, try LLM resolution
        if action_item.owner_type == 'role_inferred' and self.openai_client:
            llm_resolved = await self._resolve_role_via_llm(action_item)
            if llm_resolved:
                action_item.owner = llm_resolved
                action_item.owner_type = 'named'
                return 'llm_role_resolved'

        return 'unresolved'

    async def _resolve_role_via_llm(self, action_item: ActionItem) -> str | None:
        """
        Ask the LLM to resolve a role description to a known person.

        Returns the resolved name or None.
        """
        if not self.openai_client or self._cache.is_empty:
            return None

        named_owners = self._cache.named_owners
        if not named_owners:
            return None

        known_people_text = '\n'.join(f'- {name}' for name in named_owners)

        messages = [
            {'role': 'system', 'content': ROLE_RESOLUTION_SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': ROLE_RESOLUTION_USER_PROMPT_TEMPLATE.format(
                    role_description=action_item.owner,
                    summary=action_item.summary,
                    action_item_text=action_item.action_item_text,
                    conversation_context=action_item.conversation_context,
                    known_people_text=known_people_text,
                ),
            },
        ]

        try:
            result = await self.openai_client.chat_completion_structured(
                messages=messages,
                response_model=RoleResolutionDecision,
            )

            if (
                result.resolved_name
                and result.confidence >= ROLE_RESOLUTION_CONFIDENCE
                and result.resolved_name in named_owners
            ):
                logger.debug(
                    'owner_role_resolved_via_llm',
                    role=action_item.owner,
                    resolved=result.resolved_name,
                    confidence=result.confidence,
                )
                return result.resolved_name

        except Exception:
            logger.exception('owner_role_resolution_failed', role=action_item.owner)

        return None
