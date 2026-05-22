"""Unit tests for TopicExecutor Rule 6 (DBOS retry idempotency) fixes.

S10b (``topic_resolution_persist_step``) is
``retries_allowed=True, max_attempts=3``. The prior implementation used
``uuid4()`` for ActionItemTopic IDs and ``randomUUID()`` Cypher for
ActionItemTopicVersion IDs — each retry would create a duplicate node.
Phase F /review absorbed this as a Rule 6 hazard, mirroring the Phase B-2
fix for ``create_version_snapshot``.

The fix derives deterministic UUID5s at the call site so the repository
MERGE-keyed Cypher idempotently no-ops on the second call:
- ``_topic_id_for_resolution(tenant_id, account_id, canonical_name)``
- ``_topic_version_id(topic_id, changed_by_action_item_id)``

These tests assert that two invocations with identical inputs produce
identical IDs (the load-bearing contract for retry safety) and that
different inputs produce different IDs (the disambiguation contract).

No live Neo4j or OpenAI required — all dependencies are mocked. See
``memory/pattern_dbos_workflow_parity_rules.md`` Rule 6 and HANDOFF.md
§ "Picking up from Phase E complete" for the rationale.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from action_item_graph.pipeline.topic_executor import (
    TopicExecutor,
    _topic_id_for_resolution,
    _topic_version_id,
)
from action_item_graph.pipeline.topic_resolver import (
    TopicDecision,
    TopicResolutionResult,
)
from action_item_graph.prompts.extract_action_items import ExtractedTopic
from action_item_graph.prompts.topic_prompts import TopicSummary


TENANT_ID = UUID('11111111-1111-4111-8111-111111111111')
ACCOUNT_ID = 'acct-lightbox'
ACTION_ITEM_ID = '22222222-2222-4222-8222-222222222222'


# ---------------------------------------------------------------------------
# Deterministic helpers (call them directly)
# ---------------------------------------------------------------------------


class TestTopicIdHelperDeterminism:
    """Direct unit tests for the deterministic ID helpers.

    The topic_id key includes ``source_action_item_id`` per Codex R1
    absorption. This preserves Rule 6 retry safety (same resolution
    retries → same topic_id) AND legacy parity (two batch-mate action
    items with the same canonical_name → two distinct Topic nodes,
    matching the prior ``uuid4()`` per-call behavior).
    """

    def test_topic_id_same_inputs_produce_same_id(self):
        """Rule 6: retries of the same resolution converge to the same topic_id."""
        a = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, 'q1 sales expansion',
            source_action_item_id=ACTION_ITEM_ID,
        )
        b = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, 'q1 sales expansion',
            source_action_item_id=ACTION_ITEM_ID,
        )
        assert a == b
        assert a.version == 5

    def test_topic_id_same_canonical_name_different_source_produces_different_id(self):
        """Legacy parity: two action items in one batch with the same
        canonical_name produce DISTINCT Topic nodes (not collapsed)."""
        a = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, 'q1 sales expansion',
            source_action_item_id=ACTION_ITEM_ID,
        )
        b = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, 'q1 sales expansion',
            source_action_item_id='99999999-9999-4999-8999-999999999999',
        )
        assert a != b

    def test_topic_id_different_canonical_name_produces_different_id(self):
        a = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, 'q1 sales expansion',
            source_action_item_id=ACTION_ITEM_ID,
        )
        b = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, 'q2 sales expansion',
            source_action_item_id=ACTION_ITEM_ID,
        )
        assert a != b

    def test_topic_id_different_account_produces_different_id(self):
        a = _topic_id_for_resolution(
            TENANT_ID, 'acct-lightbox', 'shared name',
            source_action_item_id=ACTION_ITEM_ID,
        )
        b = _topic_id_for_resolution(
            TENANT_ID, 'acct-other', 'shared name',
            source_action_item_id=ACTION_ITEM_ID,
        )
        assert a != b

    def test_topic_id_different_tenant_produces_different_id(self):
        other_tenant = UUID('33333333-3333-4333-8333-333333333333')
        a = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, 'topic name',
            source_action_item_id=ACTION_ITEM_ID,
        )
        b = _topic_id_for_resolution(
            other_tenant, ACCOUNT_ID, 'topic name',
            source_action_item_id=ACTION_ITEM_ID,
        )
        assert a != b

    def test_topic_version_id_same_inputs_produce_same_id(self):
        topic_id = '44444444-4444-4444-8444-444444444444'
        a = _topic_version_id(topic_id, ACTION_ITEM_ID)
        b = _topic_version_id(topic_id, ACTION_ITEM_ID)
        assert a == b
        # Returned as a string per repository.create_topic_version's signature.
        assert isinstance(a, str)
        assert UUID(a).version == 5

    def test_topic_version_id_different_source_action_item_produces_different_id(self):
        topic_id = '44444444-4444-4444-8444-444444444444'
        a = _topic_version_id(topic_id, ACTION_ITEM_ID)
        b = _topic_version_id(topic_id, '55555555-5555-4555-8555-555555555555')
        assert a != b


# ---------------------------------------------------------------------------
# Fixtures: mocked executor
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_repository():
    """Mocked ActionItemRepository with the methods TopicExecutor calls."""
    repo = MagicMock()
    repo.create_topic = AsyncMock(return_value={})
    repo.link_action_item_to_topic = AsyncMock(return_value={})
    repo.create_topic_version = AsyncMock(return_value={})
    repo.increment_topic_action_count = AsyncMock(return_value=2)
    repo.get_topic = AsyncMock(return_value={'version': 1})
    repo.update_topic = AsyncMock(return_value={})
    return repo


@pytest.fixture
def mock_openai_create_summary():
    """Mocked OpenAIClient that returns a fixed TopicSummary."""
    openai = MagicMock()
    openai.chat_completion_structured = AsyncMock(
        return_value=TopicSummary(
            summary='A topic summary deterministic for the test.',
            should_update_embedding=False,
        )
    )
    openai.create_embedding = AsyncMock(return_value=[0.01] * 1536)
    return openai


@pytest.fixture
def executor(mock_repository, mock_openai_create_summary):
    return TopicExecutor(
        repository=mock_repository,
        openai_client=mock_openai_create_summary,
        update_summary_on_link=True,
    )


@pytest.fixture
def create_resolution():
    """A canonical CREATE_NEW resolution."""
    return TopicResolutionResult(
        action_item_id=ACTION_ITEM_ID,
        action_item_summary='Prepare Q1 review deck',
        extracted_topic=ExtractedTopic(
            name='Q1 Sales Expansion',
            context='Q1 hiring + pipeline initiative.',
        ),
        decision=TopicDecision.CREATE_NEW,
        topic_id=None,
        confidence=0.91,
        embedding=[0.02] * 1536,
        candidates=[],
    )


# ---------------------------------------------------------------------------
# Integrated executor behavior
# ---------------------------------------------------------------------------


class TestCreateTopicIdempotency:
    """Two calls to ``_create_topic`` with the same resolution produce the same MERGE keys."""

    @pytest.mark.asyncio
    async def test_create_topic_passes_deterministic_topic_id_to_repository(
        self, executor, mock_repository, create_resolution,
    ):
        await executor._create_topic(
            resolution=create_resolution,
            tenant_id=TENANT_ID,
            account_id=ACCOUNT_ID,
            action_item_text='Prepare Q1 review deck',
            owner='Peter',
        )
        topic_call_1 = mock_repository.create_topic.call_args.args[0]
        version_call_1 = mock_repository.create_topic_version.call_args.kwargs['version_id']

        mock_repository.create_topic.reset_mock()
        mock_repository.create_topic_version.reset_mock()
        mock_repository.link_action_item_to_topic.reset_mock()

        await executor._create_topic(
            resolution=create_resolution,
            tenant_id=TENANT_ID,
            account_id=ACCOUNT_ID,
            action_item_text='Prepare Q1 review deck',
            owner='Peter',
        )
        topic_call_2 = mock_repository.create_topic.call_args.args[0]
        version_call_2 = mock_repository.create_topic_version.call_args.kwargs['version_id']

        # The Topic ID must be identical across the two invocations —
        # otherwise the repository MERGE would fan out into two nodes
        # under S10b retry. Same for the TopicVersion ID.
        assert topic_call_1.id == topic_call_2.id
        assert version_call_1 == version_call_2

    @pytest.mark.asyncio
    async def test_create_topic_id_derives_from_canonical_name_and_source(
        self, executor, mock_repository, create_resolution,
    ):
        """The MERGE key is derived from (tenant, account, canonical_name,
        source_action_item_id) — verifying topic.id matches the helper's
        output applied to the same inputs."""
        await executor._create_topic(
            resolution=create_resolution,
            tenant_id=TENANT_ID,
            account_id=ACCOUNT_ID,
            action_item_text='Prepare Q1 review deck',
            owner='Peter',
        )
        topic_a = mock_repository.create_topic.call_args.args[0]

        # Predict the expected topic_id from the helper applied to the
        # canonical name (case-folded) AND the resolution's source
        # action_item_id. Including source in the key preserves legacy
        # parity (Codex R1 absorption).
        expected = _topic_id_for_resolution(
            TENANT_ID, ACCOUNT_ID, topic_a.canonical_name,
            source_action_item_id=create_resolution.action_item_id,
        )
        assert topic_a.id == expected

    @pytest.mark.asyncio
    async def test_create_topic_same_canonical_name_different_source_produces_distinct_topics(
        self, executor, mock_repository, create_resolution,
    ):
        """Legacy parity: two action items in the same batch that both
        resolve to CREATE_NEW with the same canonical_name produce two
        DISTINCT Topic nodes. The prior /process path generated a fresh
        uuid4 per call; the new deterministic-key path must preserve
        that fan-out at the entity level. Codex R1 absorption."""
        # Resolution A — first action item.
        await executor._create_topic(
            resolution=create_resolution,
            tenant_id=TENANT_ID,
            account_id=ACCOUNT_ID,
            action_item_text='Prepare Q1 review deck',
            owner='Peter',
        )
        topic_a = mock_repository.create_topic.call_args.args[0]

        mock_repository.create_topic.reset_mock()
        mock_repository.create_topic_version.reset_mock()
        mock_repository.link_action_item_to_topic.reset_mock()

        # Resolution B — different action item, same canonical_name (the
        # extractor's identical name + S10a's CREATE_NEW decision twice).
        resolution_b = TopicResolutionResult(
            action_item_id='99999999-9999-4999-8999-999999999999',
            action_item_summary=create_resolution.action_item_summary,
            extracted_topic=create_resolution.extracted_topic,
            decision=create_resolution.decision,
            topic_id=None,
            confidence=create_resolution.confidence,
            embedding=create_resolution.embedding,
            candidates=create_resolution.candidates,
        )
        await executor._create_topic(
            resolution=resolution_b,
            tenant_id=TENANT_ID,
            account_id=ACCOUNT_ID,
            action_item_text='Another Q1 prep task',
            owner='Sarah',
        )
        topic_b = mock_repository.create_topic.call_args.args[0]

        # Same canonical_name, but DIFFERENT topic_ids — preserves the
        # legacy two-Topic-nodes-per-batch behavior. Without the source
        # in the key, these would collapse to one node with
        # under-counted action_item_count.
        assert topic_a.canonical_name == topic_b.canonical_name
        assert topic_a.id != topic_b.id


class TestUpdateTopicSummaryIdempotency:
    """``_update_topic_summary`` passes a deterministic version_id under retry."""

    @pytest.mark.asyncio
    async def test_update_topic_summary_passes_deterministic_version_id(
        self, executor, mock_repository, mock_openai_create_summary,
    ):
        # The current summary differs from the LLM's output, so the path
        # reaches create_topic_version (not the no-op early return).
        mock_openai_create_summary.chat_completion_structured = AsyncMock(
            return_value=TopicSummary(
                summary='Updated summary text — different from current.',
                should_update_embedding=False,
            )
        )

        topic_id = '66666666-6666-4666-8666-666666666666'

        await executor._update_topic_summary(
            topic_id=topic_id,
            tenant_id=TENANT_ID,
            current_summary='Original summary.',
            current_name='Q1 Sales Expansion',
            new_count=3,
            action_item_text='New link triggering update',
            action_item_summary='Trigger',
            owner='Peter',
            changed_by_action_item_id=ACTION_ITEM_ID,
        )
        version_id_1 = mock_repository.create_topic_version.call_args.kwargs['version_id']

        mock_repository.create_topic_version.reset_mock()
        mock_repository.update_topic.reset_mock()

        await executor._update_topic_summary(
            topic_id=topic_id,
            tenant_id=TENANT_ID,
            current_summary='Original summary.',
            current_name='Q1 Sales Expansion',
            new_count=3,
            action_item_text='New link triggering update',
            action_item_summary='Trigger',
            owner='Peter',
            changed_by_action_item_id=ACTION_ITEM_ID,
        )
        version_id_2 = mock_repository.create_topic_version.call_args.kwargs['version_id']

        # Two retries of the same step produce the same version_id —
        # the MERGE in repository.create_topic_version no-ops on retry.
        assert version_id_1 == version_id_2

    @pytest.mark.asyncio
    async def test_update_topic_summary_version_id_distinguishes_source_action_item(
        self, executor, mock_repository, mock_openai_create_summary,
    ):
        """Two different action items updating the same topic must produce
        distinct version_ids so both version rows persist."""
        mock_openai_create_summary.chat_completion_structured = AsyncMock(
            return_value=TopicSummary(
                summary='Updated summary text — different from current.',
                should_update_embedding=False,
            )
        )

        topic_id = '77777777-7777-4777-8777-777777777777'
        action_a = ACTION_ITEM_ID
        action_b = str(uuid4())

        await executor._update_topic_summary(
            topic_id=topic_id,
            tenant_id=TENANT_ID,
            current_summary='Original.',
            current_name='Topic',
            new_count=2,
            action_item_text='A',
            action_item_summary='A summary',
            owner='Peter',
            changed_by_action_item_id=action_a,
        )
        version_id_a = mock_repository.create_topic_version.call_args.kwargs['version_id']

        mock_repository.create_topic_version.reset_mock()

        await executor._update_topic_summary(
            topic_id=topic_id,
            tenant_id=TENANT_ID,
            current_summary='Original.',
            current_name='Topic',
            new_count=3,
            action_item_text='B',
            action_item_summary='B summary',
            owner='Peter',
            changed_by_action_item_id=action_b,
        )
        version_id_b = mock_repository.create_topic_version.call_args.kwargs['version_id']

        assert version_id_a != version_id_b
