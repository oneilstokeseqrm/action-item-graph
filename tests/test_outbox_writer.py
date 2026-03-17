"""Tests for the agent outbox writer (pipeline step 11).

Follows the same failure-isolation pattern as test_pipeline_dual_write.py.
"""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from action_item_graph.models.action_item import ActionItem, ActionItemStatus
from action_item_graph.models.envelope import EnvelopeV1, ContentPayload
from action_item_graph.pipeline.pipeline import ActionItemPipeline
from action_item_graph.pipeline.merger import MergeResult


def _make_pipeline(postgres=None):
    """Create a pipeline instance with mocked clients."""
    return ActionItemPipeline(
        openai_client=MagicMock(),
        neo4j_client=MagicMock(),
        postgres_client=postgres,
    )


def _make_action_item(**kwargs):
    defaults = {
        "id": uuid4(),
        "tenant_id": UUID("11111111-1111-4111-8111-111111111111"),
        "action_item_text": "Send proposal by Friday",
        "summary": "Send proposal",
        "owner": "John",
        "status": ActionItemStatus.OPEN,
        "priority_score": 0.75,
        "is_user_owned": False,
    }
    defaults.update(kwargs)
    return ActionItem(**defaults)


def _make_merge_result(action_item_id=None, action="created"):
    """Create a MergeResult dataclass instance."""
    return MergeResult(
        action_item_id=action_item_id or str(uuid4()),
        action=action,
        was_new=(action == "created"),
        version_created=True,
        linked_interaction_id=None,
        details={},
    )


def _make_envelope():
    return EnvelopeV1(
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        user_id="user-1",
        interaction_type="transcript",
        content=ContentPayload(text="test", format="plain"),
        timestamp="2026-03-16T00:00:00Z",
        source="web-mic",
        account_id="22222222-2222-4222-8222-222222222222",
    )


def _make_mock_postgres():
    """Create a mock Postgres client with a working async context manager."""
    mock_conn = AsyncMock()
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin = MagicMock(return_value=mock_cm)
    postgres = MagicMock()
    postgres.engine = mock_engine
    return postgres, mock_conn


@pytest.mark.asyncio
@patch.dict(os.environ, {"ENABLE_AGENT_OUTBOX": "true"})
async def test_outbox_writes_on_success():
    """Normal write: creates one outbox row with correct fields."""
    postgres, mock_conn = _make_mock_postgres()
    pipeline = _make_pipeline(postgres=postgres)

    ai = _make_action_item()
    mr = _make_merge_result(action_item_id=str(ai.id), action="created")

    await pipeline._write_agent_outbox(
        merge_results=[mr],
        action_items=[ai],
        envelope=_make_envelope(),
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="22222222-2222-4222-8222-222222222222",
        interaction_id=str(uuid4()),
    )

    mock_conn.execute.assert_called_once()
    call_args = mock_conn.execute.call_args
    params = call_args[0][1]
    assert params["tenant_id"] == "11111111-1111-4111-8111-111111111111"
    assert params["account_id"] == "22222222-2222-4222-8222-222222222222"
    assert "action_item" in params["dedup_key"]


@pytest.mark.asyncio
@patch.dict(os.environ, {"ENABLE_AGENT_OUTBOX": "true"})
async def test_outbox_skips_when_no_items():
    """No-op when no created or updated items."""
    postgres, mock_conn = _make_mock_postgres()
    pipeline = _make_pipeline(postgres=postgres)

    # MergeResult with 'linked' action — not created or updated
    mr = _make_merge_result(action="linked")

    await pipeline._write_agent_outbox(
        merge_results=[mr],
        action_items=[],
        envelope=_make_envelope(),
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="a-1",
        interaction_id=str(uuid4()),
    )

    mock_conn.execute.assert_not_called()


@pytest.mark.asyncio
@patch.dict(os.environ, {"ENABLE_AGENT_OUTBOX": "false"})
async def test_outbox_skips_when_flag_disabled():
    """No-op when ENABLE_AGENT_OUTBOX is false."""
    postgres, mock_conn = _make_mock_postgres()
    pipeline = _make_pipeline(postgres=postgres)

    ai = _make_action_item()
    mr = _make_merge_result(action_item_id=str(ai.id))

    await pipeline._write_agent_outbox(
        merge_results=[mr],
        action_items=[ai],
        envelope=_make_envelope(),
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="a-1",
        interaction_id=str(uuid4()),
    )

    mock_conn.execute.assert_not_called()


@pytest.mark.asyncio
@patch.dict(os.environ, {"ENABLE_AGENT_OUTBOX": "true"})
async def test_outbox_skips_when_no_postgres():
    """No-op when Postgres client is not configured."""
    pipeline = _make_pipeline(postgres=None)

    ai = _make_action_item()
    mr = _make_merge_result(action_item_id=str(ai.id))

    # Should not raise
    await pipeline._write_agent_outbox(
        merge_results=[mr],
        action_items=[ai],
        envelope=_make_envelope(),
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="a-1",
        interaction_id=str(uuid4()),
    )


@pytest.mark.asyncio
@patch.dict(os.environ, {"ENABLE_AGENT_OUTBOX": "true"})
async def test_outbox_failure_isolation():
    """Postgres error is caught and logged, never raised."""
    postgres, mock_conn = _make_mock_postgres()
    mock_conn.execute = AsyncMock(side_effect=Exception("DB connection failed"))
    pipeline = _make_pipeline(postgres=postgres)

    ai = _make_action_item()
    mr = _make_merge_result(action_item_id=str(ai.id))

    # Should NOT raise — failure is isolated
    await pipeline._write_agent_outbox(
        merge_results=[mr],
        action_items=[ai],
        envelope=_make_envelope(),
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="a-1",
        interaction_id=str(uuid4()),
    )
