"""DBOS step functions for the action-item pipeline workflow.

14 steps total, mapping 1:1 to the step graph in
``docs/plans/2026-05-20-dbos-migration-execution-plan.md`` (lines 189-225):

  S1  ensure_account             fatal-retry × 3
  S2  extraction                 fatal-retry × 3
  S3  consolidation              fatal-retry × 3
  S4  verification               retries_allowed=False, step body fail-open
  S5  owner_resolution           fatal-retry × 3
  S6  create_interaction         fatal-retry × 3
  S7  merge_contacts_to_deal     retries_allowed=False, fail-open
  S8  matching                   fatal-retry × 3
  S9a merging_llm                fatal-retry × 3 (pure LLM)
  S9b merging_persist            fatal-retry × 3 (pure Neo4j write)
  S10a topic_resolution_llm      fatal-retry × 3 (pure LLM)
  S10b topic_resolution_persist  fatal-retry × 3 (pure Neo4j write)
  S13 postgres_dual_write        retries_allowed=False, fail-open
  S14 agent_outbox               retries_allowed=False, fail-open

Steps construct their own sub-objects (pipeline, OwnerPreResolver, etc.)
from ``get_clients()`` rather than capturing locals from the workflow
function. DBOS replay only re-runs the specific step; locals from the
workflow body don't survive replay (Peter Phase B clarification, 2026-05-20).

Fail-open ``except Exception:`` is intentional migration-parity with the
existing pipeline.py:381-395 try/except around ``merge_contacts_to_deal``
and ``_dual_write_postgres`` / ``_write_agent_outbox``. Tightening to
specific exception classes is a separate follow-up; semantic parity is
preserved for now.

The S9a/S10a steps return checkpointed payloads. Open #24 analysis caps
worst-case S9a output at ~64 KB (8 items × ~8 KB MergedActionItem with
embedding). S10a is similar shape. Both well under DBOS JSONB row limit.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from dbos import DBOS

from action_item_graph.logging import get_logger
from action_item_graph.models.entities import Interaction
from action_item_graph.models.envelope import EnvelopeV1
from action_item_graph.pipeline.merger import ActionItemMerger
from action_item_graph.pipeline.owner_resolver import OwnerPreResolver
from action_item_graph.pipeline.pipeline import ActionItemPipeline
from action_item_graph.pipeline.topic_executor import (
    TopicExecutionResult,
    TopicExecutor,
)
from action_item_graph.pipeline.topic_resolver import TopicResolver
from action_item_graph.prompts.extract_action_items import ExtractedTopic
from action_item_graph.repository import ActionItemRepository
from action_item_graph.workflows._runtime import WorkflowClients, get_clients
from action_item_graph.workflows._serialization import (
    extraction_from_dict,
    extraction_to_dict,
    match_result_from_dict,
    match_result_to_dict,
    merge_result_from_dict,
    merge_result_to_dict,
    topic_execution_to_dict,
    topic_resolution_from_dict,
    topic_resolution_to_dict,
)
from shared.contact_ops import merge_contacts_to_deal

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pipeline(clients: WorkflowClients) -> ActionItemPipeline:
    """Construct a per-step ActionItemPipeline from the client registry.

    Cheap (no I/O). Per-step construction is the DBOS-correct pattern —
    steps cannot capture pipeline instances from the workflow function
    scope because that breaks replay determinism.
    """
    return ActionItemPipeline(
        openai_client=clients.openai,
        neo4j_client=clients.neo4j,
        postgres_client=clients.postgres,
    )


# ---------------------------------------------------------------------------
# S1 — ensure_account
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
async def ensure_account_step(envelope_dict: dict[str, Any]) -> None:
    """S1: Ensure the Account node exists in Neo4j (MERGE-keyed, idempotent).

    Pre-condition: envelope.account_id is non-empty. The workflow function
    validates this BEFORE calling the step — raising inside a retryable
    step would cause DBOS to retry 3 times and wrap the eventual exception
    in ``DBOSMaxStepRetriesExceeded``, breaking the legacy fail-fast
    ``ValidationError`` semantics (Codex Round 3 absorption).
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    clients = get_clients()
    repository = ActionItemRepository(clients.neo4j)
    await repository.ensure_account(
        tenant_id=envelope.tenant_id,
        account_id=envelope.account_id or '',
        name=envelope.extras.get('account_name') if envelope.extras else None,
    )


# ---------------------------------------------------------------------------
# S2 — extraction
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
async def extraction_step(envelope_dict: dict[str, Any]) -> dict[str, Any]:
    """S2: Extract action items via F-CoT LLM. Returns ExtractionOutput as dict."""
    envelope = EnvelopeV1.model_validate(envelope_dict)
    clients = get_clients()
    pipeline = _build_pipeline(clients)
    extraction = await pipeline.extractor.extract_from_envelope(envelope)
    return extraction_to_dict(extraction)


# ---------------------------------------------------------------------------
# S3 — consolidation
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
async def consolidation_step(
    extraction_dict: dict[str, Any],
) -> dict[str, Any]:
    """S3: Consolidate near-duplicates within the batch via LLM clustering.

    Returns a dict with ``extraction`` (new ExtractionOutput) and
    ``items_consolidated`` (count of items merged into clusters).
    """
    extraction = extraction_from_dict(extraction_dict)
    clients = get_clients()
    pipeline = _build_pipeline(clients)
    new_extraction, items_consolidated = await pipeline.consolidator.consolidate(
        extraction,
    )
    return {
        'extraction': extraction_to_dict(new_extraction),
        'items_consolidated': items_consolidated,
    }


# ---------------------------------------------------------------------------
# S4 — verification (FAIL-OPEN)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def verification_step(
    extraction_dict: dict[str, Any],
) -> dict[str, Any]:
    """S4: LLM-as-Judge verification. FAIL-OPEN per Open #2.

    The broad ``except Exception:`` matches the existing
    ``verifier.verify_batch`` behavior (verifier.py:99-107) which itself
    catches LLM errors and returns ``(extraction, 0, [...])``. We wrap
    a second layer here so that even a programming bug in the verifier
    (TypeError, ImportError) doesn't sink the workflow — the pipeline
    proceeds with the unverified extraction.

    Broad-catch is intentional migration-parity; not aspirational
    hardening. Tightening to specific exception classes is a separate
    follow-up.
    """
    extraction = extraction_from_dict(extraction_dict)
    clients = get_clients()
    pipeline = _build_pipeline(clients)
    try:
        new_extraction, items_rejected, rejection_reasons = (
            await pipeline.verifier.verify_batch(extraction)
        )
        return {
            'extraction': extraction_to_dict(new_extraction),
            'items_rejected': items_rejected,
            'rejection_reasons': rejection_reasons,
            'status': 'ok',
        }
    except Exception as e:  # noqa: BLE001 — intentional migration-parity (Open #2)
        logger.warning(
            'verification_step.failed_fail_open',
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            'extraction': extraction_to_dict(extraction),
            'items_rejected': 0,
            'rejection_reasons': [f'Verification failed: {e}'],
            'status': 'skipped',
        }


# ---------------------------------------------------------------------------
# S5 — owner_resolution
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
async def owner_resolution_step(
    extraction_dict: dict[str, Any],
    envelope_dict: dict[str, Any],
) -> dict[str, Any]:
    """S5: Pre-resolve owner names against the account-scoped cache.

    Returns a dict with ``extraction`` (new ExtractionOutput with
    resolved owners — Open #1, no in-place mutation) and ``contact_map``
    (owner-name → contact-id for downstream linking).
    """
    extraction = extraction_from_dict(extraction_dict)
    envelope = EnvelopeV1.model_validate(envelope_dict)
    clients = get_clients()
    repository = ActionItemRepository(clients.neo4j)
    owner_resolver = OwnerPreResolver(repository, clients.openai)
    await owner_resolver.load_cache(
        envelope.tenant_id,
        envelope.account_id or '',
        contacts=envelope.contacts,
    )
    # Discard method_counts here; the caller only needs the resolved items.
    resolved_items = (
        await owner_resolver.resolve_batch(extraction.action_items)
    )[0]

    new_extraction = extraction_from_dict(extraction_dict)
    new_extraction.action_items = resolved_items

    contact_map: dict[str, str] = {}
    for ai in resolved_items:
        cid = owner_resolver.get_contact_id(ai.owner)
        if cid:
            contact_map[ai.owner] = cid

    return {
        'extraction': extraction_to_dict(new_extraction),
        'contact_map': contact_map,
    }


# ---------------------------------------------------------------------------
# S6 — create_interaction
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
async def create_interaction_step(interaction_dict: dict[str, Any]) -> None:
    """S6: Create Interaction node in Neo4j (MERGE-keyed, idempotent)."""
    interaction = Interaction.model_validate(interaction_dict)
    clients = get_clients()
    repository = ActionItemRepository(clients.neo4j)
    await repository.create_interaction(interaction)


# ---------------------------------------------------------------------------
# S7 — merge_contacts_to_deal (FAIL-OPEN)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def merge_contacts_to_deal_step(envelope_dict: dict[str, Any]) -> dict[str, Any]:
    """S7: Side-branch Neo4j MERGE for Contact -[:ENGAGED_ON]-> Deal.

    FAIL-OPEN per Open #2 — matches existing pipeline.py:381-395 broad
    try/except. Conditional: only runs if envelope has both
    ``opportunity_id`` and at least one contact with a contact_id.

    Returns ``{'status': 'ok' | 'skipped' | 'no_op', 'count': N}``.
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    if not envelope.opportunity_id or not envelope.contacts:
        return {'status': 'no_op', 'count': 0}
    contact_ids = [c['contact_id'] for c in envelope.contacts if c.get('contact_id')]
    if not contact_ids:
        return {'status': 'no_op', 'count': 0}
    clients = get_clients()
    try:
        await merge_contacts_to_deal(
            clients.neo4j,
            str(envelope.tenant_id),
            contact_ids,
            envelope.opportunity_id,
            source='action_item_pipeline',
        )
        return {'status': 'ok', 'count': len(contact_ids)}
    except Exception as e:  # noqa: BLE001 — intentional migration-parity (Open #2)
        logger.warning(
            'merge_contacts_to_deal_step.failed_fail_open',
            error=str(e),
            error_type=type(e).__name__,
        )
        return {'status': 'skipped', 'count': 0, 'error': str(e)}


# ---------------------------------------------------------------------------
# S8 — matching
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
async def matching_step(
    extraction_dict: dict[str, Any],
    envelope_dict: dict[str, Any],
) -> dict[str, Any]:
    """S8: LLM + Neo4j-read matching of extracted items against existing.

    Pipeline.py's ``_match_extractions`` helper already separates filtered
    items from match results; we serialize both via the dict round-trip.
    """
    extraction = extraction_from_dict(extraction_dict)
    envelope = EnvelopeV1.model_validate(envelope_dict)
    clients = get_clients()
    pipeline = _build_pipeline(clients)
    match_results, filtered_action_items = await pipeline._match_extractions(
        extraction=extraction,
        tenant_id=envelope.tenant_id,
        account_id=envelope.account_id or '',
    )
    return {
        'match_results': [match_result_to_dict(m) for m in match_results],
        'filtered_action_items': [
            ai.model_dump(mode='json') for ai in filtered_action_items
        ],
    }


# ---------------------------------------------------------------------------
# S9a — merging_llm (PURE LLM)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
async def merging_llm_step(
    match_results_list: list[dict[str, Any]],
) -> list[dict[str, Any] | None]:
    """S9a: Pure-LLM phase of merging.

    For each match_result whose decision is ``merge``, call
    ``ActionItemMerger.construct_merged_action_item_llm`` to produce the
    LLM-synthesized merge data (no Neo4j writes). Returns a list aligned
    1:1 with ``match_results_list`` — ``None`` for matches that don't
    need LLM merging (create_new / update_status / link_related paths).

    Codex #10 absorption: separating the LLM from the persist step
    means a retry of the persist step doesn't re-run the LLM and risk
    divergent ``MergedActionItem`` outputs.

    Note (Phase B instrumentation): under normal traffic this step's
    invocation count will be lower than S9b's because only the "merge"
    decision branch triggers LLM work. Observability dashboards should
    not treat the asymmetry as a bug.
    """
    clients = get_clients()
    merger = ActionItemMerger(clients.neo4j, clients.openai)
    outputs: list[dict[str, Any] | None] = []
    for m_dict in match_results_list:
        m = match_result_from_dict(m_dict)
        if m.best_match is None:
            outputs.append(None)
            continue
        _, decision = m.best_match
        if decision.merge_recommendation != 'merge':
            outputs.append(None)
            continue
        candidate, _ = m.best_match
        llm_result = await merger.construct_merged_action_item_llm(
            existing_props=candidate.node_properties,
            extraction=m.extracted_item,
        )
        outputs.append(llm_result)
    return outputs


# ---------------------------------------------------------------------------
# S9b — merging_persist (PURE NEO4J WRITE)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
async def merging_persist_step(
    match_results_list: list[dict[str, Any]],
    filtered_action_items_list: list[dict[str, Any]],
    llm_results_list: list[dict[str, Any] | None],
    interaction_dict: dict[str, Any],
    contact_map: dict[str, str],
) -> list[dict[str, Any]]:
    """S9b: Pure-Neo4j-write phase of merging.

    For each match_result, route to the appropriate persist path:
      - ``best_match is None`` → ``_create_new``
      - ``decision == 'update_status'`` → ``_update_status``
      - ``decision == 'merge'`` → ``persist_merged_action_item_neo4j`` with the
        pre-computed LLM result from S9a
      - ``decision == 'link_related'`` → ``_create_and_link``
    """
    from action_item_graph.models.action_item import ActionItem

    clients = get_clients()
    merger = ActionItemMerger(clients.neo4j, clients.openai)
    interaction = Interaction.model_validate(interaction_dict)

    # Defensive length check (Codex Round 1 γ): zip() truncates silently if
    # the three lists drift; a caller bug would mask itself rather than fail
    # loudly. S9a + matching produce these three lists from the same
    # match_results enumeration, so any length skew is a real bug worth
    # asserting early.
    if not (
        len(match_results_list)
        == len(filtered_action_items_list)
        == len(llm_results_list)
    ):
        raise RuntimeError(
            f'merging_persist_step: list length mismatch — '
            f'match_results={len(match_results_list)}, '
            f'filtered_action_items={len(filtered_action_items_list)}, '
            f'llm_results={len(llm_results_list)}'
        )

    merge_results: list[dict[str, Any]] = []
    for m_dict, ai_dict, llm_result in zip(
        match_results_list, filtered_action_items_list, llm_results_list
    ):
        m = match_result_from_dict(m_dict)
        action_item = ActionItem.model_validate(ai_dict)
        contact_id = contact_map.get(action_item.owner)

        if m.best_match is None:
            result = await merger._create_new(
                action_item=action_item,
                interaction=interaction,
                contact_id=contact_id,
            )
        else:
            candidate, decision = m.best_match
            if decision.merge_recommendation == 'update_status':
                result = await merger._update_status(
                    existing_id=candidate.action_item_id,
                    extraction=m.extracted_item,
                    interaction=interaction,
                    action_item=action_item,
                )
            elif decision.merge_recommendation == 'merge':
                if llm_result is None:
                    # Defensive — S9a should always produce a payload for 'merge'.
                    raise RuntimeError(
                        f'merging_persist_step: missing S9a payload for merge '
                        f'decision on action_item_id={candidate.action_item_id}'
                    )
                # Pass the ExtractedActionItem so create_version_snapshot
                # can disambiguate two distinct merges of the same
                # existing item from the same interaction (Codex B-2 R2).
                result = await merger.persist_merged_action_item_neo4j(
                    llm_result=llm_result,
                    existing_id=candidate.action_item_id,
                    existing_props=candidate.node_properties,
                    interaction=interaction,
                    action_item=action_item,
                    extraction=m.extracted_item,
                )
            elif decision.merge_recommendation == 'link_related':
                result = await merger._create_and_link(
                    related_to_id=candidate.action_item_id,
                    action_item=action_item,
                    interaction=interaction,
                    contact_id=contact_id,
                )
            else:
                # Fallback: treat as create_new
                result = await merger._create_new(
                    action_item=action_item,
                    interaction=interaction,
                    contact_id=contact_id,
                )
        merge_results.append(merge_result_to_dict(result))
    return merge_results


# ---------------------------------------------------------------------------
# S10a — topic_resolution_llm (PURE LLM)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
async def topic_resolution_llm_step(
    match_results_list: list[dict[str, Any]],
    merge_results_list: list[dict[str, Any]],
    filtered_action_items_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """S10a: For each merged ActionItem, run topic resolution to produce a
    TopicResolutionResult. Read-only LLM step: it calls the LLM AND reads
    candidates from Neo4j (no writes). Codex Round 1 absorption: this is
    NOT a "pure LLM" step — Neo4j reads can observe state that drifts
    between retries. Acceptable because once S10a succeeds, DBOS caches
    its output and subsequent step retries don't re-observe.

    Topic resolution is structurally split in the legacy pipeline:
    ``TopicResolver`` decides (LLM + Neo4j read), ``TopicExecutor``
    writes (Neo4j MERGE). S10a wraps the decider, S10b wraps the executor.

    Each returned dict carries an ``_action_item_context`` envelope with
    ``action_item_text`` and ``owner`` from the upstream merged
    ActionItem so S10b can pass them into ``TopicExecutor.execute_batch``
    as the legacy path does (Codex Round 1 HIGH #1 — without these,
    the LLM-driven topic summary prompts diverge from /process).
    """
    from action_item_graph.models.action_item import ActionItem

    clients = get_clients()
    resolver = TopicResolver(clients.neo4j, clients.openai)

    items_to_resolve: list[
        tuple[ExtractedTopic, str, str, UUID, str, str, str]
    ] = []
    for m_dict, mr_dict, ai_dict in zip(
        match_results_list, merge_results_list, filtered_action_items_list
    ):
        m = match_result_from_dict(m_dict)
        merge_result = merge_result_from_dict(mr_dict)
        action_item = ActionItem.model_validate(ai_dict)
        # Only resolve topics for items that have an extracted topic
        extracted = m.extracted_item
        if not extracted.topic:
            continue
        # SOURCE OF TRUTH for the topic-summary LLM prompt: the raw
        # ExtractedActionItem fields (extracted.summary, extracted.owner,
        # extracted.action_item_text). The legacy pipeline.py:_process_topics
        # (lines 870, 889, 899-900) passes these exact extracted fields, NOT
        # the post-S5 ActionItem fields. After S5 owner_resolution, the
        # ActionItem's owner may be normalized (e.g., "Peter" -> "Peter
        # O'Neill") and the topic-summary prompt would see a different
        # input string in the workflow path vs legacy. Codex Round 2
        # caught this divergence; use ExtractedActionItem for parity.
        items_to_resolve.append((
            extracted.topic,
            merge_result.action_item_id,
            extracted.summary,
            action_item.tenant_id,
            action_item.account_id or '',
            extracted.action_item_text,
            extracted.owner,
        ))

    if not items_to_resolve:
        return []

    results: list[dict[str, Any]] = []
    for et, ai_id, summary, tenant_id, account_id, ai_text, owner in items_to_resolve:
        topic_result = await resolver.resolve_topic(
            extracted_topic=et,
            action_item_id=ai_id,
            action_item_summary=summary,
            tenant_id=tenant_id,
            account_id=account_id,
        )
        result_dict = topic_resolution_to_dict(topic_result)
        # Envelope the action-item context S10b needs. Underscore prefix
        # marks it as private-to-the-step-boundary metadata, not part of
        # TopicResolutionResult's schema.
        result_dict['_action_item_context'] = {
            'action_item_text': ai_text,
            'owner': owner,
        }
        results.append(result_dict)
    return results


# ---------------------------------------------------------------------------
# S10b — topic_resolution_persist (PURE NEO4J WRITE)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
async def topic_resolution_persist_step(
    topic_resolution_results_list: list[dict[str, Any]],
    envelope_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """S10b: Persist topic resolution outcomes — create new Topic nodes,
    link ActionItems via HAS_TOPIC, update existing Topic summaries.
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    clients = get_clients()
    repository = ActionItemRepository(clients.neo4j)
    executor = TopicExecutor(repository, clients.openai)

    if not topic_resolution_results_list:
        return []

    # TopicExecutor.execute_batch expects (resolution, action_item_text,
    # owner) tuples. S10a carries the action-item context through the
    # ``_action_item_context`` envelope so the LLM-driven summary
    # prompts (_generate_initial_summary / _update_topic_summary) see
    # the same ai_text + owner the legacy pipeline.py:_process_topics
    # path passed (Codex Round 1 HIGH #1).
    batch: list[tuple] = []
    for d in topic_resolution_results_list:
        ctx = d.get('_action_item_context') or {}
        ai_text = ctx.get('action_item_text', '')
        owner = ctx.get('owner', '')
        tr = topic_resolution_from_dict(d)
        batch.append((tr, ai_text, owner))

    exec_results = await executor.execute_batch(
        resolutions=batch,
        tenant_id=envelope.tenant_id,
        account_id=envelope.account_id or '',
    )
    return [topic_execution_to_dict(r) for r in exec_results]


# ---------------------------------------------------------------------------
# S13 — postgres_dual_write (FAIL-OPEN)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def postgres_dual_write_step(
    merge_results_list: list[dict[str, Any]],
    filtered_action_items_list: list[dict[str, Any]],
    interaction_dict: dict[str, Any],
    topic_execution_results_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """S13: Dual-write to Postgres for read-model projection. FAIL-OPEN.

    Returns ``{'status': 'ok' | 'skipped' | 'no_op', 'rows_written': N}``.
    Status 'no_op' covers the case where no Postgres client is registered
    (postgres dual-write is optional infrastructure).
    """
    from action_item_graph.models.action_item import ActionItem

    clients = get_clients()
    if clients.postgres is None:
        return {'status': 'no_op', 'rows_written': 0}

    interaction = Interaction.model_validate(interaction_dict)
    action_items = [
        ActionItem.model_validate(ai) for ai in filtered_action_items_list
    ]
    merge_results = [merge_result_from_dict(d) for d in merge_results_list]
    topic_results = [
        TopicExecutionResult(**d) for d in topic_execution_results_list
    ]
    pipeline = _build_pipeline(clients)
    try:
        await pipeline._dual_write_postgres(
            merge_results=merge_results,
            action_items=action_items,
            interaction=interaction,
            topic_results=topic_results,
        )
        return {'status': 'ok', 'rows_written': len(merge_results)}
    except Exception as e:  # noqa: BLE001 — intentional migration-parity (Open #2/3)
        logger.warning(
            'postgres_dual_write_step.failed_fail_open',
            error=str(e),
            error_type=type(e).__name__,
        )
        return {'status': 'skipped', 'rows_written': 0, 'error': str(e)}


# ---------------------------------------------------------------------------
# S14 — agent_outbox (FAIL-OPEN)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def agent_outbox_step(
    merge_results_list: list[dict[str, Any]],
    filtered_action_items_list: list[dict[str, Any]],
    envelope_dict: dict[str, Any],
    interaction_dict: dict[str, Any],
) -> dict[str, Any]:
    """S14: Write to Postgres ``agent_outbox`` for downstream LLM consumers.

    FAIL-OPEN; non-blocking on workflow success.

    Takes ``interaction_dict`` as an explicit input so the interaction_id
    used here matches what extraction produced (not what the envelope
    carried). Extraction may generate an interaction_id locally if the
    envelope had none; using ``envelope.interaction_id`` here would
    silently no-op the outbox write for those envelopes because
    ``_write_agent_outbox`` casts the empty string to UUID and the dedup
    key becomes meaningless (Codex Round 1 HIGH #2).
    """
    from action_item_graph.models.action_item import ActionItem

    envelope = EnvelopeV1.model_validate(envelope_dict)
    interaction = Interaction.model_validate(interaction_dict)
    clients = get_clients()
    if clients.postgres is None:
        return {'status': 'no_op', 'rows_written': 0}

    action_items = [
        ActionItem.model_validate(ai) for ai in filtered_action_items_list
    ]
    merge_results = [merge_result_from_dict(d) for d in merge_results_list]
    pipeline = _build_pipeline(clients)
    try:
        await pipeline._write_agent_outbox(
            merge_results=merge_results,
            action_items=action_items,
            envelope=envelope,
            tenant_id=envelope.tenant_id,
            account_id=envelope.account_id or '',
            interaction_id=str(interaction.interaction_id),
        )
        return {'status': 'ok', 'rows_written': len(merge_results)}
    except Exception as e:  # noqa: BLE001 — intentional migration-parity (Open #2/3)
        logger.warning(
            'agent_outbox_step.failed_fail_open',
            error=str(e),
            error_type=type(e).__name__,
        )
        return {'status': 'skipped', 'rows_written': 0, 'error': str(e)}
