"""DBOS step functions for the deal pipeline workflow.

9 steps total, mapping to the step graph in
``docs/plans/2026-05-20-dbos-migration-execution-plan.md`` (lines 227-256):

  D1 validate_envelope           [pure, validated in workflow body — not a step]
  D2 verify_account              fatal-retry × 3
  D3 ensure_interaction          fatal-retry × 3 (conditional)
  D4 merge_contacts_to_deal_base retries_allowed=False, fail-open
  D5 fetch_existing_deal         fatal-retry × 3 (conditional)
  D6 extraction                  fatal-retry × 3
  D7 match_merge_loop            retries_allowed=False, fail-open per deal
  D8 enrich_interaction          retries_allowed=False, fail-open
  D9 postgres_dual_write         retries_allowed=False, fail-open

D1 validation lives in ``deal_workflow()`` body, NOT as a @DBOS.step,
per [[pattern-dbos-workflow-parity-rules]] Rule 1: a ValidationError
raised inside a retryable step would be retried 3 times before being
wrapped in DBOSMaxStepRetriesExceeded — breaking the legacy
fail-fast semantics.

D7 is intentionally a single @DBOS.step rather than per-deal steps. The
plan's locked decision (D3 row): "9 steps + inner merger.merge_deal
function refactor — single DBOS step boundary preserved because
per-deal failures are fail-open". The inner refactor lives in
``deal_graph/pipeline/merger.py`` (construct_merged_deal_llm +
persist_merged_deal_neo4j). At V1, the loop runs serially inside one
step; a future enhancement could promote to per-deal steps if
operational data shows that per-deal retry granularity is valuable.

Step bodies follow Phase B-1 patterns:
- Rule 2: LLM-prompt source is ExtractedDeal (upstream), not any
  post-match/post-enrich derivative.
- Rule 3: JSON serialization at boundaries via workflows/_serialization.
- Rule 4: zip()-with-parallel-arrays uses explicit length assertions.
- Rule 5: identifier source-of-truth is the downstream-produced value
  (e.g., merge_result.opportunity_id for ENGAGED_ON enrichment), not
  the upstream-carried envelope.opportunity_id when those may diverge.
"""

from __future__ import annotations

from typing import Any

from dbos import DBOS

from action_item_graph.logging import get_logger
from action_item_graph.models.envelope import EnvelopeV1
from action_item_graph.workflows._runtime import get_clients
from deal_graph.clients.event_publisher import publish_deal_processed
from deal_graph.pipeline.extractor import DealExtractor
from deal_graph.pipeline.matcher import DealMatcher
from deal_graph.pipeline.merger import DealMerger
from deal_graph.pipeline.pipeline import DealPipeline
from deal_graph.workflows._serialization import (
    deal_extraction_result_from_dict,
    deal_extraction_result_to_dict,
    deal_merge_result_to_dict,
)
from shared.contact_ops import (
    enrich_engaged_on_role,
    match_name_to_contact,
    merge_contacts_to_deal,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_deal_pipeline() -> DealPipeline:
    """Construct a per-step DealPipeline from the client registry.

    Cheap (no I/O). Per-step construction per the DBOS-correct pattern
    (steps cannot capture workflow-function locals).
    """
    clients = get_clients()
    return DealPipeline(
        neo4j_client=clients.deal_neo4j,
        openai_client=clients.openai,
        postgres_client=clients.postgres,
    )


# ---------------------------------------------------------------------------
# D2 — verify_account
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
async def verify_account_step(envelope_dict: dict[str, Any]) -> None:
    """D2: Verify the Account node exists in Neo4j.

    Legacy pipeline.py:284 calls ``repository.verify_account(tenant_id,
    account_id)``. Account presence is a precondition for any deal
    write; if the upstream skeleton hasn't materialized the account yet,
    we surface a clear failure rather than writing to ``Account('')``.

    Preconditions (validated in workflow body): envelope.account_id is
    non-empty.
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    pipeline = _build_deal_pipeline()
    await pipeline.repository.verify_account(envelope.tenant_id, envelope.account_id or '')


# ---------------------------------------------------------------------------
# D3 — ensure_interaction
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
async def ensure_interaction_step(envelope_dict: dict[str, Any]) -> dict[str, Any]:
    """D3: MERGE the Interaction node. Conditional on envelope.interaction_id.

    Returns ``{"status": "ok" | "no_op", "interaction_id": <str | None>}``.
    The downstream D8 enrich_interaction step uses the returned
    interaction_id (Rule 5: downstream-produced identifier is the
    authority — not envelope_dict.interaction_id directly because the
    workflow body needs to know whether the step actually ran).
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    if not envelope.interaction_id:
        return {'status': 'no_op', 'interaction_id': None}
    pipeline = _build_deal_pipeline()
    await pipeline.repository.ensure_interaction(
        tenant_id=envelope.tenant_id,
        interaction_id=str(envelope.interaction_id),
        content_text=envelope.content.text,
        interaction_type=envelope.interaction_type.value,
        timestamp=envelope.timestamp,
        source=envelope.source.value,
        trace_id=envelope.trace_id,
    )
    return {'status': 'ok', 'interaction_id': str(envelope.interaction_id)}


# ---------------------------------------------------------------------------
# D4 — merge_contacts_to_deal_base (FAIL-OPEN)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def merge_contacts_to_deal_base_step(
    envelope_dict: dict[str, Any],
) -> dict[str, Any]:
    """D4: Side-branch ENGAGED_ON MERGE for the Case-A targeted path.

    Conditional: only when envelope has BOTH ``opportunity_id`` and
    contacts with contact_ids. FAIL-OPEN per Rule 5 / legacy
    pipeline.py:306-318 broad try/except.

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
            clients.deal_neo4j,
            str(envelope.tenant_id),
            contact_ids,
            envelope.opportunity_id,
            source='deal_pipeline',
        )
        return {'status': 'ok', 'count': len(contact_ids)}
    except Exception as e:  # noqa: BLE001 — intentional migration-parity (Open #2)
        logger.warning(
            'merge_contacts_to_deal_base_step.failed_fail_open',
            error=str(e),
            error_type=type(e).__name__,
        )
        return {'status': 'skipped', 'count': 0, 'error': str(e)}


# ---------------------------------------------------------------------------
# D5 — fetch_existing_deal
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
async def fetch_existing_deal_step(
    envelope_dict: dict[str, Any],
) -> dict[str, Any] | None:
    """D5: For Case-A flows, fetch existing deal context from Neo4j.

    Returns the deal node dict if found, ``None`` if not found or no
    opportunity_id on envelope (the workflow body falls through to
    discovery-mode extraction in that case — matches legacy pipeline.py
    fall-through at lines 332-341).
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    if not envelope.opportunity_id:
        return None
    pipeline = _build_deal_pipeline()
    return await pipeline.repository.get_deal(
        tenant_id=envelope.tenant_id,
        opportunity_id=envelope.opportunity_id,
    )


# ---------------------------------------------------------------------------
# D6 — extraction (LLM)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2.0, backoff_rate=2.0)
async def deal_extraction_step(
    envelope_dict: dict[str, Any],
    existing_deal: dict[str, Any] | None,
) -> dict[str, Any]:
    """D6: Extract deals from the transcript via LLM.

    Returns a JSON-safe envelope bundling the DealExtractionResult with
    the parallel embeddings list, so D7 has aligned per-deal embedding
    access without re-extracting.

    LLM-prompt source-of-truth (Rule 2): the extractor consumes
    ``envelope`` and ``existing_deal`` directly — no post-transformation
    derivatives at this layer. Parity with pipeline.py:346-354.
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    clients = get_clients()
    extractor = DealExtractor(clients.openai)
    extraction_result, embeddings = await extractor.extract_from_envelope(
        envelope=envelope,
        existing_deal=existing_deal,
    )
    return deal_extraction_result_to_dict(extraction_result, embeddings)


# ---------------------------------------------------------------------------
# D7 — match_merge_loop (FAIL-OPEN per deal)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def match_merge_loop_step(
    envelope_dict: dict[str, Any],
    deal_extraction_dict: dict[str, Any],
) -> dict[str, Any]:
    """D7: Per-deal match + merge loop. Single DBOS step boundary.

    Locked architectural decision (plan D3 row): "9 steps + inner
    merger.merge_deal function refactor — single DBOS step boundary
    preserved because per-deal failures are fail-open". The inner
    function-level split lives in deal_graph/pipeline/merger.py
    (``construct_merged_deal_llm`` + ``persist_merged_deal_neo4j``).

    Per-deal exceptions are accumulated into the returned ``errors``
    list rather than aborting the loop. ENGAGED_ON enrichment for the
    discovery (no upstream opportunity_id) path also runs here, with
    its own per-deal broad try/except — Rule 5 applies: enrichment
    uses ``merge_result.opportunity_id`` (the downstream-produced
    canonical ID), not envelope.opportunity_id.

    Returns ``{
      'merge_results': [DealMergeResult-dict, ...],
      'deals_created': [opportunity_id, ...],
      'deals_merged': [opportunity_id, ...],
      'errors': [str, ...]
    }``.
    """
    envelope = EnvelopeV1.model_validate(envelope_dict)
    extraction_result, embeddings = deal_extraction_result_from_dict(
        deal_extraction_dict
    )

    if not extraction_result.has_deals or not extraction_result.deals:
        return {
            'merge_results': [],
            'deals_created': [],
            'deals_merged': [],
            'errors': [],
        }

    # Length assertion per Rule 4: embeddings and deals are produced from
    # the same extraction enumeration; a length mismatch is a real bug
    # the extractor injected upstream.
    if len(extraction_result.deals) != len(embeddings):
        raise RuntimeError(
            f'match_merge_loop_step: deals/embeddings length mismatch — '
            f'deals={len(extraction_result.deals)}, '
            f'embeddings={len(embeddings)}'
        )

    clients = get_clients()
    matcher = DealMatcher(clients.deal_neo4j, clients.openai)
    merger = DealMerger(clients.deal_neo4j, clients.openai)

    merge_results: list[dict[str, Any]] = []
    deals_created: list[str] = []
    deals_merged: list[str] = []
    errors: list[str] = []

    for i, extracted_deal in enumerate(extraction_result.deals):
        embedding = embeddings[i]
        try:
            # D7.a Match
            match_result = await matcher.find_matches(
                extracted_deal=extracted_deal,
                embedding=embedding,
                tenant_id=envelope.tenant_id,
                account_id=envelope.account_id or '',
            )
            # D7.b Merge — calls _merge_existing internally which is now
            # split into construct_merged_deal_llm + persist_merged_deal_neo4j.
            merge_result = await merger.merge_deal(
                match_result=match_result,
                tenant_id=envelope.tenant_id,
                account_id=envelope.account_id,
                source_interaction_id=envelope.interaction_id,
            )
            merge_results.append(deal_merge_result_to_dict(merge_result))
            if merge_result.action == 'created':
                deals_created.append(merge_result.opportunity_id)
            elif merge_result.action == 'merged':
                deals_merged.append(merge_result.opportunity_id)

            # D7.c ENGAGED_ON base + role enrichment. Rule 5: use the
            # downstream-produced opportunity_id (merge_result), not the
            # upstream envelope hint.
            opp_id = merge_result.opportunity_id
            if envelope.contacts and opp_id:
                try:
                    # For discovery deals (no upstream opportunity_id),
                    # base ENGAGED_ON wasn't created in D4. MERGE here.
                    if not envelope.opportunity_id:
                        cids = [
                            c['contact_id']
                            for c in envelope.contacts
                            if c.get('contact_id')
                        ]
                        await merge_contacts_to_deal(
                            clients.deal_neo4j,
                            str(envelope.tenant_id),
                            cids,
                            opp_id,
                            source='deal_pipeline',
                        )

                    # Role enrichment from MEDDIC champion / economic_buyer
                    pipeline = _build_deal_pipeline()
                    deal_node = await pipeline.repository.get_deal(
                        tenant_id=envelope.tenant_id,
                        opportunity_id=opp_id,
                    )
                    if deal_node:
                        for field_name, role_label in [
                            ('meddic_champion', 'champion'),
                            ('meddic_economic_buyer', 'economic_buyer'),
                        ]:
                            role_name = deal_node.get(field_name)
                            if role_name:
                                matched = match_name_to_contact(
                                    role_name, envelope.contacts
                                )
                                if matched:
                                    await enrich_engaged_on_role(
                                        clients.deal_neo4j,
                                        str(envelope.tenant_id),
                                        matched['contact_id'],
                                        opp_id,
                                        role_label,
                                        0.9,
                                    )
                except Exception as e:  # noqa: BLE001 — intentional parity
                    logger.warning(
                        'match_merge_loop_step.engaged_on_enrich_failed',
                        opportunity_id=opp_id,
                        error=str(e),
                    )
        except Exception as exc:  # noqa: BLE001 — intentional fail-open-per-deal
            error_msg = (
                f'Deal "{extracted_deal.opportunity_name}" '
                f'(index {i}): {exc}'
            )
            logger.error(
                'match_merge_loop_step.deal_failed',
                deal_index=i,
                deal_name=extracted_deal.opportunity_name,
                error=str(exc),
            )
            errors.append(error_msg)

    return {
        'merge_results': merge_results,
        'deals_created': deals_created,
        'deals_merged': deals_merged,
        'errors': errors,
    }


# ---------------------------------------------------------------------------
# D8 — enrich_interaction (FAIL-OPEN)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def enrich_interaction_step(
    envelope_dict: dict[str, Any],
    interaction_id: str | None,
    deals_processed: int,
) -> dict[str, Any]:
    """D8: MERGE deal-pipeline metadata onto the Interaction node.

    FAIL-OPEN: matches legacy pipeline.py:482-484 which calls
    ``_enrich_interaction`` outside the try/except (errors there are
    isolated within ``_enrich_interaction`` itself, not at the
    workflow level). Per Rule 5: interaction_id input is the
    downstream-produced authoritative value from D3, not
    envelope.interaction_id.
    """
    if not interaction_id:
        return {'status': 'no_op'}
    envelope = EnvelopeV1.model_validate(envelope_dict)
    pipeline = _build_deal_pipeline()
    try:
        await pipeline._enrich_interaction(
            envelope.tenant_id, interaction_id, deals_processed
        )
        return {'status': 'ok'}
    except Exception as e:  # noqa: BLE001 — intentional migration-parity
        logger.warning(
            'enrich_interaction_step.failed_fail_open',
            error=str(e),
            error_type=type(e).__name__,
        )
        return {'status': 'skipped', 'error': str(e)}


# ---------------------------------------------------------------------------
# D9 — postgres_dual_write (FAIL-OPEN)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def deal_postgres_dual_write_step(
    envelope_dict: dict[str, Any],
    merge_results_list: list[dict[str, Any]],
    interaction_id: str | None,
) -> dict[str, Any]:
    """D9: Dual-write deal projections to Postgres. FAIL-OPEN.

    Per Rule 5: interaction_id is the downstream-produced value from D3.
    The legacy path passes ``envelope.interaction_id`` (str | None) into
    ``_dual_write_postgres``; the workflow path passes the D3 result for
    consistency with the rest of the workflow.
    """
    from deal_graph.pipeline.merger import DealMergeResult

    envelope = EnvelopeV1.model_validate(envelope_dict)
    clients = get_clients()
    if clients.postgres is None:
        return {'status': 'no_op', 'rows_written': 0}

    merge_results = [DealMergeResult(**d) for d in merge_results_list]
    pipeline = _build_deal_pipeline()
    try:
        await pipeline._dual_write_postgres(
            merge_results=merge_results,
            tenant_id=envelope.tenant_id,
            interaction_id=interaction_id,
            source_user_id=getattr(envelope, 'user_id', None),
        )
        return {'status': 'ok', 'rows_written': len(merge_results)}
    except Exception as e:  # noqa: BLE001 — intentional migration-parity
        logger.warning(
            'deal_postgres_dual_write_step.failed_fail_open',
            error=str(e),
            error_type=type(e).__name__,
        )
        return {'status': 'skipped', 'rows_written': 0, 'error': str(e)}


# ---------------------------------------------------------------------------
# D10 — publish_deal_processed (FAIL-OPEN, side-effect only)
# ---------------------------------------------------------------------------


@DBOS.step(retries_allowed=False)
async def publish_deal_processed_step(
    *,
    tenant_id: str,
    account_id: str,
    interaction_id: str,
    deals_created: list[str],
    deals_merged: list[str],
) -> None:
    """D10: Emit ``deal.processed`` to EventBridge. FAIL-OPEN, no retries.

    Mirrors the D8/D9 (enrich_interaction, postgres_dual_write) pattern:
    ``retries_allowed=False`` because the underlying helper NEVER raises
    (broad try/except inside ``publish_deal_processed``). Letting DBOS
    retry on raise would re-do the whole workflow — much worse than a
    missing EventBridge event.

    Replay semantics: if the worker crashes between AWS-side acceptance
    and DBOS step-completion bookkeeping, the workflow resumes on a new
    worker and this step re-fires. The consumer dedupes via
    ``analyses.idempotency_key`` UNIQUE on
    ``f"opp-ingest:{deal_id}:{interaction_id}"``.

    Gated by the ``ENABLE_DEAL_PROCESSED_EVENTS`` env var (checked inside
    the pure helper). With the flag off this step persists step intent in
    DBOS but the helper short-circuits to a no-op — observable, harmless.

    boto3 ``put_events`` is synchronous; we offload to a worker thread so
    the event loop is never blocked on the ~50-200ms AWS round-trip.
    """
    import asyncio

    await asyncio.to_thread(
        publish_deal_processed,
        tenant_id=tenant_id,
        account_id=account_id,
        interaction_id=interaction_id,
        deals_created=deals_created,
        deals_merged=deals_merged,
        source="deal-pipeline",
        workflow_id=DBOS.workflow_id,
    )
