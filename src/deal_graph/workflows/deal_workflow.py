"""Deal workflow orchestrator.

Implements the 9-step DBOS workflow that replaces the synchronous
``DealPipeline.process_envelope`` call from the legacy HTTP ``/process``
route. The Lambda dispatcher (T13/Phase C) enqueues this workflow via
``DBOSClient.enqueue`` against ``DEAL_QUEUE``.

Workflow ID format (per plan locked decision):
    f"action-item-graph:deal:interaction-{envelope.interaction_id}"

Note the prefix: both pipelines share the ``action-item-graph:`` namespace
because they live in the same service. The pipeline-name segment
distinguishes them inside ``dbos.workflow_status``.

Architectural mirror of action_item_workflow: validation in workflow body
(per [[pattern-dbos-workflow-parity-rules]] Rule 1), thin orchestration,
steps construct their own sub-objects from get_clients().
"""

from __future__ import annotations

from typing import Any

from dbos import DBOS

from action_item_graph.logging import get_logger
from deal_graph.workflows.deal_steps import (
    deal_extraction_step,
    deal_postgres_dual_write_step,
    enrich_interaction_step,
    ensure_interaction_step,
    fetch_existing_deal_step,
    match_merge_loop_step,
    merge_contacts_to_deal_base_step,
    publish_deal_processed_step,
    verify_account_step,
)

logger = get_logger(__name__)


@DBOS.workflow()
async def deal_workflow(envelope_dict: dict[str, Any]) -> dict[str, Any]:
    """Orchestrate the 9-step deal pipeline.

    Args:
        envelope_dict: EnvelopeV1 serialized as a JSON-safe dict.

    Returns:
        Dict summary with the same shape as ``DealPipelineResult.to_dict()``
        produces today, suitable for the Lambda log line + ops dashboard.
    """
    # D1 — validate_envelope. Lives in workflow body, NOT a @DBOS.step
    # (Rule 1: validation inside a retryable step gets retry-wrapped).
    # Matches legacy pipeline.py:277-278 fail-fast on missing account_id.
    if not envelope_dict.get('account_id'):
        from deal_graph.errors import DealPipelineError

        raise DealPipelineError(
            'account_id is required on envelope (deal_workflow)'
        )

    envelope_interaction_id = envelope_dict.get('interaction_id') or '?'
    # opportunity_id lives in envelope.extras per EnvelopeV1's @property
    # accessor (models/envelope.py:108-111). Legacy pipeline.py:256 reads
    # via the property; the workflow path must source from the same place
    # (extras), not the top-level dict. See Rule 8 in
    # ~/.claude/projects/.../memory/pattern_dbos_workflow_parity_rules.md
    # — Pydantic @property accessors don't survive model_dump(), so
    # consumer code must either model_validate() first or replicate the
    # property's dict-key path explicitly.
    opportunity_id = (envelope_dict.get('extras') or {}).get('opportunity_id')
    logger.info(
        'deal_workflow.started',
        envelope_interaction_id=envelope_interaction_id,
        opportunity_id=opportunity_id,
    )

    # D2 — verify_account (Neo4j read)
    await verify_account_step(envelope_dict)

    # D3 — ensure_interaction (Neo4j MERGE, conditional)
    interaction_out = await ensure_interaction_step(envelope_dict)
    interaction_id = interaction_out.get('interaction_id')

    # D4 — merge_contacts_to_deal_base (Neo4j MERGE, conditional, fail-open)
    await merge_contacts_to_deal_base_step(envelope_dict)

    # D5 — fetch_existing_deal (Neo4j read, conditional on opportunity_id)
    existing_deal = await fetch_existing_deal_step(envelope_dict)
    # Behavior parity: if opportunity_id was set but no existing deal found,
    # fall through to discovery mode with existing_deal=None (legacy:
    # pipeline.py:332-341). Surface as a warning in the workflow result
    # to match legacy DealPipelineResult.warnings (Codex B-2 R1 MEDIUM).
    warnings: list[str] = []
    if opportunity_id and existing_deal is None:
        msg = (
            f'opportunity_id={opportunity_id} not found in graph; '
            'falling back to discovery extraction'
        )
        logger.warning(
            'deal_workflow.existing_deal_not_found',
            opportunity_id=opportunity_id,
        )
        warnings.append(msg)

    # D6 — extraction (LLM)
    deal_extraction_dict = await deal_extraction_step(envelope_dict, existing_deal)
    extraction_result_dict = deal_extraction_dict.get('result', {})
    total_extracted = len(extraction_result_dict.get('deals', []))
    extraction_notes = extraction_result_dict.get('extraction_notes')

    # Empty-extraction early-finalize: still enrich interaction + finish.
    # Matches legacy pipeline.py:366-372 behavior.
    if not extraction_result_dict.get('has_deals') or total_extracted == 0:
        logger.info(
            'deal_workflow.no_deals_extracted',
            interaction_id=interaction_id,
        )
        await enrich_interaction_step(envelope_dict, interaction_id, 0)
        return {
            'interaction_id': interaction_id,
            'opportunity_id': opportunity_id,
            'status': 'no_deals',
            'total_extracted': 0,
            'deals_created': [],
            'deals_merged': [],
            'errors': [],
            'warnings': warnings,
            'extraction_notes': extraction_notes,
        }

    # D7 — match_merge_loop (per-deal, fail-open per deal, single DBOS step)
    match_merge_out = await match_merge_loop_step(envelope_dict, deal_extraction_dict)
    deals_created = match_merge_out.get('deals_created', [])
    deals_merged = match_merge_out.get('deals_merged', [])
    errors = match_merge_out.get('errors', [])
    merge_results_list = match_merge_out.get('merge_results', [])

    # D8 — enrich_interaction (Neo4j MERGE, fail-open)
    deals_processed = len(deals_created) + len(deals_merged)
    await enrich_interaction_step(envelope_dict, interaction_id, deals_processed)

    # D9 — postgres_dual_write (fail-open)
    await deal_postgres_dual_write_step(envelope_dict, merge_results_list, interaction_id)

    # D10 — publish deal.processed to EventBridge (fail-open, flag-gated).
    # Feature-flag gating lives inside the helper (ENABLE_DEAL_PROCESSED_EVENTS).
    # With the flag off this step persists step intent but the helper short-
    # circuits to a no-op. Workflow-replay double-publish is absorbed by the
    # consumer-side analyses.idempotency_key UNIQUE constraint.
    await publish_deal_processed_step(
        tenant_id=str(envelope_dict.get('tenant_id', '')),
        account_id=str(envelope_dict.get('account_id') or ''),
        interaction_id=interaction_id or '',
        deals_created=deals_created,
        deals_merged=deals_merged,
    )

    logger.info(
        'deal_workflow.complete',
        interaction_id=interaction_id,
        total_extracted=total_extracted,
        deals_created=len(deals_created),
        deals_merged=len(deals_merged),
        errors=len(errors),
    )

    return {
        'interaction_id': interaction_id,
        'opportunity_id': opportunity_id,
        'status': 'ok',
        'total_extracted': total_extracted,
        'deals_created': deals_created,
        'deals_merged': deals_merged,
        'errors': errors,
        'warnings': warnings,
        'extraction_notes': extraction_notes,
    }
