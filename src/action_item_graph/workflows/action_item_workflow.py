"""Action-item workflow orchestrator.

Implements the 14-step DBOS workflow that replaces the synchronous
``ActionItemPipeline.process_envelope`` call from the legacy HTTP
``/process`` route. The Lambda dispatcher (T13/Phase C) enqueues this
workflow via ``DBOSClient.enqueue`` against ``ACTION_ITEM_QUEUE``.

Workflow ID format (per plan locked decision):
    f"action-item-graph:action-item:interaction-{envelope.interaction_id}"

Set by the caller via ``dbos.SetWorkflowID``. On FAILED re-delivery,
DBOS rejects the second enqueue with a duplicate-workflow error; recovery
requires an operator-triggered re-run via admin endpoint or manual SQL
on ``dbos.workflow_status``. Future enhancement: retry-attempt-id suffix
(deferred until operationally annoying — Open #3).

Workflow body is THIN orchestration. Each ``@DBOS.step`` checkpoints its
own input/output. Locals in this function are recreated on every replay
— DBOS replays the function from the beginning and substitutes cached
step outputs at each call site. This is why the workflow doesn't hold a
shared ``ActionItemPipeline`` instance; each step constructs what it
needs from the client registry (Peter Phase B clarification).
"""

from __future__ import annotations

from typing import Any

from dbos import DBOS

from action_item_graph.logging import get_logger
from action_item_graph.workflows.action_item_steps import (
    agent_outbox_step,
    consolidation_step,
    create_interaction_step,
    ensure_account_step,
    extraction_step,
    matching_step,
    merge_contacts_to_deal_step,
    merging_llm_step,
    merging_persist_step,
    owner_resolution_step,
    postgres_dual_write_step,
    topic_resolution_llm_step,
    topic_resolution_persist_step,
    verification_step,
)

logger = get_logger(__name__)


@DBOS.workflow()
async def action_item_workflow(
    envelope_dict: dict[str, Any],
    *,
    enable_topics: bool = True,
) -> dict[str, Any]:
    """Orchestrate the 14-step action-item pipeline.

    Args:
        envelope_dict: EnvelopeV1 serialized as a JSON-safe dict. The
            Lambda dispatcher calls ``envelope.model_dump(mode='json')``
            before enqueue.
        enable_topics: Toggle for the S10a/S10b topic-resolution branch.
            Mirrors ``ActionItemPipeline.enable_topics`` so the legacy
            and new paths produce equivalent outputs.

    Returns:
        Dict summary of the workflow outcome with the same shape the
        legacy ``PipelineResult.to_dict()`` produces, suitable for the
        Lambda log line + ops dashboard.
    """
    # Validate required fields BEFORE entering any @DBOS.step so that
    # ValidationError propagates immediately. Raising inside a retryable
    # step would cause DBOS to retry + wrap in DBOSMaxStepRetriesExceeded
    # (Codex Round 3 MEDIUM). Matches legacy pipeline.py:265-270 semantics.
    if not envelope_dict.get('account_id'):
        from action_item_graph.errors import ValidationError
        raise ValidationError(
            'EnvelopeV1 must have account_id for action item processing',
            context={'tenant_id': str(envelope_dict.get('tenant_id'))},
        )

    # Envelope interaction_id is used only as a pre-extraction trace tag.
    # The REAL interaction_id used in result/logging comes from extraction's
    # output below — extraction.interaction.interaction_id may differ when
    # the upstream envelope didn't carry an interaction_id and the extractor
    # generated one locally (Codex Round 2 MEDIUM #1).
    envelope_interaction_id = envelope_dict.get('interaction_id') or '?'
    logger.info('workflow.started', envelope_interaction_id=envelope_interaction_id)

    # S1 — ensure_account (Neo4j MERGE)
    await ensure_account_step(envelope_dict)

    # S2 — extraction (LLM)
    extraction_dict = await extraction_step(envelope_dict)
    # Authoritative interaction_id: use the one extraction settled on, which
    # matches what's persisted in Neo4j and the agent_outbox dedup key.
    interaction_id = (
        extraction_dict.get('interaction', {}).get('interaction_id')
        or envelope_interaction_id
    )
    if extraction_dict.get('action_items') == []:
        logger.info('workflow.complete_no_items', interaction_id=interaction_id)
        return {
            'interaction_id': interaction_id,
            'status': 'no_items',
            'created': 0,
            'updated': 0,
        }

    # S3 — consolidation (LLM clustering)
    consolidation_out = await consolidation_step(extraction_dict)
    extraction_dict = consolidation_out['extraction']
    items_consolidated = consolidation_out.get('items_consolidated', 0)

    # S4 — verification (LLM, fail-open)
    verification_out = await verification_step(extraction_dict)
    extraction_dict = verification_out['extraction']
    items_rejected = verification_out.get('items_rejected', 0)
    if extraction_dict.get('action_items') == []:
        logger.info(
            'workflow.complete_all_filtered',
            interaction_id=interaction_id,
            items_consolidated=items_consolidated,
            items_rejected=items_rejected,
        )
        return {
            'interaction_id': interaction_id,
            'status': 'all_filtered',
            'created': 0,
            'updated': 0,
            'items_consolidated': items_consolidated,
            'items_rejected': items_rejected,
        }

    # S5 — owner_resolution (LLM + cache; returns NEW extraction per Open #1)
    owner_out = await owner_resolution_step(extraction_dict, envelope_dict)
    extraction_dict = owner_out['extraction']
    contact_map: dict[str, str] = owner_out.get('contact_map', {})

    interaction_dict = extraction_dict['interaction']

    # S6 — create_interaction (Neo4j MERGE)
    await create_interaction_step(interaction_dict)

    # S7 — merge_contacts_to_deal (Neo4j MERGE, fail-open side-branch)
    await merge_contacts_to_deal_step(envelope_dict)

    # S8 — matching (LLM + Neo4j read)
    matching_out = await matching_step(extraction_dict, envelope_dict)
    match_results_list = matching_out['match_results']
    filtered_action_items_list = matching_out['filtered_action_items']

    # S9a — merging_llm (pure LLM)
    llm_results_list = await merging_llm_step(match_results_list)

    # S9b — merging_persist (pure Neo4j MERGE)
    merge_results_list = await merging_persist_step(
        match_results_list,
        filtered_action_items_list,
        llm_results_list,
        interaction_dict,
        contact_map,
    )

    # S10a + S10b — topic resolution (conditional on enable_topics)
    topic_execution_results_list: list[dict[str, Any]] = []
    if enable_topics:
        topic_resolution_results_list = await topic_resolution_llm_step(
            match_results_list,
            merge_results_list,
            filtered_action_items_list,
        )
        topic_execution_results_list = await topic_resolution_persist_step(
            topic_resolution_results_list,
            envelope_dict,
        )

    # S13 — postgres_dual_write (fail-open)
    await postgres_dual_write_step(
        merge_results_list,
        filtered_action_items_list,
        interaction_dict,
        topic_execution_results_list,
    )

    # S14 — agent_outbox (fail-open).
    # Pass interaction_dict explicitly so the outbox dedup key uses the
    # interaction_id extraction produced, not the envelope's (which may be
    # empty for upstream services that don't pre-assign one). Codex Round 1
    # HIGH #2 absorption.
    await agent_outbox_step(
        merge_results_list,
        filtered_action_items_list,
        envelope_dict,
        interaction_dict,
    )

    created = sum(1 for m in merge_results_list if m.get('action') == 'created')
    updated = sum(
        1
        for m in merge_results_list
        if m.get('action') in ('merged', 'status_updated')
    )
    linked = sum(1 for m in merge_results_list if m.get('action') == 'linked')

    logger.info(
        'workflow.complete',
        interaction_id=interaction_id,
        created=created,
        updated=updated,
        linked=linked,
        items_consolidated=items_consolidated,
        items_rejected=items_rejected,
        topics_processed=len(topic_execution_results_list),
    )

    return {
        'interaction_id': interaction_id,
        'status': 'ok',
        'created': created,
        'updated': updated,
        'linked': linked,
        'items_consolidated': items_consolidated,
        'items_rejected': items_rejected,
        'topics_processed': len(topic_execution_results_list),
    }
