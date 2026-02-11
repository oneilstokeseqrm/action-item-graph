#!/usr/bin/env python3
"""
Live E2E smoke test for the dual-pipeline EnvelopeDispatcher.

Runs transcripts from examples/transcripts/transcripts.json through the full
dispatcher, exercising both the Action Item pipeline and the Deal pipeline
concurrently against a single shared Neo4j database — the same path
production traffic follows.

See docs/LIVE_E2E_TEST_RESULTS.md for the validation record.

Usage:
    python scripts/run_live_e2e.py
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.models.envelope import (
    ContentFormat,
    ContentPayload,
    EnvelopeV1,
    InteractionType,
    SourceType,
)
from action_item_graph.pipeline import ActionItemPipeline

from deal_graph.clients.neo4j_client import DealNeo4jClient
from deal_graph.pipeline.pipeline import DealPipeline, DealPipelineResult

from dispatcher.dispatcher import EnvelopeDispatcher, DispatcherResult


# =============================================================================
# Data Loading
# =============================================================================


def load_transcripts() -> dict:
    """Load transcripts from the examples file."""
    path = Path(__file__).parent.parent / 'examples' / 'transcripts' / 'transcripts.json'
    with open(path) as f:
        return json.load(f)


def build_envelope(transcript: dict, tenant_id: UUID, account_id: str) -> EnvelopeV1:
    """
    Wrap a transcript dict into a full EnvelopeV1.

    Generates a fresh interaction_id so repeated runs don't collide.
    """
    return EnvelopeV1(
        tenant_id=tenant_id,
        user_id='auth0|live_test_user',
        interaction_type=InteractionType.TRANSCRIPT,
        content=ContentPayload(
            text=transcript['text'],
            format=ContentFormat.DIARIZED,
        ),
        timestamp=datetime.fromisoformat(transcript['timestamp']),
        source=SourceType.API,
        interaction_id=uuid4(),
        account_id=account_id,
        extras={'meeting_title': transcript['meeting_title']},
    )


# =============================================================================
# Database Cleanup
# =============================================================================


async def clean_database(neo4j: Neo4jClient, tenant_id: str):
    """Clean all pipeline test data for this tenant (single shared database).

    Deletes enrichment labels first (versions, then entities), then skeleton
    labels (Interaction, Account) last — respecting relationship ordering.
    """
    print("\n--- Cleaning shared database ---")

    # Order: versions first, then entities, then skeleton (Interaction, Account)
    queries = [
        "MATCH (n:ActionItemVersion) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:ActionItemTopicVersion) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:DealVersion) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:ActionItem) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:ActionItemTopic) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Owner) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Deal) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Interaction) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Account) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
    ]

    for query in queries:
        await neo4j.execute_write(query, {'tenant_id': tenant_id})

    # Verify clean state — only check labels we own (skeleton labels like
    # Entity, Topic, Community, Chunk etc. belong to upstream eq-structured-graph-core)
    our_labels = [
        'ActionItemVersion', 'ActionItemTopicVersion', 'DealVersion',
        'ActionItem', 'ActionItemTopic', 'Owner', 'Deal',
        'Interaction', 'Account',
    ]
    label_filter = ' OR '.join(f'n:{label}' for label in our_labels)
    result = await neo4j.execute_query(
        f"MATCH (n) WHERE n.tenant_id = $tenant_id AND ({label_filter}) RETURN count(n) as count",
        {'tenant_id': tenant_id},
    )
    remaining = result[0]['count'] if result else 0
    print(f"  Shared database: cleaned pipeline labels ({remaining} remaining)")
    if remaining > 0:
        raise RuntimeError(f"Database cleanup failed — {remaining} pipeline nodes remain")


# =============================================================================
# Transcript Processing
# =============================================================================


async def verify_deal_db_after_transcript(
    neo4j: DealNeo4jClient,
    tenant_id: str,
    account_id: str,
    transcript_index: int,
    interaction_id: str,
    meeting_title: str,
):
    """Count Account and Interaction nodes in Deal DB after a transcript run, and verify the exact Interaction."""
    acct_result = await neo4j.execute_query(
        "MATCH (a:Account {tenant_id: $tid, account_id: $aid}) RETURN count(a) as count",
        {'tid': tenant_id, 'aid': account_id},
    )
    int_result = await neo4j.execute_query(
        "MATCH (i:Interaction {tenant_id: $tid}) RETURN count(i) as count",
        {'tid': tenant_id},
    )
    acct_count = acct_result[0]['count'] if acct_result else 0
    int_count = int_result[0]['count'] if int_result else 0
    print(f"\n  [Deal DB Check] After transcript {transcript_index} ({meeting_title}):")
    print(f"    Accounts: {acct_count}  |  Interactions: {int_count} (expected {transcript_index})")
    if int_count != transcript_index:
        print(f"    WARNING: Expected {transcript_index} Interaction(s), got {int_count}")

    # Query the exact Interaction node by its interaction_id
    exact_result = await neo4j.execute_query(
        """MATCH (i:Interaction {tenant_id: $tid, interaction_id: $iid})
           RETURN i.deal_count as deal_count, i.processed_at as processed_at,
                  i.interaction_type as interaction_type""",
        {'tid': tenant_id, 'iid': interaction_id},
    )
    if exact_result:
        row = exact_result[0]
        dc = row.get('deal_count', 'N/A')
        processed = 'Yes' if row.get('processed_at') else 'No'
        itype = row.get('interaction_type', 'unknown')
        print(f"    Interaction {interaction_id[:12]}...: deal_count={dc} | type={itype} | processed={processed}")
    else:
        print(f"    WARNING: Interaction {interaction_id} NOT FOUND in Deal DB!")


async def run_transcript(
    dispatcher: EnvelopeDispatcher,
    transcript: dict,
    tenant_id: UUID,
    account_id: str,
    index: int,
) -> dict:
    """Dispatch a single transcript envelope and capture results."""
    print(f"\n{'=' * 70}")
    print(f"TRANSCRIPT {index}: {transcript['meeting_title']}")
    print("=" * 70)
    print(f"Sequence: {transcript['sequence']}")
    print(f"Timestamp: {transcript['timestamp']}")
    print(f"Content length: {len(transcript['text'])} characters")
    print(f"Preview: {transcript['text'][:150]}...")

    envelope = build_envelope(transcript, tenant_id, account_id)
    print(f"Interaction ID: {envelope.interaction_id}")

    result: DispatcherResult = await dispatcher.dispatch(envelope)

    # ---- Action Item pipeline results ----
    print(f"\n--- Action Item Pipeline ---")
    if result.action_item_success:
        ai = result.action_item_result
        print(f"  Success: True")
        print(f"  Items extracted: {ai.total_extracted}")
        print(f"  Created: {len(ai.created_ids)}")
        print(f"  Updated: {len(ai.updated_ids)}")
        print(f"  Linked: {len(ai.linked_ids)}")
        print(f"  Topics created: {ai.topics_created}")
        print(f"  Topics linked: {ai.topics_linked}")
        print(f"  Processing time: {ai.processing_time_ms}ms")
        if ai.errors:
            print(f"  Errors: {ai.errors}")
    else:
        exc = result.action_item_result
        print(f"  Success: False")
        print(f"  Error: {type(exc).__name__}: {exc}")

    # ---- Deal pipeline results ----
    print(f"\n--- Deal Pipeline ---")
    if result.deal_success:
        deal = result.deal_result
        print(f"  Success: True")
        print(f"  Deals extracted: {deal.total_extracted}")
        print(f"  Deals created: {len(deal.deals_created)}")
        print(f"  Deals merged: {len(deal.deals_merged)}")
        print(f"  Processing time: {deal.processing_time_ms}ms")
        if deal.errors:
            print(f"  Per-deal errors: {deal.errors}")
        if deal.warnings:
            print(f"  Warnings: {deal.warnings}")
    else:
        exc = result.deal_result
        print(f"  Success: False")
        print(f"  Error: {type(exc).__name__}: {exc}")

    # ---- Dispatcher summary ----
    print(f"\n--- Dispatcher ---")
    print(f"  Overall success: {result.overall_success}")
    print(f"  Both succeeded: {result.both_succeeded}")
    print(f"  Dispatch time: {result.dispatch_time_ms}ms")
    if result.errors:
        print(f"  Dispatcher errors: {result.errors}")

    # Build summary dict for the aggregate report
    summary = {
        'meeting_title': transcript['meeting_title'],
        'sequence': transcript['sequence'],
        'interaction_id': str(envelope.interaction_id),
        'overall_success': result.overall_success,
        'both_succeeded': result.both_succeeded,
        'dispatch_time_ms': result.dispatch_time_ms,
        # AI pipeline
        'ai_success': result.action_item_success,
        'ai_items': 0,
        'ai_created': 0,
        'ai_updated': 0,
        'ai_topics_created': 0,
        'ai_topics_linked': 0,
        'ai_time_ms': 0,
        # Deal pipeline
        'deal_success': result.deal_success,
        'deals_extracted': 0,
        'deals_created': 0,
        'deals_merged': 0,
        'deal_time_ms': 0,
        # Errors
        'errors': result.errors,
    }

    if result.action_item_success:
        ai = result.action_item_result
        summary.update({
            'ai_items': ai.total_extracted,
            'ai_created': len(ai.created_ids),
            'ai_updated': len(ai.updated_ids),
            'ai_topics_created': ai.topics_created,
            'ai_topics_linked': ai.topics_linked,
            'ai_time_ms': ai.processing_time_ms or 0,
        })

    if result.deal_success:
        deal = result.deal_result
        summary.update({
            'deals_extracted': deal.total_extracted,
            'deals_created': len(deal.deals_created),
            'deals_merged': len(deal.deals_merged),
            'deal_time_ms': deal.processing_time_ms or 0,
        })

    return summary


# =============================================================================
# Final State Queries
# =============================================================================


async def query_ai_final_state(neo4j: Neo4jClient, tenant_id: str, account_id: str) -> dict:
    """Query the final AI graph state (action items, topics, owners, interactions)."""
    print("\n--- AI DATABASE: FINAL STATE ---")

    # Action Items
    ai_query = """
    MATCH (ai:ActionItem)
    WHERE ai.tenant_id = $tenant_id AND ai.account_id = $account_id
    OPTIONAL MATCH (ai)-[:OWNED_BY]->(o:Owner)
    OPTIONAL MATCH (ai)-[:BELONGS_TO]->(t:ActionItemTopic)
    OPTIONAL MATCH (ai)-[:EXTRACTED_FROM]->(i:Interaction)
    WITH ai, o.canonical_name as resolved_owner, t.name as topic_name,
         collect(DISTINCT i.title) as source_interactions
    RETURN ai.action_item_id as action_item_id,
           ai.summary as summary,
           ai.owner as owner,
           ai.status as status,
           resolved_owner,
           topic_name,
           source_interactions
    ORDER BY ai.created_at
    """
    action_items = await neo4j.execute_query(ai_query, {
        'tenant_id': tenant_id,
        'account_id': account_id,
    })

    for idx, ai in enumerate(action_items, 1):
        sources = ', '.join(ai['source_interactions']) if ai['source_interactions'] else 'None'
        print(f"  {idx}. {ai['summary']}")
        print(f"     Owner: {ai['owner']} | Topic: {ai['topic_name'] or 'None'} | Source: {sources}")

    print(f"  Total Action Items: {len(action_items)}")

    # Topics
    topic_query = """
    MATCH (t:ActionItemTopic)
    WHERE t.tenant_id = $tenant_id AND t.account_id = $account_id
    OPTIONAL MATCH (ai:ActionItem)-[:BELONGS_TO]->(t)
    WITH t.name as name, t.action_item_count as count, collect(ai.summary) as linked_items
    RETURN name, count, linked_items
    ORDER BY name
    """
    topics = await neo4j.execute_query(topic_query, {
        'tenant_id': tenant_id,
        'account_id': account_id,
    })

    print(f"\n  Topics:")
    for topic in topics:
        item_count = len([x for x in topic['linked_items'] if x])
        print(f"    {topic['name']}: {item_count} action items")
    print(f"  Total Topics: {len(topics)}")

    # Owners
    owner_query = """
    MATCH (o:Owner)
    WHERE o.tenant_id = $tenant_id
    OPTIONAL MATCH (ai:ActionItem)-[:OWNED_BY]->(o)
    WHERE ai.account_id = $account_id
    WITH o.canonical_name as name, count(ai) as item_count
    RETURN name, item_count
    ORDER BY item_count DESC
    """
    owners = await neo4j.execute_query(owner_query, {
        'tenant_id': tenant_id,
        'account_id': account_id,
    })

    print(f"\n  Owners:")
    for owner in owners:
        print(f"    {owner['name']}: {owner['item_count']} action items")
    print(f"  Total Owners: {len(owners)}")

    # Interactions
    int_query = """
    MATCH (i:Interaction)
    WHERE i.tenant_id = $tenant_id AND i.account_id = $account_id
    OPTIONAL MATCH (ai:ActionItem)-[:EXTRACTED_FROM]->(i)
    WITH i.title as title, i.timestamp as timestamp, count(ai) as action_item_count
    RETURN title, timestamp, action_item_count
    ORDER BY timestamp
    """
    interactions = await neo4j.execute_query(int_query, {
        'tenant_id': tenant_id,
        'account_id': account_id,
    })

    print(f"\n  Interactions:")
    for interaction in interactions:
        print(f"    {interaction['title']}: {interaction['action_item_count']} action items")
    print(f"  Total Interactions: {len(interactions)}")

    return {
        'action_items': action_items,
        'topics': topics,
        'owners': owners,
        'interactions': interactions,
    }


async def query_deal_final_state(
    neo4j: DealNeo4jClient, tenant_id: str, account_id: str,
    all_results: list[dict],
) -> dict:
    """Query the final Deal graph state (deals, versions, interactions).

    Args:
        all_results: Per-transcript result dicts (must have 'interaction_id', 'meeting_title', 'sequence').
    """
    print("\n--- DEAL DATABASE: FINAL STATE ---")

    # Deals — linked to Account via account_id property (no relationship)
    deal_query = """
    MATCH (d:Deal)
    WHERE d.tenant_id = $tenant_id AND d.account_id = $account_id
    RETURN d.opportunity_id as opportunity_id,
           d.deal_ref as deal_ref,
           d.name as name,
           d.stage as stage,
           d.amount as amount,
           d.currency as currency,
           d.opportunity_summary as opportunity_summary,
           d.evolution_summary as evolution_summary,
           d.version as version,
           d.confidence as confidence,
           d.meddic_metrics as meddic_metrics,
           d.meddic_economic_buyer as meddic_economic_buyer,
           d.meddic_decision_criteria as meddic_decision_criteria,
           d.meddic_decision_process as meddic_decision_process,
           d.meddic_identified_pain as meddic_identified_pain,
           d.meddic_champion as meddic_champion,
           d.meddic_completeness as meddic_completeness,
           d.created_at as created_at
    ORDER BY d.created_at
    """
    deals = await neo4j.execute_query(deal_query, {
        'tenant_id': tenant_id,
        'account_id': account_id,
    })

    for i, deal in enumerate(deals, 1):
        amount_str = f"${deal['amount']:,.0f}" if deal.get('amount') else 'N/A'
        currency = deal.get('currency', 'USD')
        ref_str = f" ({deal['deal_ref']})" if deal.get('deal_ref') else ''
        completeness = deal.get('meddic_completeness', 0.0)
        completeness_pct = f"{completeness * 100:.0f}%" if completeness is not None else 'N/A'
        version = deal.get('version', 1)
        print(f"  {i}. {deal['name']}")
        print(f"     ID: {deal['opportunity_id']}{ref_str}")
        print(f"     Stage: {deal['stage'] or 'N/A'} | Amount: {amount_str} {currency} | Version: {version}")
        print(f"     MEDDIC Completeness: {completeness_pct}")
        # Print non-None MEDDIC dimensions
        meddic_fields = [
            ('Metrics', deal.get('meddic_metrics')),
            ('Economic Buyer', deal.get('meddic_economic_buyer')),
            ('Decision Criteria', deal.get('meddic_decision_criteria')),
            ('Decision Process', deal.get('meddic_decision_process')),
            ('Identified Pain', deal.get('meddic_identified_pain')),
            ('Champion', deal.get('meddic_champion')),
        ]
        for label, val in meddic_fields:
            if val:
                # Truncate long values for console readability
                display = val[:80] + '...' if len(val) > 80 else val
                print(f"       {label}: {display}")
        if deal.get('opportunity_summary'):
            summary = deal['opportunity_summary']
            display = summary[:100] + '...' if len(summary) > 100 else summary
            print(f"     Summary: {display}")
    print(f"  Total Deals: {len(deals)}")

    # DealVersions — show detail for forensic audit trail
    version_query = """
    MATCH (dv:DealVersion)
    WHERE dv.tenant_id = $tenant_id
    RETURN dv.version_id as version_id,
           dv.deal_opportunity_id as deal_opportunity_id,
           dv.version as version,
           dv.name as name,
           dv.stage as stage,
           dv.change_summary as change_summary,
           dv.changed_fields as changed_fields,
           dv.created_at as created_at
    ORDER BY dv.created_at
    """
    versions = await neo4j.execute_query(version_query, {'tenant_id': tenant_id})
    version_count = len(versions)
    if versions:
        print(f"\n  DealVersions:")
        for v in versions:
            fields = v.get('changed_fields', [])
            fields_str = ', '.join(fields) if fields else 'none'
            summary = v.get('change_summary', '')
            summary_display = summary[:80] + '...' if len(summary) > 80 else summary
            print(f"    v{v['version']} of {v['deal_opportunity_id'][:12]}...")
            print(f"      Changed: [{fields_str}]")
            print(f"      Reason: {summary_display}")
    print(f"  Total DealVersions: {version_count}")

    # Interactions — query each by exact interaction_id from transcript results
    # (sequence order, not DB ordering). This eliminates mis-labeling.
    interactions = []
    print(f"\n  Interactions (deal-enriched, by transcript sequence):")
    for r in sorted(all_results, key=lambda x: x['sequence']):
        iid = r['interaction_id']
        title = r['meeting_title']
        row_result = await neo4j.execute_query(
            """MATCH (i:Interaction {tenant_id: $tid, interaction_id: $iid})
               RETURN i.deal_count as deal_count,
                      i.interaction_type as interaction_type,
                      i.processed_at as processed_at,
                      i.timestamp as timestamp""",
            {'tid': tenant_id, 'iid': iid},
        )
        if row_result:
            row = row_result[0]
            dc = row.get('deal_count', 'N/A')
            itype = row.get('interaction_type', 'unknown')
            processed = 'Yes' if row.get('processed_at') else 'No'
            ts = row.get('timestamp', 'N/A')
            print(f"    {title}: {iid} | deal_count={dc} | type={itype} | processed={processed}")
            interactions.append({
                'interaction_id': iid,
                'meeting_title': title,
                'sequence': r['sequence'],
                'deal_count': dc,
                'interaction_type': itype,
                'processed': processed,
                'timestamp': ts,
            })
        else:
            print(f"    WARNING: {title}: Interaction {iid} NOT FOUND in Deal DB!")
            interactions.append({
                'interaction_id': iid,
                'meeting_title': title,
                'sequence': r['sequence'],
                'deal_count': None,
                'missing': True,
            })
    print(f"  Total Interactions: {len(interactions)}")

    return {
        'deals': deals,
        'deal_version_count': version_count,
        'interactions': interactions,
    }


# =============================================================================
# Cross-Pipeline MERGE Verification
# =============================================================================


async def verify_cross_pipeline_merge(
    neo4j: Neo4jClient,
    tenant_id: str,
    account_id: str,
    all_results: list[dict],
) -> bool:
    """Verify that both pipelines MERGEd onto the same Account and Interaction nodes.

    This is the core integration thesis: two pipelines writing to one shared
    database must converge on shared labels (Account, Interaction) rather than
    creating duplicates. We check:

    1. Exactly ONE Account node exists for this account_id (MERGE, not CREATE).
    2. Each Interaction has enrichment from BOTH pipelines:
       - action_item_count (set by ActionItemPipeline)
       - deal_count (set by DealPipeline)
    3. The key properties (account_id, interaction_id) are present and correct.
    """
    print(f"\n{'=' * 70}")
    print("CROSS-PIPELINE MERGE VERIFICATION")
    print("=" * 70)

    passed = True

    # --- 1. Account convergence ---
    acct_result = await neo4j.execute_query(
        """
        MATCH (a:Account {tenant_id: $tenant_id, account_id: $account_id})
        RETURN a.account_id AS account_id,
               a.tenant_id AS tenant_id,
               a.name AS name,
               a.created_at AS created_at
        """,
        {'tenant_id': tenant_id, 'account_id': account_id},
    )

    acct_count = len(acct_result)
    if acct_count == 1:
        acct = acct_result[0]
        print(f"\n  [PASS] Account: exactly 1 node (MERGE convergence confirmed)")
        print(f"         account_id = {acct['account_id']}")
        print(f"         tenant_id  = {acct['tenant_id']}")
        print(f"         name       = {acct['name']}")
    elif acct_count == 0:
        print(f"\n  [FAIL] Account: 0 nodes found for account_id={account_id}")
        passed = False
    else:
        print(f"\n  [FAIL] Account: {acct_count} nodes found — MERGE created duplicates!")
        passed = False

    # --- 2. Interaction convergence ---
    print(f"\n  Interactions (both pipelines should enrich the same node):")

    interaction_pass_count = 0
    for r in sorted(all_results, key=lambda x: x['sequence']):
        iid = r['interaction_id']
        title = r['meeting_title']

        int_result = await neo4j.execute_query(
            """
            MATCH (i:Interaction {tenant_id: $tenant_id, interaction_id: $iid})
            RETURN i.interaction_id AS interaction_id,
                   i.account_id AS account_id,
                   i.action_item_count AS action_item_count,
                   i.deal_count AS deal_count,
                   i.interaction_type AS interaction_type,
                   i.content_text IS NOT NULL AS has_content,
                   i.timestamp IS NOT NULL AS has_timestamp,
                   i.processed_at IS NOT NULL AS has_processed_at
            """,
            {'tenant_id': tenant_id, 'iid': iid},
        )

        node_count = len(int_result)
        if node_count == 0:
            print(f"    [FAIL] {title}: Interaction {iid[:12]}... NOT FOUND")
            passed = False
            continue
        if node_count > 1:
            print(f"    [FAIL] {title}: {node_count} nodes — MERGE created duplicates!")
            passed = False
            continue

        i = int_result[0]
        ai_count = i.get('action_item_count')
        deal_count = i.get('deal_count')

        ai_ok = ai_count is not None
        deal_ok = deal_count is not None

        if ai_ok and deal_ok:
            status = "PASS"
            interaction_pass_count += 1
        elif ai_ok:
            status = "PARTIAL"
            detail = "AI pipeline wrote, Deal pipeline did not"
        elif deal_ok:
            status = "PARTIAL"
            detail = "Deal pipeline wrote, AI pipeline did not"
        else:
            status = "FAIL"
            detail = "Neither pipeline enriched this node"
            passed = False

        print(f"    [{status}] {title}: interaction_id={i['interaction_id'][:12]}...")
        print(f"           account_id={i['account_id']}  "
              f"action_item_count={ai_count}  deal_count={deal_count}")
        print(f"           has_content={i['has_content']}  "
              f"has_timestamp={i['has_timestamp']}  "
              f"has_processed_at={i['has_processed_at']}")
        if status == "PARTIAL":
            print(f"           Note: {detail}")

    # --- 3. Summary ---
    total = len(all_results)
    print(f"\n  Summary: {interaction_pass_count}/{total} interactions enriched by both pipelines")

    if passed and interaction_pass_count == total:
        print(f"\n  [PASS] Cross-pipeline MERGE verification: ALL CHECKS PASSED")
        print(f"         Both pipelines converged on shared Account and Interaction nodes.")
    elif passed:
        print(f"\n  [WARN] Cross-pipeline MERGE verification: partial enrichment")
        print(f"         Nodes converged (no duplicates), but not all Interactions "
              f"were enriched by both pipelines.")
    else:
        print(f"\n  [FAIL] Cross-pipeline MERGE verification: FAILURES DETECTED")

    return passed


# =============================================================================
# Summary Report
# =============================================================================


def print_summary_report(
    results: list[dict],
    ai_state: dict,
    deal_state: dict,
    data: dict,
    total_time_ms: int,
):
    """Print the comprehensive summary report."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("=" * 70)

    print(f"\nAccount: {data['account_name']}")
    print(f"Tenant ID: {data['tenant_id']}")
    print(f"Account ID: {data['account_id']}")
    print(f"Total wall-clock time: {total_time_ms}ms")

    # Per-transcript table
    print(f"\n--- PROCESSING SUMMARY ---")
    header = f"{'Transcript':<30} {'AI Items':<10} {'Topics':<8} {'Deals':<8} {'Merged':<8} {'Both OK':<9} {'Time':<10}"
    print(header)
    print("-" * len(header))

    t_items = t_topics = t_deals = t_merged = 0
    t_time = 0

    for r in results:
        title = r['meeting_title'][:28] + ".." if len(r['meeting_title']) > 30 else r['meeting_title']
        both = "Yes" if r['both_succeeded'] else "No"
        print(
            f"{title:<30} {r['ai_items']:<10} {r['ai_topics_created']:<8} "
            f"{r['deals_created']:<8} {r['deals_merged']:<8} {both:<9} {r['dispatch_time_ms']}ms"
        )
        t_items += r['ai_items']
        t_topics += r['ai_topics_created']
        t_deals += r['deals_created']
        t_merged += r['deals_merged']
        t_time += r['dispatch_time_ms'] or 0

    print("-" * len(header))
    print(f"{'TOTAL':<30} {t_items:<10} {t_topics:<8} {t_deals:<8} {t_merged:<8} {'':9} {t_time}ms")

    # Graph statistics
    print(f"\n--- AI GRAPH STATISTICS ---")
    print(f"  Action Items: {len(ai_state['action_items'])}")
    print(f"  Topics: {len(ai_state['topics'])}")
    print(f"  Owners: {len(ai_state['owners'])}")
    print(f"  Interactions: {len(ai_state['interactions'])}")

    items_with_topics = sum(1 for ai in ai_state['action_items'] if ai['topic_name'])
    total = len(ai_state['action_items'])
    if total > 0:
        pct = 100 * items_with_topics / total
        print(f"  Items with Topics: {items_with_topics}/{total} ({pct:.1f}%)")

    print(f"\n--- DEAL GRAPH STATISTICS ---")
    print(f"  Deals: {len(deal_state['deals'])}")
    print(f"  DealVersions: {deal_state['deal_version_count']}")
    print(f"  Interactions: {len(deal_state['interactions'])}")

    # Error summary
    all_errors = [e for r in results for e in r['errors']]
    if all_errors:
        print(f"\n--- ERRORS ({len(all_errors)}) ---")
        for err in all_errors:
            print(f"  - {err}")
    else:
        print(f"\n  No errors across all transcripts.")

    # Overall verdict
    all_succeeded = all(r['both_succeeded'] for r in results)
    any_succeeded = all(r['overall_success'] for r in results)

    print("\n" + "=" * 70)
    if all_succeeded:
        print("TEST COMPLETE — ALL PIPELINES SUCCEEDED FOR ALL TRANSCRIPTS")
    elif any_succeeded:
        print("TEST COMPLETE — PARTIAL SUCCESS (at least one pipeline succeeded per transcript)")
    else:
        print("TEST COMPLETE — FAILURES DETECTED")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


async def main():
    # Load transcripts
    data = load_transcripts()

    tenant_id = UUID(data['tenant_id'])
    tenant_id_str = str(tenant_id)
    account_id = data['account_id']
    account_name = data['account_name']
    transcripts = sorted(data['transcripts'], key=lambda x: x['sequence'])

    print("=" * 70)
    print("LIVE E2E SMOKE TEST — DUAL-PIPELINE DISPATCHER (SHARED DATABASE)")
    print("=" * 70)
    print(f"Account: {account_name}")
    print(f"Tenant ID: {tenant_id}")
    print(f"Account ID: {account_id}")
    print(f"Transcripts to process: {len(transcripts)}")

    # Initialize clients
    openai = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))

    ai_neo4j = Neo4jClient(
        uri=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j'),
    )

    deal_neo4j = DealNeo4jClient(
        uri=os.getenv('DEAL_NEO4J_URI'),
        username=os.getenv('DEAL_NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('DEAL_NEO4J_PASSWORD'),
        database=os.getenv('DEAL_NEO4J_DATABASE', 'neo4j'),
    )

    ai_pipeline = None
    dispatcher = None

    try:
        # Connect and set up schemas
        await ai_neo4j.connect()
        await ai_neo4j.setup_schema()
        print("AI pipeline: connected and schema ready")

        await deal_neo4j.connect()
        await deal_neo4j.setup_schema()
        print("Deal pipeline: connected and enrichment schema ready")

        try:
            await deal_neo4j.verify_skeleton_schema()
            print("Shared database: skeleton schema verified")
        except RuntimeError as e:
            print(f"WARNING: Skeleton schema verification failed: {e}")
            print("Continuing anyway — MERGE-based operations may still work.")

        # Clean shared database
        print(f"\n{'=' * 70}")
        print("CLEANING SHARED DATABASE")
        print("=" * 70)
        await clean_database(ai_neo4j, tenant_id_str)

        # Build pipelines and dispatcher
        ai_pipeline = ActionItemPipeline(openai, ai_neo4j, enable_topics=True)
        deal_pipeline = DealPipeline(deal_neo4j, openai)
        dispatcher = EnvelopeDispatcher(ai_pipeline, deal_pipeline)

        # Process each transcript in sequence order
        all_results = []
        t0 = time.monotonic()

        for i, transcript in enumerate(transcripts, 1):
            result = await run_transcript(
                dispatcher, transcript, tenant_id, account_id, i,
            )
            all_results.append(result)

            if not result['overall_success']:
                print(f"\n  WARNING: Transcript {i} had failures!")

            # Verify Deal DB Account + Interaction nodes after each transcript
            await verify_deal_db_after_transcript(
                deal_neo4j, tenant_id_str, account_id, i,
                interaction_id=result['interaction_id'],
                meeting_title=transcript['meeting_title'],
            )

        total_time_ms = int((time.monotonic() - t0) * 1000)

        # Query final state from both databases
        print(f"\n{'=' * 70}")
        print("FINAL GRAPH STATE")
        print("=" * 70)
        ai_state = await query_ai_final_state(ai_neo4j, tenant_id_str, account_id)
        deal_state = await query_deal_final_state(deal_neo4j, tenant_id_str, account_id, all_results)

        # Verify cross-pipeline MERGE convergence on shared labels
        merge_ok = await verify_cross_pipeline_merge(
            ai_neo4j, tenant_id_str, account_id, all_results,
        )

        # Print comprehensive summary
        print_summary_report(all_results, ai_state, deal_state, data, total_time_ms)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        # Close clients — ai_pipeline.close() handles openai + ai_neo4j
        if ai_pipeline:
            await ai_pipeline.close()
        else:
            await ai_neo4j.close()
            await openai.close()
        await deal_neo4j.close()


if __name__ == '__main__':
    asyncio.run(main())
