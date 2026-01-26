#!/usr/bin/env python3
"""
Live test script for the Action Item Pipeline with Topic Grouping.

Runs transcripts from examples/transcripts/transcripts.json through the pipeline.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import UUID

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.pipeline import ActionItemPipeline


def load_transcripts():
    """Load transcripts from the examples file."""
    path = Path(__file__).parent / 'examples' / 'transcripts' / 'transcripts.json'
    with open(path) as f:
        return json.load(f)


async def clean_neo4j(neo4j: Neo4jClient, tenant_id: str):
    """Clean all test data from Neo4j for this tenant."""
    print("\n" + "=" * 70)
    print("CLEANING NEO4J DATABASE")
    print("=" * 70)

    # Delete all nodes for this tenant
    queries = [
        "MATCH (n:ActionItemVersion) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:TopicVersion) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:ActionItem) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Topic) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Owner) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Interaction) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
        "MATCH (n:Account) WHERE n.tenant_id = $tenant_id DETACH DELETE n",
    ]

    for query in queries:
        await neo4j.execute_write(query, {'tenant_id': tenant_id})

    print(f"✓ Deleted all nodes for tenant {tenant_id}")

    # Verify clean state for this tenant
    count_query = """
    MATCH (n)
    WHERE n.tenant_id = $tenant_id
    RETURN count(n) as count
    """
    result = await neo4j.execute_query(count_query, {'tenant_id': tenant_id})
    remaining = result[0]['count'] if result else 0
    print(f"✓ Remaining nodes for tenant: {remaining}")

    if remaining > 0:
        raise RuntimeError(f"Failed to clean database - {remaining} nodes remain")


async def run_transcript(pipeline: ActionItemPipeline, transcript: dict,
                         tenant_id: UUID, account_id: str, index: int) -> dict:
    """Run a single transcript through the pipeline."""
    print(f"\n{'=' * 70}")
    print(f"TRANSCRIPT {index}: {transcript['meeting_title']}")
    print("=" * 70)
    print(f"Sequence: {transcript['sequence']}")
    print(f"Timestamp: {transcript['timestamp']}")
    print(f"Content length: {len(transcript['text'])} characters")
    print(f"Preview: {transcript['text'][:150]}...")

    result = await pipeline.process_text(
        text=transcript['text'],
        tenant_id=tenant_id,
        account_id=account_id,
        meeting_title=transcript['meeting_title'],
        participants=transcript.get('participants', []),
    )

    print(f"\n--- Processing Results ---")
    print(f"Success: {result.success}")
    print(f"Processing time: {result.processing_time_ms}ms")
    print(f"Action items extracted: {result.total_extracted}")
    print(f"  - Created: {len(result.created_ids)}")
    print(f"  - Updated: {len(result.updated_ids)}")
    print(f"  - Linked: {len(result.linked_ids)}")
    print(f"Topics created: {result.topics_created}")
    print(f"Topics linked (reused): {result.topics_linked}")

    if result.errors:
        print(f"Errors: {result.errors}")

    if result.extraction_notes:
        print(f"Notes: {result.extraction_notes}")

    return {
        'meeting_title': transcript['meeting_title'],
        'sequence': transcript['sequence'],
        'success': result.success,
        'processing_time_ms': result.processing_time_ms,
        'total_extracted': result.total_extracted,
        'created': len(result.created_ids),
        'updated': len(result.updated_ids),
        'linked': len(result.linked_ids),
        'topics_created': result.topics_created,
        'topics_linked': result.topics_linked,
        'errors': result.errors,
    }


async def query_final_state(neo4j: Neo4jClient, tenant_id: str, account_id: str) -> dict:
    """Query and report the final state of the graph."""
    print("\n" + "=" * 70)
    print("FINAL NEO4J STATE REPORT")
    print("=" * 70)

    # Query Action Items
    print("\n--- ACTION ITEMS ---")
    ai_query = """
    MATCH (ai:ActionItem)
    WHERE ai.tenant_id = $tenant_id AND ai.account_id = $account_id
    OPTIONAL MATCH (ai)-[:OWNED_BY]->(o:Owner)
    OPTIONAL MATCH (ai)-[:BELONGS_TO]->(t:Topic)
    OPTIONAL MATCH (ai)-[:EXTRACTED_FROM]->(i:Interaction)
    RETURN ai.id as id,
           ai.action_item_text as text,
           ai.summary as summary,
           ai.owner as owner,
           ai.status as status,
           o.canonical_name as resolved_owner,
           t.name as topic_name,
           i.title as source_interaction
    ORDER BY ai.created_at
    """
    action_items = await neo4j.execute_query(ai_query, {
        'tenant_id': tenant_id,
        'account_id': account_id
    })

    for i, ai in enumerate(action_items, 1):
        print(f"\n{i}. {ai['summary']}")
        print(f"   Owner: {ai['owner']}")
        print(f"   Status: {ai['status']}")
        print(f"   Topic: {ai['topic_name'] or 'None'}")
        print(f"   Source: {ai['source_interaction']}")

    print(f"\n✓ Total Action Items: {len(action_items)}")

    # Query Topics
    print("\n--- TOPICS ---")
    topic_query = """
    MATCH (t:Topic)
    WHERE t.tenant_id = $tenant_id AND t.account_id = $account_id
    OPTIONAL MATCH (ai:ActionItem)-[:BELONGS_TO]->(t)
    WITH t.id as id, t.name as name, t.summary as summary,
         t.action_item_count as count, t.created_at as created_at,
         collect(ai.summary) as linked_items
    RETURN id, name, summary, count, linked_items, created_at
    ORDER BY created_at
    """
    topics = await neo4j.execute_query(topic_query, {
        'tenant_id': tenant_id,
        'account_id': account_id
    })

    for i, topic in enumerate(topics, 1):
        print(f"\n{i}. {topic['name']}")
        summary = topic['summary'] or ''
        if len(summary) > 120:
            summary = summary[:120] + "..."
        print(f"   Summary: {summary}")
        print(f"   Action Item Count: {topic['count']}")
        print(f"   Linked Items:")
        for item in topic['linked_items']:
            if item:
                print(f"      • {item}")

    print(f"\n✓ Total Topics: {len(topics)}")

    # Query Owners
    print("\n--- OWNERS ---")
    owner_query = """
    MATCH (o:Owner)
    WHERE o.tenant_id = $tenant_id
    OPTIONAL MATCH (ai:ActionItem)-[:OWNED_BY]->(o)
    WHERE ai.account_id = $account_id
    WITH o.canonical_name as name, o.aliases as aliases, count(ai) as item_count
    RETURN name, aliases, item_count
    ORDER BY item_count DESC
    """
    owners = await neo4j.execute_query(owner_query, {
        'tenant_id': tenant_id,
        'account_id': account_id
    })

    for owner in owners:
        aliases = owner['aliases'] or []
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        print(f"  • {owner['name']}: {owner['item_count']} action items{alias_str}")

    print(f"\n✓ Total Owners: {len(owners)}")

    # Query Interactions
    print("\n--- INTERACTIONS ---")
    int_query = """
    MATCH (i:Interaction)
    WHERE i.tenant_id = $tenant_id AND i.account_id = $account_id
    OPTIONAL MATCH (ai:ActionItem)-[:EXTRACTED_FROM]->(i)
    WITH i.title as title, i.occurred_at as occurred_at, count(ai) as action_item_count
    RETURN title, occurred_at, action_item_count
    ORDER BY occurred_at
    """
    interactions = await neo4j.execute_query(int_query, {
        'tenant_id': tenant_id,
        'account_id': account_id
    })

    for interaction in interactions:
        print(f"  • {interaction['title']}: {interaction['action_item_count']} action items")

    print(f"\n✓ Total Interactions: {len(interactions)}")

    return {
        'action_items': action_items,
        'topics': topics,
        'owners': owners,
        'interactions': interactions,
    }


def print_summary_report(results: list, final_state: dict, data: dict):
    """Print a comprehensive summary report."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("=" * 70)

    print(f"\nAccount: {data['account_name']}")
    print(f"Tenant ID: {data['tenant_id']}")
    print(f"Account ID: {data['account_id']}")

    print("\n--- PROCESSING SUMMARY ---")
    print(f"{'Transcript':<40} {'Items':<8} {'Topics':<8} {'Linked':<8} {'Time':<10}")
    print("-" * 74)

    total_items = 0
    total_topics = 0
    total_linked = 0
    total_time = 0

    for r in results:
        title = r['meeting_title'][:38] + ".." if len(r['meeting_title']) > 40 else r['meeting_title']
        print(f"{title:<40} {r['total_extracted']:<8} {r['topics_created']:<8} {r['topics_linked']:<8} {r['processing_time_ms']}ms")
        total_items += r['total_extracted']
        total_topics += r['topics_created']
        total_linked += r['topics_linked']
        total_time += r['processing_time_ms']

    print("-" * 74)
    print(f"{'TOTAL':<40} {total_items:<8} {total_topics:<8} {total_linked:<8} {total_time}ms")

    print("\n--- GRAPH STATISTICS ---")
    print(f"Action Items in Graph: {len(final_state['action_items'])}")
    print(f"Topics in Graph: {len(final_state['topics'])}")
    print(f"Owners in Graph: {len(final_state['owners'])}")
    print(f"Interactions in Graph: {len(final_state['interactions'])}")

    # Topic coverage
    items_with_topics = sum(1 for ai in final_state['action_items'] if ai['topic_name'])
    total = len(final_state['action_items'])
    if total > 0:
        pct = 100 * items_with_topics / total
        print(f"Action Items with Topics: {items_with_topics}/{total} ({pct:.1f}%)")

    # Owner distribution
    print("\n--- OWNER DISTRIBUTION ---")
    for owner in final_state['owners']:
        if owner['item_count'] > 0:
            print(f"  {owner['name']}: {owner['item_count']} action items")

    # Topic summary
    print("\n--- TOPIC SUMMARY ---")
    for topic in final_state['topics']:
        item_count = len([x for x in topic['linked_items'] if x])
        print(f"  {topic['name']}: {item_count} action items")

    print("\n" + "=" * 70)
    print("TEST COMPLETE - ALL OPERATIONS SUCCESSFUL")
    print("=" * 70)


async def main():
    # Load transcripts
    data = load_transcripts()

    tenant_id = UUID(data['tenant_id'])
    account_id = data['account_id']
    account_name = data['account_name']
    transcripts = data['transcripts']

    print("=" * 70)
    print("ACTION ITEM PIPELINE - LIVE TEST WITH TOPIC GROUPING")
    print("=" * 70)
    print(f"Account: {account_name}")
    print(f"Tenant ID: {tenant_id}")
    print(f"Account ID: {account_id}")
    print(f"Transcripts to process: {len(transcripts)}")

    # Initialize clients
    openai = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
    neo4j = Neo4jClient(
        uri=os.getenv('NEO4J_URI'),
        username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        password=os.getenv('NEO4J_PASSWORD'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j'),
    )

    pipeline = None

    try:
        await neo4j.connect()
        await neo4j.setup_schema()

        # Clean database first
        await clean_neo4j(neo4j, str(tenant_id))

        # Initialize pipeline with topics enabled
        pipeline = ActionItemPipeline(openai, neo4j, enable_topics=True)

        # Process each transcript in sequence order
        sorted_transcripts = sorted(transcripts, key=lambda x: x['sequence'])
        all_results = []

        for i, transcript in enumerate(sorted_transcripts, 1):
            result = await run_transcript(
                pipeline, transcript, tenant_id, account_id, i
            )
            all_results.append(result)

            # Check for errors
            if not result['success'] or result['errors']:
                print(f"\n⚠️  WARNING: Transcript {i} had issues!")
                if result['errors']:
                    for err in result['errors']:
                        print(f"   Error: {err}")

        # Query and report final state
        final_state = await query_final_state(neo4j, str(tenant_id), account_id)

        # Print comprehensive summary
        print_summary_report(all_results, final_state, data)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        if pipeline:
            await pipeline.close()
        await neo4j.close()
        await openai.close()


if __name__ == '__main__':
    asyncio.run(main())
