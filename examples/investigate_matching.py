#!/usr/bin/env python3
"""
Investigate matching behavior between AML-related action items.

This script:
1. Queries all action items for the test account
2. Identifies AML-related items
3. Calculates cosine similarity between them
4. Reports whether they should have matched
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from uuid import UUID

# Add src to path
import sys
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

from action_item_graph.clients.neo4j_client import Neo4jClient


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


async def investigate():
    """Run the investigation."""
    # Load test config
    transcripts_path = Path(__file__).parent / 'transcripts' / 'transcripts.json'
    with open(transcripts_path) as f:
        data = json.load(f)

    tenant_id = UUID(data['tenant_id'])
    account_id = data['account_id']

    print(f"Tenant ID: {tenant_id}")
    print(f"Account ID: {account_id}")
    print("=" * 70)

    # Connect to Neo4j
    neo4j = Neo4jClient()
    await neo4j.connect()

    try:
        # Query all action items with their embeddings and source interaction
        query = """
        MATCH (ai:ActionItem)
        WHERE ai.tenant_id = $tenant_id AND ai.account_id = $account_id
        OPTIONAL MATCH (ai)-[:EXTRACTED_FROM]->(i:Interaction)
        RETURN ai.id as id,
               ai.summary as summary,
               ai.action_item_text as text,
               ai.status as status,
               ai.owner as owner,
               ai.embedding_current as embedding,
               i.title as source_interaction,
               i.created_at as interaction_time
        ORDER BY interaction_time, ai.summary
        """

        result = await neo4j.execute_query(
            query,
            {"tenant_id": str(tenant_id), "account_id": account_id}
        )

        items = list(result)
        print(f"\nTotal action items found: {len(items)}")
        print("=" * 70)

        # Group by source interaction
        by_source = {}
        for item in items:
            source = item.get('source_interaction') or 'Unknown'
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)

        print("\nItems by source:")
        for source, source_items in by_source.items():
            print(f"\n  {source}: {len(source_items)} items")
            for item in source_items:
                print(f"    - {item['summary'][:60]}...")

        # Find AML-related items
        aml_keywords = ['aml', 'application modernization', 'bedrock', 'meeting materials', 'business case tool']
        aml_items = []

        print("\n" + "=" * 70)
        print("AML-RELATED ITEMS:")
        print("=" * 70)

        for item in items:
            text_lower = (item['summary'] + ' ' + (item['text'] or '')).lower()
            if any(kw in text_lower for kw in aml_keywords):
                aml_items.append(item)
                print(f"\n  Source: {item.get('source_interaction', 'Unknown')}")
                print(f"  Summary: {item['summary']}")
                print(f"  Owner: {item.get('owner', 'Unknown')}")
                print(f"  ID: {item['id'][:12]}...")

        print(f"\n\nTotal AML-related items: {len(aml_items)}")

        # Calculate similarity matrix between AML items
        if len(aml_items) > 1:
            print("\n" + "=" * 70)
            print("COSINE SIMILARITY MATRIX (AML items):")
            print("=" * 70)
            print("\nThreshold for matching: 0.65")
            print("-" * 70)

            # Create similarity matrix
            n = len(aml_items)
            for i in range(n):
                for j in range(i + 1, n):
                    item_a = aml_items[i]
                    item_b = aml_items[j]

                    if item_a['embedding'] and item_b['embedding']:
                        sim = cosine_similarity(item_a['embedding'], item_b['embedding'])
                        marker = "✓ SHOULD MATCH" if sim >= 0.65 else "✗ Below threshold"

                        print(f"\n{marker} (similarity: {sim:.4f})")
                        print(f"  A [{item_a.get('source_interaction', '?')}]: {item_a['summary'][:50]}...")
                        print(f"  B [{item_b.get('source_interaction', '?')}]: {item_b['summary'][:50]}...")
                    else:
                        print(f"\n⚠ Missing embedding")
                        print(f"  A: {item_a['summary'][:50]}...")
                        print(f"  B: {item_b['summary'][:50]}...")

        # Also check ALL cross-call similarities above 0.5
        print("\n" + "=" * 70)
        print("ALL CROSS-CALL SIMILARITIES > 0.5:")
        print("=" * 70)

        sources = list(by_source.keys())
        high_sims = []

        for i, src_a in enumerate(sources):
            for src_b in sources[i+1:]:
                for item_a in by_source[src_a]:
                    for item_b in by_source[src_b]:
                        if item_a['embedding'] and item_b['embedding']:
                            sim = cosine_similarity(item_a['embedding'], item_b['embedding'])
                            if sim > 0.5:
                                high_sims.append({
                                    'sim': sim,
                                    'a_source': src_a,
                                    'b_source': src_b,
                                    'a_summary': item_a['summary'],
                                    'b_summary': item_b['summary'],
                                })

        # Sort by similarity
        high_sims.sort(key=lambda x: x['sim'], reverse=True)

        print(f"\nFound {len(high_sims)} cross-call pairs with similarity > 0.5:")
        for pair in high_sims[:20]:  # Top 20
            marker = "✓ MATCH" if pair['sim'] >= 0.65 else "✗"
            print(f"\n{marker} Similarity: {pair['sim']:.4f}")
            print(f"  [{pair['a_source']}]: {pair['a_summary'][:55]}...")
            print(f"  [{pair['b_source']}]: {pair['b_summary'][:55]}...")

        # Summary
        print("\n" + "=" * 70)
        print("INVESTIGATION SUMMARY:")
        print("=" * 70)
        matches_above_threshold = len([p for p in high_sims if p['sim'] >= 0.65])
        print(f"  Total cross-call pairs checked: {sum(len(by_source[a]) * len(by_source[b]) for i, a in enumerate(sources) for b in sources[i+1:])}")
        print(f"  Pairs with similarity > 0.5: {len(high_sims)}")
        print(f"  Pairs with similarity >= 0.65 (threshold): {matches_above_threshold}")

        if matches_above_threshold == 0:
            print("\n  CONCLUSION: No cross-call pairs exceeded the 0.65 similarity threshold.")
            print("  The matching algorithm worked correctly - items are semantically distinct.")
        else:
            print(f"\n  ISSUE FOUND: {matches_above_threshold} pairs should have matched but didn't!")
            print("  This indicates a bug in the matching logic.")

    finally:
        await neo4j.close()


if __name__ == '__main__':
    asyncio.run(investigate())
