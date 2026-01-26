#!/usr/bin/env python3
"""
Example: Process a sales call transcript through the Action Item Pipeline.

This script demonstrates:
1. Processing a new transcript to extract action items
2. Processing a follow-up call that references the same items
3. Querying the resulting action items from the graph

Prerequisites:
    - Set environment variables:
        OPENAI_API_KEY=your_key
        NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
        NEO4J_PASSWORD=your_password

Usage:
    python examples/process_transcript.py
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.models.envelope import EnvelopeV1, ContentPayload
from action_item_graph.pipeline import ActionItemPipeline


# Sample transcripts for demonstration
INITIAL_CALL_TRANSCRIPT = """
Sarah: Thanks for taking the time to meet today. Let's discuss the next steps for the Acme deal.

John: Absolutely. I think we need to move quickly. I'll send over the pricing proposal by end of day Friday.

Sarah: Perfect. And I'll schedule a technical demo with their engineering team for next week.

John: Good idea. We should also loop in legal to review the contract terms before we send it.

Sarah: I'll reach out to legal today and get that process started.

John: One more thing - can you pull together the case studies they asked about?

Sarah: Sure, I'll compile those and include them with the proposal.
"""

FOLLOW_UP_CALL_TRANSCRIPT = """
Sarah: Quick status update on the Acme deal.

John: Great, where do we stand?

Sarah: I sent the pricing proposal yesterday - they're reviewing it now.

John: Excellent. What about the legal review?

Sarah: Legal finished their review, we're good to go on the contract terms.

John: Perfect. And the technical demo?

Sarah: Scheduled for Thursday at 2pm. I'll send calendar invites to everyone.

John: We should also prepare a custom ROI analysis for them before the demo.

Sarah: Good call, I'll put that together by Wednesday.
"""


async def main():
    """Run the example pipeline demonstration."""
    print("=" * 60)
    print("Action Item Pipeline Example")
    print("=" * 60)

    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set")
        return
    if not os.getenv('NEO4J_URI') or not os.getenv('NEO4J_PASSWORD'):
        print("ERROR: NEO4J_URI and NEO4J_PASSWORD must be set")
        return

    # Initialize clients
    openai = OpenAIClient()
    neo4j = Neo4jClient()

    try:
        # Connect to Neo4j
        print("\nConnecting to Neo4j...")
        await neo4j.connect()
        await neo4j.setup_schema()
        print("Connected and schema ready.")

        # Create pipeline
        pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

        # Use unique tenant and account for this demo
        tenant_id = uuid4()
        account_id = f"acct_acme_demo_{uuid4().hex[:8]}"

        print(f"\nDemo tenant: {tenant_id}")
        print(f"Demo account: {account_id}")

        # =====================================================================
        # Process initial call
        # =====================================================================
        print("\n" + "-" * 60)
        print("Processing initial sales call...")
        print("-" * 60)

        envelope1 = EnvelopeV1(
            tenant_id=tenant_id,
            user_id="demo_user",
            interaction_type="transcript",
            content=ContentPayload(
                text=INITIAL_CALL_TRANSCRIPT,
                format="plain",
            ),
            timestamp=datetime.now(),
            source="api",
            account_id=account_id,
            extras={
                "meeting_title": "Acme Corp - Initial Discovery",
            },
        )

        result1 = await pipeline.process_envelope(envelope1)

        print(f"\nResult:")
        print(f"  Action items extracted: {result1.total_extracted}")
        print(f"  New items created: {len(result1.created_ids)}")
        print(f"  Processing time: {result1.processing_time_ms}ms")

        if result1.created_ids:
            print(f"\n  Created IDs:")
            for item_id in result1.created_ids:
                print(f"    - {item_id}")

        # =====================================================================
        # Process follow-up call
        # =====================================================================
        print("\n" + "-" * 60)
        print("Processing follow-up call (with status updates)...")
        print("-" * 60)

        envelope2 = EnvelopeV1(
            tenant_id=tenant_id,
            user_id="demo_user",
            interaction_type="transcript",
            content=ContentPayload(
                text=FOLLOW_UP_CALL_TRANSCRIPT,
                format="plain",
            ),
            timestamp=datetime.now(),
            source="api",
            account_id=account_id,
            extras={
                "meeting_title": "Acme Corp - Status Update",
            },
        )

        result2 = await pipeline.process_envelope(envelope2)

        print(f"\nResult:")
        print(f"  Action items extracted: {result2.total_extracted}")
        print(f"  Matched to existing: {result2.total_matched}")
        print(f"  New items created: {len(result2.created_ids)}")
        print(f"  Existing items updated: {len(result2.updated_ids)}")
        print(f"  Processing time: {result2.processing_time_ms}ms")

        # =====================================================================
        # Query final state
        # =====================================================================
        print("\n" + "-" * 60)
        print("Final action items in graph:")
        print("-" * 60)

        all_items = await pipeline.get_action_items(
            tenant_id=tenant_id,
            account_id=account_id,
        )

        for item in all_items:
            status_emoji = {
                'open': '⏳',
                'in_progress': '🔄',
                'completed': '✅',
                'cancelled': '❌',
                'deferred': '⏸️',
            }.get(item.get('status', 'open'), '❓')

            print(f"\n  {status_emoji} {item.get('summary', 'No summary')}")
            print(f"     Owner: {item.get('owner', 'Unknown')}")
            print(f"     Status: {item.get('status', 'unknown')}")
            print(f"     ID: {item.get('id', 'N/A')[:8]}...")

        print(f"\n  Total items: {len(all_items)}")

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Initial call: {len(result1.created_ids)} items created")
        print(f"  Follow-up call: {len(result2.created_ids)} new, {len(result2.updated_ids)} updated")
        print(f"  Total in graph: {len(all_items)}")

    finally:
        await openai.close()
        await neo4j.close()
        print("\nConnections closed.")


if __name__ == "__main__":
    asyncio.run(main())
