#!/usr/bin/env python3
"""
Run real transcript tests through the Action Item Pipeline.

This script:
1. Loads transcripts from examples/transcripts/transcripts.json
2. Processes them sequentially in order
3. Shows detailed results including matches and merges
4. Saves results to examples/transcripts/results/

Usage:
    python examples/run_transcript_tests.py

    # Process only specific sequence numbers
    python examples/run_transcript_tests.py --sequences 1 2

    # Dry run (validate JSON only)
    python examples/run_transcript_tests.py --dry-run

    # Verbose output with full action item details
    python examples/run_transcript_tests.py --verbose
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Load environment variables
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)

from action_item_graph import (
    ActionItemPipeline,
    PipelineResult,
    configure_logging,
)
from action_item_graph.clients.neo4j_client import Neo4jClient
from action_item_graph.clients.openai_client import OpenAIClient


# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str) -> None:
    """Print a colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.CYAN}{'-' * 50}{Colors.ENDC}")
    print(f"{Colors.CYAN}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-' * 50}{Colors.ENDC}")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_info(label: str, value: str) -> None:
    """Print info line."""
    print(f"  {Colors.BOLD}{label}:{Colors.ENDC} {value}")


def load_transcripts(path: Path) -> dict:
    """Load and validate transcripts JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Validate required fields
    if 'tenant_id' not in data:
        raise ValueError("Missing 'tenant_id' in transcripts.json")
    if 'account_id' not in data:
        raise ValueError("Missing 'account_id' in transcripts.json")
    if 'transcripts' not in data or not data['transcripts']:
        raise ValueError("Missing or empty 'transcripts' array in transcripts.json")

    # Check for placeholder values
    if 'REPLACE' in data['tenant_id']:
        raise ValueError("Please replace the placeholder tenant_id with your actual UUID")
    if 'replace' in data['account_id'].lower():
        raise ValueError("Please replace the placeholder account_id with your actual account ID")

    # Validate each transcript
    for i, t in enumerate(data['transcripts']):
        if 'sequence' not in t:
            raise ValueError(f"Transcript {i+1} missing 'sequence' field")
        if 'text' not in t:
            raise ValueError(f"Transcript {i+1} missing 'text' field")
        if 'REPLACE' in t['text']:
            raise ValueError(f"Transcript {i+1} still has placeholder text - please add your real transcript")

    # Sort by sequence
    data['transcripts'].sort(key=lambda x: x['sequence'])

    return data


def format_duration(ms: float | None) -> str:
    """Format milliseconds as human-readable duration."""
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms/1000:.2f}s"


async def process_transcript(
    pipeline: ActionItemPipeline,
    tenant_id: UUID,
    account_id: str,
    transcript: dict,
    verbose: bool = False,
) -> dict:
    """Process a single transcript and return detailed results."""
    sequence = transcript['sequence']
    meeting_title = transcript.get('meeting_title', f'Transcript {sequence}')
    text = transcript['text']
    expected_items = transcript.get('expected_action_items')
    expected_updates = transcript.get('expected_updates')

    print_section(f"Processing: {meeting_title} (Sequence {sequence})")
    print_info("Text length", f"{len(text)} characters")

    if transcript.get('notes'):
        print_info("Notes", transcript['notes'])

    # Process through pipeline
    start_time = datetime.now()
    result = await pipeline.process_text(
        text=text,
        tenant_id=tenant_id,
        account_id=account_id,
        meeting_title=meeting_title,
        participants=transcript.get('participants'),
    )
    end_time = datetime.now()

    # Print results
    print(f"\n  {Colors.BOLD}Results:{Colors.ENDC}")
    print_info("Processing time", format_duration(result.processing_time_ms))
    print_info("Total extracted", str(result.total_extracted))
    print_info("New items", str(result.total_new_items))
    print_info("Status updates", str(result.total_status_updates))
    print_info("Matched existing", str(result.total_matched))
    print_info("Unmatched (new)", str(result.total_unmatched))

    # Stage timings
    if result.stage_timings:
        print(f"\n  {Colors.BOLD}Stage Timings:{Colors.ENDC}")
        for stage, duration in result.stage_timings.items():
            print(f"    {stage}: {format_duration(duration)}")

    # Created items
    if result.created_ids:
        print(f"\n  {Colors.GREEN}Created ({len(result.created_ids)}):{Colors.ENDC}")
        for item_id in result.created_ids:
            print(f"    + {item_id[:12]}...")

    # Updated items
    if result.updated_ids:
        print(f"\n  {Colors.BLUE}Updated ({len(result.updated_ids)}):{Colors.ENDC}")
        for item_id in result.updated_ids:
            print(f"    ~ {item_id[:12]}...")

    # Linked items
    if result.linked_ids:
        print(f"\n  {Colors.CYAN}Linked ({len(result.linked_ids)}):{Colors.ENDC}")
        for item_id in result.linked_ids:
            print(f"    → {item_id[:12]}...")

    # Verbose: show merge details
    if verbose and result.merge_results:
        print(f"\n  {Colors.BOLD}Merge Details:{Colors.ENDC}")
        for mr in result.merge_results:
            action_emoji = {
                'created': '🆕',
                'merged': '🔀',
                'status_updated': '📝',
                'linked': '🔗',
            }.get(mr.action, '❓')
            print(f"    {action_emoji} {mr.action}: {mr.action_item_id[:12]}...")
            if mr.details:
                for key, val in mr.details.items():
                    if val:  # Only show non-empty values
                        print(f"       {key}: {str(val)[:80]}")

    # Validation against expectations
    validation_passed = True
    if expected_items is not None:
        if result.total_extracted == expected_items:
            print_success(f"Expected {expected_items} items, got {result.total_extracted}")
        else:
            print_warning(f"Expected {expected_items} items, got {result.total_extracted}")
            validation_passed = False

    if expected_updates is not None:
        actual_updates = len(result.updated_ids)
        if actual_updates == expected_updates:
            print_success(f"Expected {expected_updates} updates, got {actual_updates}")
        else:
            print_warning(f"Expected {expected_updates} updates, got {actual_updates}")
            validation_passed = False

    # Return detailed result for saving
    return {
        'sequence': sequence,
        'meeting_title': meeting_title,
        'validation_passed': validation_passed,
        'result': result.to_dict(),
        'text_length': len(text),
        'processed_at': end_time.isoformat(),
    }


async def show_final_state(
    pipeline: ActionItemPipeline,
    tenant_id: UUID,
    account_id: str,
) -> list[dict]:
    """Query and display final state of all action items."""
    print_section("Final Action Item State")

    items = await pipeline.get_action_items(
        tenant_id=tenant_id,
        account_id=account_id,
        limit=100,
    )

    if not items:
        print("  No action items found.")
        return []

    status_emoji = {
        'open': '⏳',
        'in_progress': '🔄',
        'completed': '✅',
        'cancelled': '❌',
        'deferred': '⏸️',
    }

    for item in items:
        emoji = status_emoji.get(item.get('status', 'open'), '❓')
        print(f"\n  {emoji} {item.get('summary', 'No summary')}")
        print(f"     ID: {item.get('id', 'N/A')[:12]}...")
        print(f"     Status: {item.get('status', 'unknown')}")
        print(f"     Owner: {item.get('owner', 'Unknown')}")
        if item.get('due_date_text'):
            print(f"     Due: {item.get('due_date_text')}")

    print(f"\n  {Colors.BOLD}Total items: {len(items)}{Colors.ENDC}")
    return items


async def run_tests(
    sequences: list[int] | None = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the transcript tests."""
    print_header("Action Item Pipeline - Transcript Tests")

    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print_error("OPENAI_API_KEY not set")
        return
    if not os.getenv('NEO4J_URI') or not os.getenv('NEO4J_PASSWORD'):
        print_error("NEO4J_URI and NEO4J_PASSWORD must be set")
        return

    # Load transcripts
    transcripts_path = Path(__file__).parent / 'transcripts' / 'transcripts.json'
    try:
        data = load_transcripts(transcripts_path)
        print_success(f"Loaded {len(data['transcripts'])} transcripts")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print_error(str(e))
        return

    tenant_id = UUID(data['tenant_id'])
    account_id = data['account_id']
    account_name = data.get('account_name', account_id)

    print_info("Tenant ID", str(tenant_id))
    print_info("Account ID", account_id)
    print_info("Account Name", account_name)

    # Filter by sequences if specified
    transcripts = data['transcripts']
    if sequences:
        transcripts = [t for t in transcripts if t['sequence'] in sequences]
        print_info("Processing sequences", str(sequences))

    if dry_run:
        print_section("Dry Run - Transcript Validation")
        for t in transcripts:
            print(f"\n  Sequence {t['sequence']}: {t.get('meeting_title', 'Untitled')}")
            print(f"    Text length: {len(t['text'])} characters")
            print(f"    Expected items: {t.get('expected_action_items', 'not specified')}")
            print(f"    Expected updates: {t.get('expected_updates', 'not specified')}")
        print_success("All transcripts validated. Ready to process.")
        return

    # Initialize clients
    configure_logging(json_output=False)
    openai = OpenAIClient()
    neo4j = Neo4jClient()

    results = []

    try:
        # Connect
        print_section("Connecting to Services")
        await neo4j.connect()
        await neo4j.setup_schema()
        print_success("Connected to Neo4j")
        print_success("OpenAI client ready")

        # Create pipeline
        pipeline = ActionItemPipeline(openai_client=openai, neo4j_client=neo4j)

        # Process each transcript in order
        for transcript in transcripts:
            result = await process_transcript(
                pipeline=pipeline,
                tenant_id=tenant_id,
                account_id=account_id,
                transcript=transcript,
                verbose=verbose,
            )
            results.append(result)

        # Show final state
        final_items = await show_final_state(
            pipeline=pipeline,
            tenant_id=tenant_id,
            account_id=account_id,
        )

        # Summary
        print_header("Test Summary")
        total_created = sum(len(r['result']['created_ids']) for r in results)
        total_updated = sum(len(r['result']['updated_ids']) for r in results)
        total_extracted = sum(r['result']['total_extracted'] for r in results)
        validations_passed = sum(1 for r in results if r['validation_passed'])

        print_info("Transcripts processed", str(len(results)))
        print_info("Total items extracted", str(total_extracted))
        print_info("Total items created", str(total_created))
        print_info("Total items updated", str(total_updated))
        print_info("Final items in graph", str(len(final_items)))
        print_info("Validations passed", f"{validations_passed}/{len(results)}")

        # Save results
        results_dir = Path(__file__).parent / 'transcripts' / 'results'
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'results_{timestamp}.json'

        with open(results_file, 'w') as f:
            json.dump({
                'tenant_id': str(tenant_id),
                'account_id': account_id,
                'run_at': datetime.now().isoformat(),
                'transcripts_processed': len(results),
                'summary': {
                    'total_extracted': total_extracted,
                    'total_created': total_created,
                    'total_updated': total_updated,
                    'final_item_count': len(final_items),
                    'validations_passed': validations_passed,
                },
                'results': results,
                'final_items': final_items,
            }, f, indent=2, default=str)

        print_success(f"Results saved to: {results_file}")

    finally:
        await openai.close()
        await neo4j.close()
        print("\nConnections closed.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Run real transcript tests through the Action Item Pipeline'
    )
    parser.add_argument(
        '--sequences', '-s',
        type=int,
        nargs='+',
        help='Process only specific sequence numbers (e.g., -s 1 2)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed merge information'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Validate transcripts without processing'
    )

    args = parser.parse_args()

    asyncio.run(run_tests(
        sequences=args.sequences,
        verbose=args.verbose,
        dry_run=args.dry_run,
    ))


if __name__ == '__main__':
    main()
