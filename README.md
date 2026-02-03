# Action Item Graph

A temporal knowledge graph pipeline for extracting and managing action items from call transcripts. Built with Neo4j for graph storage and OpenAI for intelligent extraction and embeddings.

## Features

- **Intelligent Extraction**: Extract action items from call transcripts using GPT-4.1-mini with structured output
- **Dual-Mode Detection**: Identifies both new action items and status updates to existing items
- **Smart Matching**: Vector similarity search combined with LLM-based deduplication prevents duplicates
- **Topic Grouping**: Automatically clusters action items into high-level themes/projects for cross-conversation tracking
- **Temporal Tracking**: Full version history and bi-temporal validity for audit trails
- **Multi-Tenancy**: Complete data isolation via `tenant_id` and `account_id` scoping on all nodes
- **Dual Embeddings**: Prevents embedding drift with immutable original + mutable current embeddings
- **Deal Extraction Pipeline**: Concurrent deal detection using MEDDIC-structured extraction, vector matching with graduated thresholds, and LLM-synthesized merging
- **Dual-Pipeline Dispatcher**: Routes each envelope to both Action Item and Deal pipelines concurrently with fault isolation

## Installation

### Prerequisites

- Python 3.10+
- Neo4j 5.x (Aura cloud or local)
- OpenAI API key

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd action-item-graph

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"

# Copy environment template and configure
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

Create a `.env` file with:

```bash
# Required
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_PASSWORD=your-password

# Optional
NEO4J_USER=neo4j  # defaults to 'neo4j'
OPENAI_MODEL=gpt-4.1-mini  # defaults to gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small  # defaults to text-embedding-3-small
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json  # json or console

# Deal Graph (connects to existing neo4j_structured instance)
DEAL_NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
DEAL_NEO4J_PASSWORD=your-deal-password
# DEAL_NEO4J_USERNAME=neo4j      # defaults to 'neo4j'
# DEAL_NEO4J_DATABASE=neo4j      # defaults to 'neo4j'
# DEAL_SIMILARITY_THRESHOLD=0.70 # Vector match threshold
# DEAL_AUTO_MATCH_THRESHOLD=0.90 # Auto-match (skip LLM) threshold
```

## Quick Start

### Basic Usage

```python
import asyncio
from uuid import UUID
from action_item_graph import ActionItemPipeline, configure_logging
from action_item_graph.clients import Neo4jClient, OpenAIClient

async def main():
    # Configure logging
    configure_logging(json_output=False)

    # Initialize clients
    openai = OpenAIClient()
    neo4j = Neo4jClient()

    await neo4j.connect()
    await neo4j.setup_schema()

    # Create pipeline
    pipeline = ActionItemPipeline(
        openai_client=openai,
        neo4j_client=neo4j,
    )

    # Process a transcript
    result = await pipeline.process_text(
        text="John: I'll send the proposal by Friday. Sarah: Great, I'll review it next week.",
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="acct_acme_corp",
        meeting_title="Sales Call",
    )

    print(f"Extracted: {result.total_extracted} items")
    print(f"Created: {len(result.created_ids)} new items")
    print(f"Updated: {len(result.updated_ids)} existing items")

    # Query action items
    items = await pipeline.get_action_items(
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="acct_acme_corp",
    )

    for item in items:
        print(f"- [{item['status']}] {item['summary']} (Owner: {item['owner']})")

    # Cleanup
    await openai.close()
    await neo4j.close()

asyncio.run(main())
```

### Using the Envelope Format

For API-style payloads, use the `EnvelopeV1` format:

```python
from action_item_graph.models import EnvelopeV1
from datetime import datetime, timezone

envelope = EnvelopeV1(
    tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
    user_id="user_123",
    interaction_type="transcript",
    content={"text": "Your transcript here...", "format": "plain"},
    timestamp=datetime.now(timezone.utc),
    source="api",
    account_id="acct_acme_corp",
    extras={"meeting_title": "Discovery Call"},
)

result = await pipeline.process_envelope(envelope)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Layer                                  │
│  EnvelopeV1 payload (transcript, tenant_id, account_id, metadata)   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Extraction Pipeline                             │
│  1. ActionItemExtractor: GPT-4.1-mini structured output              │
│  2. Dual-mode: new items + status updates + topic assignment         │
│  3. Embeddings: text-embedding-3-small (1536 dims)                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Matching & Deduplication                          │
│  1. ActionItemMatcher: Vector similarity (dual index search)         │
│  2. LLM deduplication decision (same/related/new)                    │
│  3. ActionItemMerger: Execute merge/update/create                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Topic Resolution                               │
│  1. TopicResolver: Match extracted topic to existing (dual search)   │
│  2. Threshold-based: auto-link (>=0.85), auto-create (<0.70)         │
│  3. TopicExecutor: Create/link topics with evolving summaries        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Neo4j Graph Store                              │
│  Nodes: Account, Interaction, ActionItem, Topic, Owner, Versions     │
│  Vector indexes: ActionItem + Topic (embedding + embedding_current)  │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Components

| Component | Description |
|-----------|-------------|
| `ActionItemExtractor` | Extracts action items + topics using GPT-4.1-mini structured output |
| `ActionItemMatcher` | Finds similar existing items via vector search + LLM deduplication |
| `ActionItemMerger` | Executes match decisions (create, merge, update status) |
| `TopicResolver` | Matches extracted topics to existing using threshold-based logic |
| `TopicExecutor` | Creates/links topics with evolving summaries |
| `ActionItemRepository` | Low-level graph CRUD operations |
| `ActionItemPipeline` | Orchestrates the full extraction-match-merge-topic flow |
| `DealPipeline` | Orchestrates deal extraction, vector matching, and LLM-synthesized merging |
| `EnvelopeDispatcher` | Routes envelopes to both pipelines concurrently with fault isolation |

See [docs/DEAL_SERVICE_ARCHITECTURE.md](./docs/DEAL_SERVICE_ARCHITECTURE.md) for the Deal pipeline architecture.

### Graph Schema

```
(:Account)-[:HAS_INTERACTION]->(:Interaction)
(:Account)-[:HAS_ACTION_ITEM]->(:ActionItem)
(:Account)-[:HAS_TOPIC]->(:Topic)
(:ActionItem)-[:EXTRACTED_FROM]->(:Interaction)
(:ActionItem)-[:OWNED_BY]->(:Owner)
(:ActionItem)-[:BELONGS_TO]->(:Topic)
(:ActionItem)-[:HAS_VERSION]->(:ActionItemVersion)
(:Topic)-[:HAS_VERSION]->(:TopicVersion)
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed schema documentation.

## Dual Embedding Strategy

To prevent embedding drift (where evolved action items no longer match their original text):

- **`embedding`** (immutable): Original text embedding, catches similar new items
- **`embedding_current`** (mutable): Updated on significant changes, catches status updates

Both indexes are searched during matching to find candidates.

## Testing

### Run All Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=src/action_item_graph
```

### Test with Real Transcripts

```bash
# Edit examples/transcripts/transcripts.json with your transcripts

# Run transcript tests
python examples/run_transcript_tests.py

# Verbose output with merge details
python examples/run_transcript_tests.py --verbose

# Process specific sequences only
python examples/run_transcript_tests.py --sequences 1 2

# Dry run (validate JSON only)
python examples/run_transcript_tests.py --dry-run
```

See [examples/transcripts/README.md](./examples/transcripts/README.md) for transcript testing documentation.

### Live E2E Smoke Test (Dual-Pipeline)

```bash
# Requires both NEO4J_* and DEAL_NEO4J_* credentials in .env
python scripts/run_live_e2e.py
```

Runs all 4 transcripts through the full `EnvelopeDispatcher`, exercising both the
Action Item and Deal pipelines concurrently. See [docs/LIVE_E2E_TEST_RESULTS.md](./docs/LIVE_E2E_TEST_RESULTS.md)
for the latest validated run results.

## API Reference

### ActionItemPipeline

The main orchestrator class.

```python
class ActionItemPipeline:
    async def process_text(
        self,
        text: str,
        tenant_id: UUID,
        account_id: str,
        meeting_title: str | None = None,
        participants: list[str] | None = None,
        user_id: str = "system",
    ) -> PipelineResult:
        """Process raw transcript text through the full pipeline."""

    async def process_envelope(
        self,
        envelope: EnvelopeV1,
    ) -> PipelineResult:
        """Process an EnvelopeV1 payload through the full pipeline."""

    async def get_action_items(
        self,
        tenant_id: UUID,
        account_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query action items from the graph."""
```

### PipelineResult

Result object returned by pipeline processing.

```python
@dataclass
class PipelineResult:
    total_extracted: int      # Total items extracted from text
    total_new_items: int      # New action items (not status updates)
    total_status_updates: int # Status updates to existing items
    total_matched: int        # Items matched to existing
    total_unmatched: int      # Items that are genuinely new
    created_ids: list[str]    # IDs of created ActionItem nodes
    updated_ids: list[str]    # IDs of updated ActionItem nodes
    linked_ids: list[str]     # IDs of linked (related) items
    processing_time_ms: float # Total processing time
    stage_timings: dict       # Per-stage timing breakdown
    merge_results: list[MergeResult]  # Detailed merge outcomes
```

### Error Handling

```python
from action_item_graph import (
    ActionItemGraphError,  # Base error
    PipelineError,         # Pipeline-level errors
    ValidationError,       # Input validation errors
    ExtractionError,       # LLM extraction errors
    MatchingError,         # Matching phase errors
    MergeError,            # Merge phase errors
    OpenAIError,           # OpenAI API errors
    Neo4jError,            # Neo4j database errors
)

try:
    result = await pipeline.process_text(...)
except ValidationError as e:
    print(f"Invalid input: {e.message}, context: {e.context}")
except ExtractionError as e:
    print(f"Extraction failed: {e.message}")
except ActionItemGraphError as e:
    print(f"Pipeline error: {e.message}")
```

## Project Structure

```
action-item-graph/
├── src/action_item_graph/
│   ├── __init__.py          # Package exports
│   ├── config.py             # Configuration management
│   ├── errors.py             # Exception hierarchy
│   ├── logging.py            # Structured logging & timing
│   ├── repository.py         # Graph CRUD operations
│   ├── clients/
│   │   ├── neo4j_client.py   # Neo4j connection & queries
│   │   └── openai_client.py  # OpenAI API wrapper
│   ├── models/
│   │   ├── envelope.py       # EnvelopeV1 input format
│   │   ├── action_item.py    # ActionItem, ActionItemVersion
│   │   ├── topic.py          # Topic, TopicVersion, ExtractedTopic
│   │   └── entities.py       # Account, Owner, Contact, etc.
│   ├── prompts/
│   │   ├── extract_action_items.py  # Extraction prompts
│   │   ├── merge_action_items.py    # Merge synthesis prompts
│   │   └── topic_prompts.py         # Topic resolution prompts
│   └── pipeline/
│       ├── extractor.py      # Action item extraction
│       ├── matcher.py        # Similarity matching
│       ├── merger.py         # Merge decision execution
│       ├── topic_resolver.py # Topic matching logic
│       ├── topic_executor.py # Topic creation/linking
│       └── pipeline.py       # Main orchestrator
├── tests/
│   ├── test_pipeline.py      # End-to-end tests
│   ├── test_extractor.py     # Extraction tests
│   ├── test_matcher.py       # Matching tests
│   ├── test_topic_resolver.py # Topic resolution tests
│   ├── test_topic_executor.py # Topic execution tests
│   └── ...
├── examples/
│   ├── process_transcript.py # Basic usage example
│   ├── run_transcript_tests.py # Transcript test runner
│   └── transcripts/          # Real transcript testing
├── src/deal_graph/              # Deal extraction pipeline
│   ├── pipeline/                # DealExtractor, DealMatcher, DealMerger
│   ├── clients/                 # DealNeo4jClient (vector search, schema)
│   ├── models/                  # ExtractedDeal, MergedDeal
│   └── repository.py           # DealRepository (graph CRUD)
├── src/dispatcher/              # Dual-pipeline envelope dispatcher
│   └── dispatcher.py            # EnvelopeDispatcher, DispatcherResult
├── scripts/
│   └── run_live_e2e.py          # Live E2E smoke test (dual-pipeline)
├── docs/
│   ├── API.md                   # API reference
│   ├── PIPELINE_GUIDE.md        # Comprehensive pipeline guide
│   ├── PHASE7_TOPIC_GROUPING.md # Topic feature documentation
│   ├── DEAL_SERVICE_ARCHITECTURE.md # Deal pipeline architecture
│   ├── LIVE_E2E_TEST_RESULTS.md # E2E smoke test validation record
│   └── test-data-report.md      # Test results report
├── ARCHITECTURE.md           # Detailed architecture docs
├── REQUIREMENTS.md           # Functional requirements
├── CHANGELOG.md              # Version history
└── pyproject.toml            # Project configuration
```

## Configuration

### Logging

```python
from action_item_graph import configure_logging, logging_context

# Configure at startup
configure_logging(
    level="INFO",        # DEBUG, INFO, WARNING, ERROR
    json_output=True,    # JSON format for production
)

# Use context for trace propagation
with logging_context(
    trace_id="req-123",
    tenant_id="tenant-abc",
    account_id="acct-xyz",
):
    result = await pipeline.process_text(...)
```

### Timing

```python
from action_item_graph import PipelineTimer

timer = PipelineTimer()

with timer.stage("extraction"):
    # ... extraction logic

with timer.stage("matching"):
    # ... matching logic

print(timer.summary())
# {'total_ms': 1234.5, 'stages': {'extraction': 500.0, 'matching': 734.5}}
```

## Requirements

See [REQUIREMENTS.md](./REQUIREMENTS.md) for detailed functional and non-functional requirements.

## License

MIT
