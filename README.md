# Action Item Graph

A temporal knowledge graph pipeline for extracting and managing action items from call transcripts. Built with Neo4j for graph storage and OpenAI for intelligent extraction and embeddings.

## Features

- **Quality-First Extraction**: F-CoT (Focused Chain-of-Thought) prompting with Five-Field Commitment Framework extracts 2-3 high-quality items per interaction instead of 5+ noisy ones
- **Multi-Stage Validation**: Within-batch consolidation (embedding clustering) + LLM-as-Judge adversarial verification filter weak extractions
- **Owner Resolution**: Account-scoped alias caches with fuzzy matching, word-boundary awareness, and LLM role-to-name resolution
- **Priority Scoring**: Weighted composite scoring (impact, urgency, specificity, confidence) enables priority-based retrieval
- **Graduated Matching**: Three-tier thresholds (auto-match/LLM/auto-create) reduce LLM deduplication calls by ~60%
- **Dual-Mode Detection**: Identifies both new action items and status updates to existing items
- **Topic Grouping**: Automatically clusters action items into high-level themes/projects for cross-conversation tracking
- **Temporal Tracking**: Full version history and bi-temporal validity for audit trails
- **Multi-Tenancy**: Complete data isolation via `tenant_id` and `account_id` scoping on all nodes
- **Dual Embeddings**: Prevents embedding drift with immutable original + mutable current embeddings
- **Deal Extraction Pipeline**: Concurrent deal detection using MEDDIC-structured extraction, vector matching with graduated thresholds, and LLM-synthesized merging
- **Contact Enrichment**: Rich contact names in LLM prompts, Owner→Contact linking (IDENTIFIES_AS), Contact→Deal relationships (ENGAGED_ON) with champion/economic_buyer role enrichment
- **Dual-Pipeline Dispatcher**: Routes each envelope to both Action Item and Deal pipelines concurrently with fault isolation
- **Postgres Dual-Write**: Optional projection of action items, topics, versions, deals, and deal versions to Neon Postgres for frontend reads (Neo4j remains source of truth)

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
# Required — OpenAI
OPENAI_API_KEY=sk-...

# Required — Neo4j (shared database for all pipelines)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_PASSWORD=your-password
DEAL_NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io   # same instance
DEAL_NEO4J_PASSWORD=your-password                     # same credentials

# Optional — Neo4j
# NEO4J_USERNAME=neo4j                   # defaults to 'neo4j'
# NEO4J_DATABASE=neo4j                   # defaults to 'neo4j'
# DEAL_SIMILARITY_THRESHOLD=0.70         # Vector match threshold
# DEAL_AUTO_MATCH_THRESHOLD=0.90         # Auto-match (skip LLM) threshold

# Optional — Postgres Dual-Write (Neon)
# NEON_DATABASE_URL=postgresql://user:pass@host/db  # Enables Postgres projection
# When set, ActionItemPipeline writes to both Neo4j and Postgres.
# Postgres failures never block Neo4j writes.

# Optional — General
# OPENAI_MODEL=gpt-4.1-mini             # defaults to gpt-4.1-mini
# EMBEDDING_MODEL=text-embedding-3-small # defaults to text-embedding-3-small
# LOG_LEVEL=INFO                         # DEBUG, INFO, WARNING, ERROR
# LOG_FORMAT=json                        # json or console

# Railway API Service (required when running as a service)
# WORKER_API_KEY=your-shared-secret     # Bearer token for Lambda → Railway auth

# Lambda Forwarder (set in Lambda environment)
# API_BASE_URL=https://your-railway-service.up.railway.app
# WORKER_API_KEY=your-shared-secret     # Must match Railway's WORKER_API_KEY
# HTTP_TIMEOUT_SECONDS=100              # httpx timeout (default: 100)
# MAX_RETRIES=2                         # Retries for 5xx/network errors (default: 2)
```

> **Note**: `NEO4J_*` and `DEAL_NEO4J_*` both point to the **same shared Neo4j Aura
> instance**. Both pipelines write to a single database, converging on shared Account
> and Interaction nodes via defensive MERGE. See [Shared Database Architecture](#shared-database-architecture) below.

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

### Using the Dual-Pipeline Dispatcher (Recommended)

For production use, the `EnvelopeDispatcher` routes each envelope to both the
Action Item and Deal pipelines concurrently:

```python
import asyncio
from uuid import UUID
from datetime import datetime, timezone
from action_item_graph.models import EnvelopeV1
from dispatcher import EnvelopeDispatcher

async def main():
    dispatcher = EnvelopeDispatcher.from_env()

    envelope = EnvelopeV1(
        tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
        user_id="user_123",
        interaction_type="transcript",
        content={"text": "Your transcript here...", "format": "plain"},
        timestamp=datetime.now(timezone.utc),
        source="api",
        account_id="acct_acme_corp",
    )

    result = await dispatcher.dispatch(envelope)

    if result.action_item_success:
        print(f"Action items: {result.action_item_result.total_extracted}")
    if result.deal_success:
        print(f"Deals extracted: {result.deal_result.total_extracted}")
    print(f"Both succeeded: {result.both_succeeded}")

    await dispatcher.close()

asyncio.run(main())
```

The dispatcher uses `asyncio.gather(return_exceptions=True)` — one pipeline failing
never blocks the other. See [docs/DEAL_SERVICE_ARCHITECTURE.md](./docs/DEAL_SERVICE_ARCHITECTURE.md)
for full architecture details.

## Shared Database Architecture

All pipelines write to a **single shared Neo4j Aura instance**. Each pipeline owns
specific labels and their constraints, converging on shared Account and Interaction
nodes via defensive MERGE:

```
┌──────────────────────────────────────────────────┐
│              Shared Neo4j Aura Instance           │
├──────────────────────────────────────────────────┤
│  Skeleton (eq-structured-graph-core):            │
│    Account, Interaction, Contact, Entity, ...    │
│                                                  │
│  AI Pipeline (Neo4jClient):                      │
│    ActionItem, ActionItemVersion, Owner,          │
│    ActionItemTopic, ActionItemTopicVersion        │
│    Constraints: UNIQUENESS · Vector: 4 indexes   │
│                                                  │
│  Deal Pipeline (DealNeo4jClient):                │
│    Deal, DealVersion                             │
│    Constraints: UNIQUENESS · Vector: 2 indexes   │
└──────────────────────────────────────────────────┘
```

- **`Neo4jClient`** manages AI pipeline labels (UNIQUENESS constraints, 4 vector indexes)
- **`DealNeo4jClient`** manages Deal pipeline labels (UNIQUENESS constraints, 2 vector indexes)
- Both converge on shared Account/Interaction nodes via `MERGE ... ON CREATE SET ... ON MATCH SET`
- Both pipelines run concurrently via `EnvelopeDispatcher` with fault isolation
- Both pipelines optionally dual-write to Neon Postgres when `NEON_DATABASE_URL` is set:
  - Action Items: `action_items`, `action_item_versions`, `action_item_topics`, `action_item_topic_memberships`, `action_item_owners`
  - Deals: `opportunities` table (AI extraction columns: MEDDIC, ontology dimensions, embeddings) + `deal_versions` table (bi-temporal snapshots)
  - Deal writes are coordinated with the opportunity-forecasting pipeline to avoid trigger-protected columns (`stage`, `amount`, `close_date`, etc.)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Layer                                  │
│  EnvelopeV1 payload (transcript, tenant_id, account_id, metadata)   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Extraction (F-CoT)                              │
│  1. ActionItemExtractor: Commitment framework + scoring dimensions   │
│  2. Dual-mode: new items + status updates + topic assignment         │
│  3. Embeddings: text-embedding-3-small (1536 dims)                   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Quality Gates                                   │
│  1. Consolidator: Within-batch dedup (embedding clustering, 0.80)    │
│  2. Verifier: LLM-as-Judge adversarial check (confidence floor 0.4) │
│  3. Owner Pre-Resolver: Account-scoped alias cache + fuzzy matching  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Matching & Deduplication                          │
│  1. ActionItemMatcher: Graduated thresholds (0.68/0.88)              │
│  2. Auto-match (>=0.88), LLM zone (0.68-0.88), auto-create (<0.68) │
│  3. ActionItemMerger: Execute merge/update/create                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Topic Resolution + Scoring                     │
│  1. TopicResolver: Match extracted topic to existing (dual search)   │
│  2. TopicExecutor: Create/link topics with evolving summaries        │
│  3. Priority scoring: 0.40×impact + 0.35×urgency + 0.15×spec + ...  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Neo4j Graph Store (source of truth)            │
│  Nodes: Account, Interaction, ActionItem, ActionItemTopic, Owner, +  │
│  ActionItem scoring fields: priority_score, commitment_strength, ... │
│  Vector indexes: ActionItem + ActionItemTopic (embedding + current)  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼ (optional dual-write)
┌─────────────────────────────────────────────────────────────────────┐
│                    Neon Postgres (projection)                        │
│  Tables: action_items (+scoring columns), action_item_versions,     │
│  action_item_topics, action_item_topic_memberships,                 │
│  action_item_owners, opportunities (AI columns), deal_versions      │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Components

| Component | Description |
|-----------|-------------|
| `ActionItemExtractor` | F-CoT extraction with commitment framework + scoring dimensions |
| `ActionItemConsolidator` | Within-batch dedup via embedding cosine similarity clustering |
| `ActionItemVerifier` | LLM-as-Judge adversarial quality validation |
| `OwnerPreResolver` | Account-scoped owner cache with fuzzy matching + LLM role resolution |
| `ActionItemMatcher` | Graduated threshold matching (auto-match/LLM/auto-create) |
| `ActionItemMerger` | Executes match decisions (create, merge, update status) |
| `TopicResolver` | Matches extracted topics to existing using threshold-based logic |
| `TopicExecutor` | Creates/links topics with evolving summaries |
| `ActionItemRepository` | Low-level graph CRUD operations |
| `ActionItemPipeline` | Orchestrates the full 10-stage pipeline |
| `DealPipeline` | Orchestrates deal extraction, vector matching, and LLM-synthesized merging |
| `EnvelopeDispatcher` | Routes envelopes to both pipelines concurrently with fault isolation |
| `Railway API` | FastAPI service receiving events from Lambda, dispatching to both pipelines |
| `Lambda Forwarder` | Thin AWS Lambda parsing EventBridge/SQS messages, forwarding to Railway |

See [docs/DEAL_SERVICE_ARCHITECTURE.md](./docs/DEAL_SERVICE_ARCHITECTURE.md) for the Deal pipeline architecture.

### Graph Schema

```
(:Account)-[:HAS_INTERACTION]->(:Interaction)
(:Account)-[:HAS_ACTION_ITEM]->(:ActionItem)
(:Account)-[:HAS_TOPIC]->(:ActionItemTopic)
(:ActionItem)-[:EXTRACTED_FROM]->(:Interaction)
(:ActionItem)-[:OWNED_BY]->(:Owner)
(:ActionItem)-[:BELONGS_TO]->(:ActionItemTopic)
(:ActionItem)-[:HAS_VERSION]->(:ActionItemVersion)
(:ActionItemTopic)-[:HAS_VERSION]->(:ActionItemTopicVersion)
(:Owner)-[:IDENTIFIES_AS]->(:Contact)
(:Contact)-[:ENGAGED_ON]->(:Deal)
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed schema documentation.

## Dual Embedding Strategy

To prevent embedding drift (where evolved action items no longer match their original text):

- **`embedding`** (immutable): Original text embedding, catches similar new items
- **`embedding_current`** (mutable): Updated on significant changes, catches status updates

Both indexes are searched during matching to find candidates.

## Testing

### Run All Unit Tests

```bash
# Full test suite (all pipelines)
pytest tests/ -v

# With coverage for both packages
pytest tests/ --cov=src/action_item_graph --cov=src/deal_graph
```

### Run Pipeline-Specific Tests

```bash
# Action Item pipeline only
pytest tests/test_pipeline.py tests/test_extractor.py tests/test_matcher.py tests/test_merger.py -v

# Deal pipeline only
pytest tests/test_deal_extraction.py tests/test_deal_matcher.py tests/test_deal_merger.py tests/test_deal_pipeline.py -v

# Dispatcher (routes to both pipelines)
pytest tests/test_dispatcher.py -v

# Event consumer (Railway API + Lambda)
pytest tests/test_api_config.py tests/test_api_auth.py tests/test_api_health.py tests/test_api_process.py tests/test_api_main.py tests/test_lambda_config.py tests/test_lambda_envelope.py tests/test_lambda_api_client.py tests/test_lambda_handler.py -v

# Postgres dual-write
pytest tests/test_postgres_client.py tests/test_pipeline_dual_write.py -v

# Duplicate-text regression tests
pytest tests/test_duplicate_text.py -v

# Quality pipeline (consolidation, verification, owner resolution, scoring)
pytest tests/test_consolidator.py tests/test_verifier.py tests/test_owner_resolver.py tests/test_scoring.py -v
```

### Test with Real Transcripts

```bash
# Edit examples/transcripts/transcripts.json with your transcripts

# Run transcript tests
python examples/run_transcript_tests.py

# Verbose output with merge details
python examples/run_transcript_tests.py --verbose
```

See [examples/transcripts/README.md](./examples/transcripts/README.md) for transcript testing documentation.

### Live E2E Smoke Test (Dual-Pipeline)

```bash
# Requires NEO4J_* credentials in .env (shared database)
python scripts/run_live_e2e.py
```

Runs all 4 transcripts through the full `EnvelopeDispatcher`, exercising both the
Action Item and Deal pipelines concurrently against the shared Neo4j Aura instance. See
[docs/SMOKE_TEST_GUIDE.md](./docs/SMOKE_TEST_GUIDE.md) for comprehensive testing
procedures, and [docs/LIVE_E2E_TEST_RESULTS.md](./docs/LIVE_E2E_TEST_RESULTS.md)
for the latest validated run results.

### Railway API Service

The project includes a FastAPI service for receiving events from AWS infrastructure:

```bash
# Start the Railway API service locally
uvicorn action_item_graph.api.main:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health

# Process an envelope (requires WORKER_API_KEY)
curl -X POST http://localhost:8000/process \
  -H "Authorization: Bearer $WORKER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "...", "user_id": "...", ...}'
```

The service initializes Neo4j and OpenAI clients at startup and dispatches envelopes through `EnvelopeDispatcher` (both Action Item and Deal pipelines).

### Lambda Packaging & Deployment

Build and deploy the Lambda event forwarder:

```bash
# Package Lambda for arm64 deployment
./scripts/package_lambda.sh
# Output: dist/action-item-graph-ingest.zip (~19MB)

# Deploy with Pulumi (recommended)
cd infra && pulumi up --stack prod

# Or deploy manually (without IaC)
aws lambda update-function-code \
  --function-name action-item-graph-ingest \
  --zip-file fileb://dist/action-item-graph-ingest.zip
```

All AWS resources (EventBridge, SQS, Lambda, IAM, Secrets Manager) are managed by Pulumi IaC in `infra/`. The `WORKER_API_KEY` is stored in AWS Secrets Manager and fetched at Lambda cold start. See [Event Consumer Architecture](./docs/EVENT_CONSUMER_ARCHITECTURE.md) for details.

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
    # Quality pipeline metrics
    pre_consolidation_count: int   # Items before consolidation
    post_consolidation_count: int  # Items after consolidation
    items_consolidated: int        # Items removed by consolidation
    pre_verification_count: int    # Items before verification
    post_verification_count: int   # Items after verification
    items_rejected: int            # Items rejected by verifier
    rejection_reasons: list[dict]  # Why items were rejected
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
│   │   ├── openai_client.py  # OpenAI API wrapper
│   │   └── postgres_client.py # Postgres dual-write (SQLAlchemy 2.0 + asyncpg)
│   ├── models/
│   │   ├── envelope.py       # EnvelopeV1 input format
│   │   ├── action_item.py    # ActionItem, ActionItemVersion
│   │   ├── topic.py          # ActionItemTopic, ActionItemTopicVersion, ExtractedTopic
│   │   └── entities.py       # Account, Owner, Contact, etc.
│   ├── prompts/
│   │   ├── extract_action_items.py  # F-CoT extraction + commitment framework
│   │   ├── merge_action_items.py    # Merge synthesis prompts
│   │   ├── topic_prompts.py         # Topic resolution prompts
│   │   ├── consolidation_prompts.py # Within-batch dedup prompts
│   │   ├── verification_prompts.py  # LLM-as-Judge verification prompts
│   │   └── owner_prompts.py         # Role-to-name resolution prompts
│   ├── pipeline/
│   │   ├── extractor.py      # Action item extraction + priority scoring
│   │   ├── consolidator.py   # Within-batch dedup (embedding clustering)
│   │   ├── verifier.py       # LLM-as-Judge adversarial verification
│   │   ├── owner_resolver.py # Account-scoped owner pre-resolution
│   │   ├── matcher.py        # Graduated threshold matching
│   │   ├── merger.py         # Merge decision execution
│   │   ├── topic_resolver.py # Topic matching logic
│   │   ├── topic_executor.py # Topic creation/linking
│   │   └── pipeline.py       # Main 10-stage orchestrator
│   ├── api/                     # Railway FastAPI service
│   │   ├── main.py              # App with lifespan (Neo4j + OpenAI init)
│   │   ├── config.py            # Service configuration (pydantic-settings)
│   │   ├── auth.py              # Bearer token authentication
│   │   └── routes/
│   │       ├── health.py        # GET /health
│   │       └── process.py       # POST /process
│   └── lambda_ingest/           # AWS Lambda forwarder
│       ├── handler.py           # Powertools BatchProcessor entry point
│       ├── config.py            # Lambda configuration (pydantic-settings)
│       ├── secrets.py           # Secrets Manager fetch + cold-start caching
│       ├── envelope.py          # EventBridge wrapper parser
│       └── api_client.py        # Railway API client with retry
├── tests/
│   ├── test_pipeline.py           # AI pipeline end-to-end
│   ├── test_extractor.py          # F-CoT extraction + scoring
│   ├── test_consolidator.py       # Within-batch dedup (17 tests)
│   ├── test_verifier.py           # LLM-as-Judge verification (10 tests)
│   ├── test_owner_resolver.py     # Owner pre-resolution (29 tests)
│   ├── test_scoring.py            # Priority scoring + graduated thresholds (17 tests)
│   ├── test_deal_pipeline.py      # Deal pipeline end-to-end
│   ├── test_deal_extraction.py    # MEDDIC extraction
│   ├── test_deal_matcher.py       # Deal entity resolution
│   ├── test_deal_merger.py        # Deal merge synthesis
│   ├── test_dispatcher.py         # Dual-pipeline dispatch
│   ├── test_postgres_client.py     # PostgresClient unit tests (38 tests)
│   ├── test_pipeline_dual_write.py # Dual-write pipeline integration (28 tests)
│   ├── test_duplicate_text.py     # Duplicate-text regression
│   ├── test_uuid7.py              # UUIDv7 identity tests
│   ├── test_lambda_secrets.py     # Secrets Manager fetch + caching (8 tests)
│   └── ...                        # 30+ test files, 471 tests total
├── examples/
│   ├── process_transcript.py # Basic usage example
│   ├── run_transcript_tests.py # Transcript test runner
│   └── transcripts/          # Real transcript testing
├── src/deal_graph/              # Deal extraction pipeline (→ shared DB)
│   ├── utils.py                 # uuid7() wrapper (UUIDv7 via fastuuid)
│   ├── config.py                # DEAL_NEO4J_* env vars, thresholds
│   ├── repository.py            # DealRepository (graph CRUD)
│   ├── pipeline/                # DealExtractor, DealMatcher, DealMerger, DealPipeline
│   ├── clients/                 # DealNeo4jClient (schema, vector search)
│   ├── models/                  # Deal, DealVersion, ExtractedDeal, MergedDeal
│   └── prompts/                 # MEDDIC extraction, dedup, merge prompts
├── src/shared/                  # Cross-pipeline shared utilities
│   └── contact_ops.py           # ENGAGED_ON MERGE, role enrichment, contact matching
├── src/dispatcher/              # Dual-pipeline envelope dispatcher
│   └── dispatcher.py            # EnvelopeDispatcher, DispatcherResult
├── scripts/
│   ├── run_live_e2e.py          # Live E2E smoke test (dual-pipeline)
│   └── package_lambda.sh        # Lambda deployment packaging
├── docs/
│   ├── API.md                   # API reference
│   ├── PIPELINE_GUIDE.md        # Comprehensive pipeline guide
│   ├── PHASE7_TOPIC_GROUPING.md # Topic feature documentation
│   ├── DEAL_SERVICE_ARCHITECTURE.md # Deal pipeline architecture
│   ├── SMOKE_TEST_GUIDE.md      # Comprehensive smoke test & E2E guide
│   ├── LIVE_E2E_TEST_RESULTS.md # E2E smoke test validation record
│   └── test-data-report.md      # Test results report
├── infra/                       # Pulumi IaC for AWS resources
│   ├── __main__.py              # Pulumi entry point
│   ├── forwarder.py             # Reusable EventBridge→SQS→Lambda pattern
│   ├── Pulumi.yaml              # Project definition
│   ├── Pulumi.prod.yaml         # Stack config (encrypted secrets)
│   └── requirements.txt         # Pulumi Python dependencies
├── migrations/
│   └── 001_add_scoring_columns.sql  # Scoring columns for Postgres (run before deploy)
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
