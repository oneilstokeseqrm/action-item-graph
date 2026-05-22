# Changelog

All notable changes to the Action Item Graph project.

## [0.3.0] - 2026-05-22

### Added - DBOS Durable Workflow Migration

#### Architecture
- **DBOS workflow orchestration** for the ingest path — Lambda dispatcher returns 200 in sub-second time after enqueueing both pipelines; pipeline duration no longer bounded by Lambda's 120s timeout
- **Per-step retry with deterministic replay** — each LLM call, Neo4j MERGE, and Postgres write is a checkpointed `@DBOS.step` with explicit retry policy; transient OpenAI/Neon errors no longer cascade into full-pipeline failures
- **DBOS workflow registry** as the operator surface — failed workflows visible in `dbos.workflow_status` with full input/output, replay capability, and per-step granularity (replaces the DLQ as the primary operator surface)
- **No upper bound on pipeline duration** within DBOS's 15-minute workflow timeout (Open #19 / `workflow_timeout=900`)

#### New Modules
- `src/action_item_graph/dbos_runtime.py` — DBOS substrate initialization + FastAPI lifespan integration
- `src/action_item_graph/workflows/` — action-item pipeline as 14-step DBOS workflow (S1-S14 with S9/S10 LLM-vs-write splits per Codex #10) + queues + serialization helpers + client registry
- `src/deal_graph/workflows/` — deal pipeline as 9-step DBOS workflow (with D7 inner merger refactor)

#### New Infrastructure
- Neon database `eq_aig_dbos_sys` (direct non-pooler connection) for DBOS workflow state
- Pulumi secret `dbos-system-database-url` for Lambda → Neon connection
- CloudWatch metric `partial_enqueue_pair_count` for first-succeeds-second-fails split-window detection

#### Test Coverage
- 87 new tests across the migration (567 → 654 net delta vs main)
- Workflow body validation gates, authoritative interaction_id (Rule 5), per-step fail-open contracts
- Behavioral contract tests for the workflow path (T25a, scoped down to 5 contracts per the scope decision documented in `docs/plans/2026-05-20-DBOS-MIGRATION-HANDOFF.md`)
- Compatibility tests for Phase 1 mid-state (T32) + Phase 2 deploy-order mistakes (T33) + W2 concurrent execution (T34)
- Idempotency tests for the deterministic-UUID5 retry-safety pattern (Topic + TopicVersion + Deal CREATEs)
- Cross-phase architectural cohesion test (`test_api_main.py::TestDBOSWorkflowRegistration`) that pins both workflow names in the DBOS registry post-import

### Changed

- **Lambda dispatcher** (`src/action_item_graph/lambda_ingest/handler.py`) — rewritten as `DBOSClient.enqueue` dispatcher; partial-enqueue split window emits CloudWatch metric before re-raising for SQS redelivery
- **Repository idempotency for retry safety** — all CREATE-with-randomUUID patterns reached from retryable DBOS steps converted to MERGE on deterministic UUID5 keys:
  - `create_version_snapshot` (action-item + deal): UUID5 over `(entity_id, source_interaction_id, content_hash)`
  - `create_topic`: UUID5 over `(tenant_id, account_id, canonical_name, source_action_item_id)`
  - `create_topic_version`: UUID5 over `(topic_id, changed_by_action_item_id)`
  - Deal `_create_new`: UUID5 over `(tenant_id, source_interaction_id, content_hash)` (was uuid7)
- **Merger refactor** — `_merge_items` / `_merge_existing` split into `construct_*_llm` + `persist_*_neo4j` at the function level for DBOS retry safety; legacy entry paths preserved during the migration window
- **Owner resolver** (`OwnerPreResolver.resolve_batch`) — returns NEW ActionItem instances instead of mutating in-place (DBOS replay safety)
- **Workflow ID format** locked at `f"action-item-graph:{pipeline}:interaction-{uuid}"`

### Fixed

- **`deal_workflow` reads `opportunity_id` from `extras`** to match `EnvelopeV1.opportunity_id` @property — Case A targeted-deal flows would have silently fallen through to discovery mode under the prior workflow code path
- **`deal_workflow` registered in production import chain** — the module was never imported in production, so its `@DBOS.workflow()` decoration never executed and the DBOS workflow registry was missing `deal_workflow`. Phase 2 deploy would have manifested as deal pipeline silently inert. Fix: side-effect import in `action_item_graph/workflows/__init__.py`. Caught by full-PR codex review after 12 prior phase-scoped reviews missed it.
- **Topic deterministic-ID collapse** — earlier deterministic-ID work over `(tenant, account, canonical_name)` collapsed legitimate batch-mate Topics into one node (action_item_count under-reported, summary reflecting only first item). Fix: include `source_action_item_id` in the key — preserves retry safety AND legacy "two distinct Topics per same-canonical-name in one batch" parity.

### Removed

- `src/action_item_graph/lambda_ingest/api_client.py` — HTTP forwarder retired (Lambda no longer makes HTTP calls; `httpx` dropped from the Lambda zip)
- `secret_arn` Pulumi export — dead block after Phase C removed `worker-api-key` from the secrets dict; grep audit confirmed zero external consumers

### Migration Notes

This is **Phase A through Phase E** of a 3-phase deploy. Phase D (retire `/process` HTTP route + `dispatcher.py`) ships as a separate PR Day 14+ post-deploy. The 2-week rollback window depends on `/process` staying alive during Phase 1+2. See `docs/plans/2026-05-20-DBOS-MIGRATION-HANDOFF.md` for the full migration arc, locked decisions, and the deletion-seam list for the Phase D follow-up.

**T29 deploy gate:** requires `pulumi config set --secret dbos-system-database-url <DIRECT_URL>` + Railway `DBOS_SYSTEM_DATABASE_URL` env var. **T30 post-deploy:** parked DLQ message (`58863f20-3cda-48f7-973d-3002aa31331b`) redrives via `aws sqs start-message-move-task` as the live integration test.

## [0.2.0] - 2026-01-25

### Added - Topic Grouping (Phase 7)

#### New Components
- `TopicResolver` - Matches extracted topics to existing using threshold-based logic
- `TopicExecutor` - Creates topics, links action items, generates evolving summaries
- `Topic` model - High-level theme/project node with dual embeddings
- `TopicVersion` model - Historical snapshot for topic evolution tracking
- `ExtractedTopic` model - Topic extracted alongside action items

#### New Graph Elements
- `(:Topic)` node with dual embeddings (`embedding`, `embedding_current`)
- `(:TopicVersion)` node for version history
- `(:Account)-[:HAS_TOPIC]->(:Topic)` relationship
- `(:ActionItem)-[:BELONGS_TO]->(:Topic)` relationship
- `topic_embedding_idx` and `topic_embedding_current_idx` vector indexes

#### Topic Resolution Thresholds
| Threshold | Value | Behavior |
|-----------|-------|----------|
| Auto-link | >= 0.85 | Automatically link to existing topic |
| Auto-create | < 0.70 | Automatically create new topic |
| LLM confirm | 0.70-0.85 | Use LLM to confirm match |

#### Pipeline Integration
- Topic extraction happens during action item extraction (same LLM call)
- Topic resolution runs after action item merging
- `PipelineResult` includes `topics_created` and `topics_linked` counts
- Feature flag: `enable_topics=True/False` on pipeline initialization

#### Documentation
- `docs/PHASE7_TOPIC_GROUPING.md` - Feature documentation
- `docs/PIPELINE_GUIDE.md` - Comprehensive architecture guide
- `docs/test-data-report.md` - Test results report

### Fixed
- Scoping issue in `get_topic_with_action_items()` - now filters action items by tenant_id

### Changed
- `ExtractedActionItem` now includes required `topic` field
- Pipeline stages now include `topic_resolution` in timing
- Updated all tests to include topic field in test data

---

## [0.1.0] - 2026-01-23

### Added

#### Core Pipeline
- `ActionItemPipeline` - Main orchestrator for extraction, matching, and merging
- `ActionItemExtractor` - GPT-4.1-mini powered extraction with structured output
- `ActionItemMatcher` - Vector similarity search + LLM deduplication
- `ActionItemMerger` - Execute merge decisions (create, merge, update_status, link)
- `ActionItemRepository` - Graph CRUD operations

#### Models
- `EnvelopeV1` - API input payload format
- `ActionItem` - Core action item model with dual embeddings
- `ActionItemVersion` - Version history tracking
- `Account`, `Interaction`, `Owner`, `Contact`, `Deal` - Graph entity models
- `ExtractedActionItem`, `StatusUpdate` - Extraction output models

#### Clients
- `Neo4jClient` - Async Neo4j client with connection pooling and retries
- `OpenAIClient` - Async OpenAI client with structured output support

#### Features
- **Dual-mode extraction**: Extracts both new action items and status updates
- **Dual embeddings**: `embedding` (immutable original) + `embedding_current` (mutable)
- **Multi-tenancy**: `tenant_id` on all nodes with query-level isolation
- **Temporal tracking**: `valid_at`, `invalid_at`, version history
- **Vector search**: Neo4j native vector indexes for similarity matching
- **LLM deduplication**: GPT-4.1-mini decides same/related/new

#### Observability
- Structured logging with `structlog`
- `PipelineTimer` for stage-level timing
- `logging_context()` for trace ID propagation
- Confidence scores on extractions

#### Error Handling
- Typed exception hierarchy (`ActionItemGraphError` → specialized errors)
- `PartialSuccessResult` for batch operations
- Error wrapping utilities for OpenAI and Neo4j

#### Testing
- 60+ tests covering all pipeline stages
- Live integration tests with Neo4j and OpenAI
- Real transcript testing infrastructure (`examples/transcripts/`)

#### Documentation
- `README.md` - Project overview and quick start
- `ARCHITECTURE.md` - Detailed graph schema and design
- `REQUIREMENTS.md` - Functional and non-functional requirements
- `docs/API.md` - Comprehensive API reference
- `examples/transcripts/README.md` - Transcript testing guide

### Technical Details

#### Graph Schema
```
(:Account)-[:HAS_INTERACTION]->(:Interaction)
(:Account)-[:HAS_ACTION_ITEM]->(:ActionItem)
(:ActionItem)-[:EXTRACTED_FROM]->(:Interaction)
(:ActionItem)-[:OWNED_BY]->(:Owner)
(:ActionItem)-[:HAS_VERSION]->(:ActionItemVersion)
```

#### Vector Indexes
- `action_item_embedding_idx` - Original embeddings (1536 dimensions, cosine)
- `action_item_embedding_current_idx` - Current state embeddings

#### LLM Configuration
- Extraction: `gpt-4.1-mini` with structured output
- Embeddings: `text-embedding-3-small` (1536 dimensions)
- Deduplication: `gpt-4.1-mini` with Literal type constraints

#### Matching Threshold
- Default similarity threshold: 0.65 (cosine)
- Searches both `embedding` and `embedding_current` indexes
- LLM makes final same/related/new decision

### Dependencies
- Python 3.10+
- neo4j >= 5.26.0
- openai >= 1.50.0
- pydantic >= 2.11.0
- structlog >= 24.0.0
- tenacity >= 9.0.0
- numpy >= 1.26.0

---

## Future Considerations

### Potential Enhancements
- Batch processing optimization
- Custom embedding model support
- Real-time streaming extraction
- Owner resolution to Contact records
- Confidence threshold tuning
- Parallel matching for large batches

### Known Limitations
- Maximum transcript size: ~100KB
- Single LLM provider (OpenAI)
- No UI/Dashboard
- No automated CRM import
