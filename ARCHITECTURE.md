# Action Item Graph - Architecture

## Overview

A system for extracting structured intelligence (action items and deals) from call transcripts, persisting to a shared Neo4j knowledge graph. Two enrichment pipelines — ActionItemPipeline and DealPipeline — run concurrently via `EnvelopeDispatcher`, sharing a single Neo4j AuraDB instance and OpenAI client. A third upstream pipeline (eq-structured-graph-core) also writes to the same database; all three are peers that converge on shared nodes.

---

## Core Principles

These principles govern all architectural decisions. They should be preserved when extending the system.

### 1. Protect the Structured Graph

The shared Neo4j database is the source of truth. When integrating a new pipeline, the pipeline adapts to fit the existing schema — not the other way around. Established conventions (property names, constraint types, label naming) take precedence over pipeline-local preferences.

### 2. Three Pipelines as Peers

Three independent pipelines write to the same database:

| Pipeline | Repo | What it creates |
|----------|------|-----------------|
| **Structured Graph** (eq-structured-graph-core) | Separate repo | Skeleton nodes (Account, Interaction, Contact, Deal, CalendarWeek) + flesh (Entity, Topic, Chunk, Community) |
| **Action Item Pipeline** (this repo) | `src/action_item_graph/` | ActionItem, ActionItemVersion, Owner, ActionItemTopic, ActionItemTopicVersion |
| **Deal Pipeline** (this repo) | `src/deal_graph/` | Deal, DealVersion |

No pipeline is upstream of another. All three receive the same payload from an upstream ingestion service. Any pipeline may execute first. This means every pipeline must handle the case where shared nodes (Account, Interaction) don't yet exist.

### 3. Schema Ownership

Each pipeline owns the labels it creates and is responsible for their constraints, indexes, and schema evolution:

| Owner | Labels | Constraint Authority |
|-------|--------|---------------------|
| **eq-structured-graph-core** (skeleton) | Account, Interaction, Contact, Deal, CalendarWeek, Entity, Topic, Chunk, Community | Upstream — we never create or drop constraints on these |
| **AI pipeline** | ActionItem, ActionItemVersion, Owner, ActionItemTopic, ActionItemTopicVersion | `Neo4jClient.setup_schema()` |
| **Deal pipeline** | Deal, DealVersion | `DealNeo4jClient.setup_schema()` |

Shared labels (Account, Interaction) are skeleton-owned. Enrichment pipelines write to these nodes via defensive MERGE but never create or modify constraints on them.

### 4. Defensive MERGE for Shared Nodes

When multiple pipelines write to the same node type (Account, Interaction), each uses `MERGE ... ON CREATE SET ... ON MATCH SET`:

- **ON CREATE SET**: Populates base properties if this pipeline creates the node first
- **ON MATCH SET**: Only sets enrichment properties (e.g., `action_item_count`, `deal_count`, `processed_at`) — never overwrites skeleton-owned properties

This makes execution order irrelevant. Whether the structured graph pipeline, the AI pipeline, or the deal pipeline reaches the node first, the end state converges.

### 5. Pipeline Fault Isolation

`EnvelopeDispatcher` uses `asyncio.gather(return_exceptions=True)`. One pipeline crashing never blocks or cancels the other. The `DispatcherResult` captures either a successful result or the exception from each pipeline independently. `overall_success` is `True` when at least one pipeline returned a result.

### 6. Dual Embedding Strategy

Both pipelines maintain two embedding vectors per entity to prevent "embedding drift":

- **`embedding`** (immutable): Captures original semantic meaning. Used to catch similar NEW items.
- **`embedding_current`** (mutable): Updated when the entity evolves. Used to catch STATUS UPDATES to existing items.

Both indexes are searched during matching; results are deduplicated by primary key, keeping the higher score.

### 7. Label-Specific Primary Keys

Every node type uses a label-specific primary key property (not a generic `id`):

| Label | Key Property |
|-------|-------------|
| Account | `account_id` |
| Interaction | `interaction_id` |
| ActionItem | `action_item_id` |
| Owner | `owner_id` |
| ActionItemTopic | `action_item_topic_id` |
| Deal | `opportunity_id` |

This prevents ambiguity in a shared database where multiple labels coexist. All MERGE operations use `{tenant_id: $tenant_id, <label_key>: $value}`.

### 8. UNIQUENESS Constraints (Not NODE KEY)

All enrichment pipelines use `IS UNIQUE` constraints, matching the structured database convention. NODE KEY (which adds existence enforcement) is stronger but requires coordination with eq-structured-graph-core to align system-wide.

---

## System Architecture

```
                     Upstream Publishers
          ┌──────────────────┬───────────────────┐
          │                  │                   │
  live-transcription  eq-email-pipeline   eq-structured-graph-core
          │                  │                   │
          └────────┬─────────┘                   │
                   │                              │
                   ▼                              │
          ┌────────────────┐                     │
          │  EventBridge   │                     │
          │  (default bus) │                     │
          └───────┬────────┘                     │
                  │                              │
    ┌─────────────┴─────────────┐               │
    │                           │               │
    ▼                           ▼               │
┌──────────────┐     ┌──────────────────┐       │
│ thematic-lm  │     │ action-item-graph│       │
│ SQS + Lambda │     │ SQS + Lambda     │       │
│ (existing)   │     │ (new)            │       │
└──────────────┘     └────────┬─────────┘       │
                              │                 │
                    HTTPS POST /process          │
                              │                 │
                              ▼                 │
                    ┌──────────────────┐        │
                    │ Railway FastAPI  │        │
                    │ EnvelopeDispatcher│       │
                    └────────┬─────────┘       │
                             │                 │
               ┌─────────────┴──────────────┐  │
               │                            │  │
               ▼                            ▼  ▼
    ┌─────────────────────┐     ┌──────────────────┐
    │ Action Item Pipeline│     │  Deal Pipeline    │
    └──────────┬──────────┘     └──────────┬───────┘
               │                           │
               └─────────────┬─────────────┘
                             ▼
                    ┌────────────────┐
                    │ Single Neo4j   │
                    │ AuraDB         │
                    └────────────────┘
```

Within this repo, `EnvelopeDispatcher` runs the Action Item and Deal pipelines concurrently:

```
                         EnvelopeV1
                  (transcript, tenant_id,
                   account_id, metadata)
                             |
                             v
                  ┌─────────────────────┐
                  │  EnvelopeDispatcher  │
                  │  asyncio.gather()    │
                  └──────────┬──────────┘
                             |
               ┌─────────────┴─────────────┐
               |                           |
               v                           v
    ┌─────────────────────┐     ┌─────────────────────┐
    │ ActionItemPipeline  │     │    DealPipeline      │
    └──────────┬──────────┘     └──────────┬──────────┘
               |                           |
               └─────────────┬─────────────┘
                             v
                  ┌─────────────────────┐
                  │   Shared Neo4j DB   │
                  └─────────────────────┘
```

**Entry points**: `POST /process` in `src/action_item_graph/api/routes/process.py` (Railway API), or `EnvelopeDispatcher.dispatch(envelope)` in `src/dispatcher/dispatcher.py` (direct)

See [docs/DEAL_SERVICE_ARCHITECTURE.md](./docs/DEAL_SERVICE_ARCHITECTURE.md) for detailed Deal pipeline architecture.

---

## Event Consumer Architecture

The event consumer system connects upstream publishers to the dual-pipeline processor:

### Data Flow

```
EventBridge event → SQS queue → Lambda forwarder → Railway FastAPI → EnvelopeDispatcher
```

1. **EventBridge rule** matches events from `com.yourapp.transcription` (transcripts, notes, meetings) and `com.eq.email-pipeline` (emails)
2. **SQS queue** (`action-item-graph-queue`) buffers events with 720s visibility timeout and a DLQ (`action-item-graph-dlq`, maxReceiveCount=3)
3. **Lambda forwarder** (`action-item-graph-ingest`) parses the EventBridge wrapper, extracts the `detail` (EnvelopeV1 JSON), and POSTs it to Railway
4. **Railway FastAPI service** validates the envelope, initializes pipelines from persistent clients, and dispatches through `EnvelopeDispatcher`

### Railway API Service

| Aspect | Detail |
|--------|--------|
| Module | `src/action_item_graph/api/` |
| Framework | FastAPI with async lifespan |
| Endpoints | `GET /health` (Neo4j connectivity), `POST /process` (envelope processing) |
| Auth | Bearer token via `WORKER_API_KEY` |
| Clients | Neo4jClient + DealNeo4jClient + OpenAIClient (initialized once at startup, stored in `app.state`) |
| Config | pydantic-settings (`Settings` class in `api/config.py`) |
| Start command | `uvicorn action_item_graph.api.main:app --host 0.0.0.0 --port $PORT` |

### Lambda Forwarder

| Aspect | Detail |
|--------|--------|
| Module | `src/action_item_graph/lambda_ingest/` |
| Runtime | Python 3.11, arm64, 256MB, 120s timeout |
| Tools | AWS Lambda Powertools (BatchProcessor, Logger, Tracer) |
| Retry | 5xx: exponential backoff + jitter (max 2 retries). 4xx: no retry |
| Config | pydantic-settings (`LambdaConfig` in `lambda_ingest/config.py`) |
| Packaging | `scripts/package_lambda.sh` → `dist/action-item-graph-ingest.zip` (~19MB) |
| Dependencies | pydantic, pydantic-settings, httpx, aws-lambda-powertools[tracer] |

### Design Decisions

- **Own SQS queue** (not sharing with thematic-lm): SQS is point-to-point — sharing would cause message competition
- **Thin Lambda → persistent service**: Processing takes 60-90s with OpenAI API calls. Railway keeps Neo4j/OpenAI connections warm; Lambda has no state to manage
- **BatchSize=1**: Each envelope triggers 60-90s of processing. Batching would increase Lambda execution time without benefit
- **Synchronous processing**: Lambda waits for Railway's response. On success, SQS deletes the message. On failure, SQS retries up to 3x before DLQ

---

## Single Shared Neo4j Database

Both pipelines write to the same Neo4j database instance. Label namespacing and schema ownership ensure isolation:

| Aspect | ActionItem Pipeline | Deal Pipeline |
|--------|---------------------|---------------|
| Env vars | `NEO4J_URI` (fallback to `DEAL_NEO4J_URI`), `NEO4J_PASSWORD` | Same |
| Client class | `Neo4jClient` | `DealNeo4jClient` |
| Schema owner | `Neo4jClient.setup_schema()` | `DealNeo4jClient.setup_schema()` |
| Node labels | Account, Interaction, ActionItem, ActionItemVersion, Owner, ActionItemTopic, ActionItemTopicVersion | Deal, DealVersion, Interaction, Account |
| Constraints | UNIQUENESS on label-specific keys (e.g., `action_item_id`, `action_item_topic_id`) | UNIQUENESS on `(tenant_id, opportunity_id)`, etc. |
| Vector indexes | 4 (ActionItem + ActionItemTopic x embedding/embedding_current) | 2 (Deal x embedding/embedding_current) |

The shared labels (Account, Interaction) use defensive MERGE operations with `ON CREATE` / `ON MATCH` clauses to ensure idempotency when both pipelines write to the same nodes.

---

## Action Item Graph Schema

### Node Types

All nodes include `tenant_id` for multi-tenancy isolation. Constraints enforce uniqueness on label-specific primary key properties.

```
(:Account)
  - account_id: string (primary key, e.g., "acct_acme_corp_001")
  - tenant_id: UUID
  - name: string
  - domain: string (optional)
  - created_at: datetime
  - last_interaction_at: datetime

(:Interaction)
  - interaction_id: UUID (primary key)
  - tenant_id: UUID
  - account_id: string
  - interaction_type: enum (transcript, note, document, email, meeting)
  - title: string (optional)
  - content_text: string
  - timestamp: datetime
  - duration_seconds: int (optional)
  - source: string (web-mic, upload, api, import)
  - user_id: string
  - processed_at: datetime
  - action_item_count: int

(:ActionItem)
  - action_item_id: UUID (primary key)
  - tenant_id: UUID
  - account_id: string
  - action_item_text: string (verbatim from transcript)
  - summary: string (LLM-generated)
  - owner: string
  - conversation_context: string
  - due_date: datetime (optional)
  - status: enum (open, in_progress, completed, cancelled, deferred)
  - version: int
  - evolution_summary: string
  - created_at: datetime
  - last_updated_at: datetime
  - source_interaction_id: UUID
  - user_id: string
  - embedding: float[] (original, immutable)
  - embedding_current: float[] (updated on changes)
  - confidence: float
  - valid_at: datetime
  - invalid_at: datetime (optional)
  - invalidated_by: UUID (optional)

(:ActionItemVersion)
  - version_id: UUID (primary key)
  - action_item_id: UUID
  - tenant_id: UUID
  - version: int
  - action_item_text: string
  - summary: string
  - owner: string
  - status: enum
  - due_date: datetime (optional)
  - change_summary: string
  - change_source_interaction_id: UUID
  - created_at: datetime
  - valid_from: datetime
  - valid_until: datetime (optional)

(:Owner)
  - owner_id: UUID (primary key)
  - tenant_id: UUID
  - canonical_name: string
  - aliases: string[]
  - contact_id: string (optional)
  - user_id: string (optional)
  - created_at: datetime

(:ActionItemTopic)
  - action_item_topic_id: UUID (primary key)
  - tenant_id: UUID
  - account_id: string
  - name: string (display name, 3-5 words)
  - canonical_name: string (normalized: lowercase, trimmed)
  - summary: string (LLM-generated, evolves over time)
  - embedding: float[] (original, immutable)
  - embedding_current: float[] (updated on significant changes)
  - action_item_count: int (denormalized count)
  - created_at: datetime
  - updated_at: datetime
  - created_from_action_item_id: UUID (provenance)
  - version: int

(:ActionItemTopicVersion)
  - version_id: UUID (primary key)
  - action_item_topic_id: UUID
  - tenant_id: UUID
  - version_number: int
  - name: string
  - summary: string
  - embedding_snapshot: float[]
  - changed_by_action_item_id: UUID
  - created_at: datetime
```

### Relationships

```
(:Account)-[:HAS_INTERACTION]->(:Interaction)
(:Account)-[:HAS_ACTION_ITEM]->(:ActionItem)
(:Account)-[:HAS_TOPIC]->(:ActionItemTopic)

(:ActionItem)-[:EXTRACTED_FROM {is_source: bool, confidence: float}]->(:Interaction)
  # An ActionItem can be EXTRACTED_FROM multiple Interactions (grows over time)
  # is_source=true for the original extraction, false for subsequent references

(:ActionItem)-[:OWNED_BY {created_at: datetime}]->(:Owner)
(:ActionItem)-[:HAS_VERSION]->(:ActionItemVersion)
(:ActionItem)-[:INVALIDATES]->(:ActionItem)
  # When an action item supersedes another
(:ActionItem)-[:RELATED_TO]->(:ActionItem)
  # When items are semantically related but distinct

(:ActionItem)-[:BELONGS_TO {confidence: float, method: string, created_at: datetime}]->(:ActionItemTopic)
  # method: "extracted" (from extraction), "resolved" (matched to existing), "manual"

(:ActionItemTopic)-[:HAS_VERSION {version_number: int}]->(:ActionItemTopicVersion)

(:Contact)-[:PARTICIPATED_IN]->(:Interaction)
  # Contacts participate in interactions, NOT directly linked to ActionItems
```

### Visual Schema

```
                              ┌──────────┐
                              │ Account  │
                              └────┬─────┘
           ┌──────────────────────┼──────────────────────┬────────────────────────┐
           │                      │                      │                        │
           ▼                      ▼                      ▼                        ▼
    ┌─────────────┐        ┌───────────┐          ┌──────────┐     ┌──────────────────────┐
    │ Interaction │◄───────│ActionItem │─────────►│  Owner   │     │  ActionItemTopic     │
    └─────────────┘        └─────┬─────┘          └──────────┘     └────┬─────────────────┘
                                 │   │                                   │
                                 │   └───────────────BELONGS_TO──────────┘
                                 ▼                                       │
                          ┌───────────────────┐               ┌──────────┴──────────────┐
                          │ActionItemVersion  │               │ActionItemTopicVersion   │
                          └───────────────────┘               └─────────────────────────┘
```

---

## Deal Pipeline Graph Schema

See [docs/DEAL_SERVICE_ARCHITECTURE.md](./docs/DEAL_SERVICE_ARCHITECTURE.md) for the complete Deal pipeline architecture, data models, and design decisions.

### Node Types (Deal Pipeline)

```
(:Deal)
  - opportunity_id: UUID v7 (primary key, minted by deal_graph.utils.uuid7())
  - deal_ref: string (display alias: "deal_" + hex[-16:])
  - tenant_id: UUID
  - account_id: string
  - name: string
  - stage: enum (prospecting, qualification, proposal, negotiation, closed_won, closed_lost)
  - amount: float (optional)
  - currency: string (default: USD)
  - opportunity_summary: string (LLM-generated, evolves)
  - evolution_summary: string (accumulated change narrative)
  - version: int
  - confidence: float
  - source_interaction_id: UUID
  - embedding: float[] (original, immutable)
  - embedding_current: float[] (updated on merge)
  - created_at: datetime
  - last_updated_at: datetime
  - MEDDIC fields: meddic_metrics, meddic_economic_buyer, meddic_decision_criteria,
                    meddic_decision_process, meddic_identified_pain, meddic_champion
  - meddic_completeness: float (0.0-1.0)

(:DealVersion)
  - version_id: UUID
  - deal_opportunity_id: UUID (parent Deal)
  - tenant_id: UUID
  - version: int
  - name, stage, amount, opportunity_summary (snapshot)
  - MEDDIC field snapshots
  - change_summary: string
  - changed_fields: string[] (normalized property names)
  - change_source_interaction_id: UUID
  - created_at: datetime
  - valid_from: datetime
  - valid_until: datetime (optional)

(:Interaction) — shared node, enriched by Deal pipeline
  - interaction_id: UUID
  - tenant_id: UUID
  - account_id: string
  - interaction_type: string
  - deal_count: int (enriched after processing)
  - processed_at: datetime

(:Account) — shared node
  - account_id: string
  - tenant_id: UUID
```

### Relationships (Deal Pipeline)

```
(:Deal)-[:HAS_VERSION]->(:DealVersion)
```

Other associations use shared properties rather than graph edges (e.g., `Deal.account_id`, `Deal.source_interaction_id`).

---

## Constraints & Indexes

### Action Item Schema (AI-owned labels)

| Type | Name | Target |
|------|------|--------|
| UNIQUE | `action_item_unique` | ActionItem (tenant_id, action_item_id) |
| UNIQUE | `action_item_version_unique` | ActionItemVersion (tenant_id, version_id) |
| UNIQUE | `owner_unique` | Owner (tenant_id, owner_id) |
| UNIQUE | `action_item_topic_unique` | ActionItemTopic (tenant_id, action_item_topic_id) |
| UNIQUE | `action_item_topic_version_unique` | ActionItemTopicVersion (tenant_id, version_id) |
| RANGE | `action_item_tenant_idx` | ActionItem (tenant_id) |
| RANGE | `action_item_account_idx` | ActionItem (account_id) |
| RANGE | `action_item_status_idx` | ActionItem (status) |
| RANGE | `owner_tenant_idx` | Owner (tenant_id) |
| RANGE | `action_item_topic_tenant_idx` | ActionItemTopic (tenant_id) |
| RANGE | `action_item_topic_account_idx` | ActionItemTopic (account_id) |
| RANGE | `action_item_topic_canonical_name_idx` | ActionItemTopic (canonical_name) |
| VECTOR | `action_item_embedding_idx` | ActionItem.embedding (1536d, cosine) |
| VECTOR | `action_item_embedding_current_idx` | ActionItem.embedding_current (1536d, cosine) |
| VECTOR | `action_item_topic_embedding_idx` | ActionItemTopic.embedding (1536d, cosine) |
| VECTOR | `action_item_topic_embedding_current_idx` | ActionItemTopic.embedding_current (1536d, cosine) |

UNIQUENESS constraints ensure that label-specific primary keys are unique within each tenant. Shared labels (Account, Interaction) use skeleton constraints owned by upstream schema authority.

### Deal Schema (Deal-owned labels)

| Type | Name | Target |
|------|------|--------|
| UNIQUE | `dealversion_unique` | DealVersion (tenant_id, version_id) |
| RANGE | `deal_stage_idx` | Deal (tenant_id, stage) |
| RANGE | `deal_account_idx` | Deal (tenant_id, account_id) |
| VECTOR | `deal_embedding_idx` | Deal.embedding (1536d, cosine) |
| VECTOR | `deal_embedding_current_idx` | Deal.embedding_current (1536d, cosine) |

### Shared Labels (Skeleton constraints)

Skeleton constraints on Account and Interaction are owned by the upstream schema authority (`eq-structured-graph-core`). Both pipelines use defensive MERGE operations on these labels.

---

## Dual Embedding Strategy

Both pipelines use the same dual-embedding approach to prevent "embedding drift":

1. **`embedding`** (immutable): Generated from the original text when first extracted. Used to catch semantically similar NEW items.

2. **`embedding_current`** (mutable): Updated when the item evolves significantly. Used to catch status updates to EXISTING items.

### Matching Flow (Action Items)

```
New extraction arrives
        |
        v
Generate embedding for new text
        |
        +---> Search `embedding` index ---> Find similar original items
        |
        +---> Search `embedding_current` index ---> Find similar current-state items
                |
                v
        Combine results, deduplicate
                |
                v
        LLM decides: Same item? Status update? New item?
                |
        +-------+-------+
        v       v       v
    Merge   Update   Create
            Status    New
```

### Matching Flow (Deals)

Deals use graduated thresholds:

| Score Range | Decision | LLM Call? |
|-------------|----------|-----------|
| >= 0.90 | `auto_match` | No |
| 0.70 - 0.90 | LLM decides | Yes |
| < 0.70 | `create_new` | No |

---

## Multi-Tenancy & Account Scoping

### Scoping Rules

| Node Type | tenant_id | account_id | Notes |
|-----------|-----------|------------|-------|
| Account | Required | N/A | Is the account |
| Interaction | Required | Required | Scoped to account |
| ActionItem | Required | Required | Scoped to account |
| ActionItemTopic | Required | Required | Scoped to account |
| Owner | Required | - | Tenant-level (shared across accounts) |
| ActionItemVersion | Required | - | Historical record |
| ActionItemTopicVersion | Required | - | Historical record |
| Deal | Required | Required | Scoped to account |
| DealVersion | Required | - | Historical record |

### Isolation Rules

1. **tenant_id** is required on ALL nodes for complete data isolation
2. **account_id** is required on transaction-level nodes (ActionItem, ActionItemTopic, Deal, Interaction)
3. **Vector searches** filter by both tenant_id AND account_id to prevent cross-account bleeding
4. **Owner nodes** are tenant-scoped (not account-scoped) to allow name resolution across accounts
5. **Version nodes** inherit tenant_id from parent but don't need account_id (historical snapshots)
6. **Shared labels** (Account, Interaction) are written defensively using MERGE with ON CREATE / ON MATCH to ensure idempotency across both pipelines

### Query Scoping

All repository methods enforce scoping:

```cypher
-- Example: Get action items (requires both tenant_id and account_id)
MATCH (ai:ActionItem {tenant_id: $tenant_id, account_id: $account_id})
RETURN ai

-- Example: Vector search with scoping
CALL db.index.vector.queryNodes('action_item_embedding_idx', 10, $embedding)
YIELD node, score
WHERE node.tenant_id = $tenant_id AND node.account_id = $account_id
RETURN node, score
```

---

## Temporal Tracking

Inspired by Graphiti's bi-temporal model:

- **`valid_at`**: When this version of truth became valid
- **`invalid_at`**: When this version was superseded
- **`invalidated_by`**: UUID of the superseding action item

This enables:
- Time-travel queries ("What was the status on date X?")
- Audit trails
- Understanding action item evolution

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Graph Database | Neo4j Aura 5.x (Enterprise) — single shared instance |
| LLM | OpenAI GPT-4.1-mini (structured output) |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Data Validation | Pydantic 2.x |
| Async | asyncio + tenacity for retries |
| Logging | structlog (structured JSON) |
| Identity | UUIDv7 (RFC 9562) via fastuuid for Deal opportunity_id |
| Package Manager | uv |
| API Framework | FastAPI 0.115+ with uvicorn (Railway service) |
| Lambda Tools | AWS Lambda Powertools (BatchProcessor, Logger, Tracer) |
| HTTP Client | httpx (Lambda → Railway forwarding) |
| Configuration | pydantic-settings (both Railway and Lambda) |

---

## Related Documentation

- [Deal Service Architecture](./docs/DEAL_SERVICE_ARCHITECTURE.md) — Deal pipeline stages, MEDDIC extraction, data models
- [Graph Integration Proposal](./docs/GRAPH_INTEGRATION_PROPOSAL.md) — Feasibility analysis for single-database architecture, concurrency safety proofs
- [Smoke Test Guide](./docs/SMOKE_TEST_GUIDE.md) — E2E validation procedures and historical results
- [Live E2E Test Results](./docs/LIVE_E2E_TEST_RESULTS.md) — Most recent live run validation record
- [Event Consumer Design](./docs/plans/2026-02-22-event-consumer-design.md) — EventBridge → SQS → Lambda → Railway architecture design
- [Pipeline Guide](./docs/PIPELINE_GUIDE.md) — Comprehensive pipeline usage guide
- [Topic Grouping](./docs/PHASE7_TOPIC_GROUPING.md) — Topic feature documentation
- [API Reference](./docs/API.md) — API reference
