# Deal Extraction Service — Implementation Plan

> **Version**: 3.0
> **Date**: 2026-01-31
> **Status**: Approved — strict contract for coding phase (aligned with eq-structured-graph-core schema authority)
> **Scope**: New parallel pipeline for extracting Sales Opportunities (Deals) from transcripts using the MEDDIC qualification methodology

---

## Table of Contents

1. [Investigation Summary](#1-investigation-summary)
2. [Reference Framework Adoption](#2-reference-framework-adoption)
3. [Architecture Overview](#3-architecture-overview)
4. [Directory & Module Structure](#4-directory--module-structure)
5. [Configuration](#5-configuration)
6. [Graph Schema (Deal Database)](#6-graph-schema-deal-database)
7. [Data Models](#7-data-models)
8. [Pipeline Flow](#8-pipeline-flow)
9. [MEDDIC Merge Rules](#9-meddic-merge-rules)
10. [LLM Prompts](#10-llm-prompts)
11. [Implementation Phases](#11-implementation-phases)
12. [Testing Strategy](#12-testing-strategy)
13. [Strategic Recommendations](#13-strategic-recommendations)
14. [Critique & Trade-offs](#14-critique--trade-offs)
15. [Critical Files Reference](#15-critical-files-reference)

---

## 1. Investigation Summary

### Codebase Analysis

The repository (`action-item-graph` v0.2.0) is a production-grade temporal knowledge graph pipeline built on patterns from the **OpenAI Cookbook temporal graph demos** and **Zep's Graphiti library** (`docs/GRAPHITI_REFERENCE.md`).

**Technology stack:**
- Python 3.10+, fully async
- OpenAI: `gpt-4.1-mini` (extraction/synthesis), `text-embedding-3-small` (1536-dim embeddings)
- Neo4j 5.x with native vector search (cosine similarity)
- Pydantic 2.x for data validation, `structlog` for logging, `tenacity` for retries

**Existing Action Item pipeline flow:**
```
EnvelopeV1 → Validate → Ensure Account → Extract (LLM structured output)
→ Embed → Dual Vector Match → LLM Dedup → Merge/Create → Topic Resolution
→ PipelineResult
```

### MEDDIC Framework

MEDDIC (Metrics, Economic Buyer, Decision Criteria, Decision Process, Identify Pain, Champion) is a B2B sales qualification methodology developed at PTC in the 1990s. Each component maps to a structured field with a confidence score. Modern AI-augmented implementations capture these fields automatically from call transcripts and correlate completeness with deal predictability.

Extended variants (MEDDPICC) add Paper Process and Competition — we include schema slots for future extension.

---

## 2. Reference Framework Adoption

This service is built on the same two reference frameworks that informed the Action Item pipeline. Each pattern is **actively adopted** with deal-domain-specific adaptations — not passively inherited.

### Source Frameworks

| Framework | Reference | Role |
|---|---|---|
| **OpenAI Cookbook — Temporal Agents with Knowledge Graphs** | [github.com/openai/openai-cookbook/...temporal_agents_with_knowledge_graphs](https://github.com/openai/openai-cookbook/tree/main/examples/partners/temporal_agents_with_knowledge_graphs) | Foundational pipeline architecture: extract → resolve → build graph. Structured graph construction from unstructured text. |
| **Zep Graphiti** | [github.com/getzep/graphiti](https://github.com/getzep/graphiti) | Temporal knowledge graph library. Bi-temporal model, entity deduplication via LLM, episodic memory, hybrid search, edge invalidation. |
| **Local Reference** | `docs/GRAPHITI_REFERENCE.md` | Curated patterns already adopted by the Action Item pipeline. |

### Pattern Adoption Map

| # | Pattern | Source | Deal Service Adoption | Deal-Specific Adaptation |
|---|---|---|---|---|
| 1 | **Bi-Temporal Data Model** | Graphiti `nodes.py` | `DealVersion` with `valid_from`, `valid_until` | `changed_fields` list for machine-readable deltas |
| 2 | **Dual Embedding Strategy** | Graphiti + Cookbook | `embedding` (immutable) + `embedding_current` (mutable) on `Deal` | Embedding text = `"name: summary"` — captures deal identity, not MEDDIC detail |
| 3 | **Entity Deduplication via LLM** | Graphiti `dedupe_nodes.py` | `DealDeduplicationDecision` structured output | Higher thresholds (0.70/0.90 vs 0.65/0.85) — false merge is costlier for deals |
| 4 | **Threshold-Based Matching** | Graphiti + Cookbook | Three-band system: auto-create / LLM-zone / auto-match | Wider LLM zone (0.70-0.90) — more LLM involvement for high-stakes entities |
| 5 | **Episodic Memory** | Graphiti Episodes | `Interaction` stores full `content_text` — the deal's episodic memory | Enables re-processing and future "Deal Coaching" without cross-DB queries |
| 6 | **Temporal Edge Invalidation** | Graphiti `invalidate_edges.py` | MEDDIC field replacement (economic_buyer, amount) invalidates prior version | Coexists with additive composition — some fields accumulate, others replace |
| 7 | **Version Snapshots** | Graphiti + Cookbook | `DealVersion` created before every update | Includes `change_summary` (narrative) + `changed_fields` (structured) |
| 8 | **Structured Output for LLM** | Graphiti (Pydantic models) | All LLM responses are Pydantic `BaseModel` with `Field` descriptions | Strictly typed for Postgres future-proofing |
| 9 | **Provenance Tracking** | Graphiti (source edges) | `source_interaction_id` on `Deal` + `change_source_interaction_id` on `DealVersion` | Skeleton `[:RELATED_TO]` already links interactions to deals |
| 10 | **Multi-Tenancy Scoping** | Graphiti `group_id` | `tenant_id + account_id` on all nodes; enforced in all vector queries | Identical to Action Item pipeline |
| 11 | **Evolving Summaries** | Graphiti + Cookbook | `opportunity_summary` regenerated on updates; `evolution_summary` is cumulative narrative | Evolution summary explains **why** (contextual synthesis), not just **what** (data diff) |
| 12 | **Confidence Scores** | Graphiti | Per-extraction and per-MEDDIC-field confidence (0.0-1.0) | Confidence ratcheting: low-confidence fields upgraded when explicitly confirmed |

### Patterns NOT Adopted in V1 (with rationale)

| Pattern | Source | Why Not in V1 |
|---|---|---|
| BM25 Keyword Search | Graphiti hybrid search | Dual-embedding + LLM judgment is the proven approach from the Action Item pipeline. BM25 adds complexity without proven benefit for our data. See [Search Strategy Constraint](#search-strategy). |
| Reciprocal Rank Fusion (RRF) | Graphiti hybrid search | Same rationale. Max-score dedup + LLM judgment replaces statistical rank fusion. |
| Topic/Theme Clustering | Action Item pipeline | Deals are naturally singleton entities. Theme clustering may be added in a future phase for portfolio-level views. |
| `link_related` merge recommendation | Action Item pipeline | Deals are either the same opportunity (merge) or different (create new). No lateral linking needed in V1. |

---

## 3. Architecture Overview

### Enrichment Architecture

**The Deal pipeline connects to the existing `neo4j_structured` database** managed by `eq-structured-graph-core`.

- Skeleton nodes (`Deal`, `Interaction`, `Account`, `Contact`) and their base relationships are **created by the schema authority's skeleton layer** (`eq-structured-graph-core`)
- Our pipeline **enriches** those skeleton nodes with MEDDIC properties, deal versioning, embeddings, and qualification data
- **Codebase isolation** — we replicate schema definitions in our own Pydantic models; we do **not** import from `eq-structured-graph-core`
- The existing `Deal` model in `action_item_graph/models/entities.py` is legacy ghost code and must be ignored

### Skeleton vs Enrichment

The schema authority creates minimal stubs with base properties. Our pipeline adds the "flesh":

| Layer | Created By | Example Properties |
|---|---|---|
| **Skeleton** | `eq-structured-graph-core` | `Deal.tenant_id`, `Deal.opportunity_id`, `Deal.name`, `Deal.stage`, `Deal.amount` |
| **Enrichment** | Our pipeline | `Deal.meddic_*`, `Deal.embedding`, `Deal.embedding_current`, `Deal.evolution_summary`, `Deal.version` |
| **Skeleton** | `eq-structured-graph-core` | `Interaction.tenant_id`, `Interaction.interaction_id`, `Interaction.content_text`, `Interaction.timestamp` |
| **Enrichment** | Our pipeline | `Interaction.deal_count`, `Interaction.processed_at` |

### Episodic Memory (Zep/Graphiti Pattern)

The `Interaction` nodes in the database store the **full transcript text** as `content_text`. This follows Graphiti's "Episode" pattern where each discrete data input is self-contained within the graph. This ensures:

1. **Re-processability** — deals can be re-extracted from stored episodes if prompts or models change
2. **Future "Deal Coaching"** — an agent can read the full interaction history for a deal without cross-service dependencies
3. **Self-contained enrichment** — our pipeline reads `content_text` from existing `Interaction` nodes rather than querying external storage

### What IS Shared (Infrastructure Only)

| Shared Component | Source | Rationale |
|---|---|---|
| `OpenAIClient` | `action_item_graph.clients.openai_client` | Same API key, same models — no reason to duplicate |
| Base `Neo4jClient` class | `action_item_graph.clients.neo4j_client` | Connection management, retry logic, `vector_search()`, `search_both_embeddings()` are generic. New service inherits and overrides `setup_schema()` |
| `structlog` logging | `action_item_graph.logging` | `get_logger`, `logging_context`, `PipelineTimer` are infrastructure |
| Base error hierarchy | `action_item_graph.errors` | Subclass for deal-specific errors |
| `EnvelopeV1` input model | `action_item_graph.models.envelope` | The input format is identical — same transcript, same metadata |

### What is NOT Shared (Everything Else)

All Pydantic data models, pipeline components, prompts, and repository code are new and live exclusively in `src/deal_graph/`. Graph schema definitions are **replicated** (not imported) from the schema authority to maintain codebase isolation.

### Parallel Dispatch

```
EnvelopeV1
    │
    ▼
EnvelopeDispatcher.process_envelope()
    │
    ├──► ActionItemPipeline.process_envelope()  ─► Action Item Neo4j DB
    │         (existing, UNTOUCHED)
    │
    └──► DealPipeline.process_envelope()        ─► neo4j_structured DB
              (new service, enriches skeleton)        (shared with eq-structured-graph-core)
```

- `asyncio.gather()` runs both pipelines concurrently
- Each pipeline wrapped in `_safe_process_*()` — one failure does not block the other
- `DispatchResult` aggregates both results with per-pipeline error tracking
- The existing Action Item pipeline is **strictly untouched** — zero modifications

---

## 4. Directory & Module Structure

### New Packages

```
src/
├── action_item_graph/          # EXISTING — DO NOT MODIFY
│   └── ...
│
├── deal_graph/                 # NEW — Deal extraction service
│   ├── __init__.py             # Public exports: DealPipeline, DealPipelineResult
│   ├── config.py               # DealConfig (DEAL_NEO4J_* env vars, thresholds)
│   ├── errors.py               # Deal-specific error subclasses
│   ├── repository.py           # DealRepository (Cypher CRUD for Deal, DealVersion, etc.)
│   ├── clients/
│   │   ├── __init__.py
│   │   └── neo4j_client.py     # DealNeo4jClient (inherits Neo4jClient, overrides setup_schema)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deal.py             # Deal, DealVersion, MEDDICProfile, DealStage
│   │   └── extraction.py       # ExtractedDeal, DealExtractionResult
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # DealPipeline orchestrator
│   │   ├── extractor.py        # DealExtractor (MEDDIC extraction via LLM)
│   │   ├── matcher.py          # DealMatcher (dual-embedding entity resolution)
│   │   └── merger.py           # DealMerger (synthesis, versioning, delta tracking)
│   └── prompts/
│       ├── __init__.py
│       ├── extract_deals.py    # MEDDIC extraction system/user prompts
│       ├── merge_deals.py      # Deal synthesis/evolution prompts
│       └── dedup_deals.py      # Deal deduplication prompts
│
└── dispatcher/                 # NEW — Thin orchestration layer
    ├── __init__.py
    └── envelope_dispatcher.py  # EnvelopeDispatcher, DispatchResult
```

### New Test Files

```
tests/
├── test_deal_extraction.py       # MEDDIC extraction from transcripts
├── test_deal_matcher.py          # Deal entity resolution
├── test_deal_merger.py           # Deal merge synthesis + versioning
├── test_deal_pipeline.py         # End-to-end deal pipeline
├── test_deal_repository.py       # Graph CRUD operations
├── test_deal_neo4j_client.py     # Schema setup, vector search
└── test_envelope_dispatcher.py   # Parallel processing
```

---

## 5. Configuration

### New Environment Variables

Add to `.env.example` (the existing variables remain untouched):

```bash
# -----------------------------------------------------------------------------
# Deal Graph Configuration (connects to existing neo4j_structured instance)
# -----------------------------------------------------------------------------
DEAL_NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
DEAL_NEO4J_USERNAME=neo4j
DEAL_NEO4J_PASSWORD=your-deal-password
DEAL_NEO4J_DATABASE=neo4j

# Deal Pipeline Thresholds (optional)
# DEAL_SIMILARITY_THRESHOLD=0.70
# DEAL_AUTO_MATCH_THRESHOLD=0.90
# DEAL_LOG_LEVEL=INFO
```

### DealConfig Class

`src/deal_graph/config.py` — reads `DEAL_NEO4J_*` variables. These connect to the **existing `neo4j_structured` instance** managed by `eq-structured-graph-core`. Follows the same pattern as `action_item_graph/config.py` but with the `DEAL_` prefix. Includes `validate()` method that checks `DEAL_NEO4J_URI` and `DEAL_NEO4J_PASSWORD` are set.

### Database Connection Strategy

The `DEAL_NEO4J_*` variables point to the same `neo4j_structured` database where `eq-structured-graph-core` creates skeleton nodes. The code connects via standard Neo4j driver protocol using the credentials in `.env`. Our pipeline adds enrichment properties and new node types (`DealVersion`) — it does not duplicate or conflict with skeleton-managed schema elements.

---

## 6. Graph Schema (Deal Database)

### Schema Authority

> **IMPORTANT**: The `neo4j_structured` database is managed by `eq-structured-graph-core` (the **schema authority**). Skeleton nodes (`Deal`, `Interaction`, `Account`, `Contact`) and their base constraints/relationships already exist. Our pipeline adds enrichment properties to these nodes and introduces one new node type (`DealVersion`). We do NOT re-create skeleton constraints — they are `IF NOT EXISTS` no-ops.

### Skeleton Properties (created by eq-structured-graph-core)

These properties already exist on nodes in the database. Our pipeline reads and respects them:

**Deal** (minimal stub): `tenant_id`, `opportunity_id`, `name`*, `stage`*, `amount`* (*Optional, may be NULL)
**Interaction** (full): `tenant_id`, `interaction_id`, `trace_id`, `interaction_type`, `source`, `timestamp`, `content_text`, `content_format`
**Account**: `tenant_id`, `account_id`, `name`*, `industry`*, `status`*
**Contact**: `tenant_id`, `contact_id`, `name`*, `email`*, `role`*

### Skeleton Relationships (created by eq-structured-graph-core)

```
(Interaction)-[:BELONGS_TO]->(Account)
(Interaction)-[:RELATED_TO]->(Deal)         # when opportunity_id present
(Deal)-[:RELATED_TO]->(Account)             # when both opportunity_id + account_id present
(Contact)-[:WORKS_FOR]->(Account)
(Contact)-[:SENT|CREATED]->(Interaction)    # author
(Contact)-[:RECEIVED|ATTENDED]->(Interaction) # participants
(Interaction)-[:HAPPENED_IN]->(CalendarWeek)
```

### Node Labels

#### `Deal`

**Skeleton properties** (created by eq-structured-graph-core, already on node):

| Property | Type | Required | Description |
|---|---|---|---|
| `tenant_id` | UUID (string) | Yes | Multi-tenancy isolation (composite key part) |
| `opportunity_id` | string | Yes | Primary key (composite with `tenant_id`) |
| `name` | string | No* | Descriptive opportunity name (*skeleton may leave NULL) |
| `stage` | string | No* | Current deal stage (see DealStage enum) |
| `amount` | float | No | Estimated or confirmed deal value |
| `account_id` | string | No | Account-level scoping |

**Enrichment properties** (added by our pipeline):

| Property | Type | Required | Description |
|---|---|---|---|
| `currency` | string | Yes | Currency code (default: `USD`) |
| `opportunity_summary` | string | Yes | LLM-generated summary, evolves over time |
| `embedding` | float[1536] | Yes | Immutable original embedding |
| `embedding_current` | float[1536] | Yes | Mutable embedding, updated on significant changes |
| `version` | int | Yes | Incremented on each update |
| `evolution_summary` | string | Yes | Cumulative **narrative** of how and **why** the deal evolved |
| `confidence` | float | Yes | Extraction confidence (0.0-1.0) |
| `source_interaction_id` | UUID (string) | No | Original interaction that created this deal |
| `created_at` | datetime | Yes | When first created |
| `last_updated_at` | datetime | Yes | Last modification timestamp |
| `expected_close_date` | datetime | No | Projected close date |
| `closed_at` | datetime | No | Actual close date |
| **MEDDIC Fields** (flattened) | | | |
| `meddic_metrics` | string | No | Quantifiable business impact |
| `meddic_metrics_confidence` | float | No | 0.0-1.0 |
| `meddic_economic_buyer` | string | No | Person with final budget authority (e.g., `"Sarah Jones, VP Engineering"`) |
| `meddic_economic_buyer_confidence` | float | No | 0.0-1.0 |
| `meddic_decision_criteria` | string | No | Technical/business evaluation criteria |
| `meddic_decision_criteria_confidence` | float | No | 0.0-1.0 |
| `meddic_decision_process` | string | No | Steps/timeline to reach decision |
| `meddic_decision_process_confidence` | float | No | 0.0-1.0 |
| `meddic_identified_pain` | string | No | Core business problem |
| `meddic_identified_pain_confidence` | float | No | 0.0-1.0 |
| `meddic_champion` | string | No | Internal advocate (e.g., `"James Park, Director of Sales Ops"`) |
| `meddic_champion_confidence` | float | No | 0.0-1.0 |
| `meddic_completeness` | float | No | Computed: populated fields / 6 (0.0-1.0) |
| **Future Slots** | | | |
| `forecast_score` | float | No | Probabilistic close probability |
| `qualification_status` | string | No | `qualified` / `unqualified` |
| `meddic_paper_process` | string | No | MEDDPICC extension |
| `meddic_competition` | string | No | MEDDPICC extension |

> **CONSTRAINT — Flattened MEDDIC Properties**: Champion, Economic Buyer, and all MEDDIC entities are stored as **string properties** on `Deal` (e.g., `meddic_champion: "Sarah Jones"`). Do NOT create separate graph nodes for these entities (no `(:Person)` or `(:Stakeholder)` nodes). Person Entity Resolution is out of scope for V1.

#### `DealVersion` (our addition — not in schema authority)

| Property | Type | Required | Description |
|---|---|---|---|
| `version_id` | UUID (string) | Yes | Primary key (composite with `tenant_id`) |
| `deal_opportunity_id` | string (UUIDv7) | Yes | Parent Deal's `opportunity_id` |
| `tenant_id` | UUID (string) | Yes | Multi-tenancy |
| `version` | int | Yes | Version number at snapshot time |
| `name` | string | Yes | Snapshot of deal name |
| `stage` | string | Yes | Snapshot of stage |
| `amount` | float | No | Snapshot of amount |
| `opportunity_summary` | string | Yes | Snapshot of summary |
| `evolution_summary` | string | Yes | Snapshot of cumulative evolution narrative |
| `meddic_metrics` | string | No | Snapshot of MEDDIC metrics |
| `meddic_economic_buyer` | string | No | Snapshot of economic buyer |
| `meddic_decision_criteria` | string | No | Snapshot of decision criteria |
| `meddic_decision_process` | string | No | Snapshot of decision process |
| `meddic_identified_pain` | string | No | Snapshot of identified pain |
| `meddic_champion` | string | No | Snapshot of champion |
| `meddic_completeness` | float | No | Snapshot of completeness score |
| `change_summary` | string | Yes | LLM-generated **narrative** explaining why the deal changed |
| `changed_fields` | string[] | Yes | Machine-readable list of changed property names (e.g., `["meddic_champion", "stage", "amount"]`) |
| `change_source_interaction_id` | UUID (string) | No | Which interaction triggered the change |
| `created_at` | datetime | Yes | When snapshot was taken |
| `valid_from` | datetime | Yes | Start of validity window |
| `valid_until` | datetime | No | End of validity window (null if current) |

#### `Interaction`

**Skeleton properties** (created by eq-structured-graph-core, already on node):

| Property | Type | Required | Description |
|---|---|---|---|
| `tenant_id` | UUID (string) | Yes | Multi-tenancy (composite key part) |
| `interaction_id` | string | Yes | Primary key (composite with `tenant_id`) — generated upstream |
| `trace_id` | string | Yes | Trace identifier for observability |
| `interaction_type` | string | Yes | `transcript`, `note`, `document` |
| `source` | string | No | Origin (`web-mic`, `upload`, `api`, `import`) |
| `timestamp` | datetime | Yes | When the interaction took place |
| `content_text` | string | Yes | **Full raw transcript text** — the deal's episodic memory |
| `content_format` | string | No | Format of the content (e.g., `text/plain`) |

**Enrichment properties** (added by our pipeline):

| Property | Type | Required | Description |
|---|---|---|---|
| `processed_at` | datetime | No | When deals were extracted |
| `deal_count` | int | No | Number of deals extracted |

> **CONSTRAINT — Episodic Memory**: The `Interaction` node's `content_text` property (created by the skeleton layer) contains the full raw transcript text. This follows the **Zep/Graphiti "Episode" pattern** ensuring the Deal service has complete episodic memory. Our pipeline reads `content_text` from existing `Interaction` nodes — it does not need to write it.

#### `Contact` (skeleton-managed, read-only for our pipeline in V1)

| Property | Type | Required | Description |
|---|---|---|---|
| `tenant_id` | UUID (string) | Yes | Multi-tenancy (composite key part) |
| `contact_id` | string | Yes | Primary key (composite with `tenant_id`) |
| `name` | string | No | Contact name |
| `email` | string | No | Email address |
| `role` | string | No | Job title/role |

#### `Account` (skeleton-managed, read-only for our pipeline in V1)

| Property | Type | Required | Description |
|---|---|---|---|
| `tenant_id` | UUID (string) | Yes | Multi-tenancy (composite key part) |
| `account_id` | string | Yes | Primary key (composite with `tenant_id`) |
| `name` | string | No | Company name |
| `industry` | string | No | Industry classification |
| `status` | string | No | Account status |

### Relationships

```
# SKELETON (already created by eq-structured-graph-core — do not duplicate):
(Interaction)-[:RELATED_TO]->(Deal)
(Deal)-[:RELATED_TO]->(Account)
(Interaction)-[:BELONGS_TO]->(Account)
(Contact)-[:WORKS_FOR]->(Account)
(Contact)-[:SENT|CREATED]->(Interaction)
(Contact)-[:RECEIVED|ATTENDED]->(Interaction)

# OUR ENRICHMENT (additive, do not conflict with skeleton):
(Deal)-[:HAS_VERSION]->(DealVersion)         # NEW — our addition
(Deal)-[:HAS_CONTACT]->(Contact)             # NEW — future-proofing, not populated in V1
```

**Provenance tracking** (replaces v2.0 `EXTRACTED_FROM`):
- The skeleton's `(Interaction)-[:RELATED_TO]->(Deal)` already links interactions to deals — we do NOT duplicate this with a second relationship type
- `Deal.source_interaction_id` tracks the original "creating" interaction
- `DealVersion.change_source_interaction_id` tracks which interaction triggered each version change

### Constraints

```cypher
# Existing (managed by eq-structured-graph-core, already in DB — these are IF NOT EXISTS no-ops):
CREATE CONSTRAINT deal_unique IF NOT EXISTS FOR (n:Deal) REQUIRE (n.tenant_id, n.opportunity_id) IS UNIQUE
CREATE CONSTRAINT interaction_unique IF NOT EXISTS FOR (n:Interaction) REQUIRE (n.tenant_id, n.interaction_id) IS UNIQUE
CREATE CONSTRAINT account_unique IF NOT EXISTS FOR (n:Account) REQUIRE (n.tenant_id, n.account_id) IS UNIQUE
CREATE CONSTRAINT contact_unique IF NOT EXISTS FOR (n:Contact) REQUIRE (n.tenant_id, n.contact_id) IS UNIQUE

# NEW (our addition — not in schema authority):
CREATE CONSTRAINT dealversion_unique IF NOT EXISTS FOR (n:DealVersion) REQUIRE (n.tenant_id, n.version_id) IS UNIQUE
```

> **Note — Live DB Constraint Fix Required**: The live DB has a stale `deal_unique` constraint on `(tenant_id, deal_id)` from before the schema authority renamed `deal_id` → `opportunity_id`. An operator must run `DROP CONSTRAINT deal_unique` and then re-run `eq-structured-graph-core`'s `setup_db.py` to recreate with the correct property. This is outside our pipeline's scope.

### Regular Indexes

```cypher
# Existing (backing indexes from skeleton constraints — no-ops):
# deal_unique backing index covers (tenant_id, opportunity_id)
# interaction_unique backing index covers (tenant_id, interaction_id)

# NEW (our additions for query performance):
CREATE INDEX deal_stage_idx IF NOT EXISTS FOR (n:Deal) ON (n.tenant_id, n.stage)
CREATE INDEX deal_account_idx IF NOT EXISTS FOR (n:Deal) ON (n.tenant_id, n.account_id)
```

### Vector Indexes

```cypher
CREATE VECTOR INDEX deal_embedding_idx IF NOT EXISTS
FOR (n:Deal) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}

CREATE VECTOR INDEX deal_embedding_current_idx IF NOT EXISTS
FOR (n:Deal) ON (n.embedding_current)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
```

---

## 7. Data Models

> **CONSTRAINT — Postgres Future-Proofing**: All Pydantic models (`Deal`, `DealVersion`) must be **strictly typed** and keyed by **UUIDs** (`opportunity_id`, `source_interaction_id`, `version_id`, `change_source_interaction_id`). These models are designed to map **1:1 to SQL tables** to support a future Postgres synchronization feature. All fields must use explicit types (no `Any`), and all IDs must be `UUID`.

> **Future Extension — Postgres Sync**: The `Deal` Pydantic model maps directly to a `deals` SQL table, and `DealVersion` maps to a `deal_versions` table. UUID primary keys, strict typing, and flat property structures ensure seamless ORM mapping. The `meddic_*` fields are individual columns, not a JSON blob, enabling SQL queries like `WHERE meddic_completeness < 0.5`.

> **CONSTRAINT — Schema Authority Alignment**: Pydantic models replicate skeleton property names from `eq-structured-graph-core` (e.g., `opportunity_id`, `amount`, `content_text`, `role`) plus our enrichment fields. We do not import from the schema authority — we replicate definitions for codebase isolation.

### MEDDICProfile

Pydantic `BaseModel` stored as flattened properties on `Deal`.

> **CONSTRAINT**: All MEDDIC entities (Champion, Economic Buyer, etc.) are **string properties**, not graph nodes. No `(:Person)` nodes in V1. Example: `meddic_champion: "Sarah Jones, Director of Sales Ops"`.

Each MEDDIC dimension: `str` value field + `float` confidence field (0.0–1.0).

Computed property: `completeness_score` = (non-empty fields) / 6.

Method: `to_neo4j_properties()` → flattened dict with `meddic_` prefix.

Future extension slots: `paper_process`, `competition` (MEDDPICC).

### Deal (Pydantic Model)

Corresponds to the Neo4j `Deal` label. Contains skeleton properties + enrichment:
- `opportunity_id: str` — primary key (composite with `tenant_id`), from schema authority
- `tenant_id: UUID`
- `account_id: str`
- Core deal fields (skeleton): `name: str`, `stage: DealStage`, `amount: float | None`
- Core deal fields (enrichment): `currency: str`
- `meddic: MEDDICProfile`
- `opportunity_summary: str` (LLM-generated, evolves)
- `embedding: list[float] | None` (immutable original, 1536-dim)
- `embedding_current: list[float] | None` (mutable, updated on significant changes)
- `version: int` (incremented on updates)
- `evolution_summary: str` (cumulative LLM-generated **narrative** of deal progression — explains **why**, not just what)
- `confidence: float` (extraction confidence 0.0–1.0)
- `source_interaction_id: UUID | None`
- Timestamps: `created_at: datetime`, `last_updated_at: datetime`, `expected_close_date: datetime | None`, `closed_at: datetime | None`
- Future slots: `forecast_score: float | None`, `qualification_status: str | None`
- Method: `to_neo4j_properties()` → flat dict for Cypher

### DealStage (Enum)

```python
class DealStage(str, Enum):
    PROSPECTING = 'prospecting'
    QUALIFICATION = 'qualification'
    PROPOSAL = 'proposal'
    NEGOTIATION = 'negotiation'
    CLOSED_WON = 'closed_won'
    CLOSED_LOST = 'closed_lost'
```

Stage regression is **allowed** — this is a shadow forecast. If the LLM sees a deal slipping backward (e.g., negotiation → qualification), it updates the stage accordingly with reasoning captured in `evolution_summary`.

### DealVersion (Pydantic Model)

Full property snapshot of Deal at a specific version. Contains:
- `version_id: UUID` — primary key (composite with `tenant_id`)
- `deal_opportunity_id: str (UUIDv7)` — parent Deal's `opportunity_id`
- `tenant_id: UUID`
- `version: int`
- All snapshotted fields (name, stage, amount, summary, evolution_summary, MEDDIC fields, completeness)
- `change_summary: str` — LLM-generated **narrative** explaining **why** the deal changed (see [Merge Rule 6](#rule-6-evolution-narrative-over-data-diff))
- `changed_fields: list[str]` — machine-readable list of property names that changed (e.g., `["meddic_champion", "stage", "amount"]`)
- `change_source_interaction_id: UUID | None`
- Bi-temporal: `valid_from: datetime`, `valid_until: datetime | None`

### ExtractedDeal (LLM Structured Output)

Per-deal extraction from transcript:
- `opportunity_name: str`
- `opportunity_summary: str` (2-3 sentence overview)
- `stage_assessment: str` (LLM's assessment of current stage)
- MEDDIC fields: `metrics`, `economic_buyer`, `decision_criteria`, `decision_process`, `identified_pain`, `champion` — each `str | None`
- `estimated_amount: float | None` (deal value)
- `currency: str` (default `USD`)
- `expected_close_timeframe: str | None` (freetext, parsed later)
- `confidence: float` (0.0–1.0)
- `reasoning: str` (why this is a deal, what signals were present)

### DealExtractionResult (LLM Structured Output Wrapper)

```python
class DealExtractionResult(BaseModel):
    deals: list[ExtractedDeal] = []
    has_deals: bool
    extraction_notes: str | None = None
```

### DealDeduplicationDecision (LLM Structured Output)

```python
class DealDeduplicationDecision(BaseModel):
    is_same_deal: bool
    recommendation: str  # 'merge' | 'create_new'
    confidence: float    # 0.0-1.0
    reasoning: str
```

### MergedDeal (LLM Structured Output)

```python
class MergedDeal(BaseModel):
    opportunity_summary: str           # Updated summary
    evolution_summary: str             # Cumulative narrative of WHY deal evolved
    change_narrative: str              # This-update-only narrative for DealVersion.change_summary
    changed_fields: list[str]          # Machine-readable: which fields changed
    # Updated MEDDIC fields (None = keep existing, populated = replace/extend)
    metrics: str | None
    economic_buyer: str | None
    decision_criteria: str | None
    decision_process: str | None
    identified_pain: str | None
    champion: str | None
    implied_stage: str | None          # Stage assessment
    stage_reasoning: str | None        # Why stage changed (or didn't)
    amount: float | None               # Updated deal value
    expected_close_date_text: str | None
    should_update_embedding: bool      # Whether summary changed significantly
```

---

## 8. Pipeline Flow

### Search Strategy

> **CONSTRAINT**: The Deal pipeline uses the **Dual Embedding + LLM Judgment** search strategy, lifted exactly from the Action Item pipeline (`matcher.py:148-189`, `neo4j_client.py:299-352`). This V1 plan does **NOT** include BM25 keyword search or Reciprocal Rank Fusion (RRF). The proven dual-embedding cosine approach with LLM semantic judgment is the exclusive search method.

The search implementation:
1. Query `deal_embedding_idx` (original, immutable embeddings) — catches new deals similar to original state
2. Query `deal_embedding_current_idx` (current, mutable embeddings) — catches updates to evolved deals
3. Max-score deduplication by `Deal.opportunity_id` — if same node found in both indexes, keep higher score
4. Graduated thresholds gate LLM involvement
5. LLM deduplication for borderline candidates provides semantic judgment

### DealPipeline.process_envelope(envelope: EnvelopeV1) → DealPipelineResult

```
EnvelopeV1
    │
    ├─ Stage 0: Validate
    │   └─ Require: tenant_id, account_id
    │   └─ Read: opportunity_id from envelope.extras (may be None)
    │
    ├─ Stage 1: Verify Account exists
    │   └─ MATCH Account by (tenant_id, account_id) — skeleton should have created it
    │   └─ If not found: MERGE Account with skeleton base properties
    │
    ├─ Stage 2: MERGE enrichment onto existing Interaction
    │   └─ MATCH Interaction by (tenant_id, interaction_id) — skeleton created it with content_text
    │   └─ Read content_text for MEDDIC extraction
    │   └─ Add enrichment: processed_at, deal_count (after extraction)
    │   └─ This MUST happen before Deal operations
    │
    ├─ BRANCH on opportunity_id
    │
    │   ┌─────────────────────────────────────────────────┐
    │   │  CASE A: opportunity_id IS PROVIDED              │
    │   │  (skeleton already created the Deal stub)        │
    │   │                                                  │
    │   │  ├─ Stage 3a: Fetch existing Deal                │
    │   │  │   └─ MATCH by (tenant_id, opportunity_id)     │
    │   │  │   └─ If not found → error (ID should exist)   │
    │   │  │                                               │
    │   │  ├─ Stage 4a: Extract (Targeted)                │
    │   │  │   └─ LLM prompt: "Update THIS specific deal" │
    │   │  │   └─ Provide existing deal context to LLM    │
    │   │  │   └─ Output: single ExtractedDeal scoped to  │
    │   │  │       this opportunity (NO multi-deal         │
    │   │  │       discovery — V1 constraint)              │
    │   │  │                                               │
    │   │  ├─ Stage 5a: Merge/Update (enrich skeleton)    │
    │   │  │   └─ Create DealVersion snapshot              │
    │   │  │   └─ LLM synthesis (merge existing + new)     │
    │   │  │   └─ SET enrichment properties on Deal        │
    │   │  │   └─ version++ on Deal                        │
    │   │  │   └─ Conditionally update embedding_current   │
    │   │  │                                               │
    │   │  └─ Return DealPipelineResult                    │
    │   └─────────────────────────────────────────────────┘
    │
    │   ┌─────────────────────────────────────────────────┐
    │   │  CASE B: opportunity_id IS NULL (Discovery)      │
    │   │                                                  │
    │   │  ├─ Stage 3b: Extract (Discovery)               │
    │   │  │   └─ LLM prompt: "Find ALL opportunities"    │
    │   │  │   └─ Output: DealExtractionResult             │
    │   │  │       (0, 1, or N ExtractedDeal objects)      │
    │   │  │   └─ Early exit if has_deals=false            │
    │   │  │                                               │
    │   │  ├─ Stage 4b: Generate Embeddings               │
    │   │  │   └─ Embed "name: summary" per deal           │
    │   │  │                                               │
    │   │  ├─ Stage 5b: Match — for each ExtractedDeal:   │
    │   │  │   ├─ Dual vector search (both indexes)        │
    │   │  │   ├─ 0 candidates → skip LLM, create new     │
    │   │  │   ├─ Thresholds:                              │
    │   │  │   │   < 0.70 → create new                    │
    │   │  │   │   0.70–0.90 → LLM decides                │
    │   │  │   │   >= 0.90 → auto-match                   │
    │   │  │   └─ LLM dedup for borderline candidates      │
    │   │  │                                               │
    │   │  ├─ Stage 6b: Merge/Create per deal:            │
    │   │  │   ├─ NEW: MERGE Deal with skeleton base       │
    │   │  │   │   properties (tenant_id, opportunity_id)  │
    │   │  │   │   + SET enrichment properties             │
    │   │  │   │   + SET source_interaction_id             │
    │   │  │   └─ EXISTING:                                │
    │   │  │       ├─ Create DealVersion snapshot           │
    │   │  │       ├─ LLM synthesis (temporal invalidation  │
    │   │  │       │   + evolution narrative)               │
    │   │  │       ├─ SET enrichment on Deal + version++   │
    │   │  │       └─ Conditionally update embedding_curr  │
    │   │  │                                               │
    │   │  └─ Return DealPipelineResult                    │
    │   └─────────────────────────────────────────────────┘
```

### Entity Resolution Thresholds

Deals are high-stakes entities. A false merge (incorrectly combining two different deals) is far worse than a false split (creating a duplicate). Thresholds are stricter than the Action Item pipeline:

| Condition | Action | Rationale |
|---|---|---|
| 0 candidates returned | Create new Deal | Fast-path: skip LLM entirely |
| Similarity < 0.70 | Create new Deal | Below threshold — likely a different deal |
| Similarity 0.70–0.90 | LLM decides | Borderline — needs semantic judgment |
| Similarity >= 0.90 | Auto-match | High confidence — same deal |

Compare: Action Item thresholds are 0.65 / 0.85.

### Embedding Text Strategy

```python
embedding_text = f"{deal.opportunity_name}: {deal.opportunity_summary}"
```

Mirrors the topic embedding pattern (`topic_resolver.py:144`).

### Interaction Enrichment Pattern

The skeleton layer has already created the `Interaction` node with `content_text`. Our pipeline reads it and adds enrichment properties:

```cypher
// Read existing Interaction (created by skeleton)
MATCH (i:Interaction {interaction_id: $interaction_id, tenant_id: $tenant_id})
RETURN i.content_text AS content_text, i.timestamp AS timestamp, i.interaction_type AS interaction_type

// After MEDDIC extraction, add enrichment properties
MATCH (i:Interaction {interaction_id: $interaction_id, tenant_id: $tenant_id})
SET i.processed_at = datetime(),
    i.deal_count = $deal_count
```

If the `Interaction` node does not yet exist (edge case — our pipeline runs before the skeleton), we MERGE with skeleton base properties:

```cypher
MERGE (i:Interaction {interaction_id: $interaction_id, tenant_id: $tenant_id})
ON CREATE SET i.interaction_type = $type,
             i.content_text = $content_text,
             i.timestamp = $timestamp,
             i.source = $source,
             i.created_at = datetime()
ON MATCH SET  i.processed_at = datetime()
```

---

## 9. MEDDIC Merge Rules

These rules govern how the LLM synthesizes a deal update when merging new extraction data with an existing Deal. The merge logic implements **both** temporal invalidation (new values replace old) and evolution narrative (contextual synthesis).

### Rule 1: Never Lose Information (Additive Fields)

If the existing Deal has a MEDDIC field populated and the new extraction does not mention it, **keep the existing value**. The LLM merge prompt must explicitly instruct: "Preserve all existing MEDDIC fields. Only update fields where the new transcript provides new or corrected information."

### Rule 2: Additive Composition (Accumulating Fields)

Some MEDDIC fields naturally accumulate over time. If both existing and new have content for the same field, the LLM synthesizes a combined value.

**Additive fields** (information accumulates):
- `decision_criteria` — new criteria add to existing (e.g., "SOC2 compliance required" + "Must integrate with existing ERP")
- `decision_process` — new steps/stakeholders discovered (e.g., "Board review in Q2" + "Legal review added before board")
- `identified_pain` — additional pain points surface over time

### Rule 3: Temporal Invalidation (Replaceable Fields)

Some MEDDIC fields represent a single entity or value that **replaces** the prior value when new information arrives. This follows **Graphiti's temporal edge invalidation pattern** — the old value is superseded, not accumulated.

**Replaceable fields** (new value supersedes old):
- `economic_buyer` — changes when authority shifts (e.g., "VP of Sales" → "CFO took over budget authority")
- `champion` — changes when advocate shifts (e.g., "Sarah Jones" → "James Park became primary advocate")
- `amount` — replaces when revised (e.g., $200K → $500K)
- `stage` — replaces (with regression allowed)

When a replaceable field changes, the prior value is preserved in the `DealVersion` snapshot and the change is recorded in `changed_fields`.

### Rule 4: Confidence Ratcheting

If a MEDDIC field was previously low-confidence (e.g., 0.3 — inferred) and the new transcript explicitly confirms it, raise the confidence. The merge prompt instructs the LLM to output updated confidence scores.

### Rule 5: Stage Regression Allowed

This is a **shadow forecast**. If the LLM sees evidence that a deal has slipped backward (e.g., new stakeholder raised objections, timeline pushed), it should update the stage accordingly. Stage reasoning is captured in `evolution_summary` and the `DealVersion.change_summary`.

### Rule 6: Evolution Narrative over Data Diff

> **CONSTRAINT**: The `merge_deals.py` prompt must generate a high-quality `evolution_summary` that explains **why** the deal changed (contextual synthesis), not just **what** changed (data diff).

The LLM outputs two distinct narrative fields:

1. **`evolution_summary`** (on `Deal` — cumulative): The full story of the deal's progression. Appended to, not replaced. Example:
   > "Initial discovery call revealed data silo pain points (Jan 15). Sarah Jones from Engineering emerged as champion during technical deep-dive (Jan 22). Budget expanded from $200K to $500K after CEO saw demo and recognized enterprise-wide applicability (Jan 29). Deal advanced from qualification to proposal stage."

2. **`change_narrative`** (becomes `DealVersion.change_summary` — per-update): What changed **in this specific interaction** and why. Example:
   > "Budget expanded from $200K to $500K after CEO attended the demo and recognized the platform's applicability beyond the initial Sales Ops scope. CEO's involvement elevated the economic buyer from VP of Sales to CFO. Deal advanced from qualification to proposal."

The prompt must instruct: "Explain the **business context** driving each change. Do not simply list field diffs. Connect the changes to what was said in the transcript and why it matters for the deal's trajectory."

### Rule 7: Embedding Update Threshold

The LLM outputs `should_update_embedding: bool`. This should be `true` when the `opportunity_summary` has changed substantially (e.g., scope redefined, value significantly different). Minor updates (new MEDDIC field filled, same core deal) do not warrant re-embedding.

---

## 10. LLM Prompts

### MEDDIC Extraction Prompt (`extract_deals.py`)

**System prompt role:** Expert sales analyst specializing in MEDDIC qualification methodology.

**Key instructions:**
- Analyze the transcript for sales opportunities/deals
- For each opportunity, extract all 6 MEDDIC components with specific guidance on what to look for:
  - **Metrics**: ROI projections, time savings, cost reductions, revenue targets
  - **Economic Buyer**: Budget authority, VP/C-level mentions, contract signers
  - **Decision Criteria**: Must-have features, integrations, compliance, comparison criteria
  - **Decision Process**: Approval stages, evaluation timeline, POC requirements
  - **Identify Pain**: Current frustrations, inefficiencies, competitive pressure
  - **Champion**: Internal advocate, meeting organizer, enthusiastic questioner
- Zero-tolerance for fabrication — only extract what is actually discussed
- May find 0, 1, or multiple distinct opportunities in a single transcript
- Set per-field confidence based on how explicitly each element was stated
- Output: `DealExtractionResult` via structured output

**Case A variant** (targeted extraction): Provide existing deal context (current summary, MEDDIC profile, stage) and instruct: "You are updating THIS specific deal. Extract only information relevant to this opportunity. Do not identify new, unrelated deals."

### Deal Deduplication Prompt (`dedup_deals.py`)

Follows Graphiti's dedup pattern (`GRAPHITI_REFERENCE.md:52-84`).

**Compares:** deal name, opportunity summary, MEDDIC profile overlap, stage alignment, account context.

**Bias:** "When uncertain, lean toward `create_new`" — same philosophy as topic resolution in the Action Item pipeline.

**Output:** `DealDeduplicationDecision`

### Deal Merge Synthesis Prompt (`merge_deals.py`)

> **CONSTRAINT**: This prompt must prioritize the **Evolution Narrative**. The LLM must explain **why** the deal changed (contextual synthesis from the transcript), not just **what** changed (data diff).

**Input:** Existing Deal properties (full MEDDIC profile, current summary, evolution_summary, stage) + new ExtractedDeal + transcript excerpt.

**Instructions:**
1. Apply merge rules: distinguish additive fields (decision_criteria, decision_process, identified_pain) from replaceable fields (economic_buyer, champion, value, stage)
2. For replaceable fields: output the new value (temporal invalidation); the system will snapshot the old value automatically
3. For additive fields: synthesize a combined value that integrates old + new information
4. Generate `evolution_summary`: Append to the existing narrative. Explain the **business context** driving each change. Connect changes to what was said in the transcript and why it matters for the deal's trajectory.
5. Generate `change_narrative`: Describe **this specific update** with business context (this becomes `DealVersion.change_summary`)
6. Output `changed_fields`: list every property name that was modified
7. Assess stage: should it progress, regress, or stay? Provide `stage_reasoning`
8. Assess embedding: did the core deal identity change enough to warrant re-embedding?

**Output:** `MergedDeal`

---

## 11. Implementation Phases

### Phase 1: Foundation

**Goal:** Create the package skeleton and data models.

- Create `src/deal_graph/` package structure with all `__init__.py` files
- Create `src/dispatcher/` package structure
- Implement `deal_graph/config.py` — `DealConfig` class
- Implement `deal_graph/errors.py` — `DealPipelineError`, `DealExtractionError`, `DealMatchingError`, `DealMergeError`
- Implement `deal_graph/models/deal.py` — `Deal`, `DealVersion`, `MEDDICProfile`, `DealStage`
  - All models strictly typed with UUID keys for Postgres future-proofing
  - Skeleton property names replicated from schema authority (`opportunity_id`, `amount`, `content_text`, `role`)
- Implement `deal_graph/models/extraction.py` — `ExtractedDeal`, `DealExtractionResult`, `DealDeduplicationDecision`, `MergedDeal`

### Phase 2: Neo4j Client & Schema

**Goal:** Establish connection to the `neo4j_structured` database and set up enrichment schema.

> **Schema Verification Step:** Before running `setup_schema()`, connect to the database and verify skeleton constraints/labels exist. Our `setup_schema()` only creates enrichment additions (`DealVersion` constraint, vector indexes, performance indexes). Skeleton constraints are `IF NOT EXISTS` no-ops.

- Implement `deal_graph/clients/neo4j_client.py` — `DealNeo4jClient` (inherits `Neo4jClient`, overrides `setup_schema()`)
- Implement `deal_graph/repository.py` — `DealRepository` with all Cypher queries:
  - `verify_account()` — MATCH Account by (tenant_id, account_id); MERGE if missing
  - `read_interaction()` — MATCH Interaction, read `content_text` for extraction
  - `enrich_interaction()` — SET `processed_at`, `deal_count` on existing Interaction
  - `create_deal()` — MERGE Deal with skeleton base properties + SET enrichment + SET `source_interaction_id`
  - `get_deal()` — by (tenant_id, opportunity_id)
  - `update_deal()` — SET enrichment properties + version++
  - `create_version_snapshot()` — DealVersion before update (includes `changed_fields`)
  - `get_deals_for_account()` — account-scoped query
  - `get_deal_history()` — DealVersion chain via `[:HAS_VERSION]`
- Write `tests/test_deal_neo4j_client.py` — schema setup tests
- Write `tests/test_deal_repository.py` — CRUD tests including enrichment property persistence

### Phase 3: Extraction

**Goal:** MEDDIC extraction from transcripts via LLM.

- Implement `deal_graph/prompts/extract_deals.py` — system/user prompts for both Case A (targeted) and Case B (discovery)
- Implement `deal_graph/pipeline/extractor.py` — `DealExtractor`
  - `extract_from_envelope()` — branch on `opportunity_id`
  - `_extract_discovery()` — Case B: full MEDDIC scan
  - `_extract_targeted()` — Case A: scoped to specific deal, existing deal context in prompt
  - `_generate_embeddings()` — batch embed "name: summary"
- Write `tests/test_deal_extraction.py`:
  - All 6 MEDDIC fields present
  - Partial MEDDIC (2-3 fields)
  - Zero deals from casual conversation
  - Multiple distinct deals from single transcript
  - Case A: targeted extraction for known deal
  - Confidence scoring validation

### Phase 4: Entity Resolution

**Goal:** Match extracted deals against existing Deal nodes using dual-embedding + LLM judgment.

- Implement `deal_graph/prompts/dedup_deals.py` — deduplication prompt
- Implement `deal_graph/pipeline/matcher.py` — `DealMatcher`
  - `find_matches()` — dual-index vector search + LLM dedup
  - `_find_candidates()` — search both `deal_embedding_idx` and `deal_embedding_current_idx`
  - Fast-path: 0 candidates → skip LLM, return no-match immediately
  - `_deduplicate()` — LLM comparison for borderline candidates
  - `_select_best_match()` — highest confidence match
  - Thresholds: < 0.70 create, 0.70-0.90 LLM, >= 0.90 auto-match
- Write `tests/test_deal_matcher.py`:
  - Same deal across two transcripts matches
  - Different deals for same account remain separate
  - Borderline similarity triggers LLM
  - Below-threshold creates new
  - Zero-candidate fast-path

### Phase 5: Merge & Persist

**Goal:** Synthesize deal updates with temporal invalidation + evolution narrative, and persist with versioning.

- Implement `deal_graph/prompts/merge_deals.py` — merge synthesis prompt enforcing all 7 merge rules, emphasizing narrative quality
- Implement `deal_graph/pipeline/merger.py` — `DealMerger`
  - `execute_decision()` — routes to create or update
  - `_create_new()` — MERGE Deal with skeleton base properties + SET enrichment
  - `_merge_deal()` — DealVersion snapshot (with `changed_fields`) + LLM synthesis + update enrichment + embedding check
  - `_compute_changed_fields()` — compare pre/post properties to populate `changed_fields`
- Write `tests/test_deal_merger.py`:
  - MEDDIC additive accumulation (decision_criteria)
  - MEDDIC temporal invalidation (economic_buyer replacement)
  - Stage progression (forward and backward)
  - Amount update when mentioned
  - Evolution summary narrative quality (explains WHY, not just WHAT)
  - `changed_fields` correctness
  - Version snapshot before update
  - `should_update_embedding` logic

### Phase 6: Pipeline Orchestration

**Goal:** End-to-end DealPipeline.

- Implement `deal_graph/pipeline/pipeline.py` — `DealPipeline`
  - `__init__()` — accept `OpenAIClient` + `DealNeo4jClient`
  - `from_env()` — factory from environment variables
  - `process_envelope()` — full flow with Case A/B branching
  - `process_text()` — convenience method
  - `close()` — cleanup
  - Interaction enrichment happens in Stage 2 (read `content_text`, add `processed_at`/`deal_count` after extraction)
- Implement `DealPipelineResult` dataclass (mirrors `PipelineResult` pattern)
- Write `tests/test_deal_pipeline.py`:
  - Full Case B: create new deal from transcript
  - Full Case B: update existing deal from transcript
  - Full Case A: targeted update with explicit opportunity_id
  - Zero-deal transcript (no-op)
  - Interaction enrichment properties persisted (processed_at, deal_count)
  - Error handling (partial failures)

### Phase 7: Dispatcher

**Goal:** Parallel routing of transcripts to both pipelines.

- Implement `dispatcher/envelope_dispatcher.py`:
  - `EnvelopeDispatcher` class
  - `process_envelope()` — `asyncio.gather` with `_safe_process_*` wrappers
  - `from_env()` — factory creating both pipelines
  - `close()` — cleanup both
  - `DispatchResult` dataclass with `action_item_result` + `deal_result` + per-pipeline errors
- Write `tests/test_envelope_dispatcher.py`:
  - Both pipelines execute concurrently
  - One pipeline failure does not block the other
  - Composite result returned correctly

### Phase 8: Configuration & Documentation

- Update `.env.example` with `DEAL_NEO4J_*` variables
- Update `pyproject.toml` for package discovery (add `deal_graph` and `dispatcher`)
- Write `docs/DEAL_ARCHITECTURE.md` — graph schema diagram, pipeline flow, MEDDIC model reference
- Create `examples/process_deal_transcript.py` — example usage

---

## 12. Testing Strategy

### Test Fixtures

Follow the pattern in `tests/conftest.py`:
- `deal_neo4j_credentials` fixture — skips when `DEAL_NEO4J_*` vars missing
- `deal_neo4j_client` fixture — creates connected client, runs `setup_schema()`, yields, closes
- `openai_api_key` fixture — reuse existing
- Integration tests hit real APIs (established pattern)

### Test Matrix

| Test File | Key Scenarios | Type |
|---|---|---|
| `test_deal_extraction.py` | Full MEDDIC, partial, zero, multi-deal, Case A targeted, confidence | Integration |
| `test_deal_matcher.py` | Cross-transcript match, different deals, borderline, threshold, zero-candidate fast-path | Integration |
| `test_deal_merger.py` | Additive MEDDIC, temporal invalidation, stage regression, narrative quality, `changed_fields`, versioning | Integration |
| `test_deal_pipeline.py` | E2E Case A, E2E Case B create, E2E Case B update, zero-deal, enrichment persistence | Integration |
| `test_deal_repository.py` | CRUD, enrichment patterns, Interaction read/enrich, DealVersion snapshots | Integration |
| `test_deal_neo4j_client.py` | Schema setup, vector search, dual-index search | Integration |
| `test_envelope_dispatcher.py` | Concurrent execution, isolated failures, composite result | Integration |

### Verification Commands

```bash
# Run all deal tests
pytest tests/test_deal_*.py -v

# Run dispatcher tests
pytest tests/test_envelope_dispatcher.py -v

# Run all tests (existing + new)
pytest tests/ -v

# Live integration test
python examples/process_deal_transcript.py
```

---

## 13. Strategic Recommendations

### S1: MEDDIC Completeness as a First-Class Signal

Store `meddic_completeness` (0.0–1.0) directly on `Deal` and index it. Enables instant visibility:

```cypher
MATCH (d:Deal {tenant_id: $tid})
WHERE d.meddic_completeness < 0.5
RETURN d.name, d.stage, d.meddic_completeness
ORDER BY d.meddic_completeness ASC
```

**Value:** "Which deals need more qualification work?" — directly actionable for sales managers.

### S2: MEDDIC Progress Timeline via DealVersion

The versioning architecture enables reconstructing the qualification journey:

```cypher
MATCH (d:Deal {tenant_id: $tid, opportunity_id: $oid})-[:HAS_VERSION]->(v:DealVersion)
RETURN v.version, v.created_at, v.change_summary, v.changed_fields
ORDER BY v.version ASC
```

**Example output:** "v1: Pain identified (data silos) → v2: Champion identified (Sarah, VP Eng) → v3: Budget confirmed ($500K)"

The `changed_fields` property enables programmatic analysis: "Deals that changed economic buyer >2 times close at 40% lower rate."

### S3: Cross-Pipeline Correlation via Interaction ID

`Interaction.interaction_id` is the joining key across pipelines. Since both pipelines read from the same `neo4j_structured` database, correlation queries are straightforward: "What action items were discussed in the same interactions as this deal?"

### S4: Episodic Re-Processing

Because `Interaction` stores the full `content_text` (created by the skeleton layer), deals can be re-extracted if prompts or models improve. Pattern: query all Interactions for an account, re-run MEDDIC extraction, compare with existing Deals. This is a powerful "model upgrade" pathway.

### S5: Future — Agentic Deal Coaching

Once the graph is populated, an agent can:
1. Read the full episodic history (all `content_text` from a deal's linked Interactions)
2. Identify deals with low `meddic_completeness`
3. Generate suggested questions for the next call ("Ask about budget authority" if `economic_buyer` is empty)
4. Compare deal patterns to historical closed-won deals
5. Auto-flag deals with regression patterns

### S6: Future — Deal Stall Detection

```cypher
MATCH (d:Deal {tenant_id: $tid})
WHERE d.stage IN ['qualification', 'proposal', 'negotiation']
AND d.last_updated_at < datetime() - duration('P14D')
RETURN d.name, d.stage, d.last_updated_at
ORDER BY d.last_updated_at ASC
```

### S7: Future — Postgres Synchronization

The strictly-typed Pydantic models with UUID keys are designed to map 1:1 to SQL tables:

| Pydantic Model | SQL Table | Key |
|---|---|---|
| `Deal` | `deals` | `opportunity_id` (str, PK), `tenant_id` (UUID) |
| `DealVersion` | `deal_versions` | `version_id` (UUID, PK), `deal_opportunity_id` (string, UUIDv7, FK) |
| `MEDDICProfile` | Flattened into `deals` table columns | — |

The flat MEDDIC property structure (individual columns, not JSON blob) enables SQL queries like `SELECT * FROM deals WHERE meddic_completeness < 0.5 AND stage = 'qualification'`.

---

## 14. Critique & Trade-offs

### Shared Database: Schema Coordination

Both `eq-structured-graph-core` and our pipeline write to the same `neo4j_structured` database. Coordination requirements:
- Our pipeline must respect skeleton property names — never rename or repurpose skeleton-owned fields
- Enrichment properties (prefixed `meddic_*`, `embedding*`, etc.) are clearly ours
- `DealVersion` is our own node type — no conflict with skeleton
- Constraint: if the schema authority renames a property, we must update our Pydantic models accordingly

### Episodic Memory via Skeleton

`Interaction.content_text` is created by the skeleton layer, not by our pipeline. This means:
- We depend on the skeleton having already processed the interaction before our pipeline runs
- Edge case: if our pipeline runs first, we MERGE with base properties (graceful degradation)
- No transcript duplication — single source of truth in the shared database

### LLM Token Costs (per transcript)

| Call | When | Estimated Tokens |
|---|---|---|
| MEDDIC extraction | Every transcript | ~2K input + ~500 output |
| Embedding generation | Per extracted deal | ~200 input |
| LLM deduplication | Per borderline match (0-5 per deal) | ~1K input + ~200 output |
| Merge synthesis | Per existing deal update | ~2K input + ~500 output |

**Mitigation:** Skip extraction for very short transcripts (< 100 words). Batch embeddings. Cache recently-seen deals for fast-path matching.

### Threshold Tuning

The proposed thresholds (0.70 / 0.90) are starting points. Log all similarity scores during initial deployment and analyze false-positive (incorrect merges) / false-negative (missed matches) rates to calibrate.

### Case A V1 Constraint

When `opportunity_id` is provided, V1 limits extraction to that single deal. This prevents data contamination but may miss new deals mentioned in the same conversation. Consider relaxing in V2 to extract the targeted deal AND flag additional discoveries as "unlinked signals."

### Flattened MEDDIC Trade-off

Storing Champion and Economic Buyer as strings (not graph nodes) means no Person Entity Resolution. This simplifies V1 but limits queries like "Show all deals where Sarah is the Champion." V2 can introduce `(:Person)` entities with resolution logic and migrate string properties to relationships.

---

## 15. Critical Files Reference

| Purpose | Existing File Path |
|---|---|
| Pipeline orchestrator pattern | `src/action_item_graph/pipeline/pipeline.py` |
| Neo4j client (base class to inherit) | `src/action_item_graph/clients/neo4j_client.py` |
| OpenAI client (shared, reused directly) | `src/action_item_graph/clients/openai_client.py` |
| LLM extraction prompt pattern | `src/action_item_graph/prompts/extract_action_items.py` |
| Merge/synthesis pattern | `src/action_item_graph/pipeline/merger.py` |
| Dual-embedding model pattern | `src/action_item_graph/models/action_item.py` |
| Entity resolution (dual-embedding + LLM) | `src/action_item_graph/pipeline/matcher.py` |
| Neo4j dual-index search implementation | `src/action_item_graph/clients/neo4j_client.py:299-352` |
| Config pattern | `src/action_item_graph/config.py` |
| Graphiti reference patterns | `docs/GRAPHITI_REFERENCE.md` |
| Test fixtures pattern | `tests/conftest.py` |
| EnvelopeV1 input format | `src/action_item_graph/models/envelope.py` |
| Logging infrastructure | `src/action_item_graph/logging.py` |
| Error hierarchy | `src/action_item_graph/errors.py` |
| Topic dual-embedding search pattern | `src/action_item_graph/pipeline/topic_resolver.py` |
| Ghost code (DO NOT USE) | `src/action_item_graph/models/entities.py` (Deal class) |
