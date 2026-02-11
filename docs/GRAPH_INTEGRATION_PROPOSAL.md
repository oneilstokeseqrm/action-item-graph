# Graph Integration Proposal: Action Item Pipeline → Structured Graph Database

## Status: Implemented (2026-02-11)

**Date**: 2026-02-11
**Scope**: Feasibility assessment and proposed approach for persisting action item pipeline outputs to the `neo4j_structured` database instead of a separate `neo4j_action` database.

---

## 1. Executive Summary

The action item pipeline currently writes to its own Neo4j AuraDB instance (`neo4j_action`). This proposal evaluates merging that persistence layer into the shared `neo4j_structured` database — the same database used by eq-structured-graph-core and the deal pipeline.

**Verdict: Feasible.** The deal pipeline already follows this pattern. The action item pipeline would become the third pipeline writing to the shared graph, joining the structured graph pipeline and the deal pipeline.

**Core principle**: Protect the structured graph. The action item pipeline adapts to fit the existing schema, not the other way around.

---

## 2. Foundational Invariant: Shared Source Interactions

All pipelines process the same upstream interactions. Every interaction arrives with the same:

- `tenant_id` — tenant isolation boundary
- `account_id` — account scope
- `interaction_id` — unique interaction identifier
- `user_id` — originating user

These identifiers are canonical. No pipeline should duplicate the Account or Interaction nodes they represent. All pipelines MERGE onto the same nodes.

---

## 3. First-Class Nodes Across Pipelines

### Structured Graph Pipeline (eq-structured-graph-core)

Explicitly categorized in `app/models/nodes.py`:

**Skeleton Nodes** (first-class, deterministic, from upstream payload):
- `Account` — keyed on `(tenant_id, account_id)`
- `Interaction` — keyed on `(tenant_id, interaction_id)`
- `Contact` — keyed on `(tenant_id, contact_id)`
- `Deal` — keyed on `(tenant_id, opportunity_id)`
- `CalendarWeek` — keyed on `week_iso_id` (global)

**Flesh Nodes** (LLM-derived):
- `Entity` — keyed on `(tenant_id, normalized_name)`
- `Topic` — keyed on `(tenant_id, normalized_name)`
- `Chunk` — keyed on `chunk_id` (UUID)

### Action Item Pipeline (action-item-graph)

Per `REQUIREMENTS.md` FR5.1-5.5:

**First-class (from payload)**:
- `Account` — root node for CRM context (FR5.1). **Shared with structured graph.**
- `Interaction` — source context for extraction. **Shared with structured graph.**

**LLM-derived**:
- `ActionItem` — core entity, connects to Account (FR5.2) and Interaction (FR5.3)
- `Owner` — action item assignee, separate from Contact (FR5.5)
- `ActionItemTopic` (renamed from Topic) — thematic grouping of action items

**Not a first-class node** (per FR4.3):
- `tenant_id` — a property on every node, not a node itself

### Deal Pipeline (action-item-graph)

**First-class**: Account, Interaction (shared), Deal
**LLM-derived**: DealVersion (temporal audit)

### Shared First-Class Nodes (Integration Points)

| Node | Structured Graph | Action Item Pipeline | Deal Pipeline |
|------|-----------------|---------------------|---------------|
| Account | Creates (skeleton) | MERGE (defensive) | MERGE (defensive) |
| Interaction | Creates (skeleton) | MERGE (defensive) | MERGE (defensive) |
| Contact | Creates (skeleton) | Not used | Not used |
| Deal | Creates (skeleton) | Not used | MERGE (enrichment) |

---

## 4. Current State: Two Databases

### AI Database (`neo4j_action`) — Action Item Pipeline

| Aspect | Value |
|--------|-------|
| URI | `neo4j+s://1aa04126.databases.neo4j.io` |
| Data | Empty (schema provisioned 2026-02-08, no data) |
| Constraints | 7 NODE KEY on `(tenant_id, id)` |
| Constraint type | NODE KEY (existence + uniqueness) |
| Vector indexes | 4 (ActionItem × 2 + Topic × 2) |
| Node labels | Account, Interaction, ActionItem, ActionItemVersion, Owner, Topic, TopicVersion |

### Structured Database (`neo4j_structured`) — eq-structured-graph-core + Deal Pipeline

| Aspect | Value |
|--------|-------|
| URI | `neo4j+s://c6171c63.databases.neo4j.io` |
| Data | 1,718 Topics, 579 Chunks, 325 Entities, 30 Interactions, 4 Accounts, 81 Communities |
| Constraints | 13 UNIQUENESS on label-specific keys |
| Constraint type | UNIQUENESS (uniqueness, allows null) |
| Vector indexes | 3 (Topic, Community, Deal embeddings) |
| Node labels | Account, Interaction, Contact, Deal, DealVersion, Entity, Topic, Chunk, CalendarWeek, Community, SuperTheme, AnalysisRun, CanonicalIdentity, Report |

---

## 4. Compatibility Analysis

### 4.1 Account Node — Property Name Mismatch

| Pipeline | Key Property | MERGE Pattern |
|----------|-------------|---------------|
| Structured Graph | `account_id` | `merge_node("Account", {"account_id": acct})` |
| Deal Pipeline | `account_id` | `MERGE (a:Account {tenant_id: $tid, account_id: $aid})` |
| Action Item Pipeline | **`id`** | `MERGE (a:Account {id: $aid, tenant_id: $tid})` |

**Problem**: The action item pipeline uses `id` as the property name for the account identifier. The structured graph and deal pipeline both use `account_id`.

**Fix**: Rename `id` → `account_id` in the action item pipeline's Account model and all Cypher queries. The structured graph's convention wins.

### 4.2 Interaction Node — Property Name Mismatch + CREATE vs MERGE

| Pipeline | Key Property | Operation | Content Property |
|----------|-------------|-----------|------------------|
| Structured Graph | `interaction_id` | MERGE | `content_text` |
| Deal Pipeline | `interaction_id` | MERGE / MATCH | `content_text` |
| Action Item Pipeline | **`id`** | **CREATE** | **`transcript_text`** |

**Problems**:
1. `id` vs `interaction_id` — same mismatch as Account
2. `transcript_text` vs `content_text` — different property name for the same text
3. CREATE vs MERGE — the action item pipeline currently CREATEs Interaction nodes, which would fail if the node already exists (constraint violation)

**Fix**:
- Rename `id` → `interaction_id`, `transcript_text` → `content_text`
- Change CREATE to MERGE with `ON CREATE SET` / `ON MATCH SET` semantics
- The deal pipeline's `ensure_interaction()` method (`src/deal_graph/repository.py:149`) is the exact pattern to follow

### 4.3 Topic Label — Collision (Different Concepts)

| Aspect | Action Item Pipeline Topic | Structured Graph Topic |
|--------|---------------------------|------------------------|
| Identity | UUID-based: `(tenant_id, id)` | Name-based: `(tenant_id, normalized_name)` |
| Purpose | Groups related action items into thematic clusters | Semantic label extracted from interaction text |
| Embeddings | Dual: `embedding` (immutable) + `embedding_current` (mutable) | Single: `embedding` |
| Versioning | Yes — TopicVersion nodes, version counter | No |
| Key relationship | `(:ActionItem)-[:BELONGS_TO]->(:Topic)` | `(:Interaction)-[:DISCUSSED]->(:Topic)` |
| Count in DB | 0 (empty) | 1,718 |

**Problem**: These are fundamentally different concepts sharing a label name. Different identity models, different constraints, different purposes. Cannot coexist under the same label.

**Fix**: Rename the action item pipeline's Topic → `ActionItemTopic` (and TopicVersion → `ActionItemTopicVersion`). The structured graph's Topic label is established (1,718 nodes, upstream dependencies in eq-structured-graph-core) and should not change.

### 4.4 Clean Additions (No Conflicts)

These labels are unique to the action item pipeline and do not exist in the structured database:

| Label | Constraint Needed | Notes |
|-------|------------------|-------|
| `ActionItem` | `(tenant_id, action_item_id) IS UNIQUE` | Core entity |
| `ActionItemVersion` | `(tenant_id, version_id) IS UNIQUE` | Temporal audit trail |
| `Owner` | `(tenant_id, owner_id) IS UNIQUE` | Action item assignees |
| `ActionItemTopic` | `(tenant_id, action_item_topic_id) IS UNIQUE` | Renamed from Topic |
| `ActionItemTopicVersion` | `(tenant_id, version_id) IS UNIQUE` | Renamed from TopicVersion |

No schema conflicts. Pure additive schema changes.

### 4.5 Constraint Type Decision

The AI database uses NODE KEY (stronger), the structured database uses UNIQUENESS.

**Recommendation**: Use UNIQUENESS for new action item labels, matching the structured graph convention. NODE KEY is stronger but requires coordination with eq-structured-graph-core to align. This can be revisited later as a schema-wide upgrade, but should not block integration.

### 4.6 Relationship Direction Inconsistency

| Concept | Action Item Pipeline | Structured Graph |
|---------|---------------------|------------------|
| Account ↔ Interaction | `(Account)-[:HAS_INTERACTION]->(Interaction)` | `(Interaction)-[:BELONGS_TO]->(Account)` |

**Recommendation**: Adopt the structured graph's `BELONGS_TO` convention (child→parent). The action item pipeline should not introduce `HAS_INTERACTION` when `BELONGS_TO` already encodes the same relationship. Alternatively, if `HAS_INTERACTION` provides distinct semantics the pipeline needs, both can coexist — Neo4j supports multiple relationship types between the same nodes.

### 4.7 Live Database Discrepancy: Deal Constraint

**Finding**: The `deal_unique` constraint in the live structured database enforces uniqueness on `(tenant_id, deal_id)`, but the eq-structured-graph-core code defines it as `(tenant_id, opportunity_id)`, and the deal pipeline MERGEs on `opportunity_id`.

This means the live constraint is not enforcing uniqueness on the property the deal pipeline actually writes. With 0 deals in the database, this hasn't manifested as an issue, but it would on first deal creation.

**Recommendation**: Reconcile before any further work on the shared database. Drop the stale `deal_unique` constraint and recreate it on `(tenant_id, opportunity_id)` to match the code.

---

## 5. Concurrency Strategy: Parallel Pipeline Execution

### The Challenge

When an interaction is ingested, a separate upstream ingestion service dispatches the same payload to multiple pipelines concurrently:

1. **Structured Graph pipeline** (eq-structured-graph-core): validate → skeleton → extract → write_flesh
2. **Action Item pipeline** (action-item-graph): extract → match → merge → topics
3. **Deal pipeline** (action-item-graph): extract → match → merge → enrich

**Important**: No pipeline is upstream of another. All three are **peers** receiving the same payload from the upstream ingestion service. The structured graph pipeline creates the skeleton (Account, Interaction, Contact, Deal, CalendarWeek) as part of its first stage, but this is not guaranteed to execute before the other pipelines. Any pipeline could complete first.

```
                    Upstream Ingestion Service
                              |
                    (same payload to all)
                              |
               ┌──────────────┼──────────────┐
               |              |              |
               v              v              v
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │  Structured  │ │ Action Item  │ │    Deal      │
     │  Graph       │ │  Pipeline    │ │  Pipeline    │
     │  Pipeline    │ │              │ │              │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            |                |                |
            └────────────────┼────────────────┘
                             v
                    ┌────────────────┐
                    │  Single Neo4j  │
                    │  (structured)  │
                    └────────────────┘
```

Interaction IDs are minted by the upstream ingestion service — they don't exist as graph nodes beforehand. They are created during processing. Since all pipelines are peers, we cannot predict which will persist first.

### Evaluated Strategies

#### Option A — Sequential Dependency

> Run structured graph pipeline first; action item and deal pipelines wait until it finishes.

- **Pro**: Skeleton guaranteed to exist. Other pipelines can use MATCH instead of MERGE.
- **Con**: Adds total latency (structured graph time + pipeline time). Requires orchestration logic. Creates tight coupling between independent services.
- **Verdict**: Architecturally clean but operationally fragile. If the structured graph pipeline fails, all downstream pipelines are blocked.

#### Option B — Defensive MERGE (Recommended)

> All pipelines independently MERGE shared nodes (Account, Interaction). If the node exists, MERGE finds it. If not, MERGE creates it with base properties.

- **Pro**: No sequencing needed. Pipelines remain independent. Already proven by the deal pipeline.
- **Con**: First pipeline to run creates a "partial" node (e.g., Interaction without `trace_id`). Later pipelines enrich it.
- **Verdict**: **This is the correct approach.** It's already implemented and tested in the deal pipeline.

#### Option C — Hybrid: Skeleton-First with Defensive Fallback

> The upstream ingestion system runs the skeleton transaction first (~100ms, no LLM calls), then fires all LLM pipelines in parallel. Defensive MERGE remains as safety net.

- **Pro**: Skeleton is guaranteed. Fastest total latency (skeleton is trivial; all LLM work parallelized). Defense-in-depth.
- **Con**: Requires the upstream ingestion system to know about the skeleton builder. Cross-service coordination.
- **Verdict**: Best production architecture, but requires upstream changes. Not needed for Phase 1. Defensive MERGE (Option B) handles the same scenario gracefully.

### Structured Graph Pipeline: Does It Handle Pre-Existing Nodes?

**Yes.** The `TenantSession.merge_node()` method (eq-structured-graph-core `app/db/session.py:116-173`) generates:

```cypher
MERGE (n:Interaction {tenant_id: $tenant_id, interaction_id: $interaction_id})
SET n.trace_id = $other_trace_id,
    n.interaction_type = $other_interaction_type,
    n.source = $other_source,
    n.timestamp = $other_timestamp,
    n.content_text = $other_content_text,
    n.content_format = $other_content_format
RETURN n
```

Key observations:

1. **Uses MERGE** — idempotent. If the node exists, finds it. If not, creates it.
2. **Uses unconditional SET** (not `ON CREATE SET`) — always sets skeleton properties regardless of who created the node.
3. **Individual property SET** (not map replacement `SET n = $props`) — only touches named properties. Does NOT clear enrichment properties added by other pipelines (e.g., `processed_at`, `deal_count`, `action_item_count`).

This means: if the action item pipeline or deal pipeline creates the Interaction node first, the structured graph pipeline will find it via MERGE and overwrite the skeleton properties with the same values from the same upstream payload. Enrichment properties from other pipelines survive.

**The structured graph pipeline was not explicitly designed for this scenario, but it is safe by construction.** However, this safety is fragile — if someone changes `merge_node()` to use map replacement (`SET n = $props`), it would silently wipe other pipelines' enrichment properties.

### Three-Pipeline Concurrency Safety Matrix

| First to persist | Structured Graph runs | Deal Pipeline runs | Action Item Pipeline runs | Safe? |
|---|---|---|---|---|
| Structured Graph | Creates node + skeleton props | MERGE finds it, ON MATCH sets `processed_at` | MERGE finds it, ON MATCH sets enrichment | **Yes** |
| Deal Pipeline | MERGE finds it, unconditional SET overwrites skeleton props (same values) | Already ran | MERGE finds it, ON MATCH sets enrichment | **Yes** |
| Action Item Pipeline | MERGE finds it, unconditional SET overwrites skeleton props (same values) | MERGE finds it, ON MATCH sets `processed_at` | Already ran | **Yes** |
| Deal + Action Item (both before Structured Graph) | MERGE finds it, unconditional SET overwrites skeleton props (same values). All enrichment props survive. | Already ran | Already ran | **Yes** |

**Every scenario is safe** because of the layered write strategy:
- **Structured graph**: Unconditional SET (schema authority — always sets skeleton properties, never touches enrichment)
- **Deal pipeline**: ON CREATE SET / ON MATCH SET (defensive — base props on create, enrichment-only on match)
- **Action item pipeline**: ON CREATE SET / ON MATCH SET (defensive — base props on create, enrichment-only on match)

### Changes Required Per Pipeline

| Pipeline | Current State | Changes Needed |
|----------|--------------|----------------|
| Structured Graph (eq-structured-graph-core) | `merge_node()` with unconditional SET | **None** — already safe as schema authority |
| Deal Pipeline (action-item-graph) | `ensure_interaction()` with ON CREATE/ON MATCH | **None** — already implements defensive MERGE |
| Action Item Pipeline (action-item-graph) | CREATE (not MERGE), wrong property names | **Must change**: adopt MERGE with ON CREATE/ON MATCH, align property names |

### Recommended Strategy: Option B with Awareness of Option C

**For now**: All pipelines use defensive MERGE for Account and Interaction nodes. The deal pipeline's `ensure_interaction()` method is the template:

```python
# src/deal_graph/repository.py:149-200
MERGE (i:Interaction {tenant_id: $tenant_id, interaction_id: $interaction_id})
ON CREATE SET
    i.content_text = $content_text,
    i.interaction_type = $interaction_type,
    i.timestamp = $timestamp,
    i.source = $source,
    i.trace_id = $trace_id,
    i.created_at = datetime()
ON MATCH SET
    i.processed_at = datetime()
```

**Key semantics**:
- `ON CREATE SET`: Populates base properties if this pipeline creates the node first
- `ON MATCH SET`: Only sets enrichment properties if the node already exists (skeleton or another pipeline created it)
- Never overwrites skeleton-owned properties on `ON MATCH`

**For production**: Evolve to Option C when the upstream ingestion system is ready. The skeleton transaction is fast (~100ms) and deterministic — running it first eliminates the race condition entirely. But Option B is a safe foundation that works regardless.

---

## 6. Enrichment Safety: Structured Graph Phase 2

### Structured Graph Pipeline Stages

```
validate_payload → build_skeleton → extract_content → write_flesh
```

The `write_flesh` stage (Phase 2 enrichment) creates:
- **Entity** nodes via `merge_entity()` — keyed on `(tenant_id, normalized_name)`
- **Topic** nodes via `merge_topic()` — keyed on `(tenant_id, normalized_name)`
- **Chunk** nodes via `create_chunk()` — keyed on `chunk_id` (UUID)
- Relationships: `MENTIONS`, `DISCUSSED`, `PART_OF`

### Impact Analysis

| Concern | Risk Level | Explanation |
|---------|------------|-------------|
| Label collision with Entity | **None** | Action item pipeline has no Entity label |
| Label collision with Topic | **None** after rename | ActionItemTopic is a distinct label |
| Label collision with Chunk | **None** | Action item pipeline has no Chunk label |
| Relationship interference | **None** | All new relationship types (`EXTRACTED_FROM`, `OWNED_BY`, `BELONGS_TO` on ActionItem→ActionItemTopic) are distinct from structured graph relationships |
| Query interference | **None** | Resolution engine queries (`lookup_entity_by_normalized_name`, `lookup_topic_by_normalized_name`) are label-scoped. They only MATCH on `:Entity` and `:Topic` labels. ActionItem, Owner, and ActionItemTopic nodes are invisible to these queries. |
| Performance impact | **Negligible** | Additional nodes increase graph size, but constraint-backed queries (MERGE on indexed keys) are O(log n). The structured graph's MERGE operations are key-based lookups, not full scans. |
| Transaction contention | **Low** | Pipelines write to different labels. The only shared write targets are Account and Interaction nodes. MERGE is idempotent — concurrent MERGEs on the same key result in one CREATE and one no-op. |

### Verdict: Phase 2 Enrichment is Safe

The structured graph's flesh layer operates exclusively on Entity, Topic, and Chunk labels. It uses label-scoped MATCH and MERGE queries. Adding ActionItem, ActionItemVersion, Owner, and ActionItemTopic labels to the same database introduces no interference. These labels exist in a parallel namespace within the same graph.

The only shared touch points are Account and Interaction nodes, which are handled by defensive MERGE as described in Section 5.

---

## 7. Deal Pipeline Review

The deal pipeline already writes to the structured database. Here's how it currently handles the shared-node pattern:

| Operation | Method | Pattern | Aligned? |
|-----------|--------|---------|----------|
| Account | `verify_account()` | `MERGE (a:Account {tenant_id, account_id})` | Yes |
| Interaction read | `read_interaction()` | `MATCH (i:Interaction {tenant_id, interaction_id})` | Yes |
| Interaction enrich | `enrich_interaction()` | `MATCH … SET i += $updates` | Yes |
| Interaction fallback | `ensure_interaction()` | `MERGE … ON CREATE SET … ON MATCH SET` | Yes — **this is the template** |
| Deal | `create_deal()` | `MERGE (d:Deal {tenant_id, opportunity_id})` | Yes |

The deal pipeline is the **proven reference implementation** for writing to the structured database from a secondary pipeline. Its conventions should be followed exactly by the action item pipeline.

**One issue**: The deal constraint discrepancy (Section 4.7) should be resolved. The deal pipeline MERGEs on `opportunity_id`, but the live constraint is on `deal_id`. This is pre-existing and unrelated to the integration, but should be fixed.

---

## 8. Implementation Phases

### Phase 1: Property Alignment (Action Item Pipeline Only)

**Goal**: Make the action item pipeline's models and queries compatible with the structured graph's naming conventions.

**Changes**:

| File | Change |
|------|--------|
| `src/action_item_graph/models/entities.py` | `Account.id` → `Account.account_id`; `Interaction.id` → `Interaction.interaction_id`; `Interaction.transcript_text` → `Interaction.content_text`; `Interaction.occurred_at` → `Interaction.timestamp` |
| `src/action_item_graph/repository.py` | Update all Cypher queries: `{id: $account_id}` → `{account_id: $account_id}`, etc. Change Interaction CREATE → MERGE with ON CREATE/ON MATCH. |
| `src/action_item_graph/clients/neo4j_client.py` | Update constraint definitions from `(tenant_id, id)` to label-specific keys matching structured DB conventions |
| `src/action_item_graph/pipeline/pipeline.py` | Property reference updates |
| All test files | Update property references and fixtures |

**Estimated scope**: ~300 lines of changes across ~15 files. All within action-item-graph repo. No changes to eq-structured-graph-core.

### Phase 2: Topic Rename

**Goal**: Eliminate the Topic label collision.

**Changes**:

| File | Change |
|------|--------|
| `src/action_item_graph/models/topic.py` | Rename class, update label references |
| `src/action_item_graph/repository.py` | All `:Topic` → `:ActionItemTopic` in Cypher |
| `src/action_item_graph/clients/neo4j_client.py` | Constraint and index names |
| `src/action_item_graph/pipeline/stages/topic_grouping.py` | Label references |
| All test files | Label and fixture updates |

**Estimated scope**: ~100 lines of changes across ~8 files.

### Phase 3: Schema Addition

**Goal**: Add action item pipeline constraints and indexes to the structured database.

**New DDL**:
```cypher
-- Constraints
CREATE CONSTRAINT action_item_unique IF NOT EXISTS
  FOR (n:ActionItem) REQUIRE (n.tenant_id, n.action_item_id) IS UNIQUE;
CREATE CONSTRAINT action_item_version_unique IF NOT EXISTS
  FOR (n:ActionItemVersion) REQUIRE (n.tenant_id, n.version_id) IS UNIQUE;
CREATE CONSTRAINT owner_unique IF NOT EXISTS
  FOR (n:Owner) REQUIRE (n.tenant_id, n.owner_id) IS UNIQUE;
CREATE CONSTRAINT action_item_topic_unique IF NOT EXISTS
  FOR (n:ActionItemTopic) REQUIRE (n.tenant_id, n.action_item_topic_id) IS UNIQUE;
CREATE CONSTRAINT action_item_topic_version_unique IF NOT EXISTS
  FOR (n:ActionItemTopicVersion) REQUIRE (n.tenant_id, n.version_id) IS UNIQUE;

-- Range indexes
CREATE INDEX action_item_tenant_idx IF NOT EXISTS FOR (n:ActionItem) ON (n.tenant_id);
CREATE INDEX action_item_account_idx IF NOT EXISTS FOR (n:ActionItem) ON (n.account_id);
CREATE INDEX action_item_status_idx IF NOT EXISTS FOR (n:ActionItem) ON (n.status);
CREATE INDEX owner_tenant_idx IF NOT EXISTS FOR (n:Owner) ON (n.tenant_id);
CREATE INDEX action_item_topic_tenant_idx IF NOT EXISTS FOR (n:ActionItemTopic) ON (n.tenant_id);
CREATE INDEX action_item_topic_account_idx IF NOT EXISTS FOR (n:ActionItemTopic) ON (n.account_id);

-- Vector indexes (dual embedding strategy)
CREATE VECTOR INDEX action_item_embedding_idx IF NOT EXISTS
  FOR (n:ActionItem) ON n.embedding
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX action_item_embedding_current_idx IF NOT EXISTS
  FOR (n:ActionItem) ON n.embedding_current
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX action_item_topic_embedding_idx IF NOT EXISTS
  FOR (n:ActionItemTopic) ON n.embedding
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX action_item_topic_embedding_current_idx IF NOT EXISTS
  FOR (n:ActionItemTopic) ON n.embedding_current
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
```

### Phase 4: Connection Switch

**Goal**: Point the action item pipeline at the structured database.

- Update `Neo4jClient` to use `DEAL_NEO4J_URI` / `DEAL_NEO4J_PASSWORD` (or introduce a shared `STRUCTURED_NEO4J_*` env var set)
- Remove AI-DB-specific schema setup from `Neo4jClient.setup_schema()` (constraints now managed as part of the structured DB schema)
- Update `EnvelopeDispatcher` to inject a single Neo4j driver shared by both pipelines

### Phase 5: Test Alignment

- Update live E2E test (`scripts/run_live_e2e.py`) to verify all pipeline outputs in a single database
- Ensure test isolation via distinct `tenant_id` values (no test should write to another test's tenant)
- Update integration tests to reflect shared-database architecture

---

## 9. Pre-Existing Issues to Resolve

These issues exist independently of the integration but should be addressed before or during the work:

1. **Deal constraint discrepancy**: Live DB has `deal_unique` on `(tenant_id, deal_id)`, code expects `(tenant_id, opportunity_id)`. Drop and recreate.
2. **Naive timestamps**: Several files in `src/deal_graph/` use `datetime.now()` without timezone. Should be `datetime.now(tz=timezone.utc)`. (Tracked in existing plan.)
3. **Action item pipeline Account.id field**: Currently allows arbitrary strings like `"acct_acme_corp_001"`. The structured graph uses UUIDs for `account_id`. Need to verify these identifiers are consistent upstream.

---

## 10. Integrated Graph Schema (Post-Integration)

```
                              ┌──────────┐
                              │ Account  │
                              └────┬─────┘
           ┌──────────────────────┼──────────────────────┬────────────────┐
           │                      │                      │                │
           ▼                      ▼                      ▼                ▼
    ┌─────────────┐        ┌───────────┐          ┌──────────┐     ┌──────────────┐
    │ Interaction │◄───────│ActionItem │─────────►│  Owner   │     │ActionItemTopic│
    └──────┬──────┘        └─────┬─────┘          └──────────┘     └──────┬───────┘
           │                     │                                        │
    ┌──────┼──────────┐          ▼                                        ▼
    │      │          │   ┌───────────────┐                    ┌───────────────────┐
    │      │          │   │  AIVersion    │                    │ AITopicVersion    │
    ▼      ▼          ▼   └───────────────┘                    └───────────────────┘
 ┌──────┐ ┌─────┐ ┌─────┐
 │Entity│ │Topic│ │Chunk│       ← Structured Graph (unchanged)
 └──────┘ └─────┘ └─────┘

 ┌──────────┐ ┌─────────────┐ ┌────────────┐
 │  Deal    │ │ DealVersion │ │CalendarWeek│   ← Deal Pipeline + Skeleton (unchanged)
 └──────────┘ └─────────────┘ └────────────┘
```

Account and Interaction are the shared anchors. All pipeline outputs converge on them.

---

## 11. Benefits

1. **Unified knowledge graph**: Cross-pipeline queries become natural ("show me action items from interactions that discussed {topic}")
2. **Shared identity**: One Account node, one Interaction node — no duplication
3. **Simpler infrastructure**: One AuraDB instance instead of two
4. **Richer context**: ActionItems can traverse to Entities, Chunks, Communities via shared Interaction
5. **Proven pattern**: The deal pipeline already does this successfully

## 12. Risks

1. **Schema coupling**: Changes to eq-structured-graph-core constraints require coordination
2. **AuraDB Free tier**: 200K node limit; combined data accelerates toward it
3. **Test isolation**: Shared database means pipeline tests must use distinct tenant_id values
4. **Property name migration**: If any AI DB data existed, it would need migration (currently empty, so no issue)
5. **Upstream ID consistency**: Account IDs must be consistent across all upstream systems

---

## 13. Verification Checklist

Before integration is considered complete:

- [ ] Action item pipeline uses `account_id` (not `id`) for Account nodes
- [ ] Action item pipeline uses `interaction_id` (not `id`) for Interaction nodes
- [ ] Action item pipeline uses `content_text` (not `transcript_text`) for Interaction content
- [ ] Action item pipeline uses MERGE (not CREATE) for Interaction nodes
- [ ] Topic label renamed to `ActionItemTopic` throughout
- [ ] All new constraints created in structured database
- [ ] All vector indexes created in structured database
- [ ] Deal constraint discrepancy resolved (`deal_id` → `opportunity_id`)
- [ ] No `datetime.now()` without timezone in deal_graph or action_item_graph
- [ ] Phase 2 enrichment validated — entity/topic resolution unaffected
- [ ] Live E2E test passes against shared database
- [ ] All unit tests pass
- [ ] Cross-pipeline query demonstrated (ActionItem → Interaction → Topic/Entity)
