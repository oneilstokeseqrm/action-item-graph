# Graph Integration Plan: Action Item Pipeline → Structured Database

## Status: Implemented (2026-02-11)

**Date**: 2026-02-11
**Reference**: `docs/GRAPH_INTEGRATION_PROPOSAL.md` — full feasibility analysis

---

## Context

The action item pipeline currently writes to its own Neo4j AuraDB instance (`neo4j_action`, URI `1aa04126`). This plan merges that persistence layer into the shared `neo4j_structured` database (URI `c6171c63`) — the same database used by eq-structured-graph-core and the deal pipeline. The deal pipeline already does this successfully; the action item pipeline becomes the third pipeline writing to the shared graph.

**Core principle**: Protect the structured graph. The action item pipeline adapts to fit the existing schema conventions.

---

## Execution Strategy: Three Passes

### Pass 1 — Core Code Changes (Phases 1–5)
Model + repository + client + pipeline changes. Run unit tests after: `pytest tests/ -v`. All must pass.

### Pass 2 — Test Fixtures + E2E Script (Phase 6)
Update all test files and `scripts/run_live_e2e.py` for single-database architecture. Run `pytest tests/ -v` again.

### Pass 3 — Schema Provisioning + Documentation + Live E2E (Phases 7–9)
Provision new schema via MCP, update all docs, run live E2E.

---

## Phase 0: Deal Constraint Fix (MCP — Immediate)

The live `deal_unique` constraint is on `(tenant_id, deal_id)` but code expects `(tenant_id, opportunity_id)`.

**Action**: Drop and recreate on `(tenant_id, opportunity_id)`.

---

## Phase 1: Model Layer Alignment

### `src/action_item_graph/models/entities.py`
- `Account`: `to_neo4j_properties()` outputs `'account_id'` instead of `'id'`
- `Interaction`: field renames `id`→`interaction_id`, `transcript_text`→`content_text`, `occurred_at`→`timestamp`; property output aligned
- `Owner`: `to_neo4j_properties()` outputs `'owner_id'` instead of `'id'`
- `Contact`: `to_neo4j_properties()` outputs `'contact_id'` instead of `'id'`

### `src/action_item_graph/models/action_item.py`
- `ActionItem.to_neo4j_properties()`: `'id'`→`'action_item_id'`
- `ActionItemVersion.to_neo4j_properties()`: `'id'`→`'version_id'`

### `src/action_item_graph/models/topic.py`
- `Topic` → `ActionItemTopic`, `TopicVersion` → `ActionItemTopicVersion`
- `ActionItemTopic.to_neo4j_properties()`: `'id'`→`'action_item_topic_id'`
- `ActionItemTopicVersion.to_neo4j_properties()`: `'id'`→`'version_id'`

---

## Phase 2: Repository Layer — Cypher Query Updates

### `src/action_item_graph/repository.py`
- All Account MERGEs: `{id: $account_id}` → `{account_id: $account_id}`
- All Interaction refs: `{id: $interaction_id}` → `{interaction_id: $interaction_id}`
- Interaction CREATE → MERGE with ON CREATE/ON MATCH (deal pipeline template)
- All ActionItem refs: `{id: $...}` → `{action_item_id: $...}`
- All Owner refs: `{id: $...}` → `{owner_id: $...}`
- All `:Topic` → `:ActionItemTopic`, `:TopicVersion` → `:ActionItemTopicVersion`
- All Topic property refs: `{id: $topic_id}` → `{action_item_topic_id: $topic_id}`

### `src/action_item_graph/clients/neo4j_client.py`
- Vector search deduplication: `node['id']` → label-specific key property

---

## Phase 3: Neo4j Client — Connection Switch + Schema

- Connection fallback: `NEO4J_URI` → `DEAL_NEO4J_URI`
- Vector index name updates for ActionItemTopic
- `setup_schema()` rewrite: UNIQUENESS constraints (not NODE KEY), only for labels we own
- Add `verify_skeleton_schema()` method
- Remove Account/Interaction constraints (skeleton-owned)

---

## Phase 4: Pipeline + Extractor Layer

- `extractor.py`: Interaction field renames
- `pipeline.py`: Property refs, Topic→ActionItemTopic imports
- `topic_resolver.py`: Label/property/import renames
- `topic_executor.py`: Same renames
- `matcher.py`: `node['id']` → `node['action_item_id']`
- `merger.py`: `owner['id']` → `owner['owner_id']`

---

## Phase 5: Config + Environment

- `config.py`: `NEO4J_URI` falls back to `DEAL_NEO4J_URI`
- `.env`: Point `NEO4J_URI` to structured DB

---

## Phase 6: Test Updates (Pass 2)

All AI pipeline test files updated for property + label renames. `scripts/run_live_e2e.py` refactored for single-database architecture.

---

## Phase 7: Schema Provisioning (Pass 3)

Via MCP: 5 UNIQUENESS constraints, 7 range indexes, 4 vector indexes on structured DB.

---

## Phase 8: Documentation Updates (Pass 3)

All docs updated to reflect single-DB architecture, new property names, ActionItemTopic label.

---

## Phase 9: Live E2E Verification (Pass 3)

Run live E2E, verify cross-pipeline queries, regenerate test results doc.

---

## Files Modified (Complete Inventory)

See the full plan in the user's implementation instructions for the complete file inventory (~37 files across 3 passes).
