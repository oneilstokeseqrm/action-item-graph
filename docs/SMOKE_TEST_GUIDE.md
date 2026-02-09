# Smoke Test & E2E Verification Guide

> Comprehensive reference for how both pipelines (Action Item + Deal) were validated end-to-end against live infrastructure.

**Last validated**: 2026-02-03 (live E2E) | 2026-02-08 (AI DB schema re-provisioned on new AuraDB Free instance)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Infrastructure & Credentials](#2-infrastructure--credentials)
3. [Live E2E Smoke Test](#3-live-e2e-smoke-test)
4. [Integration E2E Tests (Mocked IO)](#4-integration-e2e-tests-mocked-io)
5. [Unit Test Suites by Pipeline](#5-unit-test-suites-by-pipeline)
6. [Test Data](#6-test-data)
7. [Validated Contracts & Invariants](#7-validated-contracts--invariants)
8. [Historical Results](#8-historical-results)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture Overview

The system runs **two pipelines concurrently** via `EnvelopeDispatcher`:

```
                         EnvelopeV1
                             |
                    EnvelopeDispatcher
                      /            \
                     /              \
        ActionItemPipeline      DealPipeline
          (AI Database)        (Deal Database)
```

Each pipeline writes to a **physically separate Neo4j AuraDB instance** — they share no data, constraints, or connections. A shared `OpenAIClient` handles LLM extraction and embedding generation for both.

| Component | AI Database | Deal Database |
|-----------|-------------|---------------|
| **Instance** | `NEO4J_URI` | `DEAL_NEO4J_URI` |
| **Node Labels** | Account, Interaction, ActionItem, ActionItemVersion, Owner, Topic, TopicVersion | Account, Interaction, Deal, DealVersion |
| **Constraints** | 7 NODE KEY on (tenant_id, id) | Uniqueness on opportunity_id, interaction_id |
| **Vector Indexes** | 4 (ActionItem + Topic, original + current) | 2 (Deal original + current) |
| **Embedding Dimensions** | 1536 (text-embedding-3-small) | 1536 (text-embedding-3-small) |

---

## 2. Infrastructure & Credentials

### Required Environment Variables

```bash
# OpenAI (shared by both pipelines)
OPENAI_API_KEY=sk-...

# AI Database (Action Items, Topics, Owners)
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j

# Deal Database (Deals, DealVersions)
DEAL_NEO4J_URI=neo4j+s://yyyyy.databases.neo4j.io
DEAL_NEO4J_USERNAME=neo4j
DEAL_NEO4J_PASSWORD=<password>
```

### Schema Provisioning

Both databases must have their schema provisioned before testing. Schema setup is **idempotent** (`IF NOT EXISTS` on all DDL).

**Option A — Programmatic (recommended):**

Schema is created automatically when `setup_schema()` is called at pipeline startup. The live E2E script calls this before processing:

```python
await ai_neo4j.connect()
await ai_neo4j.setup_schema()       # 7 constraints + 9 indexes + 4 vector indexes

await deal_neo4j.connect()
await deal_neo4j.setup_schema()     # constraints + indexes + 2 vector indexes
```

**Option B — Manual via MCP or Cypher:**

If provisioning a fresh AuraDB instance (e.g., after free-tier reset), run the DDL directly. See [`neo4j_client.py:setup_schema()`](../src/action_item_graph/clients/neo4j_client.py) and [`deal_graph/clients/neo4j_client.py:setup_schema()`](../src/deal_graph/clients/neo4j_client.py) for the exact statements.

**AI Database schema (20 total DDL statements):**

| Type | Count | Details |
|------|-------|---------|
| NODE KEY constraints | 7 | `(tenant_id, id)` on Account, Interaction, ActionItem, ActionItemVersion, Owner, Topic, TopicVersion |
| Property indexes | 9 | `tenant_id`, `account_id`, `status`, `canonical_name` across labels |
| Vector indexes | 4 | `ActionItem.embedding`, `ActionItem.embedding_current`, `Topic.embedding`, `Topic.embedding_current` (1536d, cosine) |

**Deal Database schema:**

| Type | Details |
|------|---------|
| Uniqueness constraints | `opportunity_id` on Deal, `interaction_id` on Interaction |
| Property indexes | `tenant_id`, `account_id`, `stage` on Deal; `tenant_id` on Interaction |
| Vector indexes | `Deal.embedding` + `Deal.embedding_current` (1536d, cosine) |

---

## 3. Live E2E Smoke Test

### What It Is

The **live E2E smoke test** (`scripts/run_live_e2e.py`) is the gold-standard validation. It processes real transcripts through the full `EnvelopeDispatcher` against live Neo4j Aura and OpenAI instances — the same code path production traffic follows.

> **Both pipelines are tested simultaneously.** Each transcript is dispatched through
> `EnvelopeDispatcher.dispatch()`, which calls `ActionItemPipeline.process_envelope()`
> and `DealPipeline.process_envelope()` **concurrently via `asyncio.gather()`** on the
> same envelope. Both pipelines execute at the same time, hitting their respective
> live Neo4j databases in parallel, sharing a single OpenAI client. This is not two
> separate test runs — it is one integrated test that validates the full concurrent
> system as it would operate in production.

### How to Run

```bash
# From project root, with .env configured:
python scripts/run_live_e2e.py
```

**Prerequisites:**
- All environment variables set (OpenAI + both Neo4j instances)
- Both database schemas provisioned
- Python environment with all dependencies installed (`uv sync` or `pip install -e ".[dev]"`)

**Expected runtime:** ~5-6 minutes (dominated by OpenAI API latency)

### What It Does

The script executes these phases in order:

#### Phase 1: Setup
1. Loads 4 transcripts from `examples/transcripts/transcripts.json`
2. Connects to both Neo4j instances and runs `setup_schema()`
3. **Cleans both databases** — deletes all nodes for the test tenant (tenant-scoped, no impact on other tenants)

#### Phase 2: Sequential Transcript Processing (Both Pipelines Per Transcript)
For each of the 4 transcripts (in sequence order):

1. Builds an `EnvelopeV1` with a fresh `interaction_id` (UUID4)
2. Dispatches through `EnvelopeDispatcher.dispatch()` — **both pipelines run concurrently on the same envelope**:
   - `ActionItemPipeline` → extracts action items, matches/deduplicates, creates topics → writes to **AI DB**
   - `DealPipeline` → extracts deals (MEDDIC), matches/deduplicates, merges → writes to **Deal DB**
   - Both execute via `asyncio.gather(return_exceptions=True)` — one failing never blocks the other
3. Prints per-pipeline results (items extracted, deals created/merged, timing)
4. Asserts `result.both_succeeded == True` (the "Both OK" column in the summary table)
5. **Verifies Deal DB state** after each transcript (Account count, Interaction count, `deal_count` enrichment)

#### Phase 3: Final State Queries
Queries both databases for complete state:

- **AI DB**: All ActionItems (with Owner, Topic, source Interaction), all Topics (with linked items), all Owners, all Interactions
- **Deal DB**: All Deals (with full MEDDIC profile), all DealVersions (with change audit), all Interactions (with `deal_count` enrichment)

#### Phase 4: Summary Report
Prints a comprehensive table showing results from **both pipelines for each transcript**:

```
Transcript                     AI Items   Topics   Deals    Merged   Both OK   Time
Call 1                         3          2        1        0        Yes       24,320ms
Call 2                         12         5        0        1        Yes       81,276ms
Call 3                         24         9        2        0        Yes       185,760ms
Call 4 — Follow-up Status ..   6          1        0        0        Yes       50,978ms
TOTAL                          45         17       3        1                  342,334ms
```

**Column definitions:**
- **AI Items** / **Topics**: Action Item pipeline results (written to AI DB)
- **Deals** / **Merged**: Deal pipeline results (written to Deal DB)
- **Both OK**: `result.both_succeeded` — `True` only when **both** pipelines returned a successful result for that transcript (no exceptions). This is the critical simultaneous-execution indicator.
- **Time**: Wall-clock time for `EnvelopeDispatcher.dispatch()` — includes both pipelines running concurrently

### What It Validates

**Simultaneous execution contracts:**

| Contract | How Verified |
|----------|--------------|
| Both pipelines run concurrently on every envelope | `EnvelopeDispatcher` uses `asyncio.gather()` — single dispatch call, both execute in parallel |
| Both pipelines succeed for all transcripts | `both_succeeded = True` for all 4 envelopes |
| Shared OpenAI client works under concurrent load | Both pipelines call the same `OpenAIClient` for extraction + embeddings without conflict |
| Separate databases receive correct writes | AI DB contains action items/topics; Deal DB contains deals/versions — no cross-contamination |
| Zero errors across entire run | `result.errors == []` for every dispatch |
| Action items persisted with topics | Final state query: 45 items, 17 topics, 100% coverage |
| Deals created with MEDDIC | Final state query: 3 deals, all 100% MEDDIC completeness |
| Deal merging works | Call 2 merges into Call 1's AML deal (version 1 → 2) |
| DealVersion snapshots | 1 version snapshot created with `changed_fields` audit |
| Interaction enrichment | `deal_count` stamped on each Deal DB Interaction node |
| UUIDv7 identity contract | `opportunity_id` values parseable as UUIDv7 |
| `deal_ref` display alias | Derived from random tail, all 3 distinct |
| Tenant-scoped isolation | All queries filter by `tenant_id` |
| Dual embedding strategy | Both `embedding` and `embedding_current` populated |
| Idempotent schema setup | `setup_schema()` runs without errors on pre-existing schema |
| Duplicate-text fix | Zip alignment prevents dict-collision (regression validated) |

### Interpreting Results

> **LLM non-determinism**: Exact item counts, text, owner names, and topic names **will vary between runs**. The purpose is to validate contracts and invariants, not pin exact outputs.

**Stable invariants** (should not change between runs):
- Both pipelines succeed for all 4 transcripts
- Zero errors
- 3 deals created (AML, IDR, EDP) — matched by name/theme, not exact text
- 1 deal merge (AML in Call 2)
- 1 DealVersion snapshot
- All MEDDIC dimensions populated for all deals
- `deal_count` values: Call 1=1, Call 2=1, Call 3=2, Call 4=0

**Variable between runs** (expected):
- Exact action item count (typically 35-50)
- Topic count and names (typically 15-20)
- Owner resolution (name aliasing varies)
- Deal amounts (LLM-estimated from transcript context)
- Processing times

### Summary: Simultaneous Pipeline Testing

| Question | Answer |
|----------|--------|
| Have both pipelines been tested simultaneously? | **Yes.** The live E2E smoke test dispatches every transcript through `EnvelopeDispatcher`, which runs both pipelines concurrently via `asyncio.gather()`. |
| When was this last validated? | **2026-02-03.** All 4 transcripts processed with `both_succeeded = True` for every envelope. Zero errors. |
| What does "simultaneously" mean technically? | A single `dispatch()` call launches `ActionItemPipeline.process_envelope()` and `DealPipeline.process_envelope()` as concurrent coroutines. They share one OpenAI client but write to separate Neo4j databases. |
| How do we know both actually executed? | The summary table shows both AI Items > 0 **and** Deal activity (3 created, 1 merged) across the run. Final state queries confirm nodes in both databases. |
| What if one pipeline fails? | `asyncio.gather(return_exceptions=True)` captures the exception without canceling the other pipeline. The "Partial System Failure" integration test (Section 4) validates this contract. |
| How to re-run? | `python scripts/run_live_e2e.py` with all credentials in `.env`. |

---

## 4. Integration E2E Tests (Mocked IO)

### What They Are

The integration E2E tests (`tests/test_integration_e2e.py`) exercise the **full object graph** — `EnvelopeDispatcher` → `ActionItemPipeline` + `DealPipeline` → extractors → matchers → mergers → repositories — with only the IO edges mocked (OpenAI, Neo4j).

**No API keys or Neo4j required.** These run in CI.

### How to Run

```bash
pytest tests/test_integration_e2e.py -v
```

### Scenarios

#### Scenario 1: "The Double Play" (8 tests)

A single transcript contains both an action item and a deal signal. Both pipelines run concurrently and succeed.

| Test | Verifies |
|------|----------|
| `test_both_pipelines_succeed` | `overall_success=True`, `both_succeeded=True` |
| `test_action_item_created` | `create_action_item` called on AI Neo4j |
| `test_deal_created` | `create_deal` called on Deal Neo4j |
| `test_action_item_result_has_created_ids` | At least 1 action item in `created_ids` |
| `test_deal_result_has_created_ids` | At least 1 deal in `deals_created` |
| `test_openai_called_for_both_extractions` | Shared OpenAI receives both `ExtractionResult` and `DealExtractionResult` |
| `test_embeddings_generated_for_both` | `create_embeddings_batch` called >= 2 times |
| `test_separate_neo4j_clients_used` | Each Neo4j mock receives `execute_write` calls |

**Test envelope:**
```
Sarah: I will email the contract to legal for review by Friday.
John: Sounds good. Also, the budget is confirmed at $50k for the data platform.
      Jane Smith from finance signed off.
```

#### Scenario 2: "Partial System Failure" (5 tests)

Deal Neo4j fails mid-flight (timeout on Deal MERGE); Action Item Neo4j works fine.

| Test | Verifies |
|------|----------|
| `test_overall_success_true` | Dispatcher succeeds because AI pipeline returned a result |
| `test_action_item_persisted` | Action item created despite deal failure |
| `test_deal_error_captured` | `DealPipelineResult.errors` contains "Neo4j connection timeout" |
| `test_deal_not_created` | `deals_created == []` |
| `test_both_pipelines_received_envelope` | Neither result slot is None — both attempted |

**Key design contract validated**: Pipeline fault isolation. One pipeline crashing does not affect the other.

---

## 5. Unit Test Suites by Pipeline

### Running All Tests

```bash
# All tests
pytest tests/ -v

# Action Item pipeline only
pytest tests/test_pipeline.py tests/test_extractor.py tests/test_matcher.py \
      tests/test_merger.py tests/test_repository.py \
      tests/test_pipeline_with_topics.py tests/test_topic_resolver.py \
      tests/test_topic_executor.py -v

# Deal pipeline only
pytest tests/test_deal_pipeline.py tests/test_deal_extraction.py \
      tests/test_deal_matcher.py tests/test_deal_merger.py \
      tests/test_deal_repository.py tests/test_deal_neo4j_client.py -v

# Dispatcher only
pytest tests/test_dispatcher.py tests/test_integration_e2e.py -v

# Regression: duplicate-text bug
pytest tests/test_duplicate_text.py -v
```

### Action Item Pipeline (9 test files)

| File | Scope | Requires Live Services |
|------|-------|-----------------------|
| `test_pipeline.py` | Full pipeline flow | Yes (OpenAI + Neo4j) |
| `test_pipeline_with_topics.py` | Pipeline with topic grouping | Yes (OpenAI + Neo4j) |
| `test_extractor.py` | LLM extraction | Yes (OpenAI) |
| `test_matcher.py` | Vector search + deduplication | Yes (OpenAI + Neo4j) |
| `test_merger.py` | Merge/create execution | Yes (OpenAI + Neo4j) |
| `test_repository.py` | Graph CRUD operations | Yes (Neo4j) |
| `test_topic_resolver.py` | Topic matching logic | Mocked |
| `test_topic_executor.py` | Topic creation/linking | Mocked |
| `test_duplicate_text.py` | Duplicate-text regression | Mocked |

### Deal Pipeline (6 test files)

| File | Scope | Requires Live Services |
|------|-------|-----------------------|
| `test_deal_pipeline.py` | Full deal pipeline orchestration | Mocked |
| `test_deal_extraction.py` | Deal extraction (Case A + B) | Mocked |
| `test_deal_matcher.py` | Graduated threshold matching | Mocked |
| `test_deal_merger.py` | Merge synthesis + version snapshots | Mocked |
| `test_deal_repository.py` | Deal graph CRUD | Yes (Deal Neo4j) |
| `test_deal_neo4j_client.py` | Deal DB client + schema | Yes (Deal Neo4j) |

### Shared / Infrastructure (6 test files)

| File | Scope |
|------|-------|
| `test_dispatcher.py` | Dispatcher concurrency, fault isolation, result aggregation |
| `test_integration_e2e.py` | Full system wiring (see [Section 4](#4-integration-e2e-tests-mocked-io)) |
| `test_neo4j_client.py` | AI Neo4j client connection, queries |
| `test_openai_client.py` | OpenAI client wrapper |
| `test_errors.py` | Custom exception hierarchy |
| `test_uuid7.py` | UUIDv7 identity contract |
| `test_logging.py` | Structured logging + timing |

### Test Counts (as of 2026-02-03)

```
266 tests passed, 0 failed
```

Tests that require live services will **skip** (not fail) when credentials are missing.

---

## 6. Test Data

### Transcripts

Located at `examples/transcripts/transcripts.json`. Contains 4 real-world diarized transcripts from the Lightbox AWS account:

| Transcript | Sequence | Content | AI Items | Deals |
|------------|----------|---------|----------|-------|
| **Call 1** | 1 | Application Modernization Lab introduction | ~3-6 | 1 created (AML) |
| **Call 2** | 2 | Follow-up: AML + multiple other topics | ~10-12 | 1 merged (AML) |
| **Call 3** | 3 | Large multi-topic discussion | ~18-24 | 2 created (IDR, EDP) |
| **Call 4** | 4 | Status update follow-up | ~6 | 0 |

**Test identity:**
- Tenant ID: `11111111-1111-4111-8111-111111111111`
- Account ID: `aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa`
- Account Name: Lightbox

### Shared Fixtures (`tests/conftest.py`)

```python
sample_tenant_id   = '11111111-1111-4111-8111-111111111111'
sample_account_id  = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
sample_transcript  = "Sarah: I'll send the proposal to the client by Friday..."
sample_deal_transcript = "Sarah: The platform helps with data silo issues..."
```

Fixtures auto-skip tests when the required `OPENAI_API_KEY` or `NEO4J_*` environment variables are not set.

---

## 7. Validated Contracts & Invariants

The following contracts have been validated through the combination of live E2E and unit tests:

### Identity & Data Model

| Contract | Evidence |
|----------|----------|
| UUIDv7 for `opportunity_id` | `test_uuid7.py` + live E2E: parseable, `.version == 7` |
| `deal_ref` = `"deal_" + hex[-16:]` | Live E2E: 3 deals, all refs distinct, derived from random tail |
| NODE KEY on `(tenant_id, id)` (AI DB) | 7 constraints verified via `SHOW CONSTRAINTS` |
| No CRM/Salesforce references | Grep across all prompts = 0 hits |
| All timestamps UTC-aware | Grep for `datetime.now()` in `src/deal_graph/` = 0 hits |

### Pipeline Behavior

| Contract | Evidence |
|----------|----------|
| Concurrent pipeline execution | `test_dispatcher.py`: timing verification |
| Pipeline fault isolation | "Partial System Failure" scenario: AI succeeds when Deal fails |
| Accumulative error handling | `test_deal_pipeline.py`: per-deal errors collected, not raised |
| Idempotent schema setup | `IF NOT EXISTS` on all DDL; re-run produces no errors |
| Idempotent ActionItem persistence | `MERGE` on `{id, tenant_id}` — second write is no-op |
| Duplicate-text fix (zip alignment) | `test_duplicate_text.py`: 3 tests validating 1:1 positional alignment |

### Deal Pipeline Specifics

| Contract | Evidence |
|----------|----------|
| Case A (targeted): uses existing deal context | `test_deal_extraction.py`, `test_deal_pipeline.py` |
| Case B (discovery): finds all deals | `test_deal_extraction.py`, live E2E (Calls 1, 3) |
| Graduated thresholds: >=0.90 auto, 0.70-0.90 LLM, <0.70 create | `test_deal_matcher.py` |
| MEDDIC field synthesis | Live E2E: all 3 deals at 100% completeness (6/6 dimensions) |
| Version snapshots before merge | Live E2E: 1 DealVersion with `changed_fields` audit |
| `changed_fields` normalization | Bare names mapped through `BARE_TO_PREFIXED`, meta-fields filtered |
| Interaction enrichment (`deal_count`) | Live E2E: Call 1=1, Call 2=1, Call 3=2, Call 4=0 |
| Dual embeddings (original + current) | Both indexes populated; `search_both_embeddings` tested |

### Action Item Pipeline Specifics

| Contract | Evidence |
|----------|----------|
| Topic grouping with 100% coverage | Live E2E: 45/45 items have topic assignments |
| Topic resolution thresholds: >=0.85 auto-link, 0.70-0.85 LLM, <0.70 create | `test-data-report.md` |
| Status update detection | Live E2E Call 4: 4 existing items updated |
| Owner resolution with alias matching | Live E2E: 7 distinct owners resolved |
| Dual embedding strategy | Both `embedding` and `embedding_current` vector indexes populated |

---

## 8. Historical Results

### Live E2E Run — 2026-02-03 (Dual Pipeline)

**Full results**: [`docs/LIVE_E2E_TEST_RESULTS.md`](./LIVE_E2E_TEST_RESULTS.md)

| Metric | AI DB | Deal DB |
|--------|-------|---------|
| Action Items / Deals | 45 | 3 |
| Topics / DealVersions | 17 | 1 |
| Owners | 7 | -- |
| Interactions | 4 | 4 |
| Topic Coverage | 100% | -- |
| Errors | 0 | 0 |
| Wall-clock Time | -- | -- |
| **Total** | -- | 343,815 ms |

**Deals created:**

| Deal | Amount | Stage | MEDDIC | Version |
|------|--------|-------|--------|---------|
| Application Modernization Lab (AML) | $497K | qualification | 100% | 2 (merged in Call 2) |
| Incident Detection & Response (IDR) | $58K | qualification | 100% | 1 |
| Enterprise Discount Program (EDP) | $7.15M | proposal | 100% | 1 |

**Key changes validated in this run:**
- AI DB constraints upgraded to tenant-scoped NODE KEY (from global UNIQUENESS)
- Duplicate-text zip alignment fix
- `create_action_item` MERGE key includes `tenant_id`

### Action Item Pipeline Run — 2026-01-25 (AI Pipeline Only)

**Full results**: [`docs/test-data-report.md`](./test-data-report.md)

| Metric | Value |
|--------|-------|
| Action Items | 41 |
| Topics | 17 |
| Topic Coverage | 100% |
| Owners | 8 |
| Status Updates | 4 |
| Processing Time | 364,638 ms |

This earlier run predates the Deal pipeline and Dispatcher. It validated the core Action Item pipeline in isolation.

### AI DB Schema Re-provisioning — 2026-02-08

The AI Database was re-provisioned on a new AuraDB Free instance (`neo4j+s://1aa04126.databases.neo4j.io`). All 20 DDL statements (7 constraints, 9 indexes, 4 vector indexes) were executed via MCP and verified ONLINE. No pipeline code was modified.

---

## 9. Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NEO4J_URI environment variable is required` | `.env` not loaded | Run from project root, or `source .env` |
| `AuthenticationError` on Neo4j | Wrong password or expired AuraDB instance | Check `.env` credentials; AuraDB Free pauses after inactivity |
| `IndexNotFoundError` on vector search | Schema not provisioned | Run `setup_schema()` or manually create vector indexes |
| Tests skip with `SKIP: No OpenAI API key` | `OPENAI_API_KEY` not set | Set in `.env` or export directly |
| Inconsistent item counts between runs | LLM non-determinism | Expected; validate contracts, not exact counts |
| `cleanup failed — N nodes remain` | Orphaned nodes from previous run | Run cleanup query manually: `MATCH (n) WHERE n.tenant_id = $tid DETACH DELETE n` |
| AuraDB Free instance unavailable | Instance paused after 72h inactivity | Resume in Aura console; may need to re-provision schema |

### Verifying Database State

Quick queries to check database health (run via MCP or Cypher shell):

```cypher
-- AI DB: Count all node types for tenant
MATCH (n) WHERE n.tenant_id = '11111111-1111-4111-8111-111111111111'
RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label

-- Deal DB: List all deals with MEDDIC completeness
MATCH (d:Deal) WHERE d.tenant_id = '11111111-1111-4111-8111-111111111111'
RETURN d.name, d.stage, d.amount, d.meddic_completeness, d.version
ORDER BY d.created_at

-- Check all indexes are ONLINE
SHOW INDEXES YIELD name, state WHERE state <> 'ONLINE' RETURN name, state

-- Check all constraints exist
SHOW CONSTRAINTS YIELD name, type RETURN name, type ORDER BY name
```

---

## Appendix: File Map

```
scripts/
  run_live_e2e.py                    # Live E2E smoke test (both pipelines)

tests/
  conftest.py                        # Shared fixtures, skip logic
  test_integration_e2e.py            # Full-system wiring (mocked IO)
  test_dispatcher.py                 # Dispatcher concurrency + fault isolation

  # Action Item Pipeline
  test_pipeline.py                   # AI pipeline integration
  test_pipeline_with_topics.py       # AI pipeline + topic grouping
  test_extractor.py                  # LLM extraction
  test_matcher.py                    # Vector matching + dedup
  test_merger.py                     # Merge/create execution
  test_repository.py                 # Graph CRUD
  test_topic_resolver.py             # Topic matching
  test_topic_executor.py             # Topic creation/linking
  test_duplicate_text.py             # Regression: zip alignment fix

  # Deal Pipeline
  test_deal_pipeline.py              # Deal pipeline orchestration
  test_deal_extraction.py            # Deal extraction (Case A + B)
  test_deal_matcher.py               # Graduated threshold matching
  test_deal_merger.py                # Merge synthesis + versions
  test_deal_repository.py            # Deal graph CRUD
  test_deal_neo4j_client.py          # Deal DB client + schema

  # Infrastructure
  test_neo4j_client.py               # AI Neo4j client
  test_openai_client.py              # OpenAI client wrapper
  test_errors.py                     # Exception hierarchy
  test_uuid7.py                      # UUIDv7 identity contract
  test_logging.py                    # Structured logging

examples/
  transcripts/transcripts.json       # 4 test transcripts (Lightbox)

docs/
  LIVE_E2E_TEST_RESULTS.md           # 2026-02-03 run results
  test-data-report.md                # 2026-01-25 AI-only run results
  SMOKE_TEST_GUIDE.md                # This document
```
