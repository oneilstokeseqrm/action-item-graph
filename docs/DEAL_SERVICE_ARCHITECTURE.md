# Deal Service Architecture

## High-Level Flow

```
                              EnvelopeV1
                                  |
                                  v
                        ┌─────────────────┐
                        │ EnvelopeDispatcher│
                        └────────┬────────┘
                                 |
                   asyncio.gather(return_exceptions=True)
                                 |
                 ┌───────────────┴───────────────┐
                 |                               |
                 v                               v
     ┌───────────────────┐           ┌───────────────────┐
     │ ActionItemPipeline │           │   DealPipeline    │
     │                   │           │                   │
     │ Extract → Match → │           │ Extract → Match → │
     │ Merge → Topics    │           │ Merge → Persist   │
     └────────┬──────────┘           └────────┬──────────┘
              |                               |
              v                               v
     ┌───────────────┐              ┌───────────────┐
     │ AI Neo4j DB   │              │ Deal Neo4j DB │
     │ (NEO4J_*)     │              │ (DEAL_NEO4J_*)│
     └───────────────┘              └───────────────┘
```

Both pipelines receive the same `EnvelopeV1` and run **concurrently**. One
pipeline failing never blocks or cancels the other. The `DispatcherResult`
captures either a successful result or the exception from each pipeline
independently.

**Entry point**: `EnvelopeDispatcher.dispatch(envelope)` in `src/dispatcher/dispatcher.py`

---

## Deal Pipeline Stages

```
EnvelopeV1
    |
    v
┌──────────────────────────────────────────────────┐
│ Stage 0: Validate & Prepare                      │
│  - Require account_id on envelope                │
│  - verify_account() → MERGE Account node         │
│  - ensure_interaction() → MERGE Interaction node  │
│  - Route: Case A (targeted) or Case B (discovery) │
└──────────────────────┬───────────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────────┐
│ Stage 1: Extract (DealExtractor)                 │
│                                                  │
│  Case A (Targeted): envelope has opportunity_id  │
│    → Extract updates for ONE known deal          │
│    → Enforces max 1 deal returned                │
│                                                  │
│  Case B (Discovery): no opportunity_id           │
│    → Find 0, 1, or N deals from transcript       │
│    → No artificial constraints on count          │
│                                                  │
│  Output: DealExtractionResult + embeddings       │
│  Embedding text: "{name}: {summary}"             │
└──────────────────────┬───────────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────────┐
│ Stage 2: Match (DealMatcher) — per extracted deal│
│                                                  │
│  1. Vector search: dual-embedding search         │
│     - deal_embedding_idx (original, immutable)   │
│     - deal_embedding_current_idx (current)       │
│     - Deduplicate by opportunity_id, keep best   │
│                                                  │
│  2. Graduated threshold decision:                │
│     ≥ 0.90  → auto_match (skip LLM)             │
│     0.70–0.90 → LLM deduplication call           │
│     < 0.70  → create_new (no candidates)         │
│                                                  │
│  Output: DealMatchResult with match_type         │
└──────────────────────┬───────────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────────┐
│ Stage 3: Merge (DealMerger)                      │
│                                                  │
│  If match_type == create_new:                    │
│    → Generate opportunity_id (UUIDv7)             │
│    → Map stage_assessment → DealStage enum       │
│    → Build MEDDIC profile from extraction        │
│    → Set embedding = embedding_current           │
│    → Persist via repo.create_deal()              │
│                                                  │
│  If match_type == auto_match or llm_match:       │
│    → LLM synthesis: merge MEDDIC fields          │
│    → Create version snapshot (BEFORE update)     │
│    → Conditionally re-embed if summary changed   │
│    → Persist via repo.update_deal()              │
│                                                  │
│  Output: DealMergeResult                         │
└──────────────────────┬───────────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────────┐
│ Stage 4: Enrich Interaction                      │
│  - Set deal_count on Interaction node            │
│  - Best-effort (failures logged, not raised)     │
└──────────────────────────────────────────────────┘
```

**Pipeline entry point**: `DealPipeline.process_envelope(envelope)` in `src/deal_graph/pipeline/pipeline.py`

---

## Key Design Decisions

### 1. Graduated Thresholds

The matcher uses a three-tier decision system to balance precision and recall:

| Score Range | Decision | LLM Call? | Rationale |
|-------------|----------|-----------|-----------|
| >= 0.90 | `auto_match` | No | High semantic similarity; LLM judgment unnecessary |
| 0.70 - 0.90 | LLM decides | Yes | Borderline zone; LLM evaluates deal-specific context |
| < 0.70 | `create_new` | No | Too dissimilar to confidently merge |

Thresholds are configurable via environment variables:
- `DEAL_SIMILARITY_THRESHOLD` (default: 0.70)
- `DEAL_AUTO_MATCH_THRESHOLD` (default: 0.90)

The LLM is biased toward `create_new` when uncertain — false merges destroy data,
while false creates can be merged later.

### 2. Fault Isolation

The `EnvelopeDispatcher` uses `asyncio.gather(return_exceptions=True)` to ensure
one pipeline failing never blocks the other:

```python
outcomes = await asyncio.gather(
    self.action_item_pipeline.process_envelope(envelope),
    self.deal_pipeline.process_envelope(envelope),
    return_exceptions=True,
)
```

`DispatcherResult` slots hold either a successful result or the exception:

- `action_item_result: PipelineResult | BaseException | None`
- `deal_result: DealPipelineResult | BaseException | None`

`overall_success` is `True` when at least one pipeline returned a result (not an
exception). Use `both_succeeded` to confirm zero exceptions.

### 3. Accumulative Error Handling

Within the deal pipeline, per-deal errors are accumulated rather than halting
the pipeline:

```python
for extracted_deal in extraction_result.deals:
    try:
        match_result = await self.matcher.find_matches(...)
        merge_result = await self.merger.merge_deal(...)
        result.merge_results.append(merge_result)
    except Exception as exc:
        result.errors.append(f'Deal "{deal.name}" (index {i}): {exc}')
```

If a 3-deal transcript has one deal that fails matching, the other two still
proceed through the full pipeline. The `DealPipelineResult.success` property
returns `True` only when `errors` is empty.

### 4. Version Snapshots

Every merge creates a `DealVersion` node **before** updating the deal:

```
(:Deal)-[:HAS_VERSION]->(:DealVersion)
```

Each version captures:
- Full deal state at that point in time (name, stage, amount, summary)
- All six MEDDIC dimension snapshots
- `change_summary` — what changed and why (from LLM)
- `changed_fields` — machine-readable list of modified properties
- `change_source_interaction_id` — which transcript triggered the change

**Bi-temporal model**: Each version records `valid_from` (when it started) and
`valid_until` (when it was superseded), enabling point-in-time history queries.

### 5. Dual Embeddings

Every deal maintains two embedding vectors:

| Embedding | Property | Mutability | Purpose |
|-----------|----------|------------|---------|
| Original | `embedding` | Immutable | Catch new deals similar to original state |
| Current | `embedding_current` | Updated on merge | Catch updates to evolved deals |

Both indexes are searched during matching via
`DealNeo4jClient.search_deals_both_embeddings()`. Results are deduplicated by
`opportunity_id`, keeping the higher score.

The merger only re-embeds `embedding_current` when the LLM synthesis indicates
the deal summary changed significantly (`should_update_embedding=True`).

### 6. MEDDIC-Structured Extraction

Deal data follows the MEDDIC sales methodology framework:

| Dimension | Field | Merge Rule |
|-----------|-------|------------|
| **M**etrics | `metrics` | Replaceable (latest is best) |
| **E**conomic Buyer | `economic_buyer` | Replaceable |
| **D**ecision Criteria | `decision_criteria` | Additive (accumulate insights) |
| **D**ecision Process | `decision_process` | Additive |
| **I**dentified Pain | `identified_pain` | Additive |
| **C**hampion | `champion` | Replaceable |

`meddic_completeness` is tracked as a 0.0-1.0 score (populated dimensions / 6).

---

## Data Models

### ExtractedDeal

Defined in `src/deal_graph/models/extraction.py`. Output of the LLM extraction stage.

| Field | Type | Description |
|-------|------|-------------|
| `opportunity_name` | `str` | Descriptive name for the opportunity |
| `opportunity_summary` | `str` | 2-3 sentence overview |
| `stage_assessment` | `str` | LLM's assessment of deal stage |
| `metrics` | `str \| None` | Quantifiable business impact |
| `economic_buyer` | `str \| None` | Person with final budget authority |
| `decision_criteria` | `str \| None` | Technical/business evaluation criteria |
| `decision_process` | `str \| None` | Steps/timeline to reach decision |
| `identified_pain` | `str \| None` | Core business problem |
| `champion` | `str \| None` | Internal advocate |
| `estimated_amount` | `float \| None` | Estimated deal value |
| `currency` | `str` | Currency code (default: USD) |
| `expected_close_timeframe` | `str \| None` | Freetext close timeframe |
| `confidence` | `float` | Extraction confidence (0.0-1.0) |
| `reasoning` | `str` | Why this is a deal and what signals were present |

### DealMatchResult

Defined in `src/deal_graph/pipeline/matcher.py`. Output of the matching stage.

| Field | Type | Description |
|-------|------|-------------|
| `extracted_deal` | `ExtractedDeal` | The extracted deal |
| `embedding` | `list[float]` | Embedding vector |
| `match_type` | `str` | `auto_match`, `llm_match`, or `create_new` |
| `matched_deal` | `DealMatchCandidate \| None` | The matched candidate (if any) |
| `decision` | `DealDeduplicationDecision \| None` | LLM decision for borderline cases |
| `candidates_evaluated` | `int` | Number of candidates checked |
| `all_candidates` | `list[DealMatchCandidate]` | Full candidate list |

### DealMergeResult

Defined in `src/deal_graph/pipeline/merger.py`. Output of the merge stage.

| Field | Type | Description |
|-------|------|-------------|
| `opportunity_id` | `str (UUIDv7 canonical format)` | The deal's unique identifier |
| `action` | `str` | `created` or `merged` |
| `was_new` | `bool` | Whether newly created |
| `version_created` | `bool` | Whether a version snapshot was created |
| `source_interaction_id` | `str \| None` | Triggering interaction |
| `embedding_updated` | `bool` | Whether `embedding_current` was re-computed |
| `details` | `dict` | Additional context (stage, MEDDIC completeness, change narrative) |

### DealPipelineResult

Defined in `src/deal_graph/pipeline/pipeline.py`. Aggregate result per envelope.

| Field | Type | Description |
|-------|------|-------------|
| `tenant_id` | `str` | Tenant identifier |
| `account_id` | `str \| None` | Account identifier |
| `interaction_id` | `str \| None` | Interaction identifier |
| `opportunity_id` | `str \| None` | Non-None only for Case A (targeted) |
| `total_extracted` | `int` | Number of deals extracted |
| `deals_created` | `list[str]` | Opportunity IDs of created deals |
| `deals_merged` | `list[str]` | Opportunity IDs of merged deals |
| `merge_results` | `list[DealMergeResult]` | Detailed per-deal results |
| `errors` | `list[str]` | Per-deal error messages |
| `warnings` | `list[str]` | Non-fatal warnings |
| `processing_time_ms` | `int` | Total processing time |
| `stage_timings` | `dict[str, float]` | Per-stage timing breakdown |

### DispatcherResult

Defined in `src/dispatcher/dispatcher.py`. Wraps results from both pipelines.

| Field | Type | Description |
|-------|------|-------------|
| `action_item_result` | `PipelineResult \| BaseException \| None` | AI pipeline outcome |
| `deal_result` | `DealPipelineResult \| BaseException \| None` | Deal pipeline outcome |
| `action_item_success` | `bool` (property) | True if AI pipeline returned a result |
| `deal_success` | `bool` (property) | True if Deal pipeline returned a result |
| `both_succeeded` | `bool` (property) | True if both pipelines succeeded |
| `overall_success` | `bool` (property) | True if at least one pipeline succeeded |
| `dispatch_time_ms` | `int` | Wall-clock dispatch time |
| `errors` | `list[str]` | Dispatcher-level error messages |

---

## File Map

```
src/deal_graph/
├── __init__.py
├── config.py                    # DEAL_NEO4J_* env vars, thresholds
├── errors.py                    # DealPipelineError, DealExtractionError
├── repository.py                # DealRepository (graph CRUD)
├── utils.py                     # uuid7() wrapper (UUIDv7 via fastuuid)
├── clients/
│   └── neo4j_client.py          # DealNeo4jClient (schema, vector search)
├── models/
│   └── extraction.py            # ExtractedDeal, DealExtractionResult, MergedDeal
└── pipeline/
    ├── pipeline.py              # DealPipeline orchestrator
    ├── extractor.py             # DealExtractor (targeted + discovery)
    ├── matcher.py               # DealMatcher (vector search + LLM dedup)
    └── merger.py                # DealMerger (LLM synthesis + versioning)

src/dispatcher/
├── __init__.py
└── dispatcher.py                # EnvelopeDispatcher, DispatcherResult
```
