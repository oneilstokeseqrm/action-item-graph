# Live E2E Smoke Test Results

**Run date**: 2026-03-12 (post-quality-overhaul)
**Script**: `scripts/run_live_e2e.py --legacy`
**Account**: Lightbox (`aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa`)
**Tenant**: `11111111-1111-4111-8111-111111111111`
**Verdict**: ALL PIPELINES SUCCEEDED FOR ALL TRANSCRIPTS. Zero errors.

> **Run variability**: This document is a point-in-time snapshot. Specific values
> (IDs, amounts, action item text, topic names, owner resolution) **will differ
> between runs** due to LLM non-determinism. The purpose of this doc is to validate
> **contracts and invariants** — not to pin exact outputs. Stable invariants include:
> deal count, pipeline success/failure, constraint behavior, and structural
> relationships.

> **Single shared Neo4j database**: Both pipelines (Action Item + Deal) write to a
> single shared Neo4j Aura instance. Shared labels (Account, Interaction) are
> managed via defensive MERGE operations. AI-owned labels (ActionItem, ActionItemTopic,
> Owner, etc.) use UNIQUENESS constraints with label-specific key properties.
> See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the unified schema.

**Changes since previous run (2026-02-11):**
- **Quality pipeline overhaul**: 10-stage pipeline with F-CoT extraction, consolidation, LLM-as-Judge verification, owner pre-resolution, graduated matching, priority scoring
- **Extraction quality**: F-CoT two-stage extraction with Five-Field Commitment Framework, capped 3-8 items
- **Within-batch consolidation**: Complete-linkage clustering at cosine similarity 0.80
- **LLM-as-Judge verification**: Adversarial quality gate (confidence floor 0.4, fail-open)
- **Owner pre-resolution**: Account-scoped alias cache with SequenceMatcher fuzzy matching
- **Graduated matching thresholds**: 0.88 auto-match, 0.68 min similarity (reduces LLM calls ~60%)
- **Priority scoring**: First-class fields on ActionItem model (impact, urgency, specificity, effort, priority_score)
- **Postgres owner upsert fix**: `ON CONFLICT (tenant_id, canonical_name)` replaces `ON CONFLICT (owner_id)`

---

## 1. Test Configuration

| Setting | Value |
|---------|-------|
| Transcripts | 4 (sorted by sequence) |
| Database | Single shared Neo4j Aura instance (both pipelines) |
| LLM | OpenAI (`gpt-4o-mini`) via structured output |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Pre-run cleanup | Label-scoped wipe for pipeline-owned nodes (preserves upstream skeleton) |

---

## 2. Processing Summary

| Transcript | AI Items | Topics | Deals Created | Deals Merged | Both OK |
|------------|----------|--------|---------------|--------------|---------|
| Call 1 | 3 | 1 | 1 | 0 | Yes |
| Call 2 | 2 | 1 | 0 | 1 | Yes |
| Call 3 | 1 | 1 | 2 | 0 | Yes |
| Call 4 — Follow-up Status Update | 0 | 0 | 0 | 0 | Yes |
| **TOTAL** | **6** | **3** | **3** | **1** | |

**Quality pipeline impact**: Previous run (2026-02-11, pre-overhaul) extracted **30** action items from the same 4 transcripts. The quality pipeline reduced this to **6** — an 80% reduction — while retaining only genuine, actionable commitments with clear owners, scoring, and completion criteria.

---

## 3. Quality Pipeline Metrics

### 3.1 Extraction Quality

All 6 persisted items are explicit commitments with named owners:

| # | Summary | Owner | Priority | Commitment |
|---|---------|-------|----------|------------|
| 1 | Jackie will initiate AML technical discovery for RIMS and related workloads | Jackie | 0.82 | explicit |
| 2 | David will open a support case and arrange a specialist meeting for Mac VPN issues | David | 0.745 | explicit |
| 3 | Alejandro to complete RDS database backup list by end of week | Alejandro | 0.74 | explicit |
| 4 | Jackie will share AML program materials and workload data with Lightbox team | Jackie | 0.675 | explicit |
| 5 | Peter to prepare proposal document summarizing engineering priorities alignment | Peter O'Neil | 0.675 | explicit |
| 6 | Jackie will provide Lightbox with a cash flow ledger Excel detailing AML program financials | Jackie | 0.59 | explicit |

### 3.2 Scoring Validation

All items have full scoring populated:

| Metric | Coverage | Range |
|--------|----------|-------|
| commitment_strength | 6/6 (100%) | All "explicit" |
| score_impact | 6/6 (100%) | 3–5 |
| score_urgency | 6/6 (100%) | 2–4 |
| score_specificity | 6/6 (100%) | 4 (all) |
| score_effort | 6/6 (100%) | 1–4 |
| priority_score | 6/6 (100%) | 0.59–0.82 |
| definition_of_done | 6/6 (100%) | All populated |
| confidence | 6/6 (100%) | 0.85–0.95 |

### 3.3 Owner Resolution

| Owner | Action Items | Resolution |
|-------|-------------|------------|
| Jackie | 3 | Named (direct extraction) |
| Peter O'Neil | 1 | Named (canonical form — resolved from "Peter") |
| David | 1 | Named (direct extraction) |
| Alejandro | 1 | Named (direct extraction) |

**0 owner fragmentation** — "Peter" correctly resolved to "Peter O'Neil" canonical form. Compare with pre-overhaul run where the same transcripts produced owner variants like "E", "B", "C", "F" (single-letter initials) alongside full names.

### 3.4 Topic Clustering

| Topic | Action Items |
|-------|-------------|
| Application Modernization Lab Engagement | 4 |
| AWS Client VPN Support | 1 |
| Database Backup Compliance | 1 |

**100% topic coverage** (6/6 items linked). Compare with pre-overhaul: 16 topics, many overlapping.

### 3.5 Structural Integrity

| Check | Result |
|-------|--------|
| EXTRACTED_FROM links | 6/6 (100%) — all items linked to source Interaction |
| BELONGS_TO topic links | 6/6 (100%) — all items linked to a topic |
| OWNED_BY links | 6/6 (100%) — all items linked to an Owner node |
| Interactions with items | 3/4 (Call 4 correctly produced 0 items) |

---

## 4. Cross-Pipeline MERGE Verification

Both pipelines converge on shared Account and Interaction nodes via MERGE.

### 4.1 Account Convergence

| Check | Result |
|-------|--------|
| Account nodes for `account_id=aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa` | **1** (MERGE convergence confirmed) |

### 4.2 Interaction Convergence

| Transcript | Interaction ID | AI Items | Deal Activity | Result |
|------------|---------------|----------|---------------|--------|
| Call 1 | `71917bf5-34f4-4dd5-9af9-19df6d627af3` | 3 | 1 deal created | PASS |
| Call 2 | `86ee0ec4-b28a-40d1-9d7a-ceea96a91511` | 2 | 1 deal merged | PASS |
| Call 3 | `eeef8401-b974-45be-b3b9-69c8e28142cc` | 1 | 2 deals created | PASS |
| Call 4 | (generated) | 0 | 0 | PASS |

**4/4 interactions processed by both pipelines. ALL CHECKS PASSED.**

---

## 5. Deal Pipeline — Final State

### 5.1 Deals (3)

The Deal pipeline results are consistent with the pre-overhaul run:

| Deal | Stage | MEDDIC Completeness |
|------|-------|-------------------|
| Lightbox Application Modernization Lab Engagement | qualification | 100% (6/6) |
| Incident Detection and Response (IDR) Service | qualification | 100% (6/6) |
| Enterprise Discount Program (EDP) Renewal | negotiation | 100% (6/6) |

---

## 6. Quality Comparison: Pre vs Post Overhaul

| Metric | Pre-Overhaul (2026-02-11) | Post-Overhaul (2026-03-12) | Change |
|--------|---------------------------|----------------------------|--------|
| Action items extracted | 30 | 6 | -80% |
| Topics | 16 | 3 | -81% |
| Owners | 5 (with single-letter initials) | 4 (all named, canonical) | Improved quality |
| Items with priority_score | 0/30 (0%) | 6/6 (100%) | New capability |
| Items with definition_of_done | 0/30 (0%) | 6/6 (100%) | New capability |
| Items with commitment_strength | 0/30 (0%) | 6/6 (100%) | New capability |
| Owner fragmentation | High (E, B, C, F initials) | Zero | Resolved |
| Observations extracted as tasks | Multiple | Zero | Eliminated |
| Deals | 3 | 3 | Unchanged |
| Pipeline errors | 0 | 0 | Unchanged |

---

## 7. Aggregate Statistics

| Metric | Value |
|--------|-------|
| Action Items | 6 |
| ActionItemTopics | 3 |
| Owners | 4 |
| Deals | 3 |
| DealVersions | 1 |
| Interactions | 4 (shared, enriched by both pipelines) |
| Accounts | 1 (shared, MERGE convergence verified) |
| Items with Topics | 6/6 (100%) |
| Items with Scoring | 6/6 (100%) |
| Items with EXTRACTED_FROM | 6/6 (100%) |
| Errors | 0 |

---

## 8. Schema & Constraints

### AI-owned labels (UNIQUENESS constraints)

| Constraint | Label | Properties |
|-----------|-------|------------|
| `action_item_unique` | ActionItem | `(tenant_id, action_item_id)` |
| `action_item_version_unique` | ActionItemVersion | `(tenant_id, version_id)` |
| `owner_unique` | Owner | `(tenant_id, owner_id)` |
| `action_item_topic_unique` | ActionItemTopic | `(tenant_id, action_item_topic_id)` |
| `action_item_topic_version_unique` | ActionItemTopicVersion | `(tenant_id, version_id)` |

### Deal-owned labels

| Constraint | Label | Properties |
|-----------|-------|------------|
| `dealversion_unique` | DealVersion | `(tenant_id, version_id)` |
| `deal_unique_v2` | Deal | `(tenant_id, opportunity_id)` |

### Shared labels (skeleton-owned)

Account and Interaction constraints are owned by upstream `eq-structured-graph-core`. Both pipelines use defensive MERGE on these labels.

---

## 9. Verification Checklist

- [x] All 4 transcripts processed — both pipelines succeeded for every call
- [x] Both OK = Yes for all 4 transcripts (zero errors)
- [x] **Cross-pipeline MERGE**: 1 Account node (no duplicates from two pipelines)
- [x] **Cross-pipeline MERGE**: 4/4 Interactions enriched by both pipelines
- [x] **Quality pipeline**: 6 items extracted (down from 30 pre-overhaul)
- [x] **Quality pipeline**: 100% commitment_strength populated (all "explicit")
- [x] **Quality pipeline**: 100% priority_score populated (range 0.59–0.82)
- [x] **Quality pipeline**: 100% definition_of_done populated
- [x] **Quality pipeline**: 0% owner fragmentation (4 distinct named owners)
- [x] **Quality pipeline**: 100% topic coverage (3 topics, all items linked via BELONGS_TO)
- [x] **Quality pipeline**: 100% EXTRACTED_FROM links
- [x] Deals=3, DealVersions=1 (unchanged from pre-overhaul)
- [x] All MEDDIC dimensions populated for all deals
- [x] UNIQUENESS constraints (not NODE KEY) for AI-owned labels
- [x] Label-scoped cleanup preserves upstream skeleton nodes
- [x] Unit tests: 494 passed, 0 failed
- [x] Postgres owner upsert fix validated (ON CONFLICT tenant_id, canonical_name)

---

## 10. Historical Comparison

| Run Date | Pipeline Version | AI Items | Topics | Owners | Deals | Errors |
|----------|-----------------|----------|--------|--------|-------|--------|
| 2026-02-11 | Pre-quality-overhaul | 30 | 16 | 5 | 3 | 0 |
| 2026-03-12 | Post-quality-overhaul | 6 | 3 | 4 | 3 | 0 |
