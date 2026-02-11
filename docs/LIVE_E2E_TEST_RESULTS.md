# Live E2E Smoke Test Results

**Run date**: 2026-02-11
**Script**: `scripts/run_live_e2e.py`
**Account**: Lightbox (`aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa`)
**Tenant**: `11111111-1111-4111-8111-111111111111`
**Total wall-clock time**: 247,843 ms
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

**Changes since previous run (2026-02-03):**
- **Graph integration**: Merged AI DB into shared structured DB — single database for both pipelines
- **Label renames**: `Topic` → `ActionItemTopic`, `TopicVersion` → `ActionItemTopicVersion`
- **Property renames**: Generic `id` → label-specific keys (`account_id`, `interaction_id`, `action_item_id`, `owner_id`, `action_item_topic_id`, `version_id`)
- **Field renames**: `transcript_text` → `content_text`, `occurred_at` → `timestamp` (on Interaction)
- **Constraint type**: NODE KEY → UNIQUENESS (matching structured DB convention)
- **Interaction persistence**: CREATE → defensive MERGE with ON CREATE / ON MATCH
- **Cleanup**: Label-scoped cleanup (respects upstream skeleton nodes in shared DB)
- **New**: Cross-pipeline MERGE verification — proves both pipelines converge on shared Account/Interaction nodes

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

| Transcript | AI Items | Topics | Deals Created | Deals Merged | Both OK | Dispatch Time |
|------------|----------|--------|---------------|--------------|---------|---------------|
| Call 1 | 3 | 2 | 1 | 0 | Yes | 21,046 ms |
| Call 2 | 12 | 5 | 0 | 1 | Yes | 87,173 ms |
| Call 3 | 9 | 7 | 2 | 0 | Yes | 80,809 ms |
| Call 4 — Follow-up Status Update | 6 | 2 | 0 | 0 | Yes | 57,379 ms |
| **TOTAL** | **30** | **16** | **3** | **1** | | **246,407 ms** |

---

## 3. Cross-Pipeline MERGE Verification

This section validates the core integration thesis: both pipelines converge on shared Account and Interaction nodes via MERGE, using the new label-specific key properties.

### 3.1 Account Convergence

| Check | Result |
|-------|--------|
| Account nodes for `account_id=aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa` | **1** (MERGE convergence confirmed) |
| `account_id` property present | Yes |
| `tenant_id` property present | Yes |

Both pipelines MERGE on `{account_id: $account_id, tenant_id: $tenant_id}`. A single Account node confirms no duplication.

### 3.2 Interaction Convergence

Each Interaction node is enriched by **both** pipelines: `action_item_count` (AI pipeline) and `deal_count` (Deal pipeline) on the same node.

| Transcript | Interaction ID | action_item_count | deal_count | has_content | has_timestamp | Result |
|------------|---------------|-------------------|------------|-------------|---------------|--------|
| Call 1 | `e44526c5-e13b-443a-abee-4ce8ff5dbff0` | 3 | 1 | Yes | Yes | PASS |
| Call 2 | `3e01308d-e05e-4d50-80f3-388db239cfc3` | 12 | 1 | Yes | Yes | PASS |
| Call 3 | `c4cf76e5-761f-4d8b-8279-41d29b5e50df` | 9 | 2 | Yes | Yes | PASS |
| Call 4 | `f17595bd-33b2-4ad2-a926-24f67e1ea455` | 6 | 0 | Yes | Yes | PASS |

**4/4 interactions enriched by both pipelines. ALL CHECKS PASSED.**

---

## 4. Identity Contract Evidence

### opportunity_id

All Deal `opportunity_id` values are UUIDv7 (RFC 9562), minted by `deal_graph.utils.uuid7()`. They are the canonical internal primary key — our system IS the CRM.

- Model layer: `Deal.opportunity_id: UUID` (Pydantic)
- Neo4j layer: serialized to `str` via `to_neo4j_properties()`
- No `deal_` prefix in canonical ID
- No Salesforce/CRM identifiers anywhere in the pipeline

### deal_ref (display-only alias)

`deal_ref` is derived deterministically from the **random-heavy tail** of the UUIDv7:

```
deal_ref = "deal_" + opportunity_id.hex[-16:]
```

| Deal | opportunity_id | deal_ref | Unique? |
|------|---------------|----------|---------|
| AML | `019c4e77-a2d2-7731-a01d-d339bbc80bcb` | `deal_a01dd339bbc80bcb` | Yes |
| IDR | `019c4e79-74ce-7061-a4bc-da0994772a7f` | `deal_a4bcda0994772a7f` | Yes |
| EDP | `019c4e79-8908-7a82-84a9-f1626b3b1f5f` | `deal_84a9f1626b3b1f5f` | Yes |

---

## 5. Deal Pipeline — Final State

### 5.1 Deals (3)

#### Deal 1: Lightbox Application Modernization Lab Engagement

| Property | Value |
|----------|-------|
| opportunity_id | `019c4e77-a2d2-7731-a01d-d339bbc80bcb` |
| deal_ref | `deal_a01dd339bbc80bcb` |
| Stage | `qualification` |
| Amount | $497,000 USD |
| Version | 2 (created in Call 1, merged in Call 2) |
| MEDDIC Completeness | 100% (6/6) |

#### Deal 2: Incident Detection and Response (IDR) Service Adoption

| Property | Value |
|----------|-------|
| opportunity_id | `019c4e79-74ce-7061-a4bc-da0994772a7f` |
| deal_ref | `deal_a4bcda0994772a7f` |
| Stage | `qualification` |
| Amount | $58,000 USD |
| Version | 1 (created in Call 3) |
| MEDDIC Completeness | 100% (6/6) |

#### Deal 3: Enterprise Discount Program (EDP) Renewal and Forecasting

| Property | Value |
|----------|-------|
| opportunity_id | `019c4e79-8908-7a82-84a9-f1626b3b1f5f` |
| deal_ref | `deal_84a9f1626b3b1f5f` |
| Stage | `negotiation` |
| Amount | $37,150,000 USD |
| Version | 1 (created in Call 3) |
| MEDDIC Completeness | 100% (6/6) |

**Amount provenance note:** All amounts are LLM-normalized estimates derived from transcript context. These are not CRM-sourced figures.

### 5.2 DealVersions (1)

One version snapshot was created when the AML deal was merged in Call 2:

| Property | Value |
|----------|-------|
| deal_opportunity_id | `019c4e77-a2d2-7731-a01d-d339bbc80bcb` |
| version | 1 (snapshot of pre-merge state) |
| changed_fields | `meddic_economic_buyer`, `meddic_decision_criteria`, `meddic_decision_process`, `meddic_identified_pain`, `meddic_champion`, `meddic_metrics`, `opportunity_summary`, `evolution_summary` |

---

## 6. AI Pipeline — Final State

### 6.1 Action Items (30)

| # | Summary | Owner | Topic |
|---|---------|-------|-------|
| 1 | Ensure session recording is captured and shared | E | Meeting Recording And Sharing |
| 2 | Start meeting recording and share afterwards | E | Meeting Recording And Sharing |
| 3 | Share business case tool outputs and materials | B | Application Modernization Lab Engagement |
| 4 | Share business case tool outputs (follow-up) | B | Follow-up Status Update |
| 5 | Track down remaining VMware credit memos | Peter | Credit Memo Reconciliation |
| 6 | Clarify VMC credit memo handling | Peter | Credit Memo Reconciliation |
| 7 | Provide scoping clarifications and Q4 offer | Jackie Rusk | Application Modernization Lab Engagement |
| 8 | Provide scoping clarifications (follow-up) | Jackie Rusk | Follow-up Status Update |
| 9 | Set up meeting with Eric, Ron for AML | Peter | Application Modernization Lab Engagement |
| 10 | Resend Aurora MySQL upgrade list | Alejandro Torres | Database Upgrade Planning |
| 11 | Inform when Livebox app ready for arch review | Alejandro Torres | Livebox Application Migration |
| 12 | Schedule arch review around Thanksgiving | Rob | Livebox Application Migration |
| 13 | Accommodate earlier scheduling if ready | Alejandro Torres | Livebox Application Migration |
| 14 | AWS backup conversation rescheduled for Monday | Alejandro Torres | AWS Backup Strategy Discussion |
| 15 | Inform Chris about key rotation resolution | Alejandro Torres | Security Key Rotation Communication |
| 16 | Send list of RDS databases without backups | Alejandro Torres | AWS Backup Strategy Discussion |
| 17 | Send RDS backup list (follow-up) | Alejandro Torres | Follow-up Status Update |
| 18 | Review credit memos Greg mentioned | Alejandro Torres | Credit Memo Reconciliation |
| 19 | Open support case for Mac VPN issues | David | AWS Client VPN Support |
| 20 | Open support case for account migration | David | AWS Account Migration Issues |
| 21 | Follow up on RDS Postgres issues | Greg | RDS Postgres Issue Resolution |
| 22 | Provide times for Bedrock team meetings | Greg | Application Modernization Lab Engagement |
| 23 | Provide times for Bedrock meetings | Greg | Bedrock Team Coordination |
| 24 | Monitor VPN and account migration cases | F | AWS Account Migration Issues |
| 25 | Arrange Zoom for DevOps Guru | E | DevOps Guru Coordination |
| 26 | Distribute meeting notes | F | Meeting Recording And Sharing |
| 27 | Follow up on IDR service trial interest | C | Incident Detection and Response Evaluation |
| 28 | Seek internal approvals for EDP renewal | Peter | EDP Renewal Approval Process |
| 29 | Prepare engineering priorities proposal | Peter | Application Modernization Lab Engagement |
| 30 | Take over DevOps Guru setup from E | C | DevOps Environment Setup |

### 6.2 Topics (16)

| Topic | Action Items |
|-------|-------------|
| Application Modernization Lab Engagement | 5 |
| Follow-up Status Update | 3 |
| Credit Memo Reconciliation | 3 |
| Livebox Application Migration | 3 |
| Meeting Recording And Sharing | 3 |
| AWS Account Migration Issues | 2 |
| AWS Backup Strategy Discussion | 2 |
| AWS Client VPN Support | 1 |
| Bedrock Team Coordination | 1 |
| Database Upgrade Planning | 1 |
| DevOps Environment Setup | 1 |
| DevOps Guru Coordination | 1 |
| EDP Renewal Approval Process | 1 |
| Incident Detection and Response Evaluation | 1 |
| RDS Postgres Issue Resolution | 1 |
| Security Key Rotation Communication | 1 |

**Topic coverage**: 30/30 action items (100%) have a topic assignment.

### 6.3 Owners (5)

| Owner | Action Items |
|-------|-------------|
| E | 18 |
| B | 2 |
| David | 2 |
| F | 2 |
| C | 2 |

---

## 7. Aggregate Statistics

| Metric | Value |
|--------|-------|
| Action Items | 30 |
| ActionItemTopics | 16 |
| Owners | 5 |
| Deals | 3 |
| DealVersions | 1 |
| Interactions | 4 (shared, enriched by both pipelines) |
| Accounts | 1 (shared, MERGE convergence verified) |
| Items with Topics | 30/30 (100%) |
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
- [x] `account_id` property present on Account node (label-specific key)
- [x] `interaction_id` property present on Interaction nodes (label-specific key)
- [x] `content_text` and `timestamp` properties present on Interaction nodes
- [x] `opportunity_id` is UUIDv7 (parseable, `.version == 7`)
- [x] No `deal_` prefix in canonical `opportunity_id`
- [x] `deal_ref` derived from random tail (`hex[-16:]`), all 3 distinct
- [x] `changed_fields` normalized: bare MEDDIC names mapped, meta-fields filtered
- [x] Deals=3, DealVersions=1
- [x] deal_count evidence: Call 1→1, Call 2→1, Call 3→2, Call 4→0
- [x] Interaction IDs matched by exact primary key (no ordering dependency)
- [x] All `:Topic` labels → `:ActionItemTopic` in Cypher (zero stale references in source)
- [x] All generic `{id:` → label-specific keys in Cypher (zero stale references in source)
- [x] UNIQUENESS constraints (not NODE KEY) for AI-owned labels
- [x] Label-scoped cleanup preserves upstream skeleton nodes (2711 nodes untouched)
- [x] No CRM/Salesforce references in pipeline prompts
- [x] Unit tests: 266 passed, 0 failed
- [x] AI pipeline: 30 items, 16 topics, 100% topic coverage
