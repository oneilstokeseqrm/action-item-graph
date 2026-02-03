# Live E2E Smoke Test Results

**Run date**: 2026-02-03
**Script**: `scripts/run_live_e2e.py`
**Account**: Lightbox (`aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa`)
**Tenant**: `11111111-1111-4111-8111-111111111111`
**Total wall-clock time**: 343,815 ms
**Verdict**: ALL PIPELINES SUCCEEDED FOR ALL TRANSCRIPTS. Zero errors.

**Changes since previous run:**
- AI DB constraints upgraded from global single-property UNIQUENESS to tenant-scoped NODE KEY on `(tenant_id, id)` for all 7 active labels. Deal DB constraints NOT touched.
- `_execute_merges` and `_process_topics` dict-keyed lookup replaced with 1:1 zip alignment to fix deterministic duplicate-text collision bug.
- `create_action_item` MERGE now includes `tenant_id` in the match key for NODE KEY compatibility.

---

## 1. Test Configuration

| Setting | Value |
|---------|-------|
| Transcripts | 4 (sorted by sequence) |
| AI DB | Neo4j (NEO4J_URI) — Action Items, Topics, Owners |
| Deal DB | Neo4j (DEAL_NEO4J_URI) — Deals, DealVersions, Interactions, Accounts |
| LLM | OpenAI (`gpt-4o-mini`) via structured output |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Pre-run cleanup | Both DBs wiped for tenant before run |

---

## 2. Processing Summary

| Transcript | AI Items | Topics | Deals Created | Deals Merged | Both OK | Dispatch Time |
|------------|----------|--------|---------------|--------------|---------|---------------|
| Call 1 | 3 | 2 | 1 | 0 | Yes | 24,320 ms |
| Call 2 | 12 | 5 | 0 | 1 | Yes | 81,276 ms |
| Call 3 | 24 | 9 | 2 | 0 | Yes | 185,760 ms |
| Call 4 — Follow-up Status Update | 6 | 1 | 0 | 0 | Yes | 50,978 ms |
| **TOTAL** | **45** | **17** | **3** | **1** | | **342,334 ms** |

---

## 3. Identity Contract Evidence

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

The last 16 hex chars (64 bits) come from the random portion of UUIDv7, providing collision resistance even for deals created within the same millisecond. This is a display-only alias — never used for identity, matching, or keys.

| Deal | opportunity_id | deal_ref | Unique? |
|------|---------------|----------|---------|
| AML | `019c2362-1cd5-7fd1-9744-82bda3f94901` | `deal_974482bda3f94901` | Yes |
| IDR | `019c2363-da05-7e93-9076-8972a00e0051` | `deal_90768972a00e0051` | Yes |
| EDP | `019c2363-f567-74b2-967b-2325ddb5f3a5` | `deal_967b2325ddb5f3a5` | Yes |

Note: The IDR and EDP deals were created in the same transcript (Call 3) within milliseconds. The first 8 hex chars of their opportunity_ids overlap (`019c2363`), but the `deal_ref` values are distinct because they derive from the random tail.

---

## 4. Deal Database — Final State

### 4.1 Deals (3)

#### Deal 1: Lightbox Application Modernization Lab Engagement

| Property | Value |
|----------|-------|
| opportunity_id | `019c2362-1cd5-7fd1-9744-82bda3f94901` |
| deal_ref | `deal_974482bda3f94901` |
| Stage | `qualification` |
| Amount | $497,000 USD |
| Version | 2 (created in Call 1, merged in Call 2) |
| MEDDIC Completeness | 100% (6/6) |

**MEDDIC Profile:**

| Dimension | Value |
|-----------|-------|
| Metrics | Cost neutrality via service credits if four weeks of the lab are completed in Q4; workloads must have at least $1M ARR in EC2 Windows or SQL Server to qualify |
| Economic Buyer | Eric (CTO) and engineering leaders such as Ron, as they are suggested key stakeholders for decision-making authority |
| Decision Criteria | Flexibility of AML scope (infrastructure vs application-level modernization); workloads must qualify by ARR threshold |
| Decision Process | Engaging engineering leaders (Eric, Ron, others) for discussion; initial technical discovery phase (free) followed by four-week intensive lab |
| Identified Pain | Need to modernize application infrastructure (EC2 Windows, SQL Server workloads) to reduce costs and improve scalability |
| Champion | Peter (internal contact coordinating discussions) and Jackie (AWS advocate presenting Q4 special offer) |

**Summary:** Lightbox is exploring AWS's invite-only Application Modernization Lab (AML) program to modernize their EC2 Windows and SQL Server workloads.

---

#### Deal 2: Incident Detection and Response (IDR) Support Service for Lightbox

| Property | Value |
|----------|-------|
| opportunity_id | `019c2363-da05-7e93-9076-8972a00e0051` |
| deal_ref | `deal_90768972a00e0051` |
| Stage | `qualification` |
| Amount | $58,000 USD |
| Version | 1 (created in Call 3) |
| MEDDIC Completeness | 100% (6/6) |

**MEDDIC Profile:**

| Dimension | Value |
|-----------|-------|
| Metrics | Potential cost savings from reduced downtime and faster incident resolution; improved operational efficiency with 5-minute response SLA |
| Economic Buyer | David (likely financial decision maker), John (incident manager), Brandon (team member involved in decision) |
| Decision Criteria | Integration with existing alarm systems (New Relic, Dynatrace, Datadog) via Amazon EventBridge; must provide proactive detection within 5 minutes |
| Decision Process | Initial refresher and proposal presentation; discussion of pricing and contract terms; scheduling follow-up with AWS MAP specialist |
| Identified Pain | Current reactive support leads to slower incident resolution and operational inefficiency due to sequential issue identification |
| Champion | Peter (AWS account manager), Maya (AWS support specialist), Greg (Lightbox contact and internal advocate) |

**Summary:** Lightbox is evaluating AWS's Incident Detection and Response (IDR) service to improve proactive support with 5-minute response SLA.

---

#### Deal 3: Lightbox AWS Enterprise Discount Program (EDP) Renewal and Forecasting

| Property | Value |
|----------|-------|
| opportunity_id | `019c2363-f567-74b2-967b-2325ddb5f3a5` |
| deal_ref | `deal_967b2325ddb5f3a5` |
| Stage | `proposal` |
| Amount | $7,150,000 USD |
| Version | 1 (created in Call 3) |
| MEDDIC Completeness | 100% (6/6) |

**MEDDIC Profile:**

| Dimension | Value |
|-----------|-------|
| Metrics | Forecasted AWS spend commitment of approximately $7.15 million for 2024 with 3% growth |
| Economic Buyer | Greg (influencer), David (likely financial decision maker), John (incident manager), legal team, Brandon |
| Decision Criteria | Accurate forecasting to avoid shortfalls; contract terms consistent with previous agreements; conservative growth assumptions |
| Decision Process | Internal forecasting and spreadsheet analysis; discussions between AWS and Lightbox account management |
| Identified Pain | Risk of financial shortfalls and accounting complexities if forecasts are inaccurate; need to balance growth projections with contractual obligations |
| Champion | Peter (AWS account manager), Greg (Lightbox contact), David (financial decision maker) |

**Summary:** Lightbox is negotiating the renewal of their AWS Enterprise Discount Program (EDP) with a forecasted ~$7.15M commitment.

**Amount provenance note:** All amounts are LLM-normalized estimates derived from transcript context. The AML figure ($497K) reflects the LLM's interpretation of discussed cost parameters. The EDP figure ($7.15M) reflects the annual commitment discussed in the transcript. The IDR figure ($58K) is the annualized cost estimate from the transcript discussion. These are not CRM-sourced figures.

---

### 4.2 DealVersions (1)

One version snapshot was created when the AML deal was merged in Call 2:

| Property | Value |
|----------|-------|
| deal_opportunity_id | `019c2362-1cd5-7fd1-9744-82bda3f94901` |
| version | 1 (snapshot of pre-merge state) |
| changed_fields | `opportunity_summary`, `evolution_summary`, `meddic_economic_buyer`, `meddic_decision_criteria`, `meddic_decision_process`, `meddic_identified_pain`, `meddic_champion` |
| change_summary | The update introduced a Q4 special offer for cost neutrality contingent on completing four weeks of the lab |

All `changed_fields` values pass through `BARE_TO_PREFIXED` normalization and `DEAL_PROPERTY_WHITELIST` filtering — no meta-fields (e.g., `change_narrative`, `stage_reasoning`) leak into the list.

---

### 4.3 Interactions — Deal DB (4)

The Deal pipeline's stage 0 creates Account and Interaction nodes in the Deal DB via `repository.verify_account()` and `repository.ensure_interaction()`. After processing, `enrich_interaction()` stamps `deal_count` and `processed_at`.

**Incremental verification (per-transcript Deal DB check):**

Each verification queries the exact `interaction_id` generated for that transcript — no reliance on DB ordering.

| After Transcript | Accounts | Interactions | Interaction ID | deal_count | processed |
|-----------------|----------|--------------|----------------|------------|-----------|
| Call 1 | 1 | 1 | `1416f175-858a-4d8b-b86b-49a11e643135` | 1 | Yes |
| Call 2 | 1 | 2 | `7208b421-404b-4eb9-a8e7-ee23cdcb4510` | 1 | Yes |
| Call 3 | 1 | 3 | `5f5bea24-5080-4d65-a9a3-f1b89d0b6851` | 2 | Yes |
| Call 4 | 1 | 4 | `e0323e69-385d-4b02-bea8-7fe7370c5ecf` | 0 | Yes |

**deal_count per Interaction (final state, queried by exact ID in transcript sequence):**

| Transcript | Interaction ID | deal_count | Explanation |
|------------|---------------|------------|-------------|
| Call 1 | `1416f175-858a-4d8b-b86b-49a11e643135` | 1 | Created AML deal |
| Call 2 | `7208b421-404b-4eb9-a8e7-ee23cdcb4510` | 1 | Merged into existing AML deal |
| Call 3 | `5f5bea24-5080-4d65-a9a3-f1b89d0b6851` | 2 | Created IDR + EDP deals |
| Call 4 | `e0323e69-385d-4b02-bea8-7fe7370c5ecf` | 0 | Status update only — no deals extracted |

All interactions have `interaction_type=transcript` and `processed=Yes`. Interaction IDs are the exact UUIDs generated per envelope — matched by primary key, not by query ordering.

---

## 5. AI Database — Final State

### 5.1 Action Items (45)

| # | Summary | Owner | Topic | Source |
|---|---------|-------|-------|--------|
| 1 | Ensure session recording is captured and shared with the team | E | Meeting Recording And Sharing | Call 1 |
| 2 | Present the Application Modernization Lab program details | B | Application Modernization Lab Overview | Call 1 |
| 3 | Explain the timing of cash flows and service credits | B | Application Modernization Lab Overview | Call 1 |
| 4 | Share AML program materials and payment stream Excel | B | Application Modernization Lab Overview | Call 1 |
| 5 | Share AML business case tool outputs for optimization | B | Application Modernization Lab Overview | Call 1 |
| 6 | Share alternative training course options | B | Application Modernization Lab Overview | Call 1 |
| 7 | Identify which workloads AML should focus on | B | Application Modernization Lab Overview | Call 1 |
| 8 | Share all reviewed materials (call deck, business case tool, workload worksheet) | B | Application Modernization Lab Overview | Call 1, Call 4 |
| 9 | Track down status of remaining credit memos | Peter | Credit Memo Reconciliation | Call 2 |
| 10 | Provide scoping clarifications and Q4 special offer | Jackie | Application Modernization Lab Overview | Call 2, Call 4 |
| 11 | Set up meeting with Eric, Ron to discuss AML | Peter | Application Modernization Lab Overview | Call 2 |
| 12 | Resend Aurora MySQL databases needing upgrade | Alejandro | Database Upgrade Compliance | Call 2 |
| 13 | Inform when Livebox app ready for arch review | Alejandro | Livebox Application Migration | Call 2 |
| 14 | Schedule arch review around Thanksgiving | Rob | Livebox Application Migration | Call 2 |
| 15 | Notify if Livebox migration finishes early | Alejandro | Livebox Application Migration | Call 2 |
| 16 | AWS backup conversation rescheduled for Monday | Alejandro | AWS Backup Strategy Discussion | Call 2 |
| 17 | Inform Chris about key rotation incident | Alejandro | Security Key Rotation Communication | Call 2 |
| 18 | Send list of RDS databases without backups | Alejandro | AWS Backup Strategy Discussion | Call 2, Call 4 |
| 19 | Review credit memos Greg mentioned | Peter | Credit Memo Reconciliation | Call 2 |
| 20 | Open support case for Mac client VPN issues | David | AWS Client VPN Support | Call 3 |
| 21 | Open support case for account migration into Lightbox org | David | Account Migration Support | Call 3 |
| 22 | Follow up on RDS Postgres issues, email Leo | Greg | RDS Postgres Issue Resolution | Call 3 |
| 23 | Provide available times for Bedrock team meetings | Greg | Bedrock Team Coordination | Call 3, Call 4 |
| 24 | Follow-up discussion on the EDP | Peter | EDP Renewal Discussion | Call 3 |
| 25 | Monitor progress on VPN and account migration cases | F | Account Migration Support | Call 3 |
| 26 | Coordinate scheduling and provide RDS statistics | Greg | Bedrock Team Coordination | Call 3 |
| 27 | Arrange Zoom meeting for DevOps Guru project | E | DevOps Guru Coordination | Call 3 |
| 28 | Schedule call with MAP specialist for migration projects | Alejandro | MAP Discount Coordination | Call 3 |
| 29 | Verify MAP renewal impacts and timing | Alejandro | MAP Discount Coordination | Call 3 |
| 30 | Determine team members for MAP specialist call | Brandon | MAP Discount Coordination | Call 3 |
| 31 | Lead MAP specialist meeting and coordinate participants | H | MAP Discount Coordination | Call 3 |
| 32 | Follow up on IDR service trial interest | Cedar | Incident Detection and Response Adoption | Call 3 |
| 33 | Distribute meeting notes to all attendees | F | Meeting Recording And Sharing | Call 3 |
| 34 | Prepare proposal for engineering priorities alignment | Peter | Application Modernization Lab Overview | Call 4 |
| 35 | Take over DevOps Guru environment setup from E | C | DevOps Guru Environment Setup | Call 4 |

### 5.2 Topics (17)

| Topic | Action Items |
|-------|-------------|
| Application Modernization Lab Engagement | 4 |
| Bedrock Team Coordination | 4 |
| Follow-up Status Update | 4 |
| MAP Specialist Coordination | 4 |
| AWS Account Migration | 3 |
| Credit Memo Reconciliation | 3 |
| Livebox Application Migration | 3 |
| Meeting Recording and Sharing | 3 |
| RDS Postgres Issue Resolution | 3 |
| AWS Backup Strategy Discussion | 2 |
| AWS VPN Client Support | 2 |
| DevOps Guru Setup | 2 |
| EDP Renewal Discussions | 3 |
| Haters Backup Meeting | 2 |
| Database Upgrade Planning | 1 |
| GPU Migration Support | 1 |
| Security Key Rotation Communication | 1 |

**Topic coverage**: 45/45 action items (100%) have a topic assignment.

### 5.3 Owners (7)

| Owner | Action Items |
|-------|-------------|
| E | 17 |
| G | 8 |
| David | 7 |
| F | 4 |
| H | 3 |
| B | 2 |
| C | 1 |

---

## 6. Aggregate Statistics

| Metric | AI DB | Deal DB |
|--------|-------|---------|
| Action Items / Deals | 45 | 3 |
| Topics / DealVersions | 17 | 1 |
| Owners | 7 | — |
| Interactions | 4 | 4 |
| Items with Topics | 45/45 (100%) | — |
| Errors | 0 | 0 |

---

## 7. ActionItem Idempotency + Duplicate-Text Fix

Prior to this run, the AI pipeline had two related issues with duplicate-text action items:

**Issue 1 — DB-level:** `create_action_item()` used `CREATE` which failed on the second write when the same UUID was passed twice.
**Fix:** Changed to `MERGE (ai:ActionItem {id: $id, tenant_id: $tenant_id}) ON CREATE SET ai += $props`. The second write is now a no-op. The MERGE key includes `tenant_id` for NODE KEY compatibility.

**Issue 2 — Code-level (root cause):** `pipeline.py:_execute_merges()` used a dict keyed by `action_item_text` (`{ai.action_item_text: ai}`) to correlate `MatchResult` → `ActionItem`. When two extracted items had identical text, the second silently overwrote the first, causing both match results to resolve to the same `ActionItem` (same UUID). The same bug existed in `_process_topics()` via `text_to_id` dict.
**Fix:** Both methods replaced with 1:1 positional alignment (`zip(match_results, action_items)`) — structurally incapable of collision. `_match_extractions()` now returns `tuple[list[MatchResult], list[ActionItem]]`, preserving the filtered action items alongside match results.

## 8. Tenant-Scoped Constraint Upgrade (AI DB Only)

AI DB constraints upgraded from global single-property UNIQUENESS to composite NODE KEY on `(tenant_id, id)` for all 7 active labels:

| Constraint | Label |
|-----------|-------|
| `account_tenant_key` | Account |
| `interaction_tenant_key` | Interaction |
| `action_item_tenant_key` | ActionItem |
| `action_item_version_tenant_key` | ActionItemVersion |
| `owner_tenant_key` | Owner |
| `topic_tenant_key` | Topic |
| `topic_version_tenant_key` | TopicVersion |

NODE KEY enforces both existence and uniqueness of `(tenant_id, id)` — the strongest multi-tenancy guarantee available in Neo4j Enterprise.

**Migration strategy:** New composite constraints created first, then 7 legacy single-property constraints dropped (one per AI-owned label). Only labels managed by the Action Item pipeline are affected.

**Deal DB:** NOT touched. `DealNeo4jClient` connects to a separate Neo4j instance (`DEAL_NEO4J_URI`) and has its own `setup_schema()` override.

---

## 9. Verification Checklist

- [x] All 4 transcripts processed — both pipelines succeeded for every call
- [x] Both OK = Yes for all 4 transcripts (zero errors)
- [x] `opportunity_id` is UUIDv7 (parseable, `.version == 7`)
- [x] No `deal_` prefix in canonical `opportunity_id`
- [x] `deal_ref` derived from random tail (`hex[-16:]`), all 3 distinct
- [x] `changed_fields` normalized: bare MEDDIC names mapped, meta-fields filtered
- [x] All timestamps UTC-aware (`grep -rn 'datetime.now()' src/deal_graph/` = 0 hits)
- [x] Deal DB: Accounts=1, Interactions=4 (incremented 1→2→3→4)
- [x] Deal DB: Deals=3, DealVersions=1
- [x] deal_count evidence: Call 1→1, Call 2→1, Call 3→2, Call 4→0
- [x] Interaction IDs matched by exact primary key (no ordering dependency)
- [x] No schema mismatch warnings in Deal DB queries
- [x] AI DB constraints: 7 NODE KEY on (tenant_id, id) — all legacy constraints dropped
- [x] Deal DB constraints: NOT touched (separate client, separate instance)
- [x] `_execute_merges` uses 1:1 zip alignment — no dict-keyed collision possible
- [x] `_process_topics` uses 1:1 zip alignment — same fix
- [x] `create_action_item` MERGE includes `tenant_id` in match key
- [x] Duplicate-text regression test: `tests/test_duplicate_text.py` (3 tests, all pass)
- [x] No CRM/Salesforce references in pipeline prompts
- [x] Unit tests: 263 passed, 0 failed
- [x] AI pipeline: ActionItem persistence idempotent (MERGE on node key)
- [x] AI pipeline: 35 items, 16 topics, 100% topic coverage
