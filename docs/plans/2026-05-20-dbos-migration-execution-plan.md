# DBOS Migration — Execution Plan

**Status:** APPROVED (post-/plan-eng-review + Codex outside voice absorption)
**Created:** 2026-05-20
**Author:** /plan-eng-review (Claude Code session)
**Design doc:** `~/.gstack/projects/oneilstokeseqrm-action-item-graph/peteroneil-main-design-20260520-065353.md` (APPROVED)
**Branch:** main → feat/dbos-migration (implementation creates branch)

---

## RESUMPTION CHECKLIST (NEXT AGENT READS THIS FIRST)

You are picking up where the planning sessions ended. **The architectural shape is locked.** Your job is implementation per the task ordering below.

### Step 1 — Orient (5–10 min)

1. Read `docs/plans/2026-05-20-DBOS-MIGRATION-HANDOFF.md` for the cross-session handoff (project trajectory, key context, reasoning behind locked decisions).
2. Read the rest of this plan file (you're in it). Pay attention to: locked decisions (D1–D5), the action-item workflow step graph (14 steps), the deal workflow step graph (9 steps), the Shared Neo4j Write Analysis, the Phased Migration Plan, the Implementation Tasks list, and the Risks table.
3. Read the design doc at `~/.gstack/projects/oneilstokeseqrm-action-item-graph/peteroneil-main-design-20260520-065353.md` for the deeper reasoning behind each decision (the plan captures conclusions; the design doc captures the why).
4. Read the reference codebase. The LTF DBOS module at `/Users/peteroneil/EQ-CORE/live-transcription-fastapi/services/` is the verbatim pattern reference — start with `services/dbos_runtime.py` and `services/account_provisioning/{workflow,steps,types,eventbridge_emit}.py`. **Don't modify live-transcription-fastapi; it's a separate repo with its own agents.**
5. Skim project memory: `~/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/MEMORY.md` and the linked files. Pay special attention to:
   - `project_dbos_migration_trajectory.md` (project state)
   - `pattern_dbos_step_decomposition.md` (why S9/S10 are split)
   - `pattern_data_independence_vs_commutativity.md` (why Shared Neo4j Write Analysis exists)
   - `pattern_conditional_lock_escape_valve.md` (how escape valves work)
   - `feedback_autonomous_execution.md` (Peter's preferred execution style)

### Step 2 — Surface session start

Before touching code, confirm with Peter that you're ready to begin Task #4. State explicitly:
- Which task you're starting with (T1)
- Your understanding of the next 2–3 tasks (T1 → T2 → T3)
- Anything that doesn't match your read of the plan (raise it before executing, not after)

### Step 3 — Execute Task #4 starting at T1

Task ordering is in the Implementation Tasks section below. Critical gates:

- **T1** (create branch) is a prerequisite for everything. ~5 min.
- **T3** (Day-1 Docker verification of `dbos` package Lambda compatibility) is the GO/NO-GO decision point for D1. If `dbos==2.22.0` doesn't install cleanly into the Lambda Amazon Linux 2 runtime, the conditional lock activates and you fall back to Approach A (HTTP endpoint). All later tasks change shape. Do T3 before T2.
- **T10/T11** (pipeline.py decomposition into DBOS steps) is the largest single piece of work. Plan to spread this across 1–2 sessions. Don't rush; the step boundary decisions encode the locked architecture.
- **T25a** (deterministic regression test) is BLOCKING for ship. Without it the external-contract preservation premise has no verification.

### Step 4 — Pause points

Surface to Peter at these natural session breaks:

- After T3 Docker verification — confirm GO on D1 conditional lock (or surface NO-GO and propose Approach A migration to the plan).
- After T10/T11 pipeline decomposition — confirm step boundaries match the plan's step graph; flag any deviations.
- After Phase E tests pass — confirm regression test outputs match expectations before /review.
- Before /ship — confirm PR is ready for Peter's merge approval.

### Hard constraints (DO NOT VIOLATE)

- **DO NOT** touch live-transcription-fastapi (separate repo, separate agent's work).
- **DO NOT** touch the EQ test tenant data (tenant_id `11111111-1111-4111-8111-111111111111`).
- **DO NOT** delete or redrive the DLQ message (MessageId `58863f20-3cda-48f7-973d-3002aa31331b`) until T30 (post-deploy live integration test).
- **DO NOT** change the external contracts: Neo4j node shapes, Postgres write contracts, EventBridge emissions are immutable byte-for-byte (schema-equivalent for LLM non-determinism counts within ±20%).
- **DO NOT** introduce a new vendor or new durable-execution engine. DBOS is the chosen tool. Inngest, Temporal, Step Functions, Lambda Durable Functions are all explicitly off the table.
- **DO NOT** re-litigate the locked premises or decisions (D1–D5) unless you find new evidence that contradicts them. If you find such evidence, surface it to Peter via AskUserQuestion before acting.
- **DO NOT** start implementation in the same session as `/plan-eng-review` was run. Fresh session = clean working context.
- **DO NOT** ship without /review + /codex review + Peter's explicit merge approval.

---

## Goal

Migrate the action-item-graph ingest path from a synchronous Lambda → Railway `/process` HTTP call (120s Lambda timeout failure class) to a DBOS-orchestrated durable workflow with per-step retry, checkpointing, and observability.

## Scope

| In scope | Out of scope |
|---|---|
| Lambda handler refactor (replace `submit_to_railway` with `DBOSClient.enqueue`) | Lambda Durable Functions, Inngest, Temporal, Step Functions (rejected) |
| New DBOS workflow + steps for action-item pipeline (14 steps after Codex split) | Rewrite of LLM extraction logic itself |
| New DBOS workflow + steps for deal pipeline (9 steps; inner merger refactor separates LLM construct from Neo4j MERGE) | Multi-replica Railway scaling (gated on orphan-workflow detector — future work) |
| `DBOS_SYSTEM_DATABASE_URL` provisioning on Neon (separate DB) | Sharing DBOS state DB with `live-transcription-fastapi` (rejected — separation chosen) |
| Pulumi IaC update (add secret + IAM permission + CloudWatch metric for partial-enqueue) | EventBridge target swap (not needed — Lambda is in-place modified) |
| External-contract regression test harness (deterministic + characterization split) | Operator dashboard hosting (V1: dbos.workflow_status via Neon SQL) |
| Retire `/process` HTTP endpoint, `api_client.py`, `dispatcher.py` (in Phase 3, post-rollback-window) | DLQ CloudWatch alarm (deferred — separate TODO) |
| Day-1 compatibility tests for migration transition states | New event types or new EnvelopeV1 fields |
| `Shared Neo4j Write Analysis` proving commutativity under W2 parallel writes | W1 parent workflow with internal fan-out (rejected V1; escape valve documented) |

---

## Locked Decisions — All 24 Open Questions Resolved

(Codex outside voice absorption added Open #24 — checkpoint payload size analysis for split steps.)

### Architectural locks (from /office-hours, refined by /plan-eng-review)

- **D1:** Lambda + DBOSClient (Approach B). Lambda's `submit_to_railway` replaced with `DBOSClient.enqueue` calls.
- **D2:** W2 two flat parallel workflows. **Escape valve:** revisit W1 if (a) a third pipeline is added that also needs to fan out from the same envelope, OR (b) the `partial_enqueue_pair_count` CloudWatch metric shows >5% of events hit the split window in steady-state production. Until then, W2 + monitoring is cheaper than W1 refactor.
- **D3:** 14 steps for action-item pipeline (12 original + 2 splits per Codex #10), 9 steps for deal pipeline (with inner `merger.merge_deal` function refactored to separate LLM construct from Neo4j MERGE).
- **D4:** Per-step retry policy via DBOS native `@DBOS.step(retries_allowed=True, max_attempts=N, interval_seconds=X, backoff_rate=Y)`. No jitter.
- **D5:** DBOS-default step checkpointing. LLM step purity verified for split-step boundaries (each LLM step is pure compute; each write step is pure write). No explicit `set_event_async` caching needed.

### Open question resolutions

**Open #1 — Stage 5 in-place mutation refactor.** RESOLVED. No external callers of `extraction.action_items` outside `pipeline.py`. Safe to refactor.

**Open #2 — DBOS step-failure semantics for fail-open steps.** RESOLVED. Step body try/except wraps the side-effect; step always returns success-with-warning. Mechanism: `@DBOS.step(retries_allowed=False)` for steps 4, 7, 13, 14 (action-item); steps 4, 8, 9 (deal).

**Open #3 — Workflow ID dedupe on FAILED re-delivery.** V1 silent ignore. Future enhancement.

**Open #4 — `dbos` package Lambda compatibility.** Day-1 verification task (T3) before lock.

**Open #5 — Lambda → Neon connectivity + IAM.** Public endpoint + Secrets Manager. NO VPC required.

**Open #6 — Pulumi IaC integration.** Existing pattern handles new secret automatically.

**Open #7 — Observability "did interaction X complete?"** SQL: `SELECT workflow_id, status FROM dbos.workflow_status WHERE workflow_id LIKE '%interaction-{id}%'`.

**Open #8 — Convergent MERGE redundancy.** Action-item + deal pipeline calls are COMPLEMENTARY (base merge + role enrichment). Accepted.

**Open #9 — Verifier fail-open in current code.** Verified at `verifier.py:99-107`. `retries_allowed=False` + step body try/except preserves current behavior exactly.

**Open #10 — LLM step side-effect audit + split-step purity.** Verified pure for pure-LLM steps. **Codex finding #10 absorbed:** mixed LLM+write steps (S9 merging, S10 topic_resolution) are split into pure-LLM step + pure-write step pairs. See D3 step table below.

**Open #11 — DBOS jitter native support.** Not native. V1 ships without jitter.

**Open #12 — DBOS state Postgres co-location.** SEPARATE. New Neon database `eq_aig_dbos_sys`.

**Open #13 — Phased migration + rollback latency.** RESOLVED. Three-phase model (see Phased Migration Plan below). Realistic rollback recovery time: ~12 min (Lambda redeploy ~2 min + in-flight DBOS workflow drain ~10 min).

**Open #14 — External-contract regression test definition.** Split into two tests:
- **Deterministic-mock regression test:** EXACT equality on all writes (Neo4j IDs, Postgres ON CONFLICT keys, EventBridge emissions). Uses canned LLM mock responses so output is reproducible.
- **Live-nondeterminism characterization test:** counts within ±20%, schema-exact, write-target ID sets exact-match. Non-blocking. Runs against real OpenAI; tolerance accounts for LLM variance.

**Open #15 — DLQ message replay.** Operational. Deferred to `/land-and-deploy` (T30).

**Open #16 — Deal pipeline step decomposition.** 9 steps. Inner `merger.merge_deal` function refactored to separate LLM construct (returns merged Deal object) from Neo4j MERGE (writes the constructed object). Step boundary stays single; function-level boundary becomes clean.

**Open #17 — DBOS checkpoint payload size.** RESOLVED-ANALYTICALLY for original 12 steps. **Re-analyzed under Open #24** for the split steps (S9a + S10a outputs are now checkpointed). See Open #24.

**Open #18 — Workflow ID namespace ownership.** Made redundant by Open #12 (separate DBs).

**Open #19 — DBOS workflow max-duration.** `workflow_timeout=900` (15 min) per workflow as safety net.

**Open #20 — DBOS dashboard operator access.** SQL on `dbos.workflow_status`. Hosted dashboard deferred.

**Open #21 — DBOSClient signature.** Verified at dbos==2.22.0.

**Open #22 — In-flight SQS at cutover.** Phased migration eliminates the swap-time concern (see Phased Migration Plan).

**Open #23 — Per-queue worker concurrency.** **Codex absorption:** start at `Queue("action-item-pipeline", concurrency=1)` and `Queue("deal-pipeline", concurrency=1)`. Raise after empirical criterion: 100+ successful invocations with no DB pool / Neo4j session / OpenAI rate-limit errors AND queue-depth metric trending positive. Queue-depth observability: confirm DBOS exposes per-queue backlog count in `dbos.workflow_queue` SQL view; if not, add custom CloudWatch metric polling the queue table.

**Open #24 (NEW) — Checkpoint payload size for split steps.** Splitting S9 → S9a (LLM construct) + S9b (write) means S9a's output (a constructed `MergedActionItem` object with embeddings, topics, match-results context) is now checkpointed. Worst-case estimate for content-heavy fixture (Anthropic-style ~5K-word envelope producing ~8 action items): per-item MergedActionItem ~1-2 KB plain text + ~6 KB embedding (1536 floats × 4 bytes) = ~8 KB per item × 8 items = ~64 KB per S9a checkpoint. Same magnitude for S10a (TopicResolutionResult list with embeddings). DBOS `operation_outputs` uses JSONB — 64 KB rows are fine, well under 1 MB row threshold and 2 GB column max. **No checkpoint-by-reference needed for V1.** If we later add LARGER embeddings (e.g., 3072-dim Gemini), revisit.

---

## Architecture Diagrams

### Data flow (unchanged from prior version)

```
EventBridge (existing rule, existing event_pattern)
        │
        ▼
SQS action-item-graph-queue (existing, kept as HA buffer)
        │ batch_size=1, visibility_timeout=720s
        ▼
Lambda action-item-graph-ingest (MODIFIED handler, same SQS attachment)
        │
        ├── Parse EnvelopeV1 from EventBridge wrapper
        ├── DBOSClient.enqueue(action_item_workflow, envelope_dict)
        ├── DBOSClient.enqueue(deal_workflow, envelope_dict)
        ├── Emit CloudWatch metric `partial_enqueue_pair_count` on partial success
        └── Return 200 (both enqueues confirmed)
        │
        ▼
DBOS system Postgres (NEW database on Neon, direct non-pooler connection)
        │
        ▼
Railway service (existing FastAPI + NEW DBOS workers co-deployed)
        │
        ├── Queue("action-item-pipeline", concurrency=1) ← raises after empirical criterion
        │     └── action_item_workflow → 14 steps
        │
        └── Queue("deal-pipeline", concurrency=1) ← raises after empirical criterion
              └── deal_workflow → 9 steps
        │
        ▼
Convergent idempotent writes:
  ├── Neo4j AuraDB (shared, MERGE-everywhere — see Shared Neo4j Write Analysis below)
  └── Neon Postgres application DB (shared, ON CONFLICT)
```

### Action-item workflow step graph (14 steps, post-Codex split)

```
ENTRY: enqueue(action_item_workflow, envelope_dict)
  │ @workflow workflow_id = action-item-graph:action-item:interaction-{id}
  ▼
S1   ensure_account             [Neo4j MERGE, fatal-retry × 3]
  ▼
S2   extraction                 [LLM, fatal-retry × 3]
  │   ↓ ExtractionOutput (count == 0 → early return)
S3   consolidation              [LLM, fatal-retry × 3]
  ▼
S4   verification               [LLM, retries_allowed=False, step body fail-open]
  │   ↓ ExtractionOutput (count == 0 → early return)
S5   owner_resolution           [LLM, fatal-retry × 3] — returns (extraction', contact_map)
  ▼
S6   create_interaction         [Neo4j MERGE, fatal-retry × 3]
  ▼
S7   merge_contacts_to_deal     [Neo4j MERGE side branch, retries_allowed=False, fail-open]
  │   ↓ conditional on envelope.opportunity_id and envelope.contacts
S8   matching                   [LLM + Neo4j read (PURE), fatal-retry × 3]
  ▼
S9a  merging_llm                [LLM construct merged ActionItem, fatal-retry × 3]
  ▼
S9b  merging_persist            [Neo4j MERGE write of merged ActionItem, fatal-retry × 3]
  ▼
S10a topic_resolution_llm       [LLM topic match decision, fatal-retry × 3]
  │   ↓ conditional on enable_topics
S10b topic_resolution_persist   [Neo4j MERGE write of HAS_TOPIC, fatal-retry × 3]
  ▼
S11 → renumbered S13 postgres_dual_write    [Postgres ON CONFLICT, retries_allowed=False, fail-open]
S12 → renumbered S14 agent_outbox           [Postgres write, retries_allowed=False, fail-open]
  ▼
EXIT: PipelineResult
```

Renumbered final step IDs: S1, S2, S3, S4, S5, S6, S7, S8, S9a, S9b, S10a, S10b, S13, S14. (Note: S11, S12 IDs are not used to avoid confusion with the prior 12-step numbering during transition.)

### Deal workflow step graph (9 steps, with inner merger refactor)

```
ENTRY: enqueue(deal_workflow, envelope_dict)
  │ @workflow workflow_id = action-item-graph:deal:interaction-{id}
  ▼
D1  validate_envelope            [pure, fatal if account_id missing]
  ▼
D2  verify_account               [Neo4j read, fatal-retry × 3]
  ▼
D3  ensure_interaction           [Neo4j MERGE, fatal-retry × 3]
  │   ↓ conditional on interaction_id
D4  merge_contacts_to_deal_base  [Neo4j MERGE, retries_allowed=False, fail-open]
  │   ↓ conditional on opportunity_id and envelope.contacts (Case A)
D5  fetch_existing_deal          [Neo4j read, fatal-retry × 3]
  │   ↓ conditional on opportunity_id; falls through to discovery if absent
D6  extraction                   [LLM, fatal-retry × 3]
  │   ↓ no deals extracted → enrich_interaction (D8) then return
D7  match_merge_loop             [Per-deal loop, retries_allowed=False, fail-open per deal]
  │   Inner refactor: merger.merge_deal() is now split into:
  │     - construct_merged_deal_llm(extracted, candidate, context) → MergedDeal (pure LLM)
  │     - persist_merged_deal_neo4j(merged_deal) → MergeResult (pure write)
  │   Single DBOS step boundary preserved because per-deal failures are non-fatal
  ▼
D8  enrich_interaction           [Neo4j MERGE, retries_allowed=False, fail-open]
  ▼
D9  postgres_dual_write          [Postgres ON CONFLICT, retries_allowed=False, fail-open]
  ▼
EXIT: DealPipelineResult
```

---

## Shared Neo4j Write Analysis (Codex #5/6/7 absorption)

Reframing the "data-independent" claim from /office-hours: **the two pipelines have no required ordering for business output**, but they DO write to shared Neo4j entities for the same interaction. Commutativity under concurrent MERGE writes must be proven, not assumed.

### Enumeration of shared Neo4j writes per interaction

| Entity / Relationship | Action-item writes | Deal writes | Commutativity argument |
|---|---|---|---|
| `Account` node | MERGE on (tenant_id, account_id), sets name if absent (COALESCE) | Read-only (`verify_account`) | Commutative — single writer (action-item only); deal pipeline reads after MERGE completes via existing dispatcher ordering OR via DBOS workflow start ordering. Race: if deal starts before action-item MERGE completes, `verify_account` could raise. **Mitigation:** D2 `verify_account` already handles missing-Account as a warning (existing code). Behavior change risk: low. |
| `Interaction` node | MERGE on (tenant_id, interaction_id) with COALESCE-pattern property sets (`user_id`, `title`, `duration_seconds`, etc.) | MERGE on same key with similar COALESCE pattern | **Commutative under COALESCE.** The COALESCE pattern was specifically introduced in the contact enrichment migration (2026-03-16) to survive race conditions. Property writes only update when current value is NULL. Net result: union of all set fields, regardless of pipeline order. |
| `Contact` nodes | Not written directly (read via envelope.contacts) | Not written directly | No conflict. |
| `(Contact)-[:ENGAGED_ON]->(Deal)` base relationship | MERGE on (contact_id, deal_id) | MERGE on same key | **Commutative.** Both pipelines call shared `merge_contacts_to_deal()`. MERGE is idempotent on the (contact_id, deal_id) pair. No property writes from action-item; deal pipeline adds role properties on top. Order: base merge first (either pipeline) → role enrichment (deal pipeline). If deal completes first, base already exists. If action-item completes first, base exists when deal arrives. **Commutative.** |
| `(Contact)-[:ENGAGED_ON]->(Deal)` role properties | Not written | SET `role`, `champion_confidence` etc. via separate enrichment query | Single writer (deal pipeline). Not a race. |
| `Deal` node | Not written | MERGE on (tenant_id, opportunity_id) with field updates | Single writer (deal pipeline). |
| `ActionItem` node | MERGE on (tenant_id, action_item_id) with field updates | Not written | Single writer (action-item pipeline). |
| `ActionItemVersion` node | CREATE per version | Not written | Single writer (action-item pipeline). |
| `ActionItemTopic` node | MERGE on (tenant_id, action_item_topic_id) | Not written | Single writer (action-item pipeline). |

### Conclusion

**All shared writes are commutative** under the existing MERGE + COALESCE pattern. The W2 parallel-pipeline architecture is safe for the current envelope contract. **No data-integrity changes required.**

### What would break this

- **A third pipeline writing to Account, Interaction, or ENGAGED_ON without COALESCE pattern.** Add to the W1 escape valve criteria.
- **Schema change that introduces non-COALESCE property writes from either pipeline.** Codify "COALESCE-only property writes for shared entities" as a contributor convention in CLAUDE.md.

### Tests to lock this

- **Concurrent write test:** in `tests/test_concurrent_pipelines.py`, simulate both pipelines processing the same envelope simultaneously (via `asyncio.gather` over the two workflow function calls). Assert final Neo4j state matches the union of expected writes regardless of completion order.
- **Property COALESCE preservation:** existing integration test `test_integration_e2e.py` already covers Interaction property COALESCE. Add a similar test for ENGAGED_ON.

---

## Phased Migration Plan (Codex #1/3/4 absorption)

Replaces the prior "Cutover Plan" section. Three explicit phases with clear gates.

### Phase 1 — DBOS infrastructure deploy (Day 0, ~1 hr)

1. Provision new Neon DB `eq_aig_dbos_sys` (direct non-pooler connection string).
2. Pulumi `up` deploys: new Secrets Manager entry, IAM permission expansion, no Lambda changes yet.
3. Deploy new Railway code with DBOS workers + DBOS lifespan integration. **`/process` route REMAINS ALIVE** in this deploy — it's the compatibility shim for in-flight messages.
4. Verify DBOS workers are healthy: `SELECT * FROM dbos.workflow_status` returns connection success; admin server disabled.
5. **Readiness gate:** synthetic envelope test — manually enqueue a test workflow via `DBOSClient.enqueue` from a local script, watch it complete via Neon SQL. Required before Phase 2.

### Phase 2 — Lambda traffic shift (Day 0–1, ~30 min)

6. Deploy new Lambda code (Pulumi `up` with new Lambda zip). Lambda now enqueues to DBOS instead of HTTP-POSTing to `/process`.
7. **In-flight messages at deploy moment:** old Lambda invocations already in progress complete the OLD path (HTTP POST to `/process`, which is still alive per Phase 1 step 3). **No message loss.**
8. Monitor first ~10 envelopes for end-to-end correctness:
   - DBOS dashboard / Neon SQL shows two workflow rows per interaction (one action-item, one deal)
   - `partial_enqueue_pair_count` CloudWatch metric = 0 (alarm if >0)
   - Both workflows reach completion within p99 ~5 min
9. **Compatibility test gate (T32-T34 below):** confirm tests cover "new Lambda + old Railway with /process alive" transition state.

### Phase 3 — Endpoint removal (Day 14+, ~30 min)

10. After 2-week rollback window with no production incidents, deploy a follow-up Railway change that removes `/process` route + deletes `api_client.py`.
11. Verify no traffic to `/process` in CloudWatch / Railway logs for the prior 7 days before deletion.
12. Delete `legacy/pre-dbos-migration` branch.

### Rollback procedure (post-Phase 2, before Phase 3)

If a subtle bug surfaces in DBOS path:

1. **Pause SQS event source mapping** (~30s via AWS console or Pulumi flag) to stop new envelopes entering Lambda.
2. **Redeploy old Lambda code** (Pulumi `up` with previous Lambda zip OR direct AWS Lambda console upload of previous version). ~2 min.
3. **Resume SQS event source mapping** — Lambda now uses old `submit_to_railway` path; `/process` is still alive per Phase 1, so messages flow normally.
4. **In-flight DBOS workflows continue running** on Railway. **Default rollback policy: let them complete naturally.** They're idempotent at the Neo4j MERGE / Postgres ON CONFLICT layer; new envelopes won't trigger new workflows (Lambda now goes to old path), so the queue drains as in-flight workflows finish.
5. **Emergency rollback policy** (only if in-flight workflows have a confirmed data-corruption bug): operator runs `UPDATE dbos.workflow_status SET status = 'CANCELLED' WHERE status IN ('PENDING', 'RUNNING')` against the DBOS system DB. This is a manual operator action with explicit incident commander approval.
6. **Realistic end-to-end recovery time:** ~12 minutes (Lambda redeploy ~2 min + in-flight workflow drain ~10 min). Satisfies "credible 5-minute rollback for new-traffic redirection" since step 2 alone (~2 min) stops the bleed. In-flight reconciliation is a tail-recovery concern.

---

## Implementation Tasks (Ordered, with Codex absorption updates)

### Phase A — Foundation (1 session, ~2h)

- [ ] **T1 (P1)** — Create branch `feat/dbos-migration` from `main`.
- [ ] **T2 (P1)** — Add `dbos>=2.22.0,<3.0` to `pyproject.toml` + `requirements.txt`. Run `uv lock`. Verify install.
- [ ] **T3 (P1)** — Day-1 verification: build Lambda zip via Docker with `public.ecr.aws/lambda/python:3.11` base. Verify dbos + psycopg[binary] install cleanly and total zip size < 50 MB. If fails: fall back to Approach A per conditional lock.
- [ ] **T4 (P1)** — Provision new Neon database `eq_aig_dbos_sys`. Capture direct (non-pooler) connection string.
- [ ] **T5 (P1)** — Update `infra/__main__.py` to add `dbos-system-database-url` to the `secrets` dict.
- [ ] **T6 (P1)** — Create `src/action_item_graph/dbos_runtime.py` adapting LTF's pattern.
- [ ] **T7 (P1)** — Wire `dbos_lifespan` into FastAPI's lifespan context manager.

### Phase B — Workflow + Steps scaffolding (1.5 sessions, ~3h, +30 min for step splits)

- [ ] **T8 (P1)** — Create `src/action_item_graph/workflows/{__init__,action_item_workflow,action_item_steps,queues}.py`.
- [ ] **T9 (P1)** — Define `Queue("action-item-pipeline", concurrency=1)` and `Queue("deal-pipeline", concurrency=1)` (Codex absorption: lower from 3). Document raise-criterion + queue-depth observability via `dbos.workflow_queue` SQL view.
- [ ] **T10 (P1)** — Decompose `pipeline.py:process_envelope` into 14 steps (with S9/S10 split per Codex #10). Stage 5 returns new `extraction'` instead of mutating in place. CRITICAL: S9a/S10a return values are checkpointed — verify worst-case payload size matches Open #24 analysis during implementation.
- [ ] **T11 (P1)** — Mirror for deal pipeline: 9 steps + inner `merger.merge_deal` refactor (split LLM construct from Neo4j write at the function level, keep single DBOS step boundary).
- [ ] **T12 (P1)** — Apply `@DBOS.step(...)` decorators per the retry policy table in D4.

### Phase C — Lambda dispatcher (1 session, ~1.5h, +30 min for partial-enqueue metric)

- [ ] **T13 (P1)** — Modify `lambda_ingest/handler.py`: two `DBOSClient.enqueue` calls + `SetWorkflowID` + `workflow_timeout=900`. Emit CloudWatch metric `partial_enqueue_pair_count` on first-succeeds-second-fails case.
- [ ] **T14 (P1)** — Modify `lambda_ingest/config.py`: add `DBOS_SYSTEM_DATABASE_URL`. Remove `API_BASE_URL`, `WORKER_API_KEY`, `HTTP_TIMEOUT_SECONDS`, `MAX_RETRIES`.
- [ ] **T15 (P1)** — Modify `lambda_ingest/secrets.py`: replace `get_worker_api_key` with `get_dbos_system_database_url`.
- [ ] **T16 (P1)** — Delete `lambda_ingest/api_client.py` (and its tests in T21).
- [ ] **T17 (P1)** — Update Pulumi `lambda_env_vars`.

### Phase D — Retire `/process` HTTP endpoint (Day 14+, NOT in Phase 2 deploy)

- [ ] **T18 (P2)** — Delete `api/routes/process.py` AFTER 2-week rollback window passes.
- [ ] **T19 (P2)** — Delete `dispatcher/dispatcher.py`.
- [ ] **T20 (P2)** — Remove `/process` route registration from `api/main.py`.

### Phase E — Tests (2 sessions, ~4h, +1h for compatibility + concurrent-write + regression-split tests)

- [ ] **T21 (P1)** — Update existing Lambda tests.
- [ ] **T22 (P1)** — Per-step unit tests for action_item_workflow (14 steps).
- [ ] **T23 (P1)** — Per-step unit tests for deal_workflow (9 steps).
- [ ] **T24 (P2)** — Crash-recovery integration test (`RUN_DBOS_E2E=1` gated).
- [ ] **T25a (P1)** — **Deterministic-mock regression test** (`tests/test_regression_deterministic.py`): canned LLM mock responses, exact equality assertions on Neo4j IDs + Postgres ON CONFLICT keys + EventBridge emissions.
- [ ] **T25b (P2)** — **Live-nondeterminism characterization test** (`tests/test_regression_live.py`): real OpenAI, ±20% count tolerance, schema-exact, non-blocking. Runs in CI only on-demand.
- [ ] **T26 (P2)** — Update `scripts/run_live_e2e.py` to use DBOSClient.enqueue.
- [ ] **T32 (P1, NEW)** — **Compatibility test: old Lambda + new Railway with `/process` alive.** Simulates Phase 1 mid-state where Railway has DBOS workers but Lambda still uses HTTP. Asserts `/process` route still works for old Lambda invocations.
- [ ] **T33 (P1, NEW)** — **Compatibility test: new Lambda + old Railway (`/process` removed).** Simulates an incorrect deploy order. Asserts Lambda fails fast with a clear error (DBOS workers not ready) rather than silently dropping messages.
- [ ] **T34 (P1, NEW)** — **Concurrent-write test** (`tests/test_concurrent_pipelines.py`): both pipelines process same envelope via `asyncio.gather`. Assert final Neo4j state matches union of expected writes regardless of completion order. Validates Shared Neo4j Write Analysis commutativity claims.
- [ ] **T35 (P2, NEW)** — Rollback-with-inflight test: simulate "redeploy old Lambda while DBOS workflow is mid-step." Assert in-flight workflow completes naturally; new envelopes route via old path; no double-processing.

### Phase F — Deploy + verify (0.5 session in /land-and-deploy)

- [ ] **T27 (P1)** — Codex consult + /review + /codex review per gstack workflow.
- [ ] **T28 (P1)** — /ship creates PR for Peter's merge approval.
- [ ] **T29 (P1)** — After merge: /land-and-deploy verifies Railway + Lambda deploy through Phase 1 + Phase 2.
- [ ] **T30 (P1)** — Redrive DLQ message via `aws sqs start-message-move-task`.
- [ ] **T31 (P2)** — Phase 3 (Day 14+): delete `/process` route + retire legacy code.

**Total revised estimate:** ~20–25 hrs of implementation CC time (up from ~16–22 hrs pre-Codex). Codex absorption added ~3 hrs (step split + compatibility tests + concurrent-write test + regression test split + partial-enqueue metric).

---

## Testing Strategy

### Unit tests per step

Each `@DBOS.step` function tested with mocked dependencies. Verify happy path, empty input, conditional path, fail-open semantics, SQL idempotency.

### Integration tests (skipped by default, `RUN_DBOS_E2E=1` gated)

Workflow end-to-end with real Neo4j + Postgres + mocked-deterministic LLM. Crash recovery via container kill. Re-delivery dedupe via duplicate enqueue. Concurrency via simultaneous enqueue.

### Regression tests — SPLIT per Codex #9 absorption

**T25a — Deterministic-mock regression** (PRIMARY, BLOCKING):
- Pytest fixture with CANNED LLM responses (same exact output every run).
- Runs same envelope through old path (mocked dispatcher) and new path (DBOS workflows with same mocks).
- Asserts: EXACT Neo4j write-target ID match, EXACT Postgres ON CONFLICT key match, EXACT EventBridge emission match.
- No tolerance — this is the contract.

**T25b — Live-nondeterminism characterization** (CHARACTERIZATION, NON-BLOCKING):
- Real OpenAI calls (5–10 envelopes from Lightbox fixture).
- Asserts: schema-exact match, write-target ID sets exact-match, LLM-driven counts within ±20%.
- Acknowledges LLM non-determinism; not a contract regression, just a sanity check.

### Compatibility tests — NEW per Codex #20 absorption

T32, T33, T34, T35 cover the migration transition states explicitly.

---

## NOT in scope (deferred)

- DLQ CloudWatch alarm (separate TODO).
- Multi-replica Railway scaling.
- DBOS hosted dashboard (app.dbos.dev).
- EventBridge API Destinations alternative.
- W1 parent-with-child workflow refactor (escape valve documented in D2).
- Retry-attempt-id workflow ID suffix.

## What already exists (reuse anchors)

- LTF `services/dbos_runtime.py` (verbatim adapt)
- LTF `services/account_provisioning/{workflow,steps,types}.py` (pattern reference)
- LTF `tests/integration/account_provisioning/test_crash_recovery.py` (test pattern)
- Existing `pipeline.py` (refactor target)
- Existing `deal_graph/pipeline/pipeline.py` (refactor target)
- Existing `infra/forwarder.py` (Pulumi pattern — add new secret entry only)
- Existing `postgres_client.py` (Neon connection handling, kept as-is)
- Existing COALESCE property write pattern in Neo4j MERGE queries (preserves Shared Neo4j Write Analysis commutativity)

## Worktree parallelization

Phase E test development naturally parallelizes:
- Lane 1: T22 (action-item workflow tests)
- Lane 2: T23 (deal workflow tests)
- Lane 3: T25a + T25b (regression tests)
- Lane 4: T32 + T33 + T34 + T35 (compatibility + concurrent-write + rollback tests)

All four can run in parallel after Phases A, B, C are merged. ~1-1.5 hr each.

## Risks + Mitigations (updated post-Codex)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `dbos` package fails to install in Lambda runtime | Low | High | T3 verification before any production deploy |
| Concurrent Neo4j writes from W2 pipelines diverge | Low | Medium | Shared Neo4j Write Analysis + T34 concurrent-write test |
| LLM step retry produces divergent merged ActionItem fields | Low (now Low after split) | Medium | S9/S10 split (T10) + T25a deterministic regression test |
| S9a/S10a checkpoint payloads exceed DBOS row size | Low | Medium | Open #24 analysis (~64 KB worst case, well under limits) + measure during T22 |
| Partial-enqueue split-brain at Lambda crashes mid-pair | Low | Low | CloudWatch metric + alarm + W1 escape valve if metric exceeds 5% threshold |
| Rollback leaves in-flight DBOS workflows in inconsistent state | Low | Low (idempotent writes) | Default policy: let complete naturally + emergency manual SQL CANCELLED |
| Concurrency=1 too low for steady-state throughput | Medium | Low (slower processing, no data loss) | Empirical raise-criterion + queue-depth observability |
| External-contract regression test misses an edge case | Medium | Medium | Split deterministic + live characterization tests + fuzz-pass on edge envelopes |
| Workflow ID format invalid per DBOS validator | Very low | High | T21 explicitly tests workflow_id format against installed dbos==2.22.0 |
| /process route prematurely deleted during Phase 2 | Low | High (in-flight message loss) | Phase 3 explicitly Day-14+; T33 compatibility test catches deploy-order mistakes |

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | — | not run (optional; engineering migration with no product-strategy implications) |
| Codex Review (Outside Voice) | `/codex review` plan-stage | Independent adversarial 2nd opinion on plan | 1 | CLEAR | 20 findings; 19 absorbed (most via plan updates), 1 rejected with escape-valve (W1 reconsideration). Quality score implied 7→9 post-absorption. |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 24 open questions resolved (added Open #24 via Codex absorption), 14+9 step decompositions locked, external-contract regression test split, Shared Neo4j Write Analysis added, phased migration explicit, concurrency=1 default with raise-criterion. |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | not applicable (internal infra refactor) |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | not applicable |

**CODEX:** 20 findings, 19 absorbed (step splits, concurrent-write analysis, phased migration, concurrency=1, rollback realism, regression test split, compatibility tests, partial-enqueue metric, checkpoint size analysis), 1 rejected (W1 reconsideration — W2 + monitoring is cheaper; escape valve at >5% partial_enqueue metric or third pipeline addition).

**CROSS-MODEL:** Strong agreement on most findings. Single divergence (W1 vs W2) resolved with escape-valve framing rather than re-litigation.

**UNRESOLVED:** 0
**VERDICT:** ENG + CODEX CLEARED — ready for implementation per T1–T35 task ordering above. Implementation starts with T1 (create feat/dbos-migration branch) and T3 (Day-1 dbos Lambda Docker container build verification) before any production-affecting work.
