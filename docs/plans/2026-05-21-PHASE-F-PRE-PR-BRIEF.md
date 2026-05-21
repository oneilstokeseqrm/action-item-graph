# Phase F Pre-PR Brief — DBOS Migration

**Status:** SKELETON — fill findings sections during /review + /codex review absorption, then surface to Peter at /ship gate.

**Created:** 2026-05-21 (Phase E handoff session)
**Branch:** `feat/dbos-migration` (10 commits ahead of `main` (9 substantive + 1 handoff-prep docs at HEAD) at handoff time)
**Last commit on branch at handoff:** `75ca5f2` (codex absorption: test rename)

This artifact is the structured pre-PR brief Peter expects to see at the /ship gate. It is pre-populated with the parts that can be filled before /review + /codex (the commit log, deletion-seam list, known-deferred items, test count delta, draft PR title + body skeleton). The next session folds findings from /review + each codex round into the **Findings absorbed** sections.

---

## 9-commit log with one-line summaries

In chronological order on `feat/dbos-migration` (oldest → newest). 9 commits total: 4 phase commits (A, B-1, B-2, C) + 2 fix commits (opp_id parity + codex-absorption rename) + 3 Phase E commits (E1, E2, E3).

| # | SHA | One-line summary |
|---|-----|-------------------|
| 1 | `1bfd42d` | **Phase A** — Foundation: DBOS v2.22.0 + psycopg[binary] added, Neon `eq_aig_dbos_sys` provisioned, Pulumi secret wired, `dbos_runtime.py` + FastAPI lifespan integration. 3 codex rounds absorbed. |
| 2 | `6e51059` | **Phase B-1** — Action-item DBOS workflow: 14-step decomposition (S1-S14 with S9/S10 splits per Codex #10), workflow body + step functions + serialization helpers + client registry. 3 codex rounds absorbed (Rules 1-5 earned). |
| 3 | `6e11307` | **Phase B-2** — Deal DBOS workflow: 9-step decomposition + D7 inner merger refactor + repository idempotency fix (create_version_snapshot from CREATE-randomUUID to MERGE-on-deterministic-UUID5; same latent bug retroactively fixed in action-item repo). 4 codex rounds absorbed (Rule 6 earned). Open #3 V1 stance locked. |
| 4 | `3ca3b1e` | **Phase C** — Lambda dispatcher cutover: `handler.py` rewritten as DBOSClient.enqueue dispatcher with CloudWatch partial-enqueue metric, config/secrets swapped from HTTP to DBOS_SYSTEM_DATABASE_URL, `api_client.py` deleted, Pulumi env vars cleaned up, `package_lambda.sh` adds dbos+psycopg. 1 codex round (no blockers). Rule 7 earned. |
| 5 | `00c849e` | **fix(dbos)** — `deal_workflow` reads `opportunity_id` from `extras` (parity with legacy `EnvelopeV1.opportunity_id` @property). Bug surfaced by Phase E T23 test writing; Case A targeted-deal flows would have silently fallen through to discovery mode under Phase B-2 code path. Rule 8 codified. |
| 6 | `8711106` | **Phase E1** — Workflow + Lambda test coverage (T21+T22+T23): 83 new tests across 6 files. T21 rewrites 3 previously-broken-at-collection Lambda test files + adds `[lambda]` extras to pyproject.toml. T22+T23 cover workflow body validation gates, authoritative interaction_id (Rule 5), per-step fail-open contracts. Foundation tests (runtime + serialization round-trips) pin Rule 3. |
| 7 | `5b136a6` | **Phase E2** — Workflow behavioral contract tests (T25a, scoped down): 5 contract tests on workflow path. Original side-by-side parity plan scoped down after two mock-strategy attempts hit different layers of `process_envelope`'s internal complexity. Scope-decision rationale documented in HANDOFF.md. `feedback_test_seam_signals.md` codifies the lesson. |
| 8 | `c042207` | **Phase E3** — Compatibility + concurrent-write tests (T32+T33+T34): 7 tests. T32 covers /process route behavior in Phase 1 mid-state (rollback safety net). T33 covers DBOS-unreachable Phase 2 deploy-order mistake. T34 covers concurrent workflow execution. |
| 9 | `75ca5f2` | **fix(dbos)** — rename Phase E2 contract test to match its actual assertion (codex-absorption commit). Phase E /codex review surfaced one non-blocking observation: docstring claimed S10b reach, assertion was S9b. Renamed `test_workflow_executes_step_chain_through_topic_creation` → `test_workflow_reaches_merge_persist_phase`. |

---

## Test count delta

- **Baseline (main):** 554 tests
- **Phase E adds:** 95 tests (T21: 24, T22: 18, T23: 18, T25a: 5, T32: 3, T33: 2, T34: 2, foundation runtime: 5, foundation serialization: 18)
- **Phase F branch:** 649 tests passing

**Confirm at /ship time** with `uv sync --extra lambda --extra dev --extra api && uv run pytest -q` — the count should hold at 649 (modulo any tests added during /codex absorption).

---

## Findings absorbed across /review + all /codex rounds

### From /review skill

**HIGH absorbed:** [FILL IN]

**MEDIUM absorbed:** [FILL IN]

**NIT addressed:** [FILL IN]

**Deliberate non-action (with reasoning):** [FILL IN]

### From /codex review full-PR rounds

#### Round 1
- **HIGH absorbed:** [FILL IN]
- **MEDIUM absorbed:** [FILL IN]
- **NIT addressed:** [FILL IN]
- **Deliberate non-action (with reasoning):** [FILL IN]
- **Verdict:** [pending / no ship-blocking found]

#### Round 2 (if needed)
[FILL IN — same structure as Round 1]

#### Round 3 (if needed)
[FILL IN]

#### Round 4 (if needed; Phase B-2 4-round precedent is the upper bound — see Step 4 of Phase F resumption guide)
[FILL IN]

#### Round 5 (if needed; this round = STOP signal per the iteration ceiling rule)
[FILL IN]

### Phase E /codex review outcome (already absorbed, FYI for full-PR context)

Run on `3ca3b1e..HEAD` with constrained scope (medium reasoning, no exploration, diff-only). Verdict: **"No high-severity ship blockers found."**

- **Non-blocking observation #1**: T25a test #5 had a docstring/assertion mismatch (claimed S10b, asserted S9b). **Absorbed** via commit `75ca5f2` (rename + docstring fix).
- **Non-blocking observation #2**: T34 is a smoke test, not commutativity proof. **Documented** in the test's docstring; commutativity is verified post-deploy by DLQ replay. No code change.
- **Specific concerns verified clean** by codex:
  - `deal_workflow.py:63` opp_id fix is correct equivalence to `EnvelopeV1.opportunity_id` @property (extras=None invalid at model-validation time)
  - T33 using `ConnectionError` instead of `sqlalchemy.OperationalError` is fine (Lambda surfaces both as BatchItemFailure)
  - Pyright info-level fixture warnings are pytest convention
  - Lambda test rewrites are faithful to the new architecture

---

## Draft PR title (<70 chars)

**Recommended:**

```
feat(dbos): migrate ingest path to DBOS durable workflows (Phases A→E)
```

(67 chars including the conventional-commit prefix.)

**Alternative options:**

- `feat(dbos): replace 120s Lambda timeout failure class with DBOS workflows` (73 chars — over limit)
- `feat(dbos): DBOS workflow migration — Phases A through E` (56 chars — terser)

---

## Draft PR body

```markdown
## Summary

Migrates the action-item-graph ingest path from a synchronous Lambda → Railway `/process` HTTP call (120s Lambda timeout failure class) to a DBOS-orchestrated durable workflow with per-step retry, checkpointing, and observability.

**What this PR contains** (9 commits, 9 phases of work):

- **Phase A** (`1bfd42d`) — DBOS runtime substrate + Neon `eq_aig_dbos_sys` state DB.
- **Phase B-1** (`6e51059`) — Action-item DBOS workflow: 14 steps mapping 1:1 to the legacy pipeline stages with S9/S10 LLM-vs-write splits per the locked step decomposition.
- **Phase B-2** (`6e11307`) — Deal DBOS workflow: 9 steps + D7 inner merger refactor + repository-layer idempotency fix (version-snapshot CREATE-randomUUID → MERGE on deterministic UUID5; same latent bug retroactively fixed in action-item repo).
- **Phase C** (`3ca3b1e`) — Lambda dispatcher cutover: `submit_to_railway` HTTP forwarder replaced with two `DBOSClient.enqueue` calls (one per pipeline workflow). CloudWatch `partial_enqueue_pair_count` metric on first-succeeds-second-fails split window.
- **Phase E1+E2+E3** (`8711106`, `5b136a6`, `c042207`) — 95 new tests across 8 files: Lambda dispatcher tests (T21), action-item workflow tests (T22), deal workflow tests (T23), behavioral contract tests (T25a scoped down — see HANDOFF.md § "T25a scope decision" for rationale), compatibility tests (T32+T33), concurrent-execution tests (T34).
- **Two fix commits** (`00c849e`, `75ca5f2`) — A parity bug in Phase B-2's `deal_workflow.py` (read `opportunity_id` from top-level dict instead of `extras` per `EnvelopeV1.opportunity_id` @property; Case A targeted-deal flows would have silently fallen through to discovery) discovered by Phase E T23 test writing; and a test rename absorbing Phase E /codex review's one non-blocking finding.

**Test count delta:** 554 → 649 (95 new tests). All 649 passing with `uv sync --extra lambda --extra dev --extra api && uv run pytest`.

**Codex absorption:** 12 rounds across phases (3 in A, 3 in B-1, 4 in B-2, 1 in C, 1 in E). Each phase reached an explicit "no ship-blocking findings" stop signal before commit. The full-PR /codex review (this PR's gate) absorbed [N] additional rounds — see PR review thread.

**Patterns earned and codified** in `~/.claude/projects/.../memory/`:
- Rules 1–6 in B-1+B-2 (validation in workflow body, LLM-prompt source-of-truth, JSON-serialization discipline, zip parallel-array length assertion, downstream-produced identifier authority, repository-method idempotency audit)
- Rule 7 in C (cross-service reference call shape must match deployment topology)
- Rule 8 in E (Pydantic @property reads through model_dump must replicate dict-key path)
- Feedback memory `feedback_test_seam_signals.md` from E (two mock strategies fighting different layers = codebase signaling natural test seams)

## What this PR does NOT contain

Intentionally deferred per the locked 3-phase migration plan and Phase E's P2/P3 prioritization:

- **Phase D (T18-T20)** — retire `/process` HTTP route + `dispatcher.py` + delete legacy code. Deferred to a follow-up PR Day 14+ post-Phase 2 deploy. The 2-week rollback window depends on `/process` staying alive during Phase 1+2. Phase D PR will also delete the migration-window-only test files (see "Deletion seam list" below).
- **T24** — Crash-recovery integration test (`RUN_DBOS_E2E=1` gated). P2 in the plan; deferred to post-deploy verification.
- **T25b** — Live-LLM characterization test. P2 sanity check (not a contract regression); deferred.
- **T26** — `scripts/run_live_e2e.py` update to use DBOSClient.enqueue. P2; only relevant when the DBOS state DB is provisioned at T29 deploy time.
- **T35** — Rollback-with-inflight test. P2; covered structurally by the rollback procedure documented in the execution plan.

## Test plan

- [ ] CI runs the full pytest suite: 649 tests passing
- [ ] `uv run pyright` clean on all touched modules (except pre-existing missing-import warnings for boto3 + aws_lambda_powertools — only bundled into the Lambda zip by `scripts/package_lambda.sh`, NOT in pyproject.toml; documented in HANDOFF.md § "Known-deferred test failures")
- [ ] `/review` skill cleared
- [ ] `/codex review` full-PR cleared with explicit "no ship-blocking findings"
- [ ] Peter manually reviews the diff and approves merge
- [ ] After merge: `/land-and-deploy` runs Phase 1 (DBOS infrastructure deploy) + Phase 2 (Lambda traffic shift) per the execution plan's Phased Migration Plan
- [ ] T29 deploy gate: Peter sets `pulumi config set --secret dbos-system-database-url <DIRECT_URL>` and the Railway `DBOS_SYSTEM_DATABASE_URL` env var (both manual; secret value not echoed in conversation per `feedback_secret_handoff_pattern.md`)
- [ ] T30: redrive the parked DLQ message (`58863f20-3cda-48f7-973d-3002aa31331b`) via `aws sqs start-message-move-task` — first live integration test of the new path

## What this enables

- **120s Lambda timeout failure class structurally eliminated.** The dispatcher returns 200 in sub-second time after two `DBOSClient.enqueue` calls; pipeline duration is no longer bounded by Lambda's timeout. Content-heavy emails (the originating Anthropic security-questionnaire incident) that exhausted the timeout now succeed.
- **Per-step retry with deterministic replay.** Each LLM call, Neo4j MERGE, and Postgres write is a checkpointed @DBOS.step with explicit retry policy. Transient OpenAI/Neon errors no longer cascade into full-pipeline failures.
- **DBOS dashboard replaces the DLQ as the operator surface.** Failed workflows are visible in `dbos.workflow_status` with full input/output, replay button, and per-step granularity.
- **Per-stage observability.** Each step's latency, retry count, and outcome is queryable independently via DBOS state DB SQL.
- **No upper bound on pipeline duration** within DBOS's workflow-duration limits (15-minute safety net per Open #19 / `workflow_timeout=900`).
- **Foundation for Phase D removal of `/process`** — once the 2-week rollback window passes without incidents, the legacy HTTP path retires.

## Deletion seam list (Phase D follow-up PR)

The following files/code are migration-window-only and will be deleted in the same PR that retires `/process` from production traffic (Phase D, Day 14+ post-deploy):

- `src/action_item_graph/api/routes/process.py` — the `/process` HTTP route handler
- `src/dispatcher/dispatcher.py` — the synchronous EnvelopeDispatcher
- `infra/__main__.py` — the deprecated `secret_arn` back-compat export block (lines 75-82) and `worker_api_key = config.require_secret("worker-api-key")` if added back
- `tests/test_compatibility_process_route.py` — T32 tests, only relevant during the Phase 1 mid-state window
- `tests/test_workflow_behavioral_contract.py` — T25a behavioral contracts, scoped to the migration window per the test docstring's deletion-seam note
- `tests/test_lambda_handler.py::TestT33DBOSUnreachable` — T33 tests, deploy-order safety net relevant only during Phase 2
- Pulumi.prod.yaml `worker-api-key` config entry (kept for back-compat during the window; remove in Phase D)
- Pulumi.prod.yaml `api-base-url` config entry (same reason)

`tests/test_concurrent_pipelines.py` (T34) STAYS alive past Phase D — concurrent W2 execution is the steady-state production behavior.

---

## Cross-phase architectural cohesion (Peter's watchpoint #1)

**Does Phase A's DBOS runtime + Phase B's step decomposition + Phase C's Lambda dispatcher + Phase E's tests hang together as one coherent migration?**

[FILL IN during /review absorption — but as a starting note]:

The phases form a clean dependency chain:
- **A → B**: A provides the DBOS runtime that B's @DBOS.workflow + @DBOS.step decorators register against. B's step decomposition is sized to the per-step retry/checkpoint policy A's runtime enables.
- **B → C**: B's `action_item_workflow` + `deal_workflow` function names are what C's Lambda dispatcher targets via `EnqueueOptions["workflow_name"]`. The dispatcher invokes the workflows; the workflows execute on Railway via A's DBOS workers.
- **C → E**: C's dispatcher schema (DBOS_SYSTEM_DATABASE_URL config, get_dbos_system_database_url secrets accessor, DBOSClient enqueue pattern) is what E1's T21 Lambda tests cover. C's two enqueue calls (one per pipeline) is what E3's T34 concurrent-write test exercises.
- **A→E together**: HANDOFF.md § "Phase A + B + C + E execution log" enumerates the commit-by-commit narrative; no phase-boundary seam between B and C (the workflow names are the contract) or between C and E (the test files are explicitly named per task).

[Phase F review may surface architectural concerns that the per-phase reviews couldn't see — fold those findings here.]

## Documentation completeness (Peter's watchpoint #2)

**Does HANDOFF.md describe the full migration arc such that someone reading it cold can understand what shipped and why?**

[FILL IN during /review absorption]:

HANDOFF.md sections (as of handoff time):
- TL;DR with 9-commit log
- Project trajectory (Sessions 1–4 narrative)
- Locked decisions consolidated (D1–D5 + Open #3 + state DB + cutover + workflow ID)
- Reasoning that doesn't survive only in the plan (6 "why" sections)
- Key artifacts (paths + purposes table)
- Picking up from Phase E complete — Phase F /review + /codex + /ship guide (8 steps)
- Picking up from Phase B complete — Phase C resumption guide (HISTORICAL — kept for trajectory)
- Phase A + B + C + E execution log (per-phase commit-level narratives)
- Phase E test-coverage backlog (largely shipped; Phase F /review may want to mark items as completed in the doc)
- Known limitations (3 deferred follow-ups from B-2 codex)
- T25a scope decision (Phase E2)
- Known-deferred test failures
- Final note on session hygiene

The cold-read test: a new engineer reading HANDOFF.md top-to-bottom should know what shipped, what's locked, what's deferred, and what the deletion seam looks like for Phase D.

## Test coverage at phase boundaries (Peter's watchpoint #3)

**Phase B-1 → B-2, Phase C → E1 — are the integration points tested, or only individual phase contents?**

[FILL IN during /review absorption]:

- **B-1 → B-2**: both workflows use the shared `WorkflowClients` registry from B-1's `_runtime.py`. `tests/test_workflows_runtime.py` covers the registry; `tests/test_concurrent_pipelines.py::TestConcurrentPipelineExecution::test_both_workflows_share_client_registry_safely` covers concurrent registry reads. The repository idempotency fix from B-2 retroactively applied to B-1's action-item repo — `tests/test_action_item_workflow.py` exercises the merger path that uses the fixed `create_version_snapshot`.
- **C → E1**: C's `lambda_ingest/handler.py` schema (workflow names, queue names, EnqueueOptions shape, workflow_id format) is pinned by `tests/test_lambda_handler.py::TestSuccessfulDispatch::test_enqueue_options_carry_lock_invariants` and `test_workflow_ids_match_locked_format`. If C's schema drifts from B-1/B-2's `@DBOS.workflow()` function names, the test asserts on the literal strings `"action_item_workflow"` and `"deal_workflow"` — would catch the drift loudly.

## Known-deferred items (Peter's watchpoint #4) — VERBATIM

The PR body's "What this PR does NOT contain" section enumerates these. Re-confirming for the watchpoint:

| Deferred | Reason | Owner / Trigger |
|---|---|---|
| Phase D (T18-T20): retire `/process`, dispatcher, api_client | Locked 3-phase migration; 2-week rollback window depends on /process alive | Day 14+ post-Phase 2 deploy; separate PR |
| T24: Crash-recovery integration test | P2 in plan, RUN_DBOS_E2E=1 gated | Post-deploy verification via real DBOS infra |
| T25b: Live-LLM characterization test | P2 sanity check, non-blocking | On-demand CI run only |
| T26: `scripts/run_live_e2e.py` DBOSClient update | P2, only relevant at T29 deploy time | T29 deploy phase |
| T35: Rollback-with-inflight test | P2, covered structurally by rollback procedure doc | Optional follow-up; not gating |

---

## Final pre-ship checklist

- [ ] `/review` skill has been run and findings absorbed
- [ ] `/codex review` full-PR has reached explicit "no ship-blocking findings" stop signal
- [ ] All HIGH findings surfaced to Peter BEFORE absorption (and absorption approach approved)
- [ ] Pre-PR brief sections filled in (findings categorization, draft PR title, draft PR body, deletion seams)
- [ ] Test count delta confirmed: 649 still holds at /ship time
- [ ] Watchpoints 1-4 addressed in the brief
- [ ] Peter has been surfaced this brief and explicitly greenlit `/ship`
- [ ] DLQ message untouched (`58863f20-3cda-48f7-973d-3002aa31331b` — wait for T30)
- [ ] `live-transcription-fastapi` untouched (separate repo)
- [ ] Phase D NOT bundled into this PR (separate follow-up Day 14+)
- [ ] `TODOS.md`, `Claude-Context-Limits.txt`, `docs/plans/2026-02-22-event-consumer-implementation.md` left as-is (pre-existing, not Phase work)
