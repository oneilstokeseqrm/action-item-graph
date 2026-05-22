# Phase F Pre-PR Brief — DBOS Migration

**Status:** SYNTHESIZED — ready for /ship gate surface. All findings absorbed across /review + Codex Rounds 1-2.

**Created:** 2026-05-21 (Phase E handoff session)
**Synthesized:** 2026-05-22 (Phase F /review + /codex absorption complete)
**Branch:** `feat/dbos-migration` — **14 commits ahead of `main`** at brief-synthesis time
**Last commit on branch:** `b7477ff` (Codex R2 absorption: drop dead secret_arn back-compat block)

This brief is the structured pre-PR artifact Peter expects to see at the /ship gate. All sections filled.

---

## 14-commit log with one-line summaries

Chronological order on `feat/dbos-migration` (oldest → newest). 10 substantive Phase A-E commits + 1 handoff-prep docs commit + 1 Rule 6 fix + 2 Codex R1 absorption commits + 1 Codex R2 absorption commit.

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
| 10 | `a67a190` | **docs(dbos)** — Phase F handoff prep: HANDOFF.md Phase F resumption guide (8 steps, conditions + watchpoints verbatim, hard constraints) + pre-PR brief skeleton (this file, pre-populated). |
| 11 | `f9d061e` | **fix(dbos): Rule 6 (Phase F /review absorption)** — Deterministic UUID5 IDs for Topic + TopicVersion + Deal CREATEs. Phase B-2 codex absorbed Rule 6 for `create_version_snapshot`; /review specialist dispatch caught three additional gaps in the same class: `repository.create_topic` (S10b retryable), `repository.create_topic_version` (S10b retryable), `_create_new` deal (D7 crash-recovery). All converted to MERGE-keyed-on-deterministic-uuid5. +13 tests (`test_topic_executor_idempotency.py` new, `TestDealCreateNewIdempotency` added). |
| 12 | `f8e282e` | **fix(dbos): register deal_workflow (Codex R1 absorption #1)** — HIGH-severity production cohesion bug: `deal_workflow` module never imported in the production runtime chain → `@DBOS.workflow()` decoration never executed → DBOS workflow registry missing the name → Lambda enqueues to "deal_workflow" would have had no consumer on Railway. 12 prior phase-scoped codex rounds missed it (each reviewed slices; tests imported deal_workflow directly, masking the production gap). Full-PR codex Round 1 caught it. Fix: 1-line side-effect import in `action_item_graph/workflows/__init__.py`. Regression test in `test_api_main.py` asserts both workflow names appear in `DBOSRegistry.workflow_info_map`. +1 test. |
| 13 | `4b155a2` | **fix(dbos): topic_id parity (Codex R1 absorption #2)** — HIGH-severity behavior regression introduced by the Rule 6 commit: keying topic_id on only `(tenant, account, canonical_name)` collapsed two same-canonical-name CREATE_NEW resolutions into one Topic node, but `_create_topic` still ran the brand-new-topic path for both → result was ONE Topic with `action_item_count=1` and summary reflecting only A, NOT legacy's TWO distinct Topics. Fix: include `source_action_item_id` in the key. Preserves Rule 6 retry safety (same resolution → same key) AND legacy parity (two batch-mates → two Topics). +2 tests (renamed + new parity assertion). |
| 14 | `b7477ff` | **fix(infra): drop dead secret_arn back-compat block (Codex R2 absorption)** — LOW: Phase C removed `worker-api-key` from secrets dict, which made the back-compat `if _worker_api_key_arn is not None: pulumi.export("secret_arn", ...)` block a silent no-op. Comment claimed back-compat preservation, code emitted nothing. Grep audit confirmed zero external consumers across this repo's CI/scripts and sibling EQ-CORE repos → safe to drop the block + replace comment with honest post-Phase-C reality. No tests (Pulumi exports not unit-testable). |

---

## Test count delta (reconciled vs main, 2026-05-22)

- **main baseline:** 567 tests (`git checkout main && uv run pytest --collect-only -q`)
- **branch HEAD:** 654 tests (`git checkout feat/dbos-migration && uv run pytest --collect-only -q`)
- **Net delta vs main:** **+87 tests**

Breakdown:
- **Phase A + B + C scaffolding tests:** modest (foundation, runtime, queues, serialization round-trips covered in Phase E1 commits primarily)
- **Phase E (commits 6+7+8): ~+71 net tests** (Phase E commit messages describe "95 new tests" added across 8 files; the +71 net accounts for `test_lambda_api_client.py` being DELETED in Phase E1 and for some pre-existing Lambda tests being rewritten rather than added)
- **Phase F absorption (commits 11+12+13+14): +16 tests** (13 from Rule 6 fix + 1 from cohesion fix + 2 from parity fix + 0 from infra cleanup)

**Reconciling earlier framing:**

The pre-PR brief skeleton (committed pre-Phase-F) claimed "554 baseline + 95 Phase E = 649." That arithmetic was wrong from the start — the actual `main` baseline is 567 not 554, and Phase E shipped +71 NET tests not +95 (the 95 number conflated added tests with net delta, missing the deleted `test_lambda_api_client.py`). The reconciliation here corrects both numbers.

**Confirm at /ship time** with `uv sync --extra lambda --extra dev --extra api && uv run pytest -q`; expected count is 654.

---

## Findings absorbed across /review + all /codex rounds

### From /review skill (single analytical pass)

**HIGH absorbed:** 1 finding (escalated from MEDIUM in data-migration specialist output; treated as HIGH per impact analysis)

- **Rule 6 gap in topic-create + topic-version-create + deal-create paths** (`topic_executor.py:174` + `repository.py:966,1071` + `deal_graph/pipeline/merger.py:242`).
  - **Why HIGH:** Same Rule 6 class as Phase B-2's `create_version_snapshot` fix; deterministic on S10b retry; documented `_create_topic_version` in HANDOFF §2 known-limits but `create_topic` (the canonical Topic node) and deal `_create_new` were NOT in §2 — genuinely new gaps the Phase B-2 audit missed.
  - **Absorbed via commit `f9d061e`** — deterministic UUID5 + MERGE pattern mirroring the Phase B-2 fix. Audit before committing confirmed Owner CREATE (read-then-CREATE provides retry-within-step dedup; §2's narrow-race framing correct, NOT bundled) and extractor uuid4 sites (not reachable from retryable DBOS paths in production topology).

**MEDIUM absorbed:** 0 findings (testing/maintainability specialist findings were INFO-class coverage/cleanup items, not defects — categorized under deliberate non-action below)

**NIT addressed:** 0 standalone NITs (the only mechanical hoist applied — `import uuid as _uuid_module` → top-level `import uuid` in both repositories — was bundled into the Rule 6 commit since those files were already being touched)

**Deliberate non-action with reasoning:**

- **Testing specialist's 6 CRITICAL-tagged + 4 INFO findings** — all are coverage gaps, NOT logic defects. Notable items:
  - `dbos_runtime.py` has no dedicated unit tests (covered indirectly by workflow + integration tests).
  - `create_version_snapshot` deterministic-id behavior asserted via E2E integration only, not unit-level idempotency tests.
  - `_extraction_content_hash` helpers untested.
  - `merging_persist_step` decision-branch coverage incomplete.
  - `match_merge_loop_step` fail-open-per-deal branches untested.
  - Merger split-methods only tested via spy.
  - **Reasoning:** All map to HANDOFF.md § "Phase E test-coverage backlog" items that were intentionally scoped out of Phase E1 per the T25a scope-down decision rationale. These are P2/P3 backlog work for a follow-up test PR, NOT migration-correctness blockers. Live integration test at T30 (DLQ replay) provides production-realistic verification.

- **Maintainability specialist's 10 INFO findings** — all are code-readability cleanups (function-local imports, 7-tuple positional types, magic `0.9` constant, near-duplicate content_hash helpers, `_build_deal_pipeline` over-coupling, long step functions). Two of the four import-hoist NITs landed in the Rule 6 commit (uuid hoist in both repositories) because those files were already being touched. The remaining 8 findings deferred to a follow-up cleanup PR — applying them within this Rule-6-focused PR would muddy the commit narratives and is poor ROI given Phase D will rewrite some of these surfaces anyway.

- **Security specialist's NO FINDINGS** — clean across secret handling, auth, input validation, crypto, IAM scope, logging hygiene, DoS surfaces.

### From /codex review full-PR rounds

#### Round 1 (post-Rule-6-commit state)

- **HIGH absorbed (2 findings):**
  - **Finding #1: `deal_workflow` never registered in production.** Catastrophic phase-cohesion bug — 12 prior phase-scoped codex rounds missed it because each reviewed slices in isolation; tests imported `deal_workflow` directly, masking the production import-chain gap. Surfaced to Peter BEFORE absorption per condition #1. Peter approved Fix A (side-effect import in workflows package __init__) with critical implementation detail (order matters: AFTER existing imports, for circular-safety). Absorbed via commit `f8e282e`. Regression test pins the contract.
  - **Finding #2: My Rule 6 topic_id collapse loses count/summary parity.** Codex tagged MEDIUM; I escalated to HIGH based on impact analysis — external-contract drift violates the migration's byte-for-byte parity hard constraint. Surfaced to Peter BEFORE absorption per condition #1. Peter approved Fix A (include `source_action_item_id` in the key) over Fix B (detect MERGE-existing + reroute) with reasoning: hard-constraint compliance + Rule 6 commit's "retry safety without changing semantics" premise + the Rule 6 commit's overclaim docstring needed retraction. Absorbed via commit `4b155a2`. Includes new test `test_create_topic_same_canonical_name_different_source_produces_distinct_topics` that would have caught the regression pre-ship.

- **MEDIUM absorbed:** 0 in this round (the 2 findings above were the only ones)
- **NIT addressed:** 0
- **Deliberate non-action:** 0
- **Verdict:** "1 ship-blocking finding; 1 non-blocking" (codex's wording) — both surfaced + absorbed per Peter's discipline

#### Round 2 (post-Codex-R1-absorption state)

- **HIGH absorbed:** 0
- **MEDIUM absorbed:** 0 in code (1 finding documented as deliberate non-action — see below)
- **NIT addressed:** 0
- **Deliberate non-action with reasoning:**
  - **Finding A (MEDIUM): Deal `opportunity_id` collapses byte-identical same-interaction extractions.** Codex flagged that my deal `_create_new` keys on `tenant + interaction + content_hash`, which would collapse two byte-identical extracted deals from the same envelope into one Deal node (where legacy `uuid7()` would have produced two duplicates). Surfaced to Peter per condition #1 with full reasoning. **Peter approved deliberate non-action.** Topic-vs-deal scenario distinction: topic case had legacy correct + new code regressed (real regression worth fixing); deal case has legacy buggy + new code happens to fix the bug (improvement, not regression). The "byte-for-byte parity" hard constraint protects the shape of external contracts, not the exact form of legacy bugs. D7's matcher already catches byte-identical extractions in 99%+ of real flows by routing iteration-2 to `_merge_existing` not `_create_new`. Three layers of plumbing (D7 → merge_deal → _create_new) to preserve duplicate-creation in an anomalous case the matcher catches is poor ROI.

- **Fix applied autonomously (LOW):**
  - **Finding B (LOW): `infra/__main__.py:75-82` claims back-compat `secret_arn` export but emits nothing.** Phase C removed `worker-api-key` from secrets dict, which silently dropped the gated `secret_arn` export. Comment was misleading, code was correct. Surfaced to Peter per "surface after Round 2 regardless of findings" instruction. Peter approved Option A (drop dead block + update comment) with grep-first discipline. **Grep audit:** in-repo refs (only `infra/`'s own files), CI workflow (uses unrelated GHA secrets), scripts/ (no `pulumi stack output` invocations), cross-repo sibling EQ-CORE projects (zero refs). Confirmed zero external consumers. Absorbed via commit `b7477ff`.

- **Verdict:** "0 ship-blocking findings; 2 non-blocking" — explicit stop signal per condition #2 (severity-driven; HIGH → MEDIUM/LOW transition + "0 ship-blocking" = stop). Round 3 NOT invoked.

#### Round 3+ — NOT INVOKED

Round 2's "0 ship-blocking" verdict + severity drop (HIGH in R1 → MEDIUM/LOW in R2) IS the stop signal per Peter's reinforcement #2 ("codex round counts are evidence, not budget"). No further rounds.

### Phase E /codex review outcome (already absorbed pre-Phase-F, FYI for full-PR context)

Run on `3ca3b1e..HEAD` with constrained scope (medium reasoning, no exploration, diff-only). Verdict: **"No high-severity ship blockers found."** Non-blocking observation #1 absorbed via commit `75ca5f2`; observation #2 (T34 smoke test framing) documented in test docstring. Both pre-date Phase F.

---

## Draft PR title (<70 chars)

**Recommended:**

```
feat(dbos): migrate ingest path to DBOS durable workflows (Phases A→E)
```

(69 chars including the conventional-commit prefix.)

---

## Draft PR body

```markdown
## Summary

Migrates the action-item-graph ingest path from a synchronous Lambda → Railway `/process` HTTP call (120s Lambda timeout failure class) to a DBOS-orchestrated durable workflow with per-step retry, checkpointing, and observability.

**14 commits.** Phase A (foundation), Phase B-1 (action-item workflow), Phase B-2 (deal workflow + repository idempotency fix), Phase C (Lambda dispatcher cutover), Phase E1+E2+E3 (test coverage), 1 docs commit, and 3 Phase F /review-and-/codex absorption commits. Codex absorption across all phases: 14 rounds reaching an explicit "no ship-blocking" stop signal each time.

**The Phase F full-PR codex pass caught a phase-boundary seam that 12 prior phase-scoped codex rounds missed:** `deal_workflow` was never imported in the production runtime chain. The `@DBOS.workflow()` decoration only runs at module import time, and `api/main.py`'s import chain reached `action_item_graph.workflows` (which decorates `action_item_workflow`) but never touched `deal_graph.workflows`. DBOS would not have known about the `deal_workflow` name. Lambda enqueues to `workflow_name="deal_workflow"` would have had no Railway consumer — deal pipeline silently inert in production after Phase 2 deploy. Tests passed because they import `deal_workflow` directly, masking the production gap. Commit `f8e282e` closes the gap (1-line side-effect import in `action_item_graph/workflows/__init__.py`) + adds a regression test (`test_api_main.py::TestDBOSWorkflowRegistration`) that asserts both workflow names appear in `DBOSRegistry.workflow_info_map` so this can't regress unnoticed. **This catch is the strongest empirical evidence for running full-PR codex on top of phase-scoped reviews** — the production import topology is a structural concern only visible when reviewing the full diff against `main`.

**What this PR contains** (commit-level):

- **Phase A** (`1bfd42d`) — DBOS runtime substrate + Neon `eq_aig_dbos_sys` state DB + FastAPI lifespan integration.
- **Phase B-1** (`6e51059`) — Action-item DBOS workflow: 14 steps mapping 1:1 to the legacy pipeline stages with S9/S10 LLM-vs-write splits per the locked step decomposition.
- **Phase B-2** (`6e11307`) — Deal DBOS workflow: 9 steps + D7 inner merger refactor + repository-layer idempotency fix (version-snapshot CREATE-randomUUID → MERGE on deterministic UUID5; same latent bug retroactively fixed in action-item repo).
- **Phase C** (`3ca3b1e`) — Lambda dispatcher cutover: `submit_to_railway` HTTP forwarder replaced with two `DBOSClient.enqueue` calls (one per pipeline workflow). CloudWatch `partial_enqueue_pair_count` metric on first-succeeds-second-fails split window.
- **Phase E1+E2+E3** (`8711106`, `5b136a6`, `c042207`) — New tests across 8 files: Lambda dispatcher tests (T21), action-item workflow tests (T22), deal workflow tests (T23), behavioral contract tests (T25a scoped down — see HANDOFF.md § "T25a scope decision" for rationale), compatibility tests (T32+T33), concurrent-execution tests (T34).
- **Phase F absorption** — `f9d061e` (Rule 6 extended to topic_executor + create_topic + create_topic_version + deal _create_new — three additional gaps the Phase B-2 audit missed, caught by /review's data-migration specialist); `f8e282e` (the deal_workflow registration cohesion fix described above); `4b155a2` (topic_id parity restoration — the Rule 6 commit's deterministic key on `(tenant, account, canonical_name)` collapsed legitimate batch-mate Topics; fix adds `source_action_item_id` to the key, preserving Rule 6 retry safety AND restoring legacy "two distinct Topics per same-canonical-name in one batch" parity); `b7477ff` (drop dead `secret_arn` Pulumi back-compat block — Phase C removed `worker-api-key` from secrets dict which silently dropped the gated export; grep audit confirmed zero external consumers).
- **Bug-fix commits during phases** — `00c849e` (Case A targeted-deal opportunity_id parity, Rule 8 codified); `75ca5f2` (Phase E codex absorption: test rename).

**Test count delta:** 567 → 654 (+87 net vs main; +16 from Phase F absorption alone). All 654 passing with `uv sync --extra lambda --extra dev --extra api && uv run pytest`.

**Patterns earned and codified** in `~/.claude/projects/.../memory/`:
- Rules 1–6 in B-1+B-2 (validation in workflow body, LLM-prompt source-of-truth, JSON-serialization discipline, zip parallel-array length assertion, downstream-produced identifier authority, repository-method idempotency audit)
- Rule 7 in C (cross-service reference call shape must match deployment topology)
- Rule 8 in E (Pydantic @property reads through model_dump must replicate dict-key path)
- Feedback memory `feedback_test_seam_signals.md` from E (two mock strategies fighting different layers = codebase signaling natural test seams)

## What this PR does NOT contain

Intentionally deferred. The PR ships Phases A through E + Phase F absorption only; everything below is documented as out-of-scope so future reviewers and ops folks can immediately distinguish "shipping in this PR" from "coming in follow-ups":

- **Phase D (T18-T20)** — retire `/process` HTTP route + `dispatcher.py` + delete legacy code. Deferred to a follow-up PR Day 14+ post-Phase 2 deploy. The 2-week rollback window depends on `/process` staying alive during Phase 1+2. Phase D PR will also delete the migration-window-only test files (see "Deletion seam list" below) and clean up the `Pulumi.prod.yaml` `worker-api-key` + `api-base-url` config entries.
- **Pulumi.prod.yaml config cleanup** — `worker-api-key` (encrypted) + `api-base-url` config entries are still in the encrypted stack config but no longer read by `forwarder`. Bundled into Phase D PR — removing encrypted-stack config has higher rollback implications than removing an output export, so deferred with the rest of Phase D.
- **T24 — Crash-recovery integration test** (`RUN_DBOS_E2E=1` gated). P2 in the plan; deferred to post-deploy verification.
- **T25b — Live-LLM characterization test.** P2 sanity check (not a contract regression); deferred.
- **T26 — `scripts/run_live_e2e.py` DBOSClient update.** P2; only relevant when the DBOS state DB is provisioned at T29 deploy time.
- **T35 — Rollback-with-inflight test.** P2; covered structurally by the rollback procedure documented in the execution plan.
- **HANDOFF.md §2 known limitations** — three bounded follow-ups documented at the time of Phase B-2 codex absorption: (a) version counter compare-and-set on `update_action_item`/`update_deal` (CAS guard against double-increment under retry), (b) Owner CREATE narrow cross-workflow race (read-then-CREATE provides retry-within-step dedup; cross-workflow concurrent path is the actual race), (c) S9a/S10a per-step instrumentation asymmetry (operational dashboards should encode the asymmetry, not a code bug). All three are bounded enough to monitor in production; address if they surface.
- **8 maintainability NITs from /review** — function-local imports in workflow bodies + step bodies (2 sites), 7-tuple positional types in S10a, magic `0.9` MEDDIC role-match confidence constant, near-duplicate `_extraction_content_hash` helpers across action-item + deal mergers, `_build_deal_pipeline` over-coupling for D2/D3/D5 (only needs the repository), 2 long step functions that could decompose (`merging_persist_step` ~80 lines, `match_merge_loop_step` ~155 lines). All readability cleanups; non-blocking. Follow-up cleanup PR.
- **Phase E test-coverage backlog from /review** — `dbos_runtime.py` lacks dedicated unit tests, `create_version_snapshot` deterministic-id behavior asserted via integration only, `_extraction_content_hash` helpers untested, `merging_persist_step` decision-branch coverage incomplete, `match_merge_loop_step` fail-open-per-deal branches untested, merger split-methods only tested via spy. Live integration at T30 (DLQ replay) provides production-realistic verification. Follow-up test PR.

## Test plan

- [ ] CI runs the full pytest suite: 654 tests passing
- [ ] `uv run pyright` clean on all touched modules (except pre-existing missing-import warnings for boto3 + aws_lambda_powertools — only bundled into the Lambda zip by `scripts/package_lambda.sh`, NOT in pyproject.toml; documented in HANDOFF.md § "Known-deferred test failures")
- [ ] `/review` skill cleared (Phase F)
- [ ] `/codex review` full-PR cleared with explicit "0 ship-blocking findings" (Codex R2, stop signal)
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
- **Rule 6 retry safety end-to-end.** Phase F /review absorption pushed deterministic-UUID5-MERGE coverage from version snapshots (Phase B-2) into Topic, TopicVersion, and Deal entity creates as well. The migration now ships with comprehensive retry idempotency across every CREATE reached from a retryable DBOS step.
- **Foundation for Phase D removal of `/process`** — once the 2-week rollback window passes without incidents, the legacy HTTP path retires.

## Deletion seam list (Phase D follow-up PR)

The following files/code are migration-window-only and will be deleted in the same PR that retires `/process` from production traffic (Phase D, Day 14+ post-deploy):

- `src/action_item_graph/api/routes/process.py` — the `/process` HTTP route handler
- `src/dispatcher/dispatcher.py` — the synchronous EnvelopeDispatcher
- `tests/test_compatibility_process_route.py` — T32 tests, only relevant during the Phase 1 mid-state window
- `tests/test_workflow_behavioral_contract.py` — T25a behavioral contracts, scoped to the migration window per the test docstring's deletion-seam note
- `tests/test_lambda_handler.py::TestT33DBOSUnreachable` — T33 tests, deploy-order safety net relevant only during Phase 2
- `Pulumi.prod.yaml` `worker-api-key` config entry (kept for back-compat during the window; remove in Phase D)
- `Pulumi.prod.yaml` `api-base-url` config entry (same reason)

`tests/test_concurrent_pipelines.py` (T34) STAYS alive past Phase D — concurrent W2 execution is the steady-state production behavior. Same for `tests/test_topic_executor_idempotency.py` and `tests/test_api_main.py::TestDBOSWorkflowRegistration` (both pin Rule 6 + cohesion invariants for the long term).
```

---

## Cross-phase architectural cohesion (Peter's watchpoint #1)

**Does Phase A's DBOS runtime + Phase B's step decomposition + Phase C's Lambda dispatcher + Phase E's tests hang together as one coherent migration?**

**Observation: Phase F full-PR review caught a real cohesion gap that 12 prior phase-scoped reviews missed.**

The phases form a clean dependency chain in the abstract:
- **A → B**: A provides the DBOS runtime that B's @DBOS.workflow + @DBOS.step decorators register against. B's step decomposition is sized to the per-step retry/checkpoint policy A's runtime enables.
- **B → C**: B's `action_item_workflow` + `deal_workflow` function names are what C's Lambda dispatcher targets via `EnqueueOptions["workflow_name"]`. The dispatcher invokes the workflows; the workflows execute on Railway via A's DBOS workers.
- **C → E**: C's dispatcher schema is what E1's T21 Lambda tests cover. C's two enqueue calls is what E3's T34 concurrent-write test exercises.

**But the production import topology was NOT verified end-to-end until Phase F codex R1.** Each phase-scoped codex round reviewed its slice in isolation:
- B-2 codex confirmed deal_workflow.py had correct `@DBOS.workflow()` decoration
- C codex confirmed handler.py had correct enqueue call shape
- E codex confirmed tests covered both workflows

But no review pass exercised the full import chain `api/main.py → action_item_graph.workflows → @DBOS.workflow() execution`. The deal_workflow module was never imported in production, so its decoration never ran, so DBOS's workflow registry never knew about it. Tests passed because they imported `deal_workflow` directly — the test process saw the registration; the production process would not have.

**This is exactly the class of bug watchpoint #1 was meant to surface.** The fix (commit `f8e282e`) added a 1-line side-effect import in the workflows package __init__ + a regression test asserting both workflow names appear in `DBOSRegistry.workflow_info_map` post-import. Future engineers can't drop the side-effect import without the test failing loudly.

Other phase-boundary observations:
- The queue-name string duplication between `queues.py` and `lambda_ingest/handler.py` is INTENTIONAL (Lambda can't import the queues module without triggering DBOS runtime registration). Pinned by `test_lambda_handler.py:142-147` literal-string assertions. Documented in code comments. This is a deliberate-and-protected seam, not a latent bug.
- The merger refactor's `_merge_items` legacy entry preserved alongside the new split methods (`construct_*_llm` + `persist_*_neo4j`) is intentional dual-path during the migration window. Phase D removes the legacy entry.
- Repository idempotency: Phase B-2 absorbed Rule 6 for version snapshots; Phase F extended Rule 6 coverage to Topic + TopicVersion + Deal CREATEs (3 commits: `f9d061e`, `4b155a2`). The migration now ships with end-to-end retry idempotency at the repository layer.

**Verdict on cohesion:** As-shipped (post-Phase-F absorption), the migration is coherent. The deal_workflow registration gap was a real seam; it's closed. The maintainability findings about long step functions and shared helpers between the two pipelines' mergers are recognized as cleanup opportunities for a follow-up PR, not migration-correctness concerns.

## Documentation completeness (Peter's watchpoint #2)

**Does HANDOFF.md describe the full migration arc such that someone reading it cold can understand what shipped and why?**

HANDOFF.md sections include TL;DR, project trajectory (Sessions 1–4 narrative), locked decisions, reasoning-not-in-the-plan (6 "why" sections), key artifacts, Phase F resumption guide (8 steps), Phase A+B+C+E execution log (per-phase commit-level narratives), Phase E test-coverage backlog, Known limitations (3 deferred follow-ups), T25a scope decision, Known-deferred test failures.

**Phase F absorption narrative is appended to HANDOFF.md as part of the same commit that lands this brief synthesis** (mirroring the existing per-phase execution-log structure). New section covers: Rule 6 gap discovery via /review specialist dispatch (commit `f9d061e`); deal_workflow registration discovery via full-PR /codex Round 1 (commit `f8e282e`) — with the watchpoint #1 attribution; topic_id parity restoration (commit `4b155a2`); the deliberate-non-action verdict on the deal opportunity_id MEDIUM finding with the topic-vs-deal scenario distinction reasoning preserved verbatim (so future engineers reading cold understand why the symmetric-looking case wasn't fixed); the `secret_arn` cleanup (commit `b7477ff`); Phase F codex closed Round 2 with 0 ship-blocking findings.

**Verdict on doc completeness:** HANDOFF.md, post-this-commit, comprehensively covers the migration arc through Phase F. A cold-read engineer can understand what shipped, what was deferred, and what the deletion seam looks like for Phase D.

## Test coverage at phase boundaries (Peter's watchpoint #3)

**Phase B-1 → B-2, Phase C → E1 — are the integration points tested, or only individual phase contents?**

- **B-1 → B-2**: Both workflows use the shared `WorkflowClients` registry from B-1's `_runtime.py`. `tests/test_workflows_runtime.py` covers the registry. `tests/test_concurrent_pipelines.py::TestConcurrentPipelineExecution::test_both_workflows_share_client_registry_safely` covers concurrent registry reads. The repository idempotency fix from B-2 retroactively applied to B-1's action-item repo — `tests/test_action_item_workflow.py` exercises the merger path that uses the fixed `create_version_snapshot`. **Now also**: Phase F Rule 6 fix (`f9d061e`) extends idempotency to Topic + TopicVersion + Deal CREATEs; `tests/test_topic_executor_idempotency.py` (10 new tests) + `TestDealCreateNewIdempotency` (3 new tests) pin the determinism + parity contracts.
- **C → E1**: C's `lambda_ingest/handler.py` schema (workflow names, queue names, EnqueueOptions shape, workflow_id format) is pinned by `tests/test_lambda_handler.py::TestSuccessfulDispatch::test_enqueue_options_carry_lock_invariants` and `test_workflow_ids_match_locked_format`. If C's schema drifts from B-1/B-2's `@DBOS.workflow()` function names, the test asserts on the literal strings `"action_item_workflow"` and `"deal_workflow"` — would catch the drift loudly.
- **NEW: C/A → Production runtime cohesion**: Phase F codex R1 caught the deal_workflow registration gap. Fix commit (`f8e282e`) adds `tests/test_api_main.py::TestDBOSWorkflowRegistration::test_both_workflows_registered_after_production_import_chain` — imports the production entrypoint chain and asserts both workflow names appear in `DBOSRegistry.workflow_info_map`. This pins the cross-phase production-import contract; future engineers can't drop the side-effect import without the test failing loudly with an actionable error message.

**Verdict on phase-boundary coverage:** As-shipped (post-Phase-F absorption), all critical phase-boundary contracts have explicit test coverage. The Phase F-added tests close the production-import-topology gap that 12 prior reviews missed.

## Known-deferred items (Peter's watchpoint #4) — VERBATIM

The PR body's "What this PR does NOT contain" section enumerates these. Re-confirming for the watchpoint:

| Deferred | Reason | Owner / Trigger |
|---|---|---|
| Phase D (T18-T20): retire `/process`, dispatcher, api_client | Locked 3-phase migration; 2-week rollback window depends on /process alive | Day 14+ post-Phase 2 deploy; separate PR |
| Pulumi.prod.yaml worker-api-key + api-base-url config cleanup | Removing encrypted-stack config has higher rollback implications than removing an output export; defer with Phase D | Bundled into Phase D PR |
| T24: Crash-recovery integration test | P2 in plan, RUN_DBOS_E2E=1 gated | Post-deploy verification via real DBOS infra |
| T25b: Live-LLM characterization test | P2 sanity check, non-blocking | On-demand CI run only |
| T26: `scripts/run_live_e2e.py` DBOSClient update | P2, only relevant at T29 deploy time | T29 deploy phase |
| T35: Rollback-with-inflight test | P2, covered structurally by rollback procedure doc | Optional follow-up; not gating |
| Phase E test-coverage backlog (dbos_runtime unit tests, content_hash helpers, merging_persist branch coverage, match_merge_loop branches, merger split-method direct tests) | Identified by Phase F /review testing specialist; P2/P3 backlog work; live integration test at T30 provides production-realistic verification | Follow-up test PR |
| 8 maintainability NITs (function-local imports x2, 7-tuple positional types, magic 0.9 MEDDIC constant, shared content_hash util, _build_deal_pipeline coupling, 2 long step functions) | Identified by Phase F /review maintainability specialist; non-blocking readability cleanups | Follow-up cleanup PR |
| HANDOFF.md §2 known limitations: version counter CAS, Owner CREATE narrow cross-workflow race, S9a/S10a per-step instrumentation asymmetry | Documented in HANDOFF.md as bounded; deferred from Phase B-2 codex absorption | Live-traffic monitoring; address if observed in production |

---

## Final pre-ship checklist

- [x] `/review` skill has been run and findings absorbed
- [x] `/codex review` full-PR has reached explicit "0 ship-blocking findings" stop signal (Codex R2)
- [x] All HIGH findings surfaced to Peter BEFORE absorption (and absorption approach approved): 3 surface events (Rule 6 gap, deal_workflow registration, topic_id parity); 0 surfaces on MEDIUM/LOW absorbed autonomously
- [x] Pre-PR brief sections filled in (findings categorization, draft PR title, draft PR body, deletion seams, watchpoints)
- [x] Test count delta confirmed: 567 main → 654 branch HEAD = +87 net; confirmed via `git checkout main && uv run pytest --collect-only -q` (2026-05-22) — supersedes earlier "638 / 649" framing
- [x] HANDOFF.md "Phase F absorption" subsection appended (lands in same commit as this brief synthesis)
- [x] Watchpoints 1-4 addressed in the brief with concrete observations
- [ ] Peter has been surfaced this brief and explicitly greenlit `/ship` ← **awaiting**
- [x] DLQ message untouched (`58863f20-3cda-48f7-973d-3002aa31331b` — wait for T30)
- [x] `live-transcription-fastapi` untouched (separate repo)
- [x] Phase D NOT bundled into this PR (separate follow-up Day 14+)
- [x] `TODOS.md`, `Claude-Context-Limits.txt`, `docs/plans/2026-02-22-event-consumer-implementation.md` left as-is (pre-existing, not Phase work)
