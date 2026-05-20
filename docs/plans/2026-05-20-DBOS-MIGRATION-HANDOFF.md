# DBOS Migration — Cross-Session Handoff

**Created:** 2026-05-20
**Purpose:** Single document the next agent reads first to pick up where the planning sessions ended. Captures project trajectory, locked decisions with reasoning, key artifacts, and the conversational context that doesn't survive in the plan file alone.

**Reading order (if you're the next agent):**
1. This document (orientation)
2. `docs/plans/2026-05-20-dbos-migration-execution-plan.md` (RESUMPTION CHECKLIST at top + full plan)
3. `~/.gstack/projects/oneilstokeseqrm-action-item-graph/peteroneil-main-design-20260520-065353.md` (deeper reasoning)
4. `/Users/peteroneil/EQ-CORE/live-transcription-fastapi/services/` (reference codebase — read but don't modify)
5. `~/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/MEMORY.md` and linked files (patterns + project state)

---

## TL;DR

A 2026-05-19 Lambda timeout incident (Anthropic security questionnaire ~5,500 words exceeded the 120s Lambda timeout) surfaced an architectural mismatch: an LLM-heavy 10-stage pipeline running synchronously inside a Lambda function with a hard 120s ceiling. The bandaid fix (raise Lambda timeout) was rejected by Peter in favor of a structural fix: migrate to a DBOS-orchestrated durable workflow that runs the pipeline asynchronously on Railway, with per-step retry, checkpointing, and observability.

Two planning sessions (2026-05-20) produced an APPROVED design doc and an APPROVED execution plan with 40 implementation tasks. Implementation has not yet started. The next agent picks up at Task #4 (T1: create feature branch, T3: Docker container verification of dbos Lambda compatibility).

---

## Project trajectory

### Pre-planning context (2026-05-19)

The `eq-synthetic-date-generation` project's Session 14 injected 3 additive synthetic accounts (Anthropic + Linear + partial Snowflake) into the EQ test tenant. Total ~38 new interactions processed in ~45 min. During this window, 1 unique message ended up in `action-item-graph-dlq` after exhausting SQS retries. CloudWatch confirmed Lambda hit its 120s timeout repeatedly on the same content. Forensic inspection revealed the message was an Anthropic vendor security questionnaire (~5,500 words, 20 sections, dense structured content) — a legitimate user input, not a malformed envelope.

The Session 15 investigation brief at `/Users/peteroneil/EQ-CORE/eq-synthetic-date-generation/docs/plans/session-15-action-item-graph-investigation-brief.md` documents the forensic details. The DLQ message MessageId is `58863f20-3cda-48f7-973d-3002aa31331b` and is preserved as the live integration test for the migration (do not touch until T30).

### Session 1 — Investigation + Architecture Brainstorm (2026-05-20 morning)

Started as an investigation session. Read the Lambda handler, the Pulumi infra, the Railway FastAPI app, and the 10-stage pipeline. Found:

- Lambda timeout: 120s, memory: 256 MB, HTTP_TIMEOUT_SECONDS: 100, MAX_RETRIES: 2 (dead code given the Lambda timeout math)
- The Lambda is a thin forwarder; the actual LLM work runs synchronously in Railway's `/process` endpoint
- The dispatcher runs action-item + deal pipelines concurrently via `asyncio.gather(return_exceptions=True)`
- CloudWatch tape showed median ~30–50s per invocation but the right tail extends past 100s — this isn't a freak event, it's a fat-distribution failure mode

Initial recommendation was a 3-option ladder (raise timeout / chunk / move out of Lambda). Peter pushed back: "Which solution or pattern would a cutting-edge startup use? What's the best architectural approach going forward, and are you sure that the three options you provided are the cutting-edge approaches?"

Research surfaced durable-execution engines (Inngest, Trigger.dev, Temporal, DBOS, Lambda Durable Functions). Initially recommended Inngest as the AI-startup default. Peter overrode: "DBOS is already integrated in eq-email-pipeline... it's not a compromise; it's the engine architecturally designed for our stack (Python + Postgres + AI workloads)."

Validation check: I couldn't find DBOS in eq-email-pipeline. Peter corrected: DBOS lives in live-transcription-fastapi's account_provisioning module, not eq-email-pipeline. Confirmed via grep — DBOS v2.22.0 is in production in LTF.

Direction redirected to /office-hours → /plan-eng-review → implementation.

### Session 2 — /office-hours (2026-05-20 mid-day)

Ran the office-hours skill to brainstorm the architectural shape. Key gates:

- **Premise check** (D1 in skill numbering): Initial 8 premises. Peter challenged #2 ("10 stages map cleanly") and #4 ("Lambda stays as thin shim") as unverified claims dressed as premises. Refined: #2 demoted to hypothesis pending stage-dependency-map verification; #4 demoted to leaning with 4 alternatives to compare in Phase 4.
- **Phase 3.5 inserted (stage dependency map):** Read `pipeline.py` lines 234–504. Verified dependency graph is linear and acyclic — clean DBOS decomposition holds. Found 3 known refactor items: stage 5 in-place mutation, stage 6.5 try/except boundary, stages 10/11 fail-open mechanism.
- **DBOSClient discovery:** Web search + DBOS Python SDK inspection confirmed `DBOSClient` is a first-class API for enqueueing workflows from outside the DBOS runtime. This eliminated the HTTP `/internal/dbos-enqueue` endpoint requirement and reframed the Lambda's role from "5-line shim" (which I'd misrepresented) to "~15 lines of DBOSClient calls."
- **Phase 4 alternatives (D3 in skill numbering):** 4 trigger architecture options (A: HTTP endpoint, B: DBOSClient, C: SQS-poller-on-Railway, D: EventBridge API destination). Peter approved B (Lambda + DBOSClient) with the framing "Cat 1 vs Cat 2" — keep Lambda as HA buffer (Cat 1) or eliminate Lambda (Cat 2).
- **Workflow shape (D4):** W1 (single parent + child sub-workflows) vs W2 (two flat parallel workflows). My initial argument for W2 had two weak points (LTF precedent — actually ambiguous; nested-workflow API — actually first-class in DBOS as of March 2026). Peter forced me to retract those and earn the conclusion via data-level independence verification. Read full deal_graph/pipeline/pipeline.py to confirm no shared computed state. W2 re-confirmed on data-independence grounds.
- **Step decomposition + retry policy + idempotency model (D5):** 12 steps for action-item pipeline. Per-step retry policy table mirroring LTF's pattern.
- **Design doc**: Written to `~/.gstack/projects/oneilstokeseqrm-action-item-graph/peteroneil-main-design-20260520-065353.md`. 3 spec-review iterations (quality 6→6→7/10). 15 issues caught + fixed across iterations. APPROVED at iteration 3.

Total: 7 architectural decisions ratified via AskUserQuestion gates with Peter. 23 open questions enumerated for /plan-eng-review.

### Session 3 — /plan-eng-review (2026-05-20 afternoon)

Ran the plan-eng-review skill. Goal: lock the 23 open questions into specific decisions with reasoning, plus run pre-implementation spikes for the items that needed empirical verification.

- **Spikes executed:** Opens #4 (dbos package Lambda compat — high-confidence analytical, Day-1 Docker verification deferred to implementation T3), #5 (Neon connectivity — public endpoint, no VPC needed), #11 (DBOS jitter — not native, ships without), #21 (DBOSClient signature — verified at dbos==2.22.0), #19 bonus (workflow_timeout — found `EnqueueOptions.workflow_timeout: float`).
- **Code reads:** Open #1 (stage 5 in-place mutation callers — all internal), Open #9 (verifier fail-open — verified at `verifier.py:99-107`), Open #10 (LLM step purity audit — pure), Open #16 (deal pipeline step decomposition — 9 steps).
- **All 23 opens resolved.** Plan written to `docs/plans/2026-05-20-dbos-migration-execution-plan.md` with 31 ordered implementation tasks.
- **Codex outside voice consult:** Ran 2 rounds. First round (5 min, high effort) timed out exploring the codebase. Second round (3 min, medium effort, no exec) returned 20 substantive findings.
- **Codex absorption:** 19 of 20 findings accepted. Plan updated:
  - **#5/6/7** — Added Shared Neo4j Write Analysis section proving commutativity of all shared writes (Account, Interaction, ENGAGED_ON) via idempotent MERGE + COALESCE pattern.
  - **#10** — Split S9 (merging) → S9a (LLM construct) + S9b (Neo4j write); split S10 (topic_resolution) → S10a + S10b. Action-item pipeline grew 12 → 14 steps. Deal pipeline kept 9 steps but inner `merger.merge_deal` function refactored to separate LLM from write at function level.
  - **#9** — Regression test split into deterministic-mock (PRIMARY, exact-equality, blocking) + live-nondeterminism characterization (±20%, non-blocking).
  - **#14/15** — CloudWatch metric `partial_enqueue_pair_count` with alarm on partial states.
  - **#16** — REJECTED with escape valve. W2 stays. W1 would be reconsidered if `partial_enqueue_pair_count` > 5% steady-state OR a third pipeline is added.
  - **#11/12/13** — Rollback recovery time updated from "~2 min" to "~12 min realistic" (Lambda redeploy ~2 min + in-flight workflow drain ~10 min). Default rollback policy: let in-flight workflows complete naturally; emergency manual SQL CANCELLED only for confirmed data-corruption.
  - **#17/18/19** — Per-queue concurrency lowered 3 → 1 at V1. Raise after empirical criterion: 100 successful invocations with no DB/Neo4j/OpenAI errors AND queue-depth trending positive. Open #24 added: checkpoint payload size analysis for split steps (S9a/S10a outputs ~64 KB worst-case, well under DBOS row limits).
  - **#1/3/4** — Cutover renamed to "Phased Migration" with 3 explicit phases (deploy DBOS infrastructure / shift Lambda traffic / Day-14+ endpoint removal).
  - **#20** — Compatibility tests added: T32 (old Lambda + new Railway), T33 (new Lambda + old Railway), T34 (concurrent-write), T35 (rollback-with-inflight).
- **Plan grew 31 → 40 tasks; ~16-22 hrs → ~20-25 hrs CC implementation time.**
- **Final approval:** APPROVED. /plan-eng-review closed cleanly. Task #3 marked complete; Task #4 pending greenlight.

---

## Locked decisions (consolidated)

| Decision | Lock state | Reasoning summary | Escape valve |
|---|---|---|---|
| **D1: Trigger architecture** = Lambda + DBOSClient | Conditional | First-class DBOS SDK pattern; ~15 LOC Lambda diff; no HTTP endpoint to write+test+secure | If T3 Docker verification fails: fall back to Approach A (HTTP `/internal/dbos-enqueue` endpoint) per plan |
| **D2: Workflow shape** = W2 two flat parallel workflows | Locked with monitoring | Data-independence verified at the pipeline level; W1 nested-workflow complexity not justified | Revisit W1 if (a) third pipeline added OR (b) `partial_enqueue_pair_count` > 5% steady-state |
| **D3: Step decomposition** = 14 action-item + 9 deal steps | Locked | 1:1 with current stages for action-item, with S9/S10 split to separate LLM compute from Neo4j write (Codex #10) | None — fundamental to retry safety per [[pattern-dbos-step-decomposition]] |
| **D4: Retry policy** = Native @DBOS.step, no jitter, concurrency=1 | Locked at V1 | DBOS has no native jitter; in-step helper overkill at our volume; concurrency=1 conservative until production evidence | Raise concurrency after 100 successful invocations with zero DB/Neo4j/OpenAI errors AND queue-depth metric trending positive |
| **D5: Idempotency model** = DBOS-default checkpointing | Locked | LLM steps verified pure (no telemetry/cache side effects); idempotent SQL writes covered by Neo4j MERGE + Postgres ON CONFLICT | If a future LLM step has side effects, split into pure-LLM + pure-side-effect step per premise 2 contingency |
| **State DB** = Separate Neon database `eq_aig_dbos_sys` in existing `super-glitter-11265514` (eq-dev) project | Locked | Plan's literal wording ("database" not "project") + matches LTF's pattern (their DBOS state lives in `neondb.dbos.*` inside the same `eq-dev` project). Clean ownership via distinct database; LTF and AIG have separate `dbos.workflow_status` tables. Migrating up to a dedicated project later is feasible. | None — separation is a one-way decision. Upgrade to dedicated project only if shared compute or cross-service `pg_database_size` quota pressure shows up. |
| **Cutover** = 3-phase migration | Locked | Phase 1 (deploy DBOS infra + new Railway with /process alive) → Phase 2 (shift Lambda traffic) → Phase 3 (Day 14+, delete /process) | Phase 3 timing extends if any incidents in Phase 2 |
| **Workflow ID format** = `f"action-item-graph:{pipeline}:interaction-{uuid}"` | Locked at base | Verified against DBOS validator at v2.22.0 | Future-enhancement: append retry-attempt-id suffix if FAILED re-delivery requires fresh attempt (Open #3) |

---

## Reasoning that doesn't survive only in the plan

Some context lives in this conversation but not fully in the plan file. Capture for future agents:

### Why Inngest was researched but DBOS was chosen

Initial research surfaced Inngest as the "AI-startup default" for durable execution. Peter overrode based on (1) DBOS already integrated in LTF, (2) Postgres-anchored is the right architectural fit for the EQ stack (already on Neon Postgres + FastAPI + Python). Adopting a second engine (Inngest) would be architectural debt. **The decision is correct but rests on the LTF integration. If LTF ever migrates away from DBOS, the EQ standard would need to be reconsidered.**

### Why W2 over W1 specifically

The W2 vs W1 decision had two false arguments I'd initially used (LTF precedent — actually ambiguous; nested-workflow API maturity — actually first-class in DBOS). Peter caught both. The actual load-bearing argument for W2 is **data-level independence**: pipelines build state from EnvelopeV1 alone, share no intermediate computed state. The "convergent writes at Neo4j MERGE layer" was caught by Codex later as not the same as data independence — required the Shared Neo4j Write Analysis to prove commutativity separately. **Don't conflate these in future architectural reviews.** See [[pattern-data-independence-vs-commutativity]].

### Why the cutover model changed mid-session

The plan initially described cutover as "EventBridge target swap with feature-flagged rollback" (from /office-hours). During plan-eng-review, I realized cutover is actually simpler: Lambda still consumes from the same SQS queue, same EventBridge rule. Only the Lambda's internal call changes. No EventBridge target swap. Codex then caught a contradiction: the plan said "cutover = single Lambda deploy" but the operational sequence was multi-day phased. Renamed to "Phased Migration" with 3 explicit phases per Codex absorption. **The 3-phase model is the right operational story; the "single Lambda deploy" framing was technically true but operationally misleading.**

### Why concurrency=1 instead of 3

Original plan had `concurrency=3` per queue based on memory math (6 × 50 MB = 300 MB working set, fits Railway). Codex (#17-19) caught that memory wasn't the real bottleneck — DB pools, Neo4j sessions, OpenAI rate limits are. Peter approved lowering to 1 with explicit raise-criterion. **The pattern: start conservative for migrations whose primary goal is durability/safety, not throughput. Raise after evidence, not pre-emptively.**

### Why the regression test was split

Original plan had a single regression test with ±20% LLM count tolerance. Codex (#9) caught that ±20% tolerance directly undermines the purity claim — if retries can produce materially different LLM outputs, downstream idempotence isn't established. Split into:

- **Deterministic-mock regression** (T25a, BLOCKING): canned LLM responses, exact equality on Neo4j IDs + Postgres ON CONFLICT keys + EventBridge emissions. This is the actual contract.
- **Live-nondeterminism characterization** (T25b, non-blocking): real OpenAI, ±20% tolerance, schema-exact match. Sanity check, not contract.

The split preserves both rigorous contract testing AND realistic-data sanity checks without conflating them.

### Why deal pipeline kept 9 steps despite Codex #10

Codex #10 (split mixed LLM+write steps) clearly applies to action-item S9/S10. For deal pipeline's `merger.merge_deal` (also LLM+write), the function is called inside `match_merge_loop` which has fail-open per-deal semantics — per-deal exceptions are accumulated, not fatal. So the retry-divergence concern is bounded: if a deal has a partial-write divergence on retry, the operator sees "deal X had an issue" rather than corrupted state.

The function-level refactor (separate LLM construct from Neo4j MERGE inside `merger.merge_deal`) was done as a code-quality improvement independent of DBOS retry safety. Step boundary at the DBOS level stayed single. **This pragmatic asymmetry — full split at action-item, function refactor at deal — is a deliberate choice based on the failure semantic, not an oversight.**

---

## Key artifacts (paths + purposes)

| Path | Purpose |
|---|---|
| `docs/plans/2026-05-20-dbos-migration-execution-plan.md` | **Execution plan** — load-bearing source of truth. RESUMPTION CHECKLIST at top. 40 tasks across 6 phases. |
| `docs/plans/2026-05-20-DBOS-MIGRATION-HANDOFF.md` | **This file** — cross-session handoff with project trajectory + reasoning |
| `~/.gstack/projects/oneilstokeseqrm-action-item-graph/peteroneil-main-design-20260520-065353.md` | **Design doc** — APPROVED via /office-hours. Deeper reasoning behind each decision. |
| `~/.gstack/projects/oneilstokeseqrm-action-item-graph/tasks-eng-review-20260520-091517.jsonl` | **Tasks JSONL** — 40 tasks as machine-readable JSONL for `/autoplan` aggregation |
| `~/.gstack/projects/oneilstokeseqrm-action-item-graph/peteroneil-main-eng-review-test-plan-20260520-091517.md` | **Test plan** — read by `/qa` and `/qa-only` skills |
| `~/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/MEMORY.md` | **Project memory index** — links to all individual memory files |
| `~/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/project_dbos_migration_trajectory.md` | DBOS migration project state |
| `~/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/pattern_dbos_step_decomposition.md` | Pattern: pure compute + pure write step is the right DBOS unit |
| `~/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/pattern_conditional_lock_escape_valve.md` | Pattern: every lock names its revisit conditions |
| `~/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/pattern_data_independence_vs_commutativity.md` | Pattern: input independence ≠ concurrent-write commutativity |
| `/Users/peteroneil/EQ-CORE/eq-synthetic-date-generation/docs/plans/session-15-action-item-graph-investigation-brief.md` | Forensic origin of the migration — DLQ message details |
| `/Users/peteroneil/EQ-CORE/live-transcription-fastapi/services/dbos_runtime.py` | LTF reference: DBOS init + FastAPI lifespan integration |
| `/Users/peteroneil/EQ-CORE/live-transcription-fastapi/services/account_provisioning/workflow.py` | LTF reference: workflow function with `@DBOS.workflow` |
| `/Users/peteroneil/EQ-CORE/live-transcription-fastapi/services/account_provisioning/steps.py` | LTF reference: step functions with `@DBOS.step` retry policies |
| `/Users/peteroneil/EQ-CORE/live-transcription-fastapi/tests/integration/account_provisioning/test_crash_recovery.py` | LTF reference: crash-recovery integration test scaffold |
| `src/action_item_graph/pipeline/pipeline.py` | Refactor target — 12 stages → 14 DBOS steps |
| `src/deal_graph/pipeline/pipeline.py` | Refactor target — 9 stages → 9 DBOS steps (with inner function refactor) |
| `src/action_item_graph/lambda_ingest/handler.py` | Refactor target — replace `submit_to_railway` with `DBOSClient.enqueue` |
| `infra/forwarder.py` + `infra/__main__.py` | Pulumi IaC — add new Secret + IAM permission expansion |

---

## Research artifacts (URLs cited during planning)

For context if the next agent needs to re-verify any claim:

- DBOS Python SDK: https://github.com/dbos-inc/dbos-transact-py
- DBOSClient docs: https://docs.dbos.dev/python/reference/client
- DBOS workflow tutorial: https://docs.dbos.dev/python/tutorials/workflow-tutorial
- DBOS step tutorial: https://docs.dbos.dev/python/tutorials/step-tutorial
- DBOS March 2026 release notes (nested workflows + parent/child indexing): https://www.dbos.dev/blog/dbos-new-features-march-2026
- AWS Lambda Durable Functions (referenced as alternative, rejected): https://codestax.medium.com/the-15-minute-wall-just-came-down-a-guide-to-aws-lambda-durable-functions-6151d3b6dd0b
- AWS Compute Blog on streaming LLM responses (background): https://aws.amazon.com/blogs/compute/serverless-strategies-for-streaming-llm-responses/
- Inngest vs Temporal (referenced as alternative, rejected): https://www.inngest.com/compare-to-temporal
- 2026 orchestration market comparison: https://www.digitalapplied.com/blog/ai-workflow-orchestration-tools-2026-comparison

---

## Coordination across the EQ ecosystem

- **`eq-synthetic-date-generation`:** Originated the DLQ incident that triggered this migration. Independent of this work post-handoff; the synthetic injection can resume once `live-transcription-fastapi` PR #23 (Problem B) is unblocked, which is unrelated to this DBOS migration.
- **`live-transcription-fastapi`:** Reference codebase. **DO NOT TOUCH** during this migration — separate repo, separate agent's work.
- **`eq-email-pipeline`:** Originating ingest pipeline that emits the EnvelopeV1 events. No changes required here for the DBOS migration; the EventBridge → SQS → Lambda chain stays the same up to and including the Lambda handler.
- **`eq-frontend`:** Reads from the same Neon Postgres database. Tables written by action-item-graph (action_items, deals, etc.) keep the same schema. Frontend Prisma schema unaffected.

---

## Resumption signal to next agent

When the next session opens, the user (Peter) will likely say something like "resume DBOS migration per plan, run T1 and T3."

Your sequence should be:

1. **Acknowledge resumption.** Read this handoff doc + the plan's RESUMPTION CHECKLIST + design doc + memory index. ~5–10 min.
2. **Confirm understanding via prose.** State which task you're starting with, the next 2–3 tasks, any deviations from your read of the plan. Wait for greenlight.
3. **Execute T1 + T3 as a paired gate.** T1 creates the feature branch (~5 min). T3 runs the Docker container build to verify `dbos` + `psycopg[binary]` install cleanly into the Lambda Amazon Linux 2 runtime (~30 min). Surface T3 result to Peter — this is the GO/NO-GO decision point for D1 conditional lock.
4. **If T3 succeeds:** proceed to T2 (add `dbos>=2.22.0,<3.0` to pyproject.toml + uv lock + verify install) and the rest of Phase A.
5. **If T3 fails:** stop. Surface to Peter the failure mode + recommendation: fall back to Approach A per plan documented contingency. Wait for greenlight before any further work.

The 2-week rollback window starts when Phase 2 deploys (T29). Phase 3 (T31 — delete /process route) happens Day 14+ as a follow-up deploy in a separate PR.

---

## Known-deferred test failures (pre-existing on main)

`tests/test_lambda_handler.py` and `tests/test_lambda_secrets.py` fail at collection time with `ModuleNotFoundError: No module named 'aws_lambda_powertools'` (and `boto3`). These libraries are installed into the Lambda zip by `scripts/package_lambda.sh` but are NOT in the project's `pyproject.toml` dev/api dependencies, so the local test runner can't import the Lambda handler modules.

This was verified to fail on `main` at commit `71dd9fd` before any Phase A work — it is NOT a Phase A regression. The Lambda tests will be rewritten in **T21 (Phase E)** as part of the dispatcher migration, at which point either the Lambda deps move into a `[lambda]` extras group in pyproject.toml, OR the affected tests get conftest.py-level skip guards.

**For future sessions:** when running `pytest`, use `--ignore=tests/test_lambda_handler.py --ignore=tests/test_lambda_secrets.py` to skip the pre-existing red. The remaining 557 tests pass.

---

## Final note on session hygiene

Peter has explicit feedback-style preferences captured in `feedback_autonomous_execution.md`: "Execute autonomously after plan approval, only stop for out-of-scope work or manual actions needed." That preference applies to this implementation phase. Per the plan's RESUMPTION CHECKLIST, surface to Peter at natural session breaks (post-T3, post-T10/T11, post-Phase E tests, pre-/ship) — not every task completion. Trust the plan; flag deviations.

If you're at a context-pressure threshold mid-implementation, save progress via `/context-save` and propose a clean session transition rather than pushing through. The plan and design doc are durable; session boundaries are flexible.
