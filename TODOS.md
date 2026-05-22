# TODOs

## Phase D Follow-up (Day 14+ post-Phase 2 deploy)

**Retire `/process` HTTP route + legacy code** (T18-T20 from the DBOS migration plan).

**What:** Delete `src/action_item_graph/api/routes/process.py`, `src/dispatcher/dispatcher.py`, and the migration-window-only test files: `tests/test_compatibility_process_route.py`, `tests/test_workflow_behavioral_contract.py`, `tests/test_lambda_handler.py::TestT33DBOSUnreachable`. Also clean up `Pulumi.prod.yaml` config entries: `worker-api-key` (encrypted) + `api-base-url` — Phase C dropped the references but the encrypted-stack config has higher rollback implications and was deferred with Phase D.

**Why:** The 2-week rollback window required `/process` to stay alive during Phase 1+2 deploys. Once the migration is stable post-Phase 2, the legacy path retires.

**When:** Day 14+ post-Phase 2 deploy. Separate PR. See deletion-seam list in v0.3.0 CHANGELOG + `docs/plans/2026-05-21-PHASE-F-PRE-PR-BRIEF.md`.

---

## Phase E Test-Coverage Backlog (P2 follow-up)

Coverage gaps identified by Phase F /review's testing specialist; deferred per the T25a scope-down decision rationale.

- `dbos_runtime.py` dedicated unit tests
- `create_version_snapshot` unit-level idempotency tests (currently asserted via E2E integration only)
- `_extraction_content_hash` helper unit tests
- `merging_persist_step` decision-branch coverage (4 branches: best_match=None / update_status / merge / link_related)
- `match_merge_loop_step` fail-open-per-deal branches
- Merger split-methods (`construct_*_llm` + `persist_*_neo4j`) direct unit tests

T30 (post-deploy DLQ replay) provides production-realistic verification. Address as P2/P3 follow-up test PR.

---

## Maintainability NITs (P3 cleanup follow-up)

Identified by Phase F /review's maintainability specialist; non-blocking readability cleanups. Some may be partially handled by Phase D rewrites.

- Function-local imports in workflow bodies (ValidationError + DealPipelineError) and action_item_steps (ActionItem at 4 sites)
- 7-tuple positional types in S10a (`topic_resolution_llm_step`)
- Magic `0.9` MEDDIC role-match confidence constant
- Near-duplicate `_extraction_content_hash` helpers across action-item + deal mergers (extract to `src/shared/pydantic_content_hash.py`)
- `_build_deal_pipeline` over-coupling for D2/D3/D5 (only needs the repository, not the full pipeline)
- Long step functions: `merging_persist_step` (~80 lines), `match_merge_loop_step` (~155 lines) — decompose into per-branch helpers

---

## HANDOFF §2 Known Limitations (production-monitor watchlist)

Bounded follow-ups documented at Phase B-2 codex absorption. Address if they surface in production.

- **Version counter compare-and-set** — `update_action_item` / `update_deal` could double-increment under DBOS retry. Historical record is correct (idempotent snapshots) but `version` field could drift by +1.
- **Owner CREATE narrow cross-workflow race** — `resolve_or_create_owner` is read-then-CREATE within a single step (retry-safe), but two concurrent workflows could race on the read and produce duplicate Owner nodes (~10ms window). Recoverable via dedup query.
- **S9a/S10a per-step instrumentation asymmetry** — S9a runs only for "merge" decision (~30% of match_results), S9b runs for every match_result. Dashboards should encode the asymmetry; not a code bug.

---

## DLQ CloudWatch Alarm (DEFERRED until post-DBOS-migration)

**What:** Add a CloudWatch alarm on `ApproximateNumberOfMessagesVisible > 0` for `action-item-graph-dlq` with SNS notification.

**Why:** The DLQ silently accumulates failed messages with no alerting. A CloudWatch alarm with SNS notification (email or Slack webhook) ensures the team is notified when messages fail processing 3 times and land in the DLQ.

**Cost:** Near-zero (~$0.10/month for the alarm + SNS).

**How:** Add `aws.cloudwatch.MetricAlarm` and `aws.sns.Topic` resources to `infra/forwarder.py`. Parameters: `threshold=0`, `evaluation_periods=1`, `period=300`, `statistic="Maximum"`, `comparison_operator="GreaterThanThreshold"`.

**Depends on:** Post-DBOS-migration. After migration ships, the DLQ becomes mostly empty (only Lambda-level enqueue failures land there), so the alarm threshold can be tightened. Do not bundle into the DBOS migration PR; keep as separate follow-up.

---

## Investigate DLQ Messages (SUBSUMED by DBOS migration T30)

**Status:** Subsumed. The 1 unique remaining DLQ message (MessageId `58863f20-3cda-48f7-973d-3002aa31331b` — Anthropic security questionnaire, 2026-05-19) is the live integration test for the DBOS migration. It will be redriven via `aws sqs start-message-move-task` in T30 (post-deploy verification) and is expected to process successfully through the new DBOS path.

**Original note:** The earlier "23 messages" reference is stale (SQS retention is 14 days, so most expired). Only 1 message remains as of 2026-05-19 forensic inspection.

**No further work required here** — T30 in the DBOS migration plan handles it.

---

## Completed

### DBOS Migration (Phases A–E + Phase F absorption)

**Completed:** v0.3.0 (2026-05-22)

Migrated the action-item-graph ingest path from synchronous Lambda → Railway `/process` HTTP call to a DBOS-orchestrated durable workflow with per-step retry, checkpointing, and observability. 14 commits, 654 tests passing (+87 vs main). 14 codex rounds absorbed across phases. See CHANGELOG v0.3.0 + `docs/plans/2026-05-20-DBOS-MIGRATION-HANDOFF.md` for the full migration arc. Phase D (retire `/process`) tracked separately at the top of this file.
