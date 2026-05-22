# Slice 2 — EventBridge Producer for `deal.processed` Events

**Date:** 2026-05-22
**Author:** Claude (this session, action-item-graph)
**Status:** DRAFT — awaiting `/plan-eng-review`
**Initiative:** Opportunity Pipeline Restoration (4-slice multi-session arc)
**Slice:** 2 of 4 (producer side, this repo)
**Prerequisite:** Slice 1 (AWS infrastructure, thematic-lm) — DONE 2026-05-22 13:53 UTC
**Upstream canonical handoff:** `~/EQ-CORE/thematic-lm/docs/superpowers/handoffs/2026-05-22-session64-slice1-close.md`

---

## 1. Goal

Add an EventBridge producer that emits **one `deal.processed` event per processed interaction** to the AWS default bus (us-east-1, account 211125681610) whenever the deal pipeline successfully aggregates `deals_created`/`deals_merged` lists. The event flows: producer → EventBridge rule (`opp-themes-eq-opportunity-themes-rule`, deployed in Slice 1) → SQS (`eq-opportunity-queue-dev`) → Lambda (`eq-opportunity-themes-ingest`) → thematic-lm `/analyze` with `scope_type='opportunity'`.

**Ship dark.** Deploy with `ENABLE_DEAL_PROCESSED_EVENTS=false`. Slice 3 flips the flag in a separate session.

---

## 2. Deviation from upstream handoff (load-bearing)

The canonical Slice 2 prompt embedded in `2026-05-22-session64-slice1-close.md` was authored **the same day the DBOS migration landed in action-item-graph** (PR #14, commit `89351a3`, merged 2026-05-22). The handoff was written against pre-DBOS code structure that no longer matches production.

**What the upstream handoff says:**
- Hook at `deal_pipeline.complete` log site in `src/action_item_graph/pipeline/pipeline.py`, after "step 11 agent_outbox write."

**Why that's wrong in current `main`:**

| Claim | Reality |
|---|---|
| `deal_pipeline.complete` lives in `src/action_item_graph/pipeline/pipeline.py` | No — it lives in `src/deal_graph/pipeline/pipeline.py:506`. The action-item pipeline file has no `deal_pipeline.*` log strings. |
| Hook sits after the agent_outbox write | Architecturally confused — `agent_outbox` is the action-item pipeline's writer (hardcoded `source_pipeline='action_item'` at `pipeline.py:1172`). It has no structural relationship to deal completion. |
| There is one `pipeline.py` to grep | No — deal orchestration runs through `src/deal_graph/workflows/deal_workflow.py` (DBOS workflow), enqueued by the Lambda at `src/action_item_graph/lambda_ingest/handler.py`. |

**Additional finding during eng review (architectural):** The dispatcher (`src/dispatcher/dispatcher.py`) — initially considered as the v1 hook in this plan — is **dead in the production traffic path**. The Lambda handler bypasses it entirely (enqueues DBOS workflows directly per `lambda_ingest/handler.py` module docstring), and the dispatcher is only invoked by the legacy `/process` HTTP route in `src/action_item_graph/api/routes/process.py:53`. That route is on the retirement list at `TODOS.md` line 7 and gets deleted in the Phase D follow-up PR. A dispatcher hook would silently no-op on Slice 3 flag flip — production Lambda traffic would never trigger it.

**Resolution:** v1 hooks inside the DBOS workflow as a `@DBOS.step` (formerly documented as "Option B / v2 hardening" in this plan's earlier draft, promoted to v1 after the dispatcher dead-code discovery). Rationale below in Section 4. The thematic-lm canonical handoff will be patched in the Slice 3 sync when Slice 2 reports complete; the corrected hook-point paragraph is captured at the end of this doc.

---

## 3. Scope

### In scope (this slice)
- New module `src/deal_graph/clients/event_publisher.py` (~90 LOC). Pure helper — no DBOS decorator on the helper itself; the helper is wrapped at the call site so the publish logic stays testable without DBOS in the loop.
- New `@DBOS.step` wrapper `publish_deal_processed_step` defined in `src/deal_graph/workflows/deal_steps.py` (alongside the other workflow steps so DBOS discovers it at package init).
- Wire the step into `src/deal_graph/workflows/deal_workflow.py` — single call inside `run_deal_workflow` right before the `'deal_workflow.complete'` log line at the current line ~146, with `deals_created` / `deals_merged` already aggregated above.
- Move `boto3>=1.34.0` from `[project.optional-dependencies] lambda` to base `dependencies` in `pyproject.toml` (Railway runtime needs it).
- New `tests/test_event_publisher.py` (5 cases on the pure helper).
- Two new tests in `tests/test_deal_workflow.py` verifying the workflow integration (step called with right args on non-empty deal lists; step not called when both lists empty).
- Railway env vars on the action-item-graph production service: `AWS_REGION=us-east-1`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `ENABLE_DEAL_PROCESSED_EVENTS=false`.
- Production deploy with feature flag **OFF**.
- 6-step verification protocol (Section 8).

### Out of scope (explicitly deferred)
- **Slice 3:** End-to-end verification by flipping the flag and observing a real transcript ingest. Founder coordinates this in a separate session.
- **Slice 4:** April 4 P2 `contributing_account_ids` resolution merge bug in `thematic-lm/src/thematic_lm/orchestrator/nodes.py:~1864`.
- **Hook in the legacy dispatcher (`src/dispatcher/dispatcher.py`)**: dead in production Lambda path; on retirement list. Belt-and-suspenders dual-hook was considered and rejected.
- CloudWatch metrics on publish success/failure (structlog log lines + DBOS step instrumentation are enough for v1).
- EventBridge rate-limit / PutEvents quota handling (well under any practical limit at expected deal-merge volume).
- Refactoring `agent_action_outbox` writer to also emit deal events (separate pattern; orthogonal).
- Fixing the `agent_action_outbox` `NotNullViolation` on `interaction_links.id` (tracked elsewhere, explicit founder constraint).

---

## 4. Hook-point decision: DBOS step inside `deal_workflow` (production path)

**Choice:** Wrap `publish_deal_processed` as a `@DBOS.step` named `publish_deal_processed_step` (defined alongside the other workflow steps in `src/deal_graph/workflows/deal_steps.py`), then call it from `run_deal_workflow` in `deal_workflow.py` immediately before the `'deal_workflow.complete'` log call. At that point, `deals_created` and `deals_merged` are already aggregated lists ready to ship.

**Why this layer (and not the dispatcher):**

The Lambda handler is the production entry point. Per `src/action_item_graph/lambda_ingest/handler.py` module docstring: *"Replaces the prior synchronous HTTP forward to Railway /process with two DBOSClient.enqueue calls (one per pipeline). Durability ownership transfers from SQS to DBOS at the moment Lambda returns 200."* Concretely:

```
SQS → Lambda (action-item-graph-ingest)
        ├─ DBOSClient.enqueue("action-item-pipeline", action_item_workflow, ...)
        └─ DBOSClient.enqueue("deal-pipeline",       deal_workflow,        ...)
            └─ DBOS worker (Railway) → run_deal_workflow → ★ HOOK HERE ★ → deal_workflow.complete
```

`src/dispatcher/dispatcher.py` is only invoked by the legacy `POST /process` HTTP route (`src/action_item_graph/api/routes/process.py:53`), which is on the retirement list in `TODOS.md` line 7. Production Lambda traffic bypasses it entirely. A dispatcher hook would silently no-op on Slice 3 flag flip.

**Properties of the DBOS-step placement:**

1. **Lives in the production code path.** Every real envelope processed by the Lambda runs through `run_deal_workflow`. Whatever we hook here will fire for every real ingest. No path-of-no-return ambiguity.

2. **At-least-once delivery via DBOS durability.** DBOS persists the step-invocation intent before execution. If the worker crashes between AWS-side acceptance and DBOS-side step-completion bookkeeping, the workflow resumes on a new worker and the step re-fires. The consumer is already keyed on `opp-ingest:<deal_id>:<interaction_id>` and dedupes via the `analyses` table's idempotency_key UNIQUE constraint, so the replay is safe (see Section 7).

3. **Step-internal failure does NOT propagate to workflow.** The step body wraps `publish_deal_processed` in a broad `try/except Exception`. On AWS error we structlog-warn and return None. The workflow continues uninterrupted to `'deal_workflow.complete'`. We never raise from inside the step because a raise would cost DBOS retry budget on the entire workflow — much worse than a missing EventBridge event.

4. **Trivial feature flag.** The flag check lives inside `publish_deal_processed` (the pure helper), not inside the step. `@DBOS.step` always fires; the step body short-circuits on flag-off. This keeps DBOS observability (step-invocation count, latency) consistent regardless of flag state and makes the flag flip a pure env-var operation with no workflow-registration concerns.

**Workflow-replay double-publish:** explicitly accepted and absorbed by consumer-side idempotency. Documented in Section 7.

**Why not dual-hook (dispatcher + DBOS step):** dispatcher is two-weeks-from-deletion; hook there would be deleted in the same Phase D follow-up PR. Not worth the noise.

---

## 5. Wire contract (locked, do not invent fields)

Source-of-truth parser: `~/EQ-CORE/thematic-lm/src/thematic_lm/opportunity/event_models.py:DealProcessedEvent.from_sqs_body`.

```
EventBridge default bus, us-east-1, account 211125681610
  Source:     "com.eq.action-item-graph"
  DetailType: "deal.processed"
  Detail (JSON):
    {
      "tenant_id":      "<uuid>",                   # REQUIRED — parser raises ValueError if missing
      "account_id":     "<uuid or empty string>",   # optional
      "interaction_id": "<uuid>",                   # REQUIRED — parser raises ValueError if missing
      "deals_created":  ["<deal_uuid>", ...],
      "deals_merged":   ["<deal_uuid>", ...],
      "timestamp":      "<ISO-8601 UTC>",
      "source":         "deal-pipeline"
    }
```

Field-name discipline: parser uses `detail["tenant_id"]` / `detail["interaction_id"]` (KeyError if missing). Variations like `tenantId`, `interactionUuid`, `created_deals` will fail at the consumer.

---

## 6. File-by-file change spec

### 6.1 `pyproject.toml`

Move `boto3>=1.34.0` from `[project.optional-dependencies] lambda` to base `dependencies`. Rationale: Railway's production deploy uses nixpacks auto-detection from `pyproject.toml`'s base deps (no Dockerfile, Procfile is `web: uvicorn ...`). The `lambda` extra is documented as "Bundled into the Lambda zip by scripts/package_lambda.sh, NOT installed into the runtime image." Leaving boto3 in the lambda extra would mean Railway tries to import `boto3` at producer call time and crashes with `ModuleNotFoundError`.

Diff:
```toml
 dependencies = [
     ...existing...
     "psycopg[binary]>=3.2.0",
+    "boto3>=1.34.0",
 ]

 [project.optional-dependencies]
 lambda = [
-    "aws-lambda-powertools[tracer]>=3.0.0",
-    "boto3>=1.34.0",
-    "pydantic-settings>=2.0.0",
+    "aws-lambda-powertools[tracer]>=3.0.0",
+    "pydantic-settings>=2.0.0",
 ]
```

Note: boto3 stays available to the Lambda zip via the base deps; no change to `scripts/package_lambda.sh` needed. The lambda extra still exists for `aws-lambda-powertools` and `pydantic-settings` which are not needed in Railway.

### 6.2 `src/deal_graph/clients/event_publisher.py` (NEW)

Pure helper. NOT decorated with `@DBOS.step` — the decorator goes on the thin wrapper in `deal_steps.py` (Section 6.3), so the publish logic stays unit-testable without DBOS context. The helper:

- Reads the feature flag on every call (`os.getenv` — lets Slice 3 flip via Railway restart, no module reload).
- Short-circuits on flag-off, empty interaction_id, or both-empty deal lists.
- Builds a `boto3.client("events")` (sync), calls `put_events`, parses the response.
- Wraps everything in a broad `try/except Exception` and logs warnings. **Never raises** — see Section 4 property #3 for why raising from inside the step would be worse than dropping an event.

```python
"""EventBridge publisher for deal.processed events.

Emits a single ``deal.processed`` event to the AWS EventBridge default bus
when ``ENABLE_DEAL_PROCESSED_EVENTS=true``. Never raises — failures log
warnings and return None so the deal workflow continues uninterrupted.

Wire contract source of truth:
    thematic-lm/src/thematic_lm/opportunity/event_models.py
    (DealProcessedEvent.from_sqs_body)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


_SOURCE = "com.eq.action-item-graph"
_DETAIL_TYPE = "deal.processed"


def _events_enabled() -> bool:
    return os.getenv("ENABLE_DEAL_PROCESSED_EVENTS", "false").lower() == "true"


def publish_deal_processed(
    *,
    tenant_id: str,
    account_id: str,
    interaction_id: str,
    deals_created: list[str],
    deals_merged: list[str],
    source: str = "deal-pipeline",
    timestamp: str | None = None,
    workflow_id: str | None = None,
) -> None:
    """Publish a deal.processed event to EventBridge. Never raises.

    No-ops when the feature flag is off, when interaction_id is empty
    (consumer requires it), or when both deal lists are empty (consumer
    would skip). ``workflow_id`` is included in log fields for tracing.
    """
    if not _events_enabled():
        return
    if not interaction_id:
        logger.warning(
            "event_publisher.skipped_no_interaction_id",
            workflow_id=workflow_id,
        )
        return
    if not deals_created and not deals_merged:
        return

    log = logger.bind(
        interaction_id=interaction_id,
        workflow_id=workflow_id,
    )

    try:
        import boto3
        client = boto3.client("events", region_name=os.getenv("AWS_REGION", "us-east-1"))
        detail = {
            "tenant_id": tenant_id,
            "account_id": account_id,
            "interaction_id": interaction_id,
            "deals_created": deals_created,
            "deals_merged": deals_merged,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        response = client.put_events(
            Entries=[
                {
                    "Source": _SOURCE,
                    "DetailType": _DETAIL_TYPE,
                    "Detail": json.dumps(detail),
                }
            ]
        )
        failed = response.get("FailedEntryCount", 0)
        entries = response.get("Entries", [])
        event_id = entries[0].get("EventId") if entries else None
        if failed:
            log.warning(
                "event_publisher.put_events_partial_failure",
                failed_count=failed,
                entries=entries,
            )
        else:
            log.info(
                "event_publisher.published",
                event_id=event_id,
                deals_created=len(deals_created),
                deals_merged=len(deals_merged),
            )
    except Exception as e:
        log.warning(
            "event_publisher.failed",
            error=str(e),
            error_type=type(e).__name__,
        )
```

### 6.3 `src/deal_graph/workflows/deal_steps.py` (ADD STEP)

Add a thin `@DBOS.step`-decorated wrapper at the end of the file. The wrapper resolves the current workflow ID via `DBOS.workflow_id`, then forwards to the pure helper. The wrapper exists for two reasons:

1. **DBOS instrumentation surface.** With `@DBOS.step`, DBOS records this side-effect in its workflow history. Step-invocation counts and durations show up in DBOS observability the same way the existing 12+ steps do (`deal_extraction_step`, `match_merge_loop_step`, etc.).
2. **Workflow-id propagation.** Inside a DBOS workflow, `DBOS.workflow_id` returns the current workflow's ID. Capturing it in log fields lets us trace which workflow run produced which EventBridge event (useful for the Slice 3 verification round-trip).

Diff (matching the existing fail-open step convention used by D7/D8/D9 in this file):
```python
# Add at the top of the file (with the other imports):
from deal_graph.clients.event_publisher import publish_deal_processed

# Note: `DBOS` is already imported at deal_steps.py:45.

# Add at the END of the file:

@DBOS.step(retries_allowed=False)
def publish_deal_processed_step(
    *,
    tenant_id: str,
    account_id: str,
    interaction_id: str,
    deals_created: list[str],
    deals_merged: list[str],
) -> None:
    """D10: emit deal.processed to EventBridge. Fail-open, no retries.

    Mirrors the D8/D9 (enrich_interaction, postgres_dual_write) pattern:
    ``retries_allowed=False`` because the underlying helper NEVER raises
    (broad try/except inside ``publish_deal_processed``). Letting DBOS
    retry on raise would re-do the whole workflow — much worse than a
    missing EventBridge event.

    Replay semantics: if the worker crashes between AWS-side acceptance
    and DBOS-side step-completion bookkeeping, the workflow resumes on a
    new worker and this step re-fires. Consumer dedupes via
    ``analyses.idempotency_key`` UNIQUE on
    ``f"opp-ingest:{deal_id}:{interaction_id}"``.
    """
    publish_deal_processed(
        tenant_id=tenant_id,
        account_id=account_id,
        interaction_id=interaction_id,
        deals_created=deals_created,
        deals_merged=deals_merged,
        source="deal-pipeline",
        workflow_id=DBOS.workflow_id,
    )
```

Notes:
- `retries_allowed=False` matches the codebase's existing fail-open side-effect pattern (`deal_steps.py:145, 243, 405, 443`). Helper never raises, so retries would only fire if our wrapper itself raises — which would be a code bug, not a transient failure.
- `DBOS` is already imported at `deal_steps.py:45` (`from dbos import DBOS`) — no new import needed for the decorator.
- `DBOS.workflow_id` is the recommended way to fetch the current workflow ID from within a step (per dbos>=2.22 API). If pyright objects at implementation time, the alternative is `DBOS.workflow_id()` as a method call — adjust based on the installed package's stubs.

**Where the worker runs:** Per `src/action_item_graph/dbos_runtime.py:120` (`DBOS.launch()`), the DBOS worker threads live inside the FastAPI process — both started by `Procfile`'s `web: uvicorn action_item_graph.api.main:app`. Section 6.1's boto3 base-dep addition therefore covers both the API and worker code paths in a single Railway service. No separate worker deployment needed.

### 6.4 `src/deal_graph/workflows/deal_workflow.py` (ADD CALL SITE)

Single new call to `publish_deal_processed_step` right before the `'deal_workflow.complete'` log line at the current line ~146. At that point both `deals_created` (list[str]) and `deals_merged` (list[str]) are already aggregated and ready to ship.

Diff:
```python
 from deal_graph.workflows.deal_steps import (
     ...existing imports...
+    publish_deal_processed_step,
 )

 ...

     # D9 — postgres_dual_write (fail-open)
     await deal_postgres_dual_write_step(envelope_dict, merge_results_list, interaction_id)

+    # D10 — emit deal.processed to EventBridge (feature-flag-gated, fail-open).
+    # Wrapped as a DBOS step so the replay-on-crash semantics are explicit;
+    # consumer dedupes via the analyses idempotency_key.
+    publish_deal_processed_step(
+        tenant_id=str(envelope_dict.get("tenant_id", "")),
+        account_id=str(envelope_dict.get("account_id") or ""),
+        interaction_id=interaction_id,
+        deals_created=deals_created,
+        deals_merged=deals_merged,
+    )

     logger.info(
         'deal_workflow.complete',
         interaction_id=interaction_id,
```

Notes:
- The step is invoked synchronously (no `await`) — DBOS sync steps are the norm in this file (`deal_postgres_dual_write_step` above it uses `await`; `publish_deal_processed_step` is sync because the underlying helper is sync). Pattern-match against `deal_steps.py`'s other steps at implementation time and align.
- `envelope_dict.get("tenant_id", "")` defensive: envelope is dict-form inside the DBOS workflow per the existing pattern (see lines 107–109 of the workflow).
- The early-finalize path (`if not extraction_result_dict.get('has_deals')` block at lines 114–130) returns BEFORE this hook, so empty-extractions don't emit. That's correct — no deals to announce.

### 6.5 `tests/test_event_publisher.py` (NEW)

Five test cases on the pure helper (no DBOS required):

1. **`test_publish_constructs_correct_payload`** — mock `boto3.client` via `unittest.mock.patch`; set flag on; call publisher; assert `put_events` was called once with `Source="com.eq.action-item-graph"`, `DetailType="deal.processed"`, and `Detail` JSON parsed back equals the expected dict shape (all 8 fields present, correct types — includes `workflow_id` when supplied).
2. **`test_feature_flag_off_does_not_publish`** — flag unset (default false); patch `boto3.client` to assert it's never called; verify publisher returns None.
3. **`test_publish_failure_does_not_raise`** — flag on; mock `boto3.client` to return a client whose `put_events` raises `ClientError`; verify publisher returns None and logged a warning.
4. **`test_empty_deals_skipped`** — flag on; pass `deals_created=[]`, `deals_merged=[]`; assert `boto3.client` not called.
5. **`test_partial_failure_logged`** — flag on; mock `put_events` to return `{"FailedEntryCount": 1, "Entries": [{"ErrorCode": "InternalFailure"}]}`; verify warning logged, no raise.

Use `unittest.mock.patch` (already the project's pattern). No `moto` — heavier dep, not needed.

### 6.6 `tests/test_deal_workflow.py` (EXTEND)

Two new tests that patch the publish step at the function-reference level inside `deal_steps`, intercepting the call from `run_deal_workflow`:

1. **`test_publish_step_called_with_aggregated_deals`** — patch `deal_graph.workflows.deal_steps.publish_deal_processed`; run a workflow path that yields non-empty `deals_created`/`deals_merged` (use existing fixtures from the file); assert publish helper called once with the aggregated lists, the right `interaction_id`, `tenant_id`, `account_id`, and a non-empty `workflow_id`.
2. **`test_publish_step_not_called_when_no_deals`** — drive the early-finalize path (no deals extracted) OR a path where both `deals_created` and `deals_merged` are empty; assert publish helper NOT called.

Patching at the `publish_deal_processed` (pure helper) reference inside `deal_steps` — NOT at `publish_deal_processed_step` — means we don't have to invoke DBOS for these tests. The step wrapper calls through, the helper is mocked, the assertion is on the helper call.

If patching the step itself proves cleaner (e.g., to assert `@DBOS.step` is exercised), an alternative is patching `deal_graph.workflows.deal_workflow.publish_deal_processed_step` directly. Decide at test-implementation time.

---

## 7. Failure-mode summary

| Failure | What happens | Observability | Workflow impact |
|---|---|---|---|
| Flag off | Helper returns immediately | No log line | None — step body is no-op |
| `interaction_id` empty/None | Helper logs warning + returns | `event_publisher.skipped_no_interaction_id` | None |
| Both deal lists empty | Helper returns immediately | No log line | None |
| boto3 import fails (shouldn't happen after 6.1) | Caught by helper's outer `try/except` | `event_publisher.failed` warning | None |
| AWS auth failure (bad creds) | `ClientError` from `put_events`, caught | `event_publisher.failed` warning | None |
| EventBridge `FailedEntryCount > 0` | Logged as warning, helper returns None | `event_publisher.put_events_partial_failure` | None |
| Network timeout | Caught by helper's outer `try/except` | `event_publisher.failed` warning | None |
| **DBOS workflow retries between step bookkeeping and AWS ack (crash window)** | Step re-fires on workflow resume; consumer receives a duplicate SQS message | Two `event_publisher.published` log lines with different `event_id` but same `interaction_id` + `workflow_id` | None at producer; consumer dedupes (see below) |
| Helper itself raises (defensive — should never happen) | DBOS would record step failure; DBOS retries the workflow | DBOS step-failure metric | Workflow retries the deal pipeline work — **bad**. Helper's outer `try/except Exception` is the guard. |

### Workflow-replay double-publish — explicitly accepted

DBOS `@DBOS.step` persists the step-invocation intent to the system database BEFORE executing the step body, and records completion AFTER. If the worker crashes inside `put_events` after AWS has accepted the event but before DBOS records completion, the workflow resumes on a new worker and the step re-fires. The same EventBridge event is published twice with different `event_id` values.

The consumer-side `lambda_opportunity_ingest` Lambda POSTs `/analyze` with `idempotency_key = f"opp-ingest:{deal_id}:{interaction_id}"`. The `analyses` table has a UNIQUE constraint on `idempotency_key`. The second POST therefore no-ops at the database constraint layer:

```
EventBridge event #1 → SQS msg → Lambda → POST /analyze (idempotency_key=K) → analyses row created
EventBridge event #2 → SQS msg → Lambda → POST /analyze (idempotency_key=K) → UNIQUE violation caught → no-op
```

Net effect of workflow replay: two SQS messages, two Lambda invocations, ONE `analyses` row, ONE downstream codebook generation. Wasted CPU on the duplicate Lambda invocation, but zero data integrity impact. The consumer Lambda's metric `MessagesSucceeded` would tick twice; `IngestSubmitted` would tick twice; `analyses` row count ticks once. Acceptable trade-off for at-least-once delivery.

---

## 8. Verification protocol (6 steps from the handoff, adapted)

### a. Unit tests pass
Run `uv run pytest tests/test_event_publisher.py tests/test_dispatcher.py -v`. All new tests green; existing 458+ tests still pass.

### b. Static check
Run `uv run pyright src/deal_graph/clients/event_publisher.py src/dispatcher/dispatcher.py`. Zero errors on new code.

### c. Deploy to Railway with flag OFF
Set env vars (see Section 9), trigger deploy via Railway MCP (or via git push if the user prefers — flag is asked of the founder separately). Wait for Active state.

### d. Smoke-test the deploy is healthy
Tail `mcp__railway__deployment_logs` for the new deployment. Look for:
- No `ImportError: No module named 'boto3'`
- No `ImportError: cannot import name 'publish_deal_processed'`
- No `NameError` / `AttributeError` from dispatcher changes
- One real envelope processing through the system (or check existing recent logs) shows `dispatcher.deal_complete` and `dispatcher.complete` logs as before
- Critically: no startup crashes

### e. Confirm AWS creds work without firing the chain
**Adapted from handoff:** the handoff suggests "from a Railway shell or via temporary diagnostic endpoint." Railway shell access isn't trivially available via MCP, and adding a temporary diagnostic endpoint is extra scope. Instead:
- Verify env vars are set (KEYS-only listing via `mcp__railway__list_service_variables` — values redacted)
- Trust that any auth misconfiguration surfaces when Slice 3 flips the flag, since the publisher's `event_publisher.failed` warning would fire on first put_events attempt
- This is acceptable because: (1) we ship dark, (2) failure is observable on flag flip, (3) failure mode is log-only

If `/plan-eng-review` insists on stronger pre-flight cred verification, alternative: add a tiny `boto3.client("events", ...).describe_event_bus(Name="default")` call inside an `/admin/aws-check` endpoint, gated by the same flag-off-still-allowed pattern. Defer unless required.

### f. DO NOT flip the flag
`ENABLE_DEAL_PROCESSED_EVENTS=false` stays on production. Slice 3 (founder-coordinated) flips it after end-to-end visibility is in place.

---

## 9. Railway env vars + IAM scope

Project `6b6205f8-a838-4c1a-8f18-de29df9fa695`, service `abf7b1cd-9783-4b4b-bee7-da3d9bfb13da`, environment `production`.

Variables to set (4 total):
- `AWS_REGION=us-east-1`
- `AWS_ACCESS_KEY_ID=<from founder>`
- `AWS_SECRET_ACCESS_KEY=<from founder>`
- `ENABLE_DEAL_PROCESSED_EVENTS=false`

**IAM scope (founder action, not Claude):** create a scoped IAM user (e.g., `aig-eventbridge-producer`) with this inline policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "events:PutEvents",
    "Resource": "arn:aws:events:us-east-1:211125681610:event-bus/default"
  }]
}
```
DO NOT reuse the founder's admin IAM user. If the scoped user doesn't exist, founder creates it before Slice 2 deploy.

---

## 10. Blast-radius assessment

| Asset | Touched? | Why safe |
|---|---|---|
| Action item pipeline + action_item_workflow | No | Different package tree; entirely independent |
| Deal pipeline (legacy `DealPipeline.run_pipeline`) | No | Not in the DBOS workflow path; we're adding a step to the workflow, not the legacy code |
| `run_deal_workflow` | Yes (additive) | One new call before the existing `deal_workflow.complete` log. Step is fail-open by design. |
| `deal_steps.py` | Yes (additive) | New step definition at end of file; alongside 12+ existing steps following the same `@DBOS.step` pattern |
| Lambda ingest (inbound side) | No | Different module tree; `lambda_ingest/handler.py` untouched |
| Dispatcher (`src/dispatcher/dispatcher.py`) | No | Intentionally untouched — slated for retirement |
| `agent_action_outbox` writer | No | Different module; explicit founder constraint |
| Existing deal_workflow tests | No regression expected | New tests are additive; existing tests don't import the publisher module |
| Existing test suite (458+ tests) | No regression expected | Boto3 base-dep addition only adds a transitive package; doesn't change runtime behavior of any existing code |

**Net assessment with flag OFF:** Each `run_deal_workflow` invocation now incurs one extra `@DBOS.step` registration boundary (DBOS persists step intent → calls helper → helper short-circuits on flag check → returns). DBOS step persistence is ~5ms overhead on Neon. At expected production volume (handful of deal-merging interactions per hour), this is invisible. The blast radius for the flag-off deploy is one new DBOS-step row per workflow run in `dbos.workflow_steps` table — observable, harmless.

**Net assessment with flag ON (Slice 3 flip):** Each successful deal workflow emits one EventBridge event. AWS auth/network errors log a warning and continue. Workflow-replay (crash inside step) double-publishes; consumer dedupes via UNIQUE constraint on idempotency_key. No data-integrity risk.

---

## 11. Future hardening notes (post-Slice-3)

(This section was previously titled "v2 hardening: DBOS step wrap." That work is now in scope as v1 — see Section 4. The remaining future-work items below are smaller polish opportunities, not blockers.)

1. **CloudWatch metric on publish outcomes.** v1 ships with structlog warnings only. If Slice 3 reveals AWS-side flakiness, add a `MetricData` emit on success/failure to `ActionItemGraph/DealEventPublisher` namespace. Useful for an alarm on sustained failure rate.

2. **Step-level retry configuration.** DBOS step defaults to no automatic retry on raised exceptions. We never raise from the helper, so this never trips for v1. If we ever change failure semantics to "transient AWS errors should retry the step," configure `@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=2)`. Right now: not needed.

3. **Idempotency at the producer side.** If consumer dedupe ever becomes insufficient (e.g., consumer-side schema change loosens the UNIQUE constraint), add a producer-side `idempotency_key` to the EventBridge Detail and have the consumer prefer it. Not needed for v1.

4. **Alarm on DBOS step failure rate.** Standard DBOS workflow observability already exposes step-level success/failure counts. A dashboard query against `dbos.workflow_steps WHERE step_name='publish_deal_processed_step' AND status='error'` gives the failure-rate signal. Set up after Slice 3 has produced enough volume for a baseline.

5. **boto3 client lifecycle (codex #4).** v1 constructs a fresh `boto3.client("events", ...)` per publish. boto3 client objects are not thread-safe by default; per-call construction is the safe pattern. At expected volume (handful of emits/hour), the ~10ms per-call setup is negligible. If volume grows materially, cache a module-level client behind a lock or use boto3's internal session pool.

6. **Real-parser contract test (codex #5).** v1 tests assert the producer's own JSON shape against expected fields. The actual `DealProcessedEvent.from_sqs_body` parser lives in a different repo (thematic-lm) and parsing it from action-item-graph would require either vendoring a fragment or cross-repo CI coordination. Mitigated for now by (a) the Slice 1 synthetic smoke test which exercised the parser end-to-end on 2026-05-22 against real data, and (b) Slice 3 will exercise the full chain with real-traffic before flag-flip declaration. A future cross-repo CI test could pin the parser shape with a JSON-schema snapshot.

7. **End-to-end DBOS-step test (codex #7).** v1 mocks `publish_deal_processed_step` in the workflow test and asserts the workflow → step interface. The step's internal `asyncio.to_thread` offload and `DBOS.workflow_id` capture are NOT directly tested. Adding a direct test would require spinning up DBOS at test time (initializing the system-database connection). Trade-off chose v1 simplicity; cover this via Slice 3 real-traffic verification when the step actually runs inside a DBOS-launched worker.

8. **Log-warning assertions (codex #6).** v1 tests verify no-raise and boto3-not-called behavior on the four short-circuits, but don't assert the warning log fields (`event_publisher.skipped_no_interaction_id`, etc.). Add `caplog`-based assertions if log-level alarms become production signals.

---

## 12. Effort estimate

| Step | Estimated minutes |
|---|---|
| Plan doc (this) | 30 (done) |
| `/plan-eng-review` + address findings | 15–30 |
| Code: pyproject.toml + event_publisher.py + dispatcher.py | 30 |
| Tests: test_event_publisher.py + test_dispatcher.py extension | 45 |
| Local pytest + pyright | 10 |
| Codex review + address findings | 30 |
| Surface diff to founder + commit + push | 15 |
| Railway env setup + trigger deploy | 15 |
| 6-step verification | 30 |
| Final report | 15 |
| **Total** | **~3.5–4 hours** |

---

## 13. Success criteria

This slice is "done" when ALL of the following are true:

1. New file `event_publisher.py` exists, with the 4 short-circuits (flag off, no interaction_id, empty deals, top-level try/except)
2. Dispatcher modified; new tests pass; existing tests still pass (zero regressions)
3. `pyproject.toml` updated; `uv sync` reproduces the lockfile cleanly
4. `pyright src/` reports zero new errors
5. `/plan-eng-review` verdict is CLEAR (or CLEAR_WITH_CONCERNS after addressing P0/P1)
6. `codex review` verdict has no unaddressed P0/P1 against the diff
7. Railway production has the 4 env vars set, with `ENABLE_DEAL_PROCESSED_EVENTS=false`
8. Latest Railway deployment is Active with no startup errors in logs
9. Founder has explicitly approved the commit + push (per project CLAUDE.md rule)
10. Final report delivered with corrected hook-point paragraph for upstream handoff sync

A partial state (e.g., env vars set but deploy crashed; tests passing but no review run) is NOT done — it's an incomplete apply that needs investigation.

---

## 14. Corrected hook-point paragraph (for upstream handoff sync)

When reporting Slice 2 complete, include this paragraph so the canonical `2026-05-22-session64-slice1-close.md` in thematic-lm can be patched in place of the current "grep `deal_pipeline.complete` in pipeline.py" instructions:

> **Hook point (post-DBOS migration, current action-item-graph `main`):** The production traffic path runs through `src/action_item_graph/lambda_ingest/handler.py`, which enqueues two DBOS workflows per envelope (`action_item_workflow` and `deal_workflow`) and returns. The legacy `EnvelopeDispatcher` at `src/dispatcher/dispatcher.py` is invoked only by the `POST /process` HTTP compatibility route and is on the retirement list at `TODOS.md` line 7 (deletes in the Phase D follow-up PR). Hooking the dispatcher would silently no-op on real Lambda traffic. **Correct hook for Slice 2:** define a `@DBOS.step`-wrapped `publish_deal_processed_step` in `src/deal_graph/workflows/deal_steps.py` (alongside the other workflow steps), and call it from `run_deal_workflow` in `src/deal_graph/workflows/deal_workflow.py` immediately before the `'deal_workflow.complete'` log line (current line ~146). At that point `deals_created` and `deals_merged` are both aggregated `list[str]` ready to ship. DBOS persists step-invocation intent, so on worker crash the workflow resumes and the step re-fires; consumer-side idempotency (`opp-ingest:<deal_id>:<interaction_id>` as `analyses.idempotency_key` UNIQUE) absorbs the replay. The publish helper itself is the pure boto3 wrapper at `src/deal_graph/clients/event_publisher.py` — feature-flag-gated by `ENABLE_DEAL_PROCESSED_EVENTS=true`, log-and-continue failure mode, never raises. Disregard the prior "step 11 agent_outbox" reference: `agent_action_outbox` is the action-item pipeline's writer (`source_pipeline='action_item'` hardcoded at `pipeline.py:1172`), orthogonal to deal events.

---

## Implementation Tasks
Synthesized from this review's findings. Each task derives from a specific finding above.

- [ ] **T1 (P1, human: ~10min / CC: ~2min)** — pyproject — move boto3>=1.34.0 from `[project.optional-dependencies] lambda` to base `dependencies`; re-lock with `uv lock`
  - Surfaced by: Architecture — boto3 not in Railway runtime (Section 6.1)
  - Files: `pyproject.toml`, `uv.lock`
  - Verify: `uv pip install -e .` succeeds; `python -c 'import boto3'` works
- [ ] **T2 (P1, human: ~30min / CC: ~5min)** — clients — create `src/deal_graph/clients/event_publisher.py` per Section 6.2 spec (pure helper, 4 short-circuits, log-and-continue, never raises)
  - Surfaced by: Section 6.2
  - Files: `src/deal_graph/clients/event_publisher.py` (new), `src/deal_graph/clients/__init__.py` if it needs an export
  - Verify: `tests/test_event_publisher.py` passes
- [ ] **T3 (P1, human: ~20min / CC: ~5min)** — workflows — add `publish_deal_processed_step` @DBOS.step in `deal_steps.py` (retries_allowed=False, fail-open, forwards workflow_id)
  - Surfaced by: Architecture Section 1 finding — dispatcher dead in production, pivot to DBOS step
  - Files: `src/deal_graph/workflows/deal_steps.py`
  - Verify: Lints; workflow imports cleanly
- [ ] **T4 (P1, human: ~10min / CC: ~2min)** — workflows — wire `publish_deal_processed_step(...)` call into `run_deal_workflow` immediately before `'deal_workflow.complete'` log
  - Surfaced by: Section 6.4
  - Files: `src/deal_graph/workflows/deal_workflow.py`
  - Verify: `tests/test_deal_workflow.py` passes including the 2 new cases
- [ ] **T5 (P1, human: ~45min / CC: ~10min)** — tests — write `tests/test_event_publisher.py` (5 cases per Section 6.5) + extend `tests/test_deal_workflow.py` (2 cases per Section 6.6)
  - Surfaced by: Test coverage diagram, gaps section
  - Files: `tests/test_event_publisher.py` (new), `tests/test_deal_workflow.py` (extend)
  - Verify: `uv run pytest tests/test_event_publisher.py tests/test_deal_workflow.py -v` green
- [ ] **T6 (P1, human: ~10min / CC: ~3min)** — static check — run `uv run pyright src/deal_graph/clients/event_publisher.py src/deal_graph/workflows/deal_steps.py src/deal_graph/workflows/deal_workflow.py`
  - Surfaced by: Section 8a/b verification
  - Verify: zero new errors
- [ ] **T7 (P1, human: ~30min / CC: ~5min)** — codex review — `/codex review` against branch diff
  - Surfaced by: gstack discipline requirement + handoff doc requirement
  - Verify: any P0/P1 findings addressed before deploy
- [ ] **T8 (P1, human: ~30min / CC: ~15min)** — Railway env + deploy — surface diff to founder, get approval, commit/push, set 4 env vars on action-item-graph production, deploy, run 6-step verification
  - Surfaced by: Section 8c-f, Section 9
  - Files: Railway env (mutation), git commit, push
  - Verify: deployment Active, no startup errors, env keys set (values redacted), flag=false confirmed

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR (PLAN) | 1 P0 architectural finding (dispatcher dead in production) → resolved by pivot to DBOS step inside deal_workflow.py; 0 code-quality findings; 0 test-coverage gaps after diagram (9/9 paths covered); 0 performance findings |
| CEO Review | `/plan-ceo-review` | Scope & strategy | 0 | SKIPPED | Scope locked by upstream handoff; founder pre-approved direction |
| Codex Review | `/codex review` | Independent 2nd opinion | 0 | DEFERRED | Runs against implementation diff after T1-T6 |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | n/a | No UI surface |

**KEY ARCHITECTURAL FINDING (Section 1):** The dispatcher (`src/dispatcher/dispatcher.py`) is invoked only by the legacy `POST /process` HTTP route and is scheduled for deletion in the Phase D follow-up PR per `TODOS.md` line 7. Production Lambda traffic enqueues DBOS workflows directly, bypassing the dispatcher entirely. A dispatcher-level hook would silently no-op on Slice 3 flag flip. **Resolution:** v1 hooks inside `run_deal_workflow` as a `@DBOS.step(retries_allowed=False)`-decorated wrapper around the pure helper. Workflow-replay double-publish is explicitly accepted and absorbed by consumer-side `analyses.idempotency_key` UNIQUE.

**SCOPE (locked):** Pure helper at `src/deal_graph/clients/event_publisher.py` + thin DBOS-step wrapper in `deal_steps.py` + one call site in `deal_workflow.py` + boto3 base-dep + 7 tests (5 helper + 2 workflow) + Railway env (flag OFF) + production deploy.

**NOT IN SCOPE:** Slice 3 flag flip, Slice 4 P2 bug, dispatcher belt-and-suspenders dual-hook, agent_action_outbox refactor, the unrelated `interaction_links.id` NotNullViolation, CloudWatch alarms.

**KEY DECISIONS:**
- DBOS-step hook (production path) over dispatcher hook (dead path) — Section 4
- `retries_allowed=False` matching the codebase's fail-open side-effect convention — Section 6.3
- Pure helper + thin step wrapper (rather than `@DBOS.step` on the helper itself) keeps unit tests DBOS-free — Section 6.2
- boto3 moved to base deps for Railway runtime — Section 6.1
- Workflow-replay double-publish accepted; consumer dedupes — Section 7

**UNRESOLVED:** None.

**VERDICT:** ENG CLEARED — ready to execute T1-T8.
