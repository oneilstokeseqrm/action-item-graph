# Dual-Write `stage` / `amount` / `deal_status` to Postgres — Implementation Spec

**Status:** Approved, ready to execute. Do NOT relitigate the architectural decisions below.
**Owner of handoff:** Peter ONeil (founder/architect — has explicit context)
**Generated:** 2026-05-16, after deep multi-session investigation.
**Primary goal (do not lose sight of this):** **AI-extracted deals must be visible in the `/pipeline` UI.** Field completeness (stage, amount) is a secondary enrichment goal. NULL values on those fields are acceptable and must not block visibility.

---

## TL;DR — the change

1. **eq-frontend (PRIMARY UNBLOCKER):** Fix the `forecast.getPipeline` tRPC filter to be NULL-aware. Currently `dealStatus: { notIn: [...] }` excludes NULL via SQL three-valued logic, hiding every AI-extracted deal. This alone makes the 4 existing test deals visible.
2. **action-item-graph (ENRICHMENT):** Have the dual-write populate `stage`, `amount`, and `deal_status='open'` so deals don't render with blank columns. Use full INSERT+UPDATE for stage/amount; INSERT-only for deal_status (so user closes aren't reverted).

Both changes are needed. The frontend fix is the visibility unblocker; the AIG fix is enrichment. **Ship both.**

---

## Critical product requirement (Peter, 2026-05-16)

> *"It should be okay for fields like 'amount' to be null. It may be that only some interactions truly mention an amount, or that none do. But we still need to be able to see the deal in the UI."*

> *"We need, as a baseline, to be able to see these deals in the UI."*

**Implications:**
- The frontend filter MUST NOT exclude rows where `amount`, `stage`, or `close_date` are NULL.
- The frontend filter MUST treat NULL `deal_status` as "not closed" (i.e., visible).
- The AIG dual-write MUST write NULL when the merger LLM has no value for `stage` / `amount` (do not COALESCE-fudge it to keep a stale value — the merger already preserves correctly via its own logic; just pass through whatever the merger produced).
- Display layer (rows, sort, aggregates) MUST handle NULL gracefully. Sort by closeDate with NULL closeDate should still render the row (push to end or wherever — just don't drop it).

---

## What has already been done in the predecessor session (do not redo)

1. **Investigated the 422 incident** (different bug, fully resolved). PR #11 merged 2026-05-15 added `'generic'` and `'zoom'` to the `SourceType` enum. Deploy `50781390-0a48-4367-99d7-ec19858637fb` is live. DLQ purged. Synthetic injection confirmed working end-to-end.
2. **Verified test data exists** in the DB:
   - Tenant: `11111111-1111-4111-8111-111111111111`
   - Account: `0e49a47e-0200-5e4f-962c-2b3df57e0624`
   - 4 opportunities in Postgres `opportunities` table with `graph_opportunity_id` values: `019e2d2e-70b7-7591-ae9c-9267182345b1`, `019e2d2c-4237-7bb3-850a-e0cce9bf1045`, `019e2d2b-dc4b-7dd2-b825-d88b34cb8980`, `019e2d2a-7950-7f33-b455-e5cf61c92277`
   - Names: "Palantir Relationship Intelligence Solution", "Palantir EQ Enterprise License Renewal", "Government Contractor Deal", "Palantir EQ Relationship Intelligence Platform POC"
   - All have NULL `stage`, `amount`, `deal_status` in Postgres
   - All have populated `stage` (e.g., "negotiation"), `amount` (e.g., 240000.0), and full MEDDIC fields in Neo4j Deal nodes
   - 17 action_items, 14 deal_versions, 1,197 deal_events in Postgres (action item & deal pipelines fully working otherwise)
3. **Verified the visibility bug**:
   - Frontend filter at `eq-frontend/lib/trpc/routers/forecast.ts:308-311` is `dealStatus: { notIn: ['closed_won', 'closed_lost'] }`
   - Running this against the DB returns **0 rows** for the test tenant
   - The null-aware variant returns 4 rows (the actual deals)
   - Confirmed via direct SQL: `SELECT COUNT(*) FROM opportunities WHERE tenant_id = '...' AND deal_status NOT IN ('closed_won','closed_lost')` → 0

---

## Architectural decisions (locked, do not relitigate)

### Decision 1: Both Postgres and Neo4j are co-equal sources of truth

The old `postgres_client.py:5` comment that calls Postgres a "read-optimized projection" is **wrong** under Peter's intended model. Frame the change as "writing the same values to both stores," not "leaking Neo4j into a projection." Update the docstring accordingly.

### Decision 2: AI extraction IS supposed to fire the forecast trigger

The `opportunity-forecasting` repo's migration `sql/021_extraction_change_trigger_registry_coverage.sql` explicitly extended the forecast trigger to watch MEDDIC fields so that "when the extraction pipeline updates a MEDDIC field" the listener fires. The original 003 migration's narrower watch-list was a pre-021 state. The trigger already fires on AI extractions today (via MEDDIC fields). Adding stage/amount to the dual-write does NOT introduce a new firing pattern — it just propagates more correct data.

### Decision 3: AIG owns stage/amount in production; opp-forecasting does not

Verified by exhaustive grep across `opportunity-forecasting/src/`, `sql/`, and `scripts/`:
- Production code never UPDATEs or INSERTs `stage` or `amount` on the `opportunities` table.
- The forecast pipeline READS `stage` via `evidence/assembler.py` and `engine/forecaster.py` (used for Track 1 vs Track 2 routing — likely what Peter remembered as "phase one / phase two").
- Stage/amount writes in opp-forecasting only exist in `scripts/enrich_account_deals.py`, `seed_rich_test_deals.py`, `run_live_e2e.py`, and `verify_event_system.py` — all manual/seed/test utilities, never run by the production pipeline.

### Decision 4: AIG's merger LLM already does cross-transcript reasoning

`src/deal_graph/prompts/merge_deals.py:98-113` shows the merger LLM prompt template includes `Existing Stage`, `Existing Amount`, `Existing Evolution History` (cumulative narrative across all transcripts), and all existing MEDDIC fields. The merger is NOT single-transcript blind. It updates stage/amount when revised, preserves when no new info (per prompt instruction at `merge_deals.py:7,40-41`).

Concerns about merger LLM accuracy on stage/amount overwrites are noted but **deferred** to the parallel user-assertion model being built in eq-frontend. **Do not** try to build that user-assertion layer as part of this change.

### Decision 5: deal_status='open' on INSERT only (never reverted)

When a user closes a deal via UI (sets `deal_status = 'closed_won'`), subsequent AIG extractions on the same deal must NOT revert to `'open'`. Solution: write `'open'` in the INSERT clause only, omit from the UPDATE SET clause. This works because `ON CONFLICT DO UPDATE SET` runs only when the row already exists — and on existing rows, we want to preserve user-set status.

### Decision 6: NULL is a first-class state for stage/amount/close_date

If the merger LLM doesn't surface a stage or amount (e.g., the deal exists from contact/relationship signals but no transcript mentioned a stage), NULL is the correct value. The UI must render NULL fields as blank (or "—"), not hide the row.

`close_date` stays out of this change entirely. AIG doesn't extract a target close_date as a structured field. Neither does opp-forecasting today. Leave it NULL across the board; revisit in a separate scope.

### Decision 7: What NOT to do

- Don't touch the forecast trigger (`opportunity-forecasting/sql/003_*.sql` or `021_*.sql`).
- Don't touch `close_date` — separate scope.
- Don't modify the merger LLM prompt or behavior.
- Don't build a user-assertion override layer.
- Don't move stage/amount inference into opportunity-forecasting (it's a forecast engine, not a state-inference engine).
- Don't add COALESCE to preserve old values when the new value is NULL — the merger already preserves correctly; we want straight passthrough.

---

## Change spec — Part A: eq-frontend (do this FIRST — it's the visibility unblocker)

**Repo:** `/Users/peteroneil/eq-frontend`
**File:** `lib/trpc/routers/forecast.ts`
**Branch:** `fix/pipeline-filter-null-aware-deal-status`

### Change 1: Page query filter (line ~308-311)

Current:
```ts
const where: Record<string, unknown> = {
  tenantId: ctx.tenantId,
  dealStatus: { notIn: ["closed_won", "closed_lost"] },
};
```

New:
```ts
const where: Record<string, unknown> = {
  tenantId: ctx.tenantId,
  OR: [
    { dealStatus: null },
    { dealStatus: { notIn: ["closed_won", "closed_lost"] } },
  ],
};
if (input.stage) where.stage = input.stage;
if (input.accountId) where.accountId = input.accountId;
```

### Change 2: Aggregate query filter (line ~367-371)

Current:
```ts
const aggRows = await prisma.opportunity.findMany({
  where: {
    tenantId: ctx.tenantId,
    dealStatus: { notIn: ["closed_won", "closed_lost"] },
  },
  ...
});
```

New:
```ts
const aggRows = await prisma.opportunity.findMany({
  where: {
    tenantId: ctx.tenantId,
    OR: [
      { dealStatus: null },
      { dealStatus: { notIn: ["closed_won", "closed_lost"] } },
    ],
  },
  ...
});
```

### Tests for Part A

Find existing tests for `forecast.getPipeline` (likely in `__tests__/` near the router, or in `lib/trpc/routers/__tests__/`).

Add tests:
1. `getPipeline returns deals with NULL dealStatus` — seed an opportunity with `dealStatus: null` and assert it appears in `items`.
2. `getPipeline excludes deals with dealStatus='closed_won'` — regression guard for existing behavior.
3. `getPipeline includes deals with dealStatus='open'` — regression guard.
4. `aggregates count includes NULL dealStatus deals` — `dealCount` should include the NULL row.

### Verify Part A works in isolation

After Part A merges and Vercel auto-deploys, load `https://eq-frontend-two.vercel.app/pipeline` with `E2E_EMAIL_USER2` (see `/Users/peteroneil/eq-frontend/.env.e2e.local`) and confirm the 4 existing deals render. Stage / amount columns will be blank. That's correct and expected at this point — Part B adds the values.

---

## Change spec — Part B: action-item-graph

**Repo:** `/Users/peteroneil/EQ-CORE/action-item-graph`
**File:** `src/action_item_graph/clients/postgres_client.py`
**Branch:** `fix/dual-write-stage-amount-deal-status`

### Change 1: Update the trigger-protected column set (~line 51-56)

Current:
```python
# Columns on opportunities table that fire notify_forecast_job() on UPDATE.
# The Deal dual-write MUST NOT write to these columns.
_DEAL_TRIGGER_PROTECTED_COLUMNS = frozenset({
    'stage', 'amount', 'close_date', 'deal_status',
    'forecast_category', 'next_step', 'description', 'lost_reason',
})
```

New:
```python
# Columns on opportunities table that AIG does not currently compute or own.
# Historical note: this list was originally framed as "trigger-protected"
# (avoiding notify_forecast_job() firing on AI updates), but migration 021
# of opportunity-forecasting (sql/021_extraction_change_trigger_registry_coverage.sql)
# explicitly extended the trigger to fire on MEDDIC field updates from AI
# extraction. So AI extractions already fire the forecast trigger today via
# MEDDIC. This frozenset now means "AIG has no upstream source for these
# fields" — not "trigger-protected."
#
# `stage`, `amount`, and `deal_status` were moved OUT of this set on 2026-05-16
# because AIG does compute stage and amount (via the deal_graph merger LLM)
# and assigns deal_status='open' as the canonical initial state for any
# AI-extracted deal.
_DEAL_FIELDS_AIG_DOES_NOT_WRITE = frozenset({
    'close_date',           # target close date; needs separate extraction scope
    'forecast_category',    # owned by opportunity-forecasting pipeline
    'next_step',            # user-facing free-form field
    'description',          # user-facing free-form field
    'lost_reason',          # only set when a user closes a deal as lost
})
```

(If the old constant name `_DEAL_TRIGGER_PROTECTED_COLUMNS` is referenced elsewhere in the codebase, either keep both names or rename references. Check with `grep -rn _DEAL_TRIGGER_PROTECTED_COLUMNS src/ tests/`.)

### Change 2: Update `upsert_deal()` docstring (~line 715)

Current:
```python
"""Upsert a Deal into the opportunities table.

Conflict resolution on graph_opportunity_id (unique index).
Does NOT write to trigger-protected columns (stage, amount, etc.).

Returns:
    The Postgres-side ``opportunities.id`` (PK) for FK references.
"""
```

New:
```python
"""Upsert a Deal into the opportunities table.

Conflict resolution on graph_opportunity_id (unique index).

Writes:
- INSERT: graph_opportunity_id, tenant_id, account_id, opportunity_name,
  deal_ref, currency, actual_close_date, deal_status='open' (literal),
  stage, amount, latest_ai_summary, ai_evolution_summary, all MEDDIC
  fields, all ontology dim_* fields, ontology_scores_json,
  ontology_completeness, ontology_version, extraction_embedding,
  extraction_embedding_current, extraction_confidence, extraction_version,
  source_interaction_id, qualification_status, source_user_id,
  ai_workflow_metadata.
- UPDATE: all of the above EXCEPT deal_status (deal_status is INSERT-only
  so user-set closure values like 'closed_won' are never reverted by
  subsequent AI extractions). stage and amount ARE updated on subsequent
  extractions; the merger LLM handles cross-transcript reasoning and
  passes through whatever the latest synthesized value is, including
  NULL when no transcript mentions an amount.

Does NOT write: close_date (target close date — separate scope),
forecast_category, next_step, description, lost_reason (see
_DEAL_FIELDS_AIG_DOES_NOT_WRITE).

Both Postgres and Neo4j are sources of truth in the system. This is a
straight dual-write of the merger output to both stores.

Returns:
    The Postgres-side ``opportunities.id`` (PK) for FK references.
"""
```

### Change 3: Update the SQL — INSERT clause (~line 730-765)

Add `deal_status`, `stage`, `amount` to the column list. Note: position matters for matching VALUES clause.

In the INSERT column list, add after `actual_close_date` (or in a sensible position — the existing order isn't alphabetical, so just be consistent):
```
deal_status, stage, amount,
```

In the corresponding VALUES list, add:
```
'open', :stage_value, :amount_value,
```

### Change 4: Update the SQL — ON CONFLICT DO UPDATE SET clause (~line 766-798)

Add the following lines to the UPDATE SET clause (NOT deal_status):
```
stage = :stage_value,
amount = :amount_value,
```

**Critical:** Do NOT add `deal_status = :something` to the UPDATE SET clause. The INSERT clause sets it to 'open' on first creation; subsequent upserts (ON CONFLICT path) leave it untouched, preserving any user-set value.

### Change 5: Update the `params` dict (~line 801)

Add:
```python
'stage_value': deal.stage.value if deal.stage else None,
'amount_value': float(deal.amount) if deal.amount is not None else None,
```

Verify the types: `deal.stage` is a `DealStage` enum (defined in `src/deal_graph/models/deal.py`), and `deal.amount` is `Decimal | float | None`. The `.value` access on DealStage returns the string like `'negotiation'`. `float()` of `Decimal` works.

If `deal.amount is None`, write NULL. If `deal.stage is None`, write NULL. Both are valid Postgres states per Decision 6.

### Tests for Part B

Locate existing `upsert_deal` tests: `grep -rln "upsert_deal" tests/` (likely in `tests/test_postgres_deal_dual_write.py` or similar).

Add tests:

1. **`test_upsert_deal_insert_sets_stage_amount_deal_status`** — Build a Deal with `stage=DealStage.NEGOTIATION, amount=240000.0`. Call `upsert_deal()`. Query Postgres directly; assert row has `stage='negotiation'`, `amount=240000.0`, `deal_status='open'`.

2. **`test_upsert_deal_insert_handles_null_stage_and_amount`** — Build a Deal with `stage=None, amount=None`. Call `upsert_deal()`. Assert row exists with `stage IS NULL`, `amount IS NULL`, `deal_status='open'`. The row must exist — NULL stage/amount must NOT prevent insertion.

3. **`test_upsert_deal_update_writes_new_stage_amount`** — Insert a deal with stage='proposal', amount=100000. Then upsert the same `graph_opportunity_id` with stage='negotiation', amount=240000. Assert Postgres now has the new values.

4. **`test_upsert_deal_update_preserves_user_set_deal_status`** — Insert a deal (deal_status='open' via INSERT). Manually UPDATE the row to deal_status='closed_won' (simulating user UI action). Then upsert again with new MEDDIC data. Assert deal_status is STILL 'closed_won' (not reverted to 'open').

5. **`test_upsert_deal_update_can_clear_amount`** — Insert a deal with amount=240000. Upsert again with `deal.amount=None`. Assert Postgres row now has amount=NULL. (This validates Decision 6: NULL is a first-class state, and the merger LLM passing through None should clear the value, not preserve via COALESCE.)

### Run the tests

```bash
cd /Users/peteroneil/EQ-CORE/action-item-graph
uv run pytest tests/test_postgres_deal_dual_write.py -v --timeout 60
# (or whichever file you added tests to)

# Then run full suite:
uv run pytest tests/ --timeout 60
```

Expect 562+ tests passing. There is one known pre-existing flaky test (`test_pipeline_with_topics.py::test_pipeline_works_without_topics`) that times out on network I/O — that's not related to this change, do not investigate.

---

## Deployment order

1. **Ship Part B (AIG) FIRST.** This is counterintuitive — but if the frontend filter change goes first and there's any issue with the AIG dual-write (e.g., type error on `deal.amount` Decimal conversion), the visible deals would render with blank values forever and confuse Peter. By shipping AIG first, the existing 4 NULL-stage/amount deals remain hidden (current state), and only after AIG ships do new deals get full values. Then the frontend fix surfaces them.

   Actually — on reflection, the reverse order is also fine and may be preferable for fastest user value. Peter wants to see the 4 deals NOW. If Part A ships first, those 4 render immediately (with blank stage/amount), and Part B retroactively fills in stage/amount on the NEXT synthetic injection. Either order works.

   **Recommended: ship Part A first (immediate visibility), then Part B (enrichment).** Either way, both must ship.

2. **Each PR uses the standard flow:**
   - Branch off `main`
   - Write code + tests
   - Commit (no emojis, follow Peter's commit style — see prior commits via `git log --oneline`)
   - Push
   - `/ship` to create PR (or `gh pr create` directly — the change is small enough)
   - `/land-and-deploy` (or just `gh pr merge` + monitor deploy)

3. **Deploy environments:**
   - action-item-graph → Railway, auto-deploys on merge to main (verified working in PR #11). Service: `api-production-c33a.up.railway.app`. Project ID `6b6205f8-a838-4c1a-8f18-de29df9fa695`, service ID `abf7b1cd-9783-4b4b-bee7-da3d9bfb13da`, env ID `c6cad37d-67a0-4703-b003-635f319b5480`.
   - eq-frontend → Vercel, auto-deploys on merge to main. URL: `https://eq-frontend-two.vercel.app/`.

---

## End-to-end verification (must pass before declaring done)

1. **Direct DB query confirms Part A works:**
   ```sql
   -- Should return 4 rows (the existing test deals)
   SELECT id, opportunity_name, deal_status, stage, amount
   FROM opportunities
   WHERE tenant_id = '11111111-1111-4111-8111-111111111111'
     AND (deal_status IS NULL OR deal_status NOT IN ('closed_won', 'closed_lost'));
   ```

2. **Visual confirmation in UI:**
   - Login at `https://eq-frontend-two.vercel.app/` with `E2E_EMAIL_USER2` / `E2E_PASSWORD_USER2` from `/Users/peteroneil/eq-frontend/.env.e2e.local`.
   - Navigate to `/pipeline`.
   - **Must see** the 4 deals: "Palantir Relationship Intelligence Solution", "Palantir EQ Enterprise License Renewal", "Government Contractor Deal", "Palantir EQ Relationship Intelligence Platform POC".
   - After Part A alone: deals render with blank stage / amount columns.
   - After Part B + a fresh synthetic injection: new deals render with stage / amount populated. (Existing 4 deals won't backfill unless re-extracted; that's expected.)
   - Use the gstack `/browse` skill for the visual check, NOT direct Playwright.

3. **Fresh synthetic injection confirms Part B works end-to-end:**
   - Trigger a synthetic injection from `/Users/peteroneil/EQ-CORE/eq-synthetic-date-generation` (see prior session's HANDOFF for invocation pattern, or check `src/eq_synthetic/injection/transcript_injector.py`).
   - Wait for Railway action-item-graph logs to show `record.success`.
   - Query the newly-created opportunity in Postgres:
     ```sql
     SELECT opportunity_name, deal_status, stage, amount
     FROM opportunities
     WHERE tenant_id = '11111111-1111-4111-8111-111111111111'
     ORDER BY created_at DESC LIMIT 1;
     ```
   - Must have `deal_status='open'`. May have NULL or populated stage/amount (depending on whether the transcript surfaced them).
   - Reload `/pipeline` UI; new deal must appear.

4. **Regression check:**
   - The 422 fix from PR #11 must still be working. Confirm `action-item-graph-dlq` SQS queue has 0 messages after fresh injection.
   - Action item pipeline must still be writing rows (count of `action_items` should grow).

---

## Useful context the next agent will need

### Key files

| File | Purpose |
|---|---|
| `src/action_item_graph/clients/postgres_client.py` | Dual-write code; upsert_deal at line 710+ |
| `src/deal_graph/models/deal.py` | `Deal` model; `DealStage` enum (`PROSPECTING`, `QUALIFICATION`, `PROPOSAL`, `NEGOTIATION`, `CLOSED`) |
| `src/deal_graph/pipeline/merger.py` | Where merger output is built; merger LLM call site |
| `src/deal_graph/prompts/merge_deals.py` | Merger LLM prompt template — DO NOT MODIFY |
| `eq-frontend/lib/trpc/routers/forecast.ts` | `getPipeline` tRPC procedure; filter at line 308-311 and 367-371 |
| `eq-frontend/prisma/schema.prisma:821` | `Opportunity` Prisma model definition |
| `opportunity-forecasting/sql/003_forecast_trigger.sql` | Original trigger (FYI, don't touch) |
| `opportunity-forecasting/sql/021_extraction_change_trigger_registry_coverage.sql` | Extended trigger that explains why AI extractions firing forecasts is intentional |

### Project memory (auto-loaded by Claude)

Already in `/Users/peteroneil/.claude/projects/-Users-peteroneil-EQ-CORE-action-item-graph/memory/MEMORY.md` — describes the three-pipeline system, deployed code state, and live E2E expectations. Read on session start.

### Live deploy state (as of 2026-05-16)

- action-item-graph Railway: deployment `50781390-0a48-4367-99d7-ec19858637fb` (live since 2026-05-15 ~14:00 UTC)
- Last successful PR: #11 (added 'generic' and 'zoom' to SourceType enum)
- eq-frontend Vercel: latest main is live; URL `https://eq-frontend-two.vercel.app/`
- DLQ: 0 messages (purged 2026-05-15)
- Main queue: 0 messages

### Conversation history pointer

Full context of how we arrived at this spec is in the Claude session ending 2026-05-16. Key insights, in order of importance for the next session:

1. The trigger isn't really "AI-write-hostile" — migration 021 made AI extractions fire forecasts on purpose.
2. opp-forecasting doesn't own stage/amount in production (only reads them).
3. AIG merger LLM does see prior state (existing_stage, existing_amount, existing_evolution_summary).
4. The visible bug is the SQL three-valued logic in the frontend filter, not the dual-write per se.
5. Peter wants both fixes — the frontend filter (visibility unblocker) and the dual-write (enrichment).
6. NULL on amount/stage is fine. Visibility is the baseline; field completeness is the polish.

---

## When this is done

1. Both PRs merged to main.
2. Both deploys live and verified.
3. UI shows the 4 deals at `/pipeline`.
4. A fresh synthetic injection produces a deal with stage/amount populated, visible in UI.
5. Update `MEMORY.md` (project memory) to note this fix landed and reference this doc.
6. Optional: write a brief lesson in `tasks/lessons.md` (or wherever project lessons live) about the three-valued-logic NULL exclusion gotcha — it's a class of bug worth flagging for future filters.
