# T29 Deploy + T30 DLQ Replay Brief — DBOS Migration

**Status:** PRE-DEPLOY — populated before the deploy executes. Update the "Deploy observations" section during/after the deploy.

**Created:** 2026-05-22 (post PR #14 merge, before T29 secret-set)
**Predecessor artifact:** `docs/plans/2026-05-21-PHASE-F-PRE-PR-BRIEF.md` (pre-merge state, historical)
**Migration state:** code MERGED, infra NOT YET DEPLOYED

This brief is the structured artifact the next session uses to execute T29 + T30 cleanly. It mirrors the pre-PR brief's pattern of "everything pre-fillable is filled; live observations land in named empty sections."

---

## What's merged

**PR #14:** https://github.com/oneilstokeseqrm/action-item-graph/pull/14
**Merge commit:** `89351a3` (merge commit, 2026-05-22)
**Merge strategy:** merge commit (preserves the 16-commit history for bisectability)
**Version on main at merge:** v0.3.0
**Tests on main at merge:** 654 passing

The 16 commits (oldest → newest on the feature branch, all on main now via merge commit `89351a3`):

| # | SHA | One-line summary |
|---|-----|-------------------|
| 1 | `1bfd42d` | Phase A — Foundation: DBOS runtime substrate + Neon `eq_aig_dbos_sys` state DB + FastAPI lifespan |
| 2 | `6e51059` | Phase B-1 — Action-item DBOS workflow: 14-step decomposition |
| 3 | `6e11307` | Phase B-2 — Deal DBOS workflow: 9-step decomposition + repository idempotency fix |
| 4 | `3ca3b1e` | Phase C — Lambda dispatcher cutover: DBOSClient.enqueue |
| 5 | `00c849e` | fix(dbos): deal_workflow reads opportunity_id from extras (Rule 8 codified) |
| 6 | `8711106` | Phase E1 — Workflow + Lambda test coverage (T21+T22+T23) |
| 7 | `5b136a6` | Phase E2 — Workflow behavioral contract tests (T25a scoped down) |
| 8 | `c042207` | Phase E3 — Compatibility + concurrent-write tests (T32-T34) |
| 9 | `75ca5f2` | fix(dbos): rename Phase E2 contract test |
| 10 | `a67a190` | docs(dbos): Phase F handoff prep |
| 11 | `f9d061e` | fix(dbos): Rule 6 — deterministic IDs for Topic + TopicVersion + Deal CREATEs (Phase F /review absorption) |
| 12 | `f8e282e` | fix(dbos): register deal_workflow at workflows package init (Codex R1 — cohesion catch) |
| 13 | `4b155a2` | fix(dbos): scope topic_id to source action item, restore legacy parity (Codex R1) |
| 14 | `b7477ff` | fix(infra): drop dead secret_arn back-compat block (Codex R2) |
| 15 | `db171c4` | docs(dbos): Phase F absorption — append to HANDOFF.md + synthesize pre-PR brief |
| 16 | `dd40558` | chore: bump version and changelog (v0.3.0) |

---

## Pre-deploy gate: GitHub Actions deploy workflow is BROKEN

The merge commit (`89351a3`) automatically triggered `.github/workflows/deploy-lambda.yml`. **It failed at the "Configure AWS credentials (OIDC)" step.** Run URL: https://github.com/oneilstokeseqrm/action-item-graph/actions/runs/26282947264

**Root cause:** Pre-existing CI infrastructure issue, NOT caused by this migration. The previous failed run (PR #10 merge, 2026-03-20) failed the same way. Most likely: `AWS_DEPLOY_ROLE_ARN` GitHub Actions secret missing OR the OIDC trust policy on the IAM role doesn't allow this repo.

**Implication:** **DO NOT RELY ON THE CI WORKFLOW FOR T29 DEPLOY.** The auto-deploy is inert. Deploy will be **manual `pulumi up --stack prod`** from a machine with AWS credentials configured + Pulumi access.

**P3 follow-up (separate from T29/T30):** fix OIDC trust + set `AWS_DEPLOY_ROLE_ARN` GHA secret so future deploys auto-run. ~30 min of work; not blocking this deploy.

---

## T29 secret handoff protocol (Peter manual)

The Pulumi `prod` stack config requires `dbos-system-database-url` to be set BEFORE any `pulumi up` can succeed. `infra/__main__.py:23` calls `config.require_secret("dbos-system-database-url")` which raises if unset.

### Secret retrieval (Peter side)

Peter has the direct-connection URL in his vault. Per `feedback_secret_handoff_pattern.md`, the URL has the form:

```
postgresql://<role>:<PASSWORD>@<endpoint>.<region>.aws.neon.tech/<database>?sslmode=require
```

Confirmed metadata (NOT the password):
- **Endpoint ID:** `ep-silent-waterfall-adtinpn1`
- **Hostname:** `ep-silent-waterfall-adtinpn1.c-2.us-east-1.aws.neon.tech` (direct, NOT `-pooler.`)
- **Role:** `neondb_owner`
- **Database:** `eq_aig_dbos_sys`
- **Project:** `super-glitter-11265514` (eq-dev) — also hosts LTF's DBOS state in `neondb`

### Pulumi config set (manual, Peter executes locally)

```bash
cd infra
pulumi config set --secret dbos-system-database-url \
  "postgresql://neondb_owner:<PASSWORD_FROM_VAULT>@ep-silent-waterfall-adtinpn1.c-2.us-east-1.aws.neon.tech/eq_aig_dbos_sys?sslmode=require"
```

The `--secret` flag encrypts the value in `Pulumi.prod.yaml`. Verify with:

```bash
pulumi config get dbos-system-database-url   # Should print the URL (Pulumi decrypts for you locally)
pulumi config              # Should show `dbos-system-database-url: [secret]`
```

### Railway env var set (manual, Peter executes via Railway dashboard)

In Railway dashboard for the `action-item-graph` service:
1. Service Settings → Variables
2. Add `DBOS_SYSTEM_DATABASE_URL` with the same direct-connection URL
3. Restart the service to pick up the new env var

Verify by checking the Railway service logs after restart — should see "DBOS launched (executor_id=...)" at startup. If the env var is missing, the FastAPI lifespan at `src/action_item_graph/dbos_runtime.py:54` fails-fast with a clear error.

### CRITICAL: pooler-vs-direct check

The hostname MUST be `ep-silent-waterfall-adtinpn1.c-2.us-east-1.aws.neon.tech` (the direct endpoint). The pooled endpoint `ep-silent-waterfall-adtinpn1-pooler.c-2.us-east-1.aws.neon.tech` BREAKS DBOS because PgBouncer's transaction-mode pooling drops the advisory locks DBOS uses for workflow coordination.

Neon's default connection-string display in the dashboard returns the pooled endpoint. **Manually strip `-pooler`** from the hostname when setting the secret.

---

## Phase 1 + Phase 2 deploy sequence

Once T29 secret is set in both Pulumi and Railway, deploy from a machine with AWS credentials configured (Peter's machine, or any machine where `aws sts get-caller-identity` works for the action-item-graph deploy role).

### Step 1: Build Lambda zip

```bash
cd /Users/peteroneil/EQ-CORE/action-item-graph
bash scripts/package_lambda.sh
```

Expected output: `dist/action-item-graph-ingest.zip` (~30 MB per Phase A T3 Docker verification — well under Lambda's 50 MB direct-upload threshold). If the zip is significantly larger (>40 MB), investigate before deploy.

### Step 2: Pulumi preview (sanity check the resource changes)

```bash
cd infra
pulumi preview --stack prod
```

Expected resource diff:
- `+` `aws.secretsmanager.Secret` for `dbos-system-database-url`
- `+` `aws.secretsmanager.SecretVersion` for the secret value
- `~` Lambda execution role IAM policy (add `secretsmanager:GetSecretValue` for the new secret ARN)
- `~` Lambda function (new zip → new code SHA + environment variable update for `SECRET_NAME_DBOS_SYSTEM_DATABASE_URL`)
- `-` Lambda environment variable removals: `API_BASE_URL`, `HTTP_TIMEOUT_SECONDS`, `MAX_RETRIES` (Phase C deletion)
- `-` Pulumi stack export `secret_arn` (back-compat alias removed per `b7477ff`)
- `+` Pulumi stack export `dbos-system-database-url_secret_arn` (per-secret export per `b7477ff`)

If the preview shows resource changes BEYOND this list (e.g., deletion of unrelated resources), STOP and investigate.

### Step 3: Pulumi up (apply Phase 1 + Phase 2 together)

```bash
pulumi up --stack prod
```

**Note on phasing:** The locked 3-phase migration plan describes Phase 1 (infra) and Phase 2 (Lambda traffic shift) as conceptually distinct, but `pulumi up` applies both atomically in a single run when the zip checksum has changed. The "phasing" is enforced by the operator running `pulumi preview` first + watching the output + having a rollback plan ready, NOT by separate Pulumi commands. This is acceptable because:
- Pulumi is transactional per-resource; partial failure rolls back
- The Lambda's old zip stays live until the new zip is uploaded; the cutover is atomic at the Lambda level
- T29 secret being set BEFORE `pulumi up` means cold start of the new Lambda can resolve DBOS_SYSTEM_DATABASE_URL

### Step 4: Readiness gate — verify DBOS workers healthy on Railway

After `pulumi up` completes, BEFORE allowing live SQS traffic to the new Lambda, verify the Railway DBOS workers are healthy:

```bash
# Railway logs
railway logs --service action-item-graph --tail 50

# Look for:
# "DBOS launched (executor_id=...)" — confirms DBOS runtime started
# "lifespan.ready" — confirms FastAPI lifespan reached its yield
# No errors at startup
```

Or via Neon SQL (use `mcp__neon__run_sql` against `databaseName=eq_aig_dbos_sys`):

```sql
SELECT current_database(), current_schema();
SELECT COUNT(*) FROM dbos.workflow_status;   -- Should return 0 or N (recovery may have kicked in)
SELECT COUNT(*) FROM dbos.workflow_queue;
```

If the DBOS schema doesn't exist yet, DBOS hasn't launched successfully. Check Railway logs for the failure reason.

### Step 5: Monitor first envelopes through the new path

Once DBOS workers are confirmed healthy, allow SQS traffic to flow naturally. Watch:

**CloudWatch Logs (Lambda):**
```
record.processing  -> envelope parsed
record.enqueued    -> both workflows enqueued successfully
```

If `record.partial_enqueue` appears: action-item enqueued but deal failed. SQS will redeliver; DBOS dedupes the action-item enqueue silently. Watch the `partial_enqueue_pair_count` CloudWatch metric.

**DBOS state DB (`eq_aig_dbos_sys`):**
```sql
SELECT workflow_id, status, name, created_at
FROM dbos.workflow_status
WHERE workflow_id LIKE 'action-item-graph:%'
ORDER BY created_at DESC LIMIT 20;
```

Expected: two rows per interaction (`action-item-graph:action-item:interaction-<uuid>` + `action-item-graph:deal:interaction-<uuid>`). Status should transition `ENQUEUED` → `PENDING` → `SUCCESS`.

**Neo4j (shared DB):**
Confirm new action_items and deals appear for the test envelopes. Live MCP query via `mcp__neo4j_structured__read_neo4j_cypher`:
```cypher
MATCH (ai:ActionItem)
WHERE ai.created_at > datetime() - duration('PT10M')
RETURN ai.action_item_id, ai.summary, ai.created_at
ORDER BY ai.created_at DESC LIMIT 10;
```

**CloudWatch metric:**
```bash
aws cloudwatch get-metric-statistics \
  --namespace ActionItemGraph/Lambda \
  --metric-name partial_enqueue_pair_count \
  --start-time "$(date -u -v-15M +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 60 \
  --statistics Sum
```

Should return 0 (or no datapoints — also fine). Any non-zero value indicates the first-succeeds-second-fails split window fired; investigate why before proceeding to T30.

---

## T30 DLQ replay (post-deploy live integration test)

Once Phase 1+2 are deployed and the first ~10 envelopes process cleanly through the new DBOS path, redrive the parked DLQ message. This is the live proof that the migration solved the originating 120s Lambda timeout problem.

### The parked message

- **MessageId:** `58863f20-3cda-48f7-973d-3002aa31331b`
- **Content:** Anthropic vendor security questionnaire, ~5,500 words, 20 sections, dense structured content (the message that triggered the original incident on 2026-05-19)
- **Why it's parked:** failed Lambda 3× via SQS retry, landed in `action-item-graph-dlq`. Confirmed via Session 15 forensic investigation. Preserved untouched per hard constraint until T30.

### Redrive command

```bash
# Get the account ID and queue URLs first
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
DLQ_URL="https://sqs.${REGION}.amazonaws.com/${ACCOUNT_ID}/action-item-graph-dlq"
MAIN_QUEUE_URL="https://sqs.${REGION}.amazonaws.com/${ACCOUNT_ID}/action-item-graph-queue"

# Initiate the move task
aws sqs start-message-move-task \
  --source-arn arn:aws:sqs:${REGION}:${ACCOUNT_ID}:action-item-graph-dlq \
  --destination-arn arn:aws:sqs:${REGION}:${ACCOUNT_ID}:action-item-graph-queue
```

The redrive is asynchronous. Check status with:

```bash
aws sqs list-message-move-tasks \
  --source-arn arn:aws:sqs:${REGION}:${ACCOUNT_ID}:action-item-graph-dlq
```

### Expected outcome

Within ~5 minutes of redrive:
- CloudWatch Logs (Lambda): `record.processing` + `record.enqueued` for MessageId `58863f20-3cda-48f7-973d-3002aa31331b`
- DBOS state DB: two new `workflow_status` rows for the interaction_id in that envelope (action-item + deal workflows)
- DBOS workflows transition `ENQUEUED` → `PENDING` → `SUCCESS` (action-item workflow may take several minutes due to ~5,500-word content + LLM step latency; well within 15-minute workflow timeout)
- Neo4j: action_items + deals + topics persisted for the security-questionnaire content
- Postgres `action_items` table: dual-write rows
- SQS DLQ: empty (the parked message was moved)

**The win:** the same message that previously exhausted Lambda's 120s timeout now succeeds via DBOS workflow's checkpointed per-step retry. Document the workflow duration in the "Deploy observations" section below.

---

## Rollback procedure (if Phase 2 surfaces issues)

If post-deploy monitoring shows the new path is broken (workflows stuck in PENDING, errors in Railway logs, missing Neo4j writes, etc.):

### Step 1: Stop the bleed (~30s)

```bash
# Pause SQS event source mapping via AWS Lambda console:
# Lambda → Configuration → Triggers → SQS trigger → Disable

# OR via CLI:
aws lambda update-event-source-mapping \
  --uuid <event-source-mapping-uuid> \
  --enabled=false
```

This stops new envelopes from entering Lambda. In-flight invocations finish; in-flight DBOS workflows continue running on Railway.

### Step 2: Redeploy old Lambda code (~5 min)

```bash
git checkout main
git revert <merge-sha-89351a3> --no-edit
git push origin main
cd infra
pulumi up --stack prod
```

OR direct AWS Lambda console: select previous version, "Use this version" → publish.

The revert creates a new commit that undoes the migration; `pulumi up` deploys the reverted Lambda code which uses the old `submit_to_railway` HTTP path.

### Step 3: Resume SQS event source mapping

```bash
aws lambda update-event-source-mapping --uuid <uuid> --enabled=true
```

Lambda now uses old `submit_to_railway` path. `/process` is still alive on Railway (Phase D not yet shipped), so messages flow normally.

### Step 4: Drain in-flight DBOS workflows

In-flight DBOS workflows complete naturally — they're idempotent at the Neo4j MERGE / Postgres ON CONFLICT layer. New envelopes won't trigger new DBOS workflows (Lambda now goes to old path), so the DBOS queue drains as in-flight workflows finish.

If the issue requires emergency stop of in-flight workflows (confirmed data-corruption), operator can manually mark them CANCELLED:

```sql
-- DANGEROUS — only with explicit incident commander approval
-- Connect to eq_aig_dbos_sys
UPDATE dbos.workflow_status
SET status = 'CANCELLED'
WHERE status IN ('PENDING', 'RUNNING');
```

### Realistic recovery time

~12 minutes end-to-end (Lambda revert + redeploy ~5 min + in-flight workflow drain ~10 min). Stops new traffic in ~30 seconds via SQS event source disable.

---

## Day 14+ follow-up: Phase D PR (separate)

After 14 days post-deploy with no incidents, open the Phase D PR to retire the legacy code:

- Delete `src/action_item_graph/api/routes/process.py`
- Delete `src/dispatcher/dispatcher.py`
- Delete `tests/test_compatibility_process_route.py` (T32 — Phase 1 mid-state only)
- Delete `tests/test_workflow_behavioral_contract.py` (T25a — migration window only per docstring deletion-seam note)
- Delete `tests/test_lambda_handler.py::TestT33DBOSUnreachable` (T33 — Phase 2 deploy-order only)
- Clean `Pulumi.prod.yaml` config entries: `worker-api-key` (encrypted), `api-base-url`

KEEP (long-term invariants, NOT migration-window-only):
- `tests/test_concurrent_pipelines.py` (T34) — W2 concurrent execution is steady-state production behavior
- `tests/test_topic_executor_idempotency.py` — pins Rule 6 retry safety for Topic + TopicVersion CREATEs
- `tests/test_api_main.py::TestDBOSWorkflowRegistration` — pins cross-phase production-import-topology invariant (the deal_workflow registration regression test)

See PR #14 body's "Deletion seam list" section + v0.3.0 CHANGELOG for the full enumeration.

---

## Deploy observations — FILLED IN POST-DEPLOY (2026-05-22)

### T29 secret-set timestamp + verification

- [x] **Pulumi config set on Peter's machine**: 2026-05-22 ~09:30 PT (commit `23afb6b` landed shortly after, encoding the new encrypted entry in `Pulumi.prod.yaml`)
- [x] **Railway env var set in dashboard**: 2026-05-22 ~09:45 PT, redeploy triggered manually
- [x] **Hostname verified as direct (non-pooler)**: YES (`ep-silent-waterfall-adtinpn1.c-2.us-east-1.aws.neon.tech` — no `-pooler` infix)
- [x] **Pulumi config encrypted entry present**: YES (`dbos-system-database-url` shown as `[secret]` in `pulumi config --stack prod`)
- [x] **Railway redeploy succeeded after env var add**: 2026-05-22T16:48:55 UTC (09:48 PT), prior 06:37 + 06:45 attempts FAILED (pre-T29; expected and confirmed the fail-fast contract in `dbos_runtime.py:54`)

### Pulumi preview output — first pass (caught a deletion the brief didn't predict)

```
+  aws:secretsmanager:Secret         action-item-graph-secret-dbos-system-database-url          create
+  aws:secretsmanager:SecretVersion  action-item-graph-secret-dbos-system-database-url-version  create
~  aws:iam:RolePolicy                action-item-graph-ingest-policy                            update [diff: ~policy]
~  aws:lambda:Function               action-item-graph-ingest                                   update [diff: ~code,environment,lastModified]
-  aws:secretsmanager:SecretVersion  action-item-graph-secret-worker-api-key-version            DELETE
-  aws:secretsmanager:Secret         action-item-graph-secret-worker-api-key                    DELETE

Resources: + 2 to create   ~ 2 to update   - 2 to DELETE   8 unchanged
```

**Resource changes did NOT match expected.** Two unexpected deletions of the `worker-api-key` AWS Secrets Manager resource (Secret + SecretVersion). The brief expected only the stack-output `secret_arn` to be renamed; not the underlying AWS Secret to be garbage-collected. Phase C's removal of `worker-api-key` from the `secrets={}` dict in `infra/__main__.py` had cascading resource-graph implications that weren't called out in the brief or codex absorption commits.

**Decision**: Surfaced to Peter per discipline #1 ("anything beyond the expected list — stop and investigate"). Peter approved **Option 2**: restore worker-api-key to the secrets dict as an explicit migration-window keep-alive resource scheduled for Phase D removal. Aligns code with brief's intent and eliminates the rollback brittleness of recreating the Secret from Pulumi.prod.yaml's encrypted backup.

**Fix landed as commit `23afb6b`**: `fix(infra): keep worker-api-key alive during DBOS migration window` (also commits the encrypted `dbos-system-database-url` Pulumi config entry from T29). Pushed to main.

### Pulumi preview output — second pass (post-fix, clean diff)

```
+  aws:secretsmanager:Secret         action-item-graph-secret-dbos-system-database-url          create
+  aws:secretsmanager:SecretVersion  action-item-graph-secret-dbos-system-database-url-version  create
~  aws:iam:RolePolicy                action-item-graph-ingest-policy                            update [diff: ~policy]
~  aws:lambda:Function               action-item-graph-ingest                                   update [diff: ~code,environment,lastModified]

Outputs:
  + dbos-system-database-url_secret_arn: [unknown]
  - secret_arn                         : "arn:aws:secretsmanager:us-east-1:211125681610:secret:/action-item-graph/worker-api-key-ezZqZq"
  + worker-api-key_secret_arn          : "arn:aws:secretsmanager:us-east-1:211125681610:secret:/action-item-graph/worker-api-key-ezZqZq"

Resources: + 2 to create   ~ 2 to update   10 unchanged
```

**Zero AWS resource deletions.** Output diff shows `secret_arn` legacy alias being replaced by `worker-api-key_secret_arn`, but both have the IDENTICAL ARN value — confirming the underlying Secret is preserved end-to-end. Matches expected list.

### Pulumi up outcome

- [x] **Pulumi up succeeded**
- [x] **Resources changed**: + 2 created (Secret + SecretVersion), ~ 2 updated (IAM RolePolicy, Lambda Function), 10 unchanged. Zero deletions.
- [x] **Time elapsed**: 50 seconds
- [x] **Lambda CodeSha256 changed**: `Aa1KxqIo89Rpnq3NViw8MVP9J1L9qMnZUjtlsrb05NU=`, LastModified `2026-05-22T14:26:07.000+0000`
- [x] **Lambda env vars verified post-deploy**: `SECRET_NAME_DBOS_SYSTEM_DATABASE_URL=/action-item-graph/dbos-system-database-url` (new) + `SECRET_NAME_WORKER_API_KEY=/action-item-graph/worker-api-key` (kept for rollback safety per Option 2)
- [x] **Warnings**: One pulumi_aws SDK deprecation warning `name is deprecated. Use region instead` — unrelated to migration, internal AWS provider call, no functional impact. Worth a follow-up PR upgrading the pulumi-aws version eventually.

### Railway DBOS launch verification

- [x] **"DBOS launched (executor_id=...)" log line present**: YES at 2026-05-22T13:50:00 UTC (first launch after T29 env var set) and again at 2026-05-22T14:24:15 UTC (post-Pulumi-up auto-redeploy from my push to main). Executor IDs: `6f553b3d-a3bb-4f94-b6e9-0255436e10b5`, then `d2cee362-1ad0-4ed8-8d06-09737e888a1a`.
- [x] **"lifespan.ready" log line present**: YES on both launches
- [x] **DBOS schema exists in `eq_aig_dbos_sys`**: YES, queryable via `mcp__neon__run_sql` against `super-glitter-11265514`/`eq_aig_dbos_sys`. `SELECT FROM dbos.workflow_status` returned 0 rows pre-T30 (clean start), no error.
- [x] **Both queue listeners active**: `action-item-pipeline` (concurrency=1) + `deal-pipeline` (concurrency=1). Confirms the Phase F `deal_workflow` registration fix (commit `f8e282e`) is operative in production.

### First N envelopes through the new path

**Observed: 0 organic envelopes in the 24-hour window prior to T30.** Peter has no production users right now — upstream services (LTF, email-pipeline) only emit envelopes when actually used. The "watch 10 organic envelopes" check assumed production traffic exists; it doesn't for this project. Per Peter's direction, organic-traffic gate was skipped; T30 DLQ redrive served as the live integration test by design.

- [N/A] First envelope processed organically: NO ORGANIC TRAFFIC in window
- [N/A] First 10 envelopes: NO ORGANIC TRAFFIC
- [x] **`partial_enqueue_pair_count` CloudWatch metric stayed at 0**: YES (no datapoints — never fired across the deploy and T30)
- [N/A] Workflow stuck in PENDING > 15 min: N/A

### T30 DLQ replay result

**Two distinct DLQ messages discovered (Session 14 forensic correction):**

| MessageId | Type | Sent | Content |
|---|---|---|---|
| `58863f20-3cda-48f7-973d-3002aa31331b` | EnvelopeV1.email | 2026-05-19T16:04:02 UTC | Anthropic security questionnaire, ~5,500 words |
| `0cc72fb0-c475-42cc-a3a0-e0019e59a4f2` | EnvelopeV1.meeting | 2026-05-19T16:24:07 UTC | Enterprise launch meeting transcript, 8,566 words / 73,405 chars |

Both same-wave Session 14 synthetic casualties: same tenant (`11111111-1111-4111-8111-111111111111`), same account (Anthropic `e008a004-95ec-5eb7-95ce-56108d0eed77`), both content-heavy enough to hit the OLD 120s Lambda timeout. The original Session 14 forensic at the eq-synthetic-date-generation HANDOFF claimed "1 unique message" — likely a receive-message-visibility timing artifact at the moment of the original receive call. **Both messages redriven together as a richer integration test.**

- [x] **DLQ messages redriven at**: 2026-05-22T14:48:34 UTC (`aws sqs start-message-move-task` returned `ApproximateNumberOfMessagesMoved: 2/2`)
- [x] **Lambda received both messages**: T+~7s at 14:48:40.997 UTC (email) + 14:48:41.510 UTC (meeting); both with `cold_start: true` (Lambda had been idle 24h)
- [x] **Both workflows enqueued per envelope**:
  - Email: `action-item-graph:action-item:interaction-ce62cbdb-a3f9-4b7d-8dfd-a5f284ef9e14` + `action-item-graph:deal:interaction-ce62cbdb-a3f9-4b7d-8dfd-a5f284ef9e14`
  - Meeting: `action-item-graph:action-item:interaction-8bcd0dfc-465b-45ea-83fa-d81f25126141` + `action-item-graph:deal:interaction-8bcd0dfc-465b-45ea-83fa-d81f25126141`
- [x] **All 4 workflow status timelines reached SUCCESS** (recovery_attempts=1 on all, meaning first attempt clean, no retries fired):

| Workflow | Status | Duration | Notes |
|---|---|---|---|
| Email action-item | SUCCESS | **187.2s** | Created 18:48:41.016, Done 18:51:48.242 UTC |
| Email deal | SUCCESS | 4.7s | `status: no_deals` — security questionnaire has no deal extraction, legitimate |
| Meeting action-item | SUCCESS | **323.7s** | Created 18:48:41.572, Done 18:54:05.259 UTC. Ran serially behind email (concurrency=1 queue) |
| Meeting deal | SUCCESS | 79.1s | `total_extracted: 1, deals_created: []` — merged with existing Anthropic deal, no new Deal created |

- [x] **Action_items in Neo4j**: 5 EXTRACTED_FROM meeting + 1 EXTRACTED_FROM email = 6 total ActionItems. Quality outputs (enterprise launch / DPA / security review themes), priority_scores 0.67–0.93, all `commitment_strength="explicit"`-class items.
- [x] **Deals created in Neo4j**: 0 NEW. Two pre-existing Anthropic deals on the account ("Anthropic Relationship Health Monitoring" v6, "Anthropic Meridian POC Expansion" v10) — meeting workflow extracted 1 deal candidate, matched and merged idempotently with existing (content matched, no version snapshot needed — Rule 6 retry-safety contract holding).
- [x] **ActionItemTopics created**: 6 for meeting (`enterprise production launch`, `enterprise deal proposal`, etc.) + 1 for email (`vendor security compliance`). All linked via `BELONGS_TO`.
- [x] **Owners resolved**: 5 distinct (Ben, Diana Vasquez, Marcus Chen, Lior Sadan, etc. for meeting) + 1 (Lior Sadan for email).
- [x] **Contacts ATTENDED on meeting**: 3 (matches `extras.contacts_count=3` from envelope — contact enrichment from Session 14's contact_ops work intact).
- [x] **Postgres dual-write rows present**: YES, 6 rows in `action_items` table with `source_interaction_id` matching T30 interactions. **All UPDATED idempotently** — created originally on 2026-05-19 via the synthetic data tool's direct injection path (separate from Lambda); today's T30 pipeline incremented `version_number` (3→12 across rows) and updated `updated_at` to 14:51–14:53 UTC. **No duplicate rows. Rule 6's MERGE-on-deterministic-UUID5 contract held end-to-end** against real-world "same envelope already partially processed via different path" conditions.
- [x] **Workflow output `error` columns**: ALL EMPTY across 4 workflows. Zero errors.
- [x] **Application version on workflow rows**: `43d87c4fca8441f6a51237e428a55eae` (matches the deployed Railway app version) — confirms workflows executed by the new DBOS code path, not legacy.
- [x] **CloudWatch Lambda metrics during T30**: 2 invocations, 0 errors, `partial_enqueue_pair_count` = 0 sum (never fired).
- [x] **THE WIN — did the message that previously exhausted Lambda's 120s timeout now succeed?** **YES.** Both load-bearing numbers:
  - **Email action-item: 187.2s** — 1.56× the old 120s ceiling; would have hard-failed on old Lambda.
  - **Meeting action-item: 323.7s** — **2.7× the old ceiling**; the dominant content-heavy proof.

Both completed cleanly inside DBOS's 900s workflow_timeout. **Problem A's failure class is structurally solved.**

### Known-limits monitoring start (HANDOFF §2)

Start watching as of post-T30 (2026-05-22T14:55 UTC):
- [ ] **Version counter drift under retry** — `update_action_item` / `update_deal` use `SET version = version + 1` without CAS. recovery_attempts=1 on all 4 T30 workflows = no retries fired today, so this concern didn't exercise. Watch organic traffic when it resumes.
- [ ] **Owner CREATE narrow cross-workflow race** — no concurrent workflows during T30 (concurrency=1 + only 2 envelopes), so the narrow-race window didn't exercise either. Watch for duplicate Owner nodes via periodic dedup query once organic traffic resumes.
- [ ] **S9a/S9b per-step instrumentation asymmetry** — dashboards should encode this; no action needed unless alerting fires on the gap.

### `concurrency=1` raise-criterion tracking starts now

Per locked decision D4: raise to 3 after **100 successful invocations with zero DB/Neo4j/OpenAI errors AND queue-depth metric trending positive**.

**Tally: 4/100 today (the 4 T30 workflows).** 96 more clean invocations to go before raising. Counter resets if any error fires.

---

## Phase 1 close summary

T29 (manual secret + Railway env var) + Pulumi up (Lambda code + new Secret + IAM policy expansion) + T30 (DLQ live integration test on 2 envelopes) all clean. Migration's structural goal — eliminate the 120s Lambda timeout failure class via per-step checkpointed DBOS workflows — verified on the highest-content envelopes that were available in production state at the time of cutover.

**Phase 1 = CLOSED.** Phase D (retire `/process` + `dispatcher.py` + migration-window-only tests + `Pulumi.prod.yaml` config cleanup including `worker-api-key`) opens as a separate PR Day 14+ post-deploy. 14-day operational monitoring window starts now.

---

## Final pre-Phase-D checklist

After 14 days post-deploy with no incidents:

- [ ] DBOS workflows in `eq_aig_dbos_sys` consistently reaching SUCCESS
- [ ] `partial_enqueue_pair_count` metric stayed at 0 for the entire window
- [ ] No HANDOFF §2 known-limits surfaced in production (or documented as benign)
- [ ] No customer-reported issues with action_item or deal extraction quality
- [ ] DLQ stayed empty (or accumulated messages classified as Lambda-level enqueue failures only, not DBOS workflow failures)

Then open Phase D PR per "Day 14+ follow-up" section above.
