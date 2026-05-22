"""Pulumi entry point for action-item-graph AWS infrastructure.

Declares the EventBridge → SQS → Lambda forwarder stack that routes
EnvelopeV1 events to the Railway API service.

Usage:
    cd infra && pulumi up --stack prod
"""

import pulumi

from forwarder import create_forwarder_stack

# ── Stack Configuration ──
config = pulumi.Config()

# DBOS system database connection (direct, non-pooler) for DBOSClient.enqueue
# from the Lambda. Must be set before `pulumi up`:
#   pulumi config set --secret dbos-system-database-url \
#     "postgresql://<role>:<pw>@<ep_id>.<region>.aws.neon.tech/eq_aig_dbos_sys?sslmode=require"
# The hostname MUST NOT contain "-pooler"; Neon's pooler breaks DBOS advisory
# locks. See docs/plans/2026-05-20-dbos-migration-execution-plan.md (T4/T5).
dbos_system_database_url = config.require_secret("dbos-system-database-url")

# Phase C cutover (2026-05-21): the Lambda dispatcher no longer POSTs to
# Railway /process; it enqueues directly via DBOSClient. The new Lambda
# code does NOT read worker-api-key.
#
# However, the worker-api-key AWS Secrets Manager resource is kept alive
# during the migration window for rollback safety. If we ever revert the
# DBOS migration (HANDOFF.md Step 6), the restored old Lambda code reads
# this secret to authenticate POSTs to Railway's /process. Letting Pulumi
# garbage-collect it now would force a re-create on rollback, with a new
# AWS ARN and a brief window where the recreated Secret's value must match
# Railway's WORKER_API_KEY env var byte-for-byte. We keep the resource
# alive end-to-end instead.
#
# Phase D follow-up PR (Day 14+ post-deploy) retires this entry alongside
# /process route deletion and the Pulumi.prod.yaml worker-api-key +
# api-base-url config cleanup. Per the deploy brief
# (docs/plans/2026-05-22-T29-T30-DEPLOY-BRIEF.md § "Day 14+ follow-up"):
# "Pulumi.prod.yaml worker-api-key config entry (kept for back-compat
# during the window; remove in Phase D)" — this explicit keep-alive
# honors that intent at the resource level too.
worker_api_key = config.require_secret("worker-api-key")

# ── Create the Forwarder Stack ──
outputs = create_forwarder_stack(
    service_name="action-item-graph",
    event_sources=[
        "com.yourapp.transcription",
        "com.eq.email-pipeline",
    ],
    detail_types=[
        "EnvelopeV1.transcript",
        "EnvelopeV1.note",
        "EnvelopeV1.meeting",
        "EnvelopeV1.email",
    ],
    lambda_handler="action_item_graph.lambda_ingest.handler.lambda_handler",
    lambda_zip_path="../dist/action-item-graph-ingest.zip",
    lambda_env_vars={
        "LOG_LEVEL": "INFO",
        "POWERTOOLS_SERVICE_NAME": "action-item-graph-ingest",
    },
    secrets={
        "dbos-system-database-url": dbos_system_database_url,
        # Migration-window keep-alive; see comment above. Retire in Phase D.
        "worker-api-key": worker_api_key,
    },
    rule_description="Routes transcript and email events to action-item-graph SQS queue",
    # Defaults match live config: 720s visibility, 3 max receives, 256MB, 120s timeout
)

# ── Stack Exports ──
pulumi.export("dlq_arn", outputs.dlq_arn)
pulumi.export("dlq_url", outputs.dlq_url)
pulumi.export("queue_arn", outputs.queue_arn)
pulumi.export("queue_url", outputs.queue_url)
pulumi.export("lambda_arn", outputs.lambda_arn)
pulumi.export("lambda_name", outputs.lambda_name)
pulumi.export("role_arn", outputs.role_arn)
pulumi.export("rule_arn", outputs.rule_arn)

# Per-secret ARN exports — each Pulumi config key in the secrets dict
# gets its own export named ``<key>_secret_arn``. External consumers
# (CI, ops tooling) read these explicitly instead of inferring from a
# positional default. Post-Phase-C the only entry is
# ``dbos-system-database-url_secret_arn``.
for secret_key, secret_arn in outputs.secret_arns_by_name.items():
    pulumi.export(f"{secret_key}_secret_arn", secret_arn)

# The historical ``secret_arn`` export (single-secret positional alias
# that meant the worker-api-key ARN pre-migration) was removed in PR #14
# alongside the broader Phase F /codex Round 2 cleanup. Per-secret named
# exports via the for-loop above are now the convention: worker-api-key's
# ARN is exported as ``worker-api-key_secret_arn``, alongside
# ``dbos-system-database-url_secret_arn``. Grep audit (CI workflow,
# scripts/, sibling EQ-CORE repos) confirmed zero consumers of the
# legacy ``secret_arn`` positional alias. Future consumers should use
# the explicit ``<key>_secret_arn`` exports above.
