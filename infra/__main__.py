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
# Railway /process; it enqueues directly via DBOSClient. The `worker-api-key`
# and `api-base-url` Pulumi config entries are kept on the stack for
# back-compat (Phase D removes them in a follow-up PR once the deprecated
# `secret_arn` export alias is retired). The forwarder stack no longer
# references them — see lambda_env_vars + secrets dicts below.

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

# Per-secret ARN exports — replaces the ambiguous "first secret" alias.
# Each Pulumi config key in the secrets dict gets its own export named
# ``<key>_secret_arn``. External consumers (CI, ops tooling) read these
# explicitly instead of inferring from a positional default.
for secret_key, secret_arn in outputs.secret_arns_by_name.items():
    pulumi.export(f"{secret_key}_secret_arn", secret_arn)

# DEPRECATED: `secret_arn` was the historical single-secret export. It
# implicitly meant the worker-api-key ARN (the only secret pre-migration).
# Kept as an alias so external consumers don't break on the cutover; once
# all consumers have migrated to the explicit `worker-api-key_secret_arn`
# export, this can be removed (planned with Phase 3 cleanup, T31).
_worker_api_key_arn = outputs.secret_arns_by_name.get("worker-api-key")
if _worker_api_key_arn is not None:
    pulumi.export("secret_arn", _worker_api_key_arn)
