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

worker_api_key = config.require_secret("worker-api-key")
api_base_url = config.require("api-base-url")

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
        "API_BASE_URL": api_base_url,
        "LOG_LEVEL": "INFO",
        "HTTP_TIMEOUT_SECONDS": "100",
        "MAX_RETRIES": "2",
        "POWERTOOLS_SERVICE_NAME": "action-item-graph-ingest",
    },
    secrets={"worker-api-key": worker_api_key},
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
pulumi.export("secret_arn", outputs.secret_arn)
