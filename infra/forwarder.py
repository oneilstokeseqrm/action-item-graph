"""Reusable EventBridge → SQS → Lambda forwarder pattern.

Creates the full set of AWS resources for an event-driven thin-forwarder:
  1. SQS Dead Letter Queue
  2. SQS Queue (with redrive to DLQ)
  3. SQS Queue Policy (allowing EventBridge to send)
  4. IAM Role + Inline Policy (SQS consume, CloudWatch Logs, X-Ray, Secrets Manager)
  5. Secrets Manager Secret (for service-to-service auth)
  6. Lambda Function (thin forwarder)
  7. EventBridge Rule (event pattern match)
  8. EventBridge Target (rule → queue)
  9. Lambda Event Source Mapping (queue → Lambda)

Data flow:
  EventBridge (default bus)
      │ rule matches source + detail-type
      ▼
  SQS Queue ──(redrive after N failures)──> SQS DLQ
      │ event source mapping (BatchSize=1)
      ▼
  Lambda (thin forwarder)
      │ HTTPS POST + Bearer token from Secrets Manager
      ▼
  Railway service
"""

from dataclasses import dataclass

import json
import pulumi
import pulumi_aws as aws


@dataclass
class ForwarderOutputs:
    """ARNs and URLs exported by the forwarder stack."""

    dlq_arn: pulumi.Output[str]
    dlq_url: pulumi.Output[str]
    queue_arn: pulumi.Output[str]
    queue_url: pulumi.Output[str]
    lambda_arn: pulumi.Output[str]
    lambda_name: pulumi.Output[str]
    role_arn: pulumi.Output[str]
    rule_arn: pulumi.Output[str]
    secret_arn: pulumi.Output[str]


def create_forwarder_stack(
    service_name: str,
    event_sources: list[str],
    detail_types: list[str],
    lambda_handler: str,
    lambda_zip_path: str,
    lambda_env_vars: dict[str, str | pulumi.Output[str]],
    secrets: dict[str, str | pulumi.Output[str]] | None = None,
    queue_visibility_timeout: int = 720,
    dlq_max_receive_count: int = 3,
    lambda_memory_mb: int = 256,
    lambda_timeout_seconds: int = 120,
    rule_description: str | None = None,
) -> ForwarderOutputs:
    """Create the full EventBridge → SQS → Lambda forwarder stack.

    Args:
        service_name: Base name for all resources (e.g., "action-item-graph").
        event_sources: EventBridge source values to match.
        detail_types: EventBridge detail-type values to match.
        lambda_handler: Python handler path (e.g., "pkg.module.handler.func").
        lambda_zip_path: Path to the Lambda deployment zip.
        lambda_env_vars: Non-secret environment variables for the Lambda.
        secrets: Dict of secret name → value to store in Secrets Manager.
            The Lambda gets secretsmanager:GetSecretValue permission.
            Keys become secret names under /{service_name}/ prefix.
        queue_visibility_timeout: SQS visibility timeout in seconds.
        dlq_max_receive_count: Max receives before routing to DLQ.
        lambda_memory_mb: Lambda memory allocation in MB.
        lambda_timeout_seconds: Lambda timeout in seconds.
        rule_description: Optional description for the EventBridge rule.

    Returns:
        ForwarderOutputs with ARNs and URLs of all created resources.
    """
    secrets = secrets or {}

    # Resolve AWS account + region dynamically for IAM policy ARNs
    caller = aws.get_caller_identity()
    region = aws.get_region()

    # ── 1. SQS Dead Letter Queue ──
    dlq = aws.sqs.Queue(
        f"{service_name}-dlq",
        name=f"{service_name}-dlq",
        max_message_size=1_048_576,  # 1 MB (AWS SQS default since late 2025)
        message_retention_seconds=14 * 24 * 3600,  # 14 days
        sqs_managed_sse_enabled=True,
    )

    # ── 2. SQS Queue ──
    queue = aws.sqs.Queue(
        f"{service_name}-queue",
        name=f"{service_name}-queue",
        visibility_timeout_seconds=queue_visibility_timeout,
        max_message_size=1_048_576,  # 1 MB (AWS SQS default since late 2025)
        message_retention_seconds=14 * 24 * 3600,  # 14 days
        sqs_managed_sse_enabled=True,
        redrive_policy=dlq.arn.apply(
            lambda arn: json.dumps(
                {
                    "deadLetterTargetArn": arn,
                    "maxReceiveCount": dlq_max_receive_count,
                }
            )
        ),
    )

    # ── 3. EventBridge Rule ──
    rule = aws.cloudwatch.EventRule(
        f"{service_name}-rule",
        name=f"{service_name}-rule",
        description=rule_description
        or f"Routes events to {service_name} SQS queue",
        event_pattern=json.dumps(
            {
                "source": event_sources,
                "detail-type": detail_types,
            }
        ),
    )

    # ── 4. EventBridge Target (rule → queue) ──
    aws.cloudwatch.EventTarget(
        f"{service_name}-target",
        rule=rule.name,
        arn=queue.arn,
        target_id=f"{service_name}-queue",
    )

    # ── 5. SQS Queue Policy (allow EventBridge to send) ──
    aws.sqs.QueuePolicy(
        f"{service_name}-queue-policy",
        queue_url=queue.url,
        policy=pulumi.Output.all(queue.arn, rule.arn).apply(
            lambda args: json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "events.amazonaws.com"},
                            "Action": "sqs:SendMessage",
                            "Resource": args[0],
                            "Condition": {
                                "ArnEquals": {"aws:SourceArn": args[1]}
                            },
                        }
                    ],
                }
            )
        ),
    )

    # ── 6. Secrets Manager Secrets ──
    secret_arns: list[pulumi.Output[str]] = []
    for secret_key, secret_value in secrets.items():
        secret = aws.secretsmanager.Secret(
            f"{service_name}-secret-{secret_key}",
            name=f"/{service_name}/{secret_key}",
            description=f"Secret for {service_name}: {secret_key}",
        )
        aws.secretsmanager.SecretVersion(
            f"{service_name}-secret-{secret_key}-version",
            secret_id=secret.id,
            secret_string=secret_value,
        )
        secret_arns.append(secret.arn)

    # ── 7. IAM Role ──
    role = aws.iam.Role(
        f"{service_name}-ingest-role",
        name=f"{service_name}-ingest-role",
        description=f"Execution role for {service_name}-ingest Lambda",
        assume_role_policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
        ),
    )

    # ── 8. IAM Inline Policy ──
    def _build_policy(args: list) -> str:
        queue_arn_val = args[0]
        log_group = f"arn:aws:logs:{region.name}:{caller.account_id}:log-group:/aws/lambda/{service_name}-ingest:*"

        statements = [
            {
                "Sid": "SQSConsume",
                "Effect": "Allow",
                "Action": [
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:GetQueueAttributes",
                ],
                "Resource": queue_arn_val,
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": log_group,
            },
            {
                "Sid": "XRayTracing",
                "Effect": "Allow",
                "Action": [
                    "xray:PutTraceSegments",
                    "xray:PutTelemetryRecords",
                ],
                "Resource": "*",
            },
        ]

        # Add Secrets Manager permissions if secrets exist
        secret_arn_vals = args[1:]
        if secret_arn_vals:
            statements.append(
                {
                    "Sid": "SecretsManagerRead",
                    "Effect": "Allow",
                    "Action": ["secretsmanager:GetSecretValue"],
                    "Resource": secret_arn_vals,
                }
            )

        return json.dumps({"Version": "2012-10-17", "Statement": statements})

    aws.iam.RolePolicy(
        f"{service_name}-ingest-policy",
        name=f"{service_name}-ingest-policy",
        role=role.id,
        policy=pulumi.Output.all(queue.arn, *secret_arns).apply(_build_policy),
    )

    # ── 9. Lambda Function ──
    # Build env vars: merge user-provided with Secrets Manager secret names
    final_env_vars = dict(lambda_env_vars)
    for secret_key in secrets:
        # Tell Lambda which secret name to fetch
        env_key = f"SECRET_NAME_{secret_key.upper().replace('-', '_')}"
        final_env_vars[env_key] = f"/{service_name}/{secret_key}"

    lambda_fn = aws.lambda_.Function(
        f"{service_name}-ingest",
        name=f"{service_name}-ingest",
        runtime="python3.11",
        architectures=["arm64"],
        handler=lambda_handler,
        role=role.arn,
        code=pulumi.FileArchive(lambda_zip_path),
        memory_size=lambda_memory_mb,
        timeout=lambda_timeout_seconds,
        environment=aws.lambda_.FunctionEnvironmentArgs(
            variables=final_env_vars,
        ),
        tracing_config=aws.lambda_.FunctionTracingConfigArgs(
            mode="Active",
        ),
    )

    # ── 10. Event Source Mapping (SQS → Lambda) ──
    aws.lambda_.EventSourceMapping(
        f"{service_name}-esm",
        event_source_arn=queue.arn,
        function_name=lambda_fn.arn,
        batch_size=1,
        function_response_types=["ReportBatchItemFailures"],
    )

    # ── Exports ──
    first_secret_arn = secret_arns[0] if secret_arns else pulumi.Output.from_input("")

    return ForwarderOutputs(
        dlq_arn=dlq.arn,
        dlq_url=dlq.url,
        queue_arn=queue.arn,
        queue_url=queue.url,
        lambda_arn=lambda_fn.arn,
        lambda_name=lambda_fn.name,
        role_arn=role.arn,
        rule_arn=rule.arn,
        secret_arn=first_secret_arn,
    )
