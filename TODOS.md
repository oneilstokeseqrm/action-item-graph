# TODOs

## DLQ CloudWatch Alarm

**What:** Add a CloudWatch alarm on `ApproximateNumberOfMessagesVisible > 0` for `action-item-graph-dlq` with SNS notification.

**Why:** The DLQ silently accumulates failed messages with no alerting. A CloudWatch alarm with SNS notification (email or Slack webhook) ensures the team is notified when messages fail processing 3 times and land in the DLQ.

**Cost:** Near-zero (~$0.10/month for the alarm + SNS).

**How:** Add `aws.cloudwatch.MetricAlarm` and `aws.sns.Topic` resources to `infra/forwarder.py`. Parameters: `threshold=0`, `evaluation_periods=1`, `period=300`, `statistic="Maximum"`, `comparison_operator="GreaterThanThreshold"`.

**Depends on:** Core IaC adoption (Phase 1–2) — complete.

---

## Investigate DLQ Messages

**What:** Investigate the 23 messages currently in `action-item-graph-dlq` to understand if there's a recurring failure pattern.

**Why:** These are test messages (not production data) that failed 3× before being dead-lettered. Understanding the failure pattern helps determine if there's a systemic issue (e.g., Railway downtime, auth mismatch, malformed envelopes) vs. one-off test noise.

**How:** Use `aws sqs receive-message` to peek at a few messages, check CloudWatch logs for the corresponding Lambda invocations, and look for patterns in error types.

**Note:** SQS retention is 14 days — some messages may have already expired. This is independent of IaC adoption.

**Depends on:** Nothing — can be investigated at any time.
