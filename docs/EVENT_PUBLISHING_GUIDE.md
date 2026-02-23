# Publishing Events to action-item-graph

This document explains how the action-item-graph event consumer works and how upstream services should publish events to it via Amazon EventBridge.

## Architecture Overview

```
Publisher (your service)
    │
    │  aws events put-events
    ▼
EventBridge (default bus)
    │
    │  Rule: action-item-graph-rule
    ▼
SQS: action-item-graph-queue
    │
    │  Event source mapping (BatchSize=1)
    ▼
Lambda: action-item-graph-ingest
    │
    │  POST /process (Bearer auth)
    ▼
Railway API: api-production-c33a.up.railway.app
    │
    ├── ActionItemPipeline (OpenAI extraction → Neo4j)
    └── DealPipeline (OpenAI extraction → Neo4j)
```

Events flow one-way: publisher → EventBridge → SQS → Lambda → Railway → Neo4j. The publisher receives no callback. Fire-and-forget.

## EventBridge Rule

**Rule name:** `action-item-graph-rule`
**Bus:** `default`
**Region:** `us-east-1`

**Event pattern (what the rule matches):**

```json
{
  "source": ["com.yourapp.transcription", "com.eq.email-pipeline"],
  "detail-type": [
    "EnvelopeV1.transcript",
    "EnvelopeV1.note",
    "EnvelopeV1.meeting",
    "EnvelopeV1.email"
  ]
}
```

Your event must match **both** a listed `source` AND a listed `detail-type` to be routed.

## How to Publish

### Required: EventBridge Entry Format

```python
import boto3, json

client = boto3.client("events", region_name="us-east-1")

client.put_events(Entries=[{
    "Source": "com.eq.email-pipeline",           # must match rule
    "DetailType": "EnvelopeV1.email",            # must match rule
    "Detail": json.dumps(envelope_payload),      # EnvelopeV1 JSON (see below)
    "EventBusName": "default",
}])
```

### Required: EnvelopeV1 Payload Schema

The `Detail` field must be a JSON string containing a valid `EnvelopeV1` object. Here is the complete schema with required/optional annotations:

```json
{
  "schema_version": "v1",

  "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "auth0|abc123def456",
  "interaction_type": "email",
  "content": {
    "text": "The full email body text goes here...",
    "format": "email"
  },
  "timestamp": "2026-02-23T10:30:00Z",
  "source": "gmail",

  "account_id": "acct_acme_corp_001",
  "interaction_id": "019c1fa0-4444-7000-8000-000000000005",
  "pg_user_id": "019c1fa0-3333-7000-8000-000000000003",
  "trace_id": "trace-abc-123",
  "extras": {
    "meeting_title": "Re: Q1 Proposal Follow-up",
    "contact_ids": ["contact_sarah_001"],
    "user_name": "Peter O'Neil"
  }
}
```

### Field Reference

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `schema_version` | No | string | Always `"v1"`. Defaults to `"v1"` if omitted. |
| `tenant_id` | **Yes** | UUID string | Organization/tenant identifier. |
| `user_id` | **Yes** | string | User identifier (Auth0 ID, type-prefixed ID, etc). |
| `interaction_type` | **Yes** | enum | One of: `transcript`, `note`, `document`, `email`, `meeting`. |
| `content.text` | **Yes** | string | The actual content body. |
| `content.format` | No | enum | One of: `plain`, `markdown`, `diarized`, `email`. Defaults to `plain`. |
| `timestamp` | **Yes** | ISO 8601 | Event creation time in UTC. |
| `source` | **Yes** | enum | One of: `web-mic`, `upload`, `api`, `import`, `email-pipeline`, `gmail`, `outlook`. |
| `account_id` | Recommended | string | Account/company identifier. Used to group action items by account in the graph. Strongly recommended — without it, action items cannot be linked to an account node. |
| `interaction_id` | No | UUID string | Unique ID for this interaction. Generated internally if omitted. |
| `pg_user_id` | No | UUID string | Postgres user UUID from identity bridge. |
| `trace_id` | No | string | Distributed tracing ID for observability. |
| `extras` | No | object | Flexible metadata dict (see below). |

### Extras Field

`extras` is a flat key-value dict for domain-specific metadata. Currently recognized keys:

| Key | Type | Purpose |
|-----|------|---------|
| `meeting_title` | string | Title/subject line. Used as context in LLM extraction prompts. |
| `contact_ids` | list[string] | Participant identifiers. Passed to extraction for context. |
| `user_name` | string | Name of the recording/sending user. Enables `is_user_owned` tagging on extracted action items. |
| `opportunity_id` | string | CRM opportunity ID. Stored in extras, available for future use. |
| `duration_seconds` | int | Duration of the interaction. Stored in extras. |

Unknown keys are stored but not actively used. The pipeline is lenient — extra keys won't cause validation errors.

## Email Pipeline: Recommended Values

For the email pipeline specifically:

```python
entry = {
    "Source": "com.eq.email-pipeline",
    "DetailType": "EnvelopeV1.email",
    "Detail": json.dumps({
        "schema_version": "v1",
        "tenant_id": str(tenant_id),
        "user_id": user_id,
        "interaction_type": "email",
        "content": {
            "text": email_body_text,       # full email body
            "format": "email",             # signals email-specific parsing
        },
        "timestamp": email_sent_at.isoformat() + "Z",
        "source": "gmail",                # or "outlook"
        "account_id": account_id,          # strongly recommended
        "extras": {
            "meeting_title": email_subject,   # subject line used as context
            "contact_ids": recipient_ids,     # list of contact IDs
            "user_name": sender_name,         # enables user-owned tagging
        },
    }),
    "EventBusName": "default",
}
```

**Notes:**
- Use `"source": "gmail"` or `"source": "outlook"` depending on the email provider.
- Use `"format": "email"` to signal the content is email-structured.
- `meeting_title` maps naturally to the email subject line — the LLM uses it for extraction context.
- `account_id` is critical — it links extracted action items to the correct account node in Neo4j.

## What Happens After Publishing

1. **EventBridge** matches the event against the rule and delivers it to the SQS queue.
2. **SQS** holds the message. Visibility timeout is 720 seconds (12 minutes) to accommodate long processing times.
3. **Lambda** picks up the message, strips the EventBridge wrapper (extracts the `detail` field), and POSTs the raw EnvelopeV1 JSON to the Railway API.
4. **Railway API** validates the envelope, then runs both pipelines concurrently:
   - **ActionItemPipeline**: Extracts action items via OpenAI, matches against existing items, merges or creates new ones in Neo4j.
   - **DealPipeline**: Extracts deal signals via OpenAI, creates/updates Deal nodes in Neo4j.
5. **Typical processing time**: 15-45 seconds per event (dominated by OpenAI API latency).

## Error Handling

| Failure Point | Behavior |
|---------------|----------|
| EventBridge delivery fails | Automatic retry by EventBridge |
| Lambda crashes or times out | SQS re-delivers after visibility timeout (720s). After 3 failures, message moves to DLQ (`action-item-graph-dlq`, 14-day retention). |
| Railway API returns 4xx | Lambda raises error → SQS retry → DLQ after 3 attempts |
| Railway API returns 5xx | Same retry behavior |
| One pipeline fails, other succeeds | Railway returns partial success (200 with `overall_success: false`). Lambda treats this as success — the envelope was processed. |

The publisher does not need to handle retries. EventBridge + SQS + DLQ provide at-least-once delivery with dead-letter safety.

## Validation Errors

If your payload is malformed, the Railway API returns 422 with details:

```json
{
  "detail": "1 validation error for EnvelopeV1\ntenant_id\n  field required (type=value_error.missing)"
}
```

Common validation failures:
- Missing required fields (`tenant_id`, `user_id`, `interaction_type`, `content`, `timestamp`, `source`)
- Invalid UUID format for `tenant_id`
- Invalid enum value for `interaction_type`, `source`, or `content.format`
- Missing `content.text` (content is an object, not a string)

## AWS Resources Reference

| Resource | ARN / URL |
|----------|-----------|
| EventBridge Rule | `arn:aws:events:us-east-1:211125681610:rule/action-item-graph-rule` |
| SQS Queue | `arn:aws:sqs:us-east-1:211125681610:action-item-graph-queue` |
| SQS DLQ | `arn:aws:sqs:us-east-1:211125681610:action-item-graph-dlq` |
| Lambda | `arn:aws:lambda:us-east-1:211125681610:function:action-item-graph-ingest` |
| Railway API | `https://api-production-c33a.up.railway.app` |

## Testing

Send a test event from the CLI:

```bash
aws events put-events \
  --region us-east-1 \
  --entries '[{
    "Source": "com.eq.email-pipeline",
    "DetailType": "EnvelopeV1.email",
    "Detail": "{\"schema_version\":\"v1\",\"tenant_id\":\"11111111-1111-4111-8111-111111111111\",\"user_id\":\"auth0|test\",\"interaction_type\":\"email\",\"content\":{\"text\":\"Hi team, please send the updated contract by Friday. Also need someone to review the pricing sheet.\",\"format\":\"email\"},\"timestamp\":\"2026-02-23T10:00:00Z\",\"source\":\"gmail\",\"account_id\":\"acct_test_001\",\"extras\":{\"meeting_title\":\"Re: Contract Review\"}}",
    "EventBusName": "default"
  }]'
```

Verify with `FailedEntryCount: 0` in the response. Processing takes 15-45 seconds. Check results in Neo4j:

```cypher
MATCH (ai:ActionItem {account_id: 'acct_test_001'})
RETURN ai.summary, ai.owner, ai.status
```
