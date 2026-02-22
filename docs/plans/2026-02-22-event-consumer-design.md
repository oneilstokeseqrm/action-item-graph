# Event Consumer Design — action-item-graph

**Date**: 2026-02-22
**Status**: Approved
**Scope**: AWS infrastructure (EventBridge rule, SQS, Lambda) + Railway API service

---

## 1. Problem

The action-item-graph pipeline exists as a library — it processes `EnvelopeV1` payloads and writes to Neo4j, but has no runtime to receive events from upstream services. Two upstream publishers need to reach this service:

- **live-transcription-fastapi** — publishes `EnvelopeV1.transcript`, `EnvelopeV1.note`, `EnvelopeV1.meeting` to EventBridge (source: `com.yourapp.transcription`)
- **eq-email-pipeline** — publishes `EnvelopeV1.email` to EventBridge (source: `com.eq.email-pipeline`)

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EventBridge (default bus)                     │
│                                                                      │
│  Source: com.yourapp.transcription       Source: com.eq.email-pipeline│
│  DetailType: EnvelopeV1.transcript      DetailType: EnvelopeV1.email │
│              EnvelopeV1.note                                         │
│              EnvelopeV1.meeting                                      │
└──────────────┬──────────────────────────────────┬────────────────────┘
               │                                  │
    ┌──────────▼──────────┐            ┌──────────▼──────────┐
    │ capture-transcripts │            │ action-item-graph-  │
    │ -rule (existing)    │            │ rule (NEW)          │
    │                     │            │ matches BOTH sources│
    └──────────┬──────────┘            └──────────┬──────────┘
               │                                  │
    ┌──────────▼──────────┐            ┌──────────▼──────────┐
    │ meeting-transcripts │            │ action-item-graph-  │
    │ -queue (existing)   │            │ queue (NEW)         │
    │                     │            │                     │
    │ → thematic-lm       │            │ → action-item-graph │
    │   Lambda (existing) │            │   Lambda (NEW)      │
    └─────────────────────┘            └──────────┬──────────┘
                                                  │
                                       ┌──────────▼──────────┐
                                       │ action-item-graph-  │
                                       │ dlq (NEW)           │
                                       │ maxReceiveCount: 3  │
                                       └─────────────────────┘

Lambda (thin forwarder)
       │
       │ HTTPS POST /process
       ▼
┌─────────────────────────┐
│ Railway: FastAPI service │
│                          │
│ EnvelopeDispatcher       │
│   ├─ ActionItemPipeline  │
│   └─ DealPipeline        │
│         │                │
│    Neo4j + OpenAI        │
└─────────────────────────┘
```

### Why this architecture

- **EventBridge fan-out**: One rule can target multiple SQS queues. Each queue gets a copy of every event. No changes to upstream publishers.
- **Own rule (not sharing thematic-lm's)**: We match both `com.yourapp.transcription` and `com.eq.email-pipeline` sources in a single rule. The existing `capture-transcripts-rule` only matches the transcription source.
- **Own SQS queue**: SQS is point-to-point — sharing `meeting-transcripts-queue` with thematic-lm would cause message competition. Separate queues ensure both consumers get every event.
- **Thin Lambda → Railway API**: Lambda parses the EventBridge-wrapped SQS message and POSTs to Railway. Processing happens on Railway where Neo4j and OpenAI clients stay warm (connection pooling, no cold starts).
- **Synchronous processing**: Lambda waits for Railway's response. On success, SQS deletes the message. On failure, SQS retries (up to 3x, then DLQ). No second queue layer needed.
- **Single Railway service**: No separate worker. SQS is the only task queue. Processing takes 60-90s — well within Lambda's 120s timeout.

## 3. AWS Infrastructure

### New resources

| Resource | Name | Configuration |
|---|---|---|
| SQS Queue | `action-item-graph-queue` | VisibilityTimeout=720s, MessageRetention=14d |
| SQS DLQ | `action-item-graph-dlq` | MessageRetention=14d |
| Redrive Policy | Queue → DLQ | maxReceiveCount=3 |
| EventBridge Rule | `action-item-graph-rule` | See event pattern below |
| Rule Target | Rule → SQS | Target: `action-item-graph-queue` |
| SQS Policy | Allow EventBridge → SQS | Condition: rule ARN |
| IAM Role | `action-item-graph-ingest-role` | sqs:ReceiveMessage/DeleteMessage/GetQueueAttributes, logs:*, xray:* |
| Lambda Function | `action-item-graph-ingest` | Python 3.11, arm64, 256MB, 120s timeout |
| Event Source Mapping | SQS → Lambda | BatchSize=1, ReportBatchItemFailures=true |

### EventBridge rule event pattern

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

### Why BatchSize=1

Each envelope triggers 60-90s of OpenAI API calls + Neo4j writes. Batching would increase Lambda execution time proportionally without benefit — the processing is not batchable. Partial batch failure reporting is still enabled for correctness.

## 4. Lambda Package

Lives within this repo at `src/action_item_graph/lambda_ingest/`:

```
src/action_item_graph/lambda_ingest/
├── __init__.py
├── handler.py       # Entry point: Powertools BatchProcessor + Logger + Tracer
├── config.py        # pydantic-settings: API_BASE_URL, WORKER_API_KEY, timeouts
├── envelope.py      # Parse EventBridge-wrapped SQS body → extract EnvelopeV1 JSON
└── api_client.py    # POST to Railway with retry on 5xx, exponential backoff
```

### Handler (Powertools pattern)

```python
from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.batch import BatchProcessor, EventType

processor = BatchProcessor(event_type=EventType.SQS)
logger = Logger(service="action-item-graph-ingest", log_uncaught_exceptions=True)
tracer = Tracer(service="action-item-graph-ingest")

@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
def lambda_handler(event, context):
    return processor.process_partial_response(
        event, record_handler=process_record, context=context
    )

@tracer.capture_method
def process_record(record):
    envelope_json = parse_sqs_record(record.body)
    submit_to_railway(config, envelope_json)
```

### Envelope parsing

SQS body contains an EventBridge event wrapper:

```json
{
  "version": "0",
  "id": "eb-event-id",
  "detail-type": "EnvelopeV1.transcript",
  "source": "com.yourapp.transcription",
  "detail": { ... EnvelopeV1 payload ... }
}
```

The Lambda extracts `detail` and POSTs it as-is to Railway.

### Railway API client

- `httpx` with configurable timeout (default 100s)
- Retry on 5xx with exponential backoff + jitter (max 2 retries)
- No retry on 4xx (persistent error)
- Returns success/failure — BatchProcessor handles SQS ack/nack

### Lambda dependencies

Minimal set, packaged via `scripts/package_lambda.sh`:

- `pydantic`, `pydantic-settings`
- `httpx`
- `aws-lambda-powertools[tracer]`

### Packaging script

Same pattern as thematic-lm: `uv pip install` targeting `aarch64-manylinux2014`, copies only `lambda_ingest/` into zip. No heavy deps (no openai, no neo4j).

### Lambda environment variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | Railway service URL (e.g., `https://action-item-graph-production.up.railway.app`) |
| `WORKER_API_KEY` | Shared secret for bearer token auth |
| `HTTP_TIMEOUT_SECONDS` | httpx timeout (default: 100) |
| `MAX_RETRIES` | Retries for 5xx/network errors (default: 2) |

## 5. Railway Service

Single FastAPI service:

```
src/action_item_graph/api/
├── __init__.py
├── main.py          # FastAPI app, lifespan (init Neo4j + OpenAI)
├── config.py        # pydantic-settings: NEO4J_URI, OPENAI_API_KEY, etc.
├── auth.py          # Bearer token validation
└── routes/
    ├── __init__.py
    ├── process.py   # POST /process — validates envelope, calls dispatcher
    └── health.py    # GET /health — Neo4j connectivity check
```

### Lifespan

Neo4j driver and OpenAI client initialized once at startup, stored in `app.state`. Shared across all requests. Cleaned up on shutdown.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create persistent clients
    app.state.neo4j = Neo4jClient(uri, user, password)
    await app.state.neo4j.connect()
    await app.state.neo4j.setup_schema()
    app.state.openai = OpenAIClient(api_key)
    yield
    # Shutdown: close connections
    await app.state.neo4j.close()
```

### POST /process

```python
@router.post("/process")
async def process_envelope(
    envelope_data: dict[str, Any],
    request: Request,
):
    envelope = EnvelopeV1.model_validate(envelope_data)
    dispatcher = EnvelopeDispatcher(
        neo4j_client=request.app.state.neo4j,
        openai_client=request.app.state.openai,
    )
    result = await dispatcher.dispatch(envelope)
    return result.to_dict()
```

### Authentication

Simple bearer token validation:

```python
async def verify_worker_token(authorization: str = Header(...)):
    if authorization != f"Bearer {settings.WORKER_API_KEY}":
        raise HTTPException(status_code=401)
```

### Health check

```python
@router.get("/health")
async def health(request: Request):
    await request.app.state.neo4j.verify_connectivity()
    return {"status": "ok"}
```

### Railway environment variables

| Variable | Description |
|---|---|
| `NEO4J_URI` | `neo4j+s://c6171c63.databases.neo4j.io` |
| `NEO4J_USERNAME` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |
| `OPENAI_API_KEY` | OpenAI API key |
| `WORKER_API_KEY` | Shared secret (must match Lambda) |

### Railway start command

```
uvicorn action_item_graph.api.main:app --host 0.0.0.0 --port $PORT
```

## 6. EnvelopeV1 Model Changes

To accept email events, three enums in `src/action_item_graph/models/envelope.py` need updates:

### InteractionType

```python
class InteractionType(str, Enum):
    TRANSCRIPT = 'transcript'
    NOTE = 'note'
    DOCUMENT = 'document'
    EMAIL = 'email'        # NEW — from eq-email-pipeline
    MEETING = 'meeting'    # NEW — from EventBridge rule pattern
```

### ContentFormat

```python
class ContentFormat(str, Enum):
    PLAIN = 'plain'
    MARKDOWN = 'markdown'
    DIARIZED = 'diarized'
    EMAIL = 'email'        # NEW — YAML front-matter + cleaned body
```

### SourceType

```python
class SourceType(str, Enum):
    WEB_MIC = 'web-mic'
    UPLOAD = 'upload'
    API = 'api'
    IMPORT = 'import'
    GMAIL = 'gmail'        # NEW — email provider
    OUTLOOK = 'outlook'    # NEW — email provider
```

No changes to EnvelopeV1 fields. Email-specific extras (`subject`, `from_email`, `direction`, `thread_key`, `has_attachments`) travel in the existing `extras` dict.

## 7. Out of Scope

- **Email-specific LLM prompts** — extraction will run on emails using the current transcript-oriented prompt. Optimizing for email content is a separate task.
- **Postgres job table** — no job tracking. SQS handles retry + DLQ.
- **Separate worker service on Railway** — single service, synchronous processing.
- **IaC (Terraform/SAM/CDK)** — AWS resources created via CLI/MCP, consistent with existing infrastructure.
- **Kinesis integration** — ignored per requirements.
- **CI/CD** — Lambda deployment is manual via packaging script + `aws lambda update-function-code`. Railway deploys from GitHub on push.
