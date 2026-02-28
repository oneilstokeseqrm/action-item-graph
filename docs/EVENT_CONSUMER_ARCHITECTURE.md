# Event Consumer Architecture — Replication Guide

> **Audience**: An engineer (or agent) building a **new service** that consumes
> `EnvelopeV1` events from EventBridge using the same pattern as
> `action-item-graph`. This document is self-contained — follow it from top to
> bottom to stand up a fully working ingestion pipeline.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Flow](#2-data-flow)
3. [Step 1 — EventBridge Rule](#3-step-1--eventbridge-rule)
4. [Step 2 — SQS Queue + DLQ](#4-step-2--sqs-queue--dlq)
5. [Step 3 — Lambda Forwarder](#5-step-3--lambda-forwarder)
6. [Step 4 — Railway FastAPI Service](#6-step-4--railway-fastapi-service)
7. [Step 5 — Dispatcher (Concurrent Pipeline Routing)](#7-step-5--dispatcher-concurrent-pipeline-routing)
8. [Step 6 — Deployment](#8-step-6--deployment)
9. [EnvelopeV1 Schema Reference](#9-envelopev1-schema-reference)
10. [Design Decisions & Rationale](#10-design-decisions--rationale)
11. [Checklist](#11-checklist)

---

## 1. System Overview

The ingestion architecture is a **thin-forwarder** pattern: AWS-native event
routing (EventBridge → SQS → Lambda) feeds events to a persistent Railway
FastAPI service where the heavy processing happens. This separation keeps the
Lambda small and cheap (~19 MB, 256 MB memory, <2s cold start) while all
stateful clients (database connections, LLM clients) stay warm on Railway.

```
┌──────────────────────────────┐
│  Upstream Publishers         │
│  (live-transcription-fastapi,│
│   eq-email-pipeline, etc.)   │
└──────────┬───────────────────┘
           │ aws events put-events
           ▼
┌──────────────────────────────┐
│  EventBridge (default bus)   │
│  Rule: {your-service}-rule   │
└──────────┬───────────────────┘
           │ target
           ▼
┌──────────────────────────────┐
│  SQS: {your-service}-queue   │
│  DLQ: {your-service}-dlq     │
│  maxReceiveCount = 3          │
└──────────┬───────────────────┘
           │ event source mapping (BatchSize=1)
           ▼
┌──────────────────────────────┐
│  Lambda: {your-service}-ingest│
│  Thin forwarder (~19 MB)     │
│  parse EB wrapper → POST     │
└──────────┬───────────────────┘
           │ HTTPS + Bearer token
           ▼
┌──────────────────────────────┐
│  Railway FastAPI              │
│  POST /process                │
│  Persistent DB/LLM clients   │
│  EnvelopeDispatcher           │
└──────────┬───────────────────┘
           │ asyncio.gather
      ┌────┴────┐
      ▼         ▼
  Pipeline A  Pipeline B   (fault-isolated)
      │         │
      ▼         ▼
   Neo4j / Postgres / etc.
```

### Why Not Consume SQS Directly on Railway?

SQS can only push to Lambda natively. Railway has no built-in SQS integration.
The Lambda acts as a protocol adapter: it converts SQS pull-based delivery into
an HTTP POST to Railway. This keeps Railway as a standard HTTP service with no
AWS SDK dependency.

---

## 2. Data Flow

1. Upstream service calls `aws events put-events` with an `EnvelopeV1` payload
   in the `detail` field.
2. EventBridge rule matches by `source` and `detail-type`, routes to SQS.
3. SQS event source mapping triggers Lambda with `BatchSize=1`.
4. Lambda strips the EventBridge wrapper (`event["detail"]`), POSTs the raw
   `EnvelopeV1` JSON to Railway with a bearer token.
5. Railway validates the token, parses the envelope, and dispatches to
   concurrent pipelines via `asyncio.gather(return_exceptions=True)`.
6. Each pipeline processes independently — one failing never blocks the other.
7. On success (2xx), SQS deletes the message. On failure, SQS retries up to 3
   times, then routes to DLQ.

---

## 3. Step 1 — EventBridge Rule

### What You Need

A rule on the **default EventBridge bus** that matches events from the upstream
publishers you care about and routes them to your SQS queue.

### Rule Pattern

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

**Adapt this for your service**: include only the `source` and `detail-type`
values you need. For example, if you only care about emails:

```json
{
  "source": ["com.eq.email-pipeline"],
  "detail-type": ["EnvelopeV1.email"]
}
```

### Creating the Rule (CLI)

```bash
# 1. Create the rule
aws events put-rule \
  --name "{your-service}-rule" \
  --event-bus-name default \
  --event-pattern '{
    "source": ["com.yourapp.transcription", "com.eq.email-pipeline"],
    "detail-type": ["EnvelopeV1.transcript", "EnvelopeV1.email"]
  }'

# 2. Set the target (your SQS queue)
aws events put-targets \
  --rule "{your-service}-rule" \
  --targets "Id=sqs-target,Arn=arn:aws:sqs:{region}:{account}:{your-service}-queue"
```

### SQS Resource Policy

EventBridge must have `sqs:SendMessage` permission on your queue. Add this
resource policy to the queue:

```json
{
  "Effect": "Allow",
  "Principal": { "Service": "events.amazonaws.com" },
  "Action": "sqs:SendMessage",
  "Resource": "arn:aws:sqs:{region}:{account}:{your-service}-queue",
  "Condition": {
    "ArnEquals": {
      "aws:SourceArn": "arn:aws:events:{region}:{account}:rule/{your-service}-rule"
    }
  }
}
```

### Important

Each consumer needs its **own SQS queue and EventBridge rule**. SQS is
point-to-point — if two services share a queue, each message goes to only one
of them. Separate queues ensure every consumer receives every matched event.

---

## 4. Step 2 — SQS Queue + DLQ

### Queue Configuration

| Resource | Name | Configuration |
|----------|------|---------------|
| Main Queue | `{your-service}-queue` | `VisibilityTimeout=720s`, `MessageRetentionPeriod=14d` |
| Dead Letter Queue | `{your-service}-dlq` | `MessageRetentionPeriod=14d` |
| Redrive Policy | queue → dlq | `maxReceiveCount=3` |

### Creating the Queues (CLI)

```bash
# 1. Create the DLQ first
aws sqs create-queue \
  --queue-name "{your-service}-dlq" \
  --attributes '{"MessageRetentionPeriod": "1209600"}'

# 2. Get the DLQ ARN
DLQ_ARN=$(aws sqs get-queue-attributes \
  --queue-url "https://sqs.{region}.amazonaws.com/{account}/{your-service}-dlq" \
  --attribute-names QueueArn --query 'Attributes.QueueArn' --output text)

# 3. Create the main queue with redrive policy
aws sqs create-queue \
  --queue-name "{your-service}-queue" \
  --attributes '{
    "VisibilityTimeout": "720",
    "MessageRetentionPeriod": "1209600",
    "RedrivePolicy": "{\"deadLetterTargetArn\": \"'$DLQ_ARN'\", \"maxReceiveCount\": 3}"
  }'
```

### Why These Values?

- **`VisibilityTimeout = 720s`** (12 minutes): Must exceed `Lambda timeout ×
  (1 + max retries) + buffer`. With a 120s Lambda timeout and 2 retries, worst
  case is ~360s. 720s provides a safe margin so a message doesn't become
  visible again while Lambda is still retrying.
- **`maxReceiveCount = 3`**: Three attempts before parking in DLQ. Balances
  retry coverage against runaway processing of poison messages.
- **`MessageRetentionPeriod = 14d`**: Maximum retention, gives you time to
  investigate DLQ messages.

---

## 5. Step 3 — Lambda Forwarder

The Lambda is intentionally thin — its only job is to bridge SQS to your
Railway HTTP endpoint. No business logic, no heavy dependencies.

### Directory Structure

```
src/{your_package}/lambda_ingest/
├── __init__.py       # empty
├── handler.py        # Lambda entry point
├── config.py         # pydantic-settings config
├── envelope.py       # EventBridge wrapper parser
└── api_client.py     # HTTP client with retry logic
```

### handler.py — Lambda Entry Point

```python
"""Lambda entry point: SQS → parse EventBridge envelope → POST to Railway."""

from aws_lambda_powertools import Logger, Tracer
from aws_lambda_powertools.utilities.batch import (
    BatchProcessor,
    EventType,
    process_partial_response,
)
from aws_lambda_powertools.utilities.data_classes.sqs_event import SQSRecord

from .config import LambdaConfig
from .envelope import parse_sqs_record_body
from .api_client import submit_to_railway

# Module-level singletons — survive across warm invocations
processor = BatchProcessor(
    event_type=EventType.SQS, raise_on_entire_batch_failure=False
)
logger = Logger(service="{your-service}-ingest", log_uncaught_exceptions=True)
tracer = Tracer(service="{your-service}-ingest")

_config: LambdaConfig | None = None


def _get_config() -> LambdaConfig:
    """Lazy-init config singleton (avoids cold-start failures)."""
    global _config
    if _config is None:
        _config = LambdaConfig()
    return _config


@tracer.capture_method
def process_record(record: SQSRecord) -> None:
    """Process a single SQS record containing an EventBridge-wrapped envelope."""
    config = _get_config()
    envelope_json = parse_sqs_record_body(record.body)

    tenant_id = envelope_json.get("tenant_id", "unknown")
    interaction_type = envelope_json.get("interaction_type", "unknown")

    logger.info(
        "record.processing",
        extra={
            "tenant_id": tenant_id,
            "interaction_type": interaction_type,
            "message_id": record.message_id,
        },
    )

    result = submit_to_railway(config, envelope_json)

    if not result.success:
        logger.error(
            "record.failed",
            extra={
                "tenant_id": tenant_id,
                "status_code": result.status_code,
                "error": result.error,
                "message_id": record.message_id,
            },
        )
        # Raising causes Powertools to mark this record as failed in
        # batchItemFailures — SQS will retry just this record.
        raise RuntimeError(
            f"Railway API failure: {result.status_code} — {result.error}"
        )

    logger.info(
        "record.success",
        extra={
            "tenant_id": tenant_id,
            "interaction_type": interaction_type,
            "status_code": result.status_code,
            "message_id": record.message_id,
        },
    )


@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
def lambda_handler(event: dict, context) -> dict:
    """Lambda entry point — SQS batch with partial failure reporting."""
    return process_partial_response(
        event=event,
        record_handler=process_record,
        processor=processor,
        context=context,
    )
```

**Key patterns:**

- `raise_on_entire_batch_failure=False` + `ReportBatchItemFailures` on the
  event source mapping = partial failure mode. If record 2 of 3 fails, only
  record 2 is retried.
- `log_event=False` keeps raw transcript/email content out of CloudWatch logs.
- `_config` is lazy-initialized (not at import time) so a misconfigured env
  var shows up as a clear runtime error, not an import crash.
- Module-level `processor`, `logger`, `tracer` are singletons that persist
  across warm Lambda invocations.

### config.py — Lambda Configuration

```python
"""Configuration for the Lambda forwarder."""

from pydantic import Field
from pydantic_settings import BaseSettings


class LambdaConfig(BaseSettings):
    """Lambda environment variables."""

    API_BASE_URL: str                                       # Railway service URL
    WORKER_API_KEY: str                                     # Shared bearer secret
    HTTP_TIMEOUT_SECONDS: int = Field(default=100, ge=5, le=120)
    MAX_RETRIES: int = Field(default=2, ge=0, le=5)
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | — | Railway service URL (e.g., `https://{service}.up.railway.app`) |
| `WORKER_API_KEY` | Yes | — | Shared secret, must match Railway's `WORKER_API_KEY` |
| `HTTP_TIMEOUT_SECONDS` | No | `100` | httpx client timeout (seconds) |
| `MAX_RETRIES` | No | `2` | Retries for 5xx / network errors (0 = no retries) |

### envelope.py — EventBridge Wrapper Parser

```python
"""Extract EnvelopeV1 payload from EventBridge-wrapped SQS body."""

import json
from typing import Any


def parse_sqs_record_body(body: str) -> dict[str, Any]:
    """
    SQS body from EventBridge looks like:
    {
        "version": "0",
        "id": "eb-event-id",
        "detail-type": "EnvelopeV1.transcript",
        "source": "com.yourapp.transcription",
        "detail": { ... EnvelopeV1 payload ... }
    }

    Returns just the "detail" dict (the raw EnvelopeV1 JSON).
    """
    try:
        event = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in SQS body: {e}") from e

    if "detail" not in event:
        raise ValueError(
            f"Missing 'detail' key in EventBridge event. "
            f"Keys present: {list(event.keys())}"
        )

    return event["detail"]
```

### api_client.py — HTTP Client with Retry

```python
"""HTTP client for forwarding envelopes to the Railway API."""

import random
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class SubmitResult:
    success: bool
    status_code: int | None = None
    error: str | None = None


def submit_to_railway(config: Any, envelope_json: dict[str, Any]) -> SubmitResult:
    """
    POST the envelope JSON to the Railway /process endpoint.

    Retry strategy:
    - 2xx → success, return immediately
    - 4xx → persistent error, return failure immediately (no retry)
    - 5xx / network error → retry with exponential backoff + jitter
    """
    url = f"{config.API_BASE_URL.rstrip('/')}/process"
    headers = {
        "Authorization": f"Bearer {config.WORKER_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error: str | None = None
    last_status: int | None = None

    for attempt in range(1 + config.MAX_RETRIES):
        try:
            with httpx.Client(timeout=config.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(url, json=envelope_json, headers=headers)
                response.raise_for_status()
                return SubmitResult(success=True, status_code=response.status_code)

        except httpx.HTTPStatusError as e:
            last_status = e.response.status_code
            last_error = f"HTTP {last_status}"
            if 400 <= last_status < 500:
                return SubmitResult(
                    success=False, status_code=last_status, error=last_error
                )
            if attempt < config.MAX_RETRIES:
                _backoff_sleep(attempt)

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = f"{type(e).__name__}: {e}"
            last_status = None
            if attempt < config.MAX_RETRIES:
                _backoff_sleep(attempt)

    return SubmitResult(success=False, status_code=last_status, error=last_error)


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff: 2^attempt + uniform jitter (0..50% of base)."""
    base = 2 ** attempt
    jitter = random.uniform(0, base * 0.5)
    time.sleep(base + jitter)
```

**Retry math**: attempt 0 → sleep 1.0–1.5s, attempt 1 → sleep 2.0–3.0s. Max
3 total attempts (1 initial + 2 retries). Each attempt opens a fresh
synchronous `httpx.Client` — Lambda doesn't need async.

### Lambda Resource Configuration

| Setting | Value |
|---------|-------|
| Function name | `{your-service}-ingest` |
| Runtime | `python3.11` |
| Architecture | `arm64` |
| Memory | `256 MB` |
| Timeout | `120 seconds` |
| Handler | `{your_package}.lambda_ingest.handler.lambda_handler` |

### Lambda IAM Role

The execution role (`{your-service}-ingest-role`) needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:{region}:{account}:{your-service}-queue"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:{region}:{account}:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "xray:PutTraceSegments",
        "xray:PutTelemetryRecords"
      ],
      "Resource": "*"
    }
  ]
}
```

### Event Source Mapping

```bash
aws lambda create-event-source-mapping \
  --function-name "{your-service}-ingest" \
  --event-source-arn "arn:aws:sqs:{region}:{account}:{your-service}-queue" \
  --batch-size 1 \
  --function-response-types '["ReportBatchItemFailures"]'
```

- **`BatchSize=1`**: Each envelope takes 60–90s to process (LLM calls).
  Batching would increase Lambda execution time proportionally without benefit.
- **`ReportBatchItemFailures`**: Enables partial failure mode with Powertools
  `BatchProcessor`.

### Lambda Packaging

The Lambda zip must be self-contained and exclude heavy pipeline deps. Use a
packaging script like `scripts/package_lambda.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR=$(mktemp -d)
OUTPUT_DIR="$PROJECT_DIR/dist"
OUTPUT="$OUTPUT_DIR/{your-service}-ingest.zip"

trap 'rm -rf "$BUILD_DIR"' EXIT

echo "=== Building Lambda package ==="

# 1. Install ONLY the Lambda's minimal dependencies (arm64 target)
uv pip install \
    --target "$BUILD_DIR" \
    --python-platform aarch64-manylinux2014 \
    --python-version 3.11 \
    --only-binary :all: \
    pydantic pydantic-settings httpx "aws-lambda-powertools[tracer]"

# 2. Copy ONLY the lambda_ingest subpackage (not the full application)
mkdir -p "$BUILD_DIR/{your_package}/lambda_ingest"
# Stub __init__.py — the real one imports heavy deps (neo4j, openai, etc.)
echo '"""Lambda-compatible stub."""' > "$BUILD_DIR/{your_package}/__init__.py"
cp "$PROJECT_DIR/src/{your_package}/lambda_ingest/"*.py \
   "$BUILD_DIR/{your_package}/lambda_ingest/"

# 3. Create deployment zip
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT"
cd "$BUILD_DIR"
zip -r9 "$OUTPUT" . -x "*.pyc" "__pycache__/*"

echo "Output: $OUTPUT"
du -h "$OUTPUT"
```

**Critical detail**: The package `__init__.py` is replaced with a minimal stub.
The real `__init__.py` eagerly imports pipeline code that depends on `openai`,
`neo4j`, etc. — none of which are in the Lambda zip. Without the stub, the
Lambda would crash on import.

### Deploying the Lambda

```bash
# Package
./scripts/package_lambda.sh

# Deploy
aws lambda update-function-code \
  --function-name "{your-service}-ingest" \
  --zip-file "fileb://dist/{your-service}-ingest.zip"
```

---

## 6. Step 4 — Railway FastAPI Service

The persistent service where all the heavy processing happens. Database
connections, LLM clients, and other stateful resources stay warm across
requests.

### Directory Structure

```
src/{your_package}/api/
├── __init__.py       # empty
├── main.py           # FastAPI app + lifespan
├── config.py         # pydantic-settings for Railway env vars
├── auth.py           # Bearer token validation
└── routes/
    ├── __init__.py   # empty
    ├── process.py    # POST /process
    └── health.py     # GET /health

Procfile                # Railway start command
pyproject.toml          # [api] optional dep group
```

### main.py — FastAPI Application with Lifespan

```python
"""FastAPI application for the Railway service."""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from {your_package}.clients.neo4j_client import Neo4jClient
from {your_package}.clients.openai_client import OpenAIClient

from .config import get_settings
from .routes.health import router as health_router
from .routes.process import router as process_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize persistent clients at startup, clean up at shutdown."""
    settings = get_settings()

    logger.info("lifespan.startup")

    # Initialize your clients here — they persist across requests
    neo4j = Neo4jClient(
        uri=settings.NEO4J_URI,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD,
        database=settings.NEO4J_DATABASE,
    )
    await neo4j.connect()
    await neo4j.setup_schema()    # idempotent (IF NOT EXISTS)

    openai = OpenAIClient(api_key=settings.OPENAI_API_KEY)

    # Store on app.state for request handlers
    app.state.neo4j = neo4j
    app.state.openai = openai

    logger.info("lifespan.ready")
    yield

    # Shutdown — clean up connections
    logger.info("lifespan.shutdown")
    await neo4j.close()


app = FastAPI(
    title="{your-service}",
    description="EnvelopeV1 event consumer",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(process_router)
```

**Key patterns:**

- `@asynccontextmanager` lifespan manages client lifecycle — everything before
  `yield` runs at startup, everything after runs at shutdown.
- Clients are stored on `app.state` (FastAPI's built-in request-scoped state)
  and accessed in route handlers via `request.app.state.{client}`.
- Schema setup is idempotent (`IF NOT EXISTS` DDL) — safe to run on every
  deploy.
- If you have optional clients (like Postgres dual-write), make them
  failure-isolated at startup:

```python
postgres: PostgresClient | None = None
if settings.NEON_DATABASE_URL:
    pg = PostgresClient(settings.NEON_DATABASE_URL)
    await pg.connect()
    if await pg.verify_connectivity():
        postgres = pg
    else:
        logger.warning("lifespan.postgres_connectivity_failed")
        await pg.close()

app.state.postgres = postgres
```

### config.py — Railway Settings

```python
"""Configuration for the Railway FastAPI service."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Railway service settings from environment variables."""

    # Your database / service clients
    NEO4J_URI: str
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"

    OPENAI_API_KEY: str

    # Auth — must match Lambda's WORKER_API_KEY
    WORKER_API_KEY: str

    # Optional services
    NEON_DATABASE_URL: str = ""


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
```

### auth.py — Bearer Token Validation

```python
"""Bearer token authentication for the Railway API."""

from fastapi import Header, HTTPException
from .config import get_settings


async def verify_worker_token(authorization: str = Header(...)) -> None:
    """Validate bearer token from the Lambda forwarder."""
    expected = f"Bearer {get_settings().WORKER_API_KEY}"
    if authorization != expected:
        raise HTTPException(
            status_code=401, detail="Invalid or missing bearer token"
        )
```

Used as a FastAPI dependency: `_auth: None = Depends(verify_worker_token)`.

### routes/process.py — POST /process

```python
"""POST /process — validate envelope and dispatch through pipelines."""

import structlog
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from {your_package}.models.envelope import EnvelopeV1
from {your_package}.pipeline.pipeline import YourPipeline
from dispatcher.dispatcher import EnvelopeDispatcher

from ..auth import verify_worker_token

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/process")
async def process_envelope(
    envelope_data: dict[str, Any],
    request: Request,
    _auth: None = Depends(verify_worker_token),
):
    """Validate an EnvelopeV1 payload and dispatch through pipelines."""
    # Parse raw dict → validated EnvelopeV1
    try:
        envelope = EnvelopeV1.model_validate(envelope_data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    log = logger.bind(
        tenant_id=str(envelope.tenant_id),
        interaction_type=envelope.interaction_type.value,
    )
    log.info("process.received")

    try:
        # Build pipeline(s) from app.state clients
        pipeline = YourPipeline(
            openai_client=request.app.state.openai,
            neo4j_client=request.app.state.neo4j,
        )
        result = await pipeline.process_envelope(envelope)
    except Exception as e:
        log.error("process.failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "overall_success": False},
        )

    log.info("process.complete")
    return result.to_dict()
```

**Notes:**
- Request body is `dict[str, Any]` (raw JSON), validated explicitly via
  `model_validate()` — this gives you control over the 422 error format.
- On pipeline failure: returns 500 with JSON body (not an unhandled exception).
  This allows the Lambda to distinguish 4xx (don't retry) from 5xx (do retry).

### routes/health.py — GET /health

```python
"""Health check endpoint."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Check primary database connectivity."""
    try:
        await request.app.state.neo4j.verify_connectivity()
        return {"status": "ok"}
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "unhealthy"})
```

### Procfile

```
web: uvicorn {your_package}.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Single `web` process. Railway reads this automatically. `${PORT:-8000}` uses
Railway's injected `PORT` in production, falls back to 8000 for local dev.

### pyproject.toml — Package Structure

The key sections for the Railway deployment:

```toml
[project]
name = "{your-service}"
requires-python = ">=3.10"
dependencies = [
    # Your core deps (pipeline, database clients, etc.)
    "pydantic>=2.11.0",
    "neo4j>=5.26.0",
    "openai>=1.50.0",
    "structlog>=24.0.0",
    # ...
]

[project.optional-dependencies]
api = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic-settings>=2.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{your_package}", "src/dispatcher"]
```

The `[api]` optional dependency group is installed on Railway but **not**
included in the Lambda zip. This keeps the two deployment surfaces independent.

Railway must install with the api extra:

```bash
pip install -e ".[api]"
# or
uv sync --extra api
```

---

## 7. Step 5 — Dispatcher (Concurrent Pipeline Routing)

If you have multiple pipelines that should run concurrently with fault
isolation, use an `EnvelopeDispatcher`:

```python
"""Concurrent pipeline dispatcher with fault isolation."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from {your_package}.models.envelope import EnvelopeV1

logger = structlog.get_logger(__name__)


@dataclass
class DispatcherResult:
    tenant_id: str
    pipeline_a_result: Any | BaseException | None = None
    pipeline_b_result: Any | BaseException | None = None
    dispatch_time_ms: int | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def pipeline_a_success(self) -> bool:
        return not isinstance(self.pipeline_a_result, BaseException)

    @property
    def pipeline_b_success(self) -> bool:
        return not isinstance(self.pipeline_b_result, BaseException)

    @property
    def overall_success(self) -> bool:
        """True if at least one pipeline succeeded."""
        return self.pipeline_a_success or self.pipeline_b_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "pipeline_a_success": self.pipeline_a_success,
            "pipeline_b_success": self.pipeline_b_success,
            "overall_success": self.overall_success,
            "dispatch_time_ms": self.dispatch_time_ms,
            "errors": self.errors,
        }


class EnvelopeDispatcher:
    def __init__(self, pipeline_a, pipeline_b):
        self.pipeline_a = pipeline_a
        self.pipeline_b = pipeline_b

    async def dispatch(self, envelope: EnvelopeV1) -> DispatcherResult:
        t0 = time.monotonic()
        result = DispatcherResult(tenant_id=str(envelope.tenant_id))

        # Run both pipelines concurrently — one failing never blocks the other
        outcomes = await asyncio.gather(
            self.pipeline_a.process_envelope(envelope),
            self.pipeline_b.process_envelope(envelope),
            return_exceptions=True,
        )

        for i, (name, outcome) in enumerate(
            zip(["pipeline_a", "pipeline_b"], outcomes)
        ):
            if isinstance(outcome, BaseException):
                logger.error(f"dispatcher.{name}_failed", error=str(outcome))
                setattr(result, f"{name}_result", outcome)
                result.errors.append(f"{name}: {type(outcome).__name__}: {outcome}")
            else:
                setattr(result, f"{name}_result", outcome)

        result.dispatch_time_ms = int((time.monotonic() - t0) * 1000)
        return result
```

**The critical line is `return_exceptions=True`**: without it, the first
exception would cancel the other coroutine. With it, exceptions are returned as
values in the results tuple — each pipeline runs to completion independently.

If you only have a single pipeline, skip the dispatcher and call
`pipeline.process_envelope(envelope)` directly in the route handler.

---

## 8. Step 6 — Deployment

### Railway

1. **Create a Railway project** and connect your GitHub repo.
2. **Set environment variables** in the Railway dashboard:

   | Variable | Description |
   |----------|-------------|
   | `NEO4J_URI` | Database connection URI |
   | `NEO4J_USERNAME` | Database username (default: `neo4j`) |
   | `NEO4J_PASSWORD` | Database password |
   | `NEO4J_DATABASE` | Database name (default: `neo4j`) |
   | `OPENAI_API_KEY` | OpenAI API key |
   | `WORKER_API_KEY` | Shared bearer secret (must match Lambda) |
   | `NEON_DATABASE_URL` | *(optional)* Postgres URL for dual-write |
   | `PORT` | *(auto-injected by Railway)* |

3. **Railway reads the `Procfile`** and runs the `web` process automatically.
4. **Auto-deploy on push**: Railway deploys from main on every git push
   (standard GitHub integration, no config files needed).
5. **No `railway.toml`, Dockerfile, or nixpacks config** required. Railway's
   Python buildpack detects `pyproject.toml` and installs dependencies
   automatically. Ensure the `[api]` extra is installed (Railway should detect
   `fastapi` in optional deps).

### Lambda

1. Run `./scripts/package_lambda.sh` to build the zip.
2. Deploy with:
   ```bash
   aws lambda update-function-code \
     --function-name "{your-service}-ingest" \
     --zip-file "fileb://dist/{your-service}-ingest.zip"
   ```
3. Set Lambda environment variables: `API_BASE_URL`, `WORKER_API_KEY`,
   `HTTP_TIMEOUT_SECONDS`, `MAX_RETRIES`.
4. Verify the event source mapping is active:
   ```bash
   aws lambda list-event-source-mappings \
     --function-name "{your-service}-ingest"
   ```

### Generating the Shared Secret

```bash
# Generate a random 32-byte hex token
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Set this value as `WORKER_API_KEY` in **both** the Lambda env vars and the
Railway env vars. They must match exactly.

---

## 9. EnvelopeV1 Schema Reference

`EnvelopeV1` is the standardized event envelope shared across all consumers.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `tenant_id` | `UUID` | Tenant/organization identifier |
| `user_id` | `str` | User identifier (Auth0 ID, type-prefixed ID, etc.) |
| `interaction_type` | `enum` | `transcript`, `note`, `document`, `email`, `meeting` |
| `content` | `ContentPayload` | `{text: str, format: "plain"|"markdown"|"diarized"|"email"}` |
| `timestamp` | `datetime` | Event creation time (UTC, ISO 8601) |
| `source` | `enum` | `web-mic`, `upload`, `api`, `import`, `email-pipeline`, `gmail`, `outlook` |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `schema_version` | `str` | `"v1"` | Schema version |
| `pg_user_id` | `UUID \| None` | `None` | Postgres user UUID from identity bridge |
| `extras` | `dict[str, Any]` | `{}` | Domain-specific metadata (`opportunity_id`, `contact_ids`, etc.) |
| `interaction_id` | `UUID \| None` | `None` | Unique interaction ID (generated if not provided) |
| `trace_id` | `str \| None` | `None` | Distributed tracing ID |
| `account_id` | `str \| None` | `None` | Account identifier for sales context |

### EventBridge Detail Types

Publishers set `detail-type` based on the interaction type:

| detail-type | Source | Description |
|-------------|--------|-------------|
| `EnvelopeV1.transcript` | `com.yourapp.transcription` | Live call transcript |
| `EnvelopeV1.note` | `com.yourapp.transcription` | Manual note |
| `EnvelopeV1.meeting` | `com.yourapp.transcription` | Meeting summary |
| `EnvelopeV1.email` | `com.eq.email-pipeline` | Email message |

### Example Payload

```json
{
  "schema_version": "v1",
  "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "auth0|abc123def456",
  "interaction_type": "transcript",
  "content": {
    "text": "John: Thanks for joining the call today...",
    "format": "diarized"
  },
  "timestamp": "2025-01-23T10:30:00Z",
  "source": "web-mic",
  "extras": {
    "opportunity_id": "019c1fa0-4444-7000-8000-000000000005",
    "contact_ids": ["contact_sarah_001"]
  },
  "account_id": "acct_acme_corp_001"
}
```

---

## 10. Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Thin Lambda forwarder** | Keeps Lambda small (~19 MB), fast cold starts (<2s), cheap (256 MB). All heavy deps (openai, neo4j, sqlalchemy) live on Railway where connections stay warm. |
| **Separate SQS queue per consumer** | SQS is point-to-point. Sharing a queue causes message competition — each message goes to only one consumer. |
| **Own EventBridge rule per consumer** | The existing `capture-transcripts-rule` only matches one source. New consumers may need different source/detail-type combinations. |
| **`BatchSize=1`** | Each envelope takes 60–90s to process (LLM calls). Batching increases Lambda timeout without benefit. |
| **Synchronous Lambda** | Lambda waits for Railway's HTTP response. Success = SQS deletes message. Failure = SQS retries. Simple, reliable. |
| **Bearer token auth** | Simple shared secret between Lambda and Railway. Both sides store the same `WORKER_API_KEY`. No OAuth complexity needed for internal service-to-service calls. |
| **`return_exceptions=True`** | Without it, `asyncio.gather` cancels remaining coroutines when one throws. With it, each pipeline runs to completion independently — partial success is better than all-or-nothing. |
| **`overall_success = any_succeeded`** | A 200 response (Lambda acks, SQS deletes) means at least one pipeline produced useful results. If one pipeline is consistently failing, the other keeps working. Monitor `errors` in the response for partial failures. |
| **`VisibilityTimeout=720s`** | Must exceed worst-case processing time. Lambda timeout (120s) × max attempts (3) = 360s. 720s provides 2× safety margin. |
| **No IaC** | AWS resources created via CLI, consistent with existing infrastructure. Each service only needs ~5 resources (rule, queue, DLQ, Lambda, event source mapping). |
| **`log_event=False`** | Prevents full transcript/email content from appearing in CloudWatch logs (privacy, cost). |
| **Stub `__init__.py` in Lambda zip** | The real package `__init__.py` eagerly imports pipeline code with heavy deps. The stub prevents import crashes in the Lambda environment. |
| **`raise_on_entire_batch_failure=False`** | Partial failure mode: if one record fails in a batch, only that record is retried. (Moot with `BatchSize=1` but future-proofs for batching.) |
| **Postgres dual-write is optional** | If `NEON_DATABASE_URL` is empty or connectivity fails, the service runs without Postgres. Neo4j is the primary store — Postgres is a projection that must never block the main write path. |

---

## 11. Checklist

Use this when standing up a new consumer service.

### AWS Resources

- [ ] Create SQS DLQ: `{your-service}-dlq`
- [ ] Create SQS queue: `{your-service}-queue` (VisibilityTimeout=720,
      redrive to DLQ, maxReceiveCount=3)
- [ ] Add SQS resource policy allowing EventBridge `sqs:SendMessage`
- [ ] Create EventBridge rule: `{your-service}-rule` with your event pattern
- [ ] Set EventBridge rule target to your SQS queue
- [ ] Create IAM role: `{your-service}-ingest-role` with SQS + logs + X-Ray
      permissions
- [ ] Create Lambda: `{your-service}-ingest` (python3.11, arm64, 256MB, 120s)
- [ ] Set Lambda env vars: `API_BASE_URL`, `WORKER_API_KEY`
- [ ] Create event source mapping: queue → Lambda (BatchSize=1,
      ReportBatchItemFailures)
- [ ] Generate shared secret (`secrets.token_hex(32)`) for `WORKER_API_KEY`

### Lambda Code

- [ ] `handler.py` — Powertools BatchProcessor + process_partial_response
- [ ] `config.py` — pydantic-settings LambdaConfig
- [ ] `envelope.py` — parse EventBridge wrapper, extract `detail`
- [ ] `api_client.py` — httpx POST with exponential backoff + jitter
- [ ] `package_lambda.sh` — minimal zip with stub `__init__.py`

### Railway Service

- [ ] `Procfile` — `web: uvicorn ...`
- [ ] `main.py` — FastAPI app with lifespan (client init/cleanup)
- [ ] `config.py` — pydantic-settings for Railway env vars
- [ ] `auth.py` — Bearer token validation dependency
- [ ] `routes/process.py` — POST /process handler
- [ ] `routes/health.py` — GET /health handler
- [ ] Set Railway env vars: `NEO4J_*`, `OPENAI_API_KEY`, `WORKER_API_KEY`
- [ ] `pyproject.toml` with `[api]` optional dep group
- [ ] Connect GitHub repo to Railway for auto-deploy

### Verification

- [ ] Deploy Lambda, send a test event via EventBridge CLI
- [ ] Verify Lambda CloudWatch logs show `record.success`
- [ ] Verify Railway logs show `process.complete`
- [ ] Verify health endpoint returns `{"status": "ok"}`
- [ ] Intentionally trigger a failure — verify message lands in DLQ after 3
      retries
