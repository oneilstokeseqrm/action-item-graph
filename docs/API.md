# Action Item Graph - API Reference

This document provides detailed API documentation for all public classes and functions.

## Table of Contents

- [Pipeline](#pipeline)
  - [ActionItemPipeline](#actionitempipeline)
  - [PipelineResult](#pipelineresult)
- [Components](#components)
  - [ActionItemExtractor](#actionitemextractor)
  - [ActionItemMatcher](#actionitemmatcher)
  - [ActionItemMerger](#actionitemmerger)
  - [TopicResolver](#topicresolver)
  - [TopicExecutor](#topicexecutor)
- [Repository](#repository)
  - [ActionItemRepository](#actionitemrepository)
- [Models](#models)
  - [EnvelopeV1](#envelopev1)
  - [ActionItem](#actionitem)
  - [ExtractedActionItem](#extractedactionitem)
  - [Topic](#topic)
  - [ExtractedTopic](#extractedtopic)
- [Clients](#clients)
  - [Neo4jClient](#neo4jclient)
  - [OpenAIClient](#openaiclient)
- [Logging](#logging)
- [Errors](#errors)

---

## Pipeline

### ActionItemPipeline

The main orchestrator that coordinates extraction, matching, and merging.

```python
from action_item_graph import ActionItemPipeline
from action_item_graph.clients import Neo4jClient, OpenAIClient

pipeline = ActionItemPipeline(
    openai_client=OpenAIClient(),
    neo4j_client=Neo4jClient(),
)
```

#### Methods

##### `process_text()`

Process raw transcript text through the full pipeline.

```python
async def process_text(
    self,
    text: str,
    tenant_id: UUID,
    account_id: str,
    meeting_title: str | None = None,
    participants: list[str] | None = None,
    user_id: str = "system",
) -> PipelineResult:
```

**Parameters:**
- `text`: The transcript text to process
- `tenant_id`: UUID for tenant isolation
- `account_id`: Account identifier for grouping related items
- `meeting_title`: Optional title for the interaction
- `participants`: Optional list of participant names
- `user_id`: User who triggered the processing (default: "system")

**Returns:** `PipelineResult` with extraction and merge outcomes

**Example:**
```python
result = await pipeline.process_text(
    text="John: I'll send the proposal by Friday.",
    tenant_id=UUID("11111111-1111-4111-8111-111111111111"),
    account_id="acct_acme",
    meeting_title="Sales Call",
)
```

##### `process_envelope()`

Process an EnvelopeV1 payload through the full pipeline.

```python
async def process_envelope(
    self,
    envelope: EnvelopeV1,
) -> PipelineResult:
```

**Parameters:**
- `envelope`: An EnvelopeV1 object containing transcript and metadata

**Returns:** `PipelineResult` with extraction and merge outcomes

##### `get_action_items()`

Query action items from the graph.

```python
async def get_action_items(
    self,
    tenant_id: UUID,
    account_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
```

**Parameters:**
- `tenant_id`: UUID for tenant isolation (required)
- `account_id`: Filter by account (optional)
- `status`: Filter by status: "open", "in_progress", "completed", "cancelled", "deferred"
- `limit`: Maximum number of results (default: 50)

**Returns:** List of action item dictionaries

---

### PipelineResult

Dataclass containing the results of pipeline processing.

```python
@dataclass
class PipelineResult:
    total_extracted: int = 0      # Total items extracted
    total_new_items: int = 0      # New action items (not status updates)
    total_status_updates: int = 0 # Status updates extracted
    total_matched: int = 0        # Items matched to existing
    total_unmatched: int = 0      # Items that are new
    created_ids: list[str]        # IDs of created nodes
    updated_ids: list[str]        # IDs of updated nodes
    linked_ids: list[str]         # IDs of linked nodes
    processing_time_ms: float     # Total processing time
    stage_timings: dict           # Per-stage timing
    merge_results: list[MergeResult]  # Detailed merge info
```

#### Methods

##### `to_dict()`

Convert to dictionary for serialization.

```python
def to_dict(self) -> dict:
```

---

## Components

### ActionItemExtractor

Extracts action items from text using GPT-4.1-mini.

```python
from action_item_graph.pipeline import ActionItemExtractor

extractor = ActionItemExtractor(openai_client=openai)
```

#### Methods

##### `extract()`

Extract action items from transcript text.

```python
async def extract(
    self,
    text: str,
    meeting_title: str | None = None,
    participants: list[str] | None = None,
) -> ExtractionOutput:
```

**Returns:** `ExtractionOutput` containing:
- `action_items`: List of new action items
- `status_updates`: List of status updates to existing items
- `embeddings`: Generated embeddings for each item

---

### ActionItemMatcher

Matches extracted items against existing items in the graph.

```python
from action_item_graph.pipeline import ActionItemMatcher

matcher = ActionItemMatcher(
    neo4j_client=neo4j,
    openai_client=openai,
)
```

#### Methods

##### `find_matches()`

Find potential matches for extracted items.

```python
async def find_matches(
    self,
    items: list[ExtractedActionItem],
    embeddings: list[list[float]],
    tenant_id: UUID,
    account_id: str,
) -> MatchResult:
```

**Parameters:**
- `items`: Extracted action items to match
- `embeddings`: Embeddings for each item
- `tenant_id`: Tenant for isolation
- `account_id`: Account to search within

**Returns:** `MatchResult` with match decisions for each item

---

### ActionItemMerger

Executes match decisions by creating, updating, or linking items.

```python
from action_item_graph.pipeline import ActionItemMerger

merger = ActionItemMerger(
    neo4j_client=neo4j,
    openai_client=openai,
)
```

#### Methods

##### `execute_decisions()`

Execute merge decisions from matching.

```python
async def execute_decisions(
    self,
    match_result: MatchResult,
    interaction_id: str,
    tenant_id: UUID,
    account_id: str,
) -> list[MergeResult]:
```

**Returns:** List of `MergeResult` objects describing what was done

---

### TopicResolver

Matches extracted topics to existing topics using threshold-based logic.

```python
from action_item_graph.pipeline import TopicResolver

resolver = TopicResolver(
    neo4j_client=neo4j,
    openai_client=openai,
    similarity_auto_link=0.85,    # Auto-link if >= this
    similarity_auto_create=0.70,  # Auto-create if < this
)
```

#### Thresholds

| Threshold | Default | Behavior |
|-----------|---------|----------|
| `similarity_auto_link` | 0.85 | Automatically link to existing topic |
| `similarity_auto_create` | 0.70 | Automatically create new topic |
| Between | - | Use LLM to confirm match |

#### Methods

##### `resolve_topic()`

Resolve an extracted topic to an existing or new topic.

```python
async def resolve_topic(
    self,
    extracted_topic: ExtractedTopic,
    action_item: ExtractedActionItem,
    tenant_id: UUID,
    account_id: str,
) -> TopicResolutionResult:
```

**Parameters:**
- `extracted_topic`: The topic extracted from the action item
- `action_item`: The associated action item (for context)
- `tenant_id`: Tenant for isolation
- `account_id`: Account for scoping

**Returns:** `TopicResolutionResult` with decision and match details

---

### TopicExecutor

Executes topic resolution decisions by creating topics and linking action items.

```python
from action_item_graph.pipeline import TopicExecutor

executor = TopicExecutor(
    neo4j_client=neo4j,
    openai_client=openai,
    repository=repository,
)
```

#### Methods

##### `execute_resolution()`

Execute a topic resolution decision.

```python
async def execute_resolution(
    self,
    resolution: TopicResolutionResult,
    action_item_id: str,
    tenant_id: UUID,
    account_id: str,
) -> str:
```

**Parameters:**
- `resolution`: The resolution decision from TopicResolver
- `action_item_id`: ID of the action item to link
- `tenant_id`: Tenant for isolation
- `account_id`: Account for scoping

**Returns:** Topic ID (created or existing)

---

## Repository

### ActionItemRepository

Low-level graph CRUD operations.

```python
from action_item_graph import ActionItemRepository

repository = ActionItemRepository(neo4j_client=neo4j)
```

#### Methods

##### `ensure_account()`

Create or retrieve an account node.

```python
async def ensure_account(
    self,
    tenant_id: UUID,
    account_id: str,
    account_name: str | None = None,
) -> str:
```

##### `create_interaction()`

Create an interaction node.

```python
async def create_interaction(
    self,
    tenant_id: UUID,
    account_id: str,
    title: str | None = None,
    user_id: str = "system",
) -> str:
```

**Returns:** ID of the created interaction

##### `create_action_item()`

Create a new action item node with relationships.

```python
async def create_action_item(
    self,
    action_item: ActionItem,
    interaction_id: str,
) -> str:
```

##### `update_action_item_status()`

Update the status of an existing action item.

```python
async def update_action_item_status(
    self,
    action_item_id: str,
    new_status: str,
    interaction_id: str,
) -> None:
```

##### `vector_search()`

Search for similar action items by embedding.

```python
async def vector_search(
    self,
    embedding: list[float],
    tenant_id: UUID,
    account_id: str,
    limit: int = 10,
    threshold: float = 0.65,
    index_name: str = "embedding_current",
) -> list[dict]:
```

##### `create_topic()`

Create a new topic node.

```python
async def create_topic(
    self,
    topic: Topic,
    account_id: str,
) -> str:
```

**Returns:** ID of the created topic

##### `get_topic()`

Retrieve a topic by ID.

```python
async def get_topic(
    self,
    topic_id: str,
    tenant_id: UUID,
) -> dict | None:
```

##### `update_topic()`

Update a topic's properties.

```python
async def update_topic(
    self,
    topic_id: str,
    tenant_id: UUID,
    updates: dict,
) -> None:
```

##### `link_action_item_to_topic()`

Create a BELONGS_TO relationship between action item and topic.

```python
async def link_action_item_to_topic(
    self,
    action_item_id: str,
    topic_id: str,
    tenant_id: UUID,
    confidence: float,
    method: str,  # "extracted", "resolved", "manual"
) -> None:
```

##### `get_topic_with_action_items()`

Get a topic with all linked action items.

```python
async def get_topic_with_action_items(
    self,
    topic_id: str,
    tenant_id: UUID,
) -> dict | None:
```

**Returns:** Topic dict with `action_items` list

##### `get_topics_for_account()`

Get all topics for an account.

```python
async def get_topics_for_account(
    self,
    tenant_id: UUID,
    account_id: str,
) -> list[dict]:
```

---

## Models

### EnvelopeV1

Input payload format for the pipeline.

```python
from action_item_graph.models import EnvelopeV1

envelope = EnvelopeV1(
    tenant_id=UUID("..."),
    user_id="user_123",
    interaction_type="transcript",  # "transcript", "note", "document"
    content={"text": "...", "format": "plain"},
    timestamp=datetime.now(timezone.utc),
    source="api",  # "web-mic", "upload", "api", "import"
    account_id="acct_xyz",
    extras={
        "meeting_title": "Discovery Call",
        "duration_seconds": 1800,
    },
)
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `tenant_id` | UUID | Yes | Tenant identifier for isolation |
| `user_id` | str | Yes | User who created the interaction |
| `interaction_type` | Literal | Yes | "transcript", "note", or "document" |
| `content` | dict | Yes | `{"text": str, "format": str}` |
| `timestamp` | datetime | Yes | When the interaction occurred |
| `source` | Literal | Yes | "web-mic", "upload", "api", "import" |
| `account_id` | str | No* | Account identifier (*recommended) |
| `interaction_id` | UUID | No | Custom ID (auto-generated if not provided) |
| `trace_id` | str | No | For distributed tracing |
| `extras` | dict | No | Additional metadata |

---

### ActionItem

Pydantic model for action item data.

```python
from action_item_graph.models import ActionItem

item = ActionItem(
    id=uuid4(),
    tenant_id=UUID("..."),
    account_id="acct_xyz",
    action_item_text="Send proposal to client",
    summary="Send proposal by Friday",
    owner="John Smith",
    status="open",
    embedding=[0.1, 0.2, ...],
    embedding_current=[0.1, 0.2, ...],
)
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `tenant_id` | UUID | Tenant for isolation |
| `account_id` | str | Account association |
| `action_item_text` | str | Original text from transcript |
| `summary` | str | LLM-generated summary |
| `owner` | str | Person responsible |
| `status` | str | open, in_progress, completed, cancelled, deferred |
| `embedding` | list[float] | Original embedding (immutable) |
| `embedding_current` | list[float] | Current state embedding (mutable) |
| `due_date` | datetime | Optional due date |
| `conversation_context` | str | Surrounding context |
| `confidence` | float | Extraction confidence (0-1) |

---

### ExtractedActionItem

Model for items extracted from text (before matching).

```python
class ExtractedActionItem(BaseModel):
    action_item_text: str
    summary: str
    owner: str
    conversation_context: str
    due_date_text: str | None
    implied_status: Literal["open", "in_progress", "completed"] | None
    confidence: float
    topic: ExtractedTopic  # Required - extracted with action item
```

---

### Topic

Pydantic model for topic data.

```python
from action_item_graph.models import Topic

topic = Topic(
    id=uuid4(),
    tenant_id=UUID("..."),
    account_id="acct_xyz",
    name="Q3 Sales Initiative",
    canonical_name="q3 sales initiative",
    summary="Focused on expanding enterprise accounts...",
    embedding=[0.1, 0.2, ...],
    embedding_current=[0.1, 0.2, ...],
    action_item_count=5,
)
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID | Unique identifier |
| `tenant_id` | UUID | Tenant for isolation |
| `account_id` | str | Account association |
| `name` | str | Display name (3-5 words) |
| `canonical_name` | str | Normalized name (lowercase, trimmed) |
| `summary` | str | LLM-generated evolving summary |
| `embedding` | list[float] | Original embedding (immutable) |
| `embedding_current` | list[float] | Current state embedding (mutable) |
| `action_item_count` | int | Number of linked action items |
| `created_at` | datetime | Creation timestamp |
| `updated_at` | datetime | Last update timestamp |

---

### ExtractedTopic

Model for topics extracted alongside action items.

```python
from action_item_graph.models import ExtractedTopic

class ExtractedTopic(BaseModel):
    name: str       # 3-5 word topic name
    context: str    # Why the action item belongs to this topic
```

**Example:**
```python
topic = ExtractedTopic(
    name="Q3 Sales Initiative",
    context="This action item relates to the quarterly sales push targeting enterprise accounts"
)
```

---

## Clients

### Neo4jClient

Async Neo4j database client with connection pooling and retries.

```python
from action_item_graph.clients import Neo4jClient

neo4j = Neo4jClient()
await neo4j.connect()
await neo4j.setup_schema()  # Creates indexes and constraints

# Execute queries
result = await neo4j.execute_query(
    "MATCH (n:ActionItem) WHERE n.tenant_id = $tenant_id RETURN n",
    {"tenant_id": "..."}
)

await neo4j.close()
```

#### Methods

- `connect()`: Establish connection
- `close()`: Close connection
- `setup_schema()`: Create indexes and constraints
- `execute_query(query, parameters)`: Execute read query
- `execute_write(query, parameters)`: Execute write query

---

### OpenAIClient

Async OpenAI API client with retries and structured output.

```python
from action_item_graph.clients import OpenAIClient

openai = OpenAIClient()

# Chat completion
response = await openai.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
)

# Structured output
from pydantic import BaseModel

class Output(BaseModel):
    items: list[str]

result = await openai.structured_output(
    messages=[...],
    response_model=Output,
)

# Embeddings
embedding = await openai.get_embedding("text to embed")

await openai.close()
```

---

## Logging

### configure_logging()

Configure structured logging for the application.

```python
from action_item_graph import configure_logging

configure_logging(
    level="INFO",        # DEBUG, INFO, WARNING, ERROR
    json_output=True,    # JSON format for production
)
```

### logging_context()

Context manager for adding context to log messages.

```python
from action_item_graph import logging_context

with logging_context(
    trace_id="req-123",
    tenant_id="tenant-abc",
    account_id="acct-xyz",
):
    # All logs within this context include these fields
    result = await pipeline.process_text(...)
```

### PipelineTimer

Timer for measuring pipeline stage durations.

```python
from action_item_graph import PipelineTimer

timer = PipelineTimer()

with timer.stage("extraction"):
    # ... extraction logic

with timer.stage("matching"):
    # ... matching logic

# Manual recording
timer.record("custom_stage", 150.5)

# Get summary
print(timer.summary())
# {'total_ms': 1234.5, 'stages': {'extraction': 500.0, 'matching': 734.5}}
```

---

## Errors

### Exception Hierarchy

```
ActionItemGraphError (base)
├── PipelineError
│   ├── ValidationError
│   ├── ExtractionError
│   ├── MatchingError
│   └── MergeError
├── OpenAIError
│   └── OpenAIRateLimitError
└── Neo4jError
    ├── Neo4jConnectionError
    └── Neo4jConstraintError
```

### Error Context

All errors include contextual information:

```python
try:
    result = await pipeline.process_text(...)
except ValidationError as e:
    print(f"Message: {e.message}")
    print(f"Context: {e.context}")
    # e.context might include: {'field': 'text', 'reason': 'empty'}
```

### PartialSuccessResult

For operations that may partially succeed:

```python
from action_item_graph import PartialSuccessResult

result = PartialSuccessResult()
result.add_success(item_id="id1", data={"status": "created"})
result.add_failure(error=ExtractionError("Failed"), item_id="id2")

print(result.success_count)  # 1
print(result.failure_count)  # 1
print(result.partial_success)  # True
print(result.to_dict())
```
