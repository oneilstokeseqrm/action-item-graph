# Action Item Pipeline - Comprehensive Architecture Guide

This document provides a detailed explanation of the Action Item Pipeline architecture, designed to help engineers (or LLMs) understand and replicate the approach for similar applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Event Ingestion](#event-ingestion)
3. [Core Concepts](#core-concepts)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Dual Embedding Strategy](#dual-embedding-strategy)
6. [Multi-Tenancy & Scoping](#multi-tenancy--scoping)
7. [Component Deep Dives](#component-deep-dives)
8. [Graph Schema Design](#graph-schema-design)
9. [LLM Integration Patterns](#llm-integration-patterns)
10. [Replication Guide](#replication-guide)
11. [Performance Considerations](#performance-considerations)

---

## Overview

The Action Item Pipeline is a temporal knowledge graph system that:

1. **Extracts** structured data (action items) from unstructured text using F-CoT prompting and a commitment framework
2. **Validates** extractions through within-batch consolidation and LLM-as-Judge verification
3. **Resolves** owners via account-scoped alias caches with fuzzy matching
4. **Deduplicates** by matching against existing items using graduated vector similarity thresholds + LLM judgment
5. **Groups** related items into high-level topics for cross-conversation tracking
6. **Scores** items with weighted priority scoring (impact, urgency, specificity, confidence)
7. **Persists** to a graph database with full version history

### Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Extract quality** | F-CoT prompting with commitment framework, capped at 3-8 items |
| **Validate quality** | Within-batch consolidation + LLM-as-Judge verification |
| **Resolve owners** | Account-scoped alias cache with fuzzy matching and word-boundary awareness |
| **Prevent duplicates** | Graduated thresholds (auto-match/LLM/auto-create) + dual embedding search |
| **Track evolution** | Version nodes + bi-temporal fields |
| **Enable clustering** | Topic grouping with threshold-based matching |
| **Score and prioritize** | Weighted priority scoring (impact 40%, urgency 35%, specificity 15%, confidence 10%) |
| **Ensure isolation** | tenant_id + account_id scoping on all queries |
| **Maintain provenance** | EXTRACTED_FROM relationships to source interactions |

---

## Event Ingestion

Before the pipeline processes an envelope, it must arrive at the system. There are two entry paths:

### Production Path: EventBridge → SQS → Lambda → Railway

```
Upstream publisher
    → EventBridge (default bus)
        → action-item-graph-queue (SQS)
            → action-item-graph-ingest (Lambda)
                → POST /process (Railway FastAPI)
                    → EnvelopeDispatcher
                        → ActionItemPipeline + DealPipeline
```

1. Upstream services publish events to EventBridge with source `com.yourapp.transcription` (transcripts, notes, meetings) or `com.eq.email-pipeline` (emails)
2. The `action-item-graph-rule` EventBridge rule routes matching events to `action-item-graph-queue` (SQS)
3. The `action-item-graph-ingest` Lambda function is triggered by SQS. It extracts the `detail` field from the EventBridge wrapper and POSTs the raw EnvelopeV1 JSON to the Railway service
4. The Railway FastAPI service at `POST /process` validates the envelope, builds pipeline instances from persistent clients (Neo4j + OpenAI), and dispatches through `EnvelopeDispatcher`

### Direct Path: Library Usage

```python
from dispatcher import EnvelopeDispatcher

dispatcher = EnvelopeDispatcher.from_env()
result = await dispatcher.dispatch(envelope)
```

Used for local development, testing, and the live E2E smoke test (`scripts/run_live_e2e.py`).

### Supported Event Types

| EventBridge detail-type | interaction_type | content_format | Typical source |
|---|---|---|---|
| `EnvelopeV1.transcript` | `transcript` | `diarized` or `plain` | live-transcription-fastapi |
| `EnvelopeV1.note` | `note` | `plain` or `markdown` | live-transcription-fastapi |
| `EnvelopeV1.meeting` | `meeting` | `diarized` or `plain` | live-transcription-fastapi |
| `EnvelopeV1.email` | `email` | `email` | eq-email-pipeline |

Email events carry additional metadata in the `extras` dict: `subject`, `from_email`, `direction`, `thread_key`, `has_attachments`.

### Error Handling

- **Lambda retry**: On 5xx from Railway, the Lambda retries with exponential backoff + jitter (max 2 retries)
- **SQS retry**: If the Lambda fails (all retries exhausted), SQS makes the message visible again. After 3 total receive attempts, the message moves to the DLQ (`action-item-graph-dlq`)
- **4xx errors**: Not retried (indicates invalid payload — would fail again)
- **Pipeline fault isolation**: Even if one pipeline (Action Item or Deal) throws an exception, the other still completes via `asyncio.gather(return_exceptions=True)`

---

## Core Concepts

### Dual-Mode Extraction

The pipeline extracts two types of information from each transcript:

1. **New Action Items**: Fresh commitments ("I'll send the proposal by Friday")
2. **Status Updates**: References to existing items ("I sent the proposal yesterday")

Both are processed through the same pipeline, but status updates trigger merge/update logic rather than creation.

### Threshold-Based Matching

Rather than binary match/no-match, the system uses graduated thresholds:

**Action Items:**
```
┌─────────────────────────────────────────────────────────┐
│                    Similarity Score                      │
├────────────┬────────────────────────────┬───────────────┤
│  < 0.68    │      0.68 - 0.88           │    >= 0.88    │
│  No Match  │  LLM Decides (borderline)  │  Auto-Match   │
│ Create New │  May match or create new   │  Link/Merge   │
└────────────┴────────────────────────────┴───────────────┘
```

**Topics** use higher thresholds (0.70 / 0.85) because incorrect grouping has more impact.

### Evolving Summaries

Both action items and topics have LLM-generated summaries that evolve:

- **Initial**: Summary generated during extraction
- **On Update**: Summary regenerated to reflect new information
- **Version Captured**: Previous summary preserved in version node

---

## Pipeline Architecture

### Stage Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INPUT                                                         │
│    EnvelopeV1 or raw text + tenant_id + account_id               │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. EXTRACTION (ActionItemExtractor)                              │
│    - F-CoT two-stage prompt: commitment signals → evaluation     │
│    - Five-Field Commitment Framework (Decision, Action, Owner,   │
│      Timeline, Definition of Done)                               │
│    - Structured output with scoring dimensions (1-5)             │
│    - Capped at 3-5 items (max 8 if truly warranted)             │
│    - Generates embeddings for each item                          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CONSOLIDATION (ActionItemConsolidator)                        │
│    - Within-batch dedup using embedding cosine similarity         │
│    - Complete-linkage clustering at 0.80 threshold                     │
│    - LLM selects best representative per cluster, merges context │
│    - Fail-open: LLM errors keep first item in cluster            │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. VERIFICATION (ActionItemVerifier)                             │
│    - LLM-as-Judge adversarial quality check                      │
│    - Evaluates: actionable? real owner? specific? commitment?    │
│    - Assigns adjusted_confidence per item                        │
│    - Drops items below confidence floor (0.4)                    │
│    - Fail-open: LLM errors pass all items through                │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. OWNER PRE-RESOLUTION (OwnerPreResolver)                       │
│    - Load account-scoped owner cache from existing Owner nodes   │
│    - Resolution cascade: exact → alias → substring → fuzzy       │
│    - Word-boundary matching (prevents "Peter" → "Peterson")      │
│    - LLM role-to-name for role_inferred owners                   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. MATCHING (ActionItemMatcher)                                  │
│    - Vector search against BOTH embedding indexes                │
│    - Graduated thresholds:                                       │
│      >= 0.88: auto-match (skip LLM)                              │
│      0.68-0.88: LLM decides (borderline)                         │
│      < 0.68: auto-create (clearly different)                     │
│    - Returns: matched items + unmatched items                    │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. MERGING (ActionItemMerger)                                    │
│    - For unmatched: Create new ActionItem node                   │
│    - For matched:                                                │
│      - If same item: Merge (update text, evolve summary)         │
│      - If status update: Update status field                     │
│      - Create version snapshot                                   │
│    - Link to Interaction via EXTRACTED_FROM                      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. TOPIC RESOLUTION (TopicResolver + TopicExecutor)              │
│    - For each action item with extracted topic:                  │
│      - Vector search against topic indexes                       │
│      - Apply thresholds: auto-link / LLM-confirm / auto-create   │
│    - Create Topic node if new                                    │
│    - Create BELONGS_TO relationship                              │
│    - Evolve topic summary via LLM                                │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. SCORING                                                       │
│    - Compute priority_score from extraction dimensions:           │
│      0.40×(impact/5) + 0.35×(urgency/5) + 0.15×(specificity/5)  │
│      + 0.10×confidence                                           │
│    - Persist scoring fields as first-class properties             │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. OUTPUT                                                       │
│    PipelineResult with:                                          │
│    - created_ids, updated_ids, linked_ids                        │
│    - topics_created, topics_linked                               │
│    - pre/post_consolidation_count, items_consolidated            │
│    - pre/post_verification_count, items_rejected                 │
│    - stage_timings, processing_time_ms                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

```python
# Simplified orchestration in pipeline.py
async def process_text(self, text, tenant_id, account_id, ...):
    # Stage 1: Ensure account exists
    await self.repository.ensure_account(tenant_id, account_id)

    # Stage 2: Create interaction + extract action items
    extraction = await self.extractor.extract_from_text(text, ...)

    # Stage 3: Consolidate within-batch duplicates
    extraction, items_consolidated = await self.consolidator.consolidate(extraction)

    # Stage 4: Verify quality via LLM-as-Judge
    extraction, items_rejected, reasons = await self.verifier.verify_batch(extraction, text)

    # Stage 5: Pre-resolve owners against account cache
    await self.owner_resolver.resolve_batch(extraction, tenant_id, account_id)

    # Stage 6: Match against existing items (graduated thresholds)
    match_result = await self.matcher.find_matches(extraction, tenant_id, account_id)

    # Stage 7: Execute merge decisions
    merge_results = await self.merger.execute_decisions(match_result, ...)

    # Stage 8: Resolve topics (if enabled)
    if self.enable_topics:
        topic_results = await self._resolve_topics(extraction, merge_results, ...)

    # Stage 9: Scoring already computed during extraction + persisted

    return PipelineResult(...)
```

---

## Dual Embedding Strategy

### The Problem: Embedding Drift

When an action item evolves over time, its text changes. If we only store one embedding:

```
Original: "Send proposal to client"  → embedding_v1
Updated:  "Sent proposal, awaiting feedback" → embedding_v2
```

A new extraction saying "Send the proposal document" would match `embedding_v1` but not `embedding_v2`. The original meaning is lost.

### The Solution: Two Embeddings

| Property | Mutability | Purpose |
|----------|------------|---------|
| `embedding` | Immutable | Captures original semantic meaning |
| `embedding_current` | Mutable | Captures current state |

### Search Strategy

```python
async def search_both_embeddings(self, query_embedding, tenant_id, account_id):
    # Search original embeddings - catches similar NEW items
    original_results = await self.vector_search(
        embedding=query_embedding,
        index_name='action_item_embedding_idx',
        tenant_id=tenant_id,
        account_id=account_id,
    )

    # Search current embeddings - catches STATUS UPDATES
    current_results = await self.vector_search(
        embedding=query_embedding,
        index_name='action_item_embedding_current_idx',
        tenant_id=tenant_id,
        account_id=account_id,
    )

    # Deduplicate by ID, keep highest score
    return merge_and_deduplicate(original_results, current_results)
```

### When to Update embedding_current

```python
# In merger - only update on significant text changes
if merged.should_update_embedding:
    new_embedding = await self.openai.get_embedding(merged.action_item_text)
    await self.repository.update_action_item(
        action_item_id,
        {'embedding_current': new_embedding}
    )
```

The LLM decides if changes are significant enough during merge synthesis.

---

## Multi-Tenancy & Scoping

### Scoping Hierarchy

```
┌─────────────────────────────────────────┐
│              TENANT                      │
│  (Complete data isolation)              │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │          ACCOUNT                    │ │
│  │  (Logical grouping within tenant)   │ │
│  │                                      │ │
│  │  ┌──────────────────────────────┐  │ │
│  │  │       ACTION ITEMS            │  │ │
│  │  │       TOPICS                  │  │ │
│  │  │       INTERACTIONS            │  │ │
│  │  └──────────────────────────────┘  │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │          OWNERS                     │ │
│  │  (Shared across accounts)          │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Required Scoping by Operation

| Operation | tenant_id | account_id |
|-----------|-----------|------------|
| Vector search | Required | Required |
| Create ActionItem | Required | Required |
| Create Topic | Required | Required |
| Create Owner | Required | - |
| Query action items | Required | Required |
| Resolve owner | Required | - |

### Preventing Cross-Account Bleeding

Vector search methods enforce both scopes:

```python
async def search_both_embeddings(
    self,
    embedding: list[float],
    tenant_id: str,      # REQUIRED - not optional
    account_id: str,     # REQUIRED - not optional
    limit: int = 10,
):
    """Both parameters required to prevent cross-account bleeding."""
```

---

## Component Deep Dives

### ActionItemExtractor

**Purpose**: Convert unstructured transcript → structured action items + topics + scoring

**Key Design Decisions**:

1. **F-CoT (Focused Chain-of-Thought)** two-stage extraction: identify commitment signals first, then evaluate actionability
2. **Five-Field Commitment Framework**: Decision, Action, Owner, Timeline, Definition of Done — items must pass criteria 1-3 or be rejected
3. **Structured output** via Pydantic models (ensures valid JSON)
4. **Dual-mode extraction** identifies both new items and status updates
5. **Capped extraction**: 3-5 items typical, max 8 if truly warranted
6. **Negative examples** in prompt prevent common false positives

**Prompt Structure**:

The extraction prompt uses two stages:
- **Stage 1**: Scan for commitment language signals (first-person future tense, explicit promises, acceptance of assignment)
- **Stage 2**: Evaluate each signal against the five-field framework; items failing criteria 1-3 are NOT extracted

The prompt includes an explicit `<not_commitments>` section with real negative examples: observations ("Salesforce numbers should be fixed"), informational statements ("Aditya is on Slack"), conditionals ("If you're looking forward to..."), expectations of third parties.

**Output Model**:

```python
class ExtractedActionItem(BaseModel):
    action_item_text: str
    summary: str
    owner: str
    conversation_context: str
    is_status_update: bool
    implied_status: Literal["open", "in_progress", "completed"] | None
    topic: ExtractedTopic
    # Five-Field Commitment Framework
    commitment_strength: Literal['explicit', 'conditional', 'weak', 'observation']
    decision_context: str | None
    definition_of_done: str | None
    # Scoring dimensions (1-5)
    score_impact: int  # Business impact
    score_urgency: int  # Time sensitivity
    score_specificity: int  # How actionable
    score_effort: int  # Effort required

class ExtractedTopic(BaseModel):
    name: str  # 3-5 words
    context: str  # Why this action item belongs
```

### ActionItemConsolidator

**Purpose**: Remove within-batch duplicates using embedding similarity

**Algorithm**:
1. Compute pairwise cosine similarity from embeddings already generated during extraction
2. Cluster items above `INTRA_BATCH_SIMILARITY = 0.80` using complete-linkage
3. For each 2+ item cluster, LLM selects the best representative and merges context
4. **Fail-open**: On LLM failure, keeps first item in cluster

### ActionItemVerifier

**Purpose**: LLM-as-Judge adversarial quality check

- Deliberately adversarial persona challenges each extracted item
- Evaluates: Is it truly actionable? Is the owner real? Is it specific enough? Is it a real commitment?
- Assigns `adjusted_confidence` (0.0-1.0) to each item
- `CONFIDENCE_FLOOR = 0.4` — items below are dropped
- Single batch call (all items in one prompt) for efficiency
- **Fail-open**: LLM errors pass all items through

### OwnerPreResolver

**Purpose**: Account-scoped canonical owner mapping with fuzzy matching

Resolution cascade:
1. Exact match against known Owner nodes
2. Case-insensitive alias match
3. Substring match (word-boundary aware — prevents "Peter" matching "Peterson")
4. Fuzzy variant match (apostrophe normalization, 80%+ SequenceMatcher ratio)
5. LLM role-to-name resolution for `role_inferred` owners

### ActionItemMatcher

**Purpose**: Find existing items that might match new extractions, using graduated thresholds to minimize LLM calls

**Algorithm**:

```python
async def find_matches(self, extracted_items, tenant_id, account_id):
    results = []

    for item, embedding in extracted_items:
        # Step 1: Vector search (dual index)
        candidates = await self._find_candidates(
            embedding, tenant_id, account_id
        )

        # Step 2: No candidates above MIN_SIMILARITY_SCORE (0.68)? → unmatched
        if not candidates:
            results.append(MatchResult(matched=False))
            continue

        # Step 3: Apply graduated thresholds
        decisions = []
        for candidate in candidates:
            if candidate.similarity_score >= LLM_ZONE_UPPER:  # 0.88
                # Auto-match: skip LLM call entirely
                decision = auto_match_decision(candidate)
            else:
                # LLM zone (0.68-0.88): ask LLM to decide
                decision = await self._deduplicate(
                    existing=candidate.node_properties,
                    new_extraction=item,
                    similarity_score=candidate.similarity_score,
                )
            decisions.append((candidate, decision))

        # Step 4: Select best match (if any)
        best = self._select_best_match(decisions)
        results.append(MatchResult(matched=best is not None, ...))

    return results
```

**Graduated Thresholds**:

| Threshold | Value | Behavior |
|-----------|-------|----------|
| `MIN_SIMILARITY_SCORE` | 0.68 | Below this: auto-create new item |
| `LLM_ZONE_UPPER` | 0.88 | Above this: auto-match, skip LLM |
| Between | 0.68-0.88 | LLM decides (existing behavior) |

This saves ~8 LLM calls per interaction (from ~12 to ~3-4).

**LLM Deduplication Prompt**:

```python
DEDUPLICATION_PROMPT = """
Given an EXISTING action item and a NEW extraction:

Existing: {existing_text} (Owner: {owner}, Status: {status})
New: {new_text} (Owner: {owner})
Similarity: {similarity_score}

Decide:
1. is_same_item: Are these the same real-world task?
2. is_status_update: Is the new one a status update?
3. merge_recommendation: "merge", "update_status", "create_new", "link_related"
"""
```

### TopicResolver

**Purpose**: Match extracted topic → existing topic or create new

**Threshold Logic**:

```python
class TopicResolver:
    SIMILARITY_AUTO_LINK = 0.85    # >= this: auto-link
    SIMILARITY_AUTO_CREATE = 0.70  # < this: auto-create
    # Between: LLM confirmation required

    async def resolve_topic(self, extracted_topic, tenant_id, account_id):
        # Generate topic embedding
        topic_text = f"{extracted_topic.name}: {extracted_topic.context}"
        embedding = await self.openai.get_embedding(topic_text)

        # Search existing topics (dual embedding)
        candidates = await self._search_topics(embedding, tenant_id, account_id)

        if not candidates:
            return TopicDecision.CREATE_NEW

        best = candidates[0]

        if best.similarity >= self.SIMILARITY_AUTO_LINK:
            return TopicDecision.LINK_EXISTING
        elif best.similarity < self.SIMILARITY_AUTO_CREATE:
            return TopicDecision.CREATE_NEW
        else:
            # Borderline: Ask LLM
            return await self._llm_confirm(best, extracted_topic)
```

### TopicExecutor

**Purpose**: Execute topic resolution decisions

**Key Operations**:

1. **Create Topic**: New node with embedding, initial summary
2. **Link Existing**: Create BELONGS_TO, evolve summary
3. **Version Tracking**: Create TopicVersion on significant changes

**Summary Evolution**:

```python
async def _evolve_topic_summary(self, topic, new_action_item):
    prompt = f"""
    Current topic summary: {topic.summary}
    New action item linked: {new_action_item.summary}

    Generate an updated summary that incorporates this new item.
    Keep it concise (1-2 sentences).
    """

    evolved_summary = await self.openai.chat_completion(prompt)

    # Update topic
    await self.repository.update_topic(topic.id, {
        'summary': evolved_summary,
        'action_item_count': topic.action_item_count + 1,
    })

    # Create version snapshot
    await self.repository.create_topic_version(topic.id, ...)
```

---

## Graph Schema Design

### Node Design Principles

1. **ID Strategy**: UUIDs for all nodes (conflict-free generation)
2. **Timestamps**: `created_at`, `updated_at` on mutable nodes
3. **Provenance**: Track source (e.g., `created_from_action_item_id`)
4. **Denormalization**: Store counts (e.g., `action_item_count`) for query efficiency

### Relationship Properties

Relationships carry metadata:

```cypher
(:ActionItem)-[:BELONGS_TO {
    confidence: 0.92,           # How confident the match
    method: "resolved",         # "extracted", "resolved", "manual"
    timestamp: datetime()       # When linked
}]->(:ActionItemTopic)

(:ActionItem)-[:EXTRACTED_FROM {
    is_source: true,            # Original extraction source
    confidence: 0.85            # Extraction confidence
}]->(:Interaction)
```

### Index Strategy

```cypher
-- Uniqueness constraints (primary keys)
CREATE CONSTRAINT action_item_id_unique FOR (n:ActionItem) REQUIRE (n.tenant_id, n.action_item_id) IS UNIQUE
CREATE CONSTRAINT action_item_topic_id_unique FOR (n:ActionItemTopic) REQUIRE (n.tenant_id, n.action_item_topic_id) IS UNIQUE

-- Query performance indexes
CREATE INDEX action_item_tenant_idx FOR (n:ActionItem) ON (n.tenant_id)
CREATE INDEX action_item_account_idx FOR (n:ActionItem) ON (n.account_id)
CREATE INDEX action_item_topic_tenant_idx FOR (n:ActionItemTopic) ON (n.tenant_id)
CREATE INDEX action_item_topic_account_idx FOR (n:ActionItemTopic) ON (n.account_id)

-- Vector indexes (1536 dimensions, cosine similarity)
CREATE VECTOR INDEX action_item_embedding_idx FOR (n:ActionItem) ON (n.embedding)
CREATE VECTOR INDEX action_item_embedding_current_idx FOR (n:ActionItem) ON (n.embedding_current)
CREATE VECTOR INDEX action_item_topic_embedding_idx FOR (n:ActionItemTopic) ON (n.embedding)
CREATE VECTOR INDEX action_item_topic_embedding_current_idx FOR (n:ActionItemTopic) ON (n.embedding_current)
```

---

## LLM Integration Patterns

### Structured Output

Use Pydantic models for reliable parsing:

```python
from pydantic import BaseModel

class DeduplicationDecision(BaseModel):
    is_same_item: bool
    is_status_update: bool
    merge_recommendation: Literal["merge", "update_status", "create_new"]
    reasoning: str
    confidence: float

# OpenAI call with structured output
decision = await openai.chat_completion_structured(
    messages=messages,
    response_model=DeduplicationDecision,
)
```

### Prompt Design Principles

1. **XML-style tags** for clear section boundaries
2. **Examples** of expected output format
3. **Explicit constraints** ("3-5 words", "1-2 sentences")
4. **Decision framework** (what to consider, in what order)

### Error Handling

```python
async def extract_with_retry(self, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await self.openai.structured_output(...)
            return result
        except ValidationError as e:
            if attempt == max_retries - 1:
                raise ExtractionError(f"Failed after {max_retries} attempts")
            # LLM sometimes returns invalid JSON - retry
            continue
```

---

## Replication Guide

### Adapting for Other Domains

This pattern can be adapted for:

- **Document extraction** (contracts → clauses)
- **Meeting notes** (discussions → decisions)
- **Support tickets** (conversations → issues)
- **Research papers** (text → citations + claims)

### Key Steps to Replicate

1. **Define your entity model**
   ```python
   class YourEntity(BaseModel):
       id: UUID
       tenant_id: UUID
       account_id: str
       # Your domain fields
       embedding: list[float]
       embedding_current: list[float]
   ```

2. **Create extraction prompts**
   ```python
   EXTRACTION_PROMPT = """
   Extract [your entities] from [your input type].
   For each entity, identify:
   - [field1]: description
   - [field2]: description
   - topic: High-level grouping (if applicable)
   """
   ```

3. **Implement the pipeline stages**
   - Extractor: LLM + embeddings
   - Matcher: Vector search + LLM deduplication
   - Merger: Create/update/link logic
   - Topic Resolution: (optional) Clustering

4. **Set up graph schema**
   - Node types with appropriate properties
   - Relationship types with edge properties
   - Vector indexes for embeddings
   - Regular indexes for scoping columns

5. **Configure thresholds**
   - Start with defaults (0.65 match, 0.70/0.85 topic)
   - Tune based on false positive/negative rates

### Configuration Template

```python
@dataclass
class PipelineConfig:
    # Action item matching thresholds (graduated)
    min_similarity: float = 0.68          # Below: auto-create new
    llm_zone_upper: float = 0.88          # Above: auto-match, skip LLM

    # Consolidation threshold
    intra_batch_similarity: float = 0.80  # Within-batch dedup clustering

    # Verification
    confidence_floor: float = 0.4         # Below: drop item
    role_resolution_confidence: float = 0.8  # LLM role-to-name min confidence

    # Topic thresholds (higher = stricter)
    topic_auto_link: float = 0.85
    topic_auto_create: float = 0.70

    # Priority scoring weights
    weight_impact: float = 0.40
    weight_urgency: float = 0.35
    weight_specificity: float = 0.15
    weight_confidence: float = 0.10

    # LLM settings
    extraction_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Feature flags
    enable_topics: bool = True
    enable_version_tracking: bool = True
```

---

## Performance Considerations

### Latency Breakdown

Typical processing for a 50KB transcript:

| Stage | Typical Time | Notes |
|-------|-------------|-------|
| Extraction | 15-30s | Single F-CoT LLM call, depends on transcript length |
| Embedding generation | 2-5s | Batch all items in one call |
| Consolidation | 0-3s | Only when clusters found (~30% of interactions) |
| Verification | 3-5s | Single batch LLM call |
| Owner pre-resolution | 0.5-1s | Cache lookup + optional LLM (~20% of interactions) |
| Vector search | 0.5-1s per item | Neo4j vector index |
| LLM deduplication | 2-3s per candidate | Fewer calls due to graduated thresholds |
| Merging | 0.5-2s per item | Graph writes |
| Topic resolution | 5-15s | Similar to matching |

### Optimization Strategies

1. **Graduated thresholds**: Auto-match/auto-create reduces LLM dedup calls by ~60%
2. **F-CoT extraction**: Fewer items extracted means fewer downstream operations
3. **Batch embeddings**: Generate all embeddings in one API call
4. **Batch verification**: Single LLM call for all items (not N individual calls)
5. **Parallel deduplication**: Run remaining LLM calls concurrently
6. **Owner cache**: In-memory cache avoids repeated Neo4j queries
7. **Fail-open quality gates**: LLM failures never block the pipeline

### Scaling Considerations

- **Vector index size**: Neo4j handles millions of vectors efficiently
- **Multi-tenant isolation**: Indexes are global; filtering is query-time
- **Concurrent processing**: Use async throughout; avoid blocking

---

## Postgres Dual-Write

Both pipelines optionally dual-write to Neon Postgres when `NEON_DATABASE_URL` is set. Neo4j remains the source of truth; Postgres is a read-optimized projection for the frontend.

### Action Item Dual-Write

The ActionItemPipeline writes action items, versions, topics, topic memberships, and owners to Postgres after Neo4j persistence. See [ARCHITECTURE.md](../ARCHITECTURE.md) for table mappings.

**Owner upsert pattern**: The `action_item_owners` table uses `ON CONFLICT (tenant_id, canonical_name)` as the upsert conflict target because `(tenant_id, canonical_name)` is the semantic business key. When Neo4j re-creates an owner node with a new UUID for the same canonical name (e.g., during re-processing), the Postgres upsert updates `owner_id` to stay in sync rather than violating the unique constraint.

### Deal Dual-Write

The DealPipeline optionally dual-writes to Postgres using the same failure-isolation pattern:

| PostgresClient Method | What It Does |
|-----------------------|-------------|
| `upsert_deal` | UPSERTs to `opportunities` table (AI extraction columns only) |
| `insert_deal_version` | Inserts bi-temporal snapshot to `deal_versions` |
| `link_deal_to_interaction` | UPSERTs to `opportunity_interaction_links` |
| `persist_deal_full` | Orchestrates all three operations for a single deal |

Failure isolation: Postgres failures never block Neo4j writes. Each operation is individually wrapped in try/except. A failed Postgres write logs a warning but does not affect the pipeline result.

The Deal pipeline does **not** write to the 8 trigger-protected columns on `opportunities` (`stage`, `amount`, `close_date`, `probability`, `forecast_category`, `pipeline_category`, `is_won`, `is_closed`) which are owned by the opportunity-forecasting pipeline.

---

## Summary

The Action Item Pipeline demonstrates a robust pattern for:

1. **Quality-first extraction** using F-CoT prompting and commitment frameworks to extract fewer, higher-quality items
2. **Multi-stage validation** with within-batch consolidation and adversarial LLM-as-Judge verification
3. **Owner resolution** via account-scoped alias caches with fuzzy matching and word-boundary awareness
4. **Intelligent deduplication** combining graduated vector similarity thresholds + LLM judgment
5. **Semantic clustering** via topic grouping with threshold-based matching
6. **Priority scoring** with weighted composite scores for surfacing the most important items
7. **Complete isolation** through comprehensive tenant + account scoping
8. **Evolution tracking** via dual embeddings and version history

The key insight is that **extraction quality compounds** — higher precision at extraction means fewer duplicates to catch downstream, fewer LLM calls for deduplication, and better signal-to-noise for end users. The quality pipeline reduces LLM calls by ~50% while dramatically improving precision.

This pattern is highly transferable to other domains requiring structured extraction with intelligent deduplication.
