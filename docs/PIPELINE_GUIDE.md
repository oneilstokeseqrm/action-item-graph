# Action Item Pipeline - Comprehensive Architecture Guide

This document provides a detailed explanation of the Action Item Pipeline architecture, designed to help engineers (or LLMs) understand and replicate the approach for similar applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Dual Embedding Strategy](#dual-embedding-strategy)
5. [Multi-Tenancy & Scoping](#multi-tenancy--scoping)
6. [Component Deep Dives](#component-deep-dives)
7. [Graph Schema Design](#graph-schema-design)
8. [LLM Integration Patterns](#llm-integration-patterns)
9. [Replication Guide](#replication-guide)
10. [Performance Considerations](#performance-considerations)

---

## Overview

The Action Item Pipeline is a temporal knowledge graph system that:

1. **Extracts** structured data (action items) from unstructured text (call transcripts)
2. **Deduplicates** by matching against existing items using vector similarity + LLM judgment
3. **Groups** related items into high-level topics for cross-conversation tracking
4. **Persists** to a graph database with full version history

### Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Prevent duplicates** | Dual embedding search + LLM deduplication |
| **Track evolution** | Version nodes + bi-temporal fields |
| **Enable clustering** | Topic grouping with threshold-based matching |
| **Ensure isolation** | tenant_id + account_id scoping on all queries |
| **Maintain provenance** | EXTRACTED_FROM relationships to source interactions |

---

## Core Concepts

### Dual-Mode Extraction

The pipeline extracts two types of information from each transcript:

1. **New Action Items**: Fresh commitments ("I'll send the proposal by Friday")
2. **Status Updates**: References to existing items ("I sent the proposal yesterday")

Both are processed through the same pipeline, but status updates trigger merge/update logic rather than creation.

### Threshold-Based Matching

Rather than binary match/no-match, the system uses graduated thresholds:

```
┌─────────────────────────────────────────────────────────┐
│                    Similarity Score                      │
├────────────┬────────────────────────────┬───────────────┤
│  < 0.65    │      0.65 - 0.85           │    >= 0.85    │
│  No Match  │  LLM Decides (borderline)  │  Auto-Match   │
│ Create New │  May match or create new   │  Link/Merge   │
└────────────┴────────────────────────────┴───────────────┘
```

For topics, thresholds are higher (0.70 / 0.85) because incorrect grouping has more impact.

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
│    - Single LLM call extracts action items + topics              │
│    - Structured output using Pydantic models                     │
│    - Identifies status updates vs new items                      │
│    - Generates embeddings for each item                          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. MATCHING (ActionItemMatcher)                                  │
│    - Vector search against BOTH embedding indexes                │
│    - Filter by tenant_id AND account_id                          │
│    - For each candidate above threshold: LLM deduplication       │
│    - Returns: matched items + unmatched items                    │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. MERGING (ActionItemMerger)                                    │
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
│ 5. TOPIC RESOLUTION (TopicResolver + TopicExecutor)              │
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
│ 6. OUTPUT                                                        │
│    PipelineResult with:                                          │
│    - created_ids, updated_ids, linked_ids                        │
│    - topics_created, topics_linked                               │
│    - stage_timings, processing_time_ms                           │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

```python
# Simplified orchestration in pipeline.py
async def process_text(self, text, tenant_id, account_id, ...):
    # Stage 1: Ensure account exists
    await self.repository.ensure_account(tenant_id, account_id)

    # Stage 2: Create interaction for this transcript
    interaction_id = await self.repository.create_interaction(...)

    # Stage 3: Extract action items + topics
    extraction = await self.extractor.extract_from_text(text, ...)

    # Stage 4: Match against existing items
    match_result = await self.matcher.find_matches(extraction, tenant_id, account_id)

    # Stage 5: Execute merge decisions
    merge_results = await self.merger.execute_decisions(match_result, ...)

    # Stage 6: Resolve topics (if enabled)
    if self.enable_topics:
        topic_results = await self._resolve_topics(extraction, merge_results, ...)

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

**Purpose**: Convert unstructured transcript → structured action items + topics

**Key Design Decisions**:

1. **Single LLM call** for action items + topics (reduces latency)
2. **Structured output** via Pydantic models (ensures valid JSON)
3. **Dual-mode extraction** identifies both new items and status updates

**Prompt Structure**:

```python
EXTRACTION_SYSTEM_PROMPT = """
You are extracting action items from a sales call transcript.

For each action item, extract:
- action_item_text: The verbatim commitment
- summary: A 1-sentence summary
- owner: Person responsible
- is_status_update: True if referencing existing commitment
- implied_status: If status update, what status?
- topic: High-level theme (3-5 words)
"""
```

**Output Model**:

```python
class ExtractedActionItem(BaseModel):
    action_item_text: str
    summary: str
    owner: str
    conversation_context: str
    is_status_update: bool
    implied_status: Literal["open", "in_progress", "completed"] | None
    topic: ExtractedTopic  # NEW: Extracted in same call

class ExtractedTopic(BaseModel):
    name: str  # 3-5 words
    context: str  # Why this action item belongs
```

### ActionItemMatcher

**Purpose**: Find existing items that might match new extractions

**Algorithm**:

```python
async def find_matches(self, extracted_items, tenant_id, account_id):
    results = []

    for item, embedding in extracted_items:
        # Step 1: Vector search (dual index)
        candidates = await self._find_candidates(
            embedding, tenant_id, account_id
        )

        # Step 2: No candidates? → unmatched
        if not candidates:
            results.append(MatchResult(matched=False))
            continue

        # Step 3: LLM deduplication for each candidate
        decisions = []
        for candidate in candidates:
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
CREATE CONSTRAINT action_item_id_unique FOR (n:ActionItem) REQUIRE (n.tenant_id, n.action_item_id) IS NODE KEY
CREATE CONSTRAINT action_item_topic_id_unique FOR (n:ActionItemTopic) REQUIRE (n.tenant_id, n.action_item_topic_id) IS NODE KEY

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
    # Matching thresholds
    min_similarity: float = 0.65
    high_confidence_threshold: float = 0.85

    # Topic thresholds (higher = stricter)
    topic_auto_link: float = 0.85
    topic_auto_create: float = 0.70

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
| Extraction | 15-30s | Single LLM call, depends on transcript length |
| Embedding generation | 2-5s | Batch all items in one call |
| Vector search | 0.5-1s per item | Neo4j vector index |
| LLM deduplication | 2-3s per candidate | Parallel where possible |
| Merging | 0.5-2s per item | Graph writes |
| Topic resolution | 5-15s | Similar to matching |

### Optimization Strategies

1. **Batch embeddings**: Generate all embeddings in one API call
2. **Parallel deduplication**: Run LLM calls concurrently
3. **Early filtering**: Skip obvious non-matches before LLM
4. **Caching**: Cache embeddings for repeated transcripts

### Scaling Considerations

- **Vector index size**: Neo4j handles millions of vectors efficiently
- **Multi-tenant isolation**: Indexes are global; filtering is query-time
- **Concurrent processing**: Use async throughout; avoid blocking

---

## Summary

The Action Item Pipeline demonstrates a robust pattern for:

1. **Structured extraction** from unstructured text using LLMs
2. **Intelligent deduplication** combining vector similarity + LLM judgment
3. **Semantic clustering** via topic grouping with graduated thresholds
4. **Complete isolation** through comprehensive tenant + account scoping
5. **Evolution tracking** via dual embeddings and version history

The key insight is that **vector similarity alone is insufficient** for deduplication - the LLM provides semantic understanding that pure distance metrics cannot. Similarly, **topic grouping benefits from human-like judgment** for borderline cases while using thresholds for obvious decisions.

This pattern is highly transferable to other domains requiring structured extraction with intelligent deduplication.
