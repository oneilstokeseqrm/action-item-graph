# Action Item Graph - Architecture

## Overview

A temporal knowledge graph pipeline for extracting and managing action items from call transcripts. Built with Neo4j for graph storage and OpenAI for extraction and embeddings.

## System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Input Layer                                  │
│  EnvelopeV1 payload (transcript, tenant_id, account_id, metadata)   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Extraction Pipeline                             │
│  1. Extract action items (OpenAI structured output)                  │
│  2. Extract status updates to existing items                         │
│  3. Generate embeddings for each extracted item                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Matching & Deduplication                          │
│  1. Vector similarity search (dual embeddings)                       │
│  2. LLM-based deduplication decision                                 │
│  3. Merge or create new action item                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Neo4j Graph Store                              │
│  Nodes: Account, Interaction, ActionItem, Owner, Contact, Deal      │
│  Vector indexes for similarity search                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Graph Schema

### Node Types

All nodes include `tenant_id` for multi-tenancy isolation.

```
(:Account)
  - id: string (primary key, e.g., "acct_acme_corp_001")
  - tenant_id: UUID
  - name: string
  - domain: string (optional)
  - created_at: datetime
  - last_interaction_at: datetime

(:Interaction)
  - id: UUID (primary key)
  - tenant_id: UUID
  - account_id: string
  - interaction_type: enum (transcript, note, document)
  - title: string (optional)
  - transcript_text: string
  - occurred_at: datetime
  - duration_seconds: int (optional)
  - source: string (web-mic, upload, api, import)
  - user_id: string
  - processed_at: datetime
  - action_item_count: int

(:ActionItem)
  - id: UUID (primary key)
  - tenant_id: UUID
  - account_id: string
  - action_item_text: string (verbatim from transcript)
  - summary: string (LLM-generated)
  - owner: string
  - conversation_context: string
  - due_date: datetime (optional)
  - status: enum (open, in_progress, completed, cancelled, deferred)
  - version: int
  - evolution_summary: string
  - created_at: datetime
  - last_updated_at: datetime
  - source_interaction_id: UUID
  - user_id: string
  - embedding: float[] (original, immutable)
  - embedding_current: float[] (updated on changes)
  - confidence: float
  - valid_at: datetime
  - invalid_at: datetime (optional)
  - invalidated_by: UUID (optional)

(:ActionItemVersion)
  - id: UUID (primary key)
  - action_item_id: UUID
  - tenant_id: UUID
  - version: int
  - action_item_text: string
  - summary: string
  - owner: string
  - status: enum
  - due_date: datetime (optional)
  - change_summary: string
  - change_source_interaction_id: UUID
  - created_at: datetime
  - valid_from: datetime
  - valid_until: datetime (optional)

(:Owner)
  - id: UUID (primary key)
  - tenant_id: UUID
  - canonical_name: string
  - aliases: string[]
  - contact_id: string (optional)
  - user_id: string (optional)
  - created_at: datetime

(:Topic)
  - id: UUID (primary key)
  - tenant_id: UUID
  - account_id: string
  - name: string (display name, 3-5 words)
  - canonical_name: string (normalized: lowercase, trimmed)
  - summary: string (LLM-generated, evolves over time)
  - embedding: float[] (original, immutable)
  - embedding_current: float[] (updated on significant changes)
  - action_item_count: int (denormalized count)
  - created_at: datetime
  - updated_at: datetime
  - created_from_action_item_id: UUID (provenance)

(:TopicVersion)
  - id: UUID (primary key)
  - topic_id: UUID
  - tenant_id: UUID
  - version_number: int
  - name: string
  - summary: string
  - embedding_snapshot: float[]
  - changed_by_action_item_id: UUID
  - created_at: datetime

(:Contact)
  - id: string (primary key)
  - tenant_id: UUID
  - account_id: string
  - name: string
  - email: string (optional)
  - title: string (optional)
  - created_at: datetime

(:Deal)
  - id: string (primary key)
  - tenant_id: UUID
  - account_id: string
  - name: string
  - stage: enum
  - value: float (optional)
  - currency: string
  - created_at: datetime
  - expected_close_date: datetime (optional)
  - closed_at: datetime (optional)
```

### Relationships

```
(:Account)-[:HAS_INTERACTION]->(:Interaction)
(:Account)-[:HAS_ACTION_ITEM]->(:ActionItem)
(:Account)-[:HAS_TOPIC]->(:Topic)
(:Account)-[:HAS_CONTACT]->(:Contact)
(:Account)-[:HAS_DEAL]->(:Deal)

(:ActionItem)-[:EXTRACTED_FROM {is_source: bool, confidence: float}]->(:Interaction)
  # An ActionItem can be EXTRACTED_FROM multiple Interactions (grows over time)
  # is_source=true for the original extraction, false for subsequent references

(:ActionItem)-[:OWNED_BY]->(:Owner)
(:ActionItem)-[:HAS_VERSION]->(:ActionItemVersion)
(:ActionItem)-[:INVALIDATES]->(:ActionItem)
  # When an action item supersedes another

(:ActionItem)-[:BELONGS_TO {confidence: float, method: string, created_at: datetime}]->(:Topic)
  # method: "extracted" (from extraction), "resolved" (matched to existing), "manual"

(:Topic)-[:HAS_VERSION]->(:TopicVersion)

(:Contact)-[:PARTICIPATED_IN]->(:Interaction)
  # Contacts participate in interactions, NOT directly linked to ActionItems
```

### Visual Schema

```
                              ┌──────────┐
                              │ Account  │
                              └────┬─────┘
           ┌──────────────────────┼──────────────────────┬────────────────┐
           │                      │                      │                │
           ▼                      ▼                      ▼                ▼
    ┌─────────────┐        ┌───────────┐          ┌──────────┐     ┌─────────┐
    │ Interaction │◄───────│ActionItem │─────────►│  Owner   │     │  Topic  │
    └──────┬──────┘        └─────┬─────┘          └──────────┘     └────┬────┘
           │                     │   │                                   │
           │                     │   └───────────────BELONGS_TO──────────┘
           │                     ▼                                       │
           │              ┌───────────────────┐                   ┌──────┴──────┐
           │              │ActionItemVersion  │                   │TopicVersion │
           │              └───────────────────┘                   └─────────────┘
           ▼
    ┌─────────────┐
    │  Contact    │
    └─────────────┘
```

## Dual Embedding Strategy

To prevent "embedding drift" causing duplicate detection failures:

1. **`embedding`** (immutable): Generated from the original action item text when first extracted. Used to catch semantically similar NEW action items.

2. **`embedding_current`** (mutable): Updated when the action item evolves significantly. Used to catch status updates to EXISTING action items.

### Matching Flow

```
New extraction arrives
        │
        ▼
Generate embedding for new text
        │
        ├──► Search `embedding` index ──► Find similar original items
        │
        └──► Search `embedding_current` index ──► Find similar current-state items
                │
                ▼
        Combine results, deduplicate
                │
                ▼
        LLM decides: Same item? Status update? New item?
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
    Merge   Update   Create
            Status    New
```

## Vector Indexes

```cypher
-- ActionItem: Original embeddings (immutable)
CREATE VECTOR INDEX action_item_embedding_idx
FOR (n:ActionItem) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}

-- ActionItem: Current state embeddings (mutable)
CREATE VECTOR INDEX action_item_embedding_current_idx
FOR (n:ActionItem) ON (n.embedding_current)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}

-- Topic: Original embeddings (immutable)
CREATE VECTOR INDEX topic_embedding_idx
FOR (n:Topic) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}

-- Topic: Current state embeddings (mutable)
CREATE VECTOR INDEX topic_embedding_current_idx
FOR (n:Topic) ON (n.embedding_current)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
```

## Multi-Tenancy & Account Scoping

### Scoping Strategy

| Node Type | tenant_id | account_id | Notes |
|-----------|-----------|------------|-------|
| Account | Required | N/A | Is the account |
| Interaction | Required | Required | Scoped to account |
| ActionItem | Required | Required | Scoped to account |
| Topic | Required | Required | Scoped to account |
| Owner | Required | - | Tenant-level (shared across accounts) |
| ActionItemVersion | Required | - | Historical record |
| TopicVersion | Required | - | Historical record |

### Isolation Rules

1. **tenant_id** is required on ALL nodes for complete data isolation
2. **account_id** is required on transaction-level nodes (ActionItem, Topic, Interaction)
3. **Vector searches** filter by both tenant_id AND account_id to prevent cross-account bleeding
4. **Owner nodes** are tenant-scoped (not account-scoped) to allow name resolution across accounts
5. **Version nodes** inherit tenant_id from parent but don't need account_id (historical snapshots)

### Query Scoping

All repository methods enforce scoping:

```cypher
-- Example: Get action items (requires both tenant_id and account_id)
MATCH (ai:ActionItem {tenant_id: $tenant_id, account_id: $account_id})
RETURN ai

-- Example: Vector search with scoping
CALL db.index.vector.queryNodes('action_item_embedding_idx', 10, $embedding)
YIELD node, score
WHERE node.tenant_id = $tenant_id AND node.account_id = $account_id
RETURN node, score
```

## Temporal Tracking

Inspired by Graphiti's bi-temporal model:

- **`valid_at`**: When this version of truth became valid
- **`invalid_at`**: When this version was superseded
- **`invalidated_by`**: UUID of the superseding action item

This enables:
- Time-travel queries ("What was the status on date X?")
- Audit trails
- Understanding action item evolution

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Graph Database | Neo4j Aura (5.x) |
| LLM | OpenAI GPT-4.1-mini |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Data Validation | Pydantic 2.x |
| Async | asyncio + tenacity for retries |
| Package Manager | uv |
