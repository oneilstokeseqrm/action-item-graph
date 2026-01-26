# Action Item Graph - Requirements

## Project Goal

Transform the OpenAI cookbook temporal knowledge graph demo into a specialized **Action Item extraction pipeline** that:

1. Extracts ONLY action items from call transcripts (not generic entities/triplets)
2. Maintains temporal awareness for action item lifecycle
3. Supports multi-tenancy with proper data isolation
4. Uses Neo4j as the graph store with native vector search

## Functional Requirements

### FR1: Action Item Extraction

- **FR1.1**: Extract action items from transcript text with the following fields:
  - `action_item_text`: Verbatim or near-verbatim text from transcript
  - `owner`: Person responsible for the action item
  - `summary`: Concise LLM-generated summary
  - `conversation_context`: Surrounding context that clarifies the item
  - `due_date`: Extracted date if mentioned (optional)
  - `status`: Initial status (default: open)

- **FR1.2**: Support dual-mode extraction:
  - **New action items**: Fresh commitments/tasks from the conversation
  - **Status updates**: References to existing items ("I sent that deck", "The proposal is done")

- **FR1.3**: Generate embeddings for each extracted action item using OpenAI

### FR2: Action Item Matching & Deduplication

- **FR2.1**: Use vector similarity search to find potentially matching action items
- **FR2.2**: Use LLM to make final deduplication decision (same item vs. related but distinct)
- **FR2.3**: Support matching against both original and current-state embeddings

### FR3: Action Item Lifecycle Management

- **FR3.1**: Track action item status: open → in_progress → completed/cancelled/deferred
- **FR3.2**: Maintain version history for all changes
- **FR3.3**: Record which interaction triggered each change
- **FR3.4**: Support temporal validity tracking (valid_at, invalid_at, invalidated_by)

### FR4: Multi-Tenancy

- **FR4.1**: All nodes must include `tenant_id` property
- **FR4.2**: All queries must filter by `tenant_id`
- **FR4.3**: `tenant_id` is NOT a first-class node in the graph

### FR5: Graph Relationships

- **FR5.1**: `Account` is the root node for CRM context
- **FR5.2**: `ActionItem` connects to `Account` directly (HAS_ACTION_ITEM)
- **FR5.3**: `ActionItem` connects to ALL related `Interaction` nodes (EXTRACTED_FROM)
  - Relationship grows over time as action item is referenced in more conversations
- **FR5.4**: `Contact` connects to `Interaction` (PARTICIPATED_IN), NOT directly to `ActionItem`
- **FR5.5**: `Owner` is separate from `Contact` (action item ownership vs. meeting participation)

### FR6: Input Payload Format

Must accept `EnvelopeV1` format:
```json
{
  "schema_version": "v1",
  "tenant_id": "UUID (required)",
  "user_id": "string (required)",
  "interaction_type": "transcript|note|document (required)",
  "content": {
    "text": "string (required)",
    "format": "plain|markdown|diarized"
  },
  "timestamp": "ISO 8601 datetime (required)",
  "source": "web-mic|upload|api|import (required)",
  "extras": {
    "opportunity_id": "string (optional)",
    "contact_ids": ["string"] (optional),
    "meeting_title": "string (optional)",
    "duration_seconds": "int (optional)"
  },
  "interaction_id": "UUID (optional)",
  "trace_id": "string (optional)",
  "account_id": "string (optional but recommended)"
}
```

## Non-Functional Requirements

### NFR1: Performance

- Extraction should complete within 10 seconds for typical transcripts (<5000 words)
- Vector search should return results within 500ms
- Support batch processing of multiple transcripts

### NFR2: Reliability

- Retry logic for OpenAI API calls (exponential backoff)
- Retry logic for Neo4j operations
- Graceful handling of partial failures

### NFR3: Observability

- Structured logging with configurable levels
- Trace IDs for distributed tracing support
- Confidence scores on extractions

### NFR4: Maintainability

- Pydantic models for all data structures
- Type hints throughout
- Comprehensive test coverage (live integration tests)

## Constraints

### C1: Technology Constraints

- Python 3.10+
- Neo4j 5.x (Aura cloud)
- OpenAI API (GPT-4.1-mini for extraction, text-embedding-3-small for embeddings)
- 1536-dimension embeddings

### C2: Data Constraints

- All embeddings stored directly on nodes (not in separate vector store)
- Maximum transcript size: ~100KB text
- Embedding dimensions: 1536 (fixed for text-embedding-3-small)

### C3: Security Constraints

- Credentials stored in environment variables
- No secrets in code or logs
- Tenant isolation via `tenant_id` filtering

## Safeguards

### S1: Embedding Drift Prevention

**Problem**: As action items evolve, their embeddings change. A status update like "I sent that deck" might not match the original "I'll send the proposal deck by Friday" if only current embeddings are used.

**Solution**: Dual embeddings
- `embedding`: Original (immutable) - catches similar new items
- `embedding_current`: Current state (mutable) - catches status updates

### S2: Status Update Detection

**Problem**: Status updates often don't look like action items ("Done!", "I finished that", "The deck is sent").

**Solution**: Dual-mode extraction prompt that explicitly looks for:
1. New action items (commitments, tasks)
2. Status updates (references to previous commitments)

## Success Criteria

1. **Extraction Accuracy**: >90% of action items in test transcripts correctly extracted
2. **Deduplication Accuracy**: >95% correct match/no-match decisions
3. **Status Update Detection**: >85% of status updates correctly linked to existing items
4. **Test Coverage**: All core functionality covered by live integration tests
5. **Multi-tenancy**: Complete data isolation verified by tests

## Out of Scope (v1)

- Real-time streaming extraction
- Custom embedding models
- Multiple LLM provider support
- UI/Dashboard
- Batch import from external CRM systems
- Automated owner resolution to Contact records
