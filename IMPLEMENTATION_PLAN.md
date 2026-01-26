# Action Item Graph - Implementation Plan

## Phase Overview

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Foundation & Connectivity | ✅ Complete |
| Phase 2 | Extraction Pipeline | ✅ Complete |
| Phase 3 | Matching & Deduplication | ✅ Complete |
| Phase 4 | Graph Operations | 🔄 Next |
| Phase 5 | Integration & Testing | ⏳ Pending |
| Phase 6 | Refinement | ⏳ Pending |

---

## Phase 1: Foundation & Connectivity ✅

### Completed Tasks

1. **Project Setup**
   - [x] Create project structure with `pyproject.toml`
   - [x] Set up virtual environment with `uv`
   - [x] Configure `.env` and `.env.example`
   - [x] Install dependencies

2. **Core Data Models** (`src/action_item_graph/models/`)
   - [x] `envelope.py` - EnvelopeV1 input payload model
   - [x] `action_item.py` - ActionItem and ActionItemVersion models
   - [x] `entities.py` - Account, Interaction, Owner, Contact, Deal models

3. **Client Wrappers** (`src/action_item_graph/clients/`)
   - [x] `openai_client.py` - Chat completions, structured output, embeddings
   - [x] `neo4j_client.py` - Connection, schema setup, vector search

4. **Live Integration Tests** (`tests/`)
   - [x] OpenAI health check, embeddings, structured output
   - [x] Neo4j health check, schema setup, CRUD, vector search
   - [x] All 13 tests passing

### Files Created

```
src/action_item_graph/
├── __init__.py
├── config.py
├── models/
│   ├── __init__.py
│   ├── envelope.py
│   ├── action_item.py
│   └── entities.py
├── clients/
│   ├── __init__.py
│   ├── openai_client.py
│   └── neo4j_client.py
└── pipeline/
    ├── __init__.py
    ├── extractor.py
    └── matcher.py

prompts/
└── extract_action_items.py

tests/
├── __init__.py
├── conftest.py
├── test_openai_client.py
├── test_neo4j_client.py
├── test_extractor.py
└── test_matcher.py
```

---

## Phase 2: Extraction Pipeline ✅

### Completed Tasks

1. **Extraction Prompts** (`prompts/`)
   - [x] `extract_action_items.py` - Extract new action items AND status updates
   - [x] `ExtractionResult` and `ExtractedActionItem` response models
   - [x] `DeduplicationDecision` model for Phase 3
   - [x] `build_extraction_prompt()` and `build_deduplication_prompt()` helpers

2. **Extraction Service** (`src/action_item_graph/pipeline/`)
   - [x] `extractor.py` - Main extraction orchestration
   - [x] `ActionItemExtractor` class with `extract_from_envelope()` and `extract_from_text()`
   - [x] `ExtractionOutput` class with `new_items` and `status_updates` properties
   - [x] Batch embedding generation for efficiency

3. **Tests** (`tests/test_extractor.py`)
   - [x] Test extraction from sample transcripts (5 tests, all passing)
   - [x] Test status update detection
   - [x] Test no action items scenario
   - [x] Test complex sales call transcript

### Extraction Prompt Design

```
System: You are an expert at identifying action items from sales call transcripts.
An action item is a specific task or commitment that someone agrees to do.

Extract TWO types of items:
1. NEW ACTION ITEMS: Fresh commitments made during this conversation
2. STATUS UPDATES: References to previously committed items (e.g., "I sent that deck", "Done!")

For each item, provide:
- action_item_text: The verbatim or near-verbatim text
- owner: Who is responsible (name as stated)
- summary: 1-sentence description of what needs to be done
- conversation_context: Surrounding context (1-2 sentences)
- due_date_text: Due date if mentioned (null if not)
- is_status_update: true if this is a status update, false if new item
- implied_status: For status updates, what status does this imply? (completed, in_progress, etc.)
```

---

## Phase 3: Matching & Deduplication ✅

### Completed Tasks

1. **Matching Service** (`src/action_item_graph/pipeline/`)
   - [x] `matcher.py` - Find candidate matches via vector search
   - [x] Query both embedding indexes with `search_both_embeddings`
   - [x] Combine and rank results by similarity score

2. **Deduplication Prompts** (`prompts/`)
   - [x] `DeduplicationDecision` model with Literal types
   - [x] `build_deduplication_prompt()` helper
   - [x] Tightened `implied_status` and `merge_recommendation` to Literal types

3. **Matcher Components**
   - [x] `MatchCandidate` dataclass for search results
   - [x] `MatchResult` dataclass with candidates and decisions
   - [x] `BatchMatchResult` for batch matching operations
   - [x] `match_batch()` convenience function

4. **Tests** (`tests/test_matcher.py`)
   - [x] Test vector search returns relevant candidates
   - [x] Test deduplication decisions (same item, status update, different items)
   - [x] Test best match selection logic
   - [x] Test end-to-end extraction + matching pipeline
   - [x] All 9 tests passing (27 total)

### Key Design Decisions

- **Dual Embedding Search**: Queries both `embedding` (original) and `embedding_current` indexes
- **LLM-based Deduplication**: Uses structured output for deterministic decisions
- **Literal Types**: `implied_status` and `merge_recommendation` use Literal types to constrain LLM output
- **Best Match Selection**: Prioritizes `is_same_item=True` with highest confidence

---

## Phase 4: Graph Operations ⏳

### Tasks

1. **Graph Repository** (`src/action_item_graph/`)
   - [ ] `repository.py` - High-level graph operations
   - [ ] Create Account if not exists
   - [ ] Create Interaction node
   - [ ] Create/update ActionItem nodes
   - [ ] Create relationships (EXTRACTED_FROM, OWNED_BY, etc.)
   - [ ] Create version snapshots

2. **Owner Resolution**
   - [ ] Match extracted owner names to existing Owner nodes
   - [ ] Create new Owner if no match
   - [ ] Handle aliases ("John", "John Smith", "JS")

3. **Tests**
   - [ ] Test full graph creation flow
   - [ ] Test relationship creation
   - [ ] Test version tracking

---

## Phase 5: Integration & Testing ⏳

### Tasks

1. **Main Pipeline** (`src/action_item_graph/`)
   - [ ] `pipeline.py` - Orchestrate full flow
   - [ ] Input: EnvelopeV1 payload
   - [ ] Output: Created/updated ActionItem IDs

2. **End-to-End Tests**
   - [ ] Test with sample transcripts
   - [ ] Test multi-turn conversations (same action item referenced multiple times)
   - [ ] Test status update detection and linking
   - [ ] Test multi-tenancy isolation

3. **Example Script**
   - [ ] `examples/process_transcript.py` - Demonstrate usage

---

## Phase 6: Refinement ⏳

### Tasks

1. **Error Handling**
   - [ ] Graceful degradation on LLM failures
   - [ ] Partial success handling

2. **Logging & Observability**
   - [ ] Structured logging
   - [ ] Trace ID propagation
   - [ ] Timing metrics

3. **Documentation**
   - [ ] API documentation
   - [ ] Usage examples
   - [ ] Troubleshooting guide

---

## Current Progress

### What's Working

- ✅ OpenAI API connectivity (chat, structured output, embeddings)
- ✅ Neo4j connectivity (Aura cloud)
- ✅ Schema setup (constraints, indexes, vector indexes)
- ✅ Vector similarity search with tenant filtering
- ✅ Dual embedding search (original + current)
- ✅ All Pydantic models defined
- ✅ Action item extraction from transcripts
- ✅ Status update detection (dual-mode extraction)
- ✅ Batch embedding generation
- ✅ Matcher service with candidate search
- ✅ LLM-based deduplication decisions
- ✅ Literal types for constrained LLM output
- ✅ 27/27 tests passing

### Next Steps

1. Implement `merger.py` - Merge action items or create new
2. Create version snapshots before updates
3. Implement graph repository for CRUD operations

---

## Running the Project

```bash
# Setup
cd /Users/peteroneil/action-item-graph
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_openai_client.py::TestOpenAIHealth -v
```

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=sk-...
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```
