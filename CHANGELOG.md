# Changelog

All notable changes to the Action Item Graph project.

## [0.2.0] - 2026-01-25

### Added - Topic Grouping (Phase 7)

#### New Components
- `TopicResolver` - Matches extracted topics to existing using threshold-based logic
- `TopicExecutor` - Creates topics, links action items, generates evolving summaries
- `Topic` model - High-level theme/project node with dual embeddings
- `TopicVersion` model - Historical snapshot for topic evolution tracking
- `ExtractedTopic` model - Topic extracted alongside action items

#### New Graph Elements
- `(:Topic)` node with dual embeddings (`embedding`, `embedding_current`)
- `(:TopicVersion)` node for version history
- `(:Account)-[:HAS_TOPIC]->(:Topic)` relationship
- `(:ActionItem)-[:BELONGS_TO]->(:Topic)` relationship
- `topic_embedding_idx` and `topic_embedding_current_idx` vector indexes

#### Topic Resolution Thresholds
| Threshold | Value | Behavior |
|-----------|-------|----------|
| Auto-link | >= 0.85 | Automatically link to existing topic |
| Auto-create | < 0.70 | Automatically create new topic |
| LLM confirm | 0.70-0.85 | Use LLM to confirm match |

#### Pipeline Integration
- Topic extraction happens during action item extraction (same LLM call)
- Topic resolution runs after action item merging
- `PipelineResult` includes `topics_created` and `topics_linked` counts
- Feature flag: `enable_topics=True/False` on pipeline initialization

#### Documentation
- `docs/PHASE7_TOPIC_GROUPING.md` - Feature documentation
- `docs/PIPELINE_GUIDE.md` - Comprehensive architecture guide
- `docs/test-data-report.md` - Test results report

### Fixed
- Scoping issue in `get_topic_with_action_items()` - now filters action items by tenant_id

### Changed
- `ExtractedActionItem` now includes required `topic` field
- Pipeline stages now include `topic_resolution` in timing
- Updated all tests to include topic field in test data

---

## [0.1.0] - 2026-01-23

### Added

#### Core Pipeline
- `ActionItemPipeline` - Main orchestrator for extraction, matching, and merging
- `ActionItemExtractor` - GPT-4.1-mini powered extraction with structured output
- `ActionItemMatcher` - Vector similarity search + LLM deduplication
- `ActionItemMerger` - Execute merge decisions (create, merge, update_status, link)
- `ActionItemRepository` - Graph CRUD operations

#### Models
- `EnvelopeV1` - API input payload format
- `ActionItem` - Core action item model with dual embeddings
- `ActionItemVersion` - Version history tracking
- `Account`, `Interaction`, `Owner`, `Contact`, `Deal` - Graph entity models
- `ExtractedActionItem`, `StatusUpdate` - Extraction output models

#### Clients
- `Neo4jClient` - Async Neo4j client with connection pooling and retries
- `OpenAIClient` - Async OpenAI client with structured output support

#### Features
- **Dual-mode extraction**: Extracts both new action items and status updates
- **Dual embeddings**: `embedding` (immutable original) + `embedding_current` (mutable)
- **Multi-tenancy**: `tenant_id` on all nodes with query-level isolation
- **Temporal tracking**: `valid_at`, `invalid_at`, version history
- **Vector search**: Neo4j native vector indexes for similarity matching
- **LLM deduplication**: GPT-4.1-mini decides same/related/new

#### Observability
- Structured logging with `structlog`
- `PipelineTimer` for stage-level timing
- `logging_context()` for trace ID propagation
- Confidence scores on extractions

#### Error Handling
- Typed exception hierarchy (`ActionItemGraphError` → specialized errors)
- `PartialSuccessResult` for batch operations
- Error wrapping utilities for OpenAI and Neo4j

#### Testing
- 60+ tests covering all pipeline stages
- Live integration tests with Neo4j and OpenAI
- Real transcript testing infrastructure (`examples/transcripts/`)

#### Documentation
- `README.md` - Project overview and quick start
- `ARCHITECTURE.md` - Detailed graph schema and design
- `REQUIREMENTS.md` - Functional and non-functional requirements
- `docs/API.md` - Comprehensive API reference
- `examples/transcripts/README.md` - Transcript testing guide

### Technical Details

#### Graph Schema
```
(:Account)-[:HAS_INTERACTION]->(:Interaction)
(:Account)-[:HAS_ACTION_ITEM]->(:ActionItem)
(:ActionItem)-[:EXTRACTED_FROM]->(:Interaction)
(:ActionItem)-[:OWNED_BY]->(:Owner)
(:ActionItem)-[:HAS_VERSION]->(:ActionItemVersion)
```

#### Vector Indexes
- `action_item_embedding_idx` - Original embeddings (1536 dimensions, cosine)
- `action_item_embedding_current_idx` - Current state embeddings

#### LLM Configuration
- Extraction: `gpt-4.1-mini` with structured output
- Embeddings: `text-embedding-3-small` (1536 dimensions)
- Deduplication: `gpt-4.1-mini` with Literal type constraints

#### Matching Threshold
- Default similarity threshold: 0.65 (cosine)
- Searches both `embedding` and `embedding_current` indexes
- LLM makes final same/related/new decision

### Dependencies
- Python 3.10+
- neo4j >= 5.26.0
- openai >= 1.50.0
- pydantic >= 2.11.0
- structlog >= 24.0.0
- tenacity >= 9.0.0
- numpy >= 1.26.0

---

## Future Considerations

### Potential Enhancements
- Batch processing optimization
- Custom embedding model support
- Real-time streaming extraction
- Owner resolution to Contact records
- Confidence threshold tuning
- Parallel matching for large batches

### Known Limitations
- Maximum transcript size: ~100KB
- Single LLM provider (OpenAI)
- No UI/Dashboard
- No automated CRM import
