# Phase 7: Topic Grouping Feature

## Overview

Topic Grouping clusters related action items into shared high-level themes/projects (e.g., "Q3 Audit", "Website Redesign"). This enables users to retrieve the full "story" of a project across conversations.

## Implementation Summary

### New Files Created

| File | Purpose |
|------|---------|
| `src/action_item_graph/models/topic.py` | ActionItemTopic, ActionItemTopicVersion, ExtractedTopic models |
| `src/action_item_graph/pipeline/topic_resolver.py` | TopicResolver - matching extracted topics to existing or new |
| `src/action_item_graph/pipeline/topic_executor.py` | TopicExecutor - creating topics and linking action items |
| `prompts/topic_prompts.py` | LLM prompts for topic matching and summary generation |
| `tests/test_topic_resolver.py` | Unit tests for topic resolution |
| `tests/test_topic_executor.py` | Unit tests for topic execution |
| `tests/test_pipeline_with_topics.py` | Integration tests for full pipeline |

### Modified Files

| File | Changes |
|------|---------|
| `prompts/extract_action_items.py` | Added `ExtractedTopic` model and `topic` field to `ExtractedActionItem` |
| `prompts/__init__.py` | Exported topic-related classes and functions |
| `src/action_item_graph/models/__init__.py` | Exported ActionItemTopic, ActionItemTopicVersion, ExtractedTopic |
| `src/action_item_graph/pipeline/__init__.py` | Exported TopicResolver, TopicExecutor, and related classes |
| `src/action_item_graph/pipeline/pipeline.py` | Integrated topic resolution phase into pipeline |
| `src/action_item_graph/repository.py` | Added topic CRUD methods |
| `src/action_item_graph/clients/neo4j_client.py` | Added topic vector indexes and search methods |

## Architecture

### Pipeline Flow

```
Extraction → Matching → Merging → [Topic Resolution] → Complete
                                         ↑
                                    NEW PHASE
```

Topic Resolution runs AFTER action item merging to ensure action items have been persisted.

### Key Thresholds

| Threshold | Value | Behavior |
|-----------|-------|----------|
| Auto-link | >= 0.85 | Automatically link to existing topic |
| Auto-create | < 0.70 | Automatically create new topic |
| LLM confirm | 0.70-0.85 | Use LLM to confirm match (bias toward create) |

These thresholds are higher than ActionItem's 0.65 because incorrect topic grouping has more impact.

### Dual Embedding Pattern

Topics use the same dual embedding strategy as ActionItems:
- `embedding`: Original (immutable) - catches items similar to original topic scope
- `embedding_current`: Current (mutable) - catches items related to evolved scope

### Neo4j Schema Additions

**Nodes:**
- `ActionItemTopic` - High-level theme/project
- `ActionItemTopicVersion` - Historical snapshot of topic state

**Relationships:**
- `(:Account)-[:HAS_TOPIC]->(:ActionItemTopic)`
- `(:ActionItem)-[:BELONGS_TO]->(:ActionItemTopic)`
- `(:ActionItemTopic)-[:HAS_VERSION]->(:ActionItemTopicVersion)`

**Vector Indexes:**
- `action_item_topic_embedding_idx` - For original topic embeddings
- `action_item_topic_embedding_current_idx` - For current topic embeddings

## Usage

### Enable/Disable Topics

```python
# Topics enabled by default
pipeline = ActionItemPipeline(openai, neo4j, enable_topics=True)

# Disable topics for gradual rollout
pipeline = ActionItemPipeline(openai, neo4j, enable_topics=False)
```

### Query Topics

```python
# Get all topics for an account
topics = await pipeline.get_topics(tenant_id, account_id)

# Get topic with linked action items
topic = await repository.get_topic_with_action_items(topic_id, tenant_id)
```

### Pipeline Results

```python
result = await pipeline.process_text(text, tenant_id, account_id)

# Topic statistics
print(f"Topics created: {result.topics_created}")
print(f"Topics linked: {result.topics_linked}")
print(f"Topic results: {result.topic_results}")

# Timing includes topic resolution
print(f"Topic resolution time: {result.stage_timings.get('topic_resolution', 0)}ms")
```

## Testing

```bash
# Run topic-specific tests
pytest tests/test_topic_resolver.py -v
pytest tests/test_topic_executor.py -v
pytest tests/test_pipeline_with_topics.py -v

# Run all tests
pytest tests/ -v
```

## Configuration

Topic thresholds can be customized:

```python
resolver = TopicResolver(
    neo4j_client,
    openai_client,
    similarity_auto_link=0.90,    # Higher threshold for stricter matching
    similarity_auto_create=0.75,  # Higher threshold for more new topics
)
```

## Future Extensions (Out of Scope)

- **Topic Aliasing**: Manual merging of duplicate topics
- **Topic Hierarchy**: Nested topics (parent/child relationships)
- **Cross-Account Topics**: Tenant-level topics for enterprise
