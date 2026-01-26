# Graphiti Reference Patterns

This document captures key patterns from [getzep/graphiti](https://github.com/getzep/graphiti) that are useful for our Action Item Graph pipeline.

## Overview

Graphiti is Zep's temporal knowledge graph library. It's the foundation referenced by the OpenAI cookbook for temporal graph patterns. Key features:
- Bi-temporal data model
- Hybrid search (embeddings + BM25 + graph traversal)
- Entity deduplication via LLM
- Edge invalidation for temporal facts

**Python Version**: >=3.10
**Key Dependencies**: pydantic>=2.11, neo4j>=5.26, openai>=1.91

---

## 1. Node Architecture (from `nodes.py`)

### Base Node Pattern
```python
class Node(BaseModel, ABC):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description='name of the node')
    group_id: str = Field(description='partition of the graph')  # Like tenant_id
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: utc_now())

    @abstractmethod
    async def save(self, driver: GraphDriver): ...
```

**Key Insight**: `group_id` serves as a partition key (similar to our `tenant_id`). All nodes are partitioned by this.

### Entity Node with Embeddings
```python
class EntityNode(Node):
    name_embedding: list[float] | None = Field(default=None)
    summary: str = Field(description='regional summary of surrounding edges')
    attributes: dict[str, Any] = Field(default={})

    async def generate_name_embedding(self, embedder: EmbedderClient):
        text = self.name.replace('\n', ' ')
        self.name_embedding = await embedder.create(input_data=[text])
```

**Adaptation for ActionItem**: Our ActionItem should follow this pattern with `embedding` stored directly on the node.

---

## 2. Deduplication Prompts (from `dedupe_nodes.py`)

### Single Node Deduplication
```python
class NodeDuplicate(BaseModel):
    id: int = Field(..., description='integer id of the entity')
    duplicate_idx: int = Field(
        ...,
        description='idx of the duplicate entity. -1 if no duplicate found.',
    )
    name: str = Field(
        ...,
        description='Name of the entity. Most complete and descriptive name.',
    )
    duplicates: list[int] = Field(
        ...,
        description='idx of all entities that are duplicates.',
    )
```

### Key Prompt Pattern
```
Given the above EXISTING ENTITIES and their attributes, MESSAGE, and PREVIOUS MESSAGES;
Determine if the NEW ENTITY extracted from the conversation is a duplicate of EXISTING ENTITIES.

Entities should only be considered duplicates if they refer to the *same real-world object or concept*.

Do NOT mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.
```

**Adaptation**: This exact pattern applies to our action item matching. We need to determine if a newly extracted action item is the same as an existing one.

---

## 3. Invalidation Prompts (from `invalidate_edges.py`)

### Contradiction Detection
```python
class InvalidatedEdges(BaseModel):
    contradicted_facts: list[int] = Field(
        ...,
        description='List of ids of facts that should be invalidated. Empty list if none.',
    )
```

### Key Prompt Pattern (v2 - simpler)
```
Based on the provided EXISTING FACTS and a NEW FACT, determine which existing facts the new fact contradicts.
Return a list containing all ids of the facts that are contradicted by the NEW FACT.
If there are no contradicted facts, return an empty list.

<EXISTING FACTS>
{existing_edges}
</EXISTING FACTS>

<NEW FACT>
{new_edge}
</NEW FACT>
```

**Adaptation**: We'll use this for detecting when an action item status update contradicts/supersedes the existing status.

---

## 4. Search Architecture (from `search/`)

### Search Config
```python
class SearchConfig(BaseModel):
    # Limits
    limit: int = 10

    # Search methods to use
    node_search_methods: list[SearchMethod] = [SearchMethod.bm25, SearchMethod.cosine]
    edge_search_methods: list[SearchMethod] = [SearchMethod.bm25, SearchMethod.cosine]

    # Reranking
    reranker: RerankerType = RerankerType.rrf
```

### Hybrid Search Pattern
1. **BM25 keyword search** - Fast exact matching
2. **Cosine similarity** - Semantic matching via embeddings
3. **RRF (Reciprocal Rank Fusion)** - Combines results

**Adaptation**: For action item matching:
1. First pass: Vector similarity on embeddings
2. Second pass (optional): Keyword matching on action_item_text
3. Combine with RRF or just use similarity threshold

---

## 5. Useful Cypher Patterns

### Save Node with Multiple Labels (Neo4j)
```python
# From get_entity_node_save_query
query = f"""
MERGE (n:{labels} {{uuid: $entity_data.uuid}})
SET n += $entity_data
RETURN n.uuid AS uuid
"""
```

### Batch Embedding Creation
```python
async def create_entity_node_embeddings(embedder: EmbedderClient, nodes: list[EntityNode]):
    filtered_nodes = [node for node in nodes if node.name]
    if not filtered_nodes:
        return

    name_embeddings = await embedder.create_batch([node.name for node in filtered_nodes])
    for node, name_embedding in zip(filtered_nodes, name_embeddings, strict=True):
        node.name_embedding = name_embedding
```

---

## 6. Project Structure Reference

```
graphiti/
├── graphiti_core/
│   ├── __init__.py
│   ├── graphiti.py          # Main orchestrator class
│   ├── nodes.py              # Node definitions
│   ├── edges.py              # Edge definitions
│   ├── driver/               # Neo4j/FalkorDB drivers
│   ├── embedder/             # Embedding clients
│   ├── llm_client/           # LLM clients (OpenAI, Anthropic, etc.)
│   ├── prompts/              # All LLM prompts
│   │   ├── dedupe_nodes.py
│   │   ├── dedupe_edges.py
│   │   ├── invalidate_edges.py
│   │   └── extract_nodes.py
│   ├── search/               # Hybrid search
│   └── utils/                # Helpers
├── tests/
└── pyproject.toml
```

---

## 7. Key Differences for Our Implementation

| Graphiti | Our Action Item Graph |
|----------|----------------------|
| Generic entities | Only ActionItem nodes |
| Edges as facts | ActionItem as hub node |
| `group_id` for partitioning | `tenant_id` on all nodes |
| Complex entity types | Simplified: ActionItem, Owner, Account, Interaction |
| Edge invalidation | ActionItem status + version tracking |

---

## 8. Patterns to Adopt

1. **Pydantic models with Field descriptions** - Self-documenting schemas
2. **Async throughout** - `async def save()`, `async def get_by_uuid()`
3. **Batch embedding creation** - Efficiency for multiple items
4. **Structured output for deduplication** - Clear JSON schemas for LLM responses
5. **Prompt versioning** - `versions: Versions = {'v1': v1, 'v2': v2}`
6. **Driver abstraction** - Clean separation from database specifics

---

## Source Links

- **Repository**: https://github.com/getzep/graphiti
- **Nodes**: `graphiti_core/nodes.py`
- **Deduplication**: `graphiti_core/prompts/dedupe_nodes.py`
- **Invalidation**: `graphiti_core/prompts/invalidate_edges.py`
- **Search**: `graphiti_core/search/search.py`
