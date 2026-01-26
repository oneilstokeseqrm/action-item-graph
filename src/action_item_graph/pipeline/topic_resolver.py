"""
Topic resolution for matching extracted topics to existing or new Topic nodes.

TopicResolver handles the decision logic for topic assignment:
1. Generate embedding for extracted topic
2. Search for existing topics via dual vector search
3. Apply thresholds and LLM confirmation for borderline cases
4. Return resolution decision (link_existing or create_new)

Key thresholds (higher than ActionItem's 0.65):
- >= 0.85: Auto-link to existing topic
- < 0.70: Auto-create new topic
- 0.70-0.85: LLM confirmation required
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import UUID

from ..clients.neo4j_client import Neo4jClient
from ..clients.openai_client import OpenAIClient
from ..logging import get_logger
from ..prompts.extract_action_items import ExtractedTopic
from ..prompts.topic_prompts import TopicMatchDecision, build_topic_match_prompt

logger = get_logger(__name__)


class TopicDecision(str, Enum):
    """Decision types for topic resolution."""

    CREATE_NEW = 'create_new'
    LINK_EXISTING = 'link_existing'


@dataclass
class TopicCandidate:
    """A candidate topic from vector search."""

    topic_id: str
    name: str
    canonical_name: str
    summary: str
    action_item_count: int
    similarity: float
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopicResolutionResult:
    """Result of topic resolution for a single extracted topic."""

    # Source information
    action_item_id: str
    action_item_summary: str
    extracted_topic: ExtractedTopic

    # Decision
    decision: TopicDecision
    topic_id: str | None  # Set for LINK_EXISTING, None for CREATE_NEW
    confidence: float

    # Match details
    candidates: list[TopicCandidate] = field(default_factory=list)
    best_candidate: TopicCandidate | None = None
    llm_decision: TopicMatchDecision | None = None

    # Generated embedding for the extracted topic
    embedding: list[float] | None = None

    @property
    def needs_creation(self) -> bool:
        """True if a new topic should be created."""
        return self.decision == TopicDecision.CREATE_NEW


class TopicResolver:
    """
    Resolves extracted topics to existing Topic nodes or marks for creation.

    Uses dual vector search and LLM confirmation for borderline cases.
    """

    # Similarity thresholds (higher than ActionItem's 0.65)
    SIMILARITY_AUTO_LINK = 0.85  # Auto-link if similarity >= this
    SIMILARITY_AUTO_CREATE = 0.70  # Auto-create if similarity < this
    # Between these values: LLM confirmation required

    # Vector search configuration
    VECTOR_SEARCH_LIMIT = 5
    MIN_SIMILARITY_SCORE = 0.65  # Don't consider candidates below this

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        openai_client: OpenAIClient,
        similarity_auto_link: float | None = None,
        similarity_auto_create: float | None = None,
    ):
        """
        Initialize the topic resolver.

        Args:
            neo4j_client: Connected Neo4j client
            openai_client: Connected OpenAI client
            similarity_auto_link: Override auto-link threshold (default: 0.85)
            similarity_auto_create: Override auto-create threshold (default: 0.70)
        """
        self.neo4j = neo4j_client
        self.openai = openai_client

        self.auto_link_threshold = similarity_auto_link or self.SIMILARITY_AUTO_LINK
        self.auto_create_threshold = similarity_auto_create or self.SIMILARITY_AUTO_CREATE

    async def resolve_topic(
        self,
        extracted_topic: ExtractedTopic,
        action_item_id: str,
        action_item_summary: str,
        tenant_id: UUID,
        account_id: str,
    ) -> TopicResolutionResult:
        """
        Resolve an extracted topic to an existing Topic node or mark for creation.

        Args:
            extracted_topic: Topic extracted alongside action item
            action_item_id: ID of the action item this topic was extracted with
            action_item_summary: Summary of the action item (for LLM context)
            tenant_id: Tenant UUID
            account_id: Account identifier

        Returns:
            TopicResolutionResult with decision and details
        """
        logger.debug(
            "resolving_topic",
            topic_name=extracted_topic.name,
            action_item_id=action_item_id,
        )

        # Generate embedding for the extracted topic
        topic_text = f"{extracted_topic.name}: {extracted_topic.context}"
        embedding = await self.openai.create_embedding(topic_text)

        # Search for existing topics
        candidates = await self._search_topics(
            embedding=embedding,
            tenant_id=tenant_id,
            account_id=account_id,
        )

        # Initialize result
        result = TopicResolutionResult(
            action_item_id=action_item_id,
            action_item_summary=action_item_summary,
            extracted_topic=extracted_topic,
            decision=TopicDecision.CREATE_NEW,
            topic_id=None,
            confidence=1.0,
            candidates=candidates,
            embedding=embedding,
        )

        # No candidates found - create new
        if not candidates:
            logger.info(
                "topic_resolution_no_candidates",
                topic_name=extracted_topic.name,
                decision="create_new",
            )
            return result

        # Get best candidate
        best = candidates[0]
        result.best_candidate = best

        # Apply thresholds
        if best.similarity >= self.auto_link_threshold:
            # High confidence match - auto-link
            result.decision = TopicDecision.LINK_EXISTING
            result.topic_id = best.topic_id
            result.confidence = best.similarity

            logger.info(
                "topic_resolution_auto_link",
                topic_name=extracted_topic.name,
                matched_topic=best.name,
                similarity=best.similarity,
            )

        elif best.similarity < self.auto_create_threshold:
            # Low similarity - auto-create
            result.decision = TopicDecision.CREATE_NEW
            result.confidence = 1.0 - best.similarity  # Higher confidence to create

            logger.info(
                "topic_resolution_auto_create",
                topic_name=extracted_topic.name,
                best_similarity=best.similarity,
            )

        else:
            # Borderline case - use LLM confirmation
            llm_decision = await self._llm_confirm_match(
                existing_topic=best,
                extracted_topic=extracted_topic,
                action_item_summary=action_item_summary,
                similarity_score=best.similarity,
            )
            result.llm_decision = llm_decision

            if llm_decision.decision == 'link_existing':
                result.decision = TopicDecision.LINK_EXISTING
                result.topic_id = best.topic_id
                result.confidence = llm_decision.confidence
            else:
                result.decision = TopicDecision.CREATE_NEW
                result.confidence = llm_decision.confidence

            logger.info(
                "topic_resolution_llm_confirmed",
                topic_name=extracted_topic.name,
                decision=result.decision.value,
                llm_confidence=llm_decision.confidence,
                reasoning=llm_decision.reasoning,
            )

        return result

    async def resolve_batch(
        self,
        items: list[tuple[ExtractedTopic, str, str]],
        tenant_id: UUID,
        account_id: str,
    ) -> list[TopicResolutionResult]:
        """
        Resolve multiple topics (useful for batch processing).

        Args:
            items: List of (extracted_topic, action_item_id, action_item_summary) tuples
            tenant_id: Tenant UUID
            account_id: Account identifier

        Returns:
            List of TopicResolutionResult objects
        """
        results = []
        for extracted_topic, action_item_id, action_item_summary in items:
            result = await self.resolve_topic(
                extracted_topic=extracted_topic,
                action_item_id=action_item_id,
                action_item_summary=action_item_summary,
                tenant_id=tenant_id,
                account_id=account_id,
            )
            results.append(result)
        return results

    async def _search_topics(
        self,
        embedding: list[float],
        tenant_id: UUID,
        account_id: str,
    ) -> list[TopicCandidate]:
        """
        Search for existing topics using dual vector search.

        Args:
            embedding: Query embedding
            tenant_id: Tenant UUID
            account_id: Account identifier

        Returns:
            List of TopicCandidate objects sorted by similarity
        """
        # Search both topic embedding indexes
        results = await self.neo4j.search_topics_both_embeddings(
            embedding=embedding,
            tenant_id=str(tenant_id),
            account_id=account_id,
            limit=self.VECTOR_SEARCH_LIMIT,
            min_score=self.MIN_SIMILARITY_SCORE,
        )

        candidates = []
        for result in results:
            node = result['node']
            candidates.append(
                TopicCandidate(
                    topic_id=node['id'],
                    name=node['name'],
                    canonical_name=node.get('canonical_name', ''),
                    summary=node.get('summary', ''),
                    action_item_count=node.get('action_item_count', 0),
                    similarity=result['score'],
                    properties=node,
                )
            )

        return candidates

    async def _llm_confirm_match(
        self,
        existing_topic: TopicCandidate,
        extracted_topic: ExtractedTopic,
        action_item_summary: str,
        similarity_score: float,
    ) -> TopicMatchDecision:
        """
        Use LLM to confirm whether topics match in borderline cases.

        Args:
            existing_topic: Best candidate from vector search
            extracted_topic: Newly extracted topic
            action_item_summary: Summary of the action item
            similarity_score: Cosine similarity between embeddings

        Returns:
            TopicMatchDecision from LLM
        """
        messages = build_topic_match_prompt(
            existing_name=existing_topic.name,
            existing_summary=existing_topic.summary,
            existing_count=existing_topic.action_item_count,
            new_name=extracted_topic.name,
            new_context=extracted_topic.context,
            action_item_summary=action_item_summary,
            similarity_score=similarity_score,
        )

        decision = await self.openai.chat_completion_structured(
            messages=messages,
            response_model=TopicMatchDecision,
            temperature=0.0,
        )

        return decision
