"""
Main pipeline orchestrator for action item extraction and graph management.

Provides end-to-end processing:
1. Validate and parse EnvelopeV1 input
2. Ensure Account exists in graph
3. Create Interaction node
4. Extract action items from transcript
5. Match against existing items (account-scoped)
6. Execute merge decisions (create/update/link)
7. Resolve topics for action items (NEW - Topic Grouping)
8. Return results with created/updated IDs
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from ..clients.neo4j_client import Neo4jClient
from ..clients.openai_client import OpenAIClient
from ..clients.postgres_client import PostgresClient
from ..errors import (
    ExtractionError,
    MatchingError,
    MergeError,
    PipelineError,
    ValidationError,
)
from ..logging import PipelineTimer, get_logger, logging_context
from ..models.action_item import ActionItem
from ..models.entities import Interaction, Owner
from ..models.envelope import EnvelopeV1
from ..models.topic import ActionItemTopic
from ..repository import ActionItemRepository
from .extractor import ActionItemExtractor, ExtractionOutput
from .matcher import ActionItemMatcher, MatchResult
from .merger import ActionItemMerger, MergeResult
from .topic_executor import TopicExecutionResult, TopicExecutor
from .topic_resolver import TopicResolver

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Result of processing an envelope through the pipeline."""

    # Identifiers
    envelope_id: str | None
    interaction_id: str
    account_id: str
    tenant_id: str

    # Action item results
    created_ids: list[str] = field(default_factory=list)
    updated_ids: list[str] = field(default_factory=list)
    linked_ids: list[str] = field(default_factory=list)

    # Statistics
    total_extracted: int = 0
    total_new_items: int = 0
    total_status_updates: int = 0
    total_matched: int = 0
    total_unmatched: int = 0

    # Details for debugging/inspection
    merge_results: list[MergeResult] = field(default_factory=list)
    extraction_notes: str | None = None

    # Topic results (Phase 7: Topic Grouping)
    topic_results: list[TopicExecutionResult] = field(default_factory=list)
    topics_created: int = 0
    topics_linked: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    processing_time_ms: int | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)

    # Error tracking (for partial success)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if pipeline completed without critical errors."""
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        """True if pipeline completed with warnings."""
        return len(self.warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'envelope_id': self.envelope_id,
            'interaction_id': self.interaction_id,
            'account_id': self.account_id,
            'tenant_id': self.tenant_id,
            'created_ids': self.created_ids,
            'updated_ids': self.updated_ids,
            'linked_ids': self.linked_ids,
            'total_extracted': self.total_extracted,
            'total_new_items': self.total_new_items,
            'total_status_updates': self.total_status_updates,
            'total_matched': self.total_matched,
            'total_unmatched': self.total_unmatched,
            'extraction_notes': self.extraction_notes,
            'topics_created': self.topics_created,
            'topics_linked': self.topics_linked,
            'processing_time_ms': self.processing_time_ms,
            'stage_timings': self.stage_timings,
            'success': self.success,
            'errors': self.errors,
            'warnings': self.warnings,
        }


class ActionItemPipeline:
    """
    End-to-end pipeline for processing transcripts into action items.

    Orchestrates:
    - ActionItemExtractor: Extract action items from text
    - ActionItemMatcher: Find matches against existing items (account-scoped)
    - ActionItemMerger: Execute create/update/merge decisions
    - ActionItemRepository: Manage graph entities and relationships

    Usage:
        pipeline = ActionItemPipeline(openai_client, neo4j_client)
        result = await pipeline.process_envelope(envelope)
    """

    def __init__(
        self,
        openai_client: OpenAIClient,
        neo4j_client: Neo4jClient,
        enable_topics: bool = True,
        postgres_client: PostgresClient | None = None,
    ):
        """
        Initialize the pipeline with required clients.

        Args:
            openai_client: Connected OpenAI client for extraction/embeddings
            neo4j_client: Connected Neo4j client for graph operations
            enable_topics: Whether to enable topic grouping (default: True)
            postgres_client: Optional Postgres client for dual-write projection
        """
        self.openai = openai_client
        self.neo4j = neo4j_client
        self.enable_topics = enable_topics
        self.postgres = postgres_client

        # Initialize pipeline components
        self.extractor = ActionItemExtractor(openai_client)
        self.matcher = ActionItemMatcher(neo4j_client, openai_client)
        self.merger = ActionItemMerger(neo4j_client, openai_client)
        self.repository = ActionItemRepository(neo4j_client)

        # Topic components (Phase 7: Topic Grouping)
        self.topic_resolver = TopicResolver(neo4j_client, openai_client)
        self.topic_executor = TopicExecutor(self.repository, openai_client)

    @classmethod
    async def from_env(cls) -> ActionItemPipeline:
        """
        Create pipeline from environment variables.

        Expects:
            OPENAI_API_KEY: OpenAI API key
            NEO4J_URI: Neo4j database URI
            NEO4J_PASSWORD: Neo4j password
            NEO4J_USERNAME: Neo4j username (optional, defaults to 'neo4j')
            NEO4J_DATABASE: Neo4j database (optional, defaults to 'neo4j')
            NEON_DATABASE_URL: Optional Neon Postgres URL for dual-write

        Returns:
            Configured and connected ActionItemPipeline
        """
        openai = OpenAIClient()
        neo4j = Neo4jClient()
        await neo4j.connect()

        postgres: PostgresClient | None = None
        neon_url = os.getenv('NEON_DATABASE_URL')
        if neon_url:
            pg = PostgresClient(neon_url)
            await pg.connect()
            postgres = pg
            logger.info('pipeline.postgres_dual_write_enabled')

        return cls(openai, neo4j, postgres_client=postgres)

    async def close(self) -> None:
        """Close all client connections."""
        await self.openai.close()
        await self.neo4j.close()
        if self.postgres is not None:
            await self.postgres.close()

    async def process_envelope(
        self,
        envelope: EnvelopeV1,
    ) -> PipelineResult:
        """
        Process an envelope through the full pipeline.

        Steps:
        1. Validate envelope has required fields
        2. Ensure Account exists in graph
        3. Create Interaction node from envelope
        4. Extract action items from transcript
        5. Match extractions against existing items (account-scoped)
        6. Execute merge decisions for each extraction
        7. Return comprehensive result

        Args:
            envelope: EnvelopeV1 with transcript content

        Returns:
            PipelineResult with created/updated action item IDs

        Raises:
            ValidationError: If envelope is missing required fields
            PipelineError: If a critical pipeline stage fails
        """
        started_at = datetime.now()
        timer = PipelineTimer()

        # Validate required fields
        if not envelope.account_id:
            raise ValidationError(
                "EnvelopeV1 must have account_id for action item processing",
                context={'tenant_id': str(envelope.tenant_id)},
            )

        tenant_id = envelope.tenant_id
        account_id = envelope.account_id
        trace_id = envelope.extras.get('trace_id') if envelope.extras else None

        # Set up logging context for this request
        with logging_context(
            trace_id=trace_id,
            tenant_id=str(tenant_id),
            account_id=account_id,
        ):
            logger.info(
                "pipeline_started",
                interaction_type=envelope.interaction_type,
                content_length=len(envelope.content.text) if envelope.content else 0,
            )

            # Initialize result
            result = PipelineResult(
                envelope_id=envelope.extras.get('envelope_id') if envelope.extras else None,
                interaction_id='',  # Will be set after creation
                account_id=account_id,
                tenant_id=str(tenant_id),
                started_at=started_at,
            )

            try:
                # Step 1: Ensure Account exists
                with timer.stage("ensure_account"):
                    await self.repository.ensure_account(
                        tenant_id=tenant_id,
                        account_id=account_id,
                        name=envelope.extras.get('account_name') if envelope.extras else None,
                    )

                # Step 2: Extract action items (creates Interaction internally)
                with timer.stage("extraction"):
                    extraction = await self.extractor.extract_from_envelope(envelope)

                result.interaction_id = str(extraction.interaction.interaction_id)
                result.total_extracted = extraction.count
                result.total_new_items = len(extraction.new_items)
                result.total_status_updates = len(extraction.status_updates)
                result.extraction_notes = extraction.extraction_notes

                logger.info(
                    "extraction_complete",
                    total_extracted=extraction.count,
                    new_items=len(extraction.new_items),
                    status_updates=len(extraction.status_updates),
                )

                # If no action items extracted, we're done
                if extraction.count == 0:
                    result.completed_at = datetime.now()
                    result.processing_time_ms = int(timer.total_ms)
                    result.stage_timings = timer.stages.copy()
                    logger.info("pipeline_complete_no_items", **timer.summary())
                    return result

                # Step 3: Create Interaction node in graph
                with timer.stage("create_interaction"):
                    await self.repository.create_interaction(extraction.interaction)

                # Step 4: Match against existing items (account-scoped)
                with timer.stage("matching"):
                    match_results, filtered_action_items = await self._match_extractions(
                        extraction=extraction,
                        tenant_id=tenant_id,
                        account_id=account_id,
                    )

                result.total_matched = sum(1 for m in match_results if m.best_match is not None)
                result.total_unmatched = len(match_results) - result.total_matched

                logger.info(
                    "matching_complete",
                    total_matched=result.total_matched,
                    total_unmatched=result.total_unmatched,
                )

                # Step 5: Execute merge decisions
                with timer.stage("merging"):
                    merge_results = await self._execute_merges(
                        match_results=match_results,
                        action_items=filtered_action_items,
                        interaction=extraction.interaction,
                    )

                # Step 6: Topic Resolution (Phase 7: Topic Grouping)
                if self.enable_topics:
                    with timer.stage("topic_resolution"):
                        topic_exec_results = await self._process_topics(
                            match_results=match_results,
                            merge_results=merge_results,
                            tenant_id=tenant_id,
                            account_id=account_id,
                        )
                        result.topic_results = topic_exec_results
                        result.topics_created = sum(1 for t in topic_exec_results if t.was_new)
                        result.topics_linked = sum(1 for t in topic_exec_results if not t.was_new)

                    logger.info(
                        "topic_resolution_complete",
                        topics_created=result.topics_created,
                        topics_linked=result.topics_linked,
                    )

                # Step 7: Dual-write to Postgres (failure-isolated)
                if self.postgres is not None:
                    with timer.stage("postgres_dual_write"):
                        await self._dual_write_postgres(
                            merge_results=merge_results,
                            action_items=filtered_action_items,
                            interaction=extraction.interaction,
                            topic_results=result.topic_results,
                        )

            except ExtractionError as e:
                logger.error("extraction_failed", error=str(e))
                result.errors.append(f"Extraction failed: {e}")
                raise
            except MatchingError as e:
                logger.error("matching_failed", error=str(e))
                result.errors.append(f"Matching failed: {e}")
                raise
            except MergeError as e:
                logger.error("merge_failed", error=str(e))
                result.errors.append(f"Merge failed: {e}")
                raise
            except Exception as e:
                logger.error("pipeline_failed", error=str(e), error_type=type(e).__name__)
                result.errors.append(f"Pipeline error: {e}")
                raise PipelineError(f"Pipeline failed: {e}", context={'stage': 'unknown'})

            # Categorize results
            for merge_result in merge_results:
                result.merge_results.append(merge_result)

                if merge_result.action == 'created':
                    result.created_ids.append(merge_result.action_item_id)
                elif merge_result.action in ('merged', 'status_updated'):
                    result.updated_ids.append(merge_result.action_item_id)
                elif merge_result.action == 'linked':
                    result.linked_ids.append(merge_result.action_item_id)

            # Finalize timing
            result.completed_at = datetime.now()
            result.processing_time_ms = int(timer.total_ms)
            result.stage_timings = timer.stages.copy()

            logger.info(
                "pipeline_complete",
                created=len(result.created_ids),
                updated=len(result.updated_ids),
                linked=len(result.linked_ids),
                **timer.summary(),
            )

            return result

    async def process_text(
        self,
        text: str,
        tenant_id: UUID,
        account_id: str,
        user_id: str | None = None,
        meeting_title: str | None = None,
        participants: list[str] | None = None,
    ) -> PipelineResult:
        """
        Process raw text through the pipeline (convenience method).

        This is a simpler interface when you don't have a full EnvelopeV1.

        Args:
            text: Transcript text to process
            tenant_id: Tenant UUID
            account_id: Account identifier (required)
            user_id: Optional user who initiated processing
            meeting_title: Optional meeting title for context
            participants: Optional list of participant names

        Returns:
            PipelineResult with created/updated action item IDs
        """
        started_at = datetime.now()
        timer = PipelineTimer()

        with logging_context(
            tenant_id=str(tenant_id),
            account_id=account_id,
        ):
            logger.info(
                "process_text_started",
                content_length=len(text),
                meeting_title=meeting_title,
            )

            # Initialize result
            result = PipelineResult(
                envelope_id=None,
                interaction_id='',
                account_id=account_id,
                tenant_id=str(tenant_id),
                started_at=started_at,
            )

            # Ensure Account exists
            with timer.stage("ensure_account"):
                await self.repository.ensure_account(
                    tenant_id=tenant_id,
                    account_id=account_id,
                )

            # Extract action items
            with timer.stage("extraction"):
                extraction = await self.extractor.extract_from_text(
                    text=text,
                    tenant_id=tenant_id,
                    account_id=account_id,
                    user_id=user_id,
                    meeting_title=meeting_title,
                    participants=participants,
                )

            result.interaction_id = str(extraction.interaction.interaction_id)
            result.total_extracted = extraction.count
            result.total_new_items = len(extraction.new_items)
            result.total_status_updates = len(extraction.status_updates)
            result.extraction_notes = extraction.extraction_notes

            if extraction.count == 0:
                result.completed_at = datetime.now()
                result.processing_time_ms = int(timer.total_ms)
                result.stage_timings = timer.stages.copy()
                logger.info("process_text_complete_no_items", **timer.summary())
                return result

            # Create Interaction in graph
            with timer.stage("create_interaction"):
                await self.repository.create_interaction(extraction.interaction)

            # Match and merge
            with timer.stage("matching"):
                match_results, filtered_action_items = await self._match_extractions(
                    extraction=extraction,
                    tenant_id=tenant_id,
                    account_id=account_id,
                )

            result.total_matched = sum(1 for m in match_results if m.best_match is not None)
            result.total_unmatched = len(match_results) - result.total_matched

            with timer.stage("merging"):
                merge_results = await self._execute_merges(
                    match_results=match_results,
                    action_items=filtered_action_items,
                    interaction=extraction.interaction,
                )

            for merge_result in merge_results:
                result.merge_results.append(merge_result)

                if merge_result.action == 'created':
                    result.created_ids.append(merge_result.action_item_id)
                elif merge_result.action in ('merged', 'status_updated'):
                    result.updated_ids.append(merge_result.action_item_id)
                elif merge_result.action == 'linked':
                    result.linked_ids.append(merge_result.action_item_id)

            # Topic Resolution (Phase 7: Topic Grouping)
            if self.enable_topics:
                with timer.stage("topic_resolution"):
                    topic_exec_results = await self._process_topics(
                        match_results=match_results,
                        merge_results=merge_results,
                        tenant_id=tenant_id,
                        account_id=account_id,
                    )
                    result.topic_results = topic_exec_results
                    result.topics_created = sum(1 for t in topic_exec_results if t.was_new)
                    result.topics_linked = sum(1 for t in topic_exec_results if not t.was_new)

            # Dual-write to Postgres (failure-isolated)
            if self.postgres is not None:
                with timer.stage("postgres_dual_write"):
                    await self._dual_write_postgres(
                        merge_results=merge_results,
                        action_items=filtered_action_items,
                        interaction=extraction.interaction,
                        topic_results=result.topic_results,
                    )

            result.completed_at = datetime.now()
            result.processing_time_ms = int(timer.total_ms)
            result.stage_timings = timer.stages.copy()

            logger.info(
                "process_text_complete",
                created=len(result.created_ids),
                updated=len(result.updated_ids),
                topics_created=result.topics_created,
                topics_linked=result.topics_linked,
                **timer.summary(),
            )

            return result

    async def _match_extractions(
        self,
        extraction: ExtractionOutput,
        tenant_id: UUID,
        account_id: str,
    ) -> tuple[list[MatchResult], list[ActionItem]]:
        """
        Match extracted items against existing items in the graph.

        Args:
            extraction: Output from extractor
            tenant_id: Tenant UUID
            account_id: Account ID (required for scoping)

        Returns:
            Tuple of (match_results, filtered_action_items), 1:1 aligned.
            Both lists have the same length and positional correspondence.
        """
        # Pair action items with their embeddings and raw extractions
        triples = [
            (ai, raw, ai.embedding)
            for ai, raw in zip(extraction.action_items, extraction.raw_extractions)
            if ai.embedding is not None
        ]

        if not triples:
            return [], []

        # Split into parallel lists — preserves positional alignment
        filtered_action_items = [ai for ai, _, _ in triples]
        extracted_pairs = [(raw, emb) for _, raw, emb in triples]

        # Run matching (account-scoped) — returns 1:1 with extracted_pairs
        match_results = await self.matcher.find_matches(
            extracted_items=extracted_pairs,
            tenant_id=tenant_id,
            account_id=account_id,
        )

        return match_results, filtered_action_items

    async def _execute_merges(
        self,
        match_results: list[MatchResult],
        action_items: list[ActionItem],
        interaction: Interaction,
    ) -> list[MergeResult]:
        """
        Execute merge decisions for all match results.

        ``match_results`` and ``action_items`` must be 1:1 aligned
        (same length, same positional order) — guaranteed by
        ``_match_extractions`` which produces both lists from the
        same filtered zip.

        Args:
            match_results: Results from matching phase
            action_items: Filtered ActionItem objects (1:1 with match_results)
            interaction: The Interaction for this processing run

        Returns:
            List of MergeResult objects (1:1 with inputs)
        """
        merge_results = []

        for match_result, action_item in zip(match_results, action_items):
            merge_result = await self.merger.execute_decision(
                match_result=match_result,
                interaction=interaction,
                action_item=action_item,
            )

            merge_results.append(merge_result)

        return merge_results

    async def get_action_items(
        self,
        tenant_id: UUID,
        account_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Retrieve action items for an account.

        Convenience method for querying the graph.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier
            status: Optional status filter (open, completed, etc.)
            limit: Maximum results

        Returns:
            List of ActionItem property dicts
        """
        return await self.repository.get_action_items_for_account(
            tenant_id=tenant_id,
            account_id=account_id,
            status=status,
            limit=limit,
        )

    async def get_topics(
        self,
        tenant_id: UUID,
        account_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Retrieve topics for an account.

        Args:
            tenant_id: Tenant UUID
            account_id: Account identifier
            limit: Maximum results

        Returns:
            List of Topic property dicts
        """
        return await self.repository.get_topics_for_account(
            tenant_id=tenant_id,
            account_id=account_id,
            limit=limit,
        )

    async def _process_topics(
        self,
        match_results: list[MatchResult],
        merge_results: list[MergeResult],
        tenant_id: UUID,
        account_id: str,
    ) -> list[TopicExecutionResult]:
        """
        Process topic resolution for all extracted action items.

        This phase runs AFTER merging to ensure action items have been persisted.
        ``match_results`` and ``merge_results`` are 1:1 aligned — both produced
        from the same filtered action item list via ``_match_extractions`` and
        ``_execute_merges``.

        Args:
            match_results: Results from matching phase (contains extracted_item with topic)
            merge_results: Results from merging phase (contains action_item_id), 1:1 with match_results
            tenant_id: Tenant UUID
            account_id: Account identifier

        Returns:
            List of TopicExecutionResult objects
        """
        topic_results = []

        for match_result, merge_result in zip(match_results, merge_results):
            extracted = match_result.extracted_item
            action_item_id = merge_result.action_item_id

            if not action_item_id:
                continue

            # Check if the extraction has a topic
            if not hasattr(extracted, 'topic') or extracted.topic is None:
                logger.debug(
                    "topic_resolution_skipped_no_topic",
                    action_item_id=action_item_id,
                )
                continue

            # Resolve the topic
            resolution = await self.topic_resolver.resolve_topic(
                extracted_topic=extracted.topic,
                action_item_id=action_item_id,
                action_item_summary=extracted.summary,
                tenant_id=tenant_id,
                account_id=account_id,
            )

            # Execute the resolution
            exec_result = await self.topic_executor.execute_resolution(
                resolution=resolution,
                tenant_id=tenant_id,
                account_id=account_id,
                action_item_text=extracted.action_item_text,
                owner=extracted.owner,
            )

            topic_results.append(exec_result)

        return topic_results

    # =========================================================================
    # Postgres Dual-Write
    # =========================================================================

    async def _dual_write_postgres(
        self,
        merge_results: list[MergeResult],
        action_items: list[ActionItem],
        interaction: Interaction,
        topic_results: list[TopicExecutionResult],
    ) -> None:
        """
        Write action items, owners, topics, and memberships to Postgres.

        This entire method is failure-isolated: any exception is logged
        and swallowed so that Neo4j (source of truth) is never affected.
        """
        if self.postgres is None:
            return

        try:
            await self._pg_write_action_items(merge_results, action_items, interaction)
        except Exception:
            logger.exception('postgres_dual_write.action_items_failed')

        try:
            await self._pg_write_topics(topic_results, action_items)
        except Exception:
            logger.exception('postgres_dual_write.topics_failed')

    async def _pg_write_action_items(
        self,
        merge_results: list[MergeResult],
        action_items: list[ActionItem],
        interaction: Interaction,
    ) -> None:
        """Dual-write action items and owners for each merge result."""
        assert self.postgres is not None

        for merge_result, action_item in zip(merge_results, action_items):
            try:
                if merge_result.was_new:
                    # For new items, the ActionItem model IS the current state
                    pg_item = action_item
                else:
                    # For updated items, fetch post-merge state from Neo4j
                    node = await self.repository.get_action_item(
                        merge_result.action_item_id, action_item.tenant_id,
                    )
                    if node is None:
                        continue
                    pg_item = _neo4j_node_to_action_item(node)

                await self.postgres.upsert_action_item(pg_item)

                # Write owner if available in merge details
                owner_name = merge_result.details.get(
                    'owner_name', action_item.owner
                )
                if owner_name:
                    owner_node = await self.repository.get_owner_by_name(
                        action_item.tenant_id, owner_name,
                    )
                    if owner_node:
                        owner = Owner(
                            id=UUID(owner_node['owner_id']),
                            tenant_id=action_item.tenant_id,
                            canonical_name=owner_node.get('canonical_name', owner_name),
                            aliases=owner_node.get('aliases', []),
                            contact_id=owner_node.get('contact_id'),
                            user_id=owner_node.get('user_id'),
                        )
                        try:
                            await self.postgres.upsert_owner(owner)
                        except Exception:
                            logger.exception(
                                'postgres_dual_write.owner_failed',
                                owner_id=str(owner.id),
                            )

                # Entity links: account + interaction
                item_id = UUID(merge_result.action_item_id)
                if pg_item.account_id:
                    try:
                        await self.postgres.link_action_item_to_entity(
                            tenant_id=pg_item.tenant_id,
                            action_item_id=item_id,
                            entity_type='account',
                            entity_id=pg_item.account_id,
                        )
                    except Exception:
                        logger.exception(
                            'postgres_dual_write.link_account_failed',
                            action_item_id=merge_result.action_item_id,
                        )

                try:
                    await self.postgres.link_action_item_to_entity(
                        tenant_id=pg_item.tenant_id,
                        action_item_id=item_id,
                        entity_type='interaction',
                        entity_id=interaction.interaction_id,
                    )
                except Exception:
                    logger.exception(
                        'postgres_dual_write.link_interaction_failed',
                        action_item_id=merge_result.action_item_id,
                    )

            except Exception:
                logger.exception(
                    'postgres_dual_write.item_failed',
                    action_item_id=merge_result.action_item_id,
                )

    async def _pg_write_topics(
        self,
        topic_results: list[TopicExecutionResult],
        action_items: list[ActionItem],
    ) -> None:
        """Dual-write topics and memberships for each topic result."""
        assert self.postgres is not None

        if not topic_results:
            return

        # Use first action item for tenant_id (all items share tenant)
        tenant_id = action_items[0].tenant_id if action_items else None
        if tenant_id is None:
            return

        for topic_result in topic_results:
            try:
                # Fetch topic from Neo4j (source of truth for current state)
                topic_node = await self.repository.get_topic(
                    topic_result.topic_id, tenant_id,
                )
                if topic_node is None:
                    continue

                topic = _neo4j_node_to_topic(topic_node)
                await self.postgres.upsert_topic(topic)

                # Membership link
                await self.postgres.upsert_topic_membership(
                    tenant_id=tenant_id,
                    action_item_id=UUID(topic_result.action_item_id),
                    topic_id=UUID(topic_result.topic_id),
                    confidence=topic_result.details.get('confidence', 1.0),
                    method=topic_result.details.get('method', 'extracted'),
                )

            except Exception:
                logger.exception(
                    'postgres_dual_write.topic_failed',
                    topic_id=topic_result.topic_id,
                    action_item_id=topic_result.action_item_id,
                )


# =============================================================================
# Neo4j → Pydantic model helpers
# =============================================================================


def _parse_iso(val: Any) -> datetime | None:
    """Parse an ISO datetime string or return None."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


def _parse_uuid(val: Any) -> UUID | None:
    """Parse a UUID string or return None."""
    if val is None:
        return None
    if isinstance(val, UUID):
        return val
    try:
        return UUID(str(val))
    except (ValueError, TypeError):
        return None


def _neo4j_node_to_action_item(node: dict[str, Any]) -> ActionItem:
    """Convert a Neo4j ActionItem node dict to an ActionItem model."""
    return ActionItem(
        id=UUID(node['action_item_id']),
        tenant_id=UUID(node['tenant_id']),
        account_id=node.get('account_id'),
        action_item_text=node.get('action_item_text', ''),
        summary=node.get('summary', ''),
        owner=node.get('owner', ''),
        owner_type=node.get('owner_type', 'named'),
        is_user_owned=node.get('is_user_owned', False),
        conversation_context=node.get('conversation_context', ''),
        due_date=_parse_iso(node.get('due_date')),
        status=node.get('status', 'open'),
        version=node.get('version', 1),
        evolution_summary=node.get('evolution_summary', ''),
        created_at=_parse_iso(node.get('created_at')) or datetime.now(),
        last_updated_at=_parse_iso(node.get('last_updated_at')) or datetime.now(),
        source_interaction_id=_parse_uuid(node.get('source_interaction_id')),
        user_id=node.get('user_id'),
        pg_user_id=_parse_uuid(node.get('pg_user_id')),
        embedding=node.get('embedding'),
        embedding_current=node.get('embedding_current'),
        confidence=node.get('confidence', 1.0),
        valid_at=_parse_iso(node.get('valid_at')),
        invalid_at=_parse_iso(node.get('invalid_at')),
        invalidated_by=_parse_uuid(node.get('invalidated_by')),
    )


def _neo4j_node_to_topic(node: dict[str, Any]) -> ActionItemTopic:
    """Convert a Neo4j ActionItemTopic node dict to an ActionItemTopic model."""
    return ActionItemTopic(
        id=UUID(node['action_item_topic_id']),
        tenant_id=UUID(node['tenant_id']),
        account_id=node.get('account_id', ''),
        name=node.get('name', ''),
        canonical_name=node.get('canonical_name', ''),
        summary=node.get('summary', ''),
        embedding=node.get('embedding'),
        embedding_current=node.get('embedding_current'),
        action_item_count=node.get('action_item_count', 0),
        created_at=_parse_iso(node.get('created_at')) or datetime.now(),
        updated_at=_parse_iso(node.get('updated_at')) or datetime.now(),
        created_from_action_item_id=_parse_uuid(node.get('created_from_action_item_id')),
        version=node.get('version', 1),
    )
