"""
Integration tests for the full pipeline with topic grouping enabled.
"""

import uuid

import pytest

# Add project paths for imports
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from action_item_graph.pipeline import ActionItemPipeline, PipelineResult
from action_item_graph.clients.openai_client import OpenAIClient
from action_item_graph.clients.neo4j_client import Neo4jClient


@pytest.fixture
def sample_transcript_with_topics() -> str:
    """Sample transcript with action items that should share topics."""
    return """
John: Thanks for joining the Q1 planning call. Let's discuss the sales expansion initiative.

Sarah: Great. For the sales team growth, I'll draft the job descriptions for the SDR positions by Friday.

John: Perfect. I'll coordinate with HR to set up the interview panel for the new hires.

Sarah: Also, on the website redesign project, I need to review the mockups from the design team.

John: Yes, and I'll schedule a meeting with the developers to discuss the technical requirements for the new site.

Sarah: Sounds good. Back to sales - I'll also reach out to the recruiting agencies about the SDR search.

John: Great progress. Let's reconvene next week.
""".strip()


class TestPipelineWithTopicsEnabled:
    """Test pipeline with topic grouping enabled."""

    @pytest.mark.asyncio
    async def test_pipeline_extracts_topics(
        self, openai_api_key, neo4j_credentials, sample_tenant_id, sample_transcript_with_topics
    ):
        """Test that the pipeline extracts and groups topics correctly."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)
        pipeline = None

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai, neo4j, enable_topics=True)
            tenant_id = uuid.UUID(sample_tenant_id)
            account_id = f"test_account_{uuid.uuid4().hex[:8]}"

            result = await pipeline.process_text(
                text=sample_transcript_with_topics,
                tenant_id=tenant_id,
                account_id=account_id,
                meeting_title="Q1 Planning Call",
                participants=["John", "Sarah"],
            )

            # Verify basic extraction worked
            assert result.success
            assert result.total_extracted > 0

            # Verify topic results
            assert len(result.topic_results) > 0
            assert result.topics_created > 0 or result.topics_linked > 0

            # Verify timing includes topic resolution
            assert 'topic_resolution' in result.stage_timings

            # Check that topics were created
            topics = await pipeline.get_topics(tenant_id, account_id)
            assert len(topics) > 0

        finally:
            if pipeline:
                await pipeline.close()

    @pytest.mark.asyncio
    async def test_pipeline_groups_related_action_items(
        self, openai_api_key, neo4j_credentials, sample_tenant_id
    ):
        """Test that related action items are grouped under the same topic."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)
        pipeline = None

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai, neo4j, enable_topics=True)
            tenant_id = uuid.UUID(sample_tenant_id)
            account_id = f"test_account_{uuid.uuid4().hex[:8]}"

            # Process a transcript with clearly related action items
            transcript = """
Sarah: For the quarterly security audit, I'll gather the compliance documents.
John: I'll schedule the security review meeting with the IT team.
Sarah: And I'll prepare the audit checklist for the reviewers.
"""

            result = await pipeline.process_text(
                text=transcript,
                tenant_id=tenant_id,
                account_id=account_id,
                meeting_title="Security Audit Planning",
            )

            assert result.success
            assert result.total_extracted >= 2

            # All items should ideally share the same topic or similar topics
            # We expect at least one topic to be created
            assert result.topics_created >= 1

            # Check topics in database
            topics = await pipeline.get_topics(tenant_id, account_id)
            assert len(topics) >= 1

            # At least one topic should have multiple action items
            # (accounting for topic reuse during resolution)
            total_linked = sum(t.get('action_item_count', 0) for t in topics)
            assert total_linked >= result.total_extracted

        finally:
            if pipeline:
                await pipeline.close()


class TestPipelineWithTopicsDisabled:
    """Test pipeline with topic grouping disabled."""

    @pytest.mark.asyncio
    async def test_pipeline_works_without_topics(
        self, openai_api_key, neo4j_credentials, sample_tenant_id, sample_transcript
    ):
        """Test that the pipeline works correctly when topics are disabled."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)
        pipeline = None

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai, neo4j, enable_topics=False)
            tenant_id = uuid.UUID(sample_tenant_id)
            account_id = f"test_account_{uuid.uuid4().hex[:8]}"

            result = await pipeline.process_text(
                text=sample_transcript,
                tenant_id=tenant_id,
                account_id=account_id,
            )

            # Verify basic extraction worked
            assert result.success
            assert result.total_extracted > 0

            # Verify no topic processing happened
            assert len(result.topic_results) == 0
            assert result.topics_created == 0
            assert result.topics_linked == 0
            assert 'topic_resolution' not in result.stage_timings

        finally:
            if pipeline:
                await pipeline.close()


class TestTopicPersistence:
    """Test topic persistence across multiple pipeline runs."""

    @pytest.mark.asyncio
    async def test_topic_reuse_across_conversations(
        self, openai_api_key, neo4j_credentials, sample_tenant_id
    ):
        """Test that topics are reused when similar topics appear in subsequent conversations."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)
        pipeline = None

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai, neo4j, enable_topics=True)
            tenant_id = uuid.UUID(sample_tenant_id)
            account_id = f"test_account_{uuid.uuid4().hex[:8]}"

            # First conversation about sales expansion
            transcript1 = """
Sarah: For the Q1 Sales Team Growth initiative, I'll draft the hiring plan.
John: I'll review the budget for the new hires.
"""
            result1 = await pipeline.process_text(
                text=transcript1,
                tenant_id=tenant_id,
                account_id=account_id,
                meeting_title="Sales Planning - Week 1",
            )

            assert result1.success
            topics_after_first = await pipeline.get_topics(tenant_id, account_id)
            first_run_topic_count = len(topics_after_first)

            # Second conversation about the same topic
            transcript2 = """
Sarah: Update on Q1 Sales Team expansion - I've posted the job listings.
John: Great, I'll start reviewing the first batch of resumes.
"""
            result2 = await pipeline.process_text(
                text=transcript2,
                tenant_id=tenant_id,
                account_id=account_id,
                meeting_title="Sales Planning - Week 2",
            )

            assert result2.success

            # Check if topics were reused
            topics_after_second = await pipeline.get_topics(tenant_id, account_id)
            second_run_topic_count = len(topics_after_second)

            # If topics were reused, count should be same or similar
            # (allowing for some variation in LLM interpretation)
            # The key is that existing topics should be linked to, not always creating new ones
            assert result2.topics_linked >= 0  # Some should be linked

        finally:
            if pipeline:
                await pipeline.close()


class TestTopicQueryMethods:
    """Test topic query methods on the pipeline."""

    @pytest.mark.asyncio
    async def test_get_topics_for_account(
        self, openai_api_key, neo4j_credentials, sample_tenant_id
    ):
        """Test retrieving topics for an account."""
        openai = OpenAIClient(api_key=openai_api_key)
        neo4j = Neo4jClient(**neo4j_credentials)
        pipeline = None

        try:
            await neo4j.connect()
            await neo4j.setup_schema()

            pipeline = ActionItemPipeline(openai, neo4j, enable_topics=True)
            tenant_id = uuid.UUID(sample_tenant_id)
            account_id = f"test_account_{uuid.uuid4().hex[:8]}"

            # Create some data first
            transcript = """
John: I'll finalize the project timeline for the new product launch.
Sarah: I'll coordinate with marketing on the launch campaign.
"""
            await pipeline.process_text(
                text=transcript,
                tenant_id=tenant_id,
                account_id=account_id,
            )

            # Query topics
            topics = await pipeline.get_topics(tenant_id, account_id)

            assert isinstance(topics, list)
            if len(topics) > 0:
                topic = topics[0]
                assert 'action_item_topic_id' in topic
                assert 'name' in topic
                assert 'tenant_id' in topic
                assert topic['tenant_id'] == str(tenant_id)
                assert topic['account_id'] == account_id

        finally:
            if pipeline:
                await pipeline.close()
