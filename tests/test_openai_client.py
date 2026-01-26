"""
Live integration tests for OpenAI client.

These tests hit the actual OpenAI API and require OPENAI_API_KEY to be set.
Run with: pytest tests/test_openai_client.py -v
"""

import pytest
from pydantic import BaseModel, Field

from action_item_graph.clients.openai_client import OpenAIClient


class TestOpenAIHealth:
    """Test OpenAI API connectivity."""

    @pytest.mark.asyncio
    async def test_health_check(self, openai_api_key: str):
        """Verify we can connect to OpenAI API."""
        client = OpenAIClient(api_key=openai_api_key)
        try:
            result = await client.health_check()
            assert result['healthy'] is True
            assert 'chat_model' in result
            assert 'embedding_model' in result
            print(f"\nOpenAI Health Check: {result}")
        finally:
            await client.close()


class TestOpenAIEmbeddings:
    """Test embedding generation."""

    @pytest.mark.asyncio
    async def test_single_embedding(self, openai_api_key: str):
        """Test creating a single embedding."""
        client = OpenAIClient(api_key=openai_api_key)
        try:
            text = "Send the updated proposal deck to the client by Friday"
            embedding = await client.create_embedding(text)

            assert isinstance(embedding, list)
            assert len(embedding) == 1536  # Default dimensions
            assert all(isinstance(x, float) for x in embedding)
            print(f"\nEmbedding dimensions: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_batch_embeddings(self, openai_api_key: str):
        """Test creating multiple embeddings in a batch."""
        client = OpenAIClient(api_key=openai_api_key)
        try:
            texts = [
                "Send the proposal by Friday",
                "Schedule a follow-up demo next week",
                "Loop in legal to review contracts",
            ]
            embeddings = await client.create_embeddings_batch(texts)

            assert len(embeddings) == 3
            assert all(len(e) == 1536 for e in embeddings)
            print(f"\nCreated {len(embeddings)} embeddings")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_embedding_similarity(self, openai_api_key: str):
        """Test that similar texts have similar embeddings."""
        import numpy as np

        client = OpenAIClient(api_key=openai_api_key)
        try:
            texts = [
                "Send the proposal deck to the client",  # Original
                "Email the proposal document to the customer",  # Similar
                "Schedule a meeting for next Tuesday",  # Different
            ]
            embeddings = await client.create_embeddings_batch(texts)

            # Calculate cosine similarities
            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            sim_01 = cosine_sim(embeddings[0], embeddings[1])  # Similar texts
            sim_02 = cosine_sim(embeddings[0], embeddings[2])  # Different texts

            print(f"\nSimilarity (proposal vs proposal): {sim_01:.4f}")
            print(f"Similarity (proposal vs meeting): {sim_02:.4f}")

            # Similar texts should have higher similarity
            assert sim_01 > sim_02
            assert sim_01 > 0.6  # Should be reasonably similar (text-embedding-3-small)
        finally:
            await client.close()


class TestOpenAIStructuredOutput:
    """Test structured output with Pydantic models."""

    @pytest.mark.asyncio
    async def test_structured_extraction(self, openai_api_key: str, sample_transcript: str):
        """Test extracting structured data from text."""

        # Define a simple extraction model
        class ExtractedActionItem(BaseModel):
            action_item_text: str = Field(description="The action item text")
            owner: str = Field(description="Who is responsible")
            due_date_text: str | None = Field(
                default=None, description="Due date if mentioned"
            )

        class ExtractionResult(BaseModel):
            action_items: list[ExtractedActionItem]

        client = OpenAIClient(api_key=openai_api_key)
        try:
            messages = [
                {
                    'role': 'system',
                    'content': 'Extract action items from the transcript. '
                    'An action item is a specific task that someone commits to doing.',
                },
                {'role': 'user', 'content': sample_transcript},
            ]

            result = await client.chat_completion_structured(
                messages=messages,
                response_model=ExtractionResult,
            )

            assert isinstance(result, ExtractionResult)
            assert len(result.action_items) > 0

            print(f"\nExtracted {len(result.action_items)} action items:")
            for i, item in enumerate(result.action_items, 1):
                print(f"  {i}. {item.owner}: {item.action_item_text}")
                if item.due_date_text:
                    print(f"     Due: {item.due_date_text}")
        finally:
            await client.close()


class TestOpenAIChatCompletion:
    """Test basic chat completion."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, openai_api_key: str):
        """Test a simple chat completion."""
        client = OpenAIClient(api_key=openai_api_key)
        try:
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Say "test successful" and nothing else.'},
            ]

            response = await client.chat_completion(messages=messages)
            assert 'test successful' in response.lower()
            print(f"\nChat response: {response}")
        finally:
            await client.close()
