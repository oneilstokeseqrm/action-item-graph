"""
OpenAI client wrapper for the Action Item Graph pipeline.

Handles:
- Chat completions with structured output (Pydantic model parsing)
- Embeddings generation (single and batch)
- Retry logic with exponential backoff
"""

import os
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# Type variable for structured output parsing
T = TypeVar('T', bound=BaseModel)


class OpenAIClient:
    """
    Async OpenAI client with structured output and embedding support.

    Configuration via environment variables:
    - OPENAI_API_KEY: Required API key
    - OPENAI_CHAT_MODEL: Chat model (default: gpt-4.1-mini)
    - OPENAI_EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small)
    - OPENAI_EMBEDDING_DIMENSIONS: Embedding dimensions (default: 1536)
    """

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            chat_model: Model for chat completions (defaults to OPENAI_CHAT_MODEL or gpt-4.1-mini)
            embedding_model: Model for embeddings (defaults to OPENAI_EMBEDDING_MODEL or text-embedding-3-small)
            embedding_dimensions: Embedding vector dimensions (defaults to OPENAI_EMBEDDING_DIMENSIONS or 1536)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY environment variable is required')

        self.chat_model = chat_model or os.getenv('OPENAI_CHAT_MODEL', 'gpt-4.1-mini')
        self.embedding_model = embedding_model or os.getenv(
            'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'
        )
        self.embedding_dimensions = embedding_dimensions or int(
            os.getenv('OPENAI_EMBEDDING_DIMENSIONS', '1536')
        )

        self._client = AsyncOpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """
        Get a chat completion response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default chat model
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            The assistant's response text
        """
        response = await self._client.chat.completions.create(
            model=model or self.chat_model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ''

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def chat_completion_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """
        Get a chat completion with structured output (Pydantic model).

        Uses OpenAI's native structured output via response_format.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class for the response
            model: Override the default chat model
            temperature: Sampling temperature

        Returns:
            Parsed Pydantic model instance
        """
        response = await self._client.beta.chat.completions.parse(
            model=model or self.chat_model,
            messages=messages,  # type: ignore
            response_format=response_model,
            temperature=temperature,
        )

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError('Failed to parse structured response')
        return parsed

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def create_embedding(self, text: str) -> list[float]:
        """
        Create an embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Clean the text - remove newlines, extra whitespace
        cleaned_text = ' '.join(text.split())

        response = await self._client.embeddings.create(
            model=self.embedding_model,
            input=cleaned_text,
            dimensions=self.embedding_dimensions,
        )
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def create_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Create embeddings for multiple texts in a single API call.

        More efficient than calling create_embedding() multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        # Clean texts
        cleaned_texts = [' '.join(t.split()) for t in texts]

        response = await self._client.embeddings.create(
            model=self.embedding_model,
            input=cleaned_texts,
            dimensions=self.embedding_dimensions,
        )

        # Sort by index to ensure order matches input
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [e.embedding for e in sorted_embeddings]

    async def health_check(self) -> dict[str, bool | str]:
        """
        Verify API connectivity with a minimal request.

        Returns:
            Dict with 'healthy' bool and optional 'error' message
        """
        try:
            # Minimal embedding request to verify connectivity
            await self._client.embeddings.create(
                model=self.embedding_model,
                input='health check',
                dimensions=self.embedding_dimensions,
            )
            return {
                'healthy': True,
                'chat_model': self.chat_model,
                'embedding_model': self.embedding_model,
                'embedding_dimensions': self.embedding_dimensions,
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}

    async def close(self):
        """Close the client connection."""
        await self._client.close()
