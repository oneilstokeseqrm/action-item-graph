"""Configuration for the Railway FastAPI service."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Railway service settings loaded from environment variables."""

    # Neo4j
    NEO4J_URI: str
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"

    # OpenAI
    OPENAI_API_KEY: str

    # Auth
    WORKER_API_KEY: str


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
