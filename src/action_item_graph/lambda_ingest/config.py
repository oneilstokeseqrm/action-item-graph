"""Configuration for the Lambda forwarder."""

from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class LambdaConfig(BaseSettings):
    """Lambda environment variables.

    WORKER_API_KEY is populated from Secrets Manager at cold start,
    not from environment variables. See handler.py._get_config().
    """

    API_BASE_URL: str
    WORKER_API_KEY: str = ""  # Set from Secrets Manager at cold start
    HTTP_TIMEOUT_SECONDS: int = Field(default=100, ge=5, le=120)
    MAX_RETRIES: int = Field(default=2, ge=0, le=5)


@lru_cache
def get_lambda_config() -> LambdaConfig:
    """Cached config singleton."""
    return LambdaConfig()
