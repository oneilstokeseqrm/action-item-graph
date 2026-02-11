"""
Configuration management for the Action Item Graph pipeline.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / '.env'
if _env_file.exists():
    load_dotenv(_env_file)


class Config:
    """Configuration settings loaded from environment."""

    # OpenAI
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    OPENAI_CHAT_MODEL: str = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4.1-mini')
    OPENAI_EMBEDDING_MODEL: str = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
    OPENAI_EMBEDDING_DIMENSIONS: int = int(os.getenv('OPENAI_EMBEDDING_DIMENSIONS', '1536'))

    # Neo4j — primary reads NEO4J_ vars, falls back to DEAL_ vars for shared DB compat
    NEO4J_URI: str = os.getenv('NEO4J_URI', '') or os.getenv('DEAL_NEO4J_URI', '')
    NEO4J_USERNAME: str = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD: str = os.getenv('NEO4J_PASSWORD', '') or os.getenv('DEAL_NEO4J_PASSWORD', '')
    NEO4J_DATABASE: str = os.getenv('NEO4J_DATABASE', 'neo4j')

    # Pipeline
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.85'))
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate that required configuration is present.

        Returns:
            List of missing required configuration keys
        """
        missing = []
        if not cls.OPENAI_API_KEY:
            missing.append('OPENAI_API_KEY')
        if not cls.NEO4J_URI:
            missing.append('NEO4J_URI')
        if not cls.NEO4J_PASSWORD:
            missing.append('NEO4J_PASSWORD')
        return missing


# Singleton config instance
config = Config()
