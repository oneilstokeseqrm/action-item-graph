"""
Configuration management for the Deal Graph pipeline.

Loads DEAL_NEO4J_* settings from environment variables. These connect to the
existing neo4j_structured database managed by eq-structured-graph-core.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root (idempotent if already loaded by action_item_graph)
_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / '.env'
if _env_file.exists():
    load_dotenv(_env_file)


class DealConfig:
    """Configuration for the Deal pipeline, loaded from environment."""

    # Neo4j (connects to existing neo4j_structured instance)
    NEO4J_URI: str = os.getenv('DEAL_NEO4J_URI', '')
    NEO4J_USERNAME: str = os.getenv('DEAL_NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD: str = os.getenv('DEAL_NEO4J_PASSWORD', '')
    NEO4J_DATABASE: str = os.getenv('DEAL_NEO4J_DATABASE', 'neo4j')

    # Pipeline thresholds
    SIMILARITY_THRESHOLD: float = float(os.getenv('DEAL_SIMILARITY_THRESHOLD', '0.70'))
    AUTO_MATCH_THRESHOLD: float = float(os.getenv('DEAL_AUTO_MATCH_THRESHOLD', '0.90'))

    # Logging
    LOG_LEVEL: str = os.getenv('DEAL_LOG_LEVEL', 'INFO')

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate that required configuration is present.

        Returns:
            List of missing required configuration keys.
        """
        missing = []
        if not cls.NEO4J_URI:
            missing.append('DEAL_NEO4J_URI')
        if not cls.NEO4J_PASSWORD:
            missing.append('DEAL_NEO4J_PASSWORD')
        return missing


# Singleton config instance
deal_config = DealConfig()
