"""Tests for Railway API configuration."""

import os
from unittest.mock import patch

from action_item_graph.api.config import Settings


class TestApiConfig:
    def test_config_loads_from_env(self):
        env = {
            "NEO4J_URI": "neo4j+s://test.databases.neo4j.io",
            "NEO4J_USERNAME": "neo4j",
            "NEO4J_PASSWORD": "secret",
            "OPENAI_API_KEY": "sk-test-key",
            "WORKER_API_KEY": "worker-secret-123",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.NEO4J_URI == "neo4j+s://test.databases.neo4j.io"
            assert settings.NEO4J_PASSWORD == "secret"
            assert settings.OPENAI_API_KEY == "sk-test-key"
            assert settings.WORKER_API_KEY == "worker-secret-123"

    def test_config_defaults(self):
        env = {
            "NEO4J_URI": "neo4j+s://test.databases.neo4j.io",
            "NEO4J_PASSWORD": "secret",
            "OPENAI_API_KEY": "sk-test",
            "WORKER_API_KEY": "worker-key",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings()
            assert settings.NEO4J_USERNAME == "neo4j"
            assert settings.NEO4J_DATABASE == "neo4j"
