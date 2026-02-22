"""Tests for Lambda forwarder configuration."""

import os
from unittest.mock import patch

from action_item_graph.lambda_ingest.config import LambdaConfig


class TestLambdaConfig:
    def test_config_loads_from_env(self):
        env = {
            "API_BASE_URL": "https://action-item-graph.up.railway.app",
            "WORKER_API_KEY": "lambda-secret",
        }
        with patch.dict(os.environ, env, clear=False):
            config = LambdaConfig()
            assert config.API_BASE_URL == "https://action-item-graph.up.railway.app"
            assert config.WORKER_API_KEY == "lambda-secret"
            assert config.HTTP_TIMEOUT_SECONDS == 100
            assert config.MAX_RETRIES == 2

    def test_config_custom_timeout(self):
        env = {
            "API_BASE_URL": "https://test.railway.app",
            "WORKER_API_KEY": "key",
            "HTTP_TIMEOUT_SECONDS": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            config = LambdaConfig()
            assert config.HTTP_TIMEOUT_SECONDS == 5
