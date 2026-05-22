"""Tests for Lambda dispatcher configuration."""

import os
from unittest.mock import patch

from action_item_graph.lambda_ingest.config import LambdaConfig


class TestLambdaConfig:
    def test_config_loads_from_env(self):
        env = {
            "DBOS_SYSTEM_DATABASE_URL": (
                "postgresql://user:pw@ep-test.us-east-1.aws.neon.tech/eq_aig_dbos_sys?sslmode=require"
            ),
        }
        with patch.dict(os.environ, env, clear=False):
            config = LambdaConfig()
            assert config.DBOS_SYSTEM_DATABASE_URL == (
                "postgresql://user:pw@ep-test.us-east-1.aws.neon.tech/eq_aig_dbos_sys?sslmode=require"
            )

    def test_config_defaults_empty_url_for_cold_start_population(self):
        """DBOS_SYSTEM_DATABASE_URL defaults to empty string so the handler
        can populate it from Secrets Manager at cold start (see
        handler._get_config). The handler then calls DBOSClient with the
        populated value; if the secret fetch fails, that error surfaces
        from get_dbos_system_database_url(), not from config validation.
        """
        with patch.dict(os.environ, {}, clear=True):
            config = LambdaConfig()
            assert config.DBOS_SYSTEM_DATABASE_URL == ""
