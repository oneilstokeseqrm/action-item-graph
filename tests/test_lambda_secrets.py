"""Tests for the Lambda Secrets Manager fetch + cache logic.

Tests the secrets.py module in isolation using mocked boto3 calls.
Requires `uv sync --extra lambda` for boto3 + aws_lambda_powertools;
without the extra, this module skips via ``pytest.importorskip``.
"""

import pytest

pytest.importorskip("boto3")

from unittest.mock import MagicMock, patch  # noqa: E402

from action_item_graph.lambda_ingest.secrets import (  # noqa: E402
    clear_cache,
    get_dbos_system_database_url,
    get_secret,
)


@pytest.fixture(autouse=True)
def _clear_secrets_cache():
    """Clear the module-level cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestGetSecret:
    """Tests for get_secret()."""

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_fetches_secret_from_secrets_manager(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "my-secret-value"}

        result = get_secret("/my-service/my-key")

        mock_boto3.client.assert_called_once_with("secretsmanager")
        mock_client.get_secret_value.assert_called_once_with(SecretId="/my-service/my-key")
        assert result == "my-secret-value"

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_caches_secret_across_calls(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "cached-value"}

        result1 = get_secret("/my-service/cached-key")
        result2 = get_secret("/my-service/cached-key")

        assert result1 == "cached-value"
        assert result2 == "cached-value"
        assert mock_boto3.client.call_count == 1

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_raises_runtime_error_on_client_error(self, mock_boto3):
        from botocore.exceptions import ClientError

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Secret not found",
                }
            },
            operation_name="GetSecretValue",
        )

        with pytest.raises(RuntimeError, match="Failed to fetch secret"):
            get_secret("/missing/secret")

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_error_message_includes_secret_name_and_aws_error(self, mock_boto3):
        from botocore.exceptions import ClientError

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.side_effect = ClientError(
            error_response={
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "Not authorized",
                }
            },
            operation_name="GetSecretValue",
        )

        with pytest.raises(RuntimeError) as exc_info:
            get_secret("/my-service/restricted")

        error_msg = str(exc_info.value)
        assert "/my-service/restricted" in error_msg
        assert "AccessDeniedException" in error_msg
        assert "Not authorized" in error_msg


class TestGetDbosSystemDatabaseUrl:
    """Tests for get_dbos_system_database_url()."""

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_reads_secret_name_from_env_and_fetches(self, mock_boto3, monkeypatch):
        monkeypatch.setenv(
            "SECRET_NAME_DBOS_SYSTEM_DATABASE_URL",
            "/action-item-graph/dbos-system-database-url",
        )
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": "postgresql://user:pw@ep-test.us-east-1.aws.neon.tech/eq_aig_dbos_sys?sslmode=require"
        }

        result = get_dbos_system_database_url()

        assert result == (
            "postgresql://user:pw@ep-test.us-east-1.aws.neon.tech/eq_aig_dbos_sys?sslmode=require"
        )
        mock_client.get_secret_value.assert_called_once_with(
            SecretId="/action-item-graph/dbos-system-database-url"
        )

    def test_raises_if_env_var_not_set(self, monkeypatch):
        monkeypatch.delenv("SECRET_NAME_DBOS_SYSTEM_DATABASE_URL", raising=False)

        with pytest.raises(RuntimeError, match="SECRET_NAME_DBOS_SYSTEM_DATABASE_URL"):
            get_dbos_system_database_url()

    def test_error_message_is_actionable(self, monkeypatch):
        monkeypatch.delenv("SECRET_NAME_DBOS_SYSTEM_DATABASE_URL", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            get_dbos_system_database_url()

        error_msg = str(exc_info.value)
        assert "environment variable" in error_msg.lower()
        assert "SECRET_NAME_DBOS_SYSTEM_DATABASE_URL" in error_msg


class TestClearCache:
    """Tests for clear_cache()."""

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_clear_cache_forces_refetch(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "value-v1"}

        get_secret("/my-service/rotating-key")
        assert mock_boto3.client.call_count == 1

        clear_cache()

        mock_client.get_secret_value.return_value = {"SecretString": "value-v2"}

        result = get_secret("/my-service/rotating-key")
        assert result == "value-v2"
        assert mock_boto3.client.call_count == 2
