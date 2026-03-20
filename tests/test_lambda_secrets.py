"""Tests for the Lambda Secrets Manager fetch + cache logic.

Tests the secrets.py module in isolation using mocked boto3 calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from action_item_graph.lambda_ingest.secrets import (
    clear_cache,
    get_secret,
    get_worker_api_key,
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
        mock_client.get_secret_value.return_value = {
            "SecretString": "my-secret-value"
        }

        result = get_secret("/my-service/my-key")

        mock_boto3.client.assert_called_once_with("secretsmanager")
        mock_client.get_secret_value.assert_called_once_with(
            SecretId="/my-service/my-key"
        )
        assert result == "my-secret-value"

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_caches_secret_across_calls(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": "cached-value"
        }

        # First call fetches from Secrets Manager
        result1 = get_secret("/my-service/cached-key")
        # Second call should use cache — no additional API call
        result2 = get_secret("/my-service/cached-key")

        assert result1 == "cached-value"
        assert result2 == "cached-value"
        # boto3.client called only once (first call)
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


class TestGetWorkerApiKey:
    """Tests for get_worker_api_key()."""

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_reads_secret_name_from_env_and_fetches(self, mock_boto3, monkeypatch):
        monkeypatch.setenv(
            "SECRET_NAME_WORKER_API_KEY", "/action-item-graph/worker-api-key"
        )
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": "the-real-api-key"
        }

        result = get_worker_api_key()

        assert result == "the-real-api-key"
        mock_client.get_secret_value.assert_called_once_with(
            SecretId="/action-item-graph/worker-api-key"
        )

    def test_raises_if_env_var_not_set(self, monkeypatch):
        monkeypatch.delenv("SECRET_NAME_WORKER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="SECRET_NAME_WORKER_API_KEY"):
            get_worker_api_key()

    def test_error_message_is_actionable(self, monkeypatch):
        monkeypatch.delenv("SECRET_NAME_WORKER_API_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            get_worker_api_key()

        error_msg = str(exc_info.value)
        assert "environment variable" in error_msg.lower()
        assert "SECRET_NAME_WORKER_API_KEY" in error_msg


class TestClearCache:
    """Tests for clear_cache()."""

    @patch("action_item_graph.lambda_ingest.secrets.boto3")
    def test_clear_cache_forces_refetch(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {
            "SecretString": "value-v1"
        }

        # Fetch and cache
        get_secret("/my-service/rotating-key")
        assert mock_boto3.client.call_count == 1

        # Clear cache
        clear_cache()

        # Update the mock to return a new value
        mock_client.get_secret_value.return_value = {
            "SecretString": "value-v2"
        }

        # Should fetch again (cache was cleared)
        result = get_secret("/my-service/rotating-key")
        assert result == "value-v2"
        assert mock_boto3.client.call_count == 2
