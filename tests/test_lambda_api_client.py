"""Tests for Lambda → Railway API client."""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import httpx

from action_item_graph.lambda_ingest.api_client import submit_to_railway, SubmitResult


@dataclass
class MockConfig:
    API_BASE_URL: str = "https://test.railway.app"
    WORKER_API_KEY: str = "test-key"
    HTTP_TIMEOUT_SECONDS: int = 10
    MAX_RETRIES: int = 2


class TestSubmitToRailway:
    @patch("action_item_graph.lambda_ingest.api_client.httpx")
    def test_success_returns_result(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"overall_success": True}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        result = submit_to_railway(MockConfig(), {"test": "envelope"})
        assert result.success is True
        assert result.status_code == 200

    @patch("action_item_graph.lambda_ingest.api_client.httpx")
    def test_4xx_no_retry(self, mock_httpx):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "422", request=MagicMock(), response=mock_response
        )

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError

        result = submit_to_railway(MockConfig(), {"bad": "data"})
        assert result.success is False
        assert result.status_code == 422
        # Should only call post once (no retry on 4xx)
        assert mock_client.post.call_count == 1

    @patch("action_item_graph.lambda_ingest.api_client.time")
    @patch("action_item_graph.lambda_ingest.api_client.httpx")
    def test_5xx_retries(self, mock_httpx, mock_time):
        mock_time.sleep = MagicMock()

        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        mock_response_500.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response_500
        )

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {"overall_success": True}
        mock_response_200.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = [mock_response_500, mock_response_200]
        mock_httpx.Client.return_value = mock_client
        mock_httpx.HTTPStatusError = httpx.HTTPStatusError
        mock_httpx.TimeoutException = httpx.TimeoutException
        mock_httpx.ConnectError = httpx.ConnectError

        result = submit_to_railway(MockConfig(), {"test": "envelope"})
        assert result.success is True
        assert mock_client.post.call_count == 2
