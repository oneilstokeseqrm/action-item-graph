"""Tests for Railway API bearer token authentication."""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from action_item_graph.api.auth import verify_worker_token


class TestBearerAuth:
    @pytest.mark.asyncio
    async def test_valid_token_passes(self):
        mock_settings = MagicMock()
        mock_settings.WORKER_API_KEY = "test-secret-key"

        with patch("action_item_graph.api.auth.get_settings", return_value=mock_settings):
            await verify_worker_token(authorization="Bearer test-secret-key")

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self):
        mock_settings = MagicMock()
        mock_settings.WORKER_API_KEY = "test-secret-key"

        with patch("action_item_graph.api.auth.get_settings", return_value=mock_settings):
            with pytest.raises(HTTPException) as exc_info:
                await verify_worker_token(authorization="Bearer wrong-key")
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_bearer_prefix_raises_401(self):
        mock_settings = MagicMock()
        mock_settings.WORKER_API_KEY = "test-secret-key"

        with patch("action_item_graph.api.auth.get_settings", return_value=mock_settings):
            with pytest.raises(HTTPException) as exc_info:
                await verify_worker_token(authorization="test-secret-key")
            assert exc_info.value.status_code == 401
