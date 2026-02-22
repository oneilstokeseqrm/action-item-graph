"""HTTP client for forwarding envelopes to the Railway API service."""

import random
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class SubmitResult:
    """Result of submitting an envelope to Railway."""

    success: bool
    status_code: int | None = None
    error: str | None = None


def submit_to_railway(config: Any, envelope_json: dict[str, Any]) -> SubmitResult:
    """
    POST the envelope JSON to the Railway /process endpoint.

    Retry strategy:
    - 2xx: return success immediately
    - 4xx: return failure immediately (persistent error, no retry)
    - 5xx / network error: retry with exponential backoff + jitter
    """
    url = f"{config.API_BASE_URL.rstrip('/')}/process"
    headers = {
        "Authorization": f"Bearer {config.WORKER_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error: str | None = None
    last_status: int | None = None

    for attempt in range(1 + config.MAX_RETRIES):
        try:
            with httpx.Client(timeout=config.HTTP_TIMEOUT_SECONDS) as client:
                response = client.post(url, json=envelope_json, headers=headers)
                response.raise_for_status()
                return SubmitResult(success=True, status_code=response.status_code)

        except httpx.HTTPStatusError as e:
            last_status = e.response.status_code
            last_error = f"HTTP {last_status}"

            # 4xx — persistent error, no retry
            if 400 <= last_status < 500:
                return SubmitResult(
                    success=False, status_code=last_status, error=last_error
                )

            # 5xx — retry
            if attempt < config.MAX_RETRIES:
                _backoff_sleep(attempt)

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = f"{type(e).__name__}: {e}"
            last_status = None
            if attempt < config.MAX_RETRIES:
                _backoff_sleep(attempt)

    return SubmitResult(success=False, status_code=last_status, error=last_error)


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff with jitter."""
    base = 2**attempt
    jitter = random.uniform(0, base * 0.5)
    time.sleep(base + jitter)
