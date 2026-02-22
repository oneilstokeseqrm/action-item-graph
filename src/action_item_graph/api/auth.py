"""Bearer token authentication for the Railway API."""

from fastapi import Header, HTTPException

from .config import get_settings


async def verify_worker_token(authorization: str = Header(...)) -> None:
    """Validate the bearer token from the Lambda forwarder."""
    expected = f"Bearer {get_settings().WORKER_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
