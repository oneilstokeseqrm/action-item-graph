"""Fetch secrets from AWS Secrets Manager with cold-start caching.

Secrets are fetched once per Lambda cold start and cached in a module-level
dict for warm invocations. boto3 is available in the Lambda runtime —
no need to bundle it in the deployment zip.
"""

import os

import boto3
from botocore.exceptions import ClientError

# Module-level cache — survives across warm Lambda invocations
_cache: dict[str, str] = {}


def get_secret(secret_name: str) -> str:
    """Fetch a secret value from AWS Secrets Manager.

    Caches the result so subsequent calls during warm invocations
    skip the API call entirely.

    Raises:
        RuntimeError: If the secret cannot be fetched (network error,
            missing permission, or secret not found).
    """
    if secret_name in _cache:
        return _cache[secret_name]

    try:
        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise RuntimeError(
            f"Failed to fetch secret '{secret_name}' from Secrets Manager: "
            f"{e.response['Error']['Code']} — {e.response['Error']['Message']}"
        ) from e

    value = response["SecretString"]
    _cache[secret_name] = value
    return value


def get_dbos_system_database_url() -> str:
    """Get the DBOS_SYSTEM_DATABASE_URL from Secrets Manager.

    Reads the secret name from the SECRET_NAME_DBOS_SYSTEM_DATABASE_URL
    environment variable (set by the Pulumi forwarder stack).

    The stored value MUST be the direct (non-pooler) Neon connection URL —
    Neon's pgbouncer endpoint drops the advisory locks DBOS relies on for
    workflow coordination. See ``memory/reference_neon_dbos_state_dbs.md``.

    Raises:
        RuntimeError: If the env var is missing or the secret fetch fails.
    """
    secret_name = os.environ.get("SECRET_NAME_DBOS_SYSTEM_DATABASE_URL")
    if not secret_name:
        raise RuntimeError(
            "SECRET_NAME_DBOS_SYSTEM_DATABASE_URL environment variable is not set. "
            "The Lambda must be deployed with this env var pointing to the "
            "Secrets Manager secret name (e.g., /action-item-graph/dbos-system-database-url)."
        )
    return get_secret(secret_name)


def clear_cache() -> None:
    """Clear the secrets cache. Used in tests."""
    _cache.clear()
