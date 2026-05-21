"""Configuration for the Lambda dispatcher.

DBOS_SYSTEM_DATABASE_URL is the direct (non-pooler) Neon URL for the
``eq_aig_dbos_sys`` database. It is populated from AWS Secrets Manager at
cold start, not from environment variables. See handler.py._get_config().

If the value cannot be resolved (Secrets Manager error, secret missing,
or env var unset in local/test runs), Pydantic's BaseSettings raises a
ValidationError on instantiation — fail-fast at cold start instead of
silently falling back to an unusable client. This mirrors LTF's
``services/dbos_runtime.py:48-57`` fail-loud pattern.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class LambdaConfig(BaseSettings):
    """Lambda environment variables.

    DBOS_SYSTEM_DATABASE_URL is REQUIRED. It MUST be the direct (non-pooler)
    Neon connection — Neon's pgbouncer endpoint breaks DBOS advisory locks.
    See ``memory/reference_neon_dbos_state_dbs.md``.
    """

    DBOS_SYSTEM_DATABASE_URL: str = ""  # Set from Secrets Manager at cold start


@lru_cache
def get_lambda_config() -> LambdaConfig:
    """Cached config singleton."""
    return LambdaConfig()
