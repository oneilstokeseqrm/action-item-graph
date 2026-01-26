"""
Custom exceptions and error handling for the Action Item Graph pipeline.

Provides:
- Typed exception hierarchy for different failure modes
- Error context preservation for debugging
- Partial success handling for batch operations
"""

from dataclasses import dataclass, field
from typing import Any


class ActionItemGraphError(Exception):
    """Base exception for all action item graph errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            return f"{self.message} | context={self.context}"
        return self.message


# =============================================================================
# Client Errors
# =============================================================================


class ClientError(ActionItemGraphError):
    """Base class for client-related errors."""

    pass


class OpenAIError(ClientError):
    """Error from OpenAI API calls."""

    pass


class OpenAIRateLimitError(OpenAIError):
    """Rate limit exceeded on OpenAI API."""

    pass


class OpenAIModelError(OpenAIError):
    """Model refused request or returned invalid response."""

    pass


class Neo4jError(ClientError):
    """Error from Neo4j database operations."""

    pass


class Neo4jConnectionError(Neo4jError):
    """Failed to connect to Neo4j database."""

    pass


class Neo4jQueryError(Neo4jError):
    """Error executing Neo4j query."""

    pass


class Neo4jConstraintError(Neo4jError):
    """Constraint violation in Neo4j (e.g., duplicate unique key)."""

    pass


# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(ActionItemGraphError):
    """Base class for pipeline-related errors."""

    pass


class ValidationError(PipelineError):
    """Input validation failed."""

    pass


class ExtractionError(PipelineError):
    """Error during action item extraction."""

    pass


class MatchingError(PipelineError):
    """Error during action item matching."""

    pass


class MergeError(PipelineError):
    """Error during action item merge operation."""

    pass


class RepositoryError(PipelineError):
    """Error during graph repository operations."""

    pass


# =============================================================================
# Partial Success Handling
# =============================================================================


@dataclass
class ItemResult:
    """Result for a single item in a batch operation."""

    item_id: str | None
    success: bool
    error: ActionItemGraphError | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PartialSuccessResult:
    """
    Result of a batch operation that may partially succeed.

    Allows processing to continue even when some items fail,
    while preserving error context for debugging.
    """

    succeeded: list[ItemResult] = field(default_factory=list)
    failed: list[ItemResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.succeeded)

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    @property
    def total_count(self) -> int:
        return self.success_count + self.failure_count

    @property
    def all_succeeded(self) -> bool:
        return self.failure_count == 0

    @property
    def all_failed(self) -> bool:
        return self.success_count == 0

    @property
    def partial_success(self) -> bool:
        return self.success_count > 0 and self.failure_count > 0

    def add_success(
        self,
        item_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Record a successful item."""
        self.succeeded.append(
            ItemResult(item_id=item_id, success=True, data=data or {})
        )

    def add_failure(
        self,
        error: ActionItemGraphError,
        item_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Record a failed item."""
        self.failed.append(
            ItemResult(item_id=item_id, success=False, error=error, data=data or {})
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'total_count': self.total_count,
            'all_succeeded': self.all_succeeded,
            'succeeded_ids': [r.item_id for r in self.succeeded if r.item_id],
            'failed_ids': [r.item_id for r in self.failed if r.item_id],
            'errors': [
                {'item_id': r.item_id, 'error': str(r.error)}
                for r in self.failed
                if r.error
            ],
        }


# =============================================================================
# Error Handling Utilities
# =============================================================================


def wrap_openai_error(exc: Exception, context: dict[str, Any] | None = None) -> OpenAIError:
    """
    Wrap an OpenAI exception in our typed error hierarchy.

    Args:
        exc: The original exception
        context: Additional context for debugging

    Returns:
        Typed OpenAIError subclass
    """
    error_str = str(exc).lower()
    ctx = context or {}
    ctx['original_error'] = str(exc)
    ctx['error_type'] = type(exc).__name__

    if 'rate limit' in error_str or 'rate_limit' in error_str:
        return OpenAIRateLimitError(
            f"OpenAI rate limit exceeded: {exc}",
            context=ctx,
        )
    elif 'content policy' in error_str or 'refused' in error_str:
        return OpenAIModelError(
            f"OpenAI model refused request: {exc}",
            context=ctx,
        )
    else:
        return OpenAIError(
            f"OpenAI API error: {exc}",
            context=ctx,
        )


def wrap_neo4j_error(exc: Exception, context: dict[str, Any] | None = None) -> Neo4jError:
    """
    Wrap a Neo4j exception in our typed error hierarchy.

    Args:
        exc: The original exception
        context: Additional context for debugging

    Returns:
        Typed Neo4jError subclass
    """
    error_str = str(exc).lower()
    ctx = context or {}
    ctx['original_error'] = str(exc)
    ctx['error_type'] = type(exc).__name__

    if 'connection' in error_str or 'connect' in error_str:
        return Neo4jConnectionError(
            f"Neo4j connection failed: {exc}",
            context=ctx,
        )
    elif 'constraint' in error_str or 'unique' in error_str:
        return Neo4jConstraintError(
            f"Neo4j constraint violation: {exc}",
            context=ctx,
        )
    else:
        return Neo4jQueryError(
            f"Neo4j query error: {exc}",
            context=ctx,
        )
