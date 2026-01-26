"""
Structured logging configuration for the Action Item Graph pipeline.

Uses structlog for structured, context-aware logging with:
- JSON output for production
- Pretty console output for development
- Automatic timing context
- Trace ID propagation from EnvelopeV1
"""

import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

import structlog
from structlog.types import Processor

from .config import config

# Context variables for request-scoped data
_trace_id: ContextVar[str | None] = ContextVar('trace_id', default=None)
_tenant_id: ContextVar[str | None] = ContextVar('tenant_id', default=None)
_account_id: ContextVar[str | None] = ContextVar('account_id', default=None)


def get_trace_id() -> str | None:
    """Get the current trace ID from context."""
    return _trace_id.get()


def get_tenant_id() -> str | None:
    """Get the current tenant ID from context."""
    return _tenant_id.get()


def get_account_id() -> str | None:
    """Get the current account ID from context."""
    return _account_id.get()


def add_context_info(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Processor that adds context variables to log entries."""
    trace_id = get_trace_id()
    tenant_id = get_tenant_id()
    account_id = get_account_id()

    if trace_id:
        event_dict['trace_id'] = trace_id
    if tenant_id:
        event_dict['tenant_id'] = tenant_id
    if account_id:
        event_dict['account_id'] = account_id

    return event_dict


def configure_logging(
    json_output: bool = False,
    log_level: str | None = None,
) -> None:
    """
    Configure structlog for the application.

    Args:
        json_output: If True, output JSON logs (for production).
                    If False, output pretty console logs (for development).
        log_level: Override log level (defaults to config.LOG_LEVEL)
    """
    level = log_level or config.LOG_LEVEL
    level_num = getattr(logging, level.upper(), logging.INFO)

    # Standard library logging config
    logging.basicConfig(
        format='%(message)s',
        stream=sys.stdout,
        level=level_num,
    )

    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_context_info,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        # Production: JSON output
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: Pretty console output
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


@contextmanager
def logging_context(
    trace_id: str | None = None,
    tenant_id: str | None = None,
    account_id: str | None = None,
) -> Generator[None, None, None]:
    """
    Context manager for setting logging context variables.

    Usage:
        with logging_context(trace_id="abc123", tenant_id="tenant_1"):
            logger.info("Processing request")  # Includes trace_id and tenant_id
    """
    old_trace = _trace_id.get()
    old_tenant = _tenant_id.get()
    old_account = _account_id.get()

    try:
        if trace_id is not None:
            _trace_id.set(trace_id)
        if tenant_id is not None:
            _tenant_id.set(tenant_id)
        if account_id is not None:
            _account_id.set(account_id)
        yield
    finally:
        _trace_id.set(old_trace)
        _tenant_id.set(old_tenant)
        _account_id.set(old_account)


class PipelineTimer:
    """
    Timer for tracking pipeline stage durations.

    Usage:
        timer = PipelineTimer()
        with timer.stage("extraction"):
            # do extraction
        with timer.stage("matching"):
            # do matching
        print(timer.summary())
    """

    def __init__(self):
        self.stages: dict[str, float] = {}
        self.start_time: float = time.perf_counter()
        self._current_stage: str | None = None
        self._stage_start: float | None = None

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Time a pipeline stage."""
        self._current_stage = name
        self._stage_start = time.perf_counter()
        try:
            yield
        finally:
            if self._stage_start is not None:
                elapsed = time.perf_counter() - self._stage_start
                self.stages[name] = elapsed * 1000  # Convert to ms
            self._current_stage = None
            self._stage_start = None

    def record(self, name: str, duration_ms: float) -> None:
        """Manually record a stage duration."""
        self.stages[name] = duration_ms

    @property
    def total_ms(self) -> float:
        """Total elapsed time since timer creation in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000

    def summary(self) -> dict[str, Any]:
        """Get timing summary as a dictionary."""
        return {
            'total_ms': round(self.total_ms, 2),
            'stages': {k: round(v, 2) for k, v in self.stages.items()},
        }


# Initialize logging on module import (development mode by default)
# Production deployments should call configure_logging(json_output=True)
configure_logging(json_output=False)
