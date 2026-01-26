"""
Action Item Graph Pipeline

A temporal knowledge graph pipeline for extracting and managing action items
from call transcripts with Neo4j storage and OpenAI-powered extraction.
"""

__version__ = '0.1.0'

# Re-export key classes for convenience
from .pipeline import (
    ActionItemPipeline,
    PipelineResult,
    ActionItemExtractor,
    ActionItemMatcher,
    ActionItemMerger,
    ExtractionOutput,
    MatchResult,
    MergeResult,
)
from .repository import ActionItemRepository
from .logging import (
    configure_logging,
    get_logger,
    logging_context,
    PipelineTimer,
)
from .errors import (
    ActionItemGraphError,
    PipelineError,
    ValidationError,
    ExtractionError,
    MatchingError,
    MergeError,
    OpenAIError,
    Neo4jError,
    PartialSuccessResult,
)

__all__ = [
    # Version
    '__version__',
    # Main Pipeline
    'ActionItemPipeline',
    'PipelineResult',
    # Components
    'ActionItemExtractor',
    'ActionItemMatcher',
    'ActionItemMerger',
    'ExtractionOutput',
    'MatchResult',
    'MergeResult',
    # Repository
    'ActionItemRepository',
    # Logging
    'configure_logging',
    'get_logger',
    'logging_context',
    'PipelineTimer',
    # Errors
    'ActionItemGraphError',
    'PipelineError',
    'ValidationError',
    'ExtractionError',
    'MatchingError',
    'MergeError',
    'OpenAIError',
    'Neo4jError',
    'PartialSuccessResult',
]
