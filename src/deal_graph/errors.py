"""
Custom exceptions for the Deal Graph pipeline.

Subclasses the base error hierarchy from action_item_graph.errors,
providing deal-specific error types for each pipeline stage.
"""

from action_item_graph.errors import PipelineError


class DealPipelineError(PipelineError):
    """Base exception for all deal pipeline errors."""

    pass


class DealExtractionError(DealPipelineError):
    """Error during MEDDIC extraction from transcript."""

    pass


class DealMatchingError(DealPipelineError):
    """Error during deal entity resolution (dual-embedding + LLM dedup)."""

    pass


class DealMergeError(DealPipelineError):
    """Error during deal merge synthesis or version snapshot."""

    pass
