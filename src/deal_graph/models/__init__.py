"""
Data models for the Deal Graph pipeline.

Provides graph node models (Deal, DealVersion) aligned with the
eq-structured-graph-core schema authority, plus LLM structured output
models for extraction, deduplication, and merge synthesis.
"""

from .deal import Deal, DealVersion, MEDDICProfile, DealStage
from .extraction import (
    ExtractedDeal,
    DealExtractionResult,
    DealDeduplicationDecision,
    MergedDeal,
)

__all__ = [
    # Graph node models
    'Deal',
    'DealVersion',
    'MEDDICProfile',
    'DealStage',
    # LLM structured output models
    'ExtractedDeal',
    'DealExtractionResult',
    'DealDeduplicationDecision',
    'MergedDeal',
]
