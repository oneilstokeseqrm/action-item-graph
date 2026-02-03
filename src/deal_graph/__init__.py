"""
Deal Graph Pipeline

MEDDIC qualification extraction pipeline that enriches skeleton Deal nodes
(created by eq-structured-graph-core) with qualification data from call
transcripts. Connects to the existing neo4j_structured database.
"""

__version__ = '0.1.0'

from .config import DealConfig, deal_config
from .errors import (
    DealPipelineError,
    DealExtractionError,
    DealMatchingError,
    DealMergeError,
)
from .models import (
    Deal,
    DealVersion,
    MEDDICProfile,
    DealStage,
    ExtractedDeal,
    DealExtractionResult,
    DealDeduplicationDecision,
    MergedDeal,
)

__all__ = [
    # Version
    '__version__',
    # Config
    'DealConfig',
    'deal_config',
    # Errors
    'DealPipelineError',
    'DealExtractionError',
    'DealMatchingError',
    'DealMergeError',
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
