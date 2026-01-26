"""
Pipeline components for action item extraction, matching, merging, and topic resolution.
"""

from .extractor import ActionItemExtractor, ExtractionOutput
from .matcher import (
    ActionItemMatcher,
    MatchCandidate,
    MatchResult,
    BatchMatchResult,
    match_batch,
)
from .merger import ActionItemMerger, MergeResult
from .pipeline import ActionItemPipeline, PipelineResult
from .topic_resolver import TopicResolver, TopicResolutionResult, TopicDecision, TopicCandidate
from .topic_executor import TopicExecutor, TopicExecutionResult

__all__ = [
    # Main Pipeline
    'ActionItemPipeline',
    'PipelineResult',
    # Extraction
    'ActionItemExtractor',
    'ExtractionOutput',
    # Matching
    'ActionItemMatcher',
    'MatchCandidate',
    'MatchResult',
    'BatchMatchResult',
    'match_batch',
    # Merging
    'ActionItemMerger',
    'MergeResult',
    # Topic Resolution (Phase 7)
    'TopicResolver',
    'TopicResolutionResult',
    'TopicDecision',
    'TopicCandidate',
    'TopicExecutor',
    'TopicExecutionResult',
]
