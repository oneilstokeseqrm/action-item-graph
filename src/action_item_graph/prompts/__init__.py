"""
LLM prompts for the Action Item Graph pipeline.
"""

from .extract_action_items import (
    ExtractedActionItem,
    ExtractedTopic,
    ExtractionResult,
    DeduplicationDecision,
    build_extraction_prompt,
    build_deduplication_prompt,
    EXTRACTION_SYSTEM_PROMPT,
    DEDUPLICATION_SYSTEM_PROMPT,
)
from .merge_action_items import (
    MergedActionItem,
    build_merge_prompt,
)
from .topic_prompts import (
    TopicMatchDecision,
    TopicSummary,
    build_topic_match_prompt,
    build_topic_summary_create_prompt,
    build_topic_summary_update_prompt,
    TOPIC_MATCH_SYSTEM_PROMPT,
    TOPIC_SUMMARY_SYSTEM_PROMPT,
)

__all__ = [
    # Action Item Extraction
    'ExtractedActionItem',
    'ExtractedTopic',
    'ExtractionResult',
    'DeduplicationDecision',
    'build_extraction_prompt',
    'build_deduplication_prompt',
    'EXTRACTION_SYSTEM_PROMPT',
    'DEDUPLICATION_SYSTEM_PROMPT',
    # Action Item Merging
    'MergedActionItem',
    'build_merge_prompt',
    # Topic Resolution
    'TopicMatchDecision',
    'TopicSummary',
    'build_topic_match_prompt',
    'build_topic_summary_create_prompt',
    'build_topic_summary_update_prompt',
    'TOPIC_MATCH_SYSTEM_PROMPT',
    'TOPIC_SUMMARY_SYSTEM_PROMPT',
]
