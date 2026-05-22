"""JSON serialization helpers for DBOS step boundaries.

DBOS checkpoints step inputs/outputs as JSON in
``dbos.operation_outputs.output``. Pydantic models serialize cleanly via
``.model_dump(mode='json')``. Plain dataclasses + tuples need explicit
conversion. This module centralizes that boundary so step bodies stay
readable.

Convention:
  - ``<type>_to_dict(obj) -> dict`` — JSON-safe representation
  - ``<type>_from_dict(d) -> obj`` — inverse, validates Pydantic fields
  - Helpers operate on plain dicts/lists; consumers may further pass
    through ``json.dumps`` if needed for diagnostics.

Pydantic models are always serialized with ``mode='json'`` so UUID and
datetime fields become strings. Failing to do that breaks DBOS
checkpoint deserialization (UUID would become a json.JSONEncoder-
incompatible object).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from action_item_graph.models.action_item import ActionItem
from action_item_graph.models.entities import Interaction
from action_item_graph.pipeline.extractor import ExtractionOutput
from action_item_graph.pipeline.matcher import MatchCandidate, MatchResult
from action_item_graph.pipeline.merger import MergeResult
from action_item_graph.pipeline.topic_resolver import (
    TopicCandidate,
    TopicDecision,
    TopicResolutionResult,
)
from action_item_graph.pipeline.topic_executor import TopicExecutionResult
from action_item_graph.prompts.extract_action_items import (
    DeduplicationDecision,
    ExtractedActionItem,
    ExtractedTopic,
)


# ---------------------------------------------------------------------------
# ExtractionOutput
# ---------------------------------------------------------------------------


def extraction_to_dict(extraction: ExtractionOutput) -> dict[str, Any]:
    return {
        'interaction': extraction.interaction.model_dump(mode='json'),
        'action_items': [
            ai.model_dump(mode='json') for ai in extraction.action_items
        ],
        'raw_extractions': [
            r.model_dump(mode='json') for r in extraction.raw_extractions
        ],
        'extraction_notes': extraction.extraction_notes,
    }


def extraction_from_dict(d: dict[str, Any]) -> ExtractionOutput:
    return ExtractionOutput(
        interaction=Interaction.model_validate(d['interaction']),
        action_items=[ActionItem.model_validate(ai) for ai in d['action_items']],
        raw_extractions=[
            ExtractedActionItem.model_validate(r) for r in d['raw_extractions']
        ],
        extraction_notes=d.get('extraction_notes'),
    )


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


def match_candidate_to_dict(c: MatchCandidate) -> dict[str, Any]:
    return dataclasses.asdict(c)


def match_candidate_from_dict(d: dict[str, Any]) -> MatchCandidate:
    return MatchCandidate(**d)


def match_result_to_dict(m: MatchResult) -> dict[str, Any]:
    """Serialize a MatchResult. Tuples in ``decisions`` and ``best_match``
    are flattened into dicts so JSON survives the round-trip."""
    return {
        'extracted_item': m.extracted_item.model_dump(mode='json'),
        'embedding': m.embedding,
        'candidates': [match_candidate_to_dict(c) for c in m.candidates],
        'decisions': [
            {
                'candidate': match_candidate_to_dict(c),
                'decision': d.model_dump(mode='json'),
            }
            for c, d in m.decisions
        ],
        'best_match': (
            {
                'candidate': match_candidate_to_dict(m.best_match[0]),
                'decision': m.best_match[1].model_dump(mode='json'),
            }
            if m.best_match is not None
            else None
        ),
    }


def match_result_from_dict(d: dict[str, Any]) -> MatchResult:
    return MatchResult(
        extracted_item=ExtractedActionItem.model_validate(d['extracted_item']),
        embedding=d['embedding'],
        candidates=[match_candidate_from_dict(c) for c in d['candidates']],
        decisions=[
            (
                match_candidate_from_dict(entry['candidate']),
                DeduplicationDecision.model_validate(entry['decision']),
            )
            for entry in d['decisions']
        ],
        best_match=(
            (
                match_candidate_from_dict(d['best_match']['candidate']),
                DeduplicationDecision.model_validate(d['best_match']['decision']),
            )
            if d.get('best_match') is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# MergeResult (dataclass, dict-only fields, asdict round-trips cleanly)
# ---------------------------------------------------------------------------


def merge_result_to_dict(m: MergeResult) -> dict[str, Any]:
    return dataclasses.asdict(m)


def merge_result_from_dict(d: dict[str, Any]) -> MergeResult:
    return MergeResult(**d)


# ---------------------------------------------------------------------------
# TopicResolutionResult
# ---------------------------------------------------------------------------


def topic_candidate_to_dict(c: TopicCandidate) -> dict[str, Any]:
    return dataclasses.asdict(c)


def topic_candidate_from_dict(d: dict[str, Any]) -> TopicCandidate:
    return TopicCandidate(**d)


def topic_resolution_to_dict(t: TopicResolutionResult) -> dict[str, Any]:
    return {
        'action_item_id': t.action_item_id,
        'action_item_summary': t.action_item_summary,
        'extracted_topic': t.extracted_topic.model_dump(mode='json'),
        'decision': t.decision.value,
        'topic_id': t.topic_id,
        'confidence': t.confidence,
        'candidates': [topic_candidate_to_dict(c) for c in t.candidates],
        'embedding': t.embedding,
        'best_candidate': (
            topic_candidate_to_dict(t.best_candidate)
            if t.best_candidate is not None
            else None
        ),
        # ``llm_decision`` field type varies; serialize via best-effort dict cast.
        'llm_decision': (
            t.llm_decision.model_dump(mode='json')
            if t.llm_decision is not None
            else None
        ),
    }


def topic_resolution_from_dict(d: dict[str, Any]) -> TopicResolutionResult:
    # The llm_decision field is set lazily by the resolver; we re-hydrate it
    # only as a generic dict because the precise type lives inside
    # topic_resolver.py and isn't part of the step's contract.
    result = TopicResolutionResult(
        action_item_id=d['action_item_id'],
        action_item_summary=d['action_item_summary'],
        extracted_topic=ExtractedTopic.model_validate(d['extracted_topic']),
        decision=TopicDecision(d['decision']),
        topic_id=d.get('topic_id'),
        confidence=d['confidence'],
        candidates=[topic_candidate_from_dict(c) for c in d['candidates']],
        embedding=d['embedding'],
    )
    if d.get('best_candidate') is not None:
        result.best_candidate = topic_candidate_from_dict(d['best_candidate'])
    if d.get('llm_decision') is not None:
        # The executor only cares whether decision == LINK_EXISTING; the raw
        # llm_decision payload is informational. Stuff it back as a dict.
        result.llm_decision = d['llm_decision']  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# TopicExecutionResult (dataclass)
# ---------------------------------------------------------------------------


def topic_execution_to_dict(t: TopicExecutionResult) -> dict[str, Any]:
    return dataclasses.asdict(t)


def topic_execution_from_dict(d: dict[str, Any]) -> TopicExecutionResult:
    return TopicExecutionResult(**d)
