"""JSON serialization helpers for the deal workflow's step boundaries.

Mirrors the discipline in
``src/action_item_graph/workflows/_serialization.py`` (per
[[pattern-dbos-workflow-parity-rules]] Rule 3). Every type that crosses
a @DBOS.step boundary in the deal workflow has a to_dict/from_dict
pair; UUID + datetime fields are serialized via ``mode='json'``;
plain dataclasses round-trip via ``dataclasses.asdict()``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from deal_graph.models.extraction import (
    DealDeduplicationDecision,
    DealExtractionResult,
    ExtractedDeal,
)
from deal_graph.pipeline.matcher import DealMatchCandidate, DealMatchResult
from deal_graph.pipeline.merger import DealMergeResult


# ---------------------------------------------------------------------------
# ExtractedDeal (Pydantic)
# ---------------------------------------------------------------------------


def extracted_deal_to_dict(d: ExtractedDeal) -> dict[str, Any]:
    return d.model_dump(mode='json')


def extracted_deal_from_dict(d: dict[str, Any]) -> ExtractedDeal:
    return ExtractedDeal.model_validate(d)


# ---------------------------------------------------------------------------
# DealExtractionResult (Pydantic, list[ExtractedDeal] + extraction_notes)
# ---------------------------------------------------------------------------


def deal_extraction_result_to_dict(
    r: DealExtractionResult, embeddings: list[list[float]]
) -> dict[str, Any]:
    """Bundle the DealExtractionResult with the parallel embeddings list.

    The legacy ``DealExtractor.extract_from_envelope`` returns ``(result,
    embeddings)``. We carry both as a single JSON-safe envelope so the
    downstream D7 step has aligned per-deal embedding access without
    re-running extraction.
    """
    return {
        'result': r.model_dump(mode='json'),
        'embeddings': embeddings,
    }


def deal_extraction_result_from_dict(
    d: dict[str, Any],
) -> tuple[DealExtractionResult, list[list[float]]]:
    return (
        DealExtractionResult.model_validate(d['result']),
        d.get('embeddings', []),
    )


# ---------------------------------------------------------------------------
# DealMatchResult (dataclass with Pydantic + dataclass fields)
# ---------------------------------------------------------------------------


def deal_match_candidate_to_dict(c: DealMatchCandidate) -> dict[str, Any]:
    return dataclasses.asdict(c)


def deal_match_candidate_from_dict(d: dict[str, Any]) -> DealMatchCandidate:
    return DealMatchCandidate(**d)


def deal_match_result_to_dict(m: DealMatchResult) -> dict[str, Any]:
    return {
        'extracted_deal': m.extracted_deal.model_dump(mode='json'),
        'embedding': m.embedding,
        'match_type': m.match_type,
        'matched_deal': (
            deal_match_candidate_to_dict(m.matched_deal)
            if m.matched_deal is not None
            else None
        ),
        'decision': (
            m.decision.model_dump(mode='json') if m.decision is not None else None
        ),
        'candidates_evaluated': m.candidates_evaluated,
        'all_candidates': [
            deal_match_candidate_to_dict(c) for c in m.all_candidates
        ],
    }


def deal_match_result_from_dict(d: dict[str, Any]) -> DealMatchResult:
    return DealMatchResult(
        extracted_deal=ExtractedDeal.model_validate(d['extracted_deal']),
        embedding=d['embedding'],
        match_type=d['match_type'],
        matched_deal=(
            deal_match_candidate_from_dict(d['matched_deal'])
            if d.get('matched_deal') is not None
            else None
        ),
        decision=(
            DealDeduplicationDecision.model_validate(d['decision'])
            if d.get('decision') is not None
            else None
        ),
        candidates_evaluated=d['candidates_evaluated'],
        all_candidates=[
            deal_match_candidate_from_dict(c)
            for c in d.get('all_candidates', [])
        ],
    )


# ---------------------------------------------------------------------------
# DealMergeResult (dataclass, scalar/dict-only fields)
# ---------------------------------------------------------------------------


def deal_merge_result_to_dict(m: DealMergeResult) -> dict[str, Any]:
    return dataclasses.asdict(m)


def deal_merge_result_from_dict(d: dict[str, Any]) -> DealMergeResult:
    return DealMergeResult(**d)
