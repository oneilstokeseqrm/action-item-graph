"""
Unit tests for Phase 4: Scoring & Surfacing.

Tests priority score computation, ActionItem scoring fields,
and graduated matching thresholds.
"""

import uuid

import pytest

from action_item_graph.models.action_item import ActionItem
from action_item_graph.pipeline.extractor import compute_priority_score
from action_item_graph.pipeline.matcher import ActionItemMatcher


# ─────────────────────────────────────────────────────────────────────────────
# Priority Score Computation
# ─────────────────────────────────────────────────────────────────────────────


class TestPriorityScore:
    """Test the weighted priority score computation."""

    def test_maximum_scores(self):
        """All 5s with perfect confidence should yield 1.0."""
        score = compute_priority_score(5, 5, 5, 1.0)
        assert score == 1.0

    def test_minimum_scores(self):
        """All 1s with zero confidence should yield minimum."""
        score = compute_priority_score(1, 1, 1, 0.0)
        assert score == pytest.approx(0.18, abs=0.001)

    def test_impact_dominates(self):
        """Impact (40% weight) should be the strongest signal."""
        high_impact = compute_priority_score(5, 1, 1, 0.5)
        high_urgency = compute_priority_score(1, 5, 1, 0.5)
        assert high_impact > high_urgency

    def test_urgency_second(self):
        """Urgency (35% weight) should be second strongest."""
        high_urgency = compute_priority_score(1, 5, 1, 0.5)
        high_specificity = compute_priority_score(1, 1, 5, 0.5)
        assert high_urgency > high_specificity

    def test_result_range(self):
        """Score should always be between 0.0 and 1.0."""
        for impact in range(1, 6):
            for urgency in range(1, 6):
                for specificity in range(1, 6):
                    for conf in [0.0, 0.5, 1.0]:
                        score = compute_priority_score(impact, urgency, specificity, conf)
                        assert 0.0 <= score <= 1.0

    def test_precision(self):
        """Score should be rounded to 3 decimal places."""
        score = compute_priority_score(3, 3, 3, 0.7)
        assert score == round(score, 3)

    def test_real_world_deal_critical(self):
        """Deal-critical item with deadline should score high."""
        score = compute_priority_score(
            impact=5,     # Deal-critical
            urgency=4,    # Hard deadline in 48h
            specificity=4,  # Clear deliverable
            confidence=0.9,
        )
        assert score >= 0.8

    def test_real_world_vague_followup(self):
        """Vague follow-up with no urgency should score low."""
        score = compute_priority_score(
            impact=2,     # Nice-to-have
            urgency=1,    # No rush
            specificity=2,  # Vague
            confidence=0.6,
        )
        assert score < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ActionItem Scoring Fields
# ─────────────────────────────────────────────────────────────────────────────


TENANT_ID = uuid.UUID('550e8400-e29b-41d4-a716-446655440000')


class TestActionItemScoringFields:
    """Test that ActionItem model properly stores and serializes scoring fields."""

    def test_scoring_fields_in_model(self):
        """ActionItem should accept scoring fields."""
        ai = ActionItem(
            tenant_id=TENANT_ID,
            action_item_text='Send pricing deck',
            summary='Send pricing',
            owner='Sarah',
            commitment_strength='explicit',
            score_impact=5,
            score_urgency=4,
            score_specificity=4,
            score_effort=2,
            priority_score=0.86,
            definition_of_done='Client receives pricing email',
        )
        assert ai.commitment_strength == 'explicit'
        assert ai.score_impact == 5
        assert ai.priority_score == 0.86
        assert ai.definition_of_done == 'Client receives pricing email'

    def test_scoring_fields_in_neo4j_properties(self):
        """to_neo4j_properties should include non-None scoring fields."""
        ai = ActionItem(
            tenant_id=TENANT_ID,
            action_item_text='Test',
            summary='Test',
            owner='Test',
            score_impact=4,
            priority_score=0.75,
            commitment_strength='explicit',
        )
        props = ai.to_neo4j_properties()
        assert props['score_impact'] == 4
        assert props['priority_score'] == 0.75
        assert props['commitment_strength'] == 'explicit'

    def test_null_scoring_fields_excluded(self):
        """None scoring fields should be excluded from Neo4j properties."""
        ai = ActionItem(
            tenant_id=TENANT_ID,
            action_item_text='Test',
            summary='Test',
            owner='Test',
        )
        props = ai.to_neo4j_properties()
        assert 'score_impact' not in props
        assert 'priority_score' not in props
        assert 'commitment_strength' not in props

    def test_score_validation(self):
        """Scores should be validated to 1-5 range."""
        with pytest.raises(Exception):  # Pydantic validation error
            ActionItem(
                tenant_id=TENANT_ID,
                action_item_text='Test',
                summary='Test',
                owner='Test',
                score_impact=6,  # Out of range
            )

    def test_priority_score_validation(self):
        """Priority score should be validated to 0.0-1.0 range."""
        with pytest.raises(Exception):
            ActionItem(
                tenant_id=TENANT_ID,
                action_item_text='Test',
                summary='Test',
                owner='Test',
                priority_score=1.5,  # Out of range
            )


# ─────────────────────────────────────────────────────────────────────────────
# Graduated Matching Thresholds
# ─────────────────────────────────────────────────────────────────────────────


class TestGraduatedThresholds:
    """Test the three-tier matching threshold constants."""

    def test_threshold_ordering(self):
        """Thresholds should be ordered: MIN < LLM_ZONE_UPPER."""
        assert ActionItemMatcher.MIN_SIMILARITY_SCORE < ActionItemMatcher.LLM_ZONE_UPPER

    def test_min_similarity(self):
        """Minimum similarity should be 0.68."""
        assert ActionItemMatcher.MIN_SIMILARITY_SCORE == 0.68

    def test_auto_match_threshold(self):
        """Auto-match threshold should be 0.88."""
        assert ActionItemMatcher.LLM_ZONE_UPPER == 0.88

    def test_thresholds_in_valid_range(self):
        """All thresholds should be between 0 and 1."""
        assert 0.0 < ActionItemMatcher.MIN_SIMILARITY_SCORE < 1.0
        assert 0.0 < ActionItemMatcher.LLM_ZONE_UPPER < 1.0
