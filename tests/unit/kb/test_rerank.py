"""Unit tests for KB reranking module."""

import pytest

from app.services.kb.rerank import (
    jaccard,
    regime_distance,
    get_candidate_regime_tags,
    compute_rerank_score,
    rerank_candidates,
    EMPTY_TAGS_JACCARD,
)
from app.services.kb.retrieval import RetrievalCandidate
from app.services.kb.types import RegimeSnapshot


# =============================================================================
# Jaccard Tests
# =============================================================================


class TestJaccard:
    """Tests for Jaccard similarity function."""

    def test_identical_tags(self):
        """Identical tags should return 1.0."""
        result = jaccard(["a", "b", "c"], ["a", "b", "c"])
        assert result == 1.0

    def test_no_overlap(self):
        """No overlap should return 0.0."""
        result = jaccard(["a", "b"], ["c", "d"])
        assert result == 0.0

    def test_partial_overlap(self):
        """Partial overlap should return correct ratio."""
        # intersection = 1 (b), union = 3 (a, b, c)
        result = jaccard(["a", "b"], ["b", "c"])
        assert result == pytest.approx(1 / 3)

    def test_empty_both(self):
        """Both empty should return EMPTY_TAGS_JACCARD."""
        result = jaccard([], [])
        assert result == EMPTY_TAGS_JACCARD

    def test_one_empty(self):
        """One empty should return 0.0."""
        result = jaccard(["a", "b"], [])
        assert result == 0.0

        result = jaccard([], ["a", "b"])
        assert result == 0.0

    def test_duplicates_ignored(self):
        """Duplicates should be handled via set conversion."""
        result = jaccard(["a", "a", "b"], ["a", "b", "b"])
        assert result == 1.0  # {a, b} == {a, b}


# =============================================================================
# Regime Distance Tests
# =============================================================================


class TestRegimeDistance:
    """Tests for regime distance function."""

    def test_identical_regimes(self):
        """Identical regimes should have distance 0."""
        regime = RegimeSnapshot(
            atr_pct=0.05,
            trend_strength=0.7,
            trend_dir=1,
            rsi=60,
            efficiency_ratio=0.6,
        )

        result = regime_distance(regime, regime)
        assert result == pytest.approx(0.0)

    def test_different_regimes(self):
        """Different regimes should have positive distance."""
        regime_a = RegimeSnapshot(
            atr_pct=0.02,
            trend_strength=0.2,
            trend_dir=-1,
            rsi=30,
            efficiency_ratio=0.3,
        )

        regime_b = RegimeSnapshot(
            atr_pct=0.08,
            trend_strength=0.9,
            trend_dir=1,
            rsi=80,
            efficiency_ratio=0.8,
        )

        result = regime_distance(regime_a, regime_b)
        assert result > 0.5  # Significantly different

    def test_none_regime(self):
        """None regime should return 1.0."""
        regime = RegimeSnapshot()

        assert regime_distance(None, regime) == 1.0
        assert regime_distance(regime, None) == 1.0
        assert regime_distance(None, None) == 1.0


# =============================================================================
# Get Candidate Regime Tags Tests
# =============================================================================


class TestGetCandidateRegimeTags:
    """Tests for extracting regime tags from payload."""

    def test_prefer_oos(self):
        """Should prefer OOS regime tags."""
        payload = {
            "regime_is": {"regime_tags": ["is_tag"]},
            "regime_oos": {"regime_tags": ["oos_tag"]},
        }

        tags, source = get_candidate_regime_tags(payload)

        assert tags == ["oos_tag"]
        assert source == "oos"

    def test_fallback_to_is(self):
        """Should fall back to IS if no OOS."""
        payload = {
            "regime_is": {"regime_tags": ["is_tag"]},
        }

        tags, source = get_candidate_regime_tags(payload)

        assert tags == ["is_tag"]
        assert source == "is"

    def test_top_level_tags(self):
        """Should use top-level regime_tags as fallback."""
        payload = {
            "regime_tags": ["top_tag"],
            "has_oos": True,
        }

        tags, source = get_candidate_regime_tags(payload)

        assert tags == ["top_tag"]
        assert source == "oos"

    def test_no_tags(self):
        """Should return empty and 'none' source if no tags."""
        payload = {}

        tags, source = get_candidate_regime_tags(payload)

        assert tags == []
        assert source == "none"


# =============================================================================
# Compute Rerank Score Tests
# =============================================================================


class TestComputeRerankScore:
    """Tests for rerank score computation."""

    def test_perfect_scores(self):
        """Perfect scores should give high result."""
        score = compute_rerank_score(
            similarity=1.0,
            jaccard_score=1.0,
            objective_score=1.0,
            max_objective=1.0,
        )

        # 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0 = 1.0
        assert score == pytest.approx(1.0)

    def test_zero_scores(self):
        """Zero scores should give low result."""
        score = compute_rerank_score(
            similarity=0.0,
            jaccard_score=0.0,
            objective_score=0.0,
            max_objective=1.0,
        )

        assert score == pytest.approx(0.0)

    def test_relaxed_penalty(self):
        """Relaxed candidates should get penalty."""
        base_score = compute_rerank_score(
            similarity=0.8,
            jaccard_score=0.8,
            objective_score=0.8,
            max_objective=1.0,
            is_relaxed=False,
        )

        relaxed_score = compute_rerank_score(
            similarity=0.8,
            jaccard_score=0.8,
            objective_score=0.8,
            max_objective=1.0,
            is_relaxed=True,
        )

        assert relaxed_score < base_score
        assert relaxed_score == pytest.approx(base_score * 0.8)

    def test_metadata_only_penalty(self):
        """Metadata-only candidates should get heavy penalty."""
        base_score = compute_rerank_score(
            similarity=0.8,
            jaccard_score=0.8,
            objective_score=0.8,
            max_objective=1.0,
        )

        metadata_score = compute_rerank_score(
            similarity=0.8,
            jaccard_score=0.8,
            objective_score=0.8,
            max_objective=1.0,
            is_metadata_only=True,
        )

        assert metadata_score < base_score
        assert metadata_score == pytest.approx(base_score * 0.5)


# =============================================================================
# Rerank Candidates Tests
# =============================================================================


class TestRerankCandidates:
    """Tests for rerank_candidates function."""

    def test_empty_candidates(self):
        """Should handle empty candidate list."""
        result = rerank_candidates([], query_tags=["a"])

        assert result.candidates == []
        assert result.warnings == []

    def test_sorts_by_rerank_score(self):
        """Should sort candidates by rerank score descending."""
        candidates = [
            RetrievalCandidate(
                point_id="low",
                payload={"objective_score": 0.5, "regime_tags": ["a"]},
                similarity_score=0.5,
            ),
            RetrievalCandidate(
                point_id="high",
                payload={"objective_score": 1.0, "regime_tags": ["a", "b"]},
                similarity_score=0.9,
            ),
        ]

        result = rerank_candidates(candidates, query_tags=["a", "b"])

        assert result.candidates[0].point_id == "high"
        assert result.candidates[1].point_id == "low"

    def test_attaches_scores(self):
        """Should attach jaccard and rerank scores."""
        candidates = [
            RetrievalCandidate(
                point_id="test",
                payload={"objective_score": 0.8, "regime_tags": ["a", "b"]},
                similarity_score=0.7,
            ),
        ]

        result = rerank_candidates(candidates, query_tags=["a", "c"])

        cand = result.candidates[0]
        assert cand.jaccard_score == pytest.approx(1 / 3)  # {a} / {a, b, c}
        assert cand.rerank_score > 0
        assert cand.used_regime_source == "is"  # From top-level, assumes has_oos=False

    def test_preserves_relaxed_flag(self):
        """Should preserve _relaxed and _metadata_only flags."""
        candidates = [
            RetrievalCandidate(
                point_id="relaxed",
                payload={"objective_score": 0.5},
                similarity_score=0.5,
                _relaxed=True,
            ),
            RetrievalCandidate(
                point_id="metadata",
                payload={"objective_score": 0.3},
                similarity_score=0.0,
                _metadata_only=True,
            ),
        ]

        result = rerank_candidates(candidates, query_tags=[])

        relaxed_cand = [c for c in result.candidates if c.point_id == "relaxed"][0]
        metadata_cand = [c for c in result.candidates if c.point_id == "metadata"][0]

        assert relaxed_cand._relaxed is True
        assert metadata_cand._metadata_only is True

    def test_warns_on_many_empty_tags(self):
        """Should warn when many candidates have no regime tags."""
        candidates = [
            RetrievalCandidate(
                point_id=f"cand_{i}",
                payload={"objective_score": 0.5},  # No regime_tags
                similarity_score=0.5,
            )
            for i in range(10)
        ]

        result = rerank_candidates(candidates, query_tags=["a"])

        assert "many_candidates_missing_regime_tags" in result.warnings
