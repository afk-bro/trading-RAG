"""Unit tests for coverage gap detection."""

import pytest

from app.services.coverage_gap.detector import (
    CoverageReasonCode,
    assess_coverage,
    compute_intent_signature,
    generate_suggestions,
)
from app.services.intent.models import MatchIntent


# =============================================================================
# Test compute_intent_signature
# =============================================================================


class TestComputeIntentSignature:
    """Tests for intent signature computation."""

    def test_signature_is_deterministic(self):
        """Same intent produces same signature."""
        intent = MatchIntent(
            strategy_archetypes=["breakout", "momentum"],
            indicators=["rsi", "volume"],
            timeframe_buckets=["swing"],
            topics=["crypto"],
            risk_terms=["stop_loss"],
        )

        sig1 = compute_intent_signature(intent)
        sig2 = compute_intent_signature(intent)

        assert sig1 == sig2

    def test_signature_is_stable_regardless_of_order(self):
        """Order of tags doesn't affect signature (sorted internally)."""
        intent1 = MatchIntent(
            strategy_archetypes=["breakout", "momentum"],
            indicators=["volume", "rsi"],
        )
        intent2 = MatchIntent(
            strategy_archetypes=["momentum", "breakout"],
            indicators=["rsi", "volume"],
        )

        assert compute_intent_signature(intent1) == compute_intent_signature(intent2)

    def test_signature_is_sha256_hex(self):
        """Signature is 64-character hex string."""
        intent = MatchIntent(strategy_archetypes=["breakout"])
        sig = compute_intent_signature(intent)

        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_different_intents_have_different_signatures(self):
        """Different intents produce different signatures."""
        intent1 = MatchIntent(strategy_archetypes=["breakout"])
        intent2 = MatchIntent(strategy_archetypes=["mean_reversion"])

        assert compute_intent_signature(intent1) != compute_intent_signature(intent2)

    def test_empty_intent_has_valid_signature(self):
        """Empty intent produces a valid signature."""
        intent = MatchIntent()
        sig = compute_intent_signature(intent)

        assert len(sig) == 64


# =============================================================================
# Test assess_coverage
# =============================================================================


class TestAssessCoverage:
    """Tests for coverage assessment."""

    def test_no_results_is_weak_coverage(self):
        """Empty results = weak coverage."""
        intent = MatchIntent(
            strategy_archetypes=["breakout"],
            overall_confidence=0.6,  # High enough to not trigger LOW_SIGNAL_INPUT
        )
        assessment = assess_coverage([], intent)

        assert assessment.weak is True
        assert CoverageReasonCode.NO_RESULTS.value in assessment.reason_codes
        assert assessment.best_score is None
        assert assessment.num_above_threshold == 0

    def test_low_best_score_is_weak_coverage(self):
        """Best score < 0.45 = weak coverage."""
        intent = MatchIntent(
            strategy_archetypes=["breakout"],
            overall_confidence=0.6,
        )
        scores = [0.40, 0.35, 0.30]
        assessment = assess_coverage(scores, intent)

        assert assessment.weak is True
        assert CoverageReasonCode.LOW_BEST_SCORE.value in assessment.reason_codes
        assert assessment.best_score == 0.40

    def test_no_results_above_threshold_is_weak_coverage(self):
        """No results >= threshold = weak coverage."""
        intent = MatchIntent(
            strategy_archetypes=["breakout"],
            overall_confidence=0.6,
        )
        scores = [0.50, 0.48, 0.45]  # All below 0.55 threshold
        assessment = assess_coverage(scores, intent, threshold=0.55)

        assert assessment.weak is True
        assert (
            CoverageReasonCode.NO_RESULTS_ABOVE_THRESHOLD.value
            in assessment.reason_codes
        )
        assert assessment.num_above_threshold == 0

    def test_good_results_is_not_weak_coverage(self):
        """Good scores = not weak coverage."""
        intent = MatchIntent(
            strategy_archetypes=["breakout"],
            overall_confidence=0.6,
        )
        scores = [0.80, 0.70, 0.60, 0.55]
        assessment = assess_coverage(scores, intent, threshold=0.55)

        assert assessment.weak is False
        assert len(assessment.reason_codes) == 0
        assert assessment.best_score == 0.80
        assert assessment.num_above_threshold == 4

    def test_low_confidence_intent_not_flagged_as_weak(self):
        """Low confidence input = LOW_SIGNAL_INPUT, not weak."""
        intent = MatchIntent(
            strategy_archetypes=[],  # Minimal extraction
            overall_confidence=0.2,  # Below 0.35 threshold
        )
        scores = [0.30, 0.25]  # Would be weak normally
        assessment = assess_coverage(scores, intent)

        assert assessment.weak is False
        assert CoverageReasonCode.LOW_SIGNAL_INPUT.value in assessment.reason_codes
        assert "low signal" in assessment.suggestions[0].lower()

    def test_avg_top_k_score_calculated_correctly(self):
        """Average score calculated correctly."""
        intent = MatchIntent(overall_confidence=0.6)
        scores = [0.80, 0.60, 0.40, 0.20]
        assessment = assess_coverage(scores, intent, top_k=4)

        expected_avg = (0.80 + 0.60 + 0.40 + 0.20) / 4
        assert assessment.avg_top_k_score == pytest.approx(expected_avg)

    def test_top_k_limits_scores_considered(self):
        """Only top_k scores are used for avg calculation."""
        intent = MatchIntent(overall_confidence=0.6)
        scores = [0.80, 0.60, 0.40, 0.20, 0.10]
        assessment = assess_coverage(scores, intent, top_k=3)

        expected_avg = (0.80 + 0.60 + 0.40) / 3
        assert assessment.avg_top_k_score == expected_avg

    def test_custom_thresholds(self):
        """Custom thresholds work correctly."""
        intent = MatchIntent(overall_confidence=0.6)

        # Scores above default threshold (0.55) - not weak
        scores_good = [0.60, 0.58, 0.56]
        assessment1 = assess_coverage(scores_good, intent)
        assert assessment1.weak is False  # All conditions pass

        # Scores below threshold - weak due to NO_RESULTS_ABOVE_THRESHOLD
        scores_low = [0.50, 0.45]
        assessment2 = assess_coverage(scores_low, intent, threshold=0.55)
        assert assessment2.weak is True
        assert (
            CoverageReasonCode.NO_RESULTS_ABOVE_THRESHOLD.value
            in assessment2.reason_codes
        )

        # Custom weak_best_score threshold
        scores_borderline = [0.50, 0.48]
        assessment3 = assess_coverage(
            scores_borderline, intent, weak_best_score=0.55, threshold=0.40
        )
        assert assessment3.weak is True  # 0.50 < 0.55
        assert CoverageReasonCode.LOW_BEST_SCORE.value in assessment3.reason_codes

    def test_both_weak_conditions_can_trigger(self):
        """Both LOW_BEST_SCORE and NO_RESULTS_ABOVE_THRESHOLD can appear."""
        intent = MatchIntent(overall_confidence=0.6)
        scores = [0.30, 0.25, 0.20]  # Best < 0.45 AND none >= 0.55
        assessment = assess_coverage(scores, intent)

        assert assessment.weak is True
        assert CoverageReasonCode.LOW_BEST_SCORE.value in assessment.reason_codes
        assert (
            CoverageReasonCode.NO_RESULTS_ABOVE_THRESHOLD.value
            in assessment.reason_codes
        )


# =============================================================================
# Test generate_suggestions
# =============================================================================


class TestGenerateSuggestions:
    """Tests for suggestion generation."""

    def test_breakout_archetype_suggestion(self):
        """Breakout archetype generates appropriate suggestion."""
        intent = MatchIntent(strategy_archetypes=["breakout"])
        suggestions = generate_suggestions(intent)

        assert any("breakout" in s.lower() for s in suggestions)

    def test_trend_following_suggestion(self):
        """Trend following generates appropriate suggestion."""
        intent = MatchIntent(strategy_archetypes=["trend_following"])
        suggestions = generate_suggestions(intent)

        assert any("trend" in s.lower() for s in suggestions)

    def test_volume_indicator_suggestion(self):
        """Volume indicator generates appropriate suggestion."""
        intent = MatchIntent(indicators=["volume"])
        suggestions = generate_suggestions(intent)

        assert any("volume" in s.lower() for s in suggestions)

    def test_options_topic_suggestion(self):
        """Options topic generates appropriate suggestion."""
        intent = MatchIntent(topics=["options"])
        suggestions = generate_suggestions(intent)

        assert any("options" in s.lower() for s in suggestions)

    def test_combination_breakout_volume_suggestion(self):
        """Breakout + volume generates combined suggestion."""
        intent = MatchIntent(
            strategy_archetypes=["breakout"],
            indicators=["volume"],
        )
        suggestions = generate_suggestions(intent)

        # Should have combined suggestion
        assert any(
            "breakout" in s.lower() and "volume" in s.lower() for s in suggestions
        )

    def test_max_three_suggestions(self):
        """Never more than 3 suggestions."""
        intent = MatchIntent(
            strategy_archetypes=["breakout", "trend_following", "momentum"],
            indicators=["rsi", "volume", "bollinger"],
            topics=["crypto", "options"],
        )
        suggestions = generate_suggestions(intent)

        assert len(suggestions) <= 3

    def test_fallback_suggestion_for_empty_intent(self):
        """Empty intent gets fallback suggestion."""
        intent = MatchIntent()
        suggestions = generate_suggestions(intent)

        assert len(suggestions) == 1
        assert "import" in suggestions[0].lower()

    def test_scalp_timeframe_suggestion(self):
        """Scalp timeframe generates appropriate suggestion."""
        intent = MatchIntent(timeframe_buckets=["scalp"])
        suggestions = generate_suggestions(intent)

        assert any("scalp" in s.lower() for s in suggestions)


# =============================================================================
# Test CoverageAssessment dataclass
# =============================================================================


class TestCoverageAssessment:
    """Tests for CoverageAssessment fields."""

    def test_assessment_has_all_required_fields(self):
        """Assessment contains all required fields."""
        intent = MatchIntent(overall_confidence=0.6)
        scores = [0.80, 0.60]
        assessment = assess_coverage(scores, intent)

        assert hasattr(assessment, "weak")
        assert hasattr(assessment, "best_score")
        assert hasattr(assessment, "avg_top_k_score")
        assert hasattr(assessment, "num_above_threshold")
        assert hasattr(assessment, "threshold")
        assert hasattr(assessment, "reason_codes")
        assert hasattr(assessment, "suggestions")

    def test_threshold_is_included_in_assessment(self):
        """Threshold value is included for client reference."""
        intent = MatchIntent(overall_confidence=0.6)
        assessment = assess_coverage([], intent, threshold=0.55)

        assert assessment.threshold == 0.55
