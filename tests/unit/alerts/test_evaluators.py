"""Tests for alert rule evaluators."""

from dataclasses import dataclass

from app.services.alerts.evaluators import RuleEvaluator
from app.services.alerts.models import DriftSpikeConfig, ConfidenceDropConfig


@dataclass
class MockBucket:
    """Mock bucket for testing."""

    drift_score: float = 0.0
    avg_confidence: float = 0.0


class TestDriftSpikeEvaluator:
    """Tests for drift spike evaluation."""

    def test_condition_met_consecutive_buckets(self):
        """Condition met when N consecutive buckets exceed threshold."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=2)
        buckets = [
            MockBucket(drift_score=0.25),
            MockBucket(drift_score=0.32),
            MockBucket(drift_score=0.35),
        ]
        result = evaluator.evaluate_drift_spike(buckets, config)
        assert result.condition_met is True
        assert result.condition_clear is False
        assert result.trigger_value == 0.35

    def test_condition_not_met_single_bucket(self):
        """Condition not met with only one bucket above threshold."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=2)
        buckets = [
            MockBucket(drift_score=0.25),
            MockBucket(drift_score=0.28),
            MockBucket(drift_score=0.35),
        ]
        result = evaluator.evaluate_drift_spike(buckets, config)
        assert result.condition_met is False

    def test_condition_clear_with_hysteresis(self):
        """Condition clears when below threshold minus hysteresis."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(
            drift_threshold=0.30, consecutive_buckets=2, hysteresis=0.05
        )
        buckets = [
            MockBucket(drift_score=0.28),
            MockBucket(drift_score=0.24),
            MockBucket(drift_score=0.20),
        ]
        result = evaluator.evaluate_drift_spike(buckets, config)
        assert result.condition_clear is True

    def test_insufficient_data(self):
        """Returns insufficient_data when not enough buckets."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=3)
        buckets = [MockBucket(drift_score=0.35), MockBucket(drift_score=0.35)]
        result = evaluator.evaluate_drift_spike(buckets, config)
        assert result.insufficient_data is True

    def test_tie_break_prioritizes_alerting(self):
        """When both condition_met and condition_clear possible, prioritize alerting."""
        evaluator = RuleEvaluator()
        config = DriftSpikeConfig(
            drift_threshold=0.30, consecutive_buckets=2, hysteresis=0.0
        )
        buckets = [MockBucket(drift_score=0.30), MockBucket(drift_score=0.30)]
        result = evaluator.evaluate_drift_spike(buckets, config)
        assert result.condition_met is True
        assert result.condition_clear is False


class TestConfidenceDropEvaluator:
    """Tests for confidence drop evaluation."""

    def test_condition_met_trend_below_threshold(self):
        """Condition met when trend delta exceeds threshold."""
        evaluator = RuleEvaluator()
        config = ConfidenceDropConfig(trend_threshold=0.05)
        buckets = [
            MockBucket(avg_confidence=0.72),
            MockBucket(avg_confidence=0.68),
            MockBucket(avg_confidence=0.62),
            MockBucket(avg_confidence=0.58),
        ]
        result = evaluator.evaluate_confidence_drop(buckets, config)
        assert result.condition_met is True
        assert result.trigger_value < 0

    def test_condition_not_met_stable_confidence(self):
        """Condition not met when confidence is stable."""
        evaluator = RuleEvaluator()
        config = ConfidenceDropConfig(trend_threshold=0.05)
        buckets = [
            MockBucket(avg_confidence=0.70),
            MockBucket(avg_confidence=0.71),
            MockBucket(avg_confidence=0.69),
            MockBucket(avg_confidence=0.70),
        ]
        result = evaluator.evaluate_confidence_drop(buckets, config)
        assert result.condition_met is False


class TestComboEvaluator:
    """Tests for combo rule evaluation."""

    def test_condition_met_both_active(self):
        """Combo condition met when both drift and confidence conditions met."""
        evaluator = RuleEvaluator()
        drift_config = DriftSpikeConfig(drift_threshold=0.30, consecutive_buckets=2)
        confidence_config = ConfidenceDropConfig(trend_threshold=0.05)
        buckets = [
            MockBucket(drift_score=0.25, avg_confidence=0.75),
            MockBucket(drift_score=0.35, avg_confidence=0.70),
            MockBucket(drift_score=0.40, avg_confidence=0.62),
            MockBucket(drift_score=0.38, avg_confidence=0.58),
        ]
        result = evaluator.evaluate_combo(buckets, drift_config, confidence_config)
        assert result.condition_met is True
        assert "drift" in result.context
        assert "confidence" in result.context

    def test_condition_clear_either_clears(self):
        """Combo clears when either underlying condition clears."""
        evaluator = RuleEvaluator()
        drift_config = DriftSpikeConfig(
            drift_threshold=0.30, consecutive_buckets=2, hysteresis=0.05
        )
        confidence_config = ConfidenceDropConfig(trend_threshold=0.05)
        buckets = [
            MockBucket(drift_score=0.20, avg_confidence=0.75),
            MockBucket(drift_score=0.18, avg_confidence=0.70),
            MockBucket(drift_score=0.15, avg_confidence=0.62),
            MockBucket(drift_score=0.12, avg_confidence=0.58),
        ]
        result = evaluator.evaluate_combo(buckets, drift_config, confidence_config)
        assert result.condition_clear is True
