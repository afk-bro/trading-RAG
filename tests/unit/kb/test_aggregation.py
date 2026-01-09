"""Unit tests for KB aggregation module."""

import pytest
from unittest.mock import MagicMock

from app.services.kb.aggregation import (
    compute_weight,
    weighted_median,
    weighted_mode,
    compute_iqr,
    aggregate_params,
    validate_and_repair_params,
    compute_confidence,
    ParamSpread,
    RELAXED_WEIGHT_MULTIPLIER,
    METADATA_ONLY_WEIGHT_MULTIPLIER,
)
from app.services.kb.rerank import RerankedCandidate
from app.services.strategies.registry import StrategySpec
from app.services.strategies.params import ParamSpec, ParamType


# =============================================================================
# Fixtures
# =============================================================================


def make_candidate(
    point_id: str,
    params: dict,
    objective_score: float = 1.0,
    rerank_score: float = 0.8,
    relaxed: bool = False,
    metadata_only: bool = False,
) -> RerankedCandidate:
    """Create a test candidate."""
    return RerankedCandidate(
        point_id=point_id,
        payload={
            "params": params,
            "objective_score": objective_score,
            "strategy_name": "test",
        },
        similarity_score=0.8,
        jaccard_score=0.5,
        rerank_score=rerank_score,
        used_regime_source="oos",
        _relaxed=relaxed,
        _metadata_only=metadata_only,
    )


@pytest.fixture
def simple_strategy_spec():
    """Create simple strategy spec for testing."""
    return StrategySpec(
        name="test_strategy",
        params={
            "ema_fast": ParamSpec(
                name="ema_fast",
                type=ParamType.INT,
                min_value=5,
                max_value=50,
                default=12,
            ),
            "ema_slow": ParamSpec(
                name="ema_slow",
                type=ParamType.INT,
                min_value=10,
                max_value=200,
                default=26,
            ),
            "threshold": ParamSpec(
                name="threshold",
                type=ParamType.FLOAT,
                min_value=0.0,
                max_value=1.0,
                default=0.5,
            ),
            "use_filter": ParamSpec(
                name="use_filter",
                type=ParamType.BOOL,
                default=True,
            ),
            "mode": ParamSpec(
                name="mode",
                type=ParamType.ENUM,
                choices=["fast", "slow", "balanced"],
                default="balanced",
            ),
        },
    )


# =============================================================================
# Weight Computation Tests
# =============================================================================


class TestComputeWeight:
    """Tests for weight computation."""

    def test_base_weight(self):
        """Should compute base weight from rerank score."""
        candidate = make_candidate("test", {}, rerank_score=0.8)

        weight = compute_weight(candidate, max_rerank_score=1.0)

        assert weight == pytest.approx(0.8)

    def test_relaxed_penalty(self):
        """Should apply relaxed penalty."""
        candidate = make_candidate("test", {}, rerank_score=0.8, relaxed=True)

        weight = compute_weight(candidate, max_rerank_score=1.0)

        assert weight == pytest.approx(0.8 * RELAXED_WEIGHT_MULTIPLIER)

    def test_metadata_only_penalty(self):
        """Should apply metadata-only penalty."""
        candidate = make_candidate("test", {}, rerank_score=0.8, metadata_only=True)

        weight = compute_weight(candidate, max_rerank_score=1.0)

        assert weight == pytest.approx(0.8 * METADATA_ONLY_WEIGHT_MULTIPLIER)

    def test_minimum_weight(self):
        """Should not go below minimum weight."""
        candidate = make_candidate("test", {}, rerank_score=0.001)

        weight = compute_weight(candidate, max_rerank_score=1.0)

        assert weight >= 0.01


# =============================================================================
# Weighted Statistics Tests
# =============================================================================


class TestWeightedMedian:
    """Tests for weighted median."""

    def test_equal_weights(self):
        """With equal weights, should be regular median."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = weighted_median(values, weights)

        assert result == 3.0

    def test_weighted_median(self):
        """Should weight towards heavier values."""
        values = [1.0, 2.0, 3.0]
        weights = [1.0, 1.0, 10.0]  # 3.0 has most weight

        result = weighted_median(values, weights)

        assert result == 3.0

    def test_single_value(self):
        """Should return single value."""
        result = weighted_median([42.0], [1.0])
        assert result == 42.0

    def test_empty_raises(self):
        """Should raise on empty input."""
        with pytest.raises(ValueError):
            weighted_median([], [])


class TestWeightedMode:
    """Tests for weighted mode."""

    def test_clear_winner(self):
        """Should return value with most weight."""
        values = ["a", "b", "c"]
        weights = [1.0, 5.0, 2.0]

        mode, fraction = weighted_mode(values, weights)

        assert mode == "b"
        assert fraction == pytest.approx(5.0 / 8.0)

    def test_bool_values(self):
        """Should work with bool values."""
        values = [True, False, True, True]
        weights = [1.0, 1.0, 1.0, 1.0]

        mode, fraction = weighted_mode(values, weights)

        assert mode is True
        assert fraction == pytest.approx(0.75)

    def test_single_value(self):
        """Should return single value with fraction 1.0."""
        mode, fraction = weighted_mode(["only"], [1.0])

        assert mode == "only"
        assert fraction == 1.0


class TestComputeIQR:
    """Tests for IQR computation."""

    def test_basic_iqr(self):
        """Should compute IQR correctly."""
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        weights = [1.0] * 8

        iqr = compute_iqr(values, weights)

        # Q1 = 2, Q3 = 6 (indices 2 and 6)
        assert iqr == 4.0

    def test_single_value(self):
        """Single value should have IQR 0."""
        assert compute_iqr([5.0], [1.0]) == 0.0


# =============================================================================
# Aggregate Params Tests
# =============================================================================


class TestAggregateParams:
    """Tests for aggregate_params function."""

    def test_empty_candidates(self):
        """Should handle empty candidates."""
        result = aggregate_params([])

        assert result.params == {}
        assert result.count_used == 0
        assert "no_candidates_for_aggregation" in result.warnings

    def test_numeric_aggregation(self, simple_strategy_spec):
        """Should use weighted median for numeric params."""
        candidates = [
            make_candidate("a", {"ema_fast": 10, "threshold": 0.3}, rerank_score=0.9),
            make_candidate("b", {"ema_fast": 12, "threshold": 0.4}, rerank_score=0.8),
            make_candidate("c", {"ema_fast": 14, "threshold": 0.5}, rerank_score=0.7),
        ]

        result = aggregate_params(candidates, simple_strategy_spec)

        assert "ema_fast" in result.params
        assert "threshold" in result.params
        # Weighted median should be close to values with higher weights
        assert isinstance(result.params["ema_fast"], int)
        assert isinstance(result.params["threshold"], float)

    def test_bool_aggregation(self, simple_strategy_spec):
        """Should use weighted mode for bool params."""
        candidates = [
            make_candidate("a", {"use_filter": True}, rerank_score=0.9),
            make_candidate("b", {"use_filter": True}, rerank_score=0.8),
            make_candidate("c", {"use_filter": False}, rerank_score=0.7),
        ]

        result = aggregate_params(candidates, simple_strategy_spec)

        assert result.params["use_filter"] is True  # More weight on True

    def test_enum_aggregation(self, simple_strategy_spec):
        """Should use weighted mode for enum params."""
        candidates = [
            make_candidate("a", {"mode": "fast"}, rerank_score=0.9),
            make_candidate("b", {"mode": "fast"}, rerank_score=0.8),
            make_candidate("c", {"mode": "slow"}, rerank_score=0.7),
        ]

        result = aggregate_params(candidates, simple_strategy_spec)

        assert result.params["mode"] == "fast"

    def test_spreads_computed(self, simple_strategy_spec):
        """Should compute spreads for params."""
        candidates = [
            make_candidate("a", {"ema_fast": 10}, rerank_score=0.9),
            make_candidate("b", {"ema_fast": 20}, rerank_score=0.8),
            make_candidate("c", {"ema_fast": 30}, rerank_score=0.7),
        ]

        result = aggregate_params(candidates, simple_strategy_spec)

        assert "ema_fast" in result.spreads
        assert result.spreads["ema_fast"].count_used == 3
        assert result.spreads["ema_fast"].spread is not None  # IQR computed


# =============================================================================
# Validate and Repair Tests
# =============================================================================


class TestValidateAndRepairParams:
    """Tests for validate_and_repair_params function."""

    def test_clamp_to_min(self, simple_strategy_spec):
        """Should clamp values below min."""
        params = {"ema_fast": 1}  # Below min of 5

        repaired, warnings = validate_and_repair_params(params, simple_strategy_spec)

        assert repaired["ema_fast"] == 5
        assert "param_ema_fast_clamped_to_min" in warnings

    def test_clamp_to_max(self, simple_strategy_spec):
        """Should clamp values above max."""
        params = {"ema_fast": 100}  # Above max of 50

        repaired, warnings = validate_and_repair_params(params, simple_strategy_spec)

        assert repaired["ema_fast"] == 50
        assert "param_ema_fast_clamped_to_max" in warnings

    def test_type_coercion(self, simple_strategy_spec):
        """Should coerce types correctly."""
        params = {"ema_fast": 12.7}  # Float instead of int

        repaired, warnings = validate_and_repair_params(params, simple_strategy_spec)

        assert repaired["ema_fast"] == 13  # Rounded
        assert isinstance(repaired["ema_fast"], int)

    def test_invalid_enum_uses_default(self, simple_strategy_spec):
        """Should use default for invalid enum."""
        params = {"mode": "invalid"}

        repaired, warnings = validate_and_repair_params(params, simple_strategy_spec)

        assert repaired["mode"] == "balanced"  # Default
        assert "param_mode_invalid_enum_using_default" in warnings

    def test_step_snapping(self):
        """Should snap to step values."""
        spec = StrategySpec(
            name="test",
            params={
                "period": ParamSpec(
                    name="period",
                    type=ParamType.INT,
                    min_value=10,
                    max_value=100,
                    step=5,
                    default=20,
                ),
            },
        )

        params = {"period": 23}  # Should snap to 25

        repaired, warnings = validate_and_repair_params(params, spec)

        assert repaired["period"] == 25


# =============================================================================
# Confidence Computation Tests
# =============================================================================


class TestComputeConfidence:
    """Tests for confidence computation."""

    def test_zero_count(self):
        """Zero count should give 0 confidence."""
        result = compute_confidence(
            spreads={},
            count_used=0,
        )

        assert result == 0.0

    def test_high_count_high_confidence(self):
        """High count should give high confidence."""
        result = compute_confidence(
            spreads={},
            count_used=100,
        )

        assert result >= 0.8

    def test_warnings_reduce_confidence(self):
        """Warnings should reduce confidence."""
        base = compute_confidence(
            spreads={},
            count_used=50,
            has_warnings=False,
        )

        with_warnings = compute_confidence(
            spreads={},
            count_used=50,
            has_warnings=True,
        )

        assert with_warnings < base

    def test_relaxed_reduces_confidence(self):
        """Relaxed filters should reduce confidence."""
        base = compute_confidence(
            spreads={},
            count_used=50,
            used_relaxed=False,
        )

        with_relaxed = compute_confidence(
            spreads={},
            count_used=50,
            used_relaxed=True,
        )

        assert with_relaxed < base

    def test_metadata_fallback_reduces_confidence(self):
        """Metadata fallback should significantly reduce confidence."""
        base = compute_confidence(
            spreads={},
            count_used=50,
        )

        with_metadata = compute_confidence(
            spreads={},
            count_used=50,
            used_metadata_fallback=True,
        )

        assert with_metadata < base - 0.2  # Significant reduction

    def test_bounded_0_to_1(self):
        """Confidence should always be in [0, 1]."""
        # With all penalties
        result = compute_confidence(
            spreads={},
            count_used=5,
            has_warnings=True,
            used_relaxed=True,
            used_metadata_fallback=True,
        )

        assert 0.0 <= result <= 1.0
