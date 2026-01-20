"""Unit tests for confidence computation module."""

from datetime import datetime, timezone, timedelta
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from app.services.intel.confidence import (
    compute_regime,
    compute_components,
    compute_confidence,
    ConfidenceContext,
    ConfidenceResult,
    DEFAULT_WEIGHTS,
    _compute_performance_component,
    _compute_drawdown_component,
    _compute_stability_component,
    _compute_freshness_component,
    _compute_regime_fit_component,
    _compute_trend_strength,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trending_ohlcv():
    """Generate OHLCV data with clear uptrend."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    # Strong uptrend: price increases steadily
    base = 100
    prices = [base + i * 0.5 + np.random.normal(0, 0.1) for i in range(100)]

    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.3 for p in prices],
            "low": [p - 0.3 for p in prices],
            "close": prices,
            "volume": [1000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def ranging_ohlcv():
    """Generate OHLCV data with sideways/ranging price action."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    # Ranging: oscillates around mean with no trend
    base = 100
    prices = [base + 2 * np.sin(i / 5) + np.random.normal(0, 0.2) for i in range(100)]

    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def volatile_ohlcv():
    """Generate OHLCV data with high volatility."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    # High volatility: large price swings
    base = 100
    prices = [base + np.random.normal(0, 5) for i in range(100)]

    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 3 for p in prices],
            "low": [p - 3 for p in prices],
            "close": prices,
            "volume": [1000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def basic_context():
    """Create basic ConfidenceContext for testing."""
    return ConfidenceContext(
        version_id=uuid4(),
        as_of_ts=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def full_context(trending_ohlcv):
    """Create ConfidenceContext with all data populated."""
    return ConfidenceContext(
        version_id=uuid4(),
        as_of_ts=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        ohlcv=trending_ohlcv,
        backtest_metrics={
            "sharpe": 1.5,
            "return_pct": 25.0,
            "max_drawdown_pct": 15.0,
            "trades": 45,
            "win_rate": 0.55,
        },
        latest_candle_ts=datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
        strategy_regime_profile={
            "good_regimes": ["trend_low_vol", "trend_mid_vol"],
            "bad_regimes": ["range_high_vol"],
        },
    )


# =============================================================================
# Regime Classification Tests
# =============================================================================


class TestComputeRegime:
    """Tests for compute_regime()."""

    def test_insufficient_data_returns_unknown(self):
        """Test that insufficient data returns unknown regime."""
        # Empty dataframe
        regime, features = compute_regime(pd.DataFrame())
        assert regime == "unknown"
        assert features["reason"] == "insufficient_data"

        # Too few rows
        small_df = pd.DataFrame({"close": [100, 101, 102]})
        regime, features = compute_regime(small_df)
        assert regime == "unknown"

    def test_none_returns_unknown(self):
        """Test that None input returns unknown."""
        regime, features = compute_regime(None)
        assert regime == "unknown"

    def test_missing_close_column(self):
        """Test handling of missing close column."""
        df = pd.DataFrame({"open": range(50), "volume": range(50)})
        regime, features = compute_regime(df)
        assert regime == "unknown"
        assert features["reason"] == "missing_close_column"

    def test_trending_data_detected(self, trending_ohlcv):
        """Test that trending data is classified correctly."""
        regime, features = compute_regime(trending_ohlcv)
        assert regime.startswith("trend_")
        assert features["is_trending"] is True
        assert features["trend_strength"] > 0.4

    def test_ranging_data_detected(self, ranging_ohlcv):
        """Test that ranging data is classified correctly."""
        regime, features = compute_regime(ranging_ohlcv)
        # Ranging should have low trend strength
        assert features["trend_strength"] < 0.4 or regime.startswith("range_")

    def test_regime_format(self, trending_ohlcv):
        """Test regime string format is valid."""
        regime, _ = compute_regime(trending_ohlcv)
        # Should be format: {trend|range}_{low_vol|mid_vol|high_vol}
        parts = regime.split("_")
        assert len(parts) >= 2
        assert parts[0] in ("trend", "range", "unknown")

    def test_features_contain_expected_keys(self, trending_ohlcv):
        """Test that features dict contains expected keys."""
        regime, features = compute_regime(trending_ohlcv)
        assert "trend_strength" in features
        assert "volatility" in features
        assert "vol_percentile" in features
        assert "is_trending" in features
        assert "bars_used" in features


class TestTrendStrength:
    """Tests for _compute_trend_strength()."""

    def test_perfect_uptrend(self):
        """Test that perfect uptrend has high R-squared."""
        prices = pd.Series([100 + i for i in range(50)])
        strength = _compute_trend_strength(prices)
        assert strength > 0.95

    def test_perfect_downtrend(self):
        """Test that perfect downtrend has high R-squared."""
        prices = pd.Series([100 - i for i in range(50)])
        strength = _compute_trend_strength(prices)
        assert strength > 0.95

    def test_random_walk_low_strength(self):
        """Test that random walk has low trend strength."""
        np.random.seed(42)
        prices = pd.Series([100 + np.random.normal(0, 1) for _ in range(50)])
        strength = _compute_trend_strength(prices)
        assert strength < 0.5

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        prices = pd.Series([100, 101, 102])
        strength = _compute_trend_strength(prices, lookback=5)
        # Should handle gracefully
        assert 0 <= strength <= 1


# =============================================================================
# Confidence Components Tests
# =============================================================================


class TestPerformanceComponent:
    """Tests for _compute_performance_component()."""

    def test_no_data_returns_neutral(self, basic_context):
        """Test that missing metrics returns neutral 0.5."""
        score = _compute_performance_component(basic_context)
        assert score == 0.5

    def test_negative_sharpe(self, basic_context):
        """Test negative Sharpe maps to low score."""
        basic_context.backtest_metrics = {"sharpe": -1.0}
        score = _compute_performance_component(basic_context)
        assert score == 0.0

        basic_context.backtest_metrics = {"sharpe": -0.5}
        score = _compute_performance_component(basic_context)
        assert 0 < score < 0.3

    def test_positive_sharpe(self, basic_context):
        """Test positive Sharpe maps correctly."""
        basic_context.backtest_metrics = {"sharpe": 0.0}
        score = _compute_performance_component(basic_context)
        assert score == pytest.approx(0.3, abs=0.01)

        basic_context.backtest_metrics = {"sharpe": 1.0}
        score = _compute_performance_component(basic_context)
        assert score == pytest.approx(0.6, abs=0.01)

        basic_context.backtest_metrics = {"sharpe": 2.0}
        score = _compute_performance_component(basic_context)
        assert score == pytest.approx(0.8, abs=0.01)

    def test_excellent_sharpe(self, basic_context):
        """Test excellent Sharpe (3+) maps to 1.0."""
        basic_context.backtest_metrics = {"sharpe": 3.0}
        score = _compute_performance_component(basic_context)
        assert score == 1.0

        basic_context.backtest_metrics = {"sharpe": 5.0}
        score = _compute_performance_component(basic_context)
        assert score == 1.0

    def test_wfo_preferred_over_backtest(self, basic_context):
        """Test that WFO metrics take precedence."""
        basic_context.backtest_metrics = {"sharpe": 0.5}
        basic_context.wfo_metrics = {"oos_sharpe": 2.0}

        score = _compute_performance_component(basic_context)
        # Should use WFO's 2.0, not backtest's 0.5
        assert score == pytest.approx(0.8, abs=0.01)


class TestDrawdownComponent:
    """Tests for _compute_drawdown_component()."""

    def test_no_data_returns_neutral(self, basic_context):
        """Test that missing metrics returns neutral."""
        score = _compute_drawdown_component(basic_context)
        assert score == 0.5

    def test_zero_drawdown(self, basic_context):
        """Test zero drawdown gives perfect score."""
        basic_context.backtest_metrics = {"max_drawdown_pct": 0}
        score = _compute_drawdown_component(basic_context)
        assert score == 1.0

    def test_moderate_drawdown(self, basic_context):
        """Test moderate drawdown scoring."""
        basic_context.backtest_metrics = {"max_drawdown_pct": 10}
        score = _compute_drawdown_component(basic_context)
        assert score == pytest.approx(0.8, abs=0.01)

        basic_context.backtest_metrics = {"max_drawdown_pct": 25}
        score = _compute_drawdown_component(basic_context)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_severe_drawdown(self, basic_context):
        """Test severe drawdown gives low score."""
        basic_context.backtest_metrics = {"max_drawdown_pct": 50}
        score = _compute_drawdown_component(basic_context)
        assert score == 0.0

        basic_context.backtest_metrics = {"max_drawdown_pct": 75}
        score = _compute_drawdown_component(basic_context)
        assert score == 0.0

    def test_handles_decimal_format(self, basic_context):
        """Test handling of decimal DD format (0.25 vs 25)."""
        basic_context.backtest_metrics = {"max_drawdown_pct": 0.25}
        score = _compute_drawdown_component(basic_context)
        # 0.25 * 100 = 25%
        assert score == pytest.approx(0.5, abs=0.01)


class TestStabilityComponent:
    """Tests for _compute_stability_component()."""

    def test_no_data_returns_neutral(self, basic_context):
        """Test that missing metrics returns neutral."""
        score = _compute_stability_component(basic_context)
        assert score == 0.5

    def test_low_variance_high_stability(self, basic_context):
        """Test low WFO variance gives high stability."""
        basic_context.wfo_metrics = {"fold_variance": 0.0}
        score = _compute_stability_component(basic_context)
        assert score == 1.0

        basic_context.wfo_metrics = {"fold_variance": 0.2}
        score = _compute_stability_component(basic_context)
        assert score == 0.8

    def test_high_variance_low_stability(self, basic_context):
        """Test high WFO variance gives low stability."""
        basic_context.wfo_metrics = {"fold_variance": 0.8}
        score = _compute_stability_component(basic_context)
        assert score == pytest.approx(0.2, abs=0.01)

        basic_context.wfo_metrics = {"fold_variance": 1.0}
        score = _compute_stability_component(basic_context)
        assert score == 0.0

    def test_trade_count_fallback(self, basic_context):
        """Test trade count as stability proxy when WFO unavailable."""
        basic_context.backtest_metrics = {"trades": 5}
        score = _compute_stability_component(basic_context)
        assert score == pytest.approx(0.3, abs=0.01)

        basic_context.backtest_metrics = {"trades": 50}
        score = _compute_stability_component(basic_context)
        assert score >= 0.8


class TestFreshnessComponent:
    """Tests for _compute_freshness_component()."""

    def test_no_candle_ts_returns_penalty(self, basic_context):
        """Test missing candle timestamp gives slight penalty."""
        score = _compute_freshness_component(basic_context)
        assert score == 0.7

    def test_fresh_data(self, basic_context):
        """Test fresh data (< 1 hour old) gives perfect score."""
        basic_context.latest_candle_ts = basic_context.as_of_ts - timedelta(minutes=30)
        score = _compute_freshness_component(basic_context)
        assert score == 1.0

    def test_stale_data(self, basic_context):
        """Test stale data gives progressively lower scores."""
        # 2 hours stale
        basic_context.latest_candle_ts = basic_context.as_of_ts - timedelta(hours=2)
        score = _compute_freshness_component(basic_context)
        assert score == 0.9

        # 12 hours stale
        basic_context.latest_candle_ts = basic_context.as_of_ts - timedelta(hours=12)
        score = _compute_freshness_component(basic_context)
        assert score == 0.7

        # 1 week stale
        basic_context.latest_candle_ts = basic_context.as_of_ts - timedelta(days=7)
        score = _compute_freshness_component(basic_context)
        assert score == 0.5


class TestRegimeFitComponent:
    """Tests for _compute_regime_fit_component()."""

    def test_no_profile_returns_neutral(self, basic_context):
        """Test missing profile returns neutral."""
        score = _compute_regime_fit_component(basic_context, "trend_low_vol")
        assert score == 0.5

    def test_good_regime_match(self, basic_context):
        """Test matching good regime gives bonus."""
        basic_context.strategy_regime_profile = {
            "good_regimes": ["trend_low_vol"],
            "bad_regimes": [],
        }
        score = _compute_regime_fit_component(basic_context, "trend_low_vol")
        assert score == 0.85

    def test_partial_good_match(self, basic_context):
        """Test partial match on good regime."""
        basic_context.strategy_regime_profile = {
            "good_regimes": ["trend_low_vol"],
            "bad_regimes": [],
        }
        # trend_mid_vol shares "trend" with trend_low_vol
        score = _compute_regime_fit_component(basic_context, "trend_mid_vol")
        assert score == 0.7

    def test_bad_regime_match(self, basic_context):
        """Test matching bad regime gives penalty."""
        basic_context.strategy_regime_profile = {
            "good_regimes": [],
            "bad_regimes": ["range_high_vol"],
        }
        score = _compute_regime_fit_component(basic_context, "range_high_vol")
        assert score == 0.15


# =============================================================================
# Main Computation Tests
# =============================================================================


class TestComputeConfidence:
    """Tests for compute_confidence()."""

    def test_returns_confidence_result(self, full_context):
        """Test that result is proper ConfidenceResult."""
        result = compute_confidence(full_context)

        assert isinstance(result, ConfidenceResult)
        assert isinstance(result.regime, str)
        assert isinstance(result.confidence_score, float)
        assert isinstance(result.confidence_components, dict)
        assert isinstance(result.inputs_hash, str)

    def test_score_bounds(self, full_context):
        """Test that confidence score is within [0, 1]."""
        result = compute_confidence(full_context)
        assert 0.0 <= result.confidence_score <= 1.0

    def test_score_bounds_extreme_cases(self, basic_context, trending_ohlcv):
        """Test score bounds with extreme metric values."""
        # Excellent metrics
        basic_context.ohlcv = trending_ohlcv
        basic_context.backtest_metrics = {
            "sharpe": 5.0,
            "max_drawdown_pct": 0,
            "trades": 100,
        }
        basic_context.latest_candle_ts = basic_context.as_of_ts
        result = compute_confidence(basic_context)
        assert 0.0 <= result.confidence_score <= 1.0

        # Terrible metrics
        basic_context.backtest_metrics = {
            "sharpe": -2.0,
            "max_drawdown_pct": 80,
            "trades": 3,
        }
        result = compute_confidence(basic_context)
        assert 0.0 <= result.confidence_score <= 1.0

    def test_components_all_present(self, full_context):
        """Test that all expected components are in result."""
        result = compute_confidence(full_context)

        expected_components = [
            "performance",
            "drawdown",
            "stability",
            "data_freshness",
            "regime_fit",
        ]
        for component in expected_components:
            assert component in result.confidence_components

    def test_component_bounds(self, full_context):
        """Test that all components are within [0, 1]."""
        result = compute_confidence(full_context)

        for name, value in result.confidence_components.items():
            assert 0.0 <= value <= 1.0, f"Component {name} out of bounds: {value}"

    def test_deterministic_hashing(self, full_context):
        """Test that same inputs produce same hash."""
        result1 = compute_confidence(full_context)
        result2 = compute_confidence(full_context)

        assert result1.inputs_hash == result2.inputs_hash
        assert len(result1.inputs_hash) == 64  # SHA256 hex

    def test_different_inputs_different_hash(self, full_context):
        """Test that different inputs produce different hash."""
        result1 = compute_confidence(full_context)

        # Change something
        full_context.backtest_metrics["sharpe"] = 2.0
        result2 = compute_confidence(full_context)

        assert result1.inputs_hash != result2.inputs_hash

    def test_custom_weights(self, full_context):
        """Test that custom weights are applied."""
        # All weight on performance
        custom_weights = {
            "performance": 1.0,
            "drawdown": 0.0,
            "stability": 0.0,
            "data_freshness": 0.0,
            "regime_fit": 0.0,
        }

        result = compute_confidence(full_context, weights=custom_weights)

        # Score should equal performance component
        assert result.confidence_score == pytest.approx(
            result.confidence_components["performance"], abs=0.01
        )

    def test_explain_contains_breakdown(self, full_context):
        """Test that explain dict contains useful breakdown."""
        result = compute_confidence(full_context)

        assert "breakdown" in result.explain
        assert "summary" in result.explain
        assert "regime" in result.explain

        breakdown = result.explain["breakdown"]
        assert len(breakdown) > 0
        assert all("component" in item for item in breakdown)
        assert all("score" in item for item in breakdown)

    def test_features_contains_regime_info(self, full_context):
        """Test that features dict contains regime info."""
        result = compute_confidence(full_context)

        assert "regime" in result.features
        assert "metrics_source" in result.features

    def test_minimal_context(self, basic_context):
        """Test computation with minimal context (no data)."""
        result = compute_confidence(basic_context)

        # Should still produce valid result
        assert result.regime == "unknown"
        assert 0.0 <= result.confidence_score <= 1.0
        assert len(result.inputs_hash) == 64


class TestWeightNormalization:
    """Tests for weight normalization behavior."""

    def test_weights_sum_to_one(self):
        """Test default weights sum to 1."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_non_normalized_weights_handled(self, full_context):
        """Test that non-normalized weights are handled correctly."""
        # Weights that don't sum to 1
        custom_weights = {
            "performance": 2.0,
            "drawdown": 2.0,
            "stability": 2.0,
            "data_freshness": 2.0,
            "regime_fit": 2.0,
        }

        result = compute_confidence(full_context, weights=custom_weights)

        # Score should still be in [0, 1]
        assert 0.0 <= result.confidence_score <= 1.0
