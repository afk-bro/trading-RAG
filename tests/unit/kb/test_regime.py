"""Unit tests for regime computation."""

import pytest
import numpy as np
import pandas as pd

from app.services.kb import (
    RegimeSnapshot,
    compute_regime_snapshot,
    regime_snapshot_to_text,
    compute_atr,
    compute_rsi,
    compute_bollinger_bands,
    compute_bb_width_pct,
    compute_efficiency_ratio,
    compute_zscore,
    compute_trend_strength,
    compute_trend_direction,
    REGIME_SCHEMA_VERSION,
)
from app.services.kb.regime import (
    evaluate_rules,
    compute_tags,
    compute_tags_with_evidence,
    DEFAULT_RULESET,
)
from app.services.kb.types import TagEvidence


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def flat_series():
    """Create a flat price series (no trend, low volatility)."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    # Flat at 100 with tiny noise
    np.random.seed(42)
    close = 100 + np.random.normal(0, 0.1, 200)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.random.uniform(1000, 2000, 200),
        },
        index=dates,
    )


@pytest.fixture
def uptrend_series():
    """Create an uptrending price series."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    # Linear uptrend with noise
    np.random.seed(42)
    trend = np.linspace(100, 150, 200)
    noise = np.random.normal(0, 1, 200)
    close = trend + noise
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.random.uniform(1000, 2000, 200),
        },
        index=dates,
    )


@pytest.fixture
def high_volatility_series():
    """Create a high volatility series."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    np.random.seed(42)
    # High volatility with large swings
    close = 100 + np.cumsum(np.random.normal(0, 3, 200))
    return pd.DataFrame(
        {
            "open": close - 2,
            "high": close + 4,
            "low": close - 4,
            "close": close,
            "volume": np.random.uniform(1000, 2000, 200),
        },
        index=dates,
    )


# =============================================================================
# Indicator Tests
# =============================================================================


class TestATR:
    """Tests for ATR computation."""

    def test_atr_basic(self, flat_series):
        """ATR should be computed without errors."""
        atr = compute_atr(
            flat_series["high"],
            flat_series["low"],
            flat_series["close"],
            period=14,
        )
        assert len(atr) == len(flat_series)
        assert not atr.iloc[14:].isna().any()

    def test_atr_flat_series_low(self, flat_series):
        """Flat series should have low ATR."""
        atr = compute_atr(
            flat_series["high"],
            flat_series["low"],
            flat_series["close"],
            period=14,
        )
        # ATR should be very small relative to price
        atr_pct = atr.iloc[-1] / flat_series["close"].iloc[-1]
        assert atr_pct < 0.01  # Less than 1%

    def test_atr_high_vol_higher(self, high_volatility_series):
        """High volatility series should have higher ATR."""
        atr = compute_atr(
            high_volatility_series["high"],
            high_volatility_series["low"],
            high_volatility_series["close"],
            period=14,
        )
        atr_pct = atr.iloc[-1] / abs(high_volatility_series["close"].iloc[-1])
        assert atr_pct > 0.02  # More than 2%


class TestRSI:
    """Tests for RSI computation."""

    def test_rsi_basic(self, flat_series):
        """RSI should be computed without errors."""
        rsi = compute_rsi(flat_series["close"], period=14)
        assert len(rsi) == len(flat_series)
        # RSI should be between 0 and 100
        assert rsi.iloc[14:].between(0, 100).all()

    def test_rsi_flat_near_50(self, flat_series):
        """Flat series RSI should be near 50."""
        rsi = compute_rsi(flat_series["close"], period=14)
        assert 40 < rsi.iloc[-1] < 60

    def test_rsi_uptrend_above_50(self, uptrend_series):
        """Uptrend RSI should be above 50."""
        rsi = compute_rsi(uptrend_series["close"], period=14)
        assert rsi.iloc[-1] > 50


class TestBollingerBands:
    """Tests for Bollinger Bands computation."""

    def test_bb_basic(self, flat_series):
        """BB should be computed correctly."""
        middle, upper, lower = compute_bollinger_bands(
            flat_series["close"], period=20, k=2.0
        )
        assert len(middle) == len(flat_series)
        # Upper > Middle > Lower
        assert (upper.iloc[20:] >= middle.iloc[20:]).all()
        assert (middle.iloc[20:] >= lower.iloc[20:]).all()

    def test_bb_width_flat_narrow(self, flat_series):
        """Flat series should have narrow BB width."""
        width = compute_bb_width_pct(flat_series["close"], period=20, k=2.0)
        # Width should be small for flat series
        assert width.iloc[-1] < 0.05  # Less than 5%


class TestEfficiencyRatio:
    """Tests for Kaufman Efficiency Ratio."""

    def test_er_basic(self, flat_series):
        """ER should be computed correctly."""
        er = compute_efficiency_ratio(flat_series["close"], period=10)
        assert len(er) == len(flat_series)
        # ER should be between 0 and 1
        valid_er = er.iloc[10:].dropna()
        assert (valid_er >= 0).all()
        assert (valid_er <= 1).all()

    def test_er_uptrend_high(self, uptrend_series):
        """Strong uptrend should have higher ER than choppy series."""
        er_uptrend = compute_efficiency_ratio(uptrend_series["close"], period=10)
        # Compare to a choppy series - uptrend should have higher ER
        np.random.seed(42)
        choppy = 100 + np.random.normal(0, 1, 200)
        choppy_series = pd.Series(choppy)
        er_choppy = compute_efficiency_ratio(choppy_series, period=10)

        # Uptrend ER should be higher than choppy ER on average
        assert er_uptrend.mean() > er_choppy.mean()

    def test_er_choppy_low(self, flat_series):
        """Flat/choppy series should have low ER."""
        er = compute_efficiency_ratio(flat_series["close"], period=10)
        # Flat series is choppy, so ER should be low
        assert er.iloc[-1] < 0.5


class TestZScore:
    """Tests for Z-score computation."""

    def test_zscore_basic(self, flat_series):
        """Z-score should be computed correctly."""
        zscore = compute_zscore(flat_series["close"], period=20)
        assert len(zscore) == len(flat_series)

    def test_zscore_flat_near_zero(self, flat_series):
        """Flat series Z-score should be near 0."""
        zscore = compute_zscore(flat_series["close"], period=20)
        assert abs(zscore.iloc[-1]) < 2


class TestTrend:
    """Tests for trend strength and direction."""

    def test_trend_strength_uptrend(self, uptrend_series):
        """Uptrend should have high trend strength."""
        strength = compute_trend_strength(uptrend_series["close"], lookback=50)
        assert strength > 0.8  # R-squared should be high

    def test_trend_strength_flat(self, flat_series):
        """Flat series should have low trend strength."""
        strength = compute_trend_strength(flat_series["close"], lookback=50)
        assert strength < 0.3

    def test_trend_direction_uptrend(self, uptrend_series):
        """Uptrend should have positive direction."""
        direction = compute_trend_direction(uptrend_series["close"], lookback=50)
        assert direction == 1

    def test_trend_direction_flat(self, flat_series):
        """Flat series should have neutral direction."""
        direction = compute_trend_direction(flat_series["close"], lookback=50)
        assert direction == 0


# =============================================================================
# Tagging Tests
# =============================================================================


class TestTags:
    """Tests for regime tagging rules."""

    def test_tags_flat_series(self, flat_series):
        """Flat series should be tagged appropriately."""
        snapshot = compute_regime_snapshot(flat_series, source="test")
        tags = snapshot.regime_tags

        assert "flat" in tags
        assert "uptrend" not in tags
        assert "downtrend" not in tags

    def test_tags_uptrend_series(self, uptrend_series):
        """Uptrend series should be tagged as uptrend."""
        snapshot = compute_regime_snapshot(uptrend_series, source="test")
        tags = snapshot.regime_tags

        assert "uptrend" in tags
        assert "flat" not in tags

    def test_tags_high_vol_series(self, high_volatility_series):
        """High vol series should be tagged appropriately."""
        snapshot = compute_regime_snapshot(high_volatility_series, source="test")
        tags = snapshot.regime_tags

        assert "high_vol" in tags

    def test_tags_sorted(self, flat_series):
        """Tags should be alphabetically sorted."""
        snapshot = compute_regime_snapshot(flat_series, source="test")
        tags = snapshot.regime_tags
        assert tags == sorted(tags)


# =============================================================================
# RegimeSnapshot Tests
# =============================================================================


class TestRegimeSnapshot:
    """Tests for RegimeSnapshot creation and conversion."""

    def test_snapshot_creation(self, flat_series):
        """Snapshot should be created with all fields."""
        snapshot = compute_regime_snapshot(
            flat_series,
            source="test",
            instrument="BTCUSD",
            timeframe="1h",
        )

        assert snapshot.schema_version == REGIME_SCHEMA_VERSION
        assert snapshot.source == "test"
        assert snapshot.instrument == "BTCUSD"
        assert snapshot.timeframe == "1h"
        assert snapshot.n_bars == 200
        assert snapshot.computed_at is not None

    def test_snapshot_to_dict(self, flat_series):
        """Snapshot should convert to dict correctly."""
        snapshot = compute_regime_snapshot(flat_series, source="test")
        d = snapshot.to_dict()

        assert isinstance(d, dict)
        assert "schema_version" in d
        assert "atr_pct" in d
        assert "regime_tags" in d

    def test_snapshot_from_dict(self, flat_series):
        """Snapshot should reconstruct from dict."""
        original = compute_regime_snapshot(flat_series, source="test")
        d = original.to_dict()
        restored = RegimeSnapshot.from_dict(d)

        assert restored.schema_version == original.schema_version
        assert restored.atr_pct == original.atr_pct
        assert restored.regime_tags == original.regime_tags

    def test_snapshot_no_nan(self, flat_series):
        """Snapshot should not have NaN values in numeric fields."""
        snapshot = compute_regime_snapshot(flat_series, source="test")

        numeric_fields = [
            snapshot.atr_pct,
            snapshot.std_pct,
            snapshot.bb_width_pct,
            snapshot.range_pct,
            snapshot.trend_strength,
            snapshot.zscore,
            snapshot.rsi,
            snapshot.return_pct,
            snapshot.drift_bps_per_bar,
            snapshot.efficiency_ratio,
        ]

        for val in numeric_fields:
            assert not np.isnan(val), "Found NaN in snapshot"


# =============================================================================
# Text Template Tests
# =============================================================================


class TestTextTemplate:
    """Tests for text template generation."""

    def test_text_deterministic(self, flat_series):
        """Text output should be deterministic."""
        snapshot = compute_regime_snapshot(flat_series, source="test")

        text1 = regime_snapshot_to_text(snapshot)
        text2 = regime_snapshot_to_text(snapshot)

        assert text1 == text2

    def test_text_contains_tags(self, flat_series):
        """Text should contain regime tags."""
        snapshot = compute_regime_snapshot(flat_series, source="test")
        text = regime_snapshot_to_text(snapshot)

        for tag in snapshot.regime_tags:
            assert tag in text

    def test_text_contains_metrics(self, flat_series):
        """Text should contain key metrics."""
        snapshot = compute_regime_snapshot(flat_series, source="test")
        text = regime_snapshot_to_text(snapshot)

        assert "ATR" in text
        assert "BB width" in text
        assert "Trend" in text
        assert "z-score" in text
        assert "RSI" in text
        assert "Efficiency" in text

    def test_text_with_timeframe(self, flat_series):
        """Text should include timeframe when provided."""
        snapshot = compute_regime_snapshot(flat_series, source="test", timeframe="1h")
        text = regime_snapshot_to_text(snapshot)

        assert "(1h)" in text


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimal_data(self):
        """Should handle minimal data gracefully."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [101] * 50,
                "low": [99] * 50,
                "close": [100] * 50,
                "volume": [1000] * 50,
            },
            index=dates,
        )

        snapshot = compute_regime_snapshot(df, source="test")
        # With 50 bars (< 200 REGIME_WINDOW_BARS), computation proceeds but may have warnings
        # The snapshot should still be computed successfully
        assert snapshot is not None
        assert snapshot.n_bars == 50

    def test_zero_variance(self):
        """Should handle zero variance gracefully."""
        dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
        # Perfectly flat price
        df = pd.DataFrame(
            {
                "open": [100.0] * 200,
                "high": [100.0] * 200,
                "low": [100.0] * 200,
                "close": [100.0] * 200,
                "volume": [1000] * 200,
            },
            index=dates,
        )

        snapshot = compute_regime_snapshot(df, source="test")
        # Should not crash, may have warnings
        assert snapshot is not None

    def test_column_normalization(self):
        """Should handle different column naming."""
        dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
        np.random.seed(42)
        close = 100 + np.random.normal(0, 1, 200)

        # Different column names
        df = pd.DataFrame(
            {
                "Open": close - 0.5,  # Title case
                "HIGH": close + 1,  # Uppercase
                "low": close - 1,  # Lowercase
                "CLOSE": close,  # Uppercase
                "Vol": [1000] * 200,  # Alias
            },
            index=dates,
        )

        snapshot = compute_regime_snapshot(df, source="test")
        assert snapshot is not None
        assert snapshot.n_bars == 200


# =============================================================================
# NaN Cleaning Tests
# =============================================================================


class TestNaNCleaning:
    """Tests for NaN/Inf cleaning in JSON serialization."""

    def test_clean_nan_basic(self):
        """Should convert NaN to None."""
        from app.services.kb.types import clean_nan_for_json

        result = clean_nan_for_json(float("nan"))
        assert result is None

    def test_clean_inf_basic(self):
        """Should convert Inf to None."""
        from app.services.kb.types import clean_nan_for_json

        result = clean_nan_for_json(float("inf"))
        assert result is None
        result = clean_nan_for_json(float("-inf"))
        assert result is None

    def test_clean_normal_float(self):
        """Should preserve normal floats."""
        from app.services.kb.types import clean_nan_for_json

        result = clean_nan_for_json(3.14)
        assert result == 3.14

    def test_clean_dict_with_nan(self):
        """Should clean NaN values in dict."""
        from app.services.kb.types import clean_nan_for_json

        data = {"a": 1.0, "b": float("nan"), "c": "text"}
        result = clean_nan_for_json(data)

        assert result["a"] == 1.0
        assert result["b"] is None
        assert result["c"] == "text"

    def test_clean_list_with_nan(self):
        """Should clean NaN values in list."""
        from app.services.kb.types import clean_nan_for_json

        data = [1.0, float("nan"), float("inf"), 2.0]
        result = clean_nan_for_json(data)

        assert result == [1.0, None, None, 2.0]

    def test_clean_nested_structure(self):
        """Should clean NaN in nested structures."""
        from app.services.kb.types import clean_nan_for_json

        data = {
            "level1": {
                "level2": [
                    {"value": float("nan")},
                    {"value": 1.5},
                ],
            },
        }
        result = clean_nan_for_json(data)

        assert result["level1"]["level2"][0]["value"] is None
        assert result["level1"]["level2"][1]["value"] == 1.5

    def test_regime_snapshot_to_dict_cleans_nan(self):
        """RegimeSnapshot.to_dict() should clean NaN values."""
        snapshot = RegimeSnapshot(
            atr_pct=float("nan"),
            rsi=50.0,
            zscore=float("inf"),
        )
        data = snapshot.to_dict()

        assert data["atr_pct"] is None
        assert data["rsi"] == 50.0
        assert data["zscore"] is None


# =============================================================================
# Rule Evaluation Tests (v1.1)
# =============================================================================


class TestRuleEvaluation:
    """Tests for declarative rule evaluation."""

    def test_and_within_group_uptrend_requires_both(self):
        """Uptrend requires both strength AND direction rules to pass."""
        # Strength passes, direction fails → no uptrend
        features = {"trend_strength": 0.7, "trend_dir": 0}
        tags, _ = evaluate_rules(features)
        assert "uptrend" not in tags

    def test_and_within_group_uptrend_both_pass(self):
        """Uptrend is assigned when both rules pass."""
        features = {"trend_strength": 0.7, "trend_dir": 1}
        tags, _ = evaluate_rules(features)
        assert "uptrend" in tags

    def test_or_across_groups_oversold_zscore_only(self):
        """Oversold fires on zscore alone (OR across groups)."""
        features = {"zscore": -1.6, "rsi": 55, "trend_strength": 0.5}
        tags, _ = evaluate_rules(features)
        assert "oversold" in tags

    def test_or_across_groups_oversold_rsi_only(self):
        """Oversold fires on RSI alone (OR across groups)."""
        features = {"zscore": 0, "rsi": 25, "trend_strength": 0.5}
        tags, _ = evaluate_rules(features)
        assert "oversold" in tags

    def test_or_across_groups_overbought_zscore_only(self):
        """Overbought fires on zscore alone."""
        features = {"zscore": 1.6, "rsi": 55, "trend_strength": 0.5}
        tags, _ = evaluate_rules(features)
        assert "overbought" in tags

    def test_or_across_groups_overbought_rsi_only(self):
        """Overbought fires on RSI alone."""
        features = {"zscore": 0, "rsi": 75, "trend_strength": 0.5}
        tags, _ = evaluate_rules(features)
        assert "overbought" in tags

    def test_abs_transform_mean_reverting_negative_zscore(self):
        """Mean-reverting uses abs(zscore) for comparison."""
        # zscore = -1.2, abs(-1.2) = 1.2 > 1.0 threshold
        features = {"trend_strength": 0.2, "zscore": -1.2}
        tags, evidence = evaluate_rules(features)
        assert "mean_reverting" in tags

        # Verify transform is recorded in evidence
        mr_evidence = [e for e in evidence if e.rule_id == "mr_zscore"][0]
        assert mr_evidence.value == -1.2  # Raw value
        assert mr_evidence.computed_value == 1.2  # After abs()
        assert mr_evidence.transform == "abs"

    def test_abs_transform_mean_reverting_positive_zscore(self):
        """Mean-reverting works with positive zscore too."""
        features = {"trend_strength": 0.2, "zscore": 1.2}
        tags, _ = evaluate_rules(features)
        assert "mean_reverting" in tags

    def test_margin_positive_when_passed_ge(self):
        """Margin is positive when >= rule passes."""
        # trend_strength=0.7 >= 0.6, margin = 0.7 - 0.6 = 0.1
        features = {"trend_strength": 0.7, "trend_dir": 1}
        _, evidence = evaluate_rules(features)
        uptrend_strength = [e for e in evidence if e.rule_id == "uptrend_strength"][0]
        assert uptrend_strength.passed is True
        assert uptrend_strength.margin == pytest.approx(0.1)

    def test_margin_negative_when_failed_ge(self):
        """Margin is negative when >= rule fails."""
        # trend_strength=0.5 >= 0.6, margin = 0.5 - 0.6 = -0.1
        features = {"trend_strength": 0.5, "trend_dir": 1}
        _, evidence = evaluate_rules(features)
        uptrend_strength = [e for e in evidence if e.rule_id == "uptrend_strength"][0]
        assert uptrend_strength.passed is False
        assert uptrend_strength.margin == pytest.approx(-0.1)

    def test_margin_positive_when_passed_lt(self):
        """Margin is positive when < rule passes."""
        # zscore=-1.6 < -1.5, margin = -1.5 - (-1.6) = 0.1
        features = {"zscore": -1.6, "rsi": 55, "trend_strength": 0.5}
        _, evidence = evaluate_rules(features)
        oversold_zscore = [e for e in evidence if e.rule_id == "oversold_zscore"][0]
        assert oversold_zscore.passed is True
        assert oversold_zscore.margin == pytest.approx(0.1)

    def test_margin_negative_when_failed_lt(self):
        """Margin is negative when < rule fails."""
        # zscore=-1.4 < -1.5, margin = -1.5 - (-1.4) = -0.1
        features = {"zscore": -1.4, "rsi": 55, "trend_strength": 0.5}
        _, evidence = evaluate_rules(features)
        oversold_zscore = [e for e in evidence if e.rule_id == "oversold_zscore"][0]
        assert oversold_zscore.passed is False
        assert oversold_zscore.margin == pytest.approx(-0.1)


class TestExclusiveFamilies:
    """Tests for exclusive family logic."""

    def test_trend_family_uptrend_wins_over_flat(self):
        """Uptrend and flat can't both be assigned (uptrend has priority)."""
        # This shouldn't happen in practice (mutually exclusive thresholds),
        # but tests the priority logic
        features = {"trend_strength": 0.7, "trend_dir": 1}
        tags, _ = evaluate_rules(features)
        assert "uptrend" in tags
        assert "flat" not in tags

    def test_trend_family_downtrend_wins_over_flat(self):
        """Downtrend blocks flat."""
        features = {"trend_strength": 0.7, "trend_dir": -1}
        tags, _ = evaluate_rules(features)
        assert "downtrend" in tags
        assert "flat" not in tags

    def test_trend_family_only_one_assigned(self):
        """At most one trend tag is assigned."""
        # Uptrend case
        features = {"trend_strength": 0.7, "trend_dir": 1}
        tags, _ = evaluate_rules(features)
        trend_tags = [
            t for t in tags if t in ["uptrend", "downtrend", "trending", "flat"]
        ]
        assert len(trend_tags) == 1

    def test_middle_band_no_trend_tag(self):
        """Middle band (0.3-0.6) yields no trend tag."""
        features = {"trend_strength": 0.45, "trend_dir": 1}
        tags, _ = evaluate_rules(features)
        trend_tags = [
            t for t in tags if t in ["uptrend", "downtrend", "trending", "flat"]
        ]
        assert len(trend_tags) == 0

    def test_trending_neutral_direction(self):
        """Trending assigned when strength high but direction neutral."""
        features = {"trend_strength": 0.7, "trend_dir": 0}
        tags, _ = evaluate_rules(features)
        assert "trending" in tags
        assert "uptrend" not in tags
        assert "downtrend" not in tags


class TestLegacyBehaviorMatch:
    """Tests verifying new evaluate_rules matches legacy compute_tags."""

    @pytest.mark.parametrize(
        "kwargs,expected_tags",
        [
            # Basic trend cases
            ({"trend_strength": 0.7, "trend_dir": 1}, ["uptrend"]),
            ({"trend_strength": 0.7, "trend_dir": -1}, ["downtrend"]),
            ({"trend_strength": 0.7, "trend_dir": 0}, ["trending"]),
            ({"trend_strength": 0.2, "trend_dir": 0}, ["flat"]),
            # Volatility
            ({"trend_strength": 0.5, "atr_pct": 0.003}, ["low_vol"]),
            ({"trend_strength": 0.5, "atr_pct": 0.02}, ["high_vol"]),
            # Efficiency
            ({"trend_strength": 0.5, "efficiency_ratio": 0.2}, ["noisy"]),
            ({"trend_strength": 0.5, "efficiency_ratio": 0.8}, ["efficient"]),
            # Oscillator extremes
            ({"trend_strength": 0.5, "zscore": -1.6, "rsi": 50}, ["oversold"]),
            ({"trend_strength": 0.5, "zscore": 0, "rsi": 25}, ["oversold"]),
            ({"trend_strength": 0.5, "zscore": 1.6, "rsi": 50}, ["overbought"]),
            ({"trend_strength": 0.5, "zscore": 0, "rsi": 75}, ["overbought"]),
        ],
    )
    def test_matches_legacy(self, kwargs, expected_tags):
        """New evaluate_rules matches legacy compute_tags."""
        # Create snapshot with neutral defaults + test kwargs
        defaults = {
            "trend_strength": 0.5,
            "trend_dir": 0,
            "atr_pct": 0.01,
            "zscore": 0,
            "rsi": 50,
            "efficiency_ratio": 0.5,
            "bb_width_pct": 0.03,
        }
        defaults.update(kwargs)
        snapshot = RegimeSnapshot(**defaults)

        legacy_tags = compute_tags(snapshot)
        new_tags, _ = compute_tags_with_evidence(snapshot)

        # Both should contain the expected tags
        for tag in expected_tags:
            assert tag in legacy_tags, f"Legacy missing {tag}"
            assert tag in new_tags, f"New missing {tag}"

        # Tags should match exactly
        assert legacy_tags == new_tags


class TestTagEvidence:
    """Tests for tag evidence structure."""

    def test_evidence_includes_all_rules(self):
        """Evidence should include all evaluated rules."""
        features = {"trend_strength": 0.5, "trend_dir": 0}
        _, evidence = evaluate_rules(features)

        # Should have evidence for all rules in DEFAULT_RULESET
        rule_ids = {e.rule_id for e in evidence}
        expected_ids = {r.rule_id for r in DEFAULT_RULESET}
        assert rule_ids == expected_ids

    def test_evidence_structure_complete(self):
        """Each evidence should have complete structure."""
        features = {"trend_strength": 0.7, "trend_dir": 1}
        _, evidence = evaluate_rules(features)

        for ev in evidence:
            assert ev.tag is not None
            assert ev.rule_id is not None
            assert isinstance(ev.passed, bool)
            assert ev.metric is not None
            assert isinstance(ev.value, (int, float))  # trend_dir is int
            assert ev.op in (">=", "<=", ">", "<", "==")
            assert isinstance(ev.threshold, (int, float))
            assert ev.margin is not None

    def test_snapshot_has_tag_evidence(self, uptrend_series):
        """RegimeSnapshot should include tag_evidence after computation."""
        snapshot = compute_regime_snapshot(uptrend_series, source="test")

        assert hasattr(snapshot, "tag_evidence")
        assert len(snapshot.tag_evidence) > 0
        assert all(isinstance(e, TagEvidence) for e in snapshot.tag_evidence)

    def test_tag_evidence_serializes_correctly(self, uptrend_series):
        """tag_evidence should serialize and deserialize correctly."""
        original = compute_regime_snapshot(uptrend_series, source="test")
        data = original.to_dict()

        # tag_evidence should be in the dict
        assert "tag_evidence" in data
        assert isinstance(data["tag_evidence"], list)

        # Restore and verify
        restored = RegimeSnapshot.from_dict(data)
        assert len(restored.tag_evidence) == len(original.tag_evidence)

        # Compare first evidence item
        orig_ev = original.tag_evidence[0]
        rest_ev = restored.tag_evidence[0]
        assert orig_ev.tag == rest_ev.tag
        assert orig_ev.rule_id == rest_ev.rule_id
        assert orig_ev.passed == rest_ev.passed

    def test_old_snapshot_without_evidence_defaults_empty(self):
        """Old snapshots without tag_evidence should default to empty list."""
        # Simulate old v1 snapshot (no tag_evidence field)
        old_data = {
            "schema_version": "regime_v1",
            "trend_strength": 0.5,
            "trend_dir": 0,
            "regime_tags": ["flat"],
        }
        snapshot = RegimeSnapshot.from_dict(old_data)

        assert snapshot.tag_evidence == []


# =============================================================================
# Regime Key Computation Tests (Step 3)
# =============================================================================


from app.services.kb.regime import (
    compute_regime_key,
    compute_regime_fingerprint,
    extract_regime_tags_for_attribution,
    DEFAULT_RULESET_ID,
)


class TestRegimeKeyComputation:
    """Tests for regime key and fingerprint computation."""

    def test_regime_key_format_all_tags(self):
        """Regime key has correct format with all tags."""
        snapshot = RegimeSnapshot(
            schema_version="regime_v1_1",
            regime_tags=["uptrend", "high_vol", "efficient"],
        )
        key = compute_regime_key(snapshot)

        assert key == "regime_v1_1|default_v1|uptrend|high_vol|efficient"

    def test_regime_key_with_missing_tags(self):
        """Regime key uses underscore for missing dimensions."""
        # Only uptrend, no vol or efficiency
        snapshot = RegimeSnapshot(
            schema_version="regime_v1_1",
            regime_tags=["uptrend"],
        )
        key = compute_regime_key(snapshot)

        assert key == "regime_v1_1|default_v1|uptrend|_|_"

    def test_regime_key_no_tags(self):
        """Regime key with no tags uses underscores for all dimensions."""
        snapshot = RegimeSnapshot(
            schema_version="regime_v1_1",
            regime_tags=[],
        )
        key = compute_regime_key(snapshot)

        assert key == "regime_v1_1|default_v1|_|_|_"

    def test_regime_key_custom_ruleset(self):
        """Regime key respects custom ruleset_id."""
        snapshot = RegimeSnapshot(
            schema_version="regime_v1_1",
            regime_tags=["flat", "low_vol"],
        )
        key = compute_regime_key(snapshot, ruleset_id="custom_v2")

        assert key == "regime_v1_1|custom_v2|flat|low_vol|_"

    def test_regime_key_trend_priority(self):
        """Trend tag priority: uptrend > downtrend > trending > flat."""
        # If multiple trend tags somehow present, priority order applies
        snapshot = RegimeSnapshot(
            schema_version="regime_v1_1",
            regime_tags=["downtrend", "uptrend"],  # Both present
        )
        key = compute_regime_key(snapshot)

        # uptrend wins due to priority order
        assert "|uptrend|" in key


class TestRegimeFingerprint:
    """Tests for regime fingerprint computation."""

    def test_fingerprint_stability(self):
        """Same key produces same fingerprint."""
        key = "regime_v1_1|default_v1|uptrend|high_vol|noisy"

        fp1 = compute_regime_fingerprint(key)
        fp2 = compute_regime_fingerprint(key)

        assert fp1 == fp2

    def test_fingerprint_is_sha256(self):
        """Fingerprint is a valid SHA256 hash (64 hex chars)."""
        key = "regime_v1_1|default_v1|flat|low_vol|efficient"
        fingerprint = compute_regime_fingerprint(key)

        assert len(fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in fingerprint)

    def test_fingerprint_changes_with_schema(self):
        """Different schema versions produce different fingerprints."""
        key_v1 = "regime_v1|default_v1|uptrend|high_vol|noisy"
        key_v2 = "regime_v1_1|default_v1|uptrend|high_vol|noisy"

        fp1 = compute_regime_fingerprint(key_v1)
        fp2 = compute_regime_fingerprint(key_v2)

        assert fp1 != fp2

    def test_fingerprint_changes_with_ruleset(self):
        """Different rulesets produce different fingerprints."""
        key_v1 = "regime_v1_1|default_v1|uptrend|_|_"
        key_v2 = "regime_v1_1|custom_v2|uptrend|_|_"

        fp1 = compute_regime_fingerprint(key_v1)
        fp2 = compute_regime_fingerprint(key_v2)

        assert fp1 != fp2


class TestRegimeTagExtraction:
    """Tests for denormalized tag extraction."""

    def test_extract_all_tags(self):
        """Extract all three tag dimensions."""
        snapshot = RegimeSnapshot(
            regime_tags=["downtrend", "low_vol", "noisy"],
        )
        trend, vol, eff = extract_regime_tags_for_attribution(snapshot)

        assert trend == "downtrend"
        assert vol == "low_vol"
        assert eff == "noisy"

    def test_extract_partial_tags(self):
        """Extract with some dimensions missing."""
        snapshot = RegimeSnapshot(
            regime_tags=["flat"],  # Only trend
        )
        trend, vol, eff = extract_regime_tags_for_attribution(snapshot)

        assert trend == "flat"
        assert vol is None
        assert eff is None

    def test_extract_no_tags(self):
        """Extract with no tags returns all None."""
        snapshot = RegimeSnapshot(regime_tags=[])
        trend, vol, eff = extract_regime_tags_for_attribution(snapshot)

        assert trend is None
        assert vol is None
        assert eff is None

    def test_extract_ignores_other_tags(self):
        """Extraction ignores non-dimension tags like oversold."""
        snapshot = RegimeSnapshot(
            regime_tags=["uptrend", "oversold", "mean_reverting", "efficient"],
        )
        trend, vol, eff = extract_regime_tags_for_attribution(snapshot)

        assert trend == "uptrend"
        assert vol is None  # oversold is not a vol tag
        assert eff == "efficient"

    def test_denormalized_tags_match_regime_key(self):
        """Denormalized tags should match what's in regime_key."""
        snapshot = RegimeSnapshot(
            schema_version="regime_v1_1",
            regime_tags=["trending", "high_vol", "noisy"],
        )
        key = compute_regime_key(snapshot)
        trend, vol, eff = extract_regime_tags_for_attribution(snapshot)

        # Key format: {schema}|{ruleset}|{trend}|{vol}|{eff}
        parts = key.split("|")
        assert parts[2] == (trend or "_")
        assert parts[3] == (vol or "_")
        assert parts[4] == (eff or "_")
