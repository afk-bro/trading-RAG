"""Unit tests for regime computation."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from app.services.kb import (
    RegimeSnapshot,
    compute_regime_snapshot,
    compute_tags,
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
            assert not np.isnan(val), f"Found NaN in snapshot"


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
        snapshot = compute_regime_snapshot(
            flat_series, source="test", timeframe="1h"
        )
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
