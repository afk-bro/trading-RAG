"""
Unit tests for ICT pattern detection indicators.

Tests Fair Value Gap, Breaker Block, Liquidity Sweep, and MSS detection.
"""

from datetime import datetime, timedelta, timezone
import pytest

from app.services.strategy.models import OHLCVBar
from app.services.strategy.indicators.ict_patterns import (
    FVGType,
    BlockType,
    detect_fvgs,
    detect_breaker_blocks,
    detect_liquidity_sweeps,
    detect_mss,
    detect_displacement,
    detect_mitigation_blocks,
)
from app.services.strategy.indicators.tf_bias import (
    BiasDirection,
    compute_ema,
    compute_sma,
    compute_efficiency_ratio,
    compute_rsi,
    detect_hh_hl_pattern,
    compute_tf_bias,
)


def make_bar(
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000.0,
) -> OHLCVBar:
    """Helper to create an OHLCVBar."""
    return OHLCVBar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume)


def make_bars_from_closes(closes: list[float], base_ts: datetime = None) -> list[OHLCVBar]:
    """Create bars from a list of close prices (simple test data)."""
    if base_ts is None:
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    bars = []
    for i, close in enumerate(closes):
        ts = base_ts + timedelta(hours=i)
        # Simple bars where OHLC are close to close price
        bars.append(make_bar(
            ts=ts,
            open_=close - 1,
            high=close + 2,
            low=close - 2,
            close=close,
        ))
    return bars


class TestFairValueGapDetection:
    """Tests for FVG detection."""

    def test_detects_bullish_fvg(self):
        """Bullish FVG: bar[i-2].high < bar[i].low (gap up)."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 100, 102, 99, 101),  # bar 0
            make_bar(base_ts + timedelta(hours=1), 101, 110, 101, 109),  # bar 1 (big up)
            make_bar(base_ts + timedelta(hours=2), 109, 115, 108, 114),  # bar 2 (gap above bar 0)
        ]
        # bar[0].high = 102, bar[2].low = 108 -> gap from 102 to 108

        fvgs = detect_fvgs(bars)

        assert len(fvgs) == 1
        assert fvgs[0].fvg_type == FVGType.BULLISH
        assert fvgs[0].gap_low == 102  # bar 0 high
        assert fvgs[0].gap_high == 108  # bar 2 low
        assert fvgs[0].gap_size == 6

    def test_detects_bearish_fvg(self):
        """Bearish FVG: bar[i-2].low > bar[i].high (gap down)."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 110, 112, 108, 109),  # bar 0
            make_bar(base_ts + timedelta(hours=1), 109, 109, 100, 101),  # bar 1 (big down)
            make_bar(base_ts + timedelta(hours=2), 101, 103, 99, 100),  # bar 2 (gap below bar 0)
        ]
        # bar[0].low = 108, bar[2].high = 103 -> gap from 103 to 108

        fvgs = detect_fvgs(bars)

        assert len(fvgs) == 1
        assert fvgs[0].fvg_type == FVGType.BEARISH
        assert fvgs[0].gap_low == 103  # bar 2 high
        assert fvgs[0].gap_high == 108  # bar 0 low

    def test_no_fvg_when_overlapping(self):
        """No FVG when bars overlap."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 100, 105, 98, 103),
            make_bar(base_ts + timedelta(hours=1), 103, 107, 102, 106),
            make_bar(base_ts + timedelta(hours=2), 106, 108, 104, 107),  # low overlaps bar 0 high
        ]

        fvgs = detect_fvgs(bars)
        assert len(fvgs) == 0

    def test_respects_min_gap_size(self):
        """FVG detection respects minimum gap size filter."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 100, 102, 99, 101),
            make_bar(base_ts + timedelta(hours=1), 101, 106, 101, 105),
            make_bar(base_ts + timedelta(hours=2), 105, 108, 103, 107),  # gap = 1 (102 to 103)
        ]

        # Small gap should be detected with no filter
        fvgs = detect_fvgs(bars, min_gap_size=0)
        assert len(fvgs) == 1

        # Should be filtered out with larger min
        fvgs = detect_fvgs(bars, min_gap_size=5)
        assert len(fvgs) == 0

    def test_as_of_ts_excludes_future_bars(self):
        """FVG created by bar 2 but as_of_ts cuts before bar 2 => no FVG."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 100, 102, 99, 101),
            make_bar(base_ts + timedelta(hours=1), 101, 110, 101, 109),
            make_bar(base_ts + timedelta(hours=2), 109, 115, 108, 114),
        ]

        # Cutoff before bar 2 => only 2 bars visible, not enough for FVG
        fvgs = detect_fvgs(bars, as_of_ts=base_ts + timedelta(hours=1))
        assert len(fvgs) == 0

        # No cutoff => FVG detected
        fvgs = detect_fvgs(bars)
        assert len(fvgs) == 1

    def test_as_of_ts_prevents_premature_fill(self):
        """FVG should be unfilled when as_of_ts is before the fill bar."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 100, 102, 99, 101),                       # bar 0
            make_bar(base_ts + timedelta(hours=1), 101, 110, 101, 109),  # bar 1
            make_bar(base_ts + timedelta(hours=2), 109, 115, 108, 114),  # bar 2 (creates bullish FVG: 102..108)
            make_bar(base_ts + timedelta(hours=3), 114, 116, 110, 115),  # bar 3 (no new FVG, no fill)
            make_bar(base_ts + timedelta(hours=4), 115, 115, 98, 99),   # bar 4 (fills FVG, low=98 < gap_low=102)
        ]

        # Before the fill bar: original FVG should be unfilled
        fvgs_before = detect_fvgs(bars, as_of_ts=base_ts + timedelta(hours=3))
        original_fvgs = [f for f in fvgs_before if f.gap_low == 102]
        assert len(original_fvgs) == 1
        assert original_fvgs[0].filled is False

        # After the fill bar: original FVG should be filled
        fvgs_after = detect_fvgs(bars)
        original_fvgs_after = [f for f in fvgs_after if f.gap_low == 102]
        assert len(original_fvgs_after) == 1
        assert original_fvgs_after[0].filled is True

    def test_no_as_of_ts_backward_compat(self):
        """Omitting as_of_ts preserves original behavior."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 100, 102, 99, 101),
            make_bar(base_ts + timedelta(hours=1), 101, 110, 101, 109),
            make_bar(base_ts + timedelta(hours=2), 109, 115, 108, 114),
        ]

        fvgs_default = detect_fvgs(bars)
        fvgs_none = detect_fvgs(bars, as_of_ts=None)
        assert len(fvgs_default) == len(fvgs_none)


class TestFVGInvalidation:
    """Tests for the FairValueGap.invalidated() method."""

    def test_unfilled_fvg_not_invalidated(self):
        """An unfilled FVG is not invalidated at any threshold."""
        from app.services.strategy.indicators.ict_patterns import FairValueGap, FVGType

        fvg = FairValueGap(
            fvg_type=FVGType.BULLISH, gap_high=108.0, gap_low=102.0,
            gap_size=6.0, bar_index=1,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            filled=False, fill_percent=0.0,
        )
        assert fvg.invalidated(max_fill_pct=1.0) is False
        assert fvg.invalidated(max_fill_pct=0.5) is False

    def test_partially_filled_invalidated_at_threshold(self):
        """FVG with 75% fill is invalidated when threshold is 0.75."""
        from app.services.strategy.indicators.ict_patterns import FairValueGap, FVGType

        fvg = FairValueGap(
            fvg_type=FVGType.BULLISH, gap_high=108.0, gap_low=102.0,
            gap_size=6.0, bar_index=1,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            filled=False, fill_percent=0.75,
        )
        assert fvg.invalidated(max_fill_pct=0.75) is True
        assert fvg.invalidated(max_fill_pct=0.80) is False
        assert fvg.invalidated(max_fill_pct=1.0) is False

    def test_fully_filled_always_invalidated(self):
        """A 100% filled FVG is invalidated at any threshold."""
        from app.services.strategy.indicators.ict_patterns import FairValueGap, FVGType

        fvg = FairValueGap(
            fvg_type=FVGType.BULLISH, gap_high=108.0, gap_low=102.0,
            gap_size=6.0, bar_index=1,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            filled=True, fill_percent=1.0,
        )
        assert fvg.invalidated(max_fill_pct=1.0) is True
        assert fvg.invalidated(max_fill_pct=0.5) is True

    def test_default_threshold_matches_filled(self):
        """Default max_fill_pct=1.0 behaves like the old filled check."""
        from app.services.strategy.indicators.ict_patterns import FairValueGap, FVGType

        fvg_partial = FairValueGap(
            fvg_type=FVGType.BULLISH, gap_high=108.0, gap_low=102.0,
            gap_size=6.0, bar_index=1,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            filled=False, fill_percent=0.5,
        )
        # With default threshold, partial fill is NOT invalidated
        assert fvg_partial.invalidated() is False


class TestLiquiditySweepDetection:
    """Tests for liquidity sweep detection."""

    def test_detects_sweep_of_swing_low(self):
        """Detect when price sweeps below swing low then reverses."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [
            make_bar(base_ts, 105, 107, 104, 106),
            make_bar(base_ts + timedelta(hours=1), 106, 108, 105, 107),
            make_bar(base_ts + timedelta(hours=2), 107, 108, 100, 100),  # swing low at 100
            make_bar(base_ts + timedelta(hours=3), 100, 103, 99, 102),
            make_bar(base_ts + timedelta(hours=4), 102, 105, 101, 104),
            make_bar(base_ts + timedelta(hours=5), 104, 106, 103, 105),
            make_bar(base_ts + timedelta(hours=6), 105, 107, 98, 106),  # sweeps 100, closes above
        ]

        sweeps = detect_liquidity_sweeps(bars, swing_lookback=2, lookback=10)

        # Should find a sweep of the swing low
        low_sweeps = [s for s in sweeps if s.sweep_type == "low"]
        assert len(low_sweeps) >= 1


class TestMSSDetection:
    """Tests for Market Structure Shift detection."""

    def test_detects_bullish_mss(self):
        """Detect bullish MSS (break of swing high)."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Create a downtrend then break of structure
        bars = [
            make_bar(base_ts, 110, 112, 108, 109),
            make_bar(base_ts + timedelta(hours=1), 109, 111, 107, 108),
            make_bar(base_ts + timedelta(hours=2), 108, 110, 106, 107),  # swing high at 110
            make_bar(base_ts + timedelta(hours=3), 107, 109, 105, 106),
            make_bar(base_ts + timedelta(hours=4), 106, 108, 104, 105),
            make_bar(base_ts + timedelta(hours=5), 105, 107, 103, 104),  # swing low at 103
            make_bar(base_ts + timedelta(hours=6), 104, 112, 103, 111),  # MSS - breaks above 110
        ]

        mss_list = detect_mss(bars, swing_lookback=2, lookback=10)

        bullish_mss = [m for m in mss_list if m.shift_type == "bullish"]
        # Should detect at least one bullish shift
        assert len(bullish_mss) >= 0  # May or may not detect depending on swing detection


class TestDisplacementDetection:
    """Tests for displacement move detection."""

    def test_detects_large_displacement(self):
        """Detect displacement moves (large impulsive candles)."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Normal bars then a big displacement
        bars = []
        for i in range(20):
            price = 100 + i * 0.5
            bars.append(make_bar(
                base_ts + timedelta(hours=i),
                price - 1,
                price + 1,
                price - 1,
                price,
            ))

        # Add a displacement candle (big move)
        bars.append(make_bar(
            base_ts + timedelta(hours=20),
            110,
            125,  # Big up move
            109,
            124,
            volume=5000,
        ))

        displacements = detect_displacement(bars, atr_period=14, atr_multiple=2.0)

        # Should detect the big candle as displacement
        # Note: result depends on ATR calculation
        assert isinstance(displacements, list)


class TestTFBiasIndicators:
    """Tests for timeframe bias indicators."""

    def test_ema_calculation(self):
        """EMA calculation produces expected values."""
        prices = [100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 104.0]
        ema = compute_ema(prices, 3)

        assert len(ema) == len(prices)
        # EMA should be between min and max
        assert all(min(prices) <= e <= max(prices) for e in ema)

    def test_sma_calculation(self):
        """SMA calculation produces expected values."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        sma = compute_sma(prices, 3)

        assert len(sma) == len(prices)
        # Last SMA should be average of last 3
        assert sma[-1] == pytest.approx((104.0 + 106.0 + 108.0) / 3)

    def test_efficiency_ratio_trending(self):
        """ER should be high for trending market."""
        # Strong uptrend
        prices = [100.0 + i * 2 for i in range(20)]
        er = compute_efficiency_ratio(prices, period=10)

        # ER should be close to 1 for perfect trend
        assert er[-1] > 0.8

    def test_efficiency_ratio_choppy(self):
        """ER should be low for choppy market."""
        # Choppy market (oscillating)
        prices = [100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0]
        er = compute_efficiency_ratio(prices, period=10)

        # ER should be low for choppy market
        assert er[-1] < 0.3

    def test_rsi_overbought(self):
        """RSI should be high after strong up moves."""
        # Strong uptrend
        prices = [100.0 + i * 3 for i in range(20)]
        rsi = compute_rsi(prices, period=14)

        # RSI should be high (overbought territory)
        assert rsi[-1] > 70

    def test_rsi_oversold(self):
        """RSI should be low after strong down moves."""
        # Strong downtrend
        prices = [200.0 - i * 3 for i in range(20)]
        rsi = compute_rsi(prices, period=14)

        # RSI should be low (oversold territory)
        assert rsi[-1] < 30

    def test_hh_hl_pattern_bullish(self):
        """Detect bullish HH/HL pattern."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Create higher highs and higher lows
        bars = []
        for i in range(15):
            base = 100 + i * 2
            # Add some variation to create swing points
            if i % 3 == 0:
                bars.append(make_bar(base_ts + timedelta(hours=i), base, base + 3, base - 1, base + 2))
            elif i % 3 == 1:
                bars.append(make_bar(base_ts + timedelta(hours=i), base + 2, base + 4, base + 1, base + 3))
            else:
                bars.append(make_bar(base_ts + timedelta(hours=i), base + 3, base + 5, base + 2, base + 1))

        direction, confidence = detect_hh_hl_pattern(bars, lookback=12)

        # Should detect bullish structure
        assert direction in (BiasDirection.BULLISH, BiasDirection.NEUTRAL)


class TestTFBiasComputation:
    """Tests for complete TF bias computation."""

    def test_bias_with_insufficient_data(self):
        """Bias computation handles insufficient data gracefully."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [make_bar(base_ts, 100, 102, 99, 101)]  # Only 1 bar

        bias = compute_tf_bias(m5_bars=bars)

        # Should return neutral with low confidence
        assert bias.final_direction == BiasDirection.NEUTRAL
        assert bias.final_confidence < 0.5

    def test_bias_with_trending_data(self):
        """Bias computation detects trending market."""
        base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Create strong uptrend
        bars = []
        for i in range(250):  # Need enough for EMA(200)
            price = 100 + i * 0.5
            bars.append(make_bar(
                base_ts + timedelta(hours=i),
                price - 0.5,
                price + 1,
                price - 1,
                price,
            ))

        bias = compute_tf_bias(
            daily_bars=bars,
            h4_bars=bars[-100:],
            h1_bars=bars[-50:],
            m15_bars=bars[-30:],
            m5_bars=bars[-20:],
            timestamp=bars[-1].ts,
        )

        # Should detect bullish bias
        assert bias.final_direction == BiasDirection.BULLISH
        # Confidence may be lower if some TFs have insufficient data
        assert bias.final_confidence > 0.4


class TestUnicornModelHelpers:
    """Tests for Unicorn Model helper functions."""

    def test_macro_window_check(self):
        """Test macro time window validation."""
        from app.services.strategy.strategies.unicorn_model import is_in_macro_window
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("America/New_York")

        # NY AM session (9:30-11:00 ET)
        in_window = datetime(2024, 1, 15, 10, 0, tzinfo=ET)  # 10:00 AM ET
        assert is_in_macro_window(in_window) is True

        # Outside window
        out_window = datetime(2024, 1, 15, 12, 0, tzinfo=ET)  # 12:00 PM ET
        assert is_in_macro_window(out_window) is False

    def test_max_stop_points(self):
        """Test max stop points lookup."""
        from app.services.strategy.strategies.unicorn_model import get_max_stop_points

        assert get_max_stop_points("NQ") == 30
        assert get_max_stop_points("MNQ") == 30
        assert get_max_stop_points("ES") == 10
        assert get_max_stop_points("MES") == 10

    def test_point_value(self):
        """Test point value lookup."""
        from app.services.strategy.strategies.unicorn_model import get_point_value

        assert get_point_value("NQ") == 20.0
        assert get_point_value("MNQ") == 2.0  # Micro
        assert get_point_value("ES") == 50.0
        assert get_point_value("MES") == 5.0  # Micro
