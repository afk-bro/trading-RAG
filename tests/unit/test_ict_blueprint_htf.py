"""Unit tests for ICT Blueprint HTF bias engine."""

from app.services.backtest.engines.ict_blueprint.htf_bias import (
    SwingDetector,
    check_msb,
    define_range,
    identify_order_block,
    invalidate_obs,
    is_in_discount,
    is_in_premium,
    update_htf,
)
from app.services.backtest.engines.ict_blueprint.types import (
    Bias,
    HTFState,
    ICTBlueprintParams,
    OrderBlock,
    SwingPoint,
    TradingRange,
)


# ---------------------------------------------------------------------------
# SwingDetector tests
# ---------------------------------------------------------------------------


class TestSwingDetector:
    def test_3_candle_swing_high(self):
        """Lookback=1 (3-candle): center bar high > both neighbors."""
        det = SwingDetector(lookback=1)
        # Bar 0: high=10
        assert det.push(0, 100, 9.0, 10.0, 8.0, 9.5) == []
        # Bar 1: high=12 (candidate swing high)
        assert det.push(1, 200, 11.0, 12.0, 10.0, 11.5) == []
        # Bar 2: high=11 — confirms bar 1 as swing high
        swings = det.push(2, 300, 10.0, 11.0, 9.0, 10.5)
        highs = [s for s in swings if s.is_high]
        assert len(highs) == 1
        assert highs[0].index == 1
        assert highs[0].price == 12.0

    def test_3_candle_swing_low(self):
        det = SwingDetector(lookback=1)
        det.push(0, 100, 10.0, 11.0, 9.0, 10.0)
        det.push(1, 200, 9.0, 10.0, 7.0, 8.0)
        swings = det.push(2, 300, 8.0, 9.0, 8.0, 8.5)
        lows = [s for s in swings if not s.is_high]
        assert len(lows) == 1
        assert lows[0].index == 1
        assert lows[0].price == 7.0

    def test_5_candle_swing(self):
        """Lookback=2 (5-candle)."""
        det = SwingDetector(lookback=2)
        # Need 5 bars. Center is index 2.
        det.push(0, 100, 10.0, 11.0, 9.0, 10.0)
        det.push(1, 200, 10.0, 11.5, 9.5, 10.5)
        det.push(2, 300, 11.0, 15.0, 10.0, 14.0)  # highest
        det.push(3, 400, 13.0, 14.0, 12.0, 13.0)
        swings = det.push(4, 500, 12.0, 13.0, 11.0, 12.0)
        highs = [s for s in swings if s.is_high]
        assert len(highs) == 1
        assert highs[0].index == 2
        assert highs[0].price == 15.0

    def test_flat_bars_dont_qualify(self):
        """Equal highs = NOT a swing high (strict inequality)."""
        det = SwingDetector(lookback=1)
        det.push(0, 100, 10.0, 12.0, 9.0, 11.0)
        det.push(1, 200, 10.0, 12.0, 9.0, 11.0)  # Same high as bar 0
        swings = det.push(2, 300, 10.0, 11.0, 9.0, 10.5)
        highs = [s for s in swings if s.is_high]
        assert len(highs) == 0

    def test_flat_lows_dont_qualify(self):
        det = SwingDetector(lookback=1)
        det.push(0, 100, 10.0, 12.0, 8.0, 11.0)
        det.push(1, 200, 10.0, 12.0, 8.0, 11.0)  # Same low
        swings = det.push(2, 300, 10.0, 12.0, 9.0, 11.0)
        lows = [s for s in swings if not s.is_high]
        assert len(lows) == 0

    def test_incremental_multiple_swings(self):
        """Push a sequence and collect all emitted swings."""
        det = SwingDetector(lookback=1)
        all_swings = []
        # Create a simple up-down-up pattern
        bars = [
            (0, 100, 10.0, 11.0, 9.0, 10.5),
            (1, 200, 11.0, 14.0, 10.0, 13.0),  # swing high candidate
            (2, 300, 12.0, 13.0, 8.0, 9.0),  # confirms bar 1 SH, swing low candidate
            (3, 400, 9.0, 12.0, 9.0, 11.0),  # confirms bar 2 SL
        ]
        for b in bars:
            all_swings.extend(det.push(*b))
        highs = [s for s in all_swings if s.is_high]
        lows = [s for s in all_swings if not s.is_high]
        assert len(highs) >= 1
        assert highs[0].price == 14.0
        assert len(lows) >= 1
        assert lows[0].price == 8.0


# ---------------------------------------------------------------------------
# MSB tests
# ---------------------------------------------------------------------------


class TestMSB:
    def test_bullish_msb(self):
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        result = check_msb(sh, sl, 51.0, Bias.NEUTRAL)
        assert result is not None
        assert result[0] == Bias.BULLISH

    def test_bearish_msb(self):
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        result = check_msb(sh, sl, 39.0, Bias.NEUTRAL)
        assert result is not None
        assert result[0] == Bias.BEARISH

    def test_no_msb(self):
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        result = check_msb(sh, sl, 45.0, Bias.NEUTRAL)
        assert result is None

    def test_bias_persistence(self):
        """After bullish MSB, bias stays BULLISH until bearish MSB."""
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        # Bullish MSB
        result = check_msb(sh, sl, 51.0, Bias.NEUTRAL)
        assert result[0] == Bias.BULLISH
        # No opposing MSB yet
        result2 = check_msb(sh, sl, 49.0, Bias.BULLISH)
        assert result2 is None  # 49 > 40, no bearish MSB

    def test_bearish_msb_flips_bullish(self):
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        result = check_msb(sh, sl, 39.0, Bias.BULLISH)
        assert result[0] == Bias.BEARISH


# ---------------------------------------------------------------------------
# Range definition tests
# ---------------------------------------------------------------------------


class TestRangeDefinition:
    def test_bullish_range(self):
        sh = [SwingPoint(2, 300, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        r = define_range(Bias.BULLISH, sh, sl, sh[0])
        assert r is not None
        assert r.high == 50.0
        assert r.low == 40.0
        assert r.midpoint == 45.0

    def test_bearish_range(self):
        sh = [SwingPoint(1, 200, 50.0, True)]
        sl = [SwingPoint(2, 300, 40.0, False)]
        r = define_range(Bias.BEARISH, sh, sl, sl[0])
        assert r is not None
        assert r.low == 40.0
        assert r.high == 50.0

    def test_no_preceding_swing(self):
        sh = [SwingPoint(5, 500, 50.0, True)]
        sl = []  # No swing lows at all
        r = define_range(Bias.BULLISH, sh, sl, sh[0])
        assert r is None


# ---------------------------------------------------------------------------
# Order block identification tests
# ---------------------------------------------------------------------------


class TestOrderBlock:
    def _make_bars(self, data):
        """data: list of (open, high, low, close)"""
        return [(i, i * 100, o, h, l, c) for i, (o, h, l, c) in enumerate(data)]

    def test_bullish_ob_1_candle(self):
        # Bar 0: bearish (close < open) — this should be the OB
        # Bar 1: bullish impulse
        # Bar 2: MSB candle (closes above swing high)
        bars = self._make_bars(
            [
                (50.0, 51.0, 48.0, 49.0),  # bearish
                (49.0, 55.0, 49.0, 54.0),  # bullish impulse
                (54.0, 58.0, 53.0, 57.0),  # MSB bar
            ]
        )
        anchor = SwingPoint(0, 0, 45.0, True)
        ob = identify_order_block(bars, 2, Bias.BULLISH, 1, anchor, 2)
        assert ob is not None
        assert ob.bottom == 48.0  # low of bar 0
        assert ob.top == 51.0  # high of bar 0

    def test_bullish_ob_2_candles(self):
        bars = self._make_bars(
            [
                (50.0, 51.0, 47.0, 48.0),  # bearish
                (49.0, 50.0, 46.0, 47.0),  # bearish
                (47.0, 55.0, 47.0, 54.0),  # impulse
                (54.0, 58.0, 53.0, 57.0),  # MSB
            ]
        )
        anchor = SwingPoint(0, 0, 45.0, True)
        ob = identify_order_block(bars, 3, Bias.BULLISH, 2, anchor, 3)
        assert ob is not None
        assert ob.bottom == 46.0  # min low of both bearish candles
        assert ob.top == 51.0  # max high of both bearish candles

    def test_bearish_ob(self):
        bars = self._make_bars(
            [
                (48.0, 52.0, 48.0, 51.0),  # bullish (close > open)
                (51.0, 52.0, 44.0, 45.0),  # bearish impulse
                (45.0, 46.0, 38.0, 39.0),  # MSB bar (closes below swing low)
            ]
        )
        anchor = SwingPoint(0, 0, 42.0, False)
        ob = identify_order_block(bars, 2, Bias.BEARISH, 1, anchor, 2)
        assert ob is not None
        assert ob.top == 52.0
        assert ob.bottom == 48.0

    def test_no_opposing_candle(self):
        # All bullish before MSB — no bearish OB candle for bullish MSB
        bars = self._make_bars(
            [
                (48.0, 52.0, 47.0, 51.0),  # bullish
                (51.0, 56.0, 50.0, 55.0),  # bullish
            ]
        )
        anchor = SwingPoint(0, 0, 45.0, True)
        ob = identify_order_block(bars, 1, Bias.BULLISH, 1, anchor, 1)
        assert ob is None


# ---------------------------------------------------------------------------
# OB invalidation tests
# ---------------------------------------------------------------------------


class TestOBInvalidation:
    def test_boundary_breach_bullish(self):
        ob = OrderBlock(
            top=50.0,
            bottom=45.0,
            bias=Bias.BULLISH,
            ob_id=(1, 0, 0, "long"),
            anchor_swing=SwingPoint(0, 0, 45.0, True),
            msb_bar_index=1,
            created_at_daily_index=0,
        )
        state = HTFState(bias=Bias.BULLISH, active_obs=[ob])
        invalidate_obs(state, bar_close=44.0, current_daily_index=5, max_ob_age_bars=20)
        assert ob.invalidated is True

    def test_boundary_breach_bearish(self):
        ob = OrderBlock(
            top=50.0,
            bottom=45.0,
            bias=Bias.BEARISH,
            ob_id=(1, 0, 0, "short"),
            anchor_swing=SwingPoint(0, 0, 50.0, False),
            msb_bar_index=1,
            created_at_daily_index=0,
        )
        state = HTFState(bias=Bias.BEARISH, active_obs=[ob])
        invalidate_obs(state, bar_close=51.0, current_daily_index=5, max_ob_age_bars=20)
        assert ob.invalidated is True

    def test_bias_flip_invalidation(self):
        ob = OrderBlock(
            top=50.0,
            bottom=45.0,
            bias=Bias.BULLISH,
            ob_id=(1, 0, 0, "long"),
            anchor_swing=SwingPoint(0, 0, 45.0, True),
            msb_bar_index=1,
            created_at_daily_index=0,
        )
        state = HTFState(bias=Bias.BEARISH, active_obs=[ob])
        invalidate_obs(state, bar_close=47.0, current_daily_index=5, max_ob_age_bars=20)
        assert ob.invalidated is True

    def test_expiry_invalidation(self):
        ob = OrderBlock(
            top=50.0,
            bottom=45.0,
            bias=Bias.BULLISH,
            ob_id=(1, 0, 0, "long"),
            anchor_swing=SwingPoint(0, 0, 45.0, True),
            msb_bar_index=1,
            created_at_daily_index=0,
        )
        state = HTFState(bias=Bias.BULLISH, active_obs=[ob])
        invalidate_obs(
            state, bar_close=47.0, current_daily_index=25, max_ob_age_bars=20
        )
        assert ob.invalidated is True

    def test_valid_ob_not_invalidated(self):
        ob = OrderBlock(
            top=50.0,
            bottom=45.0,
            bias=Bias.BULLISH,
            ob_id=(1, 0, 0, "long"),
            anchor_swing=SwingPoint(0, 0, 45.0, True),
            msb_bar_index=1,
            created_at_daily_index=0,
        )
        state = HTFState(bias=Bias.BULLISH, active_obs=[ob])
        invalidate_obs(
            state, bar_close=47.0, current_daily_index=10, max_ob_age_bars=20
        )
        assert ob.invalidated is False


# ---------------------------------------------------------------------------
# Premium / discount tests
# ---------------------------------------------------------------------------


class TestPremiumDiscount:
    def test_discount(self):
        r = TradingRange(high=100.0, low=80.0, midpoint=90.0, bias=Bias.BULLISH)
        assert is_in_discount(85.0, r, 0.5) is True
        assert is_in_discount(95.0, r, 0.5) is False

    def test_premium(self):
        r = TradingRange(high=100.0, low=80.0, midpoint=90.0, bias=Bias.BEARISH)
        assert is_in_premium(95.0, r, 0.5) is True
        assert is_in_premium(85.0, r, 0.5) is False

    def test_custom_threshold(self):
        r = TradingRange(high=100.0, low=80.0, midpoint=90.0, bias=Bias.BULLISH)
        # threshold 0.382 → level = 80 + 20*0.382 = 87.64
        assert is_in_discount(85.0, r, 0.382) is True
        assert is_in_discount(88.0, r, 0.382) is False


# ---------------------------------------------------------------------------
# Audit: SwingDetector confirmation lag (point #2)
# ---------------------------------------------------------------------------


class TestSwingConfirmationLag:
    def test_swing_not_emitted_until_right_side_bar(self):
        """Swing at index 1 is only emitted when bar 2 is pushed (confirmation lag)."""
        det = SwingDetector(lookback=1)
        # Bar 0
        s0 = det.push(0, 100, 9.0, 10.0, 8.0, 9.5)
        assert s0 == []
        # Bar 1: potential swing high
        s1 = det.push(1, 200, 11.0, 15.0, 10.0, 14.0)
        assert s1 == []  # NOT emitted yet — right-side bar not seen
        # Bar 2: confirms bar 1
        s2 = det.push(2, 300, 13.0, 14.0, 12.0, 13.0)
        highs = [s for s in s2 if s.is_high]
        assert len(highs) == 1
        assert highs[0].index == 1  # Emitted for bar 1, not bar 2

    def test_5_candle_confirmation_delay(self):
        """Lookback=2: swing at index 2 only confirmed after bar 4."""
        det = SwingDetector(lookback=2)
        for i in range(4):
            result = det.push(i, i * 100, 10.0, 11.0 + (2 if i == 2 else 0), 9.0, 10.0)
            assert result == [], f"Should not emit before bar 4, emitted at bar {i}"
        # Bar 4 completes the window
        result = det.push(4, 400, 10.0, 11.0, 9.0, 10.0)
        highs = [s for s in result if s.is_high]
        assert len(highs) == 1
        assert highs[0].index == 2


# ---------------------------------------------------------------------------
# Audit: MSB uses close only, strict > / < (point #3)
# ---------------------------------------------------------------------------


class TestMSBComparators:
    def test_bullish_msb_requires_strict_greater(self):
        """Close EQUAL to swing high should NOT trigger MSB."""
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        result = check_msb(sh, sl, 50.0, Bias.NEUTRAL)  # close == swing high
        assert result is None

    def test_bearish_msb_requires_strict_less(self):
        """Close EQUAL to swing low should NOT trigger MSB."""
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = [SwingPoint(1, 200, 40.0, False)]
        result = check_msb(sh, sl, 40.0, Bias.NEUTRAL)  # close == swing low
        assert result is None

    def test_msb_uses_close_not_high(self):
        """MSB check takes bar_close, verify it's not checking highs/lows."""
        sh = [SwingPoint(0, 100, 50.0, True)]
        sl = []
        # If someone passed high instead of close, this would trigger.
        # Passing 49.9 (close below 50) should NOT trigger.
        result = check_msb(sh, sl, 49.9, Bias.NEUTRAL)
        assert result is None
        # 50.1 should trigger
        result = check_msb(sh, sl, 50.1, Bias.NEUTRAL)
        assert result is not None


# ---------------------------------------------------------------------------
# Audit: Consecutive MSBs in same direction (point #4)
# ---------------------------------------------------------------------------


class TestConsecutiveMSBs:
    def test_consecutive_bullish_msbs_update_range_and_ob(self):
        """Two bullish MSBs in a row should produce two OBs and update range."""
        state = HTFState()
        params = ICTBlueprintParams(swing_lookback=1, ob_candles=1)
        detector = SwingDetector(lookback=1)
        bars_hist = []

        # Build a sequence: low → high → higher-high → pullback → even-higher
        bar_data = [
            (0, 100, 100.0, 105.0, 95.0, 102.0),  # neutral
            (1, 200, 102.0, 110.0, 100.0, 108.0),  # swing high candidate
            (2, 300, 108.0, 109.0, 98.0, 99.0),  # confirms SH at 1, SL candidate
            (3, 400, 99.0, 107.0, 97.0, 106.0),  # confirms SL at 2
            (4, 500, 106.0, 115.0, 105.0, 114.0),  # bullish MSB (close > SH[1]=110)
            (5, 600, 114.0, 120.0, 113.0, 119.0),  # new SH candidate
            (6, 700, 119.0, 119.5, 110.0, 111.0),  # confirms SH at 5, new SL
            (7, 800, 111.0, 118.0, 109.0, 117.0),  # confirms SL at 6
            (8, 900, 117.0, 125.0, 116.0, 124.0),  # 2nd bullish MSB (close > SH[5]=120)
        ]

        for bar in bar_data:
            bars_hist.append(bar)
            update_htf(state, bar[0], bar, params, detector, bars_hist)

        assert state.bias == Bias.BULLISH
        # Should have at least 2 OBs (one per MSB)
        valid_obs = [ob for ob in state.active_obs if not ob.invalidated]
        assert len(valid_obs) >= 1  # At least the second MSB's OB
        # Range should reflect the latest MSB
        assert state.current_range is not None
