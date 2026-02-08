"""
Unit tests for ATR trailing stop logic in resolve_bar_exit.
"""

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.services.strategy.models import OHLCVBar
from app.services.strategy.indicators.tf_bias import BiasDirection
from app.services.backtest.engines.unicorn_runner import (
    TradeRecord,
    CriteriaCheck,
    TradingSession,
    resolve_bar_exit,
    IntrabarPolicy,
    _compute_trail_distance,
)

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(ts, open_, high, low, close):
    return OHLCVBar(ts=ts, open=open_, high=high, low=low, close=close, volume=100)


def _make_criteria():
    """Minimal CriteriaCheck for trade construction."""
    return CriteriaCheck(
        htf_bias_aligned=True,
        htf_bias_confidence=0.7,
        stop_valid=True,
        in_macro_window=True,
        mss_found=True,
        displacement_valid=True,
        liquidity_sweep_found=True,
        htf_fvg_found=True,
        breaker_block_found=True,
        ltf_fvg_found=True,
    )


def _make_long_trade(entry=100.0, stop=95.0, risk=5.0, trail_dist=7.5, atr=5.0):
    """Long trade with trailing stop fields pre-configured."""
    return TradeRecord(
        entry_time=datetime(2024, 6, 1, 10, 0, tzinfo=ET),
        entry_price=entry,
        direction=BiasDirection.BULLISH,
        quantity=1,
        session=TradingSession.NY_AM,
        criteria=_make_criteria(),
        stop_price=stop,
        target_price=float("inf"),
        risk_points=risk,
        initial_stop=stop,
        entry_atr=atr,
        trail_distance=trail_dist,
    )


def _make_short_trade(entry=100.0, stop=105.0, risk=5.0, trail_dist=7.5, atr=5.0):
    """Short trade with trailing stop fields pre-configured."""
    return TradeRecord(
        entry_time=datetime(2024, 6, 1, 10, 0, tzinfo=ET),
        entry_price=entry,
        direction=BiasDirection.BEARISH,
        quantity=1,
        session=TradingSession.NY_AM,
        criteria=_make_criteria(),
        stop_price=stop,
        target_price=float("-inf"),
        risk_points=risk,
        initial_stop=stop,
        entry_atr=atr,
        trail_distance=trail_dist,
    )


def _ts(minutes_offset=0):
    return datetime(2024, 6, 1, 10, 0, tzinfo=ET) + timedelta(minutes=minutes_offset)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrailNotActiveBeforeOneR:
    """Trail should not activate when MFE < 1R."""

    def test_trail_not_active_before_1r(self):
        trade = _make_long_trade()  # entry=100, risk=5
        # Bar moves +0.8R (4 points), high=104
        bar = _make_bar(_ts(1), 101, 104, 100, 103)
        result = resolve_bar_exit(
            trade, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5,
        )
        assert result is None
        assert not trade.trail_active
        assert trade.stop_price == 95.0  # unchanged


class TestTrailActivatesAtOneR:
    """Trail activates at +1R MFE, stop moves to at least entry."""

    def test_trail_activates_at_1r(self):
        trade = _make_long_trade()  # entry=100, stop=95, risk=5, trail_dist=7.5
        # Bar hits exactly +1R: high=105
        bar = _make_bar(_ts(1), 101, 105, 100, 104)
        result = resolve_bar_exit(
            trade, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5,
        )
        assert trade.trail_active is True
        # BE floor: stop >= entry
        assert trade.stop_price >= 100.0
        # Trail formula: 105 - 7.5 = 97.5, but BE floor = max(97.5, 100) = 100
        assert trade.stop_price == 100.0


class TestTrailRatchetsForward:
    """Sequential bars: stop only increases (long)."""

    def test_trail_ratchets_forward(self):
        trade = _make_long_trade()
        # Bar 1: +1.2R = 106, activates trail
        bar1 = _make_bar(_ts(1), 101, 106, 100, 105)
        resolve_bar_exit(trade, bar1, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_active
        # trail_high=106, new_stop = 106 - 7.5 = 98.5, but BE floor = 100
        stop_after_1 = trade.stop_price
        assert stop_after_1 == 100.0

        # Bar 2: +1.5R = 107.5
        bar2 = _make_bar(_ts(2), 105, 107.5, 104, 107)
        resolve_bar_exit(trade, bar2, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        # trail_high=107.5, new_stop = 107.5 - 7.5 = 100.0
        stop_after_2 = trade.stop_price
        assert stop_after_2 >= stop_after_1

        # Bar 3: +2.0R = 110
        bar3 = _make_bar(_ts(3), 107, 110, 106, 109)
        resolve_bar_exit(trade, bar3, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        # trail_high=110, new_stop = 110 - 7.5 = 102.5
        stop_after_3 = trade.stop_price
        assert stop_after_3 == 102.5
        assert stop_after_3 >= stop_after_2

        # Bar 4: pullback — high=108 (lower than trail_high), stop stays
        bar4 = _make_bar(_ts(4), 109, 108, 105, 106)
        resolve_bar_exit(trade, bar4, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.stop_price == stop_after_3  # no regression


class TestTrailExitReasonIsTrailStop:
    """When trailed stop is hit, exit reason should be trail_stop."""

    def test_trail_exit_reason_is_trail_stop(self):
        trade = _make_long_trade()
        # Activate trail: +2R = 110
        bar1 = _make_bar(_ts(1), 101, 110, 100, 109)
        resolve_bar_exit(trade, bar1, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_active
        # trail_high=110, stop = 110 - 7.5 = 102.5

        # Bar 2: price drops to hit trailed stop
        bar2 = _make_bar(_ts(2), 105, 106, 102.0, 102.5)
        result = resolve_bar_exit(trade, bar2, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert result is not None
        assert result.exit_reason == "trail_stop"


class TestOriginalStopReasonIsStopLoss:
    """When original stop hit without trail activation, reason is stop_loss."""

    def test_original_stop_reason_is_stop_loss(self):
        trade = _make_long_trade()  # stop=95
        # Bar goes down and hits original stop, no activation
        bar = _make_bar(_ts(1), 100, 101, 94, 95)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert result is not None
        assert result.exit_reason == "stop_loss"
        assert not trade.trail_active


class TestTrailAndBreakevenMutuallyExclusive:
    """CLI mutual exclusion between trail and breakeven."""

    def test_trail_and_breakeven_mutually_exclusive(self):
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable, "scripts/run_unicorn_backtest.py",
                "--symbol", "NQ",
                "--breakeven-at-r", "1.0",
                "--trail-atr-mult", "1.5",
            ],
            capture_output=True,
            text=True,
            cwd="/home/x/dev/automation-infra/trading-RAG",
        )
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr.lower()


class TestTargetSentinelInf:
    """Target should be inf (long) or -inf (short) when trail is active."""

    def test_target_sentinel_inf_long(self):
        trade = _make_long_trade()
        assert trade.target_price == float("inf")
        # Target should never trigger — bar goes very high
        bar = _make_bar(_ts(1), 101, 999999, 100, 500000)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        # Should not exit via target (only trail_stop if activated and hit)
        # With high=999999, trail activates and stop moves way up.
        # But low=100 >= stop so no stop hit on this bar.
        assert result is None or result.exit_reason != "target"

    def test_target_sentinel_inf_short(self):
        trade = _make_short_trade()
        assert trade.target_price == float("-inf")


class TestBEFloorOnActivation:
    """Even when trail_distance > risk_points, stop floors at entry after activation."""

    def test_be_floor_on_activation(self):
        # trail_distance = 10, risk = 5, so trail formula could put stop below entry
        trade = _make_long_trade(entry=100.0, stop=95.0, risk=5.0, trail_dist=10.0, atr=5.0)
        # Activate: high = 105 (+1R)
        bar = _make_bar(_ts(1), 101, 105, 100, 104)
        resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=2.0)
        assert trade.trail_active
        # trail_high=105, trail formula: 105 - 10 = 95. But BE floor = max(95, 100) = 100
        assert trade.stop_price >= 100.0


class TestTrailDistanceFrozen:
    """trail_distance is set once at trade creation and doesn't change."""

    def test_trail_distance_frozen(self):
        trade = _make_long_trade(trail_dist=7.5)
        original_dist = trade.trail_distance

        # Process multiple bars
        for i in range(1, 5):
            bar = _make_bar(_ts(i), 100 + i, 102 + i, 99 + i, 101 + i)
            resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
            assert trade.trail_distance == original_dist


class TestShortDirectionTrail:
    """Mirror logic for shorts: trail_low tracks, stop moves down."""

    def test_short_direction_trail(self):
        trade = _make_short_trade()  # entry=100, stop=105, risk=5, trail_dist=7.5
        # Bar 1: +1R favorable = price drops to 95 (low=95)
        bar1 = _make_bar(_ts(1), 99, 100, 95, 96)
        resolve_bar_exit(trade, bar1, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_active
        assert trade.trail_low == 95.0
        # BE floor: stop <= entry = 100
        assert trade.stop_price <= 100.0
        # Trail: 95 + 7.5 = 102.5, min(105, 102.5) = 102.5
        assert trade.stop_price == 100.0  # BE floor dominates (102.5 > 100 but min)

        # Bar 2: further drop to 90
        bar2 = _make_bar(_ts(2), 96, 97, 90, 91)
        resolve_bar_exit(trade, bar2, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_low == 90.0
        # Trail: 90 + 7.5 = 97.5, min(100, 97.5) = 97.5
        assert trade.stop_price == 97.5

        # Bar 3: pullback, low=92 (higher than trail_low=90), stop stays
        bar3 = _make_bar(_ts(3), 91, 93, 92, 92)
        resolve_bar_exit(trade, bar3, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.stop_price == 97.5  # no regression

    def test_short_trail_stop_exit(self):
        trade = _make_short_trade()
        # Activate: low=95
        bar1 = _make_bar(_ts(1), 99, 100, 90, 91)
        resolve_bar_exit(trade, bar1, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        # trail_low=90, stop = 90 + 7.5 = 97.5
        assert trade.stop_price == 97.5

        # Hit stop: high >= 97.5
        bar2 = _make_bar(_ts(2), 92, 98, 91, 97)
        result = resolve_bar_exit(trade, bar2, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert result is not None
        assert result.exit_reason == "trail_stop"


class TestActivationBarGetsImmediateTrail:
    """On the bar that crosses +1R, trail formula also applies (no delay)."""

    def test_activation_bar_gets_immediate_trail(self):
        trade = _make_long_trade()  # entry=100, risk=5, trail_dist=7.5
        # Bar crosses +3R in one shot: high=115
        bar = _make_bar(_ts(1), 101, 115, 100, 114)
        resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_active
        # trail_high=115, trail formula: 115 - 7.5 = 107.5
        # BE floor = max(107.5, 100) = 107.5
        assert trade.stop_price == 107.5


class TestEodExitStillWorksWithTrail:
    """EOD exit should override trail when time expires."""

    def test_eod_exit_still_works_with_trail(self):
        trade = _make_long_trade()
        # Bar at EOD time, no stop/target hit, trail active
        # Activate trail first
        bar1 = _make_bar(_ts(1), 101, 110, 100, 109)
        resolve_bar_exit(trade, bar1, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_active

        # EOD bar: price above trailed stop but EOD forces exit
        eod_ts = datetime(2024, 6, 1, 15, 45, tzinfo=ET)
        bar_eod = _make_bar(eod_ts, 108, 109, 107, 108)
        result = resolve_bar_exit(
            trade, bar_eod, IntrabarPolicy.WORST, 0.0,
            eod_exit=True, eod_time=time(15, 45),
            trail_atr_mult=1.5,
        )
        assert result is not None
        assert result.exit_reason == "eod"


class TestActivationPlusStopHitSameBar:
    """Edge case: bar activates trail AND hits the trailed stop in one bar."""

    def test_activation_and_stop_hit_same_bar_long(self):
        # entry=100, risk=5, stop=95, trail_dist=7.5
        trade = _make_long_trade()
        # Bar spikes to +1R (105) then crashes through the new stop.
        # trail_high=106, trail formula: 106 - 7.5 = 98.5, BE floor = max(98.5, 100) = 100
        # Bar low=94 < stop=100 → stop hit
        bar = _make_bar(_ts(1), 101, 106, 94, 95)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_active
        assert result is not None
        assert result.exit_reason == "trail_stop"
        # Stop was at entry (100) due to BE floor, fill at min(100, open=101) = 100
        assert result.exit_price == 100.0

    def test_activation_and_stop_hit_same_bar_short(self):
        # entry=100, risk=5, stop=105, trail_dist=7.5
        trade = _make_short_trade()
        # Bar dips to 95 (+1R), then spikes to 101 (above BE floor stop=100)
        bar = _make_bar(_ts(1), 99, 101, 94, 100)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5)
        assert trade.trail_active
        assert result is not None
        assert result.exit_reason == "trail_stop"


class TestTrailRatchetMonotonicity:
    """Stop can only move in favorable direction — never loosens."""

    def test_monotonicity_long_50_bars(self):
        trade = _make_long_trade()
        prev_stop = trade.stop_price

        # Simulate 50 bars: uptrend then pullback then new high then chop
        prices = (
            [100 + i * 0.5 for i in range(20)]   # grind up
            + [110 - i * 0.3 for i in range(10)]  # pullback
            + [107 + i * 0.8 for i in range(10)]  # new high
            + [113 - i * 0.2 for i in range(10)]  # chop down
        )
        for i, base in enumerate(prices):
            bar = _make_bar(
                _ts(i + 1),
                open_=base,
                high=base + 1.5,
                low=base - 1.5,
                close=base + 0.5,
            )
            result = resolve_bar_exit(
                trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5,
            )
            assert trade.stop_price >= prev_stop, (
                f"Stop regressed on bar {i}: {trade.stop_price} < {prev_stop}"
            )
            prev_stop = trade.stop_price
            if result is not None:
                break

    def test_monotonicity_short_50_bars(self):
        trade = _make_short_trade()
        prev_stop = trade.stop_price

        prices = (
            [100 - i * 0.5 for i in range(20)]
            + [90 + i * 0.3 for i in range(10)]
            + [93 - i * 0.8 for i in range(10)]
            + [87 + i * 0.2 for i in range(10)]
        )
        for i, base in enumerate(prices):
            bar = _make_bar(
                _ts(i + 1),
                open_=base,
                high=base + 1.5,
                low=base - 1.5,
                close=base - 0.5,
            )
            result = resolve_bar_exit(
                trade, bar, IntrabarPolicy.WORST, 0.0, trail_atr_mult=1.5,
            )
            assert trade.stop_price <= prev_stop, (
                f"Stop loosened on bar {i}: {trade.stop_price} > {prev_stop}"
            )
            prev_stop = trade.stop_price
            if result is not None:
                break


class TestTrailDistanceCap:
    """trail_distance should be capped by trail_cap_mult * risk_points."""

    def test_cap_reduces_distance(self):
        # ATR=10, trail_atr_mult=2.0 → raw=20, risk=5, cap_mult=1.0 → cap=5
        dist = _compute_trail_distance(10.0, 2.0, 1.0, 5.0)
        assert dist == 5.0

    def test_no_cap_when_none(self):
        dist = _compute_trail_distance(10.0, 2.0, None, 5.0)
        assert dist == 20.0

    def test_cap_not_needed_when_raw_smaller(self):
        # ATR=2, trail_atr_mult=1.5 → raw=3, risk=5, cap_mult=1.0 → cap=5, min=3
        dist = _compute_trail_distance(2.0, 1.5, 1.0, 5.0)
        assert dist == 3.0

    def test_cap_zero_risk_no_crash(self):
        dist = _compute_trail_distance(10.0, 1.5, 1.0, 0.0)
        assert dist == 15.0  # cap division by zero avoided, raw returned

    def test_cap_075_tighter(self):
        # ATR=10, mult=2.0 → raw=20, risk=10, cap_mult=0.75 → cap=7.5
        dist = _compute_trail_distance(10.0, 2.0, 0.75, 10.0)
        assert dist == 7.5


class TestTrailActivateR:
    """Configurable activation threshold."""

    def test_activate_at_half_r(self):
        trade = _make_long_trade()  # entry=100, risk=5
        # Bar moves +0.5R = 102.5
        bar = _make_bar(_ts(1), 101, 102.5, 100, 102)
        resolve_bar_exit(
            trade, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5, trail_activate_r=0.5,
        )
        assert trade.trail_active is True
        assert trade.stop_price >= 100.0  # BE floor

    def test_no_activate_below_half_r(self):
        trade = _make_long_trade()
        # Bar moves +0.4R = 102.0
        bar = _make_bar(_ts(1), 101, 102.0, 100, 101.5)
        resolve_bar_exit(
            trade, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5, trail_activate_r=0.5,
        )
        assert trade.trail_active is False

    def test_default_1r_still_works(self):
        trade = _make_long_trade()
        # +0.9R = 104.5, should NOT activate at default 1.0
        bar = _make_bar(_ts(1), 101, 104.5, 100, 104)
        resolve_bar_exit(
            trade, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5,
        )
        assert trade.trail_active is False

    def test_lower_threshold_increases_activation(self):
        """With lower threshold, more bars trigger activation."""
        # At 1.0R: +0.8R doesn't activate
        trade_1r = _make_long_trade()
        bar = _make_bar(_ts(1), 101, 104, 100, 103)
        resolve_bar_exit(
            trade_1r, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5, trail_activate_r=1.0,
        )
        assert not trade_1r.trail_active

        # At 0.5R: same bar activates
        trade_05r = _make_long_trade()
        resolve_bar_exit(
            trade_05r, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5, trail_activate_r=0.5,
        )
        assert trade_05r.trail_active

    def test_short_activate_at_075r(self):
        trade = _make_short_trade()  # entry=100, risk=5
        # +0.75R favorable = price drops to 96.25 (low=96.25)
        bar = _make_bar(_ts(1), 99, 100, 96.25, 97)
        resolve_bar_exit(
            trade, bar, IntrabarPolicy.WORST, 0.0,
            trail_atr_mult=1.5, trail_activate_r=0.75,
        )
        assert trade.trail_active is True
        assert trade.stop_price <= 100.0  # BE floor
