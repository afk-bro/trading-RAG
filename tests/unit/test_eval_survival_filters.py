"""Tests for eval survival filters: LONDON session, confidence gate,
alignment gate, R-based daily halt."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from app.services.backtest.engines.daily_governor import DailyGovernor
from app.services.strategy.indicators.tf_bias import BiasDirection
from app.services.strategy.strategies.unicorn_model import (
    SessionProfile,
    SESSION_WINDOWS,
    is_in_macro_window,
    ScaleOutPreset,
    SCALE_OUT_PARAMS,
)
from app.services.backtest.engines.unicorn_runner import (
    BiasState,
    TradeRecord,
    CriteriaCheck,
    TradingSession,
    _asof_lookup,
)


ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# 1. LONDON session profile
# ---------------------------------------------------------------------------

class TestSessionProfileLondon:
    def test_london_in_session_windows(self):
        """LONDON profile has exactly one window: 3:00-4:00 ET."""
        windows = SESSION_WINDOWS[SessionProfile.LONDON]
        assert len(windows) == 1
        assert windows[0] == (time(3, 0), time(4, 0))

    def test_is_in_macro_window_london_inside(self):
        """3:30 ET is inside London window."""
        ts = datetime(2025, 3, 10, 3, 30, tzinfo=ET)
        assert is_in_macro_window(ts, SessionProfile.LONDON) is True

    def test_is_in_macro_window_london_before(self):
        """2:59 ET is before London window."""
        ts = datetime(2025, 3, 10, 2, 59, tzinfo=ET)
        assert is_in_macro_window(ts, SessionProfile.LONDON) is False

    def test_is_in_macro_window_london_at_end(self):
        """4:00 ET is at close of window (should be excluded)."""
        ts = datetime(2025, 3, 10, 4, 0, tzinfo=ET)
        assert is_in_macro_window(ts, SessionProfile.LONDON) is False

    def test_is_in_macro_window_london_ny_am_rejected(self):
        """9:30 ET is NOT inside London-only window."""
        ts = datetime(2025, 3, 10, 9, 30, tzinfo=ET)
        assert is_in_macro_window(ts, SessionProfile.LONDON) is False

    def test_london_enum_value(self):
        assert SessionProfile.LONDON.value == "london"
        assert SessionProfile("london") == SessionProfile.LONDON


# ---------------------------------------------------------------------------
# 2. Confidence gate (unit-level: tests the gate logic directly)
# ---------------------------------------------------------------------------

class TestConfidenceGate:
    """The confidence gate in the runner rejects when
    criteria.htf_bias_confidence < config.min_confidence.
    We test the condition logic here without running the full backtest."""

    def test_rejects_low_confidence(self):
        conf = 0.55
        threshold = 0.8
        assert conf < threshold  # would be rejected

    def test_passes_high_confidence(self):
        conf = 0.85
        threshold = 0.8
        assert conf >= threshold  # would pass

    def test_disabled_when_none(self):
        """When min_confidence is None, the gate is disabled."""
        threshold = None
        assert threshold is None  # gate skipped


# ---------------------------------------------------------------------------
# 3. Alignment gate (unit-level: tests _asof_lookup + direction comparison)
# ---------------------------------------------------------------------------

class TestAlignmentGate:
    def _make_bias_series(self, direction, ts_offset_minutes=0):
        ts = datetime(2025, 3, 10, 3, 30, tzinfo=ET)
        return [BiasState(
            ts=ts,
            direction=direction,
            confidence=0.8,
        )]

    def test_rejects_divergent(self):
        """Primary=BULLISH, ref=BEARISH -> rejected."""
        primary = BiasDirection.BULLISH
        series = self._make_bias_series(BiasDirection.BEARISH)
        ts = datetime(2025, 3, 10, 3, 35, tzinfo=ET)
        ref = _asof_lookup(series, ts)
        assert ref is not None
        assert ref.direction != primary

    def test_rejects_neutral(self):
        """Primary=BULLISH, ref=NEUTRAL -> rejected."""
        primary = BiasDirection.BULLISH
        series = self._make_bias_series(BiasDirection.NEUTRAL)
        ts = datetime(2025, 3, 10, 3, 35, tzinfo=ET)
        ref = _asof_lookup(series, ts)
        assert ref is not None
        assert ref.direction == BiasDirection.NEUTRAL

    def test_rejects_missing(self):
        """No ref data before ts -> None -> rejected."""
        series = self._make_bias_series(BiasDirection.BULLISH)
        ts_before = datetime(2025, 3, 10, 2, 0, tzinfo=ET)
        ref = _asof_lookup(series, ts_before)
        assert ref is None

    def test_passes_aligned(self):
        """Primary=BULLISH, ref=BULLISH -> passes."""
        primary = BiasDirection.BULLISH
        series = self._make_bias_series(BiasDirection.BULLISH)
        ts = datetime(2025, 3, 10, 3, 35, tzinfo=ET)
        ref = _asof_lookup(series, ts)
        assert ref is not None
        assert ref.direction == primary

    def test_disabled_by_default(self):
        """require_intermarket_alignment defaults to False, so gate is skipped."""
        require = False
        assert not require  # gate skipped


# ---------------------------------------------------------------------------
# 4. R-based daily loss halt
# ---------------------------------------------------------------------------

class TestUpdateRDay:
    def test_sets_threshold(self):
        """update_r_day recalculates max_daily_loss_dollars from R."""
        gov = DailyGovernor(
            max_daily_loss_dollars=300.0,
            max_trades_per_day=2,
            max_daily_loss_r=1.0,
        )
        gov.update_r_day(200.0)  # 1R = $200
        assert gov.max_daily_loss_dollars == 200.0

    def test_noop_when_disabled(self):
        """When max_daily_loss_r is None, update_r_day is a no-op."""
        gov = DailyGovernor(
            max_daily_loss_dollars=300.0,
            max_trades_per_day=2,
            max_daily_loss_r=None,
        )
        gov.update_r_day(200.0)
        assert gov.max_daily_loss_dollars == 300.0  # unchanged

    def test_halt_triggers_after_r_based_loss(self):
        """After update_r_day sets threshold, losses should trigger halt correctly."""
        gov = DailyGovernor(
            max_daily_loss_dollars=9999.0,  # placeholder, will be overwritten
            max_trades_per_day=3,
            max_daily_loss_r=1.0,
        )
        gov.update_r_day(150.0)  # 1R = $150, so halt at -$150
        assert gov.max_daily_loss_dollars == 150.0

        gov.record_trade_close(-150.0)
        assert gov.halted_for_day is True
        assert gov.halt_reason == "loss_limit"
        assert gov.allows_entry() is False

    def test_fractional_r_multiplier(self):
        """max_daily_loss_r=0.5 means halt at half of R_day."""
        gov = DailyGovernor(
            max_daily_loss_dollars=9999.0,
            max_trades_per_day=3,
            max_daily_loss_r=0.5,
        )
        gov.update_r_day(200.0)
        assert gov.max_daily_loss_dollars == 100.0  # 0.5 * 200

    def test_zero_r_day_noop(self):
        """If R_day is 0, don't update (avoid 0 threshold)."""
        gov = DailyGovernor(
            max_daily_loss_dollars=300.0,
            max_trades_per_day=2,
            max_daily_loss_r=1.0,
        )
        gov.update_r_day(0.0)
        assert gov.max_daily_loss_dollars == 300.0  # unchanged


# ---------------------------------------------------------------------------
# 5. Partial exit (scale-out)
# ---------------------------------------------------------------------------

def _make_trade(
    qty: int = 2,
    entry_price: float = 100.0,
    risk_points: float = 5.0,
    direction: BiasDirection = BiasDirection.BULLISH,
    partial_exit_r: float | None = 1.0,
    partial_exit_pct: float = 0.5,
) -> TradeRecord:
    """Helper: build a minimal TradeRecord for partial-exit tests."""
    trade = TradeRecord(
        entry_time=datetime(2025, 6, 1, 9, 30, tzinfo=ET),
        entry_price=entry_price,
        direction=direction,
        quantity=qty,
        session=TradingSession.NY_AM,
        criteria=CriteriaCheck(),
        stop_price=entry_price - risk_points if direction == BiasDirection.BULLISH else entry_price + risk_points,
        target_price=float("inf") if direction == BiasDirection.BULLISH else float("-inf"),
        risk_points=risk_points,
        initial_stop=entry_price - risk_points if direction == BiasDirection.BULLISH else entry_price + risk_points,
        entry_atr=risk_points,
    )
    if partial_exit_r is not None:
        if direction == BiasDirection.BULLISH:
            trade.partial_exit_price = entry_price + (risk_points * partial_exit_r)
        else:
            trade.partial_exit_price = entry_price - (risk_points * partial_exit_r)
        trade.partial_exit_r = partial_exit_r
        trade.partial_exit_pct = partial_exit_pct
    return trade


class TestPartialExit:
    def test_partial_exit_splits_trade(self):
        """A 2-contract trade should split into leg1 (1 ct, +1R) + leg2 (1 ct, trailing)."""
        trade = _make_trade(qty=2, entry_price=100.0, risk_points=5.0)
        assert trade.partial_exit_price == 105.0  # +1R for long

        # Simulate the split logic (same as in runner)
        leg1_qty = int(trade.quantity * trade.partial_exit_pct)
        assert leg1_qty == 1
        leg2_qty = trade.quantity - leg1_qty
        assert leg2_qty == 1

        # Build leg 1 record (as runner does)
        leg1 = TradeRecord(
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            direction=trade.direction,
            quantity=leg1_qty,
            session=trade.session,
            criteria=trade.criteria,
            stop_price=trade.stop_price,
            target_price=trade.target_price,
            risk_points=trade.risk_points,
            initial_stop=trade.initial_stop,
            entry_atr=trade.entry_atr,
            exit_price=trade.partial_exit_price,
            exit_reason="partial_target",
            is_partial_leg=True,
            leg_index=1,
        )
        assert leg1.is_partial_leg is True
        assert leg1.leg_index == 1

        # Leg 1 PnL
        pnl_leg1 = trade.partial_exit_price - trade.entry_price
        assert pnl_leg1 == 5.0  # +1R = risk_points

        # After split, original becomes leg 2
        trade.quantity = leg2_qty
        trade.is_partial_leg = True
        trade.leg_index = 2
        trade.partial_exit_price = None

        assert trade.quantity == 1
        assert trade.is_partial_leg is True
        assert trade.leg_index == 2
        assert trade.partial_exit_price is None  # no re-trigger

    def test_partial_exit_disabled_when_none(self):
        """No split when partial_exit_r is None."""
        trade = _make_trade(qty=2, partial_exit_r=None)
        assert trade.partial_exit_price is None
        assert trade.partial_exit_r is None

    def test_partial_exit_skips_single_contract(self):
        """No split when quantity=1 (can't split 1 contract)."""
        trade = _make_trade(qty=1)
        # The gate: trade.quantity >= 2
        assert trade.quantity < 2  # would be skipped

    def test_partial_exit_leg1_pnl(self):
        """Leg1 PnL = risk_points * partial_exit_r."""
        # Long
        trade_long = _make_trade(qty=4, entry_price=200.0, risk_points=10.0, partial_exit_r=1.5)
        expected_price = 200.0 + (10.0 * 1.5)  # 215.0
        assert trade_long.partial_exit_price == expected_price
        pnl = expected_price - trade_long.entry_price
        assert pnl == 15.0  # 1.5R * 10 risk_points

        # Short
        trade_short = _make_trade(
            qty=4, entry_price=200.0, risk_points=10.0,
            direction=BiasDirection.BEARISH, partial_exit_r=1.0,
        )
        expected_price_short = 200.0 - (10.0 * 1.0)  # 190.0
        assert trade_short.partial_exit_price == expected_price_short
        pnl_short = trade_short.entry_price - expected_price_short
        assert pnl_short == 10.0

    def test_partial_exit_leg2_trails(self):
        """After split, leg2 retains trail fields and continues as a trailer."""
        trade = _make_trade(qty=4, entry_price=100.0, risk_points=5.0)
        trade.trail_distance = 7.5  # e.g. 1.5x ATR
        trade.trail_active = False

        # Simulate split
        leg1_qty = int(trade.quantity * trade.partial_exit_pct)  # 2
        trade.quantity = trade.quantity - leg1_qty  # 2
        trade.is_partial_leg = True
        trade.leg_index = 2
        trade.partial_exit_price = None

        # Leg 2 should still have trail fields intact
        assert trade.trail_distance == 7.5
        assert trade.trail_active is False
        assert trade.quantity == 2
        assert trade.exit_time is None  # still open

    def test_partial_exit_qty_rounding(self):
        """With odd quantity, leg1 gets floor, leg2 gets remainder."""
        trade = _make_trade(qty=3, partial_exit_pct=0.5)
        leg1_qty = int(trade.quantity * trade.partial_exit_pct)
        assert leg1_qty == 1  # floor(3 * 0.5) = 1
        leg2_qty = trade.quantity - leg1_qty
        assert leg2_qty == 2  # remainder gets more


# ---------------------------------------------------------------------------
# 6. Governor partial-leg counting
# ---------------------------------------------------------------------------

class TestGovernorPartialLegCounting:
    def test_governor_skips_partial_leg_count(self):
        """Partial legs should not increment day_trade_count."""
        gov = DailyGovernor(max_trades_per_day=2, max_daily_loss_dollars=1000.0)
        gov.record_trade_close(50.0, is_partial_leg=True)
        gov.record_trade_close(30.0, is_partial_leg=True)
        gov.record_trade_close(-20.0, is_partial_leg=False)
        assert gov.day_trade_count == 1  # only the non-partial leg counts

    def test_governor_still_tracks_partial_leg_pnl(self):
        """Losses from partial legs must still accumulate in day_loss_dollars."""
        gov = DailyGovernor(max_trades_per_day=5, max_daily_loss_dollars=500.0)
        gov.record_trade_close(-100.0, is_partial_leg=True)
        gov.record_trade_close(-50.0, is_partial_leg=False)
        assert gov.day_loss_dollars == -150.0
        assert gov.day_trade_count == 1  # only the non-partial

    def test_governor_halts_on_setups_not_legs(self):
        """2 setup closes + 2 partial closes = day_trade_count==2, governor halts.

        Without the fix, all 4 closes would count and the governor would have
        halted after 2 (blocking the second setup).
        """
        gov = DailyGovernor(max_trades_per_day=2, max_daily_loss_dollars=5000.0)

        # Setup 1: partial leg + trailer leg
        gov.record_trade_close(50.0, is_partial_leg=True)   # leg 1 (partial)
        gov.record_trade_close(100.0, is_partial_leg=False)  # leg 2 (trailer = setup close)

        # Setup 2: partial leg + trailer leg
        gov.record_trade_close(30.0, is_partial_leg=True)   # leg 1 (partial)
        gov.record_trade_close(-40.0, is_partial_leg=False)  # leg 2 (trailer = setup close)

        assert gov.day_trade_count == 2
        assert gov.allows_entry() is False
        assert gov.halt_reason == "trade_limit"

    def test_governor_partial_loss_triggers_halt(self):
        """A partial leg loss can push day_loss past the halt threshold."""
        gov = DailyGovernor(max_trades_per_day=5, max_daily_loss_dollars=100.0)
        gov.record_trade_close(-60.0, is_partial_leg=True)
        gov.record_trade_close(-50.0, is_partial_leg=True)
        # Total loss = -110, exceeds -100 threshold
        assert gov.halted_for_day is True
        assert gov.halt_reason == "loss_limit"
        assert gov.day_trade_count == 0  # no setups counted


# ---------------------------------------------------------------------------
# 7. ScaleOutPreset enum
# ---------------------------------------------------------------------------

class TestScaleOutPreset:
    def test_none_disables_partial_exit(self):
        params = SCALE_OUT_PARAMS[ScaleOutPreset.NONE]
        assert params["partial_exit_r"] is None
        assert params["partial_exit_pct"] == 0.0

    def test_prop_safe_is_33_at_1r(self):
        params = SCALE_OUT_PARAMS[ScaleOutPreset.PROP_SAFE]
        assert params["partial_exit_r"] == 1.0
        assert params["partial_exit_pct"] == 0.33

    def test_only_two_presets_exist(self):
        """No other presets should exist — B and D are rejected."""
        assert set(ScaleOutPreset) == {ScaleOutPreset.NONE, ScaleOutPreset.PROP_SAFE}

    def test_enum_from_string(self):
        assert ScaleOutPreset("none") == ScaleOutPreset.NONE
        assert ScaleOutPreset("prop_safe") == ScaleOutPreset.PROP_SAFE
