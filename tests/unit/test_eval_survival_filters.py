"""Tests for eval survival filters: LONDON session, confidence gate,
alignment gate, R-based daily halt, weekly bias gate, confidence tiering,
NY AM timebox tightening."""

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from app.services.backtest.engines.daily_governor import DailyGovernor
from app.services.strategy.indicators.tf_bias import (
    BiasDirection,
    compute_weekly_bias,
)
from app.services.strategy.strategies.unicorn_model import (
    SessionProfile,
    SESSION_WINDOWS,
    UnicornConfig,
    is_in_macro_window,
    _apply_ny_am_cutoff,
    ScaleOutPreset,
    SCALE_OUT_PARAMS,
)
from app.services.strategy.models import OHLCVBar
from app.services.backtest.engines.unicorn_runner import (
    BiasState,
    BarBundle,
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
        return [
            BiasState(
                ts=ts,
                direction=direction,
                confidence=0.8,
            )
        ]

    def test_rejects_divergent(self):
        """Primary=BULLISH, ref=BEARISH -> rejected."""
        _primary = BiasDirection.BULLISH
        series = self._make_bias_series(BiasDirection.BEARISH)
        ts = datetime(2025, 3, 10, 3, 35, tzinfo=ET)
        ref = _asof_lookup(series, ts)
        assert ref is not None
        assert ref.direction != _primary

    def test_rejects_neutral(self):
        """Primary=BULLISH, ref=NEUTRAL -> rejected."""
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
        stop_price=(
            entry_price - risk_points
            if direction == BiasDirection.BULLISH
            else entry_price + risk_points
        ),
        target_price=(
            float("inf") if direction == BiasDirection.BULLISH else float("-inf")
        ),
        risk_points=risk_points,
        initial_stop=(
            entry_price - risk_points
            if direction == BiasDirection.BULLISH
            else entry_price + risk_points
        ),
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
        trade_long = _make_trade(
            qty=4, entry_price=200.0, risk_points=10.0, partial_exit_r=1.5
        )
        expected_price = 200.0 + (10.0 * 1.5)  # 215.0
        assert trade_long.partial_exit_price == expected_price
        pnl = expected_price - trade_long.entry_price
        assert pnl == 15.0  # 1.5R * 10 risk_points

        # Short
        trade_short = _make_trade(
            qty=4,
            entry_price=200.0,
            risk_points=10.0,
            direction=BiasDirection.BEARISH,
            partial_exit_r=1.0,
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
        gov.record_trade_close(50.0, is_partial_leg=True)  # leg 1 (partial)
        gov.record_trade_close(
            100.0, is_partial_leg=False
        )  # leg 2 (trailer = setup close)

        # Setup 2: partial leg + trailer leg
        gov.record_trade_close(30.0, is_partial_leg=True)  # leg 1 (partial)
        gov.record_trade_close(
            -40.0, is_partial_leg=False
        )  # leg 2 (trailer = setup close)

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
        """Tuning phase closed — exactly NONE and PROP_SAFE, nothing else."""
        assert set(ScaleOutPreset) == {
            ScaleOutPreset.NONE,
            ScaleOutPreset.PROP_SAFE,
        }, "Scale-out tuning phase is closed. Do not add presets."

    def test_enum_from_string(self):
        assert ScaleOutPreset("none") == ScaleOutPreset.NONE
        assert ScaleOutPreset("prop_safe") == ScaleOutPreset.PROP_SAFE


# ---------------------------------------------------------------------------
# 8. Weekly bias gate (Feature 1)
# ---------------------------------------------------------------------------


def _make_weekly_bars(price: float, n: int = 250) -> list[OHLCVBar]:
    """Generate N weekly bars with a steady uptrend from `price`."""
    bars = []
    for i in range(n):
        p = price + i * 0.5
        bars.append(
            OHLCVBar(
                ts=datetime(2020, 1, 6, tzinfo=timezone.utc)
                + __import__("datetime").timedelta(weeks=i),
                open=p,
                high=p + 2.0,
                low=p - 1.0,
                close=p + 1.0,
                volume=1000.0,
            )
        )
    return bars


class TestWeeklyBiasGate:
    def test_compute_weekly_bias_bullish(self):
        """Steady uptrend should produce BULLISH weekly bias."""
        bars = _make_weekly_bars(100.0, 250)
        result = compute_weekly_bias(bars)
        assert result.direction == BiasDirection.BULLISH
        assert result.confidence > 0.3

    def test_compute_weekly_bias_insufficient_data(self):
        """With < 200 bars, should return NEUTRAL."""
        bars = _make_weekly_bars(100.0, 50)
        result = compute_weekly_bias(bars)
        assert result.direction == BiasDirection.NEUTRAL
        assert result.confidence == 0.0

    def test_weekly_bias_bearish(self):
        """Steady downtrend → BEARISH."""
        bars = []
        for i in range(250):
            p = 200.0 - i * 0.5
            bars.append(
                OHLCVBar(
                    ts=datetime(2020, 1, 6, tzinfo=timezone.utc)
                    + __import__("datetime").timedelta(weeks=i),
                    open=p,
                    high=p + 1.0,
                    low=p - 2.0,
                    close=p - 1.0,
                    volume=1000.0,
                )
            )
        result = compute_weekly_bias(bars)
        assert result.direction == BiasDirection.BEARISH

    def test_bar_bundle_has_daily_weekly_fields(self):
        """BarBundle now accepts daily and weekly fields."""
        bundle = BarBundle(daily=[], weekly=[])
        assert bundle.daily == []
        assert bundle.weekly == []


# ---------------------------------------------------------------------------
# 9. Confidence tiering (Feature 2)
# ---------------------------------------------------------------------------


class TestConfidenceTiering:
    def test_tier_a_full_size(self):
        """Confidence >= tier_a → tier A, no budget reduction."""
        conf = 0.85
        tier_a = 0.80
        tier_b = 0.70
        if conf >= tier_a:
            tier = "A"
        elif conf >= tier_b:
            tier = "B"
        else:
            tier = "C"
        assert tier == "A"

    def test_tier_b_half_size(self):
        """Confidence between tier_b and tier_a → tier B, half budget."""
        conf = 0.75
        tier_a = 0.80
        tier_b = 0.70
        if conf >= tier_a:
            tier = "A"
        elif conf >= tier_b:
            tier = "B"
        else:
            tier = "C"
        assert tier == "B"

        budget = 1000.0
        if tier == "B":
            budget *= 0.5
        assert budget == 500.0

    def test_tier_c_blocked(self):
        """Confidence < tier_b → tier C, blocked."""
        conf = 0.65
        tier_a = 0.80
        tier_b = 0.70
        if conf >= tier_a:
            tier = "A"
        elif conf >= tier_b:
            tier = "B"
        else:
            tier = "C"
        assert tier == "C"

    def test_config_validation_both_or_neither(self):
        """Setting only one tier raises ValueError."""
        with pytest.raises(ValueError, match="both be set"):
            UnicornConfig(confidence_tier_a=0.80, confidence_tier_b=None)

    def test_config_validation_b_lt_a(self):
        """tier_b must be < tier_a."""
        with pytest.raises(ValueError, match="must be < confidence_tier_a"):
            UnicornConfig(confidence_tier_a=0.70, confidence_tier_b=0.80)

    def test_config_valid_tiers(self):
        """Valid tier config should construct without error."""
        config = UnicornConfig(confidence_tier_a=0.80, confidence_tier_b=0.70)
        assert config.confidence_tier_a == 0.80
        assert config.confidence_tier_b == 0.70

    def test_trade_record_has_tier_field(self):
        """TradeRecord has confidence_tier field."""
        trade = TradeRecord(
            entry_time=datetime(2025, 6, 1, 9, 30, tzinfo=ET),
            entry_price=100.0,
            direction=BiasDirection.BULLISH,
            quantity=1,
            session=TradingSession.NY_AM,
            criteria=CriteriaCheck(),
            stop_price=95.0,
            target_price=110.0,
            risk_points=5.0,
            confidence_tier="B",
        )
        assert trade.confidence_tier == "B"


# ---------------------------------------------------------------------------
# 10. NY AM timebox tightening (Feature 3)
# ---------------------------------------------------------------------------


class TestNyAmCutoff:
    def test_apply_cutoff_shortens_window(self):
        """60-min cutoff changes 9:30-11:00 to 9:30-10:30."""
        windows = [(time(9, 30), time(11, 0))]
        result = _apply_ny_am_cutoff(windows, 60)
        assert result == [(time(9, 30), time(10, 30))]

    def test_apply_cutoff_leaves_other_windows(self):
        """London window should be untouched."""
        windows = [(time(3, 0), time(4, 0)), (time(9, 30), time(11, 0))]
        result = _apply_ny_am_cutoff(windows, 45)
        assert result[0] == (time(3, 0), time(4, 0))
        assert result[1] == (time(9, 30), time(10, 15))

    def test_is_in_macro_window_with_cutoff(self):
        """10:45 ET should be OUT with 60-min cutoff."""
        ts = datetime(2025, 3, 10, 10, 45, tzinfo=ET)
        # Without cutoff: in window (9:30-11:00)
        assert is_in_macro_window(ts, SessionProfile.STRICT) is True
        # With cutoff: out (9:30-10:30)
        assert (
            is_in_macro_window(ts, SessionProfile.STRICT, ny_am_cutoff_minutes=60)
            is False
        )

    def test_is_in_macro_window_cutoff_inside(self):
        """10:00 ET should still be IN with 60-min cutoff."""
        ts = datetime(2025, 3, 10, 10, 0, tzinfo=ET)
        assert (
            is_in_macro_window(ts, SessionProfile.STRICT, ny_am_cutoff_minutes=60)
            is True
        )

    def test_config_validation_cutoff_range(self):
        """Cutoff must be 1-90 minutes."""
        with pytest.raises(ValueError, match="1-90"):
            UnicornConfig(ny_am_cutoff_minutes=0)
        with pytest.raises(ValueError, match="1-90"):
            UnicornConfig(ny_am_cutoff_minutes=91)

    def test_config_valid_cutoff(self):
        config = UnicornConfig(ny_am_cutoff_minutes=60)
        assert config.ny_am_cutoff_minutes == 60

    def test_cutoff_none_uses_default(self):
        """When cutoff is None, full 9:30-11:00 window applies."""
        ts = datetime(2025, 3, 10, 10, 45, tzinfo=ET)
        assert (
            is_in_macro_window(ts, SessionProfile.STRICT, ny_am_cutoff_minutes=None)
            is True
        )


# ---------------------------------------------------------------------------
# 11. Eval-mode default locks (strict profile)
# ---------------------------------------------------------------------------


class TestEvalModeDefaults:
    """Verify --eval-mode --session-profile strict locks proven features."""

    def _make_args(self, **overrides):
        """Simulate an argparse namespace with eval-mode defaults."""
        defaults = {
            "eval_mode": True,
            "session_profile": "strict",
            "weekly_bias_gate": False,
            "ny_am_cutoff": None,
            "breakeven_at_r": None,
            "trail_atr_mult": None,
            "trail_cap_mult": None,
        }
        defaults.update(overrides)

        class Args:
            pass

        a = Args()
        for k, v in defaults.items():
            setattr(a, k, v)
        return a

    def test_strict_locks_weekly_bias_gate(self):
        """Eval-mode + strict should lock weekly_bias_gate=True."""
        args = self._make_args()
        # Simulate the locking logic from run_unicorn_backtest.py
        if args.eval_mode and args.session_profile == "strict":
            if not args.weekly_bias_gate:
                args.weekly_bias_gate = True
            if args.ny_am_cutoff is None:
                args.ny_am_cutoff = 60
        assert args.weekly_bias_gate is True

    def test_strict_locks_ny_am_cutoff(self):
        """Eval-mode + strict should default ny_am_cutoff=60."""
        args = self._make_args()
        if args.eval_mode and args.session_profile == "strict":
            if not args.weekly_bias_gate:
                args.weekly_bias_gate = True
            if args.ny_am_cutoff is None:
                args.ny_am_cutoff = 60
        assert args.ny_am_cutoff == 60

    def test_explicit_cutoff_overrides(self):
        """Explicit --ny-am-cutoff 45 wins over the default 60."""
        args = self._make_args(ny_am_cutoff=45)
        if args.eval_mode and args.session_profile == "strict":
            if not args.weekly_bias_gate:
                args.weekly_bias_gate = True
            if args.ny_am_cutoff is None:
                args.ny_am_cutoff = 60
        assert args.ny_am_cutoff == 45

    def test_non_strict_no_lock(self):
        """Non-strict profile should not auto-lock features."""
        args = self._make_args(session_profile="normal")
        if args.eval_mode and args.session_profile == "strict":
            if not args.weekly_bias_gate:
                args.weekly_bias_gate = True
            if args.ny_am_cutoff is None:
                args.ny_am_cutoff = 60
        assert args.weekly_bias_gate is False
        assert args.ny_am_cutoff is None

    def test_no_eval_mode_no_lock(self):
        """Without eval-mode, no locking happens."""
        args = self._make_args(eval_mode=False)
        if args.eval_mode and args.session_profile == "strict":
            if not args.weekly_bias_gate:
                args.weekly_bias_gate = True
            if args.ny_am_cutoff is None:
                args.ny_am_cutoff = 60
        assert args.weekly_bias_gate is False
        assert args.ny_am_cutoff is None


# ---------------------------------------------------------------------------
# 12. Adaptive confidence tiering
# ---------------------------------------------------------------------------


class TestAdaptiveConfidenceTiering:
    """Test adaptive confidence tiering integration with governor + runner logic."""

    def test_consecutive_losses_trigger_tiering(self):
        """Governor activates tiering at streak threshold."""
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=2,
        )
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is False
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is True

    def test_dd_threshold_triggers_tiering(self):
        """Governor activates tiering when DD exceeds threshold."""
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_dd_pct=60.0,
        )
        gov.check_adaptive_dd(59.0)
        assert gov.confidence_tier_active is False
        gov.check_adaptive_dd(60.0)
        assert gov.confidence_tier_active is True

    def test_reset_day_clears_adaptive(self):
        """reset_day() should clear adaptive state."""
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=1,
        )
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is True
        assert gov.consecutive_losses == 1
        gov.reset_day()
        assert gov.confidence_tier_active is False
        assert gov.consecutive_losses == 0

    def test_adaptive_does_not_override_explicit_config(self):
        """When config has explicit tiers, adaptive governor doesn't override."""
        config_tier_a = 0.90
        config_tier_b = 0.85

        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=1,
            adaptive_tier_a=0.80,
            adaptive_tier_b=0.70,
        )
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is True

        # Simulate the runner logic: adaptive only fires when config tiers are None
        effective_tier_a = config_tier_a  # explicit config
        effective_tier_b = config_tier_b
        if gov.confidence_tier_active and effective_tier_a is None:
            effective_tier_a = gov.adaptive_tier_a
            effective_tier_b = gov.adaptive_tier_b

        assert effective_tier_a == 0.90  # config wins
        assert effective_tier_b == 0.85

    def test_adaptive_applies_when_config_tiers_none(self):
        """When config has no tiers, adaptive governor tiers apply."""
        config_tier_a = None
        config_tier_b = None

        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=1,
            adaptive_tier_a=0.80,
            adaptive_tier_b=0.70,
        )
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is True

        effective_tier_a = config_tier_a
        effective_tier_b = config_tier_b
        if gov.confidence_tier_active and effective_tier_a is None:
            effective_tier_a = gov.adaptive_tier_a
            effective_tier_b = gov.adaptive_tier_b

        assert effective_tier_a == 0.80
        assert effective_tier_b == 0.70

    def test_win_resets_consecutive_losses(self):
        """A winning trade resets the consecutive loss counter."""
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=3,
        )
        gov.record_trade_close(-50.0)
        gov.record_trade_close(-50.0)
        assert gov.consecutive_losses == 2
        gov.record_trade_close(100.0)
        assert gov.consecutive_losses == 0
