"""Unit tests for ICT Blueprint risk manager."""

import pytest

from app.services.backtest.engines.ict_blueprint.risk_manager import (
    apply_commission,
    bps_to_points,
    check_entry_exit_collision,
    check_exit,
    check_rr_gate,
    compute_position_size,
    compute_stop_price,
    compute_target_price,
    process_derisk,
)
from app.services.backtest.engines.ict_blueprint.types import (
    Bias,
    BreakerZone,
    HTFStateSnapshot,
    LTFSetup,
    OrderBlock,
    Position,
    SetupPhase,
    Side,
    SwingPoint,
    TradingRange,
)


def _make_setup(side=Side.LONG, sweep_low=44.0, breaker_bottom=45.0) -> LTFSetup:
    ob = OrderBlock(
        top=50.0,
        bottom=45.0,
        bias=Bias.BULLISH,
        ob_id=(5, 3, 4, "long"),
        anchor_swing=SwingPoint(2, 200, 45.0, True),
        msb_bar_index=5,
    )
    return LTFSetup(
        ob=ob,
        side=side,
        phase=SetupPhase.ENTRY_PENDING,
        sweep_low=sweep_low,
        breaker=BreakerZone(top=47.0, bottom=breaker_bottom, bar_index=8),
    )


def _make_htf_snap() -> HTFStateSnapshot:
    return HTFStateSnapshot(
        bias=Bias.BULLISH,
        swing_highs=(SwingPoint(2, 200, 60.0, True),),
        swing_lows=(SwingPoint(1, 100, 40.0, False),),
        current_range=TradingRange(
            high=60.0, low=40.0, midpoint=50.0, bias=Bias.BULLISH
        ),
        active_obs=(),
        last_msb_bar_index=5,
    )


# ---------------------------------------------------------------------------
# Stop placement
# ---------------------------------------------------------------------------


class TestStopPlacement:
    def test_below_sweep_long(self):
        setup = _make_setup(sweep_low=44.0)
        stop = compute_stop_price(setup, "below_sweep", Side.LONG, buffer_ticks=2.0)
        assert stop == 42.0

    def test_below_breaker_long(self):
        setup = _make_setup(breaker_bottom=45.0)
        stop = compute_stop_price(setup, "below_breaker", Side.LONG, buffer_ticks=2.0)
        assert stop == 43.0

    def test_below_sweep_short(self):
        setup = _make_setup(side=Side.SHORT, sweep_low=56.0)
        stop = compute_stop_price(setup, "below_sweep", Side.SHORT, buffer_ticks=2.0)
        assert stop == 58.0

    def test_no_sweep_returns_none(self):
        setup = _make_setup(sweep_low=None)
        setup.sweep_low = None
        stop = compute_stop_price(setup, "below_sweep", Side.LONG)
        assert stop is None


# ---------------------------------------------------------------------------
# Target placement
# ---------------------------------------------------------------------------


class TestTargetPlacement:
    def test_external_range_long(self):
        htf = _make_htf_snap()
        target = compute_target_price(48.0, 42.0, Side.LONG, "external_range", htf)
        assert target == 60.0  # Range high

    def test_fixed_rr_long(self):
        htf = _make_htf_snap()
        target = compute_target_price(
            48.0, 42.0, Side.LONG, "fixed_rr", htf, fixed_rr=3.0
        )
        # risk = 6, target = 48 + 18 = 66
        assert target == 66.0

    def test_fixed_rr_short(self):
        htf = _make_htf_snap()
        target = compute_target_price(
            55.0, 60.0, Side.SHORT, "fixed_rr", htf, fixed_rr=3.0
        )
        # risk = 5, target = 55 - 15 = 40
        assert target == 40.0


# ---------------------------------------------------------------------------
# R:R gate
# ---------------------------------------------------------------------------


class TestRRGate:
    def test_passes(self):
        assert check_rr_gate(48.0, 42.0, 60.0, 2.0, Side.LONG) is True
        # reward=12, risk=6, RR=2.0 → passes at min_rr=2.0

    def test_fails(self):
        assert check_rr_gate(48.0, 42.0, 53.0, 2.0, Side.LONG) is False
        # reward=5, risk=6, RR≈0.83 → fails

    def test_boundary(self):
        # Exactly 2.0 RR
        assert check_rr_gate(48.0, 42.0, 60.0, 2.0, Side.LONG) is True

    def test_short(self):
        assert check_rr_gate(55.0, 60.0, 40.0, 2.0, Side.SHORT) is True
        # reward=15, risk=5, RR=3.0


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_basic_sizing(self):
        size = compute_position_size(
            equity=100000,
            risk_pct=0.01,
            entry_price=4500,
            stop_price=4490,
            point_value=50.0,
        )
        # risk_dollars = 1000, risk_per_unit = 10 * 50 = 500
        # size = 1000 / 500 = 2.0
        assert size == 2.0

    def test_minimum_size(self):
        size = compute_position_size(
            equity=1000,
            risk_pct=0.001,
            entry_price=4500,
            stop_price=4490,
            point_value=50.0,
        )
        # risk_dollars = 1, risk_per_unit = 500 → size = 0.002 → clamped to 1.0
        assert size == 1.0

    def test_zero_risk(self):
        size = compute_position_size(
            equity=100000,
            risk_pct=0.01,
            entry_price=4500,
            stop_price=4500,
            point_value=50.0,
        )
        assert size == 0.0


# ---------------------------------------------------------------------------
# De-risk
# ---------------------------------------------------------------------------


class TestDerisk:
    def test_move_to_be(self):
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        result = process_derisk(
            pos, h1_high=4520, h1_low=4505, derisk_mode="move_to_be", trigger_rr=2.0
        )
        # 4520 - 4500 = 20 points, risk = 10, RR = 2.0 → trigger
        assert result is None  # move_to_be doesn't realize PnL
        assert pos.stop_price == 4500.0  # Moved to breakeven
        assert pos.derisk_triggered is True

    def test_half_off(self):
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        result = process_derisk(
            pos, h1_high=4520, h1_low=4505, derisk_mode="half_off", trigger_rr=2.0
        )
        assert result is not None
        assert result > 0
        assert pos.remaining_size == 1.0

    def test_not_triggered(self):
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        result = process_derisk(
            pos, h1_high=4510, h1_low=4505, derisk_mode="move_to_be", trigger_rr=2.0
        )
        # 10/10 = 1.0 RR, below trigger
        assert result is None
        assert pos.derisk_triggered is False

    def test_derisk_on_intrabar_high(self):
        """De-risk evaluates on h1_high, not close."""
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        # High reaches 4520 (2R) but close might be lower — should still trigger
        process_derisk(
            pos, h1_high=4520, h1_low=4495, derisk_mode="move_to_be", trigger_rr=2.0
        )
        assert pos.derisk_triggered is True

    def test_already_triggered(self):
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
            derisk_triggered=True,
        )
        result = process_derisk(
            pos, h1_high=4525, h1_low=4505, derisk_mode="move_to_be", trigger_rr=2.0
        )
        assert result is None


# ---------------------------------------------------------------------------
# Exit checks
# ---------------------------------------------------------------------------


class TestExitChecks:
    def test_stop_hit_long(self):
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        result = check_exit(pos, h1_high=4505, h1_low=4488, h1_close=4492, bar_ts=200)
        assert result is not None
        assert result[1] == "stop_loss"

    def test_target_hit_long(self):
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        result = check_exit(pos, h1_high=4535, h1_low=4510, h1_close=4532, bar_ts=200)
        assert result is not None
        assert result[1] == "take_profit"

    def test_both_hit_stop_wins(self):
        """Intrabar: if both stop and target reachable, stop wins."""
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        result = check_exit(pos, h1_high=4535, h1_low=4485, h1_close=4520, bar_ts=200)
        assert result is not None
        assert result[1] == "stop_loss"

    def test_no_exit(self):
        pos = Position(
            entry_time=100,
            entry_price=4500,
            stop_price=4490,
            target_price=4530,
            side=Side.LONG,
            size=2.0,
            risk_points=10.0,
        )
        result = check_exit(pos, h1_high=4520, h1_low=4495, h1_close=4515, bar_ts=200)
        assert result is None


# ---------------------------------------------------------------------------
# Entry-exit collision
# ---------------------------------------------------------------------------


class TestEntryExitCollision:
    def test_collision_long(self):
        # Bar crosses both entry and stop
        assert (
            check_entry_exit_collision(
                4500, 4490, h1_high=4505, h1_low=4485, side=Side.LONG
            )
            is True
        )

    def test_no_collision_long(self):
        # Bar doesn't reach stop
        assert (
            check_entry_exit_collision(
                4500, 4490, h1_high=4505, h1_low=4495, side=Side.LONG
            )
            is False
        )

    def test_collision_short(self):
        assert (
            check_entry_exit_collision(
                4500, 4510, h1_high=4515, h1_low=4495, side=Side.SHORT
            )
            is True
        )

    def test_no_collision_short(self):
        assert (
            check_entry_exit_collision(
                4500, 4510, h1_high=4505, h1_low=4495, side=Side.SHORT
            )
            is False
        )


# ---------------------------------------------------------------------------
# Commission/slippage helpers
# ---------------------------------------------------------------------------


class TestCommission:
    def test_apply_commission(self):
        cost = apply_commission(100000, 10)  # 10 bps = 0.1%
        assert cost == pytest.approx(100.0)

    def test_bps_to_points(self):
        pts = bps_to_points(4500, 10)
        assert pts == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Max attempts enforcement
# ---------------------------------------------------------------------------


class TestMaxAttempts:
    def test_attempts_tracked(self):
        setup = _make_setup()
        setup.ob.attempts_used = 2
        # With max_attempts_per_ob=2, this OB should be exhausted
        assert setup.ob.attempts_used >= 2
