"""Tests for the daily risk governor."""

from datetime import date

from app.services.backtest.engines.daily_governor import (
    DailyGovernor,
    DailyGovernorStats,
)


class TestDailyGovernorInit:
    def test_default_state_allows_trading(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        assert gov.allows_entry() is True
        assert gov.risk_multiplier == 1.0
        assert gov.halted_for_day is False

    def test_custom_params(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=600.0,
            max_trades_per_day=3,
            half_size_multiplier=0.25,
        )
        assert gov.max_daily_loss_dollars == 600.0
        assert gov.max_trades_per_day == 3
        assert gov.half_size_multiplier == 0.25


class TestDailyGovernorReset:
    def test_reset_clears_state(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.day_loss_dollars = -200.0
        gov.day_trade_count = 2
        gov.risk_multiplier = 0.5
        gov.halted_for_day = True

        gov.reset_day()

        assert gov.day_loss_dollars == 0.0
        assert gov.day_trade_count == 0
        assert gov.risk_multiplier == 1.0
        assert gov.halted_for_day is False
        assert gov.halt_reason == ""


class TestGovernorHalt:
    def test_halts_after_full_loss(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-300.0)
        assert gov.halted_for_day is True
        assert gov.halt_reason == "loss_limit"
        assert gov.allows_entry() is False

    def test_halts_after_exceeding_max_trades(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.record_trade_close(100.0)  # win
        gov.record_trade_close(50.0)  # win
        # 2 trades taken, at cap
        assert gov.allows_entry() is False
        assert gov.halt_reason == "trade_limit"

    def test_not_halted_after_winning_trade(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(200.0)
        assert gov.halted_for_day is False
        assert gov.allows_entry() is True
        assert gov.risk_multiplier == 1.0


class TestGovernorStepdown:
    def test_half_loss_triggers_stepdown(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-150.0)  # exactly half_loss_threshold
        assert gov.risk_multiplier == 0.5
        assert gov.halted_for_day is False  # not fully halted yet
        assert gov.allows_entry() is True  # can still trade at half size

    def test_small_loss_no_stepdown(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-100.0)  # below half threshold
        assert gov.risk_multiplier == 1.0

    def test_stepdown_then_halt_on_second_loss(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-150.0)  # half -> stepdown to 0.5
        assert gov.risk_multiplier == 0.5
        gov.record_trade_close(-150.0)  # cumulative -300 -> halt
        assert gov.halted_for_day is True
        assert gov.allows_entry() is False

    def test_custom_half_size_multiplier(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=600.0,
            max_trades_per_day=3,
            half_size_multiplier=0.25,
        )
        # half_loss_threshold = 600 * 0.25 = 150
        gov.record_trade_close(-150.0)
        assert gov.risk_multiplier == 0.25


class TestGovernorDayReset:
    def test_maybe_reset_on_new_date(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.maybe_reset(date(2024, 1, 2))
        gov.record_trade_close(-300.0)
        assert gov.halted_for_day is True

        gov.maybe_reset(date(2024, 1, 3))  # new day
        assert gov.halted_for_day is False
        assert gov.allows_entry() is True
        assert gov.risk_multiplier == 1.0

    def test_maybe_reset_same_date_noop(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.maybe_reset(date(2024, 1, 2))
        gov.record_trade_close(-200.0)
        gov.maybe_reset(date(2024, 1, 2))  # same day, no reset
        assert gov.day_loss_dollars == -200.0

    def test_maybe_reset_returns_halt_reason(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=2)
        gov.maybe_reset(date(2024, 1, 2))
        gov.record_trade_close(-300.0)
        assert gov.halted_for_day is True

        halt_reason = gov.maybe_reset(date(2024, 1, 3))
        assert halt_reason == "loss_limit"

        halt_reason = gov.maybe_reset(date(2024, 1, 4))
        assert halt_reason == ""  # wasn't halted on day 3

    def test_maybe_reset_returns_trade_limit_reason(self):
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=1)
        gov.maybe_reset(date(2024, 1, 2))
        gov.record_trade_close(100.0)  # 1 trade taken
        gov.allows_entry()  # triggers trade_limit halt
        assert gov.halt_reason == "trade_limit"

        halt_reason = gov.maybe_reset(date(2024, 1, 3))
        assert halt_reason == "trade_limit"


class TestGovernorWinsOnlyTrack:
    def test_winning_trade_does_not_reduce_day_loss(self):
        """day_loss_dollars only accumulates losses, not wins."""
        gov = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=3)
        gov.record_trade_close(-200.0)
        gov.record_trade_close(500.0)  # big win
        # day_loss stays at -200, not offset by win
        assert gov.day_loss_dollars == -200.0


class TestDailyGovernorStats:
    def test_stats_track_halted_days_and_skipped(self):
        stats = DailyGovernorStats()
        stats.signals_skipped += 3
        stats.days_halted += 1
        stats.half_size_trades += 1

        assert stats.signals_skipped == 3
        assert stats.days_halted == 1
        assert stats.half_size_trades == 1

    def test_stats_default_all_zero(self):
        stats = DailyGovernorStats()
        assert stats.signals_skipped == 0
        assert stats.days_halted == 0
        assert stats.half_size_trades == 0
        assert stats.total_days_traded == 0


class TestConsecutiveLossTracking:
    def test_increments_on_loss(self):
        gov = DailyGovernor(max_daily_loss_dollars=1000.0, max_trades_per_day=5)
        gov.record_trade_close(-50.0)
        assert gov.consecutive_losses == 1
        gov.record_trade_close(-50.0)
        assert gov.consecutive_losses == 2

    def test_resets_on_win(self):
        gov = DailyGovernor(max_daily_loss_dollars=1000.0, max_trades_per_day=5)
        gov.record_trade_close(-50.0)
        gov.record_trade_close(-50.0)
        assert gov.consecutive_losses == 2
        gov.record_trade_close(100.0)
        assert gov.consecutive_losses == 0

    def test_partial_legs_dont_affect_streak(self):
        gov = DailyGovernor(max_daily_loss_dollars=1000.0, max_trades_per_day=5)
        gov.record_trade_close(-50.0)  # setup close -> streak=1
        gov.record_trade_close(-30.0, is_partial_leg=True)  # partial leg ignored
        assert gov.consecutive_losses == 1

    def test_resets_on_day_boundary(self):
        gov = DailyGovernor(max_daily_loss_dollars=1000.0, max_trades_per_day=5)
        gov.record_trade_close(-50.0)
        gov.record_trade_close(-50.0)
        assert gov.consecutive_losses == 2
        gov.reset_day()
        assert gov.consecutive_losses == 0


class TestAdaptiveConfidenceTieringGovernor:
    def test_streak_trigger(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=2,
        )
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is False
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is True

    def test_streak_not_triggered_below_threshold(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=3,
        )
        gov.record_trade_close(-50.0)
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is False

    def test_dd_trigger(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_dd_pct=50.0,
        )
        gov.check_adaptive_dd(49.0)
        assert gov.confidence_tier_active is False
        gov.check_adaptive_dd(50.0)
        assert gov.confidence_tier_active is True

    def test_dd_disabled_when_none(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_dd_pct=None,
        )
        gov.check_adaptive_dd(100.0)
        assert gov.confidence_tier_active is False

    def test_reset_clears_adaptive_state(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=1,
        )
        gov.record_trade_close(-50.0)
        assert gov.confidence_tier_active is True
        gov.reset_day()
        assert gov.confidence_tier_active is False
        assert gov.consecutive_losses == 0

    def test_adaptive_tier_defaults(self):
        gov = DailyGovernor(
            max_daily_loss_dollars=1000.0,
            max_trades_per_day=5,
            adaptive_tier_streak=2,
        )
        assert gov.adaptive_tier_a == 0.80
        assert gov.adaptive_tier_b == 0.70
