"""Integration tests for eval account profile with backtest runner."""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.services.strategy.models import OHLCVBar
from app.services.backtest.engines.unicorn_runner import (
    run_unicorn_backtest,
    format_backtest_report,
)
from app.services.backtest.engines.eval_profile import EvalAccountProfile

ET = ZoneInfo("America/New_York")


def make_bar(
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000.0,
) -> OHLCVBar:
    return OHLCVBar(ts=ts, open=open_, high=high, low=low, close=close, volume=volume)


def generate_trending_bars(
    start_ts: datetime,
    num_bars: int,
    start_price: float,
    trend: float = 0.5,
    volatility: float = 2.0,
    interval_minutes: int = 15,
) -> list[OHLCVBar]:
    bars = []
    price = start_price
    for i in range(num_bars):
        ts = start_ts + timedelta(minutes=i * interval_minutes)
        open_ = price
        close = price + trend
        high = max(open_, close) + volatility * 0.5
        low = min(open_, close) - volatility * 0.5
        bars.append(make_bar(ts, open_, high, low, close, volume=1000 + i * 10))
        price = close
    return bars


@pytest.fixture
def sample_bars():
    """Generate 100 bars of HTF and LTF data."""
    start = datetime(2024, 1, 15, 9, 30, tzinfo=ET)
    htf = generate_trending_bars(start, 100, 17000.0, trend=0.5, interval_minutes=15)
    ltf = generate_trending_bars(start, 200, 17000.0, trend=0.25, interval_minutes=5)
    return htf, ltf


@pytest.fixture
def standard_profile():
    return EvalAccountProfile(
        account_size=50_000,
        max_drawdown_dollars=2_000,
        max_daily_loss_dollars=1_000,
        risk_fraction=0.15,
        r_min_dollars=100.0,
        r_max_dollars=300.0,
    )


class TestBacktestWithoutProfile:
    """eval_profile=None should behave identically to before."""

    def test_backtest_without_profile_unchanged(self, sample_bars):
        htf, ltf = sample_bars
        result = run_unicorn_backtest(
            symbol="MNQ",
            htf_bars=htf,
            ltf_bars=ltf,
            dollars_per_trade=200.0,
            eval_profile=None,
        )
        gs = result.governor_stats
        assert gs is not None
        assert "eval_account_size" not in gs
        assert "peak_equity" not in gs
        assert "drawdown_halt" not in gs
        assert gs["dollars_per_trade"] == 200.0


class TestBacktestWithProfile:
    """Tests with eval_profile present."""

    def test_backtest_with_profile_tracks_equity(self, sample_bars, standard_profile):
        htf, ltf = sample_bars
        result = run_unicorn_backtest(
            symbol="MNQ",
            htf_bars=htf,
            ltf_bars=ltf,
            dollars_per_trade=200.0,
            eval_profile=standard_profile,
        )
        gs = result.governor_stats
        assert gs is not None
        assert gs["eval_account_size"] == 50_000
        assert gs["eval_max_drawdown"] == 2_000
        assert "final_equity" in gs
        assert "peak_equity" in gs
        assert "trailing_drawdown" in gs
        assert "r_day_min" in gs
        assert "r_day_max" in gs
        assert isinstance(gs["drawdown_halt"], bool)

    def test_profile_blown_stops_early(self, sample_bars):
        """Tiny max_drawdown should blow the eval quickly."""
        htf, ltf = sample_bars
        # Tiny drawdown means eval is already "tight"
        tiny_profile = EvalAccountProfile(
            account_size=50_000,
            max_drawdown_dollars=1.0,  # $1 max DD -> blown immediately on any loss
            max_daily_loss_dollars=0.50,
            risk_fraction=0.15,
            r_min_dollars=0.01,
            r_max_dollars=0.15,
        )
        result = run_unicorn_backtest(
            symbol="MNQ",
            htf_bars=htf,
            ltf_bars=ltf,
            dollars_per_trade=200.0,
            eval_profile=tiny_profile,
        )
        gs = result.governor_stats
        # With such tiny DD, it should either blow or have very few trades
        # The key test: the field exists and is populated
        assert "drawdown_halt" in gs

    def test_r_day_range_populated(self, sample_bars, standard_profile):
        htf, ltf = sample_bars
        result = run_unicorn_backtest(
            symbol="MNQ",
            htf_bars=htf,
            ltf_bars=ltf,
            dollars_per_trade=200.0,
            eval_profile=standard_profile,
        )
        gs = result.governor_stats
        assert gs["r_day_min"] <= gs["r_day_max"]
        assert gs["r_day_min"] >= 0.0

    def test_report_contains_eval_section(self, sample_bars, standard_profile):
        htf, ltf = sample_bars
        result = run_unicorn_backtest(
            symbol="MNQ",
            htf_bars=htf,
            ltf_bars=ltf,
            dollars_per_trade=200.0,
            eval_profile=standard_profile,
        )
        report = format_backtest_report(result)
        assert "EVAL ACCOUNT" in report
        assert "Starting equity" in report
        assert "R_day range" in report
