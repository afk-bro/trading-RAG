"""
Unit tests for Unicorn Model backtest runner.
"""

from datetime import datetime, timedelta
import pytest

from app.services.strategy.models import OHLCVBar
from app.services.backtest.engines.unicorn_runner import (
    TradingSession,
    classify_session,
    check_criteria,
    run_unicorn_backtest,
    format_backtest_report,
    CriteriaCheck,
)
from app.services.strategy.indicators.tf_bias import BiasDirection


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


def generate_trending_bars(
    start_ts: datetime,
    num_bars: int,
    start_price: float,
    trend: float = 0.5,  # Price change per bar
    volatility: float = 2.0,
    interval_minutes: int = 15,
) -> list[OHLCVBar]:
    """Generate synthetic trending price data."""
    bars = []
    price = start_price

    for i in range(num_bars):
        ts = start_ts + timedelta(minutes=i * interval_minutes)
        open_ = price
        close = price + trend

        # Add some volatility
        high = max(open_, close) + volatility * 0.5
        low = min(open_, close) - volatility * 0.5

        bars.append(make_bar(ts, open_, high, low, close, volume=1000 + i * 10))
        price = close

    return bars


class TestSessionClassification:
    """Tests for trading session classification."""

    def test_ny_am_session(self):
        """Correctly identifies NY AM session."""
        ts = datetime(2024, 1, 15, 10, 30)
        assert classify_session(ts) == TradingSession.NY_AM

    def test_ny_pm_session(self):
        """Correctly identifies NY PM session."""
        ts = datetime(2024, 1, 15, 14, 0)
        assert classify_session(ts) == TradingSession.NY_PM

    def test_london_session(self):
        """Correctly identifies London session."""
        ts = datetime(2024, 1, 15, 3, 30)
        assert classify_session(ts) == TradingSession.LONDON

    def test_asia_session(self):
        """Correctly identifies Asia session."""
        ts = datetime(2024, 1, 15, 19, 30)
        assert classify_session(ts) == TradingSession.ASIA

    def test_off_hours(self):
        """Correctly identifies off-hours."""
        ts = datetime(2024, 1, 15, 12, 0)  # Between sessions
        assert classify_session(ts) == TradingSession.OFF_HOURS


class TestCriteriaCheck:
    """Tests for criteria checking."""

    def test_criteria_count(self):
        """Criteria count works correctly."""
        check = CriteriaCheck()
        assert check.criteria_met_count == 0

        check.htf_bias_aligned = True
        check.liquidity_sweep_found = True
        assert check.criteria_met_count == 2

    def test_all_criteria_met(self):
        """All criteria met flag works."""
        check = CriteriaCheck()
        assert not check.all_criteria_met

        check.htf_bias_aligned = True
        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = True
        check.mss_found = True
        check.stop_valid = True
        check.in_macro_window = True

        assert check.all_criteria_met
        assert check.criteria_met_count == 8

    def test_missing_criteria(self):
        """Missing criteria list is correct."""
        check = CriteriaCheck()
        check.htf_bias_aligned = True
        check.liquidity_sweep_found = True

        missing = check.missing_criteria()
        assert "htf_fvg" in missing
        assert "breaker_block" in missing
        assert "htf_bias" not in missing
        assert "liquidity_sweep" not in missing


class TestBacktestRunner:
    """Tests for the backtest runner."""

    def test_insufficient_data_raises(self):
        """Runner raises on insufficient data."""
        bars = [make_bar(datetime.now(), 100, 101, 99, 100)]

        with pytest.raises(ValueError, match="Insufficient bars"):
            run_unicorn_backtest("NQ", bars, bars)

    def test_backtest_with_synthetic_data(self):
        """Runner completes with synthetic data."""
        # Generate enough bars for backtest
        start_ts = datetime(2024, 1, 2, 9, 30)  # Start in NY AM
        htf_bars = generate_trending_bars(
            start_ts, 200, 17000, trend=2.0, interval_minutes=15
        )
        ltf_bars = generate_trending_bars(
            start_ts, 600, 17000, trend=0.67, interval_minutes=5
        )

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        # Should complete without error
        assert result.symbol == "NQ"
        assert result.total_bars == 200
        assert result.total_setups_scanned > 0

    def test_report_formatting(self):
        """Report formatter produces valid output."""
        start_ts = datetime(2024, 1, 2, 9, 30)
        htf_bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 300, 17000, interval_minutes=5)

        result = run_unicorn_backtest("NQ", htf_bars, ltf_bars)
        report = format_backtest_report(result)

        assert "UNICORN MODEL BACKTEST REPORT" in report
        assert "NQ" in report
        assert "SETUP ANALYSIS" in report
        assert "TRADE RESULTS" in report
        assert "SESSION BREAKDOWN" in report
        assert "CRITERIA BOTTLENECK" in report


class TestCriteriaChecker:
    """Tests for the criteria checker function."""

    def test_check_criteria_returns_check_object(self):
        """check_criteria returns a CriteriaCheck object."""
        start_ts = datetime(2024, 1, 2, 10, 0)  # In NY AM
        bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)

        # Use timestamp within NY AM window (10:30)
        check_ts = datetime(2024, 1, 2, 10, 30)

        result = check_criteria(
            bars=bars,
            htf_bars=bars,
            ltf_bars=bars[-60:],
            symbol="NQ",
            ts=check_ts,
        )

        assert isinstance(result, CriteriaCheck)
        assert result.session == TradingSession.NY_AM  # Should be in NY AM

    def test_check_criteria_detects_macro_window(self):
        """Macro window detection works."""
        # In macro window
        ts_in = datetime(2024, 1, 2, 10, 0)
        bars = generate_trending_bars(ts_in, 60, 17000, interval_minutes=15)

        result = check_criteria(bars, bars, bars[-30:], "NQ", ts_in)
        assert result.in_macro_window is True

        # Outside macro window
        ts_out = datetime(2024, 1, 2, 12, 0)
        bars_out = generate_trending_bars(ts_out, 60, 17000, interval_minutes=15)

        result_out = check_criteria(bars_out, bars_out, bars_out[-30:], "NQ", ts_out)
        assert result_out.in_macro_window is False
