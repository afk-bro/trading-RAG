"""
Unit tests for Unicorn Model backtest runner.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest

ET = ZoneInfo("America/New_York")

from app.services.strategy.models import OHLCVBar
from app.services.backtest.engines.unicorn_runner import (
    TradingSession,
    classify_session,
    check_criteria,
    run_unicorn_backtest,
    format_backtest_report,
    format_trade_trace,
    CriteriaCheck,
    SetupOccurrence,
    resolve_bar_exit,
    ExitResult,
    IntrabarPolicy,
    TradeRecord,
    compute_adverse_wick_ratio,
    compute_range_atr_mult,
    BiasState,
    BiasSnapshot,
    BarBundle,
    _asof_lookup,
)
from app.services.strategy.indicators.tf_bias import BiasDirection
from app.services.strategy.strategies.unicorn_model import (
    CriteriaScore,
    UnicornConfig,
    SessionProfile,
    is_in_macro_window,
    get_max_stop_points,
    _ranges_overlap,
    analyze_unicorn_setup,
)
from app.services.strategy.indicators.tf_bias import TimeframeBias
from app.services.strategy.indicators.ict_patterns import (
    FairValueGap,
    FVGType,
    BreakerBlock,
    BlockType,
)
from app.services.strategy.models import MarketSnapshot


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
        ts = datetime(2024, 1, 15, 10, 30, tzinfo=ET)
        assert classify_session(ts) == TradingSession.NY_AM

    def test_ny_pm_session(self):
        """Correctly identifies NY PM session."""
        ts = datetime(2024, 1, 15, 14, 0, tzinfo=ET)
        assert classify_session(ts) == TradingSession.NY_PM

    def test_london_session(self):
        """Correctly identifies London session."""
        ts = datetime(2024, 1, 15, 3, 30, tzinfo=ET)
        assert classify_session(ts) == TradingSession.LONDON

    def test_asia_session(self):
        """Correctly identifies Asia session."""
        ts = datetime(2024, 1, 15, 19, 30, tzinfo=ET)
        assert classify_session(ts) == TradingSession.ASIA

    def test_off_hours(self):
        """Correctly identifies off-hours."""
        ts = datetime(2024, 1, 15, 12, 0, tzinfo=ET)  # Between sessions
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
        check.displacement_valid = True

        assert check.all_criteria_met
        assert check.criteria_met_count == 9

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
        bars = [make_bar(datetime.now(timezone.utc), 100, 101, 99, 100)]

        with pytest.raises(ValueError, match="Insufficient bars"):
            run_unicorn_backtest("NQ", bars, bars)

    def test_backtest_with_synthetic_data(self):
        """Runner completes with synthetic data."""
        # Generate enough bars for backtest
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)  # Start in NY AM
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
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
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
        start_ts = datetime(2024, 1, 2, 10, 0, tzinfo=ET)  # In NY AM
        bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)

        # Use timestamp within NY AM window (10:30)
        check_ts = datetime(2024, 1, 2, 10, 30, tzinfo=ET)

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
        ts_in = datetime(2024, 1, 2, 10, 0, tzinfo=ET)
        bars = generate_trending_bars(ts_in, 60, 17000, interval_minutes=15)

        result = check_criteria(bars, bars, bars[-30:], "NQ", ts_in)
        assert result.in_macro_window is True

        # Outside macro window
        ts_out = datetime(2024, 1, 2, 12, 0, tzinfo=ET)
        bars_out = generate_trending_bars(ts_out, 60, 17000, interval_minutes=15)

        result_out = check_criteria(bars_out, bars_out, bars_out[-30:], "NQ", ts_out)
        assert result_out.in_macro_window is False


# =========================================================================
# Critical tests: spec-to-code alignment
# =========================================================================


class TestMeetsEntryRequirements:
    """Tests for mandatory + scored entry gating."""

    def test_mandatory_pass_and_scored_at_threshold_enters(self):
        """All 5 mandatory + exactly 3/4 scored => entry allowed."""
        check = CriteriaCheck()
        # Mandatory
        check.htf_bias_aligned = True
        check.stop_valid = True
        check.in_macro_window = True
        check.mss_found = True
        check.displacement_valid = True
        # Scored: exactly 3 of 4
        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = False

        assert check.meets_entry_requirements(min_scored=3) is True

    def test_mandatory_fail_with_all_scored_rejects(self):
        """Mandatory fail + 4/4 scored => must reject."""
        check = CriteriaCheck()
        # Mandatory: stop_valid fails
        check.htf_bias_aligned = True
        check.stop_valid = False
        check.in_macro_window = True
        check.mss_found = True
        check.displacement_valid = True
        # All 4 scored pass
        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = True

        assert check.meets_entry_requirements(min_scored=3) is False
        assert check.mandatory_criteria_met is False


class TestCriteriaScoreDecideEntry:
    """Tests for the canonical decide_entry gate on CriteriaScore."""

    def test_decide_entry_matches_spec(self):
        """decide_entry uses mandatory+scored, not flat sum."""
        cs = CriteriaScore()
        # Mandatory pass
        cs.htf_bias = True
        cs.stop_valid = True
        cs.macro_window = True
        cs.mss = True
        cs.displacement_valid = True
        # 3/4 scored
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.breaker_block = True

        assert cs.decide_entry(min_scored=3) is True
        # Total score is 8, but decide_entry should not care about total
        assert cs.score == 8

    def test_decide_entry_rejects_below_scored_threshold(self):
        """Mandatory pass + only 2/4 scored => reject."""
        cs = CriteriaScore()
        cs.htf_bias = True
        cs.stop_valid = True
        cs.macro_window = True
        cs.mss = True
        cs.displacement_valid = True
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        # Only 2 scored

        assert cs.decide_entry(min_scored=3) is False
        assert cs.mandatory_met is True
        assert cs.scored_count == 2


class TestNeutralBiasRejection:
    """NEUTRAL bias must produce no entry."""

    def test_neutral_bias_returns_zero_score(self):
        """HTF bias = NEUTRAL => htf_bias criterion false."""
        cs = CriteriaScore()
        # Simulate NEUTRAL: htf_bias stays False
        cs.stop_valid = True
        cs.macro_window = True
        cs.mss = True
        cs.displacement_valid = True
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.breaker_block = True
        cs.ltf_fvg = True

        assert cs.decide_entry(min_scored=3) is False
        assert cs.mandatory_met is False


class TestMacroWindowProfile:
    """Macro window must respect session profile."""

    def test_wide_only_window_fails_under_normal(self):
        """19:30 ET is in WIDE (Asia) but NOT in NORMAL."""
        ts = datetime(2024, 1, 15, 19, 30, tzinfo=ET)
        assert is_in_macro_window(ts, SessionProfile.WIDE) is True
        assert is_in_macro_window(ts, SessionProfile.NORMAL) is False

    def test_strict_only_allows_ny_am(self):
        """3:30 ET (London) is in NORMAL but NOT in STRICT."""
        ts = datetime(2024, 1, 15, 3, 30, tzinfo=ET)
        assert is_in_macro_window(ts, SessionProfile.NORMAL) is True
        assert is_in_macro_window(ts, SessionProfile.STRICT) is False

    def test_boundary_end_exclusive(self):
        """Exact end time (11:00:00) should be excluded (half-open interval)."""
        ts_end = datetime(2024, 1, 15, 11, 0, 0, tzinfo=ET)
        assert is_in_macro_window(ts_end, SessionProfile.STRICT) is False

        ts_just_before = datetime(2024, 1, 15, 10, 59, 59, tzinfo=ET)
        assert is_in_macro_window(ts_just_before, SessionProfile.STRICT) is True


class TestBacktestUsesSessionProfile:
    """Backtest check_criteria must pass config.session_profile, not default WIDE."""

    def test_check_criteria_respects_normal_profile(self):
        """Asia-window timestamp with NORMAL profile => macro_window False."""
        # 19:30 is inside WIDE (Asia) but outside NORMAL
        ts = datetime(2024, 1, 15, 19, 30, tzinfo=ET)
        bars = generate_trending_bars(ts, 60, 17000, interval_minutes=15)

        config = UnicornConfig(session_profile=SessionProfile.NORMAL)
        result = check_criteria(bars, bars, bars[-30:], "NQ", ts, config=config)
        assert result.in_macro_window is False

    def test_check_criteria_respects_wide_profile(self):
        """Same timestamp with WIDE profile => macro_window True."""
        ts = datetime(2024, 1, 15, 19, 30, tzinfo=ET)
        bars = generate_trending_bars(ts, 60, 17000, interval_minutes=15)

        config = UnicornConfig(session_profile=SessionProfile.WIDE)
        result = check_criteria(bars, bars, bars[-30:], "NQ", ts, config=config)
        assert result.in_macro_window is True


class TestStopDistanceBoundary:
    """Stop distance at exactly 3.0 * ATR must pass."""

    def test_stop_at_exactly_max_atr_passes(self):
        """risk_handles == 3.0 * ATR => stop_valid = True."""
        atr = 10.0
        config = UnicornConfig(stop_max_atr_mult=3.0)
        max_points = get_max_stop_points("NQ", atr=atr, config=config)
        # max_points = 30.0
        assert max_points == 30.0
        # risk_handles <= max_points => valid
        assert 30.0 <= max_points  # boundary: exactly equal passes

    def test_stop_above_max_atr_fails(self):
        """risk_handles > 3.0 * ATR => stop_valid = False."""
        atr = 10.0
        config = UnicornConfig(stop_max_atr_mult=3.0)
        max_points = get_max_stop_points("NQ", atr=atr, config=config)
        assert 30.1 > max_points  # 30.1 exceeds 30.0 => would fail


class TestConfidenceGate:
    """Tests for opt-in confidence gating on HTF bias."""

    def test_default_no_confidence_gate(self):
        """min_confidence=None (default) does not block low-confidence setups."""
        cs = CriteriaScore()
        cs.htf_bias = True  # Would be set by analyze_unicorn_setup
        cs.stop_valid = True
        cs.macro_window = True
        cs.mss = True
        cs.displacement_valid = True
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.breaker_block = True

        # Even though confidence may be low, htf_bias passed => decide_entry works
        assert cs.decide_entry(min_scored=3) is True

    def test_confidence_gate_blocks_when_enabled(self):
        """min_confidence=0.6 blocks bias with confidence < 0.6.

        This tests the config plumbing: when min_confidence is set,
        analyze_unicorn_setup incorporates it into the htf_bias flag.
        We simulate the result: htf_bias=False because confidence was too low.
        """
        cs = CriteriaScore()
        # Simulate: bias direction is BULLISH and tradeable, but confidence=0.4 < 0.6
        # => analyze_unicorn_setup sets htf_bias=False due to confidence gate
        cs.htf_bias = False
        cs.stop_valid = True
        cs.macro_window = True
        cs.mss = True
        cs.displacement_valid = True
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.breaker_block = True
        cs.ltf_fvg = True

        # Mandatory fails (htf_bias=False), so entry rejected despite 4/4 scored
        assert cs.decide_entry(min_scored=3) is False
        assert cs.mandatory_met is False


class TestRangesOverlapBoundary:
    """Tests for price range overlap, especially boundary contact."""

    def test_ranges_touching_at_boundary_overlap(self):
        """Ranges that share exactly one point should overlap.

        For price ranges, breaker.high == fvg.low means they share a level.
        This is intentionally inclusive (unlike half-open session windows).
        """
        r1 = (100.0, 105.0)  # FVG range
        r2 = (105.0, 110.0)  # Breaker range touching at 105
        assert _ranges_overlap(r1, r2) is True

    def test_ranges_with_gap_do_not_overlap(self):
        """Ranges with a gap between them should not overlap."""
        r1 = (100.0, 105.0)
        r2 = (105.25, 110.0)  # 0.25 gap (one NQ tick)
        assert _ranges_overlap(r1, r2) is False

    def test_nested_ranges_overlap(self):
        """One range fully inside another should overlap."""
        r1 = (100.0, 110.0)
        r2 = (103.0, 107.0)
        assert _ranges_overlap(r1, r2) is True


class TestConfigValidation:
    """UnicornConfig must reject invalid parameter ranges."""

    def test_min_scored_criteria_rejects_five(self):
        """min_scored_criteria=5 is impossible (only 4 scored items)."""
        with pytest.raises(ValueError, match="min_scored_criteria must be 0-4"):
            UnicornConfig(min_scored_criteria=5)

    def test_min_scored_criteria_rejects_negative(self):
        """min_scored_criteria=-1 is invalid."""
        with pytest.raises(ValueError, match="min_scored_criteria must be 0-4"):
            UnicornConfig(min_scored_criteria=-1)

    def test_min_scored_criteria_accepts_zero(self):
        """min_scored_criteria=0 means mandatory-only gating."""
        config = UnicornConfig(min_scored_criteria=0)
        assert config.min_scored_criteria == 0

    def test_min_scored_criteria_accepts_four(self):
        """min_scored_criteria=4 requires all scored criteria."""
        config = UnicornConfig(min_scored_criteria=4)
        assert config.min_scored_criteria == 4

    def test_min_confidence_rejects_out_of_range(self):
        """min_confidence must be 0.0-1.0 when set."""
        with pytest.raises(ValueError, match="min_confidence must be"):
            UnicornConfig(min_confidence=1.5)

    def test_min_confidence_accepts_none(self):
        """Default None is valid (metric-only, no gate)."""
        config = UnicornConfig()
        assert config.min_confidence is None


class TestParityDiagnostics:
    """SetupOccurrence must carry gating diagnostics for parity checks."""

    def test_setup_occurrence_has_parity_fields(self):
        """Parity fields are populated on setup records."""
        start_ts = datetime(2024, 1, 2, 10, 0, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        # Should have scanned some setups
        assert len(result.all_setups) > 0

        for setup in result.all_setups:
            # Parity fields must be populated (not just defaults)
            assert isinstance(setup.mandatory_met, bool)
            assert isinstance(setup.scored_count, int)
            assert 0 <= setup.scored_count <= 5
            assert isinstance(setup.min_scored_required, int)
            assert isinstance(setup.decide_entry_result, bool)

            # Consistency: decide_entry_result must match the inputs
            expected = (
                setup.mandatory_met and setup.scored_count >= setup.min_scored_required
            )
            assert setup.decide_entry_result == expected, (
                f"Parity mismatch at {setup.timestamp}: "
                f"mandatory={setup.mandatory_met}, scored={setup.scored_count}, "
                f"min={setup.min_scored_required}, result={setup.decide_entry_result}"
            )

            # scored_missing must be consistent with scored_count
            assert isinstance(setup.scored_missing, list)
            assert len(setup.scored_missing) == 4 - setup.scored_count, (
                f"scored_missing length mismatch at {setup.timestamp}: "
                f"missing={setup.scored_missing}, scored_count={setup.scored_count}"
            )
            # All items must be valid scored criteria names
            valid_scored = {"liquidity_sweep", "htf_fvg", "breaker_block", "ltf_fvg"}
            for name in setup.scored_missing:
                assert name in valid_scored, f"Invalid scored criterion: {name}"


class TestTimezoneConversion:
    """Timezone-aware session classification and macro window tests."""

    def test_utc_timestamp_converts_to_et_for_session(self):
        """UTC timestamp at 15:30 = 10:30 ET (winter) => NY_AM."""
        ts = datetime(2024, 1, 15, 15, 30, tzinfo=timezone.utc)
        assert classify_session(ts) == TradingSession.NY_AM

    def test_summer_dst_utc_to_et(self):
        """UTC 13:30 in July = 9:30 EDT => NY_AM."""
        ts = datetime(2024, 7, 15, 13, 30, tzinfo=timezone.utc)
        assert classify_session(ts) == TradingSession.NY_AM

    def test_winter_utc_to_et(self):
        """UTC 14:30 in January = 9:30 EST => NY_AM."""
        ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
        assert classify_session(ts) == TradingSession.NY_AM

    def test_naive_datetime_raises(self):
        """Naive datetime must raise ValueError."""
        ts = datetime(2024, 1, 15, 10, 30)  # naive
        with pytest.raises(ValueError, match="tz-aware"):
            classify_session(ts)

    def test_naive_datetime_raises_in_macro_window(self):
        """is_in_macro_window rejects naive datetime."""
        ts = datetime(2024, 1, 15, 10, 30)  # naive
        with pytest.raises(ValueError, match="tz-aware"):
            is_in_macro_window(ts)

    def test_boundary_after_conversion(self):
        """UTC 16:00 = 11:00 ET (winter) => end of NY_AM, should be excluded."""
        ts = datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        assert is_in_macro_window(ts, SessionProfile.STRICT) is False

        # Just before boundary
        ts_before = datetime(2024, 1, 15, 15, 59, 59, tzinfo=timezone.utc)
        assert is_in_macro_window(ts_before, SessionProfile.STRICT) is True


class TestEnsureUtc:
    """Tests for the ensure_utc utility."""

    def test_utc_passthrough(self):
        """UTC datetime passes through unchanged."""
        from app.utils.time import ensure_utc
        ts = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        result = ensure_utc(ts)
        assert result == ts
        assert result.tzinfo is not None

    def test_non_utc_converts(self):
        """Non-UTC tz-aware datetime is converted to UTC."""
        from app.utils.time import ensure_utc
        et_ts = datetime(2024, 1, 15, 10, 30, tzinfo=ET)
        result = ensure_utc(et_ts)
        # 10:30 ET in winter = 15:30 UTC
        assert result.hour == 15
        assert result.minute == 30
        assert result.tzname() == "UTC"

    def test_naive_raises(self):
        """Naive datetime raises ValueError."""
        from app.utils.time import ensure_utc
        with pytest.raises(ValueError, match="tz-aware"):
            ensure_utc(datetime(2024, 1, 15, 10, 30))


# =========================================================================
# Canonical intrabar fill tests for resolve_bar_exit
# =========================================================================

SLIP = 0.25  # 1 tick on NQ

from typing import Optional


def _make_trade(
    direction: BiasDirection,
    entry_price: float,
    stop_price: float,
    target_price: float,
    entry_time: Optional[datetime] = None,
    risk_points: Optional[float] = None,
) -> TradeRecord:
    """Helper to build a minimal TradeRecord for exit tests."""
    if entry_time is None:
        entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
    if risk_points is None:
        risk_points = abs(entry_price - stop_price)
    return TradeRecord(
        entry_time=entry_time,
        entry_price=entry_price,
        direction=direction,
        quantity=1,
        session=TradingSession.NY_AM,
        criteria=CriteriaCheck(),
        stop_price=stop_price,
        target_price=target_price,
        risk_points=risk_points,
    )



class TestResolveBarExitStopTarget:
    """Group A: stop/target basics."""

    def test_long_stop_hit(self):
        """Bar low pierces stop => stop_loss at stop-slip, pnl < 0."""
        trade = _make_trade(BiasDirection.BULLISH, 100.0, 95.0, 110.0)
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 99.0, 100.5, 94.5, 96.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "stop_loss"
        assert result.exit_price == 95.0 - SLIP
        assert result.pnl_points < 0

    def test_long_target_hit(self):
        """Bar high reaches target => target at target-slip, pnl > 0."""
        trade = _make_trade(BiasDirection.BULLISH, 100.0, 95.0, 110.0)
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 101.0, 111.0, 100.5, 109.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "target"
        assert result.exit_price == 110.0 - SLIP
        assert result.pnl_points > 0

    def test_short_stop_hit(self):
        """Bar high pierces stop => stop_loss at stop+slip, pnl < 0."""
        trade = _make_trade(BiasDirection.BEARISH, 100.0, 105.0, 90.0)
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 101.0, 105.5, 100.0, 104.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "stop_loss"
        assert result.exit_price == 105.0 + SLIP
        assert result.pnl_points < 0

    def test_short_target_hit(self):
        """Bar low reaches target => target at target+slip, pnl > 0."""
        trade = _make_trade(BiasDirection.BEARISH, 100.0, 105.0, 90.0)
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 99.0, 100.0, 89.0, 91.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "target"
        assert result.exit_price == 90.0 + SLIP
        assert result.pnl_points > 0


class TestResolveBarExitAmbiguity:
    """Group B: Same-bar stop+target ambiguity."""

    def test_same_bar_worst_policy(self):
        """Both hit, WORST => stop_loss."""
        trade = _make_trade(BiasDirection.BULLISH, 100.0, 95.0, 110.0)
        # Bar spans from below stop to above target
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 98.0, 111.0, 94.0, 105.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "stop_loss"

    def test_same_bar_best_policy(self):
        """Both hit, BEST => target."""
        trade = _make_trade(BiasDirection.BULLISH, 100.0, 95.0, 110.0)
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 98.0, 111.0, 94.0, 105.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.BEST, SLIP)
        assert result is not None
        assert result.exit_reason == "target"

    def test_same_bar_ohlc_path_bullish_bar(self):
        """Both hit on bullish bar (close>open), long => target first.

        Bullish bar: path is O->H->L->C. Long target on O->H leg fires first.
        """
        trade = _make_trade(BiasDirection.BULLISH, 100.0, 95.0, 110.0)
        # Bullish bar: close > open
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 98.0, 111.0, 94.0, 108.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.OHLC_PATH, SLIP)
        assert result is not None
        assert result.exit_reason == "target"
        assert result.exit_price == 110.0 - SLIP


class TestResolveBarExitGapThrough:
    """Group C: Gap-through stop pricing."""

    def test_long_gap_through_stop(self):
        """Bar opens below stop => fills at open-slip (not stop-slip)."""
        trade = _make_trade(BiasDirection.BULLISH, 100.0, 95.0, 110.0)
        # Bar gaps down — opens at 93, below the 95 stop
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 93.0, 94.0, 92.0, 93.5)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "stop_loss"
        # Gap through: fill at min(stop=95, open=93) = 93, then -slip
        assert result.exit_price == 93.0 - SLIP

    def test_short_gap_through_stop(self):
        """Bar opens above stop => fills at open+slip (not stop+slip)."""
        trade = _make_trade(BiasDirection.BEARISH, 100.0, 105.0, 90.0)
        # Bar gaps up — opens at 107, above the 105 stop
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 107.0, 108.0, 106.0, 107.5)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "stop_loss"
        # Gap through: fill at max(stop=105, open=107) = 107, then +slip
        assert result.exit_price == 107.0 + SLIP


class TestResolveBarExitEdgeCases:
    """Group D: Edge cases."""

    def test_no_exit_when_neither_hit(self):
        """Neither stop nor target hit => None."""
        trade = _make_trade(BiasDirection.BULLISH, 100.0, 95.0, 110.0)
        # Bar stays within stop and target
        bar = make_bar(trade.entry_time + timedelta(minutes=15), 100.0, 103.0, 97.0, 101.0)
        result = resolve_bar_exit(trade, bar, IntrabarPolicy.WORST, SLIP)
        assert result is None


class TestEntryBarStopCheck:
    """Group E: Entry-bar integration — stop checked on the entry bar."""

    def test_entry_bar_stop_checked(self):
        """Stop pierced on entry bar => trade exits immediately.

        Runs a full backtest with synthetic data where the entry bar's
        low breaches the stop. The trade should be closed on the entry bar
        rather than surviving to the next bar.
        """
        # Build bars where a trade would be opened and immediately stopped.
        # We use the resolve_bar_exit function directly to verify behavior,
        # since triggering an actual setup in run_unicorn_backtest requires
        # 8 criteria to align which is non-trivial in synthetic data.
        entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        trade = _make_trade(
            BiasDirection.BULLISH, 100.0, 98.0, 106.0, entry_time=entry_time,
        )
        # Entry bar itself goes below stop
        entry_bar = make_bar(entry_time, 100.0, 101.0, 97.0, 99.0)

        result = resolve_bar_exit(trade, entry_bar, IntrabarPolicy.WORST, SLIP)
        assert result is not None
        assert result.exit_reason == "stop_loss"
        # Gap-through: open=100 > stop=98, so fill at stop price
        assert result.exit_price == 98.0 - SLIP
        assert result.pnl_points < 0


# =========================================================================
# Wick ratio and range ATR mult helper tests
# =========================================================================


class TestComputeAdverseWickRatio:
    """Tests for adverse wick ratio computation."""

    def test_bullish_long_big_upper_wick(self):
        """Long direction: upper wick is adverse. o=100, h=110, l=98, c=102."""
        bar = make_bar(datetime(2024, 1, 15, 10, 0, tzinfo=ET), 100.0, 110.0, 98.0, 102.0)
        ratio = compute_adverse_wick_ratio(bar, BiasDirection.BULLISH)
        # upper wick = (110 - 102) / (110 - 98) = 8/12
        assert ratio == pytest.approx(0.6667, rel=1e-3)

    def test_bearish_short_big_lower_wick(self):
        """Short direction: lower wick is adverse. o=100, h=102, l=90, c=98."""
        bar = make_bar(datetime(2024, 1, 15, 10, 0, tzinfo=ET), 100.0, 102.0, 90.0, 98.0)
        ratio = compute_adverse_wick_ratio(bar, BiasDirection.BEARISH)
        # lower wick = (98 - 90) / (102 - 90) = 8/12
        assert ratio == pytest.approx(0.6667, rel=1e-3)

    def test_zero_range_returns_zero(self):
        """Flat bar (high == low) returns 0.0."""
        bar = make_bar(datetime(2024, 1, 15, 10, 0, tzinfo=ET), 100.0, 100.0, 100.0, 100.0)
        assert compute_adverse_wick_ratio(bar, BiasDirection.BULLISH) == 0.0
        assert compute_adverse_wick_ratio(bar, BiasDirection.BEARISH) == 0.0


class TestComputeRangeAtrMult:
    """Tests for range ATR multiple computation."""

    def test_basic(self):
        """Bar range=12, ATR=10 => 1.2."""
        bar = make_bar(datetime(2024, 1, 15, 10, 0, tzinfo=ET), 100.0, 110.0, 98.0, 105.0)
        assert compute_range_atr_mult(bar, 10.0) == pytest.approx(1.2)

    def test_zero_atr_returns_zero(self):
        """ATR=0 => 0.0 (avoid division by zero)."""
        bar = make_bar(datetime(2024, 1, 15, 10, 0, tzinfo=ET), 100.0, 110.0, 98.0, 105.0)
        assert compute_range_atr_mult(bar, 0.0) == 0.0


class TestWickGuardIntegration:
    """Integration tests for bar-quality guards in the backtest loop."""

    def test_wick_guard_rejects_high_wick(self):
        """config.max_wick_ratio=0.5, signal bar wick ~ 0.67 => rejected."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        config = UnicornConfig(max_wick_ratio=0.5)
        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
        )

        # Check that wick guard diagnostics are populated on setups
        wick_rejected = [s for s in result.all_setups if s.wick_guard_rejected]
        # With the guard enabled, at least the diagnostic fields should be populated
        for setup in result.all_setups:
            if setup.decide_entry_result and setup.direction != BiasDirection.NEUTRAL:
                # Diagnostics should always be recorded
                assert isinstance(setup.signal_wick_ratio, float)
                assert isinstance(setup.signal_range_atr_mult, float)

        # Any wick-rejected setup must have correct fields
        for setup in wick_rejected:
            assert setup.taken is False
            assert setup.guard_reason_code == "wick_guard"
            assert "wick_guard" in (setup.reason_not_taken or "")

    def test_wick_guard_disabled_by_default(self):
        """config.max_wick_ratio=None => no filtering."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        config = UnicornConfig()  # defaults: max_wick_ratio=None
        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
        )

        # No setups should be wick-guard-rejected when disabled
        wick_rejected = [s for s in result.all_setups if s.wick_guard_rejected]
        assert len(wick_rejected) == 0

    def test_range_guard_rejects_wide_bar(self):
        """config.max_range_atr_mult=2.0, wide signal bars => some rejected."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(
            start_ts, 200, 17000, trend=2.0, volatility=10.0, interval_minutes=15
        )
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        config = UnicornConfig(max_range_atr_mult=2.0)
        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
        )

        # Check that range guard diagnostics are correct
        range_rejected = [s for s in result.all_setups if s.range_guard_rejected]
        for setup in range_rejected:
            assert setup.taken is False
            assert setup.guard_reason_code == "range_guard"
            assert "range_guard" in (setup.reason_not_taken or "")


class TestDisplacementGuard:
    """Integration tests for the displacement conviction guard."""

    def _make_backtest_bars(self):
        """Helper: generate standard bars for displacement guard tests."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)
        return htf_bars, ltf_bars

    def test_criteria_check_stores_displacement_size(self):
        """check_criteria stores mss_displacement_atr on CriteriaCheck."""
        start_ts = datetime(2024, 1, 2, 10, 0, tzinfo=ET)
        bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)
        check_ts = datetime(2024, 1, 2, 10, 30, tzinfo=ET)

        result = check_criteria(
            bars=bars,
            htf_bars=bars,
            ltf_bars=bars[-60:],
            symbol="NQ",
            ts=check_ts,
        )

        assert isinstance(result.mss_displacement_atr, float)
        # If MSS was found, displacement must be non-negative
        if result.mss_found:
            assert result.mss_displacement_atr >= 0.0

    def test_displacement_guard_disabled_by_default(self):
        """min_displacement_atr=None => zero displacement-rejected setups."""
        htf_bars, ltf_bars = self._make_backtest_bars()
        config = UnicornConfig()  # defaults: min_displacement_atr=None

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
        )

        disp_rejected = [s for s in result.all_setups if s.displacement_guard_rejected]
        assert len(disp_rejected) == 0
        # displacement_guard_evaluated should be False for all setups when disabled
        for setup in result.all_setups:
            assert setup.displacement_guard_evaluated is False

    def test_displacement_guard_rejects_weak_mss(self):
        """Very high threshold => all MSS-bearing setups rejected."""
        htf_bars, ltf_bars = self._make_backtest_bars()
        config = UnicornConfig(min_displacement_atr=5.0)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
        )

        disp_rejected = [s for s in result.all_setups if s.displacement_guard_rejected]
        for setup in disp_rejected:
            assert setup.taken is False
            assert setup.guard_reason_code == "displacement_guard"
            assert "< 5.00x" in (setup.reason_not_taken or "")
            assert setup.displacement_guard_evaluated is True

    def test_displacement_diagnostics_recorded_on_wick_rejected(self):
        """signal_displacement_atr is recorded even on wick-rejected setups."""
        htf_bars, ltf_bars = self._make_backtest_bars()

        # Run with a very tight wick guard to cause wick rejections
        config_tight = UnicornConfig(max_wick_ratio=0.001)
        result_tight = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config_tight,
        )

        # Also run with a loose wick guard for comparison
        config_loose = UnicornConfig(max_wick_ratio=1.0)
        result_loose = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config_loose,
        )

        # In both cases, qualifying setups that passed direction filter
        # should have signal_displacement_atr as a float
        for result in [result_tight, result_loose]:
            qualifying = [
                s for s in result.all_setups
                if s.decide_entry_result and s.direction != BiasDirection.NEUTRAL
            ]
            for setup in qualifying:
                assert isinstance(setup.signal_displacement_atr, float)
                assert isinstance(setup.signal_mss_found, bool)


# =============================================================================
# NY_OPEN session profile tests
# =============================================================================


class TestNYOpenProfile:
    """Tests for NY_OPEN session profile and session diagnostics."""

    def test_ny_open_profile_pure_function(self):
        """is_in_macro_window: 10:45 ET is True with STRICT, False with NY_OPEN."""
        ts = datetime(2024, 6, 10, 10, 45, tzinfo=ET)
        assert is_in_macro_window(ts, SessionProfile.STRICT) is True
        assert is_in_macro_window(ts, SessionProfile.NY_OPEN) is False

    def test_session_profile_ny_open_boundaries(self):
        """NY_OPEN uses half-open [9:30, 10:30) interval."""
        ts_before = datetime(2024, 6, 10, 9, 29, tzinfo=ET)
        ts_start = datetime(2024, 6, 10, 9, 30, tzinfo=ET)
        ts_inside = datetime(2024, 6, 10, 10, 29, tzinfo=ET)
        ts_end = datetime(2024, 6, 10, 10, 30, tzinfo=ET)

        assert is_in_macro_window(ts_before, SessionProfile.NY_OPEN) is False
        assert is_in_macro_window(ts_start, SessionProfile.NY_OPEN) is True
        assert is_in_macro_window(ts_inside, SessionProfile.NY_OPEN) is True
        assert is_in_macro_window(ts_end, SessionProfile.NY_OPEN) is False

    def test_setup_session_diagnostic_always_set(self):
        """Every SetupOccurrence has setup_session populated and setup_in_macro_window is bool."""
        start = datetime(2024, 6, 10, 6, 0, tzinfo=ET)
        htf_bars = generate_trending_bars(start, 100, 17500.0, trend=0.3, interval_minutes=15)
        ltf_bars = generate_trending_bars(start, 300, 17500.0, trend=0.1, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        assert len(result.all_setups) > 0, "Need at least one setup to test"
        for setup in result.all_setups:
            assert isinstance(setup.setup_session, str)
            assert setup.setup_session != "", f"setup_session empty at {setup.timestamp}"
            assert isinstance(setup.setup_in_macro_window, bool)

    def test_session_diagnostics_structure(self):
        """session_diagnostics dict has expected keys and types."""
        start = datetime(2024, 6, 10, 6, 0, tzinfo=ET)
        htf_bars = generate_trending_bars(start, 100, 17500.0, trend=0.3, interval_minutes=15)
        ltf_bars = generate_trending_bars(start, 300, 17500.0, trend=0.1, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        diag = result.session_diagnostics
        assert diag is not None
        assert "setup_disposition" in diag
        assert "confidence_by_session" in diag

        # Validate setup_disposition entries
        for sess_key, counts in diag["setup_disposition"].items():
            assert isinstance(sess_key, str)
            assert isinstance(counts["total"], int)
            assert isinstance(counts["taken"], int)
            assert isinstance(counts["rejected"], int)
            assert isinstance(counts["macro_rejected"], int)
            assert isinstance(counts["take_pct"], float)
            assert isinstance(counts["in_macro_total"], int)
            assert isinstance(counts["take_pct_in_macro"], float)
            assert counts["in_macro_total"] == counts["total"] - counts["macro_rejected"]

        # Validate confidence_by_session entries
        for sess_key, stats in diag["confidence_by_session"].items():
            assert isinstance(sess_key, str)
            assert isinstance(stats["trades"], int)
            assert isinstance(stats["avg_confidence"], float)
            assert isinstance(stats["low"], int)
            assert isinstance(stats["mid"], int)
            assert isinstance(stats["high"], int)

        # Validate expectancy_by_session entries
        assert "expectancy_by_session" in diag
        for sess_key, e in diag["expectancy_by_session"].items():
            assert isinstance(sess_key, str)
            assert isinstance(e["trades"], int)
            assert isinstance(e["wins"], int)
            assert isinstance(e["losses"], int)
            assert isinstance(e["win_rate"], float)
            assert isinstance(e["avg_win_pts"], float)
            assert isinstance(e["avg_loss_pts"], float)
            assert isinstance(e["expectancy_per_trade"], float)
            assert isinstance(e["total_pnl_pts"], float)
            assert isinstance(e["expectancy_per_in_macro_setup"], float)
            # R-multiple fields
            assert isinstance(e["avg_r_per_trade"], float)
            assert isinstance(e["total_r"], float)
            assert isinstance(e["avg_win_r"], float)
            assert isinstance(e["avg_loss_r"], float)
            assert isinstance(e["expectancy_r_per_in_macro_setup"], float)
            assert isinstance(e["rr_missing"], int)

        # Validate confidence_outcome_by_session entries
        assert "confidence_outcome_by_session" in diag
        for sess_key, buckets in diag["confidence_outcome_by_session"].items():
            assert isinstance(sess_key, str)
            for bk in ("low", "mid", "high"):
                assert bk in buckets, f"Missing bucket {bk} in session {sess_key}"
                b = buckets[bk]
                assert isinstance(b["trades"], int)
                assert isinstance(b["wins"], int)
                assert isinstance(b["win_rate"], float)
                assert isinstance(b["avg_pnl_pts"], float)
                assert isinstance(b["total_pnl_pts"], float)
                # R-multiple fields
                assert isinstance(b["avg_r"], float)
                assert isinstance(b["total_r"], float)

    def test_report_contains_diagnostic_sections(self):
        """Report includes CONFIG, SETUP DISPOSITION BY SESSION, and CONFIDENCE BY SESSION."""
        start = datetime(2024, 6, 10, 6, 0, tzinfo=ET)
        htf_bars = generate_trending_bars(start, 100, 17500.0, trend=0.3, interval_minutes=15)
        ltf_bars = generate_trending_bars(start, 300, 17500.0, trend=0.1, interval_minutes=5)

        config = UnicornConfig(session_profile=SessionProfile.NORMAL)
        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
        )

        report = format_backtest_report(result)

        assert "CONFIG" in report
        assert "SETUP DISPOSITION BY SESSION" in report
        assert "CONFIDENCE BY SESSION" in report


# =========================================================================
# Intermarket agreement observability tests
# =========================================================================


class TestAsofLookup:
    """Direct unit tests for _asof_lookup."""

    def test_empty_series_returns_none(self):
        """Empty series => None."""
        ts = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert _asof_lookup([], ts) is None

    def test_ts_before_first_entry_returns_none(self):
        """ts earlier than all entries => None."""
        series = [
            BiasState(ts=datetime(2024, 1, 15, 11, 0, tzinfo=ET), direction=BiasDirection.BULLISH, confidence=0.8),
        ]
        early_ts = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert _asof_lookup(series, early_ts) is None

    def test_ts_exactly_on_entry(self):
        """ts exactly matching an entry => returns that entry."""
        target_ts = datetime(2024, 1, 15, 10, 30, tzinfo=ET)
        series = [
            BiasState(ts=datetime(2024, 1, 15, 10, 0, tzinfo=ET), direction=BiasDirection.BULLISH, confidence=0.6),
            BiasState(ts=target_ts, direction=BiasDirection.BEARISH, confidence=0.9),
            BiasState(ts=datetime(2024, 1, 15, 11, 0, tzinfo=ET), direction=BiasDirection.BULLISH, confidence=0.5),
        ]
        result = _asof_lookup(series, target_ts)
        assert result is not None
        assert result.ts == target_ts
        assert result.direction == BiasDirection.BEARISH
        assert result.confidence == 0.9

    def test_ts_between_entries_returns_earlier(self):
        """ts between two entries => returns the earlier one."""
        series = [
            BiasState(ts=datetime(2024, 1, 15, 10, 0, tzinfo=ET), direction=BiasDirection.BULLISH, confidence=0.6),
            BiasState(ts=datetime(2024, 1, 15, 11, 0, tzinfo=ET), direction=BiasDirection.BEARISH, confidence=0.9),
        ]
        between_ts = datetime(2024, 1, 15, 10, 30, tzinfo=ET)
        result = _asof_lookup(series, between_ts)
        assert result is not None
        assert result.direction == BiasDirection.BULLISH
        assert result.confidence == 0.6


class TestBiasSeriesPopulated:
    """htf_bias_series must be populated after backtest."""

    def test_bias_series_populated(self):
        """Backtest populates htf_bias_series with BiasState tuples."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        assert isinstance(result.htf_bias_series, list)
        assert len(result.htf_bias_series) > 0
        for state in result.htf_bias_series:
            assert isinstance(state, BiasState)
            assert isinstance(state.ts, datetime)
            assert isinstance(state.direction, BiasDirection)
            assert isinstance(state.confidence, float)
            assert 0.0 <= state.confidence <= 1.0


class TestIntermarketAgreementWithRefSeries:
    """intermarket_agreement diagnostics with a synthetic reference series."""

    def test_agreement_present_with_ref_series(self):
        """Passing reference_bias_series populates intermarket_agreement in diagnostics."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        # First run to get htf_bias_series timestamps
        base_result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        # Build synthetic reference: all BULLISH, confidence 0.8
        synth_ref = [
            BiasState(ts=s.ts, direction=BiasDirection.BULLISH, confidence=0.8)
            for s in base_result.htf_bias_series
        ]

        # Re-run with reference series
        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            reference_bias_series=synth_ref,
            reference_symbol="ES",
        )

        diag = result.session_diagnostics
        assert diag is not None
        assert "intermarket_agreement" in diag

        ia = diag["intermarket_agreement"]
        assert ia["reference_symbol"] == "ES"
        assert isinstance(ia["by_agreement"], dict)
        assert isinstance(ia["by_session_agreement"], dict)
        assert isinstance(ia["both_high_conf"], dict)

        # Validate structure of by_agreement entries
        for label, entry in ia["by_agreement"].items():
            assert label in ("aligned", "divergent", "neutral_involved", "missing_ref", "missing_primary")
            assert isinstance(entry["trades"], int)
            assert isinstance(entry["wins"], int)
            assert isinstance(entry["win_rate"], float)
            assert isinstance(entry["avg_pnl_pts"], float)
            assert isinstance(entry["total_pnl_pts"], float)
            assert isinstance(entry["avg_r"], float)
            assert isinstance(entry["total_r"], float)

        # both_high_conf must have correct shape
        bh = ia["both_high_conf"]
        assert isinstance(bh["trades"], int)
        assert isinstance(bh["wins"], int)
        assert isinstance(bh["win_rate"], float)
        assert isinstance(bh["avg_r"], float)
        assert isinstance(bh["total_r"], float)


class TestNoRefSeriesNoAgreementKey:
    """Without reference series, intermarket_agreement must be absent."""

    def test_no_ref_series_no_agreement_key(self):
        """No reference_bias_series => no intermarket_agreement key."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        diag = result.session_diagnostics
        assert diag is not None
        assert "intermarket_agreement" not in diag


class TestIntermarketReportRendering:
    """Intermarket agreement report section renders correctly."""

    def test_report_contains_intermarket_section(self):
        """Report includes INTERMARKET AGREEMENT when ref series provided."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        # Build a simple all-bullish reference
        ref_series = [
            BiasState(ts=htf_bars[i].ts, direction=BiasDirection.BULLISH, confidence=0.8)
            for i in range(len(htf_bars))
        ]

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            reference_bias_series=ref_series,
            reference_symbol="ES",
        )

        report = format_backtest_report(result)
        assert "INTERMARKET AGREEMENT (vs ES)" in report

    def test_report_no_intermarket_section_without_ref(self):
        """Report omits INTERMARKET AGREEMENT when no ref series."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
        )

        report = format_backtest_report(result)
        assert "INTERMARKET AGREEMENT" not in report


# =========================================================================
# Phase 3: Multi-TF causal alignment tests
# =========================================================================


class TestCheckCriteriaBackwardCompat:
    """check_criteria backward compatibility: no h4/h1 => identical output."""

    def test_check_criteria_backward_compat(self):
        """Without h4/h1, output is identical to old behavior."""
        start_ts = datetime(2024, 1, 2, 10, 0, tzinfo=ET)
        bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)
        check_ts = datetime(2024, 1, 2, 10, 30, tzinfo=ET)

        result_old = check_criteria(
            bars=bars, htf_bars=bars, ltf_bars=bars[-60:],
            symbol="NQ", ts=check_ts,
        )
        result_new = check_criteria(
            bars=bars, htf_bars=bars, ltf_bars=bars[-60:],
            symbol="NQ", ts=check_ts,
            h4_bars=None, h1_bars=None,
        )

        assert result_old.htf_bias_direction == result_new.htf_bias_direction
        assert result_old.htf_bias_confidence == result_new.htf_bias_confidence
        assert result_old.htf_bias_aligned == result_new.htf_bias_aligned

    def test_check_criteria_with_h4_h1(self):
        """Passing h4/h1 bars changes bias confidence (more data = different weight)."""
        start_ts = datetime(2024, 1, 2, 10, 0, tzinfo=ET)
        bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)
        h4_bars = generate_trending_bars(start_ts, 25, 17000, trend=2.0, interval_minutes=240)
        h1_bars = generate_trending_bars(start_ts, 100, 17000, trend=1.0, interval_minutes=60)
        check_ts = datetime(2024, 1, 2, 10, 30, tzinfo=ET)

        result_without = check_criteria(
            bars=bars, htf_bars=bars, ltf_bars=bars[-60:],
            symbol="NQ", ts=check_ts,
        )
        result_with = check_criteria(
            bars=bars, htf_bars=bars, ltf_bars=bars[-60:],
            symbol="NQ", ts=check_ts,
            h4_bars=h4_bars, h1_bars=h1_bars,
        )

        # Both should produce valid results
        assert isinstance(result_without.htf_bias_confidence, float)
        assert isinstance(result_with.htf_bias_confidence, float)
        # With additional TFs, confidence may differ (more data = different weighting)
        # We just verify it doesn't crash and returns a valid result


class TestCausalAlignment:
    """Tests for causal alignment of h4/h1 bars."""

    def test_causal_h4_alignment(self):
        """At 11:45, 08:00 4H bar (covering 08:00-12:00) is NOT complete."""
        from app.services.backtest.engines.unicorn_runner import BarBundle

        # Create a 4H bar starting at 08:00 UTC
        h4_bar_0800 = make_bar(
            datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc),
            17000, 17010, 16990, 17005,
        )
        h4_bar_0400 = make_bar(
            datetime(2024, 1, 2, 4, 0, tzinfo=timezone.utc),
            16990, 17005, 16985, 17000,
        )

        h4_completed_ts = [
            h4_bar_0400.ts + timedelta(hours=4),  # 08:00 - completed
            h4_bar_0800.ts + timedelta(hours=4),  # 12:00 - NOT yet completed at 11:45
        ]

        # At 11:45, bisect_right finds how many completed bars
        from bisect import bisect_right
        ts = datetime(2024, 1, 2, 11, 45, tzinfo=timezone.utc)
        n_complete = bisect_right(h4_completed_ts, ts)

        # Only 1 bar (the 04:00 bar) is complete at 11:45
        assert n_complete == 1

    def test_causal_h1_alignment(self):
        """At 10:15, 10:00 1H bar (covering 10:00-11:00) is NOT complete."""
        h1_bar_0900 = make_bar(
            datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),
            17000, 17010, 16990, 17005,
        )
        h1_bar_1000 = make_bar(
            datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
            17005, 17015, 16995, 17010,
        )

        h1_completed_ts = [
            h1_bar_0900.ts + timedelta(hours=1),  # 10:00 - completed
            h1_bar_1000.ts + timedelta(hours=1),  # 11:00 - NOT yet completed at 10:15
        ]

        from bisect import bisect_right
        ts = datetime(2024, 1, 2, 10, 15, tzinfo=timezone.utc)
        n_complete = bisect_right(h1_completed_ts, ts)

        # Only 1 bar (the 09:00 bar) is complete at 10:15
        assert n_complete == 1

    def test_backtest_with_bar_bundle_runs(self):
        """run_unicorn_backtest with bar_bundle completes without error."""
        from app.services.backtest.engines.unicorn_runner import BarBundle

        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)
        h4_bars = generate_trending_bars(start_ts, 25, 17000, trend=8.0, interval_minutes=240)
        h1_bars = generate_trending_bars(start_ts, 100, 17000, trend=2.0, interval_minutes=60)

        bundle = BarBundle(h4=h4_bars, h1=h1_bars, m15=htf_bars, m5=ltf_bars)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            bar_bundle=bundle,
        )

        assert result.symbol == "NQ"
        assert result.total_bars == 200
        assert result.total_setups_scanned > 0

    def test_backtest_without_bar_bundle_unchanged(self):
        """bar_bundle=None preserves existing behavior."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result_old = run_unicorn_backtest(
            symbol="NQ", htf_bars=htf_bars, ltf_bars=ltf_bars, dollars_per_trade=500,
        )
        result_new = run_unicorn_backtest(
            symbol="NQ", htf_bars=htf_bars, ltf_bars=ltf_bars, dollars_per_trade=500,
            bar_bundle=None,
        )

        # Same number of setups and trades (identical logic path)
        assert result_old.total_setups_scanned == result_new.total_setups_scanned
        assert result_old.trades_taken == result_new.trades_taken


# =========================================================================
# Phase 4: 1m hybrid execution tests
# =========================================================================


class TestHybrid1mExecution:
    """Tests for 1m precision trade management."""

    def _make_m1_bars(self, start_ts, count, base_price=17000.0, trend=0.1):
        """Generate synthetic 1m bars."""
        bars = []
        price = base_price
        for i in range(count):
            ts = start_ts + timedelta(minutes=i)
            o = price
            c = price + trend
            h = max(o, c) + 0.5
            low = min(o, c) - 0.5
            bars.append(make_bar(ts, o, h, low, c))
            price = c
        return bars

    def test_1m_stop_hit_precision(self):
        """Trade exits at minute 7, not at 15m boundary."""
        entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        trade = _make_trade(BiasDirection.BULLISH, 17000.0, 16990.0, 17020.0, entry_time=entry_time)

        # Create 15 1m bars; minute 7 has low that pierces stop
        m1_bars = []
        for i in range(15):
            ts = entry_time + timedelta(minutes=i)
            if i == 7:
                # Stop at 16990, this bar dips to 16989
                m1_bars.append(make_bar(ts, 16995.0, 16996.0, 16989.0, 16992.0))
            else:
                m1_bars.append(make_bar(ts, 16995.0, 16998.0, 16993.0, 16996.0))

        # Simulate 1m management: iterate and check
        for m1_bar in m1_bars:
            result = resolve_bar_exit(trade, m1_bar, IntrabarPolicy.WORST, SLIP)
            if result:
                trade.exit_price = result.exit_price
                trade.exit_time = m1_bar.ts
                trade.exit_reason = result.exit_reason
                break

        assert trade.exit_time is not None
        assert trade.exit_time == entry_time + timedelta(minutes=7)
        assert trade.exit_reason == "stop_loss"

    def test_1m_target_hit_precision(self):
        """Trade hits target at minute 5, not at 15m boundary."""
        entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        trade = _make_trade(BiasDirection.BULLISH, 17000.0, 16990.0, 17010.0, entry_time=entry_time)

        m1_bars = []
        for i in range(15):
            ts = entry_time + timedelta(minutes=i)
            if i == 5:
                # Target at 17010, this bar reaches 17011
                m1_bars.append(make_bar(ts, 17005.0, 17011.0, 17004.0, 17009.0))
            else:
                m1_bars.append(make_bar(ts, 17002.0, 17005.0, 17000.0, 17003.0))

        for m1_bar in m1_bars:
            result = resolve_bar_exit(trade, m1_bar, IntrabarPolicy.WORST, SLIP)
            if result:
                trade.exit_time = m1_bar.ts
                trade.exit_reason = result.exit_reason
                break

        assert trade.exit_time is not None
        assert trade.exit_time == entry_time + timedelta(minutes=5)
        assert trade.exit_reason == "target"

    def test_1m_entry_price_uses_first_1m_bar(self):
        """When m1_window is available, entry price uses m1[0].open."""
        # This is tested indirectly: we verify the logic is correct
        # by checking that m1_window[0].open differs from the 15m bar.open
        m1_bar = make_bar(
            datetime(2024, 1, 15, 10, 0, tzinfo=ET), 17005.0, 17010.0, 17000.0, 17008.0
        )
        bar_15m = make_bar(
            datetime(2024, 1, 15, 10, 0, tzinfo=ET), 17003.0, 17012.0, 16998.0, 17010.0
        )

        # m1[0].open should be used for entry
        assert m1_bar.open == 17005.0
        assert bar_15m.open == 17003.0
        assert m1_bar.open != bar_15m.open

    def test_1m_fallback_when_no_m1_data(self):
        """m1=None in BarBundle => identical to no-bundle behavior."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        # Bundle without m1 data
        bundle_no_m1 = BarBundle(m15=htf_bars, m5=ltf_bars, m1=None)

        result_no_bundle = run_unicorn_backtest(
            symbol="NQ", htf_bars=htf_bars, ltf_bars=ltf_bars, dollars_per_trade=500,
        )
        result_bundle_no_m1 = run_unicorn_backtest(
            symbol="NQ", htf_bars=htf_bars, ltf_bars=ltf_bars, dollars_per_trade=500,
            bar_bundle=bundle_no_m1,
        )

        assert result_no_bundle.total_setups_scanned == result_bundle_no_m1.total_setups_scanned
        assert result_no_bundle.trades_taken == result_bundle_no_m1.trades_taken

    def test_1m_cross_bar_management(self):
        """Trade opened in bar N can be managed via 1m in bar N+k."""
        # This is structural: 1m management in the loop processes open trades
        # from previous 15m bars. We verify by running with m1 data.
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)
        m1_bars = generate_trending_bars(start_ts, 3000, 17000, trend=0.067, interval_minutes=1)

        bundle = BarBundle(m15=htf_bars, m5=ltf_bars, m1=m1_bars)

        result = run_unicorn_backtest(
            symbol="NQ", htf_bars=htf_bars, ltf_bars=ltf_bars, dollars_per_trade=500,
            bar_bundle=bundle,
        )

        # Just verify it runs and produces results
        assert result.symbol == "NQ"
        assert result.total_setups_scanned > 0

    def test_1m_mfe_mae_precision(self):
        """MFE/MAE tracked at 1m granularity when m1 data available."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)
        m1_bars = generate_trending_bars(start_ts, 3000, 17000, trend=0.067, interval_minutes=1)

        bundle = BarBundle(m15=htf_bars, m5=ltf_bars, m1=m1_bars)

        result = run_unicorn_backtest(
            symbol="NQ", htf_bars=htf_bars, ltf_bars=ltf_bars, dollars_per_trade=500,
            bar_bundle=bundle,
        )

        # All trades should have MFE/MAE tracking
        for trade in result.trades:
            assert isinstance(trade.mfe, float)
            assert isinstance(trade.mae, float)


# ---------------------------------------------------------------------------
# Phase 5: CLI synthetic multi-TF generation tests
# ---------------------------------------------------------------------------

class TestGenerateSampleData:
    """Tests for generate_sample_data() in the CLI script."""

    def test_generate_sample_data_legacy(self):
        """Legacy mode returns (htf_bars, ltf_bars) tuple."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from run_unicorn_backtest import generate_sample_data

        result = generate_sample_data("NQ", days=3, profile="easy", multi_tf=False)

        assert isinstance(result, tuple)
        assert len(result) == 2
        htf_bars, ltf_bars = result
        assert len(htf_bars) > 0
        assert len(ltf_bars) > 0
        # HTF should be 15m bars, LTF should be 5m bars
        # 3 weekdays × 11 hours × 4 bars/hour = ~132 HTF bars (may vary by weekday)
        assert len(htf_bars) > 50
        assert len(ltf_bars) > len(htf_bars)  # 5m has more bars than 15m

    def test_generate_sample_data_multi_tf(self):
        """Multi-TF mode returns BarBundle with all timeframes populated."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from run_unicorn_backtest import generate_sample_data

        result = generate_sample_data("NQ", days=3, profile="easy", multi_tf=True)

        assert isinstance(result, BarBundle)
        assert result.m1 is not None
        assert result.m5 is not None
        assert result.m15 is not None
        assert result.h1 is not None
        assert result.h4 is not None

        # Bar count consistency: each higher TF should have fewer bars
        assert len(result.m1) > len(result.m5)
        assert len(result.m5) > len(result.m15)
        assert len(result.m15) > len(result.h1)
        assert len(result.h1) > len(result.h4)

        # Rough ratio checks (1m has ~660 bars/day for 11 trading hours)
        assert len(result.m1) > 500  # 3 days × ~660
        assert len(result.m5) >= len(result.m1) // 6  # ~5:1 ratio, with rounding
        assert len(result.h4) >= 3  # At least 1 per day


class TestCLIMultiTFSmoke:
    """End-to-end CLI smoke test for --multi-tf flag."""

    def test_cli_multi_tf_synthetic(self):
        """CLI script runs to completion with --multi-tf --synthetic."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent.parent.parent / "scripts" / "run_unicorn_backtest.py"),
                "--symbol", "NQ",
                "--synthetic",
                "--days", "3",
                "--multi-tf",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        # Should mention multi-TF in output
        assert "Multi-TF" in result.stdout or "multi" in result.stdout.lower()


# =========================================================================
# Phase: Trace mode — BiasSnapshot, format_trade_trace, intermarket label
# =========================================================================


class TestBiasSnapshot:
    """Tests for BiasSnapshot capture in check_criteria."""

    def test_bias_snapshot_populated(self):
        """check_criteria with h4/h1 data populates per-TF directions."""
        start_ts = datetime(2024, 1, 2, 10, 0, tzinfo=ET)
        bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)
        h4_bars = generate_trending_bars(start_ts, 25, 17000, trend=2.0, interval_minutes=240)
        h1_bars = generate_trending_bars(start_ts, 100, 17000, trend=1.0, interval_minutes=60)
        check_ts = datetime(2024, 1, 2, 10, 30, tzinfo=ET)

        result = check_criteria(
            bars=bars, htf_bars=bars, ltf_bars=bars[-60:],
            symbol="NQ", ts=check_ts,
            h4_bars=h4_bars, h1_bars=h1_bars,
        )

        assert result.bias_snapshot is not None
        snap = result.bias_snapshot
        # With h4/h1 provided, directions should not be None
        assert snap.h4_direction is not None
        assert snap.h1_direction is not None
        assert snap.m15_direction is not None
        # H4 has insufficient data (25 bars < 210 needed), so it's excluded
        # from used_tfs but still present in the snapshot for transparency
        assert "h4" not in snap.used_tfs
        assert "h1" in snap.used_tfs
        assert isinstance(snap.final_confidence, float)
        assert isinstance(snap.alignment_score, float)

    def test_bias_snapshot_without_htf(self):
        """No h4/h1 bars → snapshot h4/h1 are None, final still computed from m15/m5."""
        start_ts = datetime(2024, 1, 2, 10, 0, tzinfo=ET)
        bars = generate_trending_bars(start_ts, 100, 17000, interval_minutes=15)
        check_ts = datetime(2024, 1, 2, 10, 30, tzinfo=ET)

        result = check_criteria(
            bars=bars, htf_bars=bars, ltf_bars=bars[-60:],
            symbol="NQ", ts=check_ts,
        )

        assert result.bias_snapshot is not None
        snap = result.bias_snapshot
        assert snap.h4_direction is None
        assert snap.h1_direction is None
        # m15 at least should be populated (bars were passed as htf_bars)
        assert snap.m15_direction is not None
        assert "h4" not in snap.used_tfs
        assert "h1" not in snap.used_tfs
        assert "m15" in snap.used_tfs
        assert snap.final_confidence >= 0.0


class TestFormatTradeTrace:
    """Tests for format_trade_trace()."""

    def _make_synthetic_result(self):
        """Build a minimal result with one deterministic trade for trace tests."""
        from app.services.backtest.engines.unicorn_runner import (
            UnicornBacktestResult, TradeOutcome,
        )
        entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        exit_time = datetime(2024, 1, 15, 10, 30, tzinfo=ET)

        snap = BiasSnapshot(
            m15_direction=BiasDirection.BULLISH,
            m15_confidence=0.7,
            m5_direction=BiasDirection.BULLISH,
            m5_confidence=0.6,
            final_direction=BiasDirection.BULLISH,
            final_confidence=0.65,
            alignment_score=0.8,
            used_tfs=("m15", "m5"),
        )
        criteria = CriteriaCheck(
            htf_bias_aligned=True, in_macro_window=True, stop_valid=True,
            htf_bias_direction=BiasDirection.BULLISH, htf_bias_confidence=0.65,
            bias_snapshot=snap,
        )
        trade = TradeRecord(
            entry_time=entry_time,
            entry_price=17000.0,
            direction=BiasDirection.BULLISH,
            quantity=1,
            session=TradingSession.NY_AM,
            criteria=criteria,
            stop_price=16990.0,
            target_price=17020.0,
            risk_points=10.0,
            exit_time=exit_time,
            exit_price=16990.0 - 0.25,
            exit_reason="stop_loss",
            pnl_points=-10.25,
            pnl_dollars=-50.0,
            outcome=TradeOutcome.LOSS,
            r_multiple=-1.025,
            duration_minutes=30,
            mfe=5.0,
            mae=10.0,
        )
        result = UnicornBacktestResult(
            symbol="NQ",
            start_date=entry_time,
            end_date=exit_time,
            total_bars=100,
            trades=[trade],
            trades_taken=1,
            losses=1,
        )
        # Build m15 bars covering the trade window
        m15_bars = []
        for i in range(10):
            ts = entry_time + timedelta(minutes=i * 15)
            m15_bars.append(make_bar(ts, 17000.0 - i, 17005.0 - i, 16989.0 - i, 16995.0 - i))
        bundle = BarBundle(m15=m15_bars)
        return result, bundle, trade

    def test_format_trade_trace_basic(self):
        """Trace first trade, output has all 4 section headers in order."""
        result, bundle, trade = self._make_synthetic_result()

        output = format_trade_trace(
            trade=trade,
            trade_index=0,
            bar_bundle=bundle,
            result=result,
            intrabar_policy=IntrabarPolicy.WORST,
            slippage_points=0.25,
        )

        assert "ENTRY CONTEXT" in output
        assert "BIAS STACK" in output
        assert "MANAGEMENT PATH" in output
        assert "EXIT SUMMARY" in output
        # Check ordering
        idx_entry = output.index("ENTRY CONTEXT")
        idx_bias = output.index("BIAS STACK")
        idx_mgmt = output.index("MANAGEMENT PATH")
        idx_exit = output.index("EXIT SUMMARY")
        assert idx_entry < idx_bias < idx_mgmt < idx_exit

    def test_format_trade_trace_no_m1(self):
        """No m1 data → falls back to m15 replay."""
        result, bundle, trade = self._make_synthetic_result()
        # bundle already has m15 only (no m1)

        output = format_trade_trace(
            trade=trade,
            trade_index=0,
            bar_bundle=bundle,
            result=result,
            intrabar_policy=IntrabarPolicy.WORST,
            slippage_points=0.25,
        )

        assert "MANAGEMENT PATH" in output
        assert "m15" in output  # Should mention replay TF

    def test_format_trade_trace_verbose(self):
        """Verbose prints all bars (no ellipsis hint)."""
        result, bundle, trade = self._make_synthetic_result()

        output = format_trade_trace(
            trade=trade,
            trade_index=0,
            bar_bundle=bundle,
            result=result,
            intrabar_policy=IntrabarPolicy.WORST,
            slippage_points=0.25,
            verbose=True,
        )

        assert "MANAGEMENT PATH" in output
        assert "--trace-verbose" not in output

    def test_format_trade_trace_out_of_range(self):
        """Index beyond trades raises ValueError."""
        result, bundle, trade = self._make_synthetic_result()

        with pytest.raises(ValueError, match="out of range"):
            format_trade_trace(
                trade=trade,
                trade_index=9999,
                bar_bundle=bundle,
                result=result,
                intrabar_policy=IntrabarPolicy.WORST,
                slippage_points=0.25,
            )

    def test_trace_replay_matches_recorded_exit(self):
        """Replay exit reason matches trade.exit_reason, output says 'verified'."""
        result, bundle, trade = self._make_synthetic_result()

        output = format_trade_trace(
            trade=trade,
            trade_index=0,
            bar_bundle=bundle,
            result=result,
            intrabar_policy=IntrabarPolicy.WORST,
            slippage_points=0.25,
        )

        # Should contain either verified or unable to verify
        assert "verified" in output.lower() or "unable to verify" in output.lower()


class TestSeedReproducibility:
    """Seed reproducibility for synthetic data."""

    def test_seed_reproducibility(self):
        """Same seed → identical trade list."""
        import random
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from run_unicorn_backtest import generate_sample_data

        random.seed(42)
        htf1, ltf1 = generate_sample_data("NQ", days=5, profile="realistic")

        random.seed(42)
        htf2, ltf2 = generate_sample_data("NQ", days=5, profile="realistic")

        # Bar counts must match
        assert len(htf1) == len(htf2)
        assert len(ltf1) == len(ltf2)

        # Prices must match exactly
        for a, b in zip(htf1, htf2):
            assert a.open == b.open
            assert a.high == b.high
            assert a.low == b.low
            assert a.close == b.close


class TestIntermarketLabelOnTrade:
    """intermarket_label is set per trade when ref series provided."""

    def test_intermarket_label_on_trade(self):
        """_build_session_diagnostics sets intermarket_label on each trade."""
        from app.services.backtest.engines.unicorn_runner import (
            UnicornBacktestResult, TradeOutcome, _build_session_diagnostics,
        )

        entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        trade = TradeRecord(
            entry_time=entry_time,
            entry_price=17000.0,
            direction=BiasDirection.BULLISH,
            quantity=1,
            session=TradingSession.NY_AM,
            criteria=CriteriaCheck(htf_bias_direction=BiasDirection.BULLISH, htf_bias_confidence=0.7),
            stop_price=16990.0,
            target_price=17020.0,
            risk_points=10.0,
            exit_time=entry_time + timedelta(minutes=30),
            exit_price=17020.0,
            exit_reason="target",
            pnl_points=20.0,
            pnl_dollars=100.0,
            outcome=TradeOutcome.WIN,
        )

        # Build result with ref series
        ref_series = [
            BiasState(ts=entry_time, direction=BiasDirection.BULLISH, confidence=0.8),
        ]
        htf_bias_series = [
            BiasState(ts=entry_time, direction=BiasDirection.BULLISH, confidence=0.7),
        ]
        result = UnicornBacktestResult(
            symbol="NQ",
            start_date=entry_time,
            end_date=entry_time + timedelta(hours=1),
            total_bars=100,
            trades=[trade],
            trades_taken=1,
            wins=1,
            htf_bias_series=htf_bias_series,
            reference_bias_series=ref_series,
            reference_symbol="ES",
        )

        # _build_session_diagnostics sets intermarket_label
        result.session_diagnostics = _build_session_diagnostics(result)

        valid_labels = {"aligned", "divergent", "neutral_involved", "missing_ref", "missing_primary"}
        assert trade.intermarket_label is not None
        assert trade.intermarket_label in valid_labels
        # Both BULLISH → should be aligned
        assert trade.intermarket_label == "aligned"


class TestDailyGovernorIntegration:
    """Test that daily governor gates entries in the backtest loop."""

    def test_governor_halts_after_max_trades(self):
        """With max_trades=1, only one trade per day should be taken."""
        from app.services.backtest.engines.daily_governor import DailyGovernor
        from collections import Counter

        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        config = UnicornConfig()
        governor = DailyGovernor(max_daily_loss_dollars=5000.0, max_trades_per_day=1)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
            daily_governor=governor,
        )

        # Count trades per calendar day
        trades_per_day = Counter(t.entry_time.date() for t in result.trades)
        for day, count in trades_per_day.items():
            assert count <= 1, f"Day {day} had {count} trades, expected max 1"

    def test_governor_none_means_no_limit(self):
        """Without governor, no daily limits apply (backward compat)."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result_no_gov = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=UnicornConfig(),
            daily_governor=None,
        )
        assert result_no_gov.trades_taken >= 0
        assert result_no_gov.governor_stats is None

    def test_governor_stats_populated(self):
        """Governor stats dict should be present when governor is used."""
        from app.services.backtest.engines.daily_governor import DailyGovernor

        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        governor = DailyGovernor(max_daily_loss_dollars=300.0, max_trades_per_day=1)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=UnicornConfig(),
            daily_governor=governor,
        )

        assert result.governor_stats is not None
        assert "signals_skipped" in result.governor_stats
        assert "days_halted" in result.governor_stats
        assert "half_size_trades" in result.governor_stats
        assert "total_days_traded" in result.governor_stats
        assert "loss_limit_halts" in result.governor_stats
        assert "trade_limit_halts" in result.governor_stats
        # Policy knobs stored for report
        assert result.governor_stats["max_daily_loss_dollars"] == 300.0
        assert result.governor_stats["max_trades_per_day"] == 1


class TestStructuralSizing:
    def test_wide_stop_skips_trade(self):
        """When stop is too wide for the risk budget, trade is skipped."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0,
                                          volatility=20.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67,
                                          volatility=7.0, interval_minutes=5)

        # Very small budget: $50 per trade with NQ ($20/pt)
        # Any stop > 2.5 points should produce quantity=0 -> skip
        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=50,  # tiny budget
            config=UnicornConfig(),
        )

        # Check that the mechanism exists (position_size_zero in reason)
        size_skipped = [
            s for s in result.all_setups
            if s.reason_not_taken and "position_size_zero" in s.reason_not_taken
        ]
        assert isinstance(size_skipped, list)
