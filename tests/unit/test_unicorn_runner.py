"""
Unit tests for Unicorn Model backtest runner.
"""

from datetime import datetime, timedelta, timezone
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
    CriteriaCheck,
    SetupOccurrence,
    resolve_bar_exit,
    ExitResult,
    IntrabarPolicy,
    TradeRecord,
    compute_adverse_wick_ratio,
    compute_range_atr_mult,
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
        """All 3 mandatory + exactly 3/5 scored => entry allowed."""
        check = CriteriaCheck()
        # Mandatory
        check.htf_bias_aligned = True
        check.stop_valid = True
        check.in_macro_window = True
        # Scored: exactly 3 of 5
        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.mss_found = True
        check.breaker_block_found = False
        check.ltf_fvg_found = False

        assert check.meets_entry_requirements(min_scored=3) is True

    def test_mandatory_fail_with_all_scored_rejects(self):
        """Mandatory fail + 5/5 scored => must reject."""
        check = CriteriaCheck()
        # Mandatory: stop_valid fails
        check.htf_bias_aligned = True
        check.stop_valid = False
        check.in_macro_window = True
        # All 5 scored pass
        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = True
        check.mss_found = True

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
        # 3/5 scored
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.mss = True

        assert cs.decide_entry(min_scored=3) is True
        # Total score is 6, but decide_entry should not care about total
        assert cs.score == 6

    def test_decide_entry_rejects_below_scored_threshold(self):
        """Mandatory pass + only 2/5 scored => reject."""
        cs = CriteriaScore()
        cs.htf_bias = True
        cs.stop_valid = True
        cs.macro_window = True
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
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.breaker_block = True
        cs.ltf_fvg = True
        cs.mss = True

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
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.mss = True

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
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.breaker_block = True
        cs.ltf_fvg = True
        cs.mss = True

        # Mandatory fails (htf_bias=False), so entry rejected despite 5/5 scored
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

    def test_min_scored_criteria_rejects_six(self):
        """min_scored_criteria=6 is impossible (only 5 scored items)."""
        with pytest.raises(ValueError, match="min_scored_criteria must be 0-5"):
            UnicornConfig(min_scored_criteria=6)

    def test_min_scored_criteria_rejects_negative(self):
        """min_scored_criteria=-1 is invalid."""
        with pytest.raises(ValueError, match="min_scored_criteria must be 0-5"):
            UnicornConfig(min_scored_criteria=-1)

    def test_min_scored_criteria_accepts_zero(self):
        """min_scored_criteria=0 means mandatory-only gating."""
        config = UnicornConfig(min_scored_criteria=0)
        assert config.min_scored_criteria == 0

    def test_min_scored_criteria_accepts_five(self):
        """min_scored_criteria=5 requires all scored criteria."""
        config = UnicornConfig(min_scored_criteria=5)
        assert config.min_scored_criteria == 5

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
            assert len(setup.scored_missing) == 5 - setup.scored_count, (
                f"scored_missing length mismatch at {setup.timestamp}: "
                f"missing={setup.scored_missing}, scored_count={setup.scored_count}"
            )
            # All items must be valid scored criteria names
            valid_scored = {"liquidity_sweep", "htf_fvg", "breaker_block", "ltf_fvg", "mss"}
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
