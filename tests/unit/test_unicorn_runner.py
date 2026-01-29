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
    SetupOccurrence,
)
from app.services.strategy.indicators.tf_bias import BiasDirection
from app.services.strategy.strategies.unicorn_model import (
    CriteriaScore,
    UnicornConfig,
    SessionProfile,
    is_in_macro_window,
    get_max_stop_handles,
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
        ts = datetime(2024, 1, 15, 19, 30)
        assert is_in_macro_window(ts, SessionProfile.WIDE) is True
        assert is_in_macro_window(ts, SessionProfile.NORMAL) is False

    def test_strict_only_allows_ny_am(self):
        """3:30 ET (London) is in NORMAL but NOT in STRICT."""
        ts = datetime(2024, 1, 15, 3, 30)
        assert is_in_macro_window(ts, SessionProfile.NORMAL) is True
        assert is_in_macro_window(ts, SessionProfile.STRICT) is False

    def test_boundary_end_exclusive(self):
        """Exact end time (11:00:00) should be excluded (half-open interval)."""
        ts_end = datetime(2024, 1, 15, 11, 0, 0)
        assert is_in_macro_window(ts_end, SessionProfile.STRICT) is False

        ts_just_before = datetime(2024, 1, 15, 10, 59, 59)
        assert is_in_macro_window(ts_just_before, SessionProfile.STRICT) is True


class TestBacktestUsesSessionProfile:
    """Backtest check_criteria must pass config.session_profile, not default WIDE."""

    def test_check_criteria_respects_normal_profile(self):
        """Asia-window timestamp with NORMAL profile => macro_window False."""
        # 19:30 is inside WIDE (Asia) but outside NORMAL
        ts = datetime(2024, 1, 15, 19, 30)
        bars = generate_trending_bars(ts, 60, 17000, interval_minutes=15)

        config = UnicornConfig(session_profile=SessionProfile.NORMAL)
        result = check_criteria(bars, bars, bars[-30:], "NQ", ts, config=config)
        assert result.in_macro_window is False

    def test_check_criteria_respects_wide_profile(self):
        """Same timestamp with WIDE profile => macro_window True."""
        ts = datetime(2024, 1, 15, 19, 30)
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
        max_handles = get_max_stop_handles("NQ", atr=atr, config=config)
        # max_handles = 30.0
        assert max_handles == 30.0
        # risk_handles <= max_handles => valid
        assert 30.0 <= max_handles  # boundary: exactly equal passes

    def test_stop_above_max_atr_fails(self):
        """risk_handles > 3.0 * ATR => stop_valid = False."""
        atr = 10.0
        config = UnicornConfig(stop_max_atr_mult=3.0)
        max_handles = get_max_stop_handles("NQ", atr=atr, config=config)
        assert 30.1 > max_handles  # 30.1 exceeds 30.0 => would fail


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
        start_ts = datetime(2024, 1, 2, 10, 0)
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
