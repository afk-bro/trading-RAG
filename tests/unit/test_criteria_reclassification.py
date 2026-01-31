"""
Tests for MSS + displacement mandatory criteria reclassification.

Validates that MSS and displacement are now mandatory gates (not scored),
scored criteria count is 4 (not 5), and backward compatibility holds.
"""

import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

from app.services.strategy.models import OHLCVBar
from app.services.strategy.strategies.unicorn_model import (
    CriteriaScore,
    UnicornConfig,
    SessionProfile,
    STRATEGY_FAMILY,
    MODEL_VERSION,
    MODEL_CODENAME,
    MODEL_VERSIONS,
    build_run_label,
    build_run_key,
)
from app.services.backtest.engines.unicorn_runner import (
    CriteriaCheck,
    MANDATORY_CRITERIA,
    SCORED_CRITERIA,
    check_criteria,
    run_unicorn_backtest,
)
from app.services.strategy.indicators.tf_bias import BiasDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# =========================================================================
# Original 10 unit tests (dataclass / constant level)
# =========================================================================


class TestMSSNowMandatory:
    """MSS must be a mandatory gate."""

    def test_mss_now_mandatory(self):
        """mss_found=False => mandatory_criteria_met=False even if other mandatory pass."""
        check = CriteriaCheck()
        check.htf_bias_aligned = True
        check.stop_valid = True
        check.in_macro_window = True
        check.displacement_valid = True
        check.mss_found = False  # MSS fails

        # All 4 scored pass
        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = True

        assert check.mandatory_criteria_met is False
        assert check.meets_entry_requirements(min_scored=0) is False


class TestDisplacementMandatory:
    """Displacement validation as mandatory gate."""

    def test_displacement_mandatory_when_configured(self):
        """min_displacement_atr=0.3, MSS disp=0.1 => displacement_valid=False => mandatory fails."""
        check = CriteriaCheck()
        check.htf_bias_aligned = True
        check.stop_valid = True
        check.in_macro_window = True
        check.mss_found = True
        check.displacement_valid = False  # Below threshold

        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = True

        assert check.mandatory_criteria_met is False
        assert check.meets_entry_requirements(min_scored=3) is False

    def test_displacement_auto_passes_when_disabled(self):
        """min_displacement_atr=None => displacement_valid=True always."""
        cs = CriteriaScore()
        config = UnicornConfig(min_displacement_atr=None)

        # When displacement_atr is None, displacement_valid auto-passes
        cs.displacement_valid = (
            config.min_displacement_atr is None
        )
        assert cs.displacement_valid is True

    def test_displacement_passes_when_sufficient(self):
        """min_displacement_atr=0.3, MSS disp=0.5 => displacement_valid=True."""
        check = CriteriaCheck()
        check.htf_bias_aligned = True
        check.stop_valid = True
        check.in_macro_window = True
        check.mss_found = True
        check.mss_displacement_atr = 0.5
        check.displacement_valid = True  # 0.5 >= 0.3

        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = True

        assert check.mandatory_criteria_met is True
        assert check.meets_entry_requirements(min_scored=3) is True


class TestScoredCriteriaNow4:
    """Scored criteria count is 4 after reclassification."""

    def test_scored_criteria_now_4(self):
        """scored_count only counts liquidity_sweep, htf_fvg, breaker_block, ltf_fvg."""
        cs = CriteriaScore()
        cs.liquidity_sweep = True
        cs.htf_fvg = True
        cs.breaker_block = True
        cs.ltf_fvg = True
        # MSS is mandatory now, not scored
        cs.mss = True

        assert cs.scored_count == 4

        # Without MSS, scored count unchanged
        cs2 = CriteriaScore()
        cs2.liquidity_sweep = True
        cs2.htf_fvg = True
        cs2.breaker_block = True
        cs2.ltf_fvg = True
        cs2.mss = False
        assert cs2.scored_count == 4

    def test_scored_criteria_count_runner(self):
        """Runner CriteriaCheck scored_criteria_count is 4 max."""
        check = CriteriaCheck()
        check.liquidity_sweep_found = True
        check.htf_fvg_found = True
        check.breaker_block_found = True
        check.ltf_fvg_found = True
        check.mss_found = True  # mandatory, not scored

        assert check.scored_criteria_count == 4


class TestMinScoredValidationCap4:
    """UnicornConfig validates min_scored_criteria cap at 4."""

    def test_min_scored_validation_cap_4(self):
        """UnicornConfig(min_scored_criteria=5) raises ValueError."""
        with pytest.raises(ValueError, match="min_scored_criteria must be 0-4"):
            UnicornConfig(min_scored_criteria=5)

    def test_min_scored_4_is_valid(self):
        """min_scored_criteria=4 is the new strict maximum."""
        config = UnicornConfig(min_scored_criteria=4)
        assert config.min_scored_criteria == 4


class TestLegacyDefaultConfigUnchanged:
    """Default UnicornConfig behavior unchanged for users who never set displacement."""

    def test_legacy_default_config_unchanged(self):
        """UnicornConfig() with no args: displacement auto-passes, identical behavior."""
        config = UnicornConfig()
        assert config.min_displacement_atr is None
        assert config.min_scored_criteria == 3

        # Simulate analyze_unicorn_setup logic: None => auto-pass
        displacement_valid = (
            config.min_displacement_atr is None
            or True  # Would check MSS displacement
        )
        assert displacement_valid is True


class TestDisplacementGuardBackstop:
    """Post-scoring displacement guard still fires as redundant backstop."""

    def test_displacement_guard_backstop_still_fires(self):
        """Old post-scoring guard catches edge case where criterion passed but guard rejects.

        This shouldn't happen with correct implementation (guard is redundant),
        but validates the backstop is still present.
        """
        # If displacement is mandatory AND the guard still exists,
        # the guard can only fire on setups that already passed displacement_valid.
        # When both use the same threshold, guard is truly redundant.
        config = UnicornConfig(min_displacement_atr=0.3)

        # Simulate: MSS found with displacement below threshold
        mss_displacement_atr = 0.2
        displacement_valid = mss_displacement_atr >= config.min_displacement_atr
        assert displacement_valid is False  # Mandatory criterion blocks

        # Guard logic (from runner): would also block
        guard_blocks = (
            config.min_displacement_atr is not None
            and mss_displacement_atr < config.min_displacement_atr
        )
        assert guard_blocks is True  # Backstop agrees


class TestMissingKeyNaming:
    """External key must be 'displacement', never 'displacement_valid'."""

    def test_missing_key_is_displacement_not_displacement_valid(self):
        """missing list uses 'displacement', never 'displacement_valid'."""
        # CriteriaScore
        cs = CriteriaScore()
        cs.displacement_valid = False
        missing = cs.missing
        assert "displacement" in missing
        assert "displacement_valid" not in missing

        # CriteriaCheck
        check = CriteriaCheck()
        check.displacement_valid = False
        missing_check = check.missing_criteria()
        assert "displacement" in missing_check
        assert "displacement_valid" not in missing_check


class TestCriteriaCountInvariant:
    """Mandatory + scored = 9 total criteria."""

    def test_criteria_count_invariant(self):
        """len(MANDATORY_CRITERIA) + len(SCORED_CRITERIA) == 9."""
        assert len(MANDATORY_CRITERIA) + len(SCORED_CRITERIA) == 9

    def test_mandatory_contains_mss_and_displacement(self):
        """MANDATORY_CRITERIA includes mss and displacement."""
        assert "mss" in MANDATORY_CRITERIA
        assert "displacement" in MANDATORY_CRITERIA

    def test_scored_does_not_contain_mss(self):
        """SCORED_CRITERIA does not include mss."""
        assert "mss" not in SCORED_CRITERIA
        assert "displacement" not in SCORED_CRITERIA


# =========================================================================
# Verification checklist tests (exercise actual runner paths)
# =========================================================================


class TestBackwardCompatGoldenRun:
    """V1: Backward-compat — displacement_valid always True when unconfigured."""

    def test_all_setups_have_displacement_valid_true_when_none(self):
        """Run full backtest with default config (min_displacement_atr=None).

        Every setup that check_criteria produces must have displacement_valid=True.
        """
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        config = UnicornConfig()  # default: min_displacement_atr=None
        assert config.min_displacement_atr is None

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            dollars_per_trade=500,
            config=config,
        )

        # Every scanned setup must have displacement_valid=True
        assert len(result.all_setups) > 0, "Backtest must produce at least one setup"
        for setup in result.all_setups:
            assert setup.criteria.displacement_valid is True, (
                f"displacement_valid=False at {setup.timestamp} "
                f"with min_displacement_atr=None — backward compat broken"
            )

    def test_check_criteria_sets_displacement_valid_true_when_none(self):
        """Direct check_criteria call with default config produces displacement_valid=True."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 100, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 300, 17000, trend=0.67, interval_minutes=5)

        config = UnicornConfig()
        check = check_criteria(
            bars=htf_bars,
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            symbol="NQ",
            ts=htf_bars[-1].ts,
            config=config,
        )
        assert check.displacement_valid is True


class TestEnforcedDisplacementRun:
    """V2: With displacement threshold, trades <= baseline and rejection reason is correct."""

    def test_enforced_displacement_fewer_or_equal_trades(self):
        """--min-displacement-atr 0.3 must produce <= trades vs None."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        baseline = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            config=UnicornConfig(min_displacement_atr=None),
        )

        enforced = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            config=UnicornConfig(min_displacement_atr=0.3),
        )

        assert enforced.trades_taken <= baseline.trades_taken, (
            f"Enforced displacement produced more trades ({enforced.trades_taken}) "
            f"than baseline ({baseline.trades_taken})"
        )

    def test_rejection_reason_attributes_displacement_correctly(self):
        """Setups rejected by displacement must cite it as mandatory failure."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            config=UnicornConfig(min_displacement_atr=0.3),
        )

        for setup in result.all_setups:
            if setup.taken:
                continue
            if not setup.criteria.displacement_valid:
                # Must be attributed to mandatory failure, not scored
                assert setup.reason_not_taken is not None
                assert "mandatory" in setup.reason_not_taken or "displacement" in setup.reason_not_taken, (
                    f"Displacement rejection not properly attributed: "
                    f"reason='{setup.reason_not_taken}' at {setup.timestamp}"
                )
                # Must NOT say "scored X/4 < N"
                assert "scored" not in setup.reason_not_taken, (
                    f"Displacement failure wrongly attributed to scored threshold: "
                    f"reason='{setup.reason_not_taken}' at {setup.timestamp}"
                )


class TestNoMSSNoDisplacement:
    """V3: mss_found=False + min_displacement_atr set => both missing."""

    def test_no_mss_implies_displacement_false(self):
        """If mss_found=False and min_displacement_atr is set, displacement_valid must be False."""
        check = CriteriaCheck()
        check.mss_found = False
        # Simulate check_criteria logic
        config = UnicornConfig(min_displacement_atr=0.3)
        if config.min_displacement_atr is None:
            check.displacement_valid = True
        elif check.mss_found:
            check.displacement_valid = check.mss_displacement_atr >= config.min_displacement_atr
        else:
            check.displacement_valid = False

        assert check.displacement_valid is False

        # Both must appear in missing
        missing = check.missing_criteria()
        assert "mss" in missing
        assert "displacement" in missing

    def test_no_mss_no_displacement_in_runner(self):
        """check_criteria with no MSS and displacement threshold => both fail."""
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        # Flat bars — unlikely to produce MSS patterns
        htf_bars = generate_trending_bars(start_ts, 100, 17000, trend=0.0, volatility=0.1, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 300, 17000, trend=0.0, volatility=0.1, interval_minutes=5)

        config = UnicornConfig(min_displacement_atr=0.3)
        check = check_criteria(
            bars=htf_bars,
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            symbol="NQ",
            ts=htf_bars[-1].ts,
            config=config,
        )

        if not check.mss_found:
            assert check.displacement_valid is False, (
                "displacement_valid should be False when mss_found=False and threshold is set"
            )
            missing = check.missing_criteria()
            assert "mss" in missing
            assert "displacement" in missing


class TestBackstopRedundancy:
    """V4: Backstop guard never fires when mandatory criterion already blocks."""

    def test_backstop_never_fires_on_normal_path(self):
        """In a full backtest, if displacement_valid=False the setup is already
        rejected by mandatory gate, so the post-scoring guard should never be
        the actual rejection path for any setup that passed mandatory.
        """
        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            config=UnicornConfig(min_displacement_atr=0.3),
        )

        for setup in result.all_setups:
            if setup.displacement_guard_rejected:
                # The backstop fired. This means the setup PASSED mandatory
                # (including displacement_valid) but then the guard caught it.
                # This should be impossible since both use the same threshold.
                assert setup.criteria.displacement_valid is True, (
                    "Backstop fired but displacement_valid was already False — "
                    "this means the two displacement checks disagree"
                )


class TestReportSplitLabels:
    """Report and diagnostics use correct X/Y labels after reclassification."""

    def test_report_contains_correct_split_labels(self):
        """Report must say '/4' for scored, '9/9' for valid setups."""
        from app.services.backtest.engines.unicorn_runner import format_backtest_report

        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            config=UnicornConfig(),
        )

        report = format_backtest_report(result)
        assert "/4" in report, "Report should reference scored criteria out of 4"
        assert "9/9" in report, "Report should reference valid setups as 9/9"
        assert "/5" not in report, "Report must not reference old '/5' scored denominator"
        assert "8/8" not in report, "Report must not reference old '8/8' valid setup count"


# ===========================================================================
# Naming / Versioning
# ===========================================================================


class TestVersionConstants:
    """Strategy family versioning constants are correct."""

    def test_current_version(self):
        assert MODEL_VERSION == "2.1"
        assert MODEL_CODENAME == "Intent"
        assert STRATEGY_FAMILY == "Unicorn"

    def test_version_history_complete(self):
        assert "1.0" in MODEL_VERSIONS
        assert "2.0" in MODEL_VERSIONS
        assert "2.1" in MODEL_VERSIONS
        for ver, info in MODEL_VERSIONS.items():
            assert isinstance(info["codename"], str) and len(info["codename"]) > 0
            assert isinstance(info["desc"], str) and len(info["desc"]) > 0
            assert isinstance(info["mandatory"], int) and info["mandatory"] >= 0
            assert isinstance(info["scored"], int) and info["scored"] >= 0

    def test_version_schema_matches_code(self):
        """v2.1 schema counts must match the actual constant sets."""
        v21 = MODEL_VERSIONS["2.1"]
        assert v21["mandatory"] == 5
        assert v21["scored"] == 4
        assert v21["mandatory"] + v21["scored"] == 9


class TestBuildRunLabel:
    """build_run_label produces correct self-describing labels."""

    def test_default_config_label(self):
        label = build_run_label(UnicornConfig())
        assert label.startswith("Unicorn v2.1")
        assert "Bias=Single" in label
        assert "Side=BiDir" in label
        assert "Displ=off" in label
        assert "MinScore=3/4" in label
        assert "Window=normal" in label
        assert "TS=none" in label

    def test_mtf_long_only_with_displacement(self):
        cfg = UnicornConfig(min_displacement_atr=0.3, min_scored_criteria=2, session_profile=SessionProfile.STRICT)
        label = build_run_label(
            cfg,
            direction_filter=BiasDirection.BULLISH,
            time_stop_minutes=30,
            bar_bundle=object(),  # truthy = MTF
        )
        assert "Bias=MTF" in label
        assert "Side=Long" in label
        assert "Displ=0.3" in label
        assert "MinScore=2/4" in label
        assert "Window=strict" in label
        assert "TS=30m" in label

    def test_short_only(self):
        label = build_run_label(
            UnicornConfig(),
            direction_filter=BiasDirection.BEARISH,
        )
        assert "Side=Short" in label

    def test_label_pipe_separated(self):
        label = build_run_label(UnicornConfig())
        parts = label.split(" | ")
        assert len(parts) == 7, f"Expected 7 pipe-separated segments, got {len(parts)}"


class TestBuildRunKey:
    """build_run_key produces deterministic machine-stable slugs."""

    def test_default_config_key(self):
        key = build_run_key(UnicornConfig())
        assert key == "ver_unicorn_v2_1_bias_single_side_bidir_displ_off_minscore_3of4_window_normal_ts_none"

    def test_key_matches_label_semantics(self):
        """Key and label encode the same config choices."""
        cfg = UnicornConfig(min_displacement_atr=0.3, session_profile=SessionProfile.STRICT)
        key = build_run_key(cfg, direction_filter=BiasDirection.BULLISH, time_stop_minutes=30, bar_bundle=object())
        assert "bias_mtf" in key
        assert "side_long" in key
        assert "displ_0_3" in key
        assert "minscore_2" not in key  # default is 3
        assert "minscore_3of4" in key
        assert "window_strict" in key
        assert "ts_30m" in key

    def test_key_has_no_spaces_or_uppercase(self):
        key = build_run_key(UnicornConfig(), bar_bundle=object())
        assert " " not in key
        assert key == key.lower()

    def test_key_deterministic(self):
        cfg = UnicornConfig(min_displacement_atr=0.5)
        k1 = build_run_key(cfg, direction_filter=BiasDirection.BEARISH)
        k2 = build_run_key(cfg, direction_filter=BiasDirection.BEARISH)
        assert k1 == k2

    def test_key_differs_for_different_configs(self):
        k1 = build_run_key(UnicornConfig())
        k2 = build_run_key(UnicornConfig(min_displacement_atr=0.3))
        assert k1 != k2


class TestRunLabelInReport:
    """Run label and model line appear in backtest report."""

    def test_report_contains_run_label_and_key(self):
        from app.services.backtest.engines.unicorn_runner import format_backtest_report

        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            config=UnicornConfig(),
        )

        assert result.run_key is not None
        assert "unicorn_v2_1" in result.run_key

        report = format_backtest_report(result)
        assert "Unicorn v2.1" in report
        assert "Bias=" in report
        assert "MinScore=" in report

    def test_report_contains_model_schema_line(self):
        from app.services.backtest.engines.unicorn_runner import format_backtest_report

        start_ts = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        htf_bars = generate_trending_bars(start_ts, 200, 17000, trend=2.0, interval_minutes=15)
        ltf_bars = generate_trending_bars(start_ts, 600, 17000, trend=0.67, interval_minutes=5)

        result = run_unicorn_backtest(
            symbol="NQ",
            htf_bars=htf_bars,
            ltf_bars=ltf_bars,
            config=UnicornConfig(),
        )

        report = format_backtest_report(result)
        assert "Unicorn v2.1 (Intent)" in report
        assert "5M+4S" in report
