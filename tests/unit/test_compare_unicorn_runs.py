"""
Tests for scripts/compare_unicorn_runs.py and _build_output_dict helper.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

# Allow importing the script as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from compare_unicorn_runs import compare_runs, MANDATORY_CRITERIA
from run_unicorn_backtest import _build_output_dict


def _make_run(
    *,
    run_key="ver_unicorn_v2_1_test",
    trades_taken=10,
    win_rate=0.6,
    profit_factor=1.5,
    expectancy_points=2.0,
    total_pnl_points=20.0,
    total_pnl_dollars=400.0,
    largest_loss_points=-5.0,
    bottlenecks=None,
) -> dict:
    if bottlenecks is None:
        bottlenecks = [
            {"criterion": "macro_window", "fail_rate": 0.4},
            {"criterion": "htf_bias", "fail_rate": 0.3},
            {"criterion": "mss", "fail_rate": 0.2},
            {"criterion": "liquidity_sweep", "fail_rate": 0.35},
            {"criterion": "htf_fvg", "fail_rate": 0.25},
            {"criterion": "breaker_block", "fail_rate": 0.15},
        ]
    return {
        "run_key": run_key,
        "trades_taken": trades_taken,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_points": expectancy_points,
        "total_pnl_points": total_pnl_points,
        "total_pnl_dollars": total_pnl_dollars,
        "largest_loss_points": largest_loss_points,
        "criteria_bottlenecks": bottlenecks,
    }


class TestCompareIdenticalRuns:
    def test_compare_identical_runs_zero_deltas(self):
        a = _make_run()
        b = _make_run()
        output = compare_runs(a, b)

        # All deltas should be zero / +0
        assert "+0" in output or "0.00" in output
        # Identity section present
        assert "Run A:" in output
        assert "Run B:" in output
        assert "ver_unicorn_v2_1_test" in output


class TestCompareDifferentRuns:
    def test_compare_different_runs_correct_deltas(self):
        a = _make_run(trades_taken=10, total_pnl_points=20.0, win_rate=0.5)
        b = _make_run(
            run_key="ver_unicorn_v2_1_other",
            trades_taken=15,
            total_pnl_points=35.0,
            win_rate=0.7,
        )
        output = compare_runs(a, b)

        # Trades delta = +5
        assert "+5" in output
        # PnL delta = +15.00
        assert "+15.00" in output
        # Both keys present
        assert "ver_unicorn_v2_1_test" in output
        assert "ver_unicorn_v2_1_other" in output


class TestBottleneckSplit:
    def test_bottleneck_split_mandatory_vs_scored(self):
        a = _make_run()
        b = _make_run()
        output = compare_runs(a, b)

        assert "[M]" in output
        assert "[S]" in output
        assert "Mandatory (top 3):" in output
        assert "Scored (top 3):" in output

        # Mandatory criteria appear under [M]
        for line in output.split("\n"):
            if "[M]" in line:
                crit = line.split("[M]")[1].strip().split()[0]
                assert crit in MANDATORY_CRITERIA, f"{crit} not in MANDATORY_CRITERIA"
            if "[S]" in line:
                crit = line.split("[S]")[1].strip().split()[0]
                assert crit not in MANDATORY_CRITERIA, f"{crit} should not be mandatory"


class TestMissingOptionalFields:
    def test_missing_optional_fields_no_crash(self):
        a = {"run_key": "a_key"}
        b = {}
        # Should not raise
        output = compare_runs(a, b)
        assert "RUN COMPARISON" in output
        assert "a_key" in output
        assert "N/A" in output  # b has no run_key

    def test_empty_bottlenecks_no_crash(self):
        a = _make_run(bottlenecks=[])
        b = _make_run(bottlenecks=[])
        output = compare_runs(a, b)
        assert "no bottleneck data" in output


# ---------------------------------------------------------------------------
# Helpers for _build_output_dict tests
# ---------------------------------------------------------------------------

class _FakeSession:
    """Hashable stand-in for TradingSession enum."""
    def __init__(self, value: str):
        self.value = value
    def __hash__(self):
        return hash(self.value)
    def __eq__(self, other):
        return self.value == getattr(other, "value", other)


def _fake_session_stats():
    """Minimal session stats mapping keyed by a session-like enum."""
    _sess = _FakeSession("ny_am")
    return {
        _sess: SimpleNamespace(
            total_setups=5, valid_setups=3, trades_taken=2,
            win_rate=0.5, total_pnl_points=10.0,
        ),
    }


def _fake_result(**overrides):
    """Build a minimal result namespace that _build_output_dict can consume."""
    defaults = dict(
        run_key="ver_unicorn_v2_1_test",
        run_label="Unicorn v2.1 test",
        symbol="NQ",
        start_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
        total_bars=500,
        total_setups_scanned=100,
        partial_setups=40,
        valid_setups=20,
        trades_taken=10,
        wins=6,
        losses=4,
        win_rate=0.6,
        profit_factor=1.5,
        total_pnl_points=20.0,
        total_pnl_dollars=400.0,
        expectancy_points=2.0,
        avg_mfe=5.0,
        avg_mae=-3.0,
        mfe_capture_rate=0.6,
        avg_r_multiple=0.8,
        largest_loss_points=-5.0,
        confidence_win_correlation=0.3,
        criteria_bottlenecks=[
            SimpleNamespace(criterion="macro_window", fail_rate=0.4),
            SimpleNamespace(criterion="htf_fvg", fail_rate=0.25),
        ],
        session_stats=_fake_session_stats(),
        confidence_buckets=[
            SimpleNamespace(
                min_confidence=0.5, max_confidence=0.7,
                trade_count=4, win_rate=0.75, avg_r_multiple=1.2,
            ),
        ],
        session_diagnostics={"intermarket_agreement": None},
        governor_stats=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_config():
    return SimpleNamespace(
        min_scored_criteria=3,
        min_displacement_atr=None,
        session_profile=SimpleNamespace(value="normal"),
    )


def _fake_args(**overrides):
    defaults = dict(long_only=False, time_stop=None)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# _build_output_dict tests
# ---------------------------------------------------------------------------

class TestBuildOutputDict:
    """Verify _build_output_dict produces all keys compare_runs reads."""

    COMPARE_KEYS = {
        "run_key", "trades_taken", "win_rate", "profit_factor",
        "expectancy_points", "total_pnl_points", "total_pnl_dollars",
        "largest_loss_points", "criteria_bottlenecks",
    }

    def test_build_output_dict_has_required_keys(self):
        result = _fake_result()
        config = _fake_config()
        args = _fake_args()

        output = _build_output_dict(result, config, args)

        for key in self.COMPARE_KEYS:
            assert key in output, f"Missing key: {key}"

        # largest_loss_points must be numeric
        assert isinstance(output["largest_loss_points"], (int, float))


class TestBaselineCompareIntegration:
    """Build two output dicts and verify compare_runs produces sane output."""

    def test_baseline_compare_integration(self):
        config = _fake_config()
        args = _fake_args()

        a = _build_output_dict(
            _fake_result(run_key="baseline_run", total_pnl_points=10.0),
            config, args,
        )
        b = _build_output_dict(
            _fake_result(run_key="current_run", total_pnl_points=25.0),
            config, args,
        )

        output = compare_runs(a, b)

        # Identity lines present
        assert "baseline_run" in output
        assert "current_run" in output
        # Delta markers present
        assert "+" in output
        # Section headers present
        assert "METRICS DELTA" in output


class TestWriteBaseline:
    """Verify _build_output_dict output round-trips through JSON."""

    def test_write_baseline_round_trip(self, tmp_path):
        import json

        config = _fake_config()
        args = _fake_args()
        output = _build_output_dict(_fake_result(), config, args)

        path = tmp_path / "baseline.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        with open(path) as f:
            loaded = json.load(f)

        # All compare keys survive round-trip
        for key in TestBuildOutputDict.COMPARE_KEYS:
            assert key in loaded, f"Missing key after round-trip: {key}"
            assert loaded[key] == output[key]
