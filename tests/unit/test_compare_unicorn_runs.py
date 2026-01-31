"""
Tests for scripts/compare_unicorn_runs.py
"""

import sys
from pathlib import Path

# Allow importing the script as a module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from compare_unicorn_runs import compare_runs, MANDATORY_CRITERIA


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
