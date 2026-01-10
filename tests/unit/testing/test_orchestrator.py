"""Unit tests for RunOrchestrator - TDD style."""

import csv
import io
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4, uuid5

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.testing.models import (
    RunPlan,
    RunVariant,
    RunResult,
    RunResultStatus,
    VariantMetrics,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_run_plan() -> RunPlan:
    """Create a sample RunPlan for testing."""
    base_spec = {
        "strategy_id": "breakout_52w_high",
        "name": "Test Strategy",
        "workspace_id": str(uuid4()),
        "symbols": ["AAPL"],
        "timeframe": "daily",
        "entry": {"type": "breakout_52w_high", "lookback_days": 252},
        "exit": {"type": "eod"},
        "risk": {"dollars_per_trade": 1000.0, "max_positions": 5},
    }

    return RunPlan(
        workspace_id=uuid4(),
        base_spec=base_spec,
        variants=[
            RunVariant(
                variant_id="abc123",
                label="baseline",
                spec_overrides={},
                tags=["baseline"],
            ),
            RunVariant(
                variant_id="def456",
                label="lookback=126",
                spec_overrides={"entry.lookback_days": 126},
                tags=["grid"],
            ),
        ],
        objective="sharpe_dd_penalty",
        dataset_ref="btc_2023",
    )


@pytest.fixture
def sample_csv_content() -> bytes:
    """Create sample OHLCV CSV content."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ts", "open", "high", "low", "close", "volume"])
    writer.writerow(["2023-01-01T00:00:00", "100.0", "105.0", "99.0", "104.0", "1000"])
    writer.writerow(["2023-01-02T00:00:00", "104.0", "108.0", "103.0", "107.0", "1200"])
    writer.writerow(["2023-01-03T00:00:00", "107.0", "110.0", "106.0", "109.0", "1100"])
    return output.getvalue().encode("utf-8")


@pytest.fixture
def sample_csv_5_bars() -> bytes:
    """Create sample OHLCV CSV with 5 bars for more realistic testing."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ts", "open", "high", "low", "close", "volume"])
    writer.writerow(["2023-01-01T00:00:00", "100.0", "105.0", "99.0", "104.0", "1000"])
    writer.writerow(["2023-01-02T00:00:00", "104.0", "108.0", "103.0", "107.0", "1200"])
    writer.writerow(["2023-01-03T00:00:00", "107.0", "110.0", "106.0", "109.0", "1100"])
    writer.writerow(["2023-01-04T00:00:00", "109.0", "112.0", "108.0", "111.0", "1300"])
    writer.writerow(["2023-01-05T00:00:00", "111.0", "115.0", "110.0", "114.0", "1400"])
    return output.getvalue().encode("utf-8")


# =============================================================================
# A) Pure Math Functions Tests
# =============================================================================


class TestComputeMaxDrawdown:
    """Tests for _compute_max_drawdown static method."""

    def test_empty_curve_returns_zero(self):
        """Empty equity curve should return 0.0 drawdown."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        result = RunOrchestrator._compute_max_drawdown([])
        assert result == 0.0

    def test_single_element_returns_zero(self):
        """Single element curve should return 0.0 drawdown."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        result = RunOrchestrator._compute_max_drawdown([100.0])
        assert result == 0.0

    def test_known_curve_correct_drawdown(self):
        """Known equity curve should return correct max drawdown.

        Curve: [100, 110, 95, 105]
        Peak = 110, Trough after peak = 95
        Drawdown = (110 - 95) / 110 = 15/110 = 13.636...%
        """
        from app.services.testing.run_orchestrator import RunOrchestrator

        curve = [100.0, 110.0, 95.0, 105.0]
        result = RunOrchestrator._compute_max_drawdown(curve)
        expected = (110.0 - 95.0) / 110.0 * 100  # 13.636...
        assert abs(result - expected) < 0.01

    def test_monotonic_increasing_returns_zero(self):
        """Monotonically increasing curve should return 0.0 drawdown."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        curve = [100.0, 110.0, 120.0, 130.0]
        result = RunOrchestrator._compute_max_drawdown(curve)
        assert result == 0.0

    def test_all_losses_correct_max_dd(self):
        """All losses curve should return correct max drawdown.

        Curve: [100, 90, 80, 70]
        Peak = 100, Trough = 70
        Drawdown = (100 - 70) / 100 = 30%
        """
        from app.services.testing.run_orchestrator import RunOrchestrator

        curve = [100.0, 90.0, 80.0, 70.0]
        result = RunOrchestrator._compute_max_drawdown(curve)
        expected = 30.0  # 30%
        assert abs(result - expected) < 0.01

    def test_multiple_drawdowns_returns_max(self):
        """Should return the maximum of multiple drawdowns.

        Curve: [100, 95, 110, 90, 115]
        First drawdown: 100 -> 95 = 5%
        Second drawdown: 110 -> 90 = 18.18%
        Max should be 18.18%
        """
        from app.services.testing.run_orchestrator import RunOrchestrator

        curve = [100.0, 95.0, 110.0, 90.0, 115.0]
        result = RunOrchestrator._compute_max_drawdown(curve)
        expected = (110.0 - 90.0) / 110.0 * 100  # 18.18%
        assert abs(result - expected) < 0.01


class TestComputeTradeSharpe:
    """Tests for _compute_trade_sharpe static method."""

    def test_less_than_3_elements_returns_none(self):
        """Less than 3 elements (means <2 trades) should return None."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        assert RunOrchestrator._compute_trade_sharpe([]) is None
        assert RunOrchestrator._compute_trade_sharpe([100.0]) is None
        assert RunOrchestrator._compute_trade_sharpe([100.0, 110.0]) is None

    def test_zero_std_returns_none(self):
        """Zero standard deviation (all same returns) should return None."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # Constant equity -> all zero returns -> std = 0
        curve = [100.0, 100.0, 100.0, 100.0]
        result = RunOrchestrator._compute_trade_sharpe(curve)
        assert result is None

    def test_known_returns_expected_sharpe(self):
        """Known returns should produce expected Sharpe.

        Curve: [100, 110, 105, 115]
        Returns: [10%, -4.545%, 9.524%]
        Mean = (10 - 4.545 + 9.524) / 3 = 4.993%
        Sharpe = mean / std
        """
        from app.services.testing.run_orchestrator import RunOrchestrator

        curve = [100.0, 110.0, 105.0, 115.0]
        result = RunOrchestrator._compute_trade_sharpe(curve)
        # Should be a positive number (mean is positive)
        assert result is not None
        assert result > 0

    def test_negative_returns_negative_sharpe(self):
        """Negative returns should produce negative Sharpe."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # Mostly negative returns
        curve = [100.0, 95.0, 90.0, 85.0]  # -5%, -5.26%, -5.56%
        result = RunOrchestrator._compute_trade_sharpe(curve)
        assert result is not None
        assert result < 0


class TestComputeObjectiveScore:
    """Tests for _compute_objective_score static method."""

    def test_sharpe_objective_with_none_sharpe_returns_sentinel(self):
        """Sharpe objective with None sharpe should return -999.0."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        metrics = VariantMetrics(
            sharpe=None,
            return_pct=10.0,
            max_drawdown_pct=5.0,
            trade_count=5,
            win_rate=0.6,
            ending_equity=11000.0,
            gross_profit=1500.0,
            gross_loss=-500.0,
        )
        result = RunOrchestrator._compute_objective_score(metrics, "sharpe")
        assert result == -999.0

    def test_sharpe_objective_returns_sharpe(self):
        """Sharpe objective should return sharpe value."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        metrics = VariantMetrics(
            sharpe=1.5,
            return_pct=10.0,
            max_drawdown_pct=5.0,
            trade_count=5,
            win_rate=0.6,
            ending_equity=11000.0,
            gross_profit=1500.0,
            gross_loss=-500.0,
        )
        result = RunOrchestrator._compute_objective_score(metrics, "sharpe")
        assert result == 1.5

    def test_sharpe_dd_penalty_objective(self):
        """sharpe_dd_penalty should be sharpe - 0.5 * max_dd/100."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        metrics = VariantMetrics(
            sharpe=2.0,
            return_pct=20.0,
            max_drawdown_pct=10.0,  # 10%
            trade_count=10,
            win_rate=0.6,
            ending_equity=12000.0,
            gross_profit=2500.0,
            gross_loss=-500.0,
        )
        result = RunOrchestrator._compute_objective_score(metrics, "sharpe_dd_penalty")
        # 2.0 - 0.5 * (10/100) = 2.0 - 0.05 = 1.95
        expected = 2.0 - 0.5 * (10.0 / 100.0)
        assert abs(result - expected) < 0.001

    def test_return_objective(self):
        """Return objective should return return_pct."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        metrics = VariantMetrics(
            sharpe=1.0,
            return_pct=15.5,
            max_drawdown_pct=8.0,
            trade_count=8,
            win_rate=0.5,
            ending_equity=11550.0,
            gross_profit=2000.0,
            gross_loss=-450.0,
        )
        result = RunOrchestrator._compute_objective_score(metrics, "return")
        assert result == 15.5

    def test_calmar_with_zero_drawdown_returns_zero(self):
        """Calmar with zero drawdown should return 0.0 (avoid division by zero)."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        metrics = VariantMetrics(
            sharpe=1.0,
            return_pct=20.0,
            max_drawdown_pct=0.0,  # Zero drawdown
            trade_count=5,
            win_rate=0.8,
            ending_equity=12000.0,
            gross_profit=2000.0,
            gross_loss=0.0,
        )
        result = RunOrchestrator._compute_objective_score(metrics, "calmar")
        assert result == 0.0

    def test_calmar_normal_calculation(self):
        """Calmar should be return_pct / abs(max_drawdown_pct)."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        metrics = VariantMetrics(
            sharpe=1.5,
            return_pct=30.0,
            max_drawdown_pct=10.0,
            trade_count=12,
            win_rate=0.65,
            ending_equity=13000.0,
            gross_profit=3500.0,
            gross_loss=-500.0,
        )
        result = RunOrchestrator._compute_objective_score(metrics, "calmar")
        # 30.0 / 10.0 = 3.0
        assert result == 3.0

    def test_return_dd_penalty_objective(self):
        """return_dd_penalty should be return_pct - lambda * max_dd."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        metrics = VariantMetrics(
            sharpe=1.0,
            return_pct=25.0,
            max_drawdown_pct=10.0,
            trade_count=10,
            win_rate=0.6,
            ending_equity=12500.0,
            gross_profit=3000.0,
            gross_loss=-500.0,
        )
        result = RunOrchestrator._compute_objective_score(metrics, "return_dd_penalty")
        # 25.0 - 0.5 * 10.0 = 25.0 - 5.0 = 20.0
        expected = 25.0 - 0.5 * 10.0
        assert abs(result - expected) < 0.001


class TestCalculateMetrics:
    """Tests for _calculate_metrics method."""

    def test_no_trades_returns_zero_metrics(self):
        """No trades should return trade_count=0, win_rate=0.0, sharpe=None."""
        from app.services.testing.run_orchestrator import RunOrchestrator
        from app.schemas import PaperState

        starting_equity = 10000.0
        paper_state = PaperState(
            workspace_id=uuid4(),
            starting_equity=starting_equity,
            cash=starting_equity,
            realized_pnl=0.0,
        )
        closed_trades = []

        metrics = RunOrchestrator._calculate_metrics(
            starting_equity, paper_state, closed_trades
        )

        assert metrics.trade_count == 0
        assert metrics.win_rate == 0.0
        assert metrics.sharpe is None
        assert metrics.return_pct == 0.0
        assert metrics.max_drawdown_pct == 0.0

    def test_profit_factor_none_if_no_losses(self):
        """Profit factor should be None if no losses."""
        from app.services.testing.run_orchestrator import RunOrchestrator
        from app.schemas import PaperState

        starting_equity = 10000.0
        paper_state = PaperState(
            workspace_id=uuid4(),
            starting_equity=starting_equity,
            cash=11000.0,  # Made profit
            realized_pnl=1000.0,
        )
        # All winning trades
        closed_trades = [
            {"pnl": 500.0, "exit_equity": 10500.0},
            {"pnl": 500.0, "exit_equity": 11000.0},
        ]

        metrics = RunOrchestrator._calculate_metrics(
            starting_equity, paper_state, closed_trades
        )

        assert metrics.profit_factor is None  # No losses
        assert metrics.gross_loss == 0.0

    def test_win_rate_calculation(self):
        """Win rate should be wins / total."""
        from app.services.testing.run_orchestrator import RunOrchestrator
        from app.schemas import PaperState

        starting_equity = 10000.0
        paper_state = PaperState(
            workspace_id=uuid4(),
            starting_equity=starting_equity,
            cash=10300.0,
            realized_pnl=300.0,
        )
        # 2 wins, 1 loss
        closed_trades = [
            {"pnl": 200.0, "exit_equity": 10200.0},
            {"pnl": -100.0, "exit_equity": 10100.0},
            {"pnl": 200.0, "exit_equity": 10300.0},
        ]

        metrics = RunOrchestrator._calculate_metrics(
            starting_equity, paper_state, closed_trades
        )

        assert metrics.trade_count == 3
        assert abs(metrics.win_rate - 2 / 3) < 0.001

    def test_equity_curve_step_logic(self):
        """Equity curve should be built correctly at closed trade points."""
        from app.services.testing.run_orchestrator import RunOrchestrator
        from app.schemas import PaperState

        starting_equity = 10000.0
        paper_state = PaperState(
            workspace_id=uuid4(),
            starting_equity=starting_equity,
            cash=10500.0,
            realized_pnl=500.0,
        )
        closed_trades = [
            {"pnl": 200.0, "exit_equity": 10200.0},
            {"pnl": -100.0, "exit_equity": 10100.0},
            {"pnl": 400.0, "exit_equity": 10500.0},
        ]

        metrics = RunOrchestrator._calculate_metrics(
            starting_equity, paper_state, closed_trades
        )

        # Should have computed max drawdown from equity curve:
        # [10000, 10200, 10100, 10500]
        # Peak = 10200, Trough = 10100, DD = (10200-10100)/10200 = 0.98%
        expected_dd = (10200.0 - 10100.0) / 10200.0 * 100
        assert abs(metrics.max_drawdown_pct - expected_dd) < 0.1


# =============================================================================
# B) Variant Isolation Tests
# =============================================================================


class TestVariantIsolation:
    """Tests for variant isolation using uuid5."""

    def test_different_variants_get_different_namespaces(self, sample_run_plan):
        """Two variants with different IDs should get different namespace UUIDs."""
        from app.services.testing.run_orchestrator import VARIANT_NS

        run_plan_id = sample_run_plan.run_plan_id
        variant1_id = sample_run_plan.variants[0].variant_id
        variant2_id = sample_run_plan.variants[1].variant_id

        ns1 = uuid5(VARIANT_NS, f"{run_plan_id}:{variant1_id}")
        ns2 = uuid5(VARIANT_NS, f"{run_plan_id}:{variant2_id}")

        assert ns1 != ns2

    def test_same_variant_produces_consistent_uuid(self, sample_run_plan):
        """Same variant should always produce the same namespace UUID."""
        from app.services.testing.run_orchestrator import VARIANT_NS

        run_plan_id = sample_run_plan.run_plan_id
        variant_id = sample_run_plan.variants[0].variant_id

        ns1 = uuid5(VARIANT_NS, f"{run_plan_id}:{variant_id}")
        ns2 = uuid5(VARIANT_NS, f"{run_plan_id}:{variant_id}")

        assert ns1 == ns2


# =============================================================================
# C) CSV Parsing Tests
# =============================================================================


class TestParseOhlcvCsv:
    """Tests for _parse_ohlcv_csv method."""

    def test_valid_csv_correct_bar_count(self, sample_csv_content):
        """Valid CSV should return correct number of bars."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        bars = RunOrchestrator._parse_ohlcv_csv(sample_csv_content)
        assert len(bars) == 3

    def test_valid_csv_bar_values(self, sample_csv_content):
        """Valid CSV should parse bar values correctly."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        bars = RunOrchestrator._parse_ohlcv_csv(sample_csv_content)

        # Check first bar
        assert bars[0].open == 100.0
        assert bars[0].high == 105.0
        assert bars[0].low == 99.0
        assert bars[0].close == 104.0
        assert bars[0].volume == 1000.0

    def test_invalid_csv_missing_columns_raises_value_error(self):
        """Invalid CSV with missing columns should raise ValueError."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # Missing 'volume' column
        bad_csv = b"ts,open,high,low,close\n2023-01-01T00:00:00,100,105,99,104"

        with pytest.raises(ValueError, match="Missing required columns"):
            RunOrchestrator._parse_ohlcv_csv(bad_csv)

    def test_less_than_2_bars_raises_value_error(self):
        """Less than 2 bars should raise ValueError."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        single_bar_csv = b"ts,open,high,low,close,volume\n2023-01-01T00:00:00,100,105,99,104,1000"

        with pytest.raises(ValueError, match="at least 2 bars"):
            RunOrchestrator._parse_ohlcv_csv(single_bar_csv)

    def test_empty_csv_raises_value_error(self):
        """Empty CSV should raise ValueError."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        empty_csv = b"ts,open,high,low,close,volume\n"

        with pytest.raises(ValueError, match="at least 2 bars"):
            RunOrchestrator._parse_ohlcv_csv(empty_csv)


# =============================================================================
# D) select_best_variant Tests
# =============================================================================


class TestSelectBestVariant:
    """Tests for select_best_variant function."""

    def test_empty_results_returns_none_none(self):
        """Empty results should return (None, None)."""
        from app.services.testing.run_orchestrator import select_best_variant

        best_id, best_score = select_best_variant([])
        assert best_id is None
        assert best_score is None

    def test_single_result_returns_it(self):
        """Single result should be returned as best."""
        from app.services.testing.run_orchestrator import select_best_variant

        result = RunResult(
            run_plan_id=uuid4(),
            variant_id="abc123",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.5,
                return_pct=15.0,
                max_drawdown_pct=8.0,
                trade_count=10,
                win_rate=0.6,
                ending_equity=11500.0,
                gross_profit=2000.0,
                gross_loss=-500.0,
            ),
            objective_score=1.5,
            started_at=datetime.now(timezone.utc),
        )

        best_id, best_score = select_best_variant([result])
        assert best_id == "abc123"
        assert best_score == 1.5

    def test_highest_objective_wins(self):
        """Highest objective score should win."""
        from app.services.testing.run_orchestrator import select_best_variant

        result1 = RunResult(
            run_plan_id=uuid4(),
            variant_id="low_score",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.0,
                return_pct=10.0,
                max_drawdown_pct=5.0,
                trade_count=5,
                win_rate=0.6,
                ending_equity=11000.0,
                gross_profit=1200.0,
                gross_loss=-200.0,
            ),
            objective_score=1.0,
            started_at=datetime.now(timezone.utc),
        )
        result2 = RunResult(
            run_plan_id=uuid4(),
            variant_id="high_score",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=2.0,
                return_pct=20.0,
                max_drawdown_pct=7.0,
                trade_count=8,
                win_rate=0.7,
                ending_equity=12000.0,
                gross_profit=2500.0,
                gross_loss=-500.0,
            ),
            objective_score=2.0,
            started_at=datetime.now(timezone.utc),
        )

        best_id, best_score = select_best_variant([result1, result2])
        assert best_id == "high_score"
        assert best_score == 2.0

    def test_tie_break_by_higher_return(self):
        """Same objective score should tie-break by higher return."""
        from app.services.testing.run_orchestrator import select_best_variant

        result1 = RunResult(
            run_plan_id=uuid4(),
            variant_id="lower_return",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.5,
                return_pct=10.0,
                max_drawdown_pct=5.0,
                trade_count=5,
                win_rate=0.6,
                ending_equity=11000.0,
                gross_profit=1200.0,
                gross_loss=-200.0,
            ),
            objective_score=1.5,
            started_at=datetime.now(timezone.utc),
        )
        result2 = RunResult(
            run_plan_id=uuid4(),
            variant_id="higher_return",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.5,
                return_pct=15.0,  # Higher return
                max_drawdown_pct=7.0,
                trade_count=8,
                win_rate=0.7,
                ending_equity=11500.0,
                gross_profit=1800.0,
                gross_loss=-300.0,
            ),
            objective_score=1.5,
            started_at=datetime.now(timezone.utc),
        )

        best_id, best_score = select_best_variant([result1, result2])
        assert best_id == "higher_return"

    def test_tie_break_by_lower_drawdown(self):
        """Same objective and return should tie-break by lower drawdown."""
        from app.services.testing.run_orchestrator import select_best_variant

        result1 = RunResult(
            run_plan_id=uuid4(),
            variant_id="higher_dd",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.5,
                return_pct=15.0,
                max_drawdown_pct=10.0,  # Higher drawdown
                trade_count=5,
                win_rate=0.6,
                ending_equity=11500.0,
                gross_profit=1700.0,
                gross_loss=-200.0,
            ),
            objective_score=1.5,
            started_at=datetime.now(timezone.utc),
        )
        result2 = RunResult(
            run_plan_id=uuid4(),
            variant_id="lower_dd",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.5,
                return_pct=15.0,
                max_drawdown_pct=5.0,  # Lower drawdown
                trade_count=8,
                win_rate=0.7,
                ending_equity=11500.0,
                gross_profit=1700.0,
                gross_loss=-200.0,
            ),
            objective_score=1.5,
            started_at=datetime.now(timezone.utc),
        )

        best_id, best_score = select_best_variant([result1, result2])
        assert best_id == "lower_dd"

    def test_tie_break_by_smaller_variant_id(self):
        """All else equal, tie-break by smaller variant_id (deterministic)."""
        from app.services.testing.run_orchestrator import select_best_variant

        result1 = RunResult(
            run_plan_id=uuid4(),
            variant_id="zzz999",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.5,
                return_pct=15.0,
                max_drawdown_pct=5.0,
                trade_count=5,
                win_rate=0.6,
                ending_equity=11500.0,
                gross_profit=1700.0,
                gross_loss=-200.0,
            ),
            objective_score=1.5,
            started_at=datetime.now(timezone.utc),
        )
        result2 = RunResult(
            run_plan_id=uuid4(),
            variant_id="aaa111",  # Smaller lexicographically
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.5,
                return_pct=15.0,
                max_drawdown_pct=5.0,
                trade_count=8,
                win_rate=0.7,
                ending_equity=11500.0,
                gross_profit=1700.0,
                gross_loss=-200.0,
            ),
            objective_score=1.5,
            started_at=datetime.now(timezone.utc),
        )

        best_id, best_score = select_best_variant([result1, result2])
        assert best_id == "aaa111"

    def test_failed_results_are_excluded(self):
        """Failed results should be excluded from best selection."""
        from app.services.testing.run_orchestrator import select_best_variant

        result1 = RunResult(
            run_plan_id=uuid4(),
            variant_id="failed_one",
            status=RunResultStatus.failed,
            metrics=None,
            objective_score=None,
            error="Simulation error",
            started_at=datetime.now(timezone.utc),
        )
        result2 = RunResult(
            run_plan_id=uuid4(),
            variant_id="success_one",
            status=RunResultStatus.success,
            metrics=VariantMetrics(
                sharpe=1.0,
                return_pct=10.0,
                max_drawdown_pct=5.0,
                trade_count=5,
                win_rate=0.6,
                ending_equity=11000.0,
                gross_profit=1200.0,
                gross_loss=-200.0,
            ),
            objective_score=1.0,
            started_at=datetime.now(timezone.utc),
        )

        best_id, best_score = select_best_variant([result1, result2])
        assert best_id == "success_one"

    def test_all_failed_returns_none(self):
        """All failed results should return (None, None)."""
        from app.services.testing.run_orchestrator import select_best_variant

        result1 = RunResult(
            run_plan_id=uuid4(),
            variant_id="failed1",
            status=RunResultStatus.failed,
            metrics=None,
            objective_score=None,
            error="Error 1",
            started_at=datetime.now(timezone.utc),
        )
        result2 = RunResult(
            run_plan_id=uuid4(),
            variant_id="failed2",
            status=RunResultStatus.failed,
            metrics=None,
            objective_score=None,
            error="Error 2",
            started_at=datetime.now(timezone.utc),
        )

        best_id, best_score = select_best_variant([result1, result2])
        assert best_id is None
        assert best_score is None


# =============================================================================
# E) Orchestrator Execute Tests (Integration with Mocks)
# =============================================================================


class TestOrchestratorExecute:
    """Integration tests for RunOrchestrator.execute() with mocks."""

    @pytest.mark.asyncio
    async def test_execute_returns_results_for_all_variants(
        self, sample_run_plan, sample_csv_5_bars
    ):
        """Execute should return a RunResult for each variant."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # Mock dependencies
        mock_events_repo = AsyncMock()
        mock_events_repo.insert.return_value = uuid4()

        mock_runner = MagicMock()

        orchestrator = RunOrchestrator(mock_events_repo, mock_runner)

        # Mock _execute_variant to return success
        async def mock_execute_variant(run_plan, variant, bars):
            return RunResult(
                run_plan_id=run_plan.run_plan_id,
                variant_id=variant.variant_id,
                status=RunResultStatus.success,
                metrics=VariantMetrics(
                    sharpe=1.0,
                    return_pct=10.0,
                    max_drawdown_pct=5.0,
                    trade_count=3,
                    win_rate=0.67,
                    ending_equity=11000.0,
                    gross_profit=1200.0,
                    gross_loss=-200.0,
                ),
                objective_score=1.0,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

        with patch.object(orchestrator, "_execute_variant", side_effect=mock_execute_variant):
            results = await orchestrator.execute(sample_run_plan, sample_csv_5_bars)

        # Should have one result per variant
        assert len(results) == len(sample_run_plan.variants)

    @pytest.mark.asyncio
    async def test_execute_journals_run_started_event(
        self, sample_run_plan, sample_csv_5_bars
    ):
        """Execute should journal a RUN_STARTED event."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        mock_events_repo = AsyncMock()
        mock_events_repo.insert.return_value = uuid4()

        mock_runner = MagicMock()

        orchestrator = RunOrchestrator(mock_events_repo, mock_runner)

        # Mock _execute_variant
        async def mock_execute_variant(run_plan, variant, bars):
            return RunResult(
                run_plan_id=run_plan.run_plan_id,
                variant_id=variant.variant_id,
                status=RunResultStatus.success,
                metrics=VariantMetrics(
                    sharpe=1.0,
                    return_pct=10.0,
                    max_drawdown_pct=5.0,
                    trade_count=3,
                    win_rate=0.67,
                    ending_equity=11000.0,
                    gross_profit=1200.0,
                    gross_loss=-200.0,
                ),
                objective_score=1.0,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

        with patch.object(orchestrator, "_execute_variant", side_effect=mock_execute_variant):
            await orchestrator.execute(sample_run_plan, sample_csv_5_bars)

        # Check that insert was called (for RUN_STARTED and RUN_COMPLETED)
        assert mock_events_repo.insert.call_count >= 1


class TestVARIANT_NS:
    """Tests for VARIANT_NS constant."""

    def test_variant_ns_is_valid_uuid(self):
        """VARIANT_NS should be a valid UUID constant (alias for TESTING_VARIANT_NAMESPACE)."""
        from app.services.testing.run_orchestrator import VARIANT_NS
        from app.services.testing.models import TESTING_VARIANT_NAMESPACE

        assert isinstance(VARIANT_NS, UUID)
        # VARIANT_NS is now an alias for TESTING_VARIANT_NAMESPACE
        assert VARIANT_NS == TESTING_VARIANT_NAMESPACE
        assert str(VARIANT_NS) == "8f3c2a2e-9d9a-4b2e-9f6b-3c3a4c1e9a01"


class TestSkippedVariants:
    """Tests for skipped variant handling."""

    @pytest.mark.asyncio
    async def test_skipped_variant_has_skip_reason_not_error(self, sample_csv_5_bars):
        """Skipped variants should set skip_reason, not error field."""
        from app.services.testing.run_orchestrator import RunOrchestrator
        from app.services.testing.models import RunPlan, RunVariant

        # Create run plan with a variant that has invalid params
        base_spec = {
            "strategy_id": "breakout_52w_high",
            "name": "Test Strategy",
            "workspace_id": str(uuid4()),
            "symbols": ["AAPL"],
            "timeframe": "daily",
            "entry": {"type": "breakout_52w_high", "lookback_days": 252},
            "exit": {"type": "eod"},
            "risk": {"dollars_per_trade": 1000.0, "max_positions": 5},
        }

        run_plan = RunPlan(
            workspace_id=uuid4(),
            base_spec=base_spec,
            variants=[
                RunVariant(
                    variant_id="invalid_variant",
                    label="negative_dollars",
                    spec_overrides={"risk.dollars_per_trade": -100.0},  # Invalid!
                    tags=["grid"],
                ),
            ],
            objective="sharpe_dd_penalty",
            dataset_ref="test",
        )

        mock_events_repo = AsyncMock()
        mock_events_repo.insert.return_value = uuid4()
        mock_runner = MagicMock()

        orchestrator = RunOrchestrator(mock_events_repo, mock_runner)
        results = await orchestrator.execute(run_plan, sample_csv_5_bars)

        assert len(results) == 1
        result = results[0]
        assert result.status == RunResultStatus.skipped
        assert result.skip_reason is not None
        assert "dollars_per_trade" in result.skip_reason
        assert result.error is None  # error should be None for skipped

    @pytest.mark.asyncio
    async def test_skipped_variants_journal_in_run_completed(self, sample_csv_5_bars):
        """RUN_COMPLETED event should include skipped count."""
        from app.services.testing.run_orchestrator import RunOrchestrator
        from app.services.testing.models import RunPlan, RunVariant

        base_spec = {
            "strategy_id": "breakout_52w_high",
            "name": "Test Strategy",
            "workspace_id": str(uuid4()),
            "symbols": ["AAPL"],
            "timeframe": "daily",
            "entry": {"type": "breakout_52w_high", "lookback_days": 252},
            "exit": {"type": "eod"},
            "risk": {"dollars_per_trade": 1000.0, "max_positions": 5},
        }

        run_plan = RunPlan(
            workspace_id=uuid4(),
            base_spec=base_spec,
            variants=[
                RunVariant(
                    variant_id="valid_variant",
                    label="baseline",
                    spec_overrides={},
                    tags=["baseline"],
                ),
                RunVariant(
                    variant_id="invalid_variant",
                    label="negative_dollars",
                    spec_overrides={"risk.dollars_per_trade": -100.0},
                    tags=["grid"],
                ),
            ],
            objective="sharpe_dd_penalty",
            dataset_ref="test",
        )

        mock_events_repo = AsyncMock()
        mock_events_repo.insert.return_value = uuid4()
        mock_runner = MagicMock()

        orchestrator = RunOrchestrator(mock_events_repo, mock_runner)
        await orchestrator.execute(run_plan, sample_csv_5_bars)

        # Find the RUN_COMPLETED event call
        calls = mock_events_repo.insert.call_args_list
        run_completed_call = None
        for call in calls:
            event = call[0][0]
            if event.payload.get("run_event_type") == "RUN_COMPLETED":
                run_completed_call = event
                break

        assert run_completed_call is not None
        assert run_completed_call.payload["n_skipped"] == 1


class TestCsvRobustness:
    """Tests for CSV parsing robustness features."""

    def test_bom_stripped_from_csv(self):
        """UTF-8 BOM should be stripped from CSV content."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # Create CSV with UTF-8 BOM (EF BB BF)
        bom = b"\xef\xbb\xbf"
        csv_content = b"ts,open,high,low,close,volume\n"
        csv_content += b"2023-01-01T00:00:00,100,105,99,104,1000\n"
        csv_content += b"2023-01-02T00:00:00,104,108,103,107,1200\n"
        csv_with_bom = bom + csv_content

        bars = RunOrchestrator._parse_ohlcv_csv(csv_with_bom)
        assert len(bars) == 2

    def test_bom_in_column_name_stripped(self):
        """BOM embedded in first column name should be handled."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # BOM in first column name (can happen with some Excel exports)
        csv_content = "\ufeffts,open,high,low,close,volume\n"
        csv_content += "2023-01-01T00:00:00,100,105,99,104,1000\n"
        csv_content += "2023-01-02T00:00:00,104,108,103,107,1200\n"

        bars = RunOrchestrator._parse_ohlcv_csv(csv_content.encode("utf-8"))
        assert len(bars) == 2

    def test_monotonic_timestamps_required(self):
        """Non-monotonic timestamps should raise ValueError."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # Second timestamp is earlier than first
        csv_content = b"ts,open,high,low,close,volume\n"
        csv_content += b"2023-01-02T00:00:00,104,108,103,107,1200\n"
        csv_content += b"2023-01-01T00:00:00,100,105,99,104,1000\n"  # Earlier!

        with pytest.raises(ValueError, match="Non-monotonic timestamp"):
            RunOrchestrator._parse_ohlcv_csv(csv_content)

    def test_duplicate_timestamps_rejected(self):
        """Duplicate timestamps should be rejected (not strictly increasing)."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        # Same timestamp twice
        csv_content = b"ts,open,high,low,close,volume\n"
        csv_content += b"2023-01-01T00:00:00,100,105,99,104,1000\n"
        csv_content += b"2023-01-01T00:00:00,104,108,103,107,1200\n"  # Duplicate!

        with pytest.raises(ValueError, match="Non-monotonic timestamp"):
            RunOrchestrator._parse_ohlcv_csv(csv_content)

    def test_monotonic_timestamps_pass(self):
        """Properly ordered timestamps should pass validation."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        csv_content = b"ts,open,high,low,close,volume\n"
        csv_content += b"2023-01-01T00:00:00,100,105,99,104,1000\n"
        csv_content += b"2023-01-02T00:00:00,104,108,103,107,1200\n"
        csv_content += b"2023-01-03T00:00:00,107,110,106,109,1100\n"

        bars = RunOrchestrator._parse_ohlcv_csv(csv_content)
        assert len(bars) == 3
        # Verify order
        assert bars[0].ts < bars[1].ts < bars[2].ts

    def test_invalid_timestamp_format_fails_fast(self):
        """Invalid timestamp format should fail immediately."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        csv_content = b"ts,open,high,low,close,volume\n"
        csv_content += b"not-a-date,100,105,99,104,1000\n"
        csv_content += b"2023-01-02T00:00:00,104,108,103,107,1200\n"

        with pytest.raises(ValueError, match="Invalid row"):
            RunOrchestrator._parse_ohlcv_csv(csv_content)

    def test_missing_header_raises_error(self):
        """CSV with no header should raise clear error."""
        from app.services.testing.run_orchestrator import RunOrchestrator

        csv_content = b""

        with pytest.raises(ValueError, match="no header"):
            RunOrchestrator._parse_ohlcv_csv(csv_content)
