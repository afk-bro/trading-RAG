"""Tests for WFO segments endpoint: stitched equity, stability stats, segment table."""

from uuid import uuid4

from app.routers.backtests.wfo_chart import (
    EquityPoint,
    WFOSegment,
    _compute_stability,
    _stitch_equity_curves,
)


# =============================================================================
# Stability stats
# =============================================================================


class TestComputeStability:
    """Param stability calculation."""

    def test_all_same_params(self):
        segments = [
            WFOSegment(fold_index=i, tune_id=str(uuid4()), status="completed",
                       best_params={"a": 1, "b": 2})
            for i in range(5)
        ]
        result = _compute_stability(segments)
        assert result is not None
        assert result.unique_param_sets == 1
        assert result.consistency == 1.0
        assert result.param_changes == []
        assert result.most_common_count == 5

    def test_all_different_params(self):
        segments = [
            WFOSegment(fold_index=i, tune_id=str(uuid4()), status="completed",
                       best_params={"a": i})
            for i in range(4)
        ]
        result = _compute_stability(segments)
        assert result is not None
        assert result.unique_param_sets == 4
        assert result.consistency == 0.25
        assert len(result.param_changes) == 3

    def test_mixed_params(self):
        params_a = {"x": 10}
        params_b = {"x": 20}
        segments = [
            WFOSegment(fold_index=0, tune_id=str(uuid4()), status="completed",
                       best_params=params_a),
            WFOSegment(fold_index=1, tune_id=str(uuid4()), status="completed",
                       best_params=params_a),
            WFOSegment(fold_index=2, tune_id=str(uuid4()), status="completed",
                       best_params=params_b),
            WFOSegment(fold_index=3, tune_id=str(uuid4()), status="completed",
                       best_params=params_a),
        ]
        result = _compute_stability(segments)
        assert result is not None
        assert result.unique_param_sets == 2
        assert result.consistency == 0.75
        assert result.most_common_params == params_a
        assert result.most_common_count == 3
        # Changes at fold 2 (a->b) and fold 3 (b->a)
        assert result.param_changes == [2, 3]

    def test_no_completed_folds(self):
        segments = [
            WFOSegment(fold_index=0, tune_id=str(uuid4()), status="failed"),
            WFOSegment(fold_index=1, tune_id=str(uuid4()), status="failed"),
        ]
        result = _compute_stability(segments)
        assert result is None

    def test_single_fold(self):
        result = _compute_stability([
            WFOSegment(fold_index=0, tune_id=str(uuid4()), status="completed",
                       best_params={"k": 5}),
        ])
        assert result is not None
        assert result.unique_param_sets == 1
        assert result.consistency == 1.0
        assert result.param_changes == []

    def test_skips_folds_without_params(self):
        segments = [
            WFOSegment(fold_index=0, tune_id=str(uuid4()), status="completed",
                       best_params={"a": 1}),
            WFOSegment(fold_index=1, tune_id=str(uuid4()), status="failed",
                       best_params=None),
            WFOSegment(fold_index=2, tune_id=str(uuid4()), status="completed",
                       best_params={"a": 1}),
        ]
        result = _compute_stability(segments)
        assert result is not None
        assert result.total_folds == 2  # Only folds with params


# =============================================================================
# Stitched equity
# =============================================================================


class TestStitchEquityCurves:
    """OOS equity curve stitching."""

    def test_single_fold(self):
        points = [
            EquityPoint(t="2024-01-01T00:00:00Z", equity=10000),
            EquityPoint(t="2024-01-02T00:00:00Z", equity=10500),
            EquityPoint(t="2024-01-03T00:00:00Z", equity=10200),
        ]
        result = _stitch_equity_curves([(0, points)], initial_equity=10000)
        assert len(result) == 3
        assert result[0].equity == 10000.0
        assert result[1].equity == 10500.0
        assert result[2].equity == 10200.0

    def test_two_folds_continuous(self):
        fold0 = [
            EquityPoint(t="2024-01-01T00:00:00Z", equity=10000),
            EquityPoint(t="2024-01-15T00:00:00Z", equity=11000),  # +10%
        ]
        fold1 = [
            EquityPoint(t="2024-02-01T00:00:00Z", equity=10000),
            EquityPoint(t="2024-02-15T00:00:00Z", equity=10500),  # +5%
        ]
        result = _stitch_equity_curves([(0, fold0), (1, fold1)], initial_equity=10000)

        # Fold 0: 10000 -> 11000 (+10%)
        assert result[0].equity == 10000.0
        assert result[1].equity == 11000.0

        # Fold 1: starts at 11000 (carry forward), +5% = 11550
        assert result[2].equity == 11000.0
        assert result[3].equity == 11550.0

    def test_empty_input(self):
        result = _stitch_equity_curves([])
        assert result == []

    def test_fold_with_zero_start_skipped(self):
        points = [
            EquityPoint(t="2024-01-01T00:00:00Z", equity=0),
            EquityPoint(t="2024-01-02T00:00:00Z", equity=100),
        ]
        result = _stitch_equity_curves([(0, points)])
        assert result == []

    def test_folds_ordered_by_index(self):
        """Even if folds are passed out of order, they're sorted by index."""
        fold1 = [
            EquityPoint(t="2024-02-01T00:00:00Z", equity=10000),
            EquityPoint(t="2024-02-15T00:00:00Z", equity=10500),
        ]
        fold0 = [
            EquityPoint(t="2024-01-01T00:00:00Z", equity=10000),
            EquityPoint(t="2024-01-15T00:00:00Z", equity=11000),
        ]
        result = _stitch_equity_curves([(1, fold1), (0, fold0)], initial_equity=10000)

        # Fold 0 comes first regardless of input order
        assert result[0].t == "2024-01-01T00:00:00Z"
        assert result[0].equity == 10000.0
        assert result[1].equity == 11000.0

    def test_drawdown_fold(self):
        fold0 = [
            EquityPoint(t="2024-01-01T00:00:00Z", equity=10000),
            EquityPoint(t="2024-01-15T00:00:00Z", equity=9000),  # -10%
        ]
        fold1 = [
            EquityPoint(t="2024-02-01T00:00:00Z", equity=10000),
            EquityPoint(t="2024-02-15T00:00:00Z", equity=11000),  # +10%
        ]
        result = _stitch_equity_curves([(0, fold0), (1, fold1)], initial_equity=10000)

        assert result[0].equity == 10000.0
        assert result[1].equity == 9000.0  # -10%
        assert result[2].equity == 9000.0  # Fold 1 starts where fold 0 ended
        assert result[3].equity == 9900.0  # 9000 * 1.1 = 9900


class TestWFOSegmentModel:
    """WFOSegment data model tests."""

    def test_segment_with_run_id(self):
        run_id = str(uuid4())
        s = WFOSegment(
            fold_index=0,
            tune_id=str(uuid4()),
            run_id=run_id,
            status="completed",
            train_start="2024-01-01",
            train_end="2024-03-01",
            test_start="2024-03-01",
            test_end="2024-04-01",
            best_params={"or_minutes": 30},
            best_score=1.5,
            oos_metrics={"sharpe": 1.2, "max_dd": -0.08},
        )
        assert s.run_id == run_id
        assert s.oos_metrics["sharpe"] == 1.2

    def test_segment_without_run_id(self):
        s = WFOSegment(
            fold_index=0,
            tune_id=str(uuid4()),
            status="failed",
        )
        assert s.run_id is None
        assert s.best_params is None
