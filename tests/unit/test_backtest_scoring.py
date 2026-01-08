"""Unit tests for backtest scoring functions."""

import math
import pytest

from app.services.backtest.scoring import compute_score, rank_trials


class TestComputeScore:
    """Tests for compute_score function."""

    def test_sharpe_with_valid_data(self):
        """Valid sharpe score should be returned."""
        summary = {"sharpe": 1.5, "trades": 10}
        score = compute_score(summary, objective="sharpe", min_trades=5)
        assert score == 1.5

    def test_sharpe_with_insufficient_trades(self):
        """Insufficient trades should return None (skipped)."""
        summary = {"sharpe": 2.0, "trades": 3}
        score = compute_score(summary, objective="sharpe", min_trades=5)
        assert score is None

    def test_sharpe_with_missing_metric(self):
        """Missing sharpe should return None."""
        summary = {"trades": 10}
        score = compute_score(summary, objective="sharpe", min_trades=5)
        assert score is None

    def test_sharpe_with_nan(self):
        """NaN sharpe should return None."""
        summary = {"sharpe": float("nan"), "trades": 10}
        score = compute_score(summary, objective="sharpe", min_trades=5)
        assert score is None

    def test_sharpe_with_inf(self):
        """Inf sharpe should return None."""
        summary = {"sharpe": float("inf"), "trades": 10}
        score = compute_score(summary, objective="sharpe", min_trades=5)
        assert score is None

    def test_return_with_valid_data(self):
        """Valid return score should be returned."""
        summary = {"return_pct": 25.5, "trades": 10}
        score = compute_score(summary, objective="return", min_trades=5)
        assert score == 25.5

    def test_return_with_negative_value(self):
        """Negative return should still be valid."""
        summary = {"return_pct": -10.0, "trades": 10}
        score = compute_score(summary, objective="return", min_trades=5)
        assert score == -10.0

    def test_calmar_with_valid_data(self):
        """Valid calmar score should be return/abs(drawdown)."""
        summary = {"return_pct": 20.0, "max_drawdown_pct": -10.0, "trades": 10}
        score = compute_score(summary, objective="calmar", min_trades=5)
        assert score == 2.0  # 20 / 10

    def test_calmar_with_zero_drawdown(self):
        """Zero drawdown with positive return should give bonus."""
        summary = {"return_pct": 15.0, "max_drawdown_pct": 0.0, "trades": 10}
        score = compute_score(summary, objective="calmar", min_trades=5)
        assert score == 150.0  # 15 * 10 bonus

    def test_calmar_with_zero_drawdown_negative_return(self):
        """Zero drawdown with negative return should return that value."""
        summary = {"return_pct": -5.0, "max_drawdown_pct": 0.0, "trades": 10}
        score = compute_score(summary, objective="calmar", min_trades=5)
        assert score == -5.0

    def test_unknown_objective(self):
        """Unknown objective should return None."""
        summary = {"sharpe": 1.5, "trades": 10}
        score = compute_score(summary, objective="unknown", min_trades=5)
        assert score is None

    def test_zero_trades(self):
        """Zero trades should return None."""
        summary = {"sharpe": 5.0, "trades": 0}
        score = compute_score(summary, objective="sharpe", min_trades=5)
        assert score is None

    def test_exact_min_trades(self):
        """Exactly min_trades should be valid."""
        summary = {"sharpe": 1.0, "trades": 5}
        score = compute_score(summary, objective="sharpe", min_trades=5)
        assert score == 1.0


class TestRankTrials:
    """Tests for rank_trials function."""

    def test_ranks_by_score_descending(self):
        """Trials should be ranked by score descending."""
        trials = [
            {"params": {"a": 1}, "summary": {"sharpe": 1.0, "trades": 10}},
            {"params": {"a": 2}, "summary": {"sharpe": 3.0, "trades": 10}},
            {"params": {"a": 3}, "summary": {"sharpe": 2.0, "trades": 10}},
        ]
        ranked = rank_trials(trials, objective="sharpe", min_trades=5)

        assert len(ranked) == 3
        assert ranked[0]["params"]["a"] == 2  # highest sharpe
        assert ranked[0]["rank"] == 1
        assert ranked[0]["score"] == 3.0
        assert ranked[1]["params"]["a"] == 3
        assert ranked[1]["rank"] == 2
        assert ranked[2]["params"]["a"] == 1
        assert ranked[2]["rank"] == 3

    def test_filters_invalid_trials(self):
        """Trials with insufficient trades should be filtered."""
        trials = [
            {"params": {"a": 1}, "summary": {"sharpe": 5.0, "trades": 2}},  # filtered
            {"params": {"a": 2}, "summary": {"sharpe": 1.0, "trades": 10}},
            {"params": {"a": 3}, "summary": {"sharpe": 2.0, "trades": 10}},
        ]
        ranked = rank_trials(trials, objective="sharpe", min_trades=5)

        assert len(ranked) == 2
        assert ranked[0]["params"]["a"] == 3  # highest valid
        assert ranked[1]["params"]["a"] == 2

    def test_top_n_limit(self):
        """Should return only top N results."""
        trials = [
            {"params": {"a": i}, "summary": {"sharpe": float(i), "trades": 10}}
            for i in range(20)
        ]
        ranked = rank_trials(trials, top_n=5)

        assert len(ranked) == 5
        assert ranked[0]["params"]["a"] == 19  # highest
        assert ranked[4]["params"]["a"] == 15

    def test_empty_input(self):
        """Empty input should return empty list."""
        ranked = rank_trials([])
        assert ranked == []

    def test_all_filtered(self):
        """All filtered trials should return empty list."""
        trials = [
            {"params": {"a": 1}, "summary": {"sharpe": 5.0, "trades": 1}},
            {"params": {"a": 2}, "summary": {"sharpe": 3.0, "trades": 2}},
        ]
        ranked = rank_trials(trials, min_trades=5)
        assert ranked == []

    def test_preserves_original_fields(self):
        """Original trial fields should be preserved."""
        trials = [
            {"params": {"a": 1}, "summary": {"sharpe": 1.0, "trades": 10}, "run_id": "abc"},
        ]
        ranked = rank_trials(trials)

        assert ranked[0]["run_id"] == "abc"
        assert ranked[0]["params"] == {"a": 1}
        assert ranked[0]["summary"] == {"sharpe": 1.0, "trades": 10}
