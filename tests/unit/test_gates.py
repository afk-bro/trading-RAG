"""Tests for gate evaluation and winner selection."""

import pytest
import os
from unittest.mock import patch

from app.services.backtest.tuner import (
    evaluate_gates,
    GATE_MAX_DD_PCT,
    GATE_MIN_TRADES,
)


class TestEvaluateGates:
    """Tests for the evaluate_gates function."""

    def test_passes_when_all_gates_pass(self):
        """Gates should pass when metrics meet all thresholds."""
        metrics = {
            "return_pct": 15.0,
            "sharpe": 1.5,
            "max_drawdown_pct": -10.0,  # Better than -20 threshold
            "win_rate": 0.55,
            "trades": 25,  # More than 10 threshold
            "profit_factor": 1.8,
        }

        passed, failures = evaluate_gates(metrics)

        assert passed is True
        assert failures == []

    def test_fails_on_excessive_drawdown(self):
        """Gates should fail when drawdown exceeds threshold."""
        metrics = {
            "max_drawdown_pct": -25.0,  # Worse than -20 threshold
            "trades": 25,
        }

        passed, failures = evaluate_gates(metrics)

        assert passed is False
        assert len(failures) == 1
        assert "gate:max_drawdown_pct" in failures[0]
        assert "-25.0" in failures[0]
        assert "-20.0" in failures[0]

    def test_fails_on_insufficient_trades(self):
        """Gates should fail when trades below minimum."""
        metrics = {
            "max_drawdown_pct": -10.0,
            "trades": 5,  # Less than 10 threshold
        }

        passed, failures = evaluate_gates(metrics)

        assert passed is False
        assert len(failures) == 1
        assert "gate:trades" in failures[0]
        assert "5 < 10" in failures[0]

    def test_fails_on_empty_metrics(self):
        """Gates should fail when metrics dict is empty (falsy)."""
        passed, failures = evaluate_gates({})

        assert passed is False
        # Empty dict is falsy, triggers early return
        assert failures == ["gate:missing_metrics"]

    def test_fails_on_missing_required_keys(self):
        """Gates should fail when required keys are missing but dict exists."""
        # Dict with other keys but missing dd and trades
        metrics = {"return_pct": 10.0, "sharpe": 1.5}

        passed, failures = evaluate_gates(metrics)

        assert passed is False
        assert "gate:max_drawdown_pct (missing)" in failures
        assert "gate:trades (missing)" in failures

    def test_fails_on_none_metrics(self):
        """Gates should fail when metrics is None."""
        passed, failures = evaluate_gates(None)

        assert passed is False
        assert failures == ["gate:missing_metrics"]

    def test_multiple_gate_failures_combined(self):
        """Multiple gate failures should all be reported."""
        metrics = {
            "max_drawdown_pct": -30.0,  # Fails
            "trades": 3,  # Fails
        }

        passed, failures = evaluate_gates(metrics)

        assert passed is False
        assert len(failures) == 2
        assert any("max_drawdown_pct" in f for f in failures)
        assert any("trades" in f for f in failures)

    def test_passes_at_exact_thresholds(self):
        """Gates should pass at exact threshold values."""
        metrics = {
            "max_drawdown_pct": -20.0,  # Exactly at threshold
            "trades": 10,  # Exactly at threshold
        }

        passed, failures = evaluate_gates(metrics)

        assert passed is True
        assert failures == []

    def test_passes_with_none_values_for_optional_metrics(self):
        """Gates should handle None values for metrics not checked by gates."""
        metrics = {
            "return_pct": None,
            "sharpe": None,
            "max_drawdown_pct": -15.0,
            "win_rate": None,
            "trades": 20,
            "profit_factor": None,
        }

        passed, failures = evaluate_gates(metrics)

        assert passed is True
        assert failures == []

    def test_failure_message_format(self):
        """Failure messages should match expected format for UI aggregation."""
        metrics = {
            "max_drawdown_pct": -27.4,
            "trades": 20,
        }

        passed, failures = evaluate_gates(metrics)

        assert passed is False
        # Format: "gate:<metric> (<value> < <threshold>)"
        assert failures[0] == "gate:max_drawdown_pct (-27.4 < -20.0)"


class TestGateConfiguration:
    """Tests for gate configuration via environment variables."""

    def test_default_thresholds(self):
        """Default thresholds should be reasonable values."""
        assert GATE_MAX_DD_PCT == 20
        assert GATE_MIN_TRADES == 10

    def test_env_override_max_dd(self):
        """TUNER_GATE_MAX_DD_PCT should override default."""
        # Test that the module reads from environment
        # The actual loading happens at import time, so we verify the pattern
        with patch.dict(os.environ, {"TUNER_GATE_MAX_DD_PCT": "30"}):
            # Re-import to pick up new value
            import importlib
            from app.services.backtest import tuner

            importlib.reload(tuner)

            assert tuner.GATE_MAX_DD_PCT == 30.0

            # Restore
            importlib.reload(tuner)

    def test_env_override_min_trades(self):
        """TUNER_GATE_MIN_TRADES should override default."""
        with patch.dict(os.environ, {"TUNER_GATE_MIN_TRADES": "20"}):
            import importlib
            from app.services.backtest import tuner

            importlib.reload(tuner)

            assert tuner.GATE_MIN_TRADES == 20

            # Restore
            importlib.reload(tuner)


class TestGateSkipReasonFormat:
    """Tests for skip_reason format with gates."""

    def test_single_gate_failure_format(self):
        """Single gate failure should produce clean skip_reason."""
        metrics = {"max_drawdown_pct": -25.0, "trades": 20}

        passed, failures = evaluate_gates(metrics)
        skip_reason = "; ".join(failures)

        assert skip_reason == "gate:max_drawdown_pct (-25.0 < -20.0)"

    def test_multiple_gate_failures_joined(self):
        """Multiple gate failures should be joined with semicolon."""
        metrics = {"max_drawdown_pct": -30.0, "trades": 5}

        passed, failures = evaluate_gates(metrics)
        skip_reason = "; ".join(failures)

        assert "gate:max_drawdown_pct" in skip_reason
        assert "gate:trades" in skip_reason
        assert "; " in skip_reason

    def test_skip_reason_distinguishable_from_min_trades(self):
        """Gate trades failure should be distinguishable from min_trades scoring skip."""
        # Gate failure (policy violation)
        gate_metrics = {"max_drawdown_pct": -10.0, "trades": 5}
        _, failures = evaluate_gates(gate_metrics)
        gate_skip = "; ".join(failures)

        # min_trades scoring skip format (from tuner)
        scoring_skip = "min_trades_not_met (5<10)"

        # Should be distinguishable
        assert "gate:" in gate_skip
        assert "gate:" not in scoring_skip


class TestGateIntegrationBehavior:
    """Tests documenting expected integration behavior."""

    def test_gate_failure_preserves_run_id(self):
        """Gate-failed trials should still have run_id for drill-through.

        The tuner should persist run_id even when gates fail, allowing
        operators to inspect the actual backtest results.
        """
        # This is a documentation test - actual behavior tested in integration
        pass

    def test_gate_failure_preserves_scores(self):
        """Gate-failed trials should still have scores for comparison.

        Operators may want to see what the scores would have been
        even for gate-failed trials.
        """
        # This is a documentation test
        pass

    def test_gate_failure_vs_runtime_failure_status(self):
        """Gate failures use 'skipped' status, runtime errors use 'failed'.

        This ontology allows UI to distinguish policy violations from errors:
        - 'skipped': Trial ran but violated policy (gates, min_trades)
        - 'failed': Trial could not complete (timeout, exception)
        """
        # This is a documentation test - verified by skip_reason prefix
        metrics_gate_fail = {"max_drawdown_pct": -30.0, "trades": 5}
        _, failures = evaluate_gates(metrics_gate_fail)

        # Gate failures have 'gate:' prefix
        assert all(f.startswith("gate:") for f in failures)


class TestWinnerSelectionWithGates:
    """Tests documenting winner selection behavior with gates."""

    def test_winner_from_gate_passing_trials_only(self):
        """Winner should be selected from gate-passing trials only.

        Gate-failed trials return None from run_trial(), so they're
        automatically excluded from valid_results and leaderboard.
        """
        # This is a documentation test - verified by code structure
        pass

    def test_warning_when_no_trials_pass_gates(self):
        """Warning should be generated when all trials fail gates.

        Format: "No trials passed gates (N skipped due to gate violations)"
        """
        # This is a documentation test - verified by tuner code
        pass

    def test_no_best_run_when_all_gates_fail(self):
        """best_run_id should be None when no trials pass gates.

        The tune completes but with null best_* fields.
        """
        # This is a documentation test
        pass
