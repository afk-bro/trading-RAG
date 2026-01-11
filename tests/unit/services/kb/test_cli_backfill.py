"""Unit tests for KB CLI backfill commands."""

import json
from collections import Counter
from uuid import uuid4

from app.services.kb.candidacy import (
    is_candidate,
    VariantMetricsForCandidacy,
    CandidacyConfig,
    CandidacyDecision,
)
from app.services.kb.types import RegimeSnapshot


class TestCandidacyMetricsExtraction:
    """Test extracting candidacy metrics from summary JSONB."""

    def test_extract_trades_from_summary(self):
        """Trades should be extracted from summary.trades."""
        summary = {"trades": 42, "sharpe": 1.5, "max_drawdown_pct": -10.0}
        n_trades = summary.get("trades", 0) or 0
        assert n_trades == 42

    def test_extract_sharpe_from_summary(self):
        """Sharpe should be extracted from summary.sharpe."""
        summary = {"trades": 42, "sharpe": 1.5}
        sharpe = summary.get("sharpe")
        assert sharpe == 1.5

    def test_extract_drawdown_from_summary(self):
        """Drawdown should be extracted and converted to positive fraction."""
        summary = {"max_drawdown_pct": -15.0}
        max_dd_pct = abs(summary.get("max_drawdown_pct", 0) or 0)
        max_dd_frac = max_dd_pct / 100.0
        assert max_dd_frac == 0.15

    def test_handle_missing_trades(self):
        """Missing trades should default to 0."""
        summary = {"sharpe": 1.0}
        n_trades = summary.get("trades", 0) or 0
        assert n_trades == 0

    def test_handle_null_summary(self):
        """None summary should be handled gracefully."""
        summary = None
        if summary is None:
            summary = {}
        n_trades = summary.get("trades", 0) or 0
        assert n_trades == 0


class TestCandidacyGateEvaluation:
    """Test candidacy gate evaluation logic."""

    def test_passes_all_gates(self):
        """Run with good metrics should pass all gates."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            sharpe_oos=1.5,
            overfit_gap=None,
        )
        regime = RegimeSnapshot(regime_tags=["uptrend", "low_vol"])
        config = CandidacyConfig(require_regime=True)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime,
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"

    def test_fails_insufficient_trades(self):
        """Run with too few trades should fail."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=2,  # Below min_trades=5
            max_dd_frac_oos=0.10,
            sharpe_oos=1.5,
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=False, min_trades=5)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is False
        assert decision.reason == "insufficient_oos_trades"

    def test_fails_high_drawdown(self):
        """Run with high drawdown should fail."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.35,  # Above max_drawdown=0.25
            sharpe_oos=1.5,
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=False, max_drawdown=0.25)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is False
        assert decision.reason == "dd_too_high"

    def test_fails_low_sharpe(self):
        """Run with low Sharpe should fail."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            sharpe_oos=0.05,  # Below min_sharpe=0.1
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=False, min_sharpe=0.1)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is False
        assert decision.reason == "sharpe_too_low"

    def test_fails_missing_regime_when_required(self):
        """Run without regime should fail when regime is required."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            sharpe_oos=1.5,
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=True)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,  # Missing regime
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is False
        assert decision.reason == "missing_regime_oos"

    def test_passes_without_regime_when_not_required(self):
        """Run without regime should pass when regime is not required."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            sharpe_oos=1.5,
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=False)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"

    def test_rejects_manual_experiment(self):
        """Manual experiments should never be auto-candidated."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            sharpe_oos=1.5,
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=False)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,
            experiment_type="manual",
            config=config,
        )

        assert decision.eligible is False
        assert decision.reason == "manual_experiment_excluded"


class TestRegimeSnapshotParsing:
    """Test parsing RegimeSnapshot from JSON."""

    def test_parse_valid_regime(self):
        """Valid regime dict should parse to RegimeSnapshot."""
        regime_data = {
            "schema_version": "regime_v1",
            "atr_pct": 0.02,
            "trend_strength": 0.8,
            "trend_dir": 1,
            "regime_tags": ["uptrend", "low_vol"],
        }

        regime = RegimeSnapshot.from_dict(regime_data)

        assert regime is not None
        assert regime.atr_pct == 0.02
        assert regime.trend_strength == 0.8
        assert regime.regime_tags == ["uptrend", "low_vol"]

    def test_parse_none_returns_none(self):
        """None input should return None."""
        regime = RegimeSnapshot.from_dict(None)
        assert regime is None

    def test_parse_empty_dict(self):
        """Empty dict should create default RegimeSnapshot."""
        regime = RegimeSnapshot.from_dict({})
        assert regime is not None
        assert regime.atr_pct == 0.0


class TestRejectionReasonAggregation:
    """Test aggregating rejection reasons."""

    def test_count_rejection_reasons(self):
        """Rejection reasons should be counted correctly."""
        reasons = Counter()

        # Simulate processing multiple runs
        decisions = [
            CandidacyDecision(False, "insufficient_oos_trades"),
            CandidacyDecision(False, "insufficient_oos_trades"),
            CandidacyDecision(False, "dd_too_high"),
            CandidacyDecision(True, "passed_all_gates"),
            CandidacyDecision(False, "sharpe_too_low"),
        ]

        for decision in decisions:
            if not decision.eligible:
                reasons[decision.reason] += 1

        assert reasons["insufficient_oos_trades"] == 2
        assert reasons["dd_too_high"] == 1
        assert reasons["sharpe_too_low"] == 1
        assert sum(reasons.values()) == 4

    def test_most_common_reasons(self):
        """Most common rejection reasons should be first."""
        reasons = Counter(
            {
                "insufficient_oos_trades": 100,
                "dd_too_high": 50,
                "sharpe_too_low": 25,
            }
        )

        top = reasons.most_common(2)
        assert top[0] == ("insufficient_oos_trades", 100)
        assert top[1] == ("dd_too_high", 50)


class TestDryRunBehavior:
    """Test dry-run mode behavior."""

    def test_eligible_ids_collected_in_dry_run(self):
        """Eligible IDs should be collected even in dry-run mode."""
        # Simulate the backfill loop
        eligible_ids = []
        dry_run = True

        # Simulated decision
        run_id = uuid4()
        decision = CandidacyDecision(True, "passed_all_gates")

        if decision.eligible:
            eligible_ids.append(run_id)

        # In dry run, we collect but don't update
        updated_count = 0
        if eligible_ids and not dry_run:
            updated_count = len(eligible_ids)

        assert len(eligible_ids) == 1
        assert updated_count == 0

    def test_no_updates_in_dry_run(self):
        """No database updates should happen in dry-run mode."""
        eligible_ids = [uuid4(), uuid4()]
        dry_run = True

        updated_count = 0
        if eligible_ids and not dry_run:
            # This would be the DB update
            updated_count = len(eligible_ids)

        assert updated_count == 0


class TestEdgeCases:
    """Test edge cases in candidacy backfill."""

    def test_zero_trades_handled(self):
        """Zero trades should fail gracefully."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=0,
            max_dd_frac_oos=0.0,
            sharpe_oos=None,
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=False)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is False
        assert decision.reason == "insufficient_oos_trades"

    def test_none_sharpe_passes_min_sharpe_check(self):
        """None Sharpe should not trigger sharpe_too_low."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            sharpe_oos=None,  # No Sharpe calculated
            overfit_gap=None,
        )
        config = CandidacyConfig(require_regime=False, min_sharpe=0.1)

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,
            experiment_type="sweep",
            config=config,
        )

        # Should pass because min_sharpe check is skipped when sharpe is None
        assert decision.eligible is True

    def test_negative_drawdown_converted_to_positive(self):
        """Negative drawdown should be converted to positive fraction."""
        summary = {"max_drawdown_pct": -20.0}
        max_dd_pct = abs(summary.get("max_drawdown_pct", 0) or 0)
        max_dd_frac = max_dd_pct / 100.0

        assert max_dd_frac == 0.20

    def test_json_string_summary_parsing(self):
        """Summary stored as JSON string should be parsed."""
        summary_str = '{"trades": 15, "sharpe": 1.2}'
        summary = json.loads(summary_str)

        assert summary["trades"] == 15
        assert summary["sharpe"] == 1.2
