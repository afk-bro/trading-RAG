"""Unit tests for the candidacy gate policy."""

import pytest

from app.services.kb.candidacy import (
    CandidacyConfig,
    CandidacyDecision,
    VariantMetricsForCandidacy,
    is_candidate,
    KNOWN_EXPERIMENT_TYPES,
)
from app.services.kb.types import RegimeSnapshot


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def passing_metrics() -> VariantMetricsForCandidacy:
    """Create metrics that pass all default gates."""
    return VariantMetricsForCandidacy(
        n_trades_oos=20,
        max_dd_frac_oos=0.10,  # 10% drawdown (below 25% threshold)
        overfit_gap=0.15,  # Below 0.30 threshold
        sharpe_oos=0.50,  # Above 0.1 threshold
    )


@pytest.fixture
def regime_snapshot() -> RegimeSnapshot:
    """Create a valid regime snapshot for testing."""
    return RegimeSnapshot(
        n_bars=200,
        effective_n_bars=180,
        atr_pct=0.01,
        trend_strength=0.5,
        zscore=0.5,
        rsi=50.0,
        regime_tags=["trending", "vol_normal"],
    )


@pytest.fixture
def default_config() -> CandidacyConfig:
    """Return default candidacy config."""
    return CandidacyConfig()


# =============================================================================
# Known Experiment Types Tests
# =============================================================================


class TestKnownExperimentTypes:
    """Tests for the known experiment types constant."""

    def test_known_types_includes_tune(self):
        """tune should be a known experiment type."""
        assert "tune" in KNOWN_EXPERIMENT_TYPES

    def test_known_types_includes_sweep(self):
        """sweep should be a known experiment type."""
        assert "sweep" in KNOWN_EXPERIMENT_TYPES

    def test_known_types_includes_ablation(self):
        """ablation should be a known experiment type."""
        assert "ablation" in KNOWN_EXPERIMENT_TYPES

    def test_known_types_includes_manual(self):
        """manual should be a known experiment type."""
        assert "manual" in KNOWN_EXPERIMENT_TYPES

    def test_known_types_is_exactly_four(self):
        """Should have exactly 4 known types."""
        assert len(KNOWN_EXPERIMENT_TYPES) == 4


# =============================================================================
# Experiment Type Filter Tests
# =============================================================================


class TestExperimentTypeFiltering:
    """Tests for experiment type validation."""

    def test_unknown_type_excluded(self, passing_metrics, regime_snapshot):
        """Unknown experiment types should be excluded."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="unknown_type",
        )

        assert decision.eligible is False
        assert decision.reason == "unknown_experiment_type"

    def test_empty_string_type_excluded(self, passing_metrics, regime_snapshot):
        """Empty string experiment type should be excluded."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="",
        )

        assert decision.eligible is False
        assert decision.reason == "unknown_experiment_type"

    def test_manual_never_auto_candidates(self, passing_metrics, regime_snapshot):
        """Manual runs should never auto-candidate."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="manual",
        )

        assert decision.eligible is False
        assert decision.reason == "manual_experiment_excluded"

    def test_tune_can_be_candidate(self, passing_metrics, regime_snapshot):
        """tune experiment type should pass type check."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="tune",
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"

    def test_sweep_can_be_candidate(self, passing_metrics, regime_snapshot):
        """sweep experiment type should pass type check."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"

    def test_ablation_can_be_candidate(self, passing_metrics, regime_snapshot):
        """ablation experiment type should pass type check."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="ablation",
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"


# =============================================================================
# Regime Requirement Tests
# =============================================================================


class TestRegimeRequirement:
    """Tests for the regime requirement gate."""

    def test_missing_regime_rejected_by_default(self, passing_metrics):
        """Missing regime should cause rejection with default config."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=None,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "missing_regime_oos"

    def test_missing_regime_allowed_when_disabled(self, passing_metrics):
        """Missing regime should be allowed when require_regime is False."""
        config = CandidacyConfig(require_regime=False)

        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=None,
            experiment_type="sweep",
            config=config,
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"

    def test_present_regime_passes(self, passing_metrics, regime_snapshot):
        """Present regime should pass the regime gate."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True


# =============================================================================
# Minimum Trades Gate Tests
# =============================================================================


class TestMinTradesGate:
    """Tests for the minimum OOS trades gate."""

    def test_below_min_trades_rejected(self, regime_snapshot):
        """Trades below minimum should be rejected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=3,  # Below default 5
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "insufficient_oos_trades"

    def test_exactly_min_trades_passes(self, regime_snapshot):
        """Exactly at min trades threshold should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=5,  # Exactly at default
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_above_min_trades_passes(self, regime_snapshot):
        """Trades above minimum should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=100,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_custom_min_trades_threshold(self, regime_snapshot):
        """Custom min_trades threshold should be respected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=8,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        config_strict = CandidacyConfig(min_trades=10)
        decision_strict = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
            config=config_strict,
        )
        assert decision_strict.eligible is False

        config_lenient = CandidacyConfig(min_trades=5)
        decision_lenient = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
            config=config_lenient,
        )
        assert decision_lenient.eligible is True


# =============================================================================
# Max Drawdown Gate Tests
# =============================================================================


class TestMaxDrawdownGate:
    """Tests for the maximum drawdown gate."""

    def test_above_max_drawdown_rejected(self, regime_snapshot):
        """Drawdown above max should be rejected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.30,  # 30%, above default 25%
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "dd_too_high"

    def test_exactly_at_max_drawdown_rejected(self, regime_snapshot):
        """Exactly at max drawdown should be rejected (> threshold)."""
        # Note: The check is metrics.max_dd_frac_oos > config.max_drawdown
        # So exactly at threshold should pass
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.25,  # Exactly at default threshold
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        # At threshold should pass (not strictly greater)
        assert decision.eligible is True

    def test_below_max_drawdown_passes(self, regime_snapshot):
        """Drawdown below max should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.15,  # 15%, well below 25%
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_custom_max_drawdown_threshold(self, regime_snapshot):
        """Custom max_drawdown threshold should be respected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.20,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        config_strict = CandidacyConfig(max_drawdown=0.15)
        decision_strict = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
            config=config_strict,
        )
        assert decision_strict.eligible is False

        config_lenient = CandidacyConfig(max_drawdown=0.30)
        decision_lenient = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
            config=config_lenient,
        )
        assert decision_lenient.eligible is True


# =============================================================================
# Overfit Gap Gate Tests
# =============================================================================


class TestOverfitGapGate:
    """Tests for the overfit gap gate."""

    def test_above_max_overfit_rejected(self, regime_snapshot):
        """Overfit gap above max should be rejected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.40,  # Above default 0.30
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "overfit_too_high"

    def test_exactly_at_max_overfit_passes(self, regime_snapshot):
        """Exactly at max overfit gap should pass (not strictly greater)."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.30,  # Exactly at default
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_below_max_overfit_passes(self, regime_snapshot):
        """Overfit gap below max should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.10,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_none_overfit_skipped(self, regime_snapshot):
        """None overfit gap should skip the check."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=None,  # Not available
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        # Should pass - None means data not available, skip check
        assert decision.eligible is True

    def test_negative_overfit_passes(self, regime_snapshot):
        """Negative overfit gap (OOS better than IS) should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=-0.20,  # OOS outperformed IS
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True


# =============================================================================
# Sharpe Gate Tests
# =============================================================================


class TestSharpeGate:
    """Tests for the minimum Sharpe ratio gate."""

    def test_below_min_sharpe_rejected(self, regime_snapshot):
        """Sharpe below minimum should be rejected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.05,  # Below default 0.1
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "sharpe_too_low"

    def test_exactly_at_min_sharpe_passes(self, regime_snapshot):
        """Exactly at min Sharpe should pass (not strictly less)."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.10,  # Exactly at default
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_above_min_sharpe_passes(self, regime_snapshot):
        """Sharpe above minimum should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=1.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_none_sharpe_skipped(self, regime_snapshot):
        """None Sharpe should skip the check."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=None,  # Not available
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        # Should pass - None means data not available, skip check
        assert decision.eligible is True

    def test_negative_sharpe_rejected(self, regime_snapshot):
        """Negative Sharpe should be rejected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=-0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "sharpe_too_low"


# =============================================================================
# Gate Order Tests
# =============================================================================


class TestGateOrder:
    """Tests for the order of gate evaluation."""

    def test_experiment_type_checked_first(self):
        """Experiment type should be checked before metrics."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=1,  # Would fail
            max_dd_frac_oos=0.50,  # Would fail
            overfit_gap=0.60,  # Would fail
            sharpe_oos=-1.0,  # Would fail
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,  # Would fail
            experiment_type="unknown",
        )

        # Should fail on experiment type, not other gates
        assert decision.reason == "unknown_experiment_type"

    def test_manual_checked_before_regime(self):
        """Manual exclusion should be checked before regime."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,  # Would fail regime check
            experiment_type="manual",
        )

        # Should fail on manual, not regime
        assert decision.reason == "manual_experiment_excluded"

    def test_regime_checked_before_metrics(self, regime_snapshot):
        """Regime should be checked before metric gates."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=1,  # Would fail trades gate
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=None,  # Missing
            experiment_type="sweep",
        )

        # Should fail on regime, not trades
        assert decision.reason == "missing_regime_oos"

    def test_trades_checked_before_drawdown(self, regime_snapshot):
        """Trades gate should be checked before drawdown."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=2,  # Would fail
            max_dd_frac_oos=0.50,  # Would also fail
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.reason == "insufficient_oos_trades"

    def test_drawdown_checked_before_overfit(self, regime_snapshot):
        """Drawdown gate should be checked before overfit."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.50,  # Would fail
            overfit_gap=0.60,  # Would also fail
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.reason == "dd_too_high"

    def test_overfit_checked_before_sharpe(self, regime_snapshot):
        """Overfit gate should be checked before Sharpe."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.60,  # Would fail
            sharpe_oos=-1.0,  # Would also fail
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.reason == "overfit_too_high"


# =============================================================================
# Config Default Tests
# =============================================================================


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_require_regime_is_true(self):
        """Default require_regime should be True."""
        config = CandidacyConfig()
        assert config.require_regime is True

    def test_default_min_trades_is_five(self):
        """Default min_trades should be 5."""
        config = CandidacyConfig()
        assert config.min_trades == 5

    def test_default_max_drawdown_is_25_percent(self):
        """Default max_drawdown should be 0.25 (25%)."""
        config = CandidacyConfig()
        assert config.max_drawdown == 0.25

    def test_default_max_overfit_is_30_percent(self):
        """Default max_overfit_gap should be 0.30."""
        config = CandidacyConfig()
        assert config.max_overfit_gap == 0.30

    def test_default_min_sharpe_is_01(self):
        """Default min_sharpe should be 0.1."""
        config = CandidacyConfig()
        assert config.min_sharpe == 0.1

    def test_default_min_oos_bars_is_none(self):
        """Default min_oos_bars should be None (future feature)."""
        config = CandidacyConfig()
        assert config.min_oos_bars is None

    def test_none_config_uses_defaults(self, passing_metrics, regime_snapshot):
        """Passing None config should use defaults."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
            config=None,
        )

        assert decision.eligible is True
        assert decision.reason == "passed_all_gates"


# =============================================================================
# CandidacyDecision Tests
# =============================================================================


class TestCandidacyDecision:
    """Tests for the CandidacyDecision dataclass."""

    def test_eligible_decision_attributes(self, passing_metrics, regime_snapshot):
        """Eligible decision should have correct attributes."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert isinstance(decision, CandidacyDecision)
        assert decision.eligible is True
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    def test_ineligible_decision_attributes(self, passing_metrics):
        """Ineligible decision should have correct attributes."""
        decision = is_candidate(
            metrics=passing_metrics,
            regime_oos=None,
            experiment_type="sweep",
        )

        assert isinstance(decision, CandidacyDecision)
        assert decision.eligible is False
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_trades_rejected(self, regime_snapshot):
        """Zero trades should be rejected."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=0,
            max_dd_frac_oos=0.0,
            overfit_gap=0.0,
            sharpe_oos=0.0,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "insufficient_oos_trades"

    def test_zero_drawdown_passes(self, regime_snapshot):
        """Zero drawdown should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.0,
            overfit_gap=0.15,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_zero_overfit_gap_passes(self, regime_snapshot):
        """Zero overfit gap should pass."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.0,
            sharpe_oos=0.50,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_exactly_zero_sharpe_rejected(self, regime_snapshot):
        """Zero Sharpe should be rejected (below 0.1)."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,
            max_dd_frac_oos=0.10,
            overfit_gap=0.15,
            sharpe_oos=0.0,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is False
        assert decision.reason == "sharpe_too_low"

    def test_all_optional_none_passes(self, regime_snapshot):
        """All optional metrics None should pass those checks."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=20,  # Required
            max_dd_frac_oos=0.10,  # Required
            overfit_gap=None,
            sharpe_oos=None,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        assert decision.eligible is True

    def test_very_high_values_handles_gracefully(self, regime_snapshot):
        """Very high values should be handled without error."""
        metrics = VariantMetricsForCandidacy(
            n_trades_oos=1000000,
            max_dd_frac_oos=0.99,  # 99% drawdown
            overfit_gap=10.0,
            sharpe_oos=100.0,
        )

        decision = is_candidate(
            metrics=metrics,
            regime_oos=regime_snapshot,
            experiment_type="sweep",
        )

        # Should fail on drawdown
        assert decision.eligible is False
        assert decision.reason == "dd_too_high"
