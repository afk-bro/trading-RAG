"""Candidacy gate policy for KB trial ingestion.

Implements the gate function that determines whether a test variant
should be auto-promoted to 'candidate' status for KB ingestion.

This is Phase 2 of the trial ingestion design.
"""

from dataclasses import dataclass
from typing import Optional

from app.services.kb.types import RegimeSnapshot

# Known experiment types that can be processed
KNOWN_EXPERIMENT_TYPES = {"tune", "sweep", "ablation", "manual"}


@dataclass
class CandidacyDecision:
    """Result of candidacy gate evaluation.

    Attributes:
        eligible: Whether the trial passes all gates
        reason: The gate that determined the decision (pass or fail reason)
    """

    eligible: bool
    reason: str


@dataclass
class CandidacyConfig:
    """Configuration for candidacy gate thresholds.

    Attributes:
        require_regime: Whether OOS regime snapshot is required
        min_trades: Minimum number of OOS trades
        min_oos_bars: Future: OOS coverage sanity check
        max_drawdown: Maximum allowed drawdown fraction (0.25 = 25%)
        max_overfit_gap: Maximum sharpe_is - sharpe_oos gap
        min_sharpe: Minimum OOS Sharpe ratio
    """

    require_regime: bool = True
    min_trades: int = 5
    min_oos_bars: Optional[int] = None  # Future: OOS coverage sanity
    max_drawdown: float = 0.25
    max_overfit_gap: float = 0.30
    min_sharpe: float = 0.1


@dataclass
class VariantMetricsForCandidacy:
    """Subset of metrics needed for candidacy check.

    Attributes:
        n_trades_oos: Number of trades in OOS period
        max_dd_frac_oos: Maximum drawdown as positive fraction (0.10 = 10%)
        overfit_gap: sharpe_is - sharpe_oos, or None if unavailable
        sharpe_oos: OOS Sharpe ratio, or None if unavailable
    """

    n_trades_oos: int
    max_dd_frac_oos: float  # Positive fraction (0.10 = 10%)
    overfit_gap: Optional[float]  # sharpe_is - sharpe_oos
    sharpe_oos: Optional[float]


def is_candidate(
    metrics: VariantMetricsForCandidacy,
    regime_oos: Optional[RegimeSnapshot],
    experiment_type: str,
    config: Optional[CandidacyConfig] = None,
) -> CandidacyDecision:
    """Evaluate whether a test variant passes quality gates for auto-candidacy.

    Pure function - no DB access, no side effects.

    The gate checks are evaluated in order:
    1. Experiment type validation (unknown types excluded)
    2. Manual runs exclusion (never auto-candidate)
    3. Regime requirement (configurable)
    4. Minimum trades check
    5. Maximum drawdown check
    6. Overfit gap check (if available)
    7. Minimum Sharpe check (if available)

    Args:
        metrics: Performance metrics from OOS period
        regime_oos: Market regime snapshot from OOS period, or None
        experiment_type: Type of experiment (tune, sweep, ablation, manual)
        config: Gate thresholds configuration (uses defaults if None)

    Returns:
        CandidacyDecision with eligibility and reason
    """
    if config is None:
        config = CandidacyConfig()

    # Unknown types excluded
    if experiment_type not in KNOWN_EXPERIMENT_TYPES:
        return CandidacyDecision(False, "unknown_experiment_type")

    # Manual runs never auto-candidate
    if experiment_type == "manual":
        return CandidacyDecision(False, "manual_experiment_excluded")

    # Regime requirement (configurable)
    if config.require_regime and regime_oos is None:
        return CandidacyDecision(False, "missing_regime_oos")

    # Hard gates
    if metrics.n_trades_oos < config.min_trades:
        return CandidacyDecision(False, "insufficient_oos_trades")

    if metrics.max_dd_frac_oos > config.max_drawdown:
        return CandidacyDecision(False, "dd_too_high")

    if metrics.overfit_gap is not None and metrics.overfit_gap > config.max_overfit_gap:
        return CandidacyDecision(False, "overfit_too_high")

    if metrics.sharpe_oos is not None and metrics.sharpe_oos < config.min_sharpe:
        return CandidacyDecision(False, "sharpe_too_low")

    return CandidacyDecision(True, "passed_all_gates")
