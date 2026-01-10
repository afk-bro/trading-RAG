"""
Core types for the Trading Knowledge Base.

Contains RegimeSnapshot and TrialDoc dataclasses with all associated
metadata, conversion utilities, and warning computation.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal
from uuid import UUID

from app.services.kb.constants import (
    REGIME_SCHEMA_VERSION,
    TRIAL_DOC_SCHEMA_VERSION,
    DEFAULT_FEATURE_PARAMS,
    OVERFIT_GAP_MODERATE,
    OVERFIT_GAP_HIGH,
    LOW_TRADES_THRESHOLD,
    HIGH_DRAWDOWN_THRESHOLD,
)


# =============================================================================
# RegimeSnapshot
# =============================================================================


@dataclass
class RegimeSnapshot:
    """
    Market regime features computed from an OHLCV window.

    Designed for:
    - Embedding similarity (via text template)
    - Deterministic tagging (via compute_tags)
    - Audit trail (computed_at, computation_source)
    """

    # Versioning (reproducibility)
    schema_version: str = REGIME_SCHEMA_VERSION
    feature_params: dict = field(default_factory=lambda: DEFAULT_FEATURE_PARAMS.copy())
    timeframe: str | None = None

    # Audit
    computed_at: str | None = None  # ISO UTC
    computation_source: str | None = None  # "live" | "backfill" | "query"

    # Window metadata
    n_bars: int = 0
    effective_n_bars: int = 0  # After NaN drop from rolling
    ts_start: str = ""  # ISO UTC
    ts_end: str = ""

    # Volatility / dispersion
    atr_pct: float = 0.0  # ATR / close (fraction)
    std_pct: float = 0.0  # Rolling std / close
    bb_width_pct: float = 0.0  # (upper - lower) / middle
    range_pct: float = 0.0  # (high - low) / close over window

    # Trend (magnitude + direction)
    trend_strength: float = 0.0  # 0-1 normalized
    trend_dir: int = 0  # -1, 0, +1

    # Position in distribution
    zscore: float = 0.0  # Current price vs rolling mean/std
    rsi: float = 50.0  # 0-100

    # Return
    return_pct: float = 0.0  # Window return (fraction)
    drift_bps_per_bar: float = 0.0  # Avg return per bar in bps

    # Noise proxy
    efficiency_ratio: float = 0.5  # Kaufman ER: |net| / sum(|bars|), 0-1

    # Optional identifiers
    instrument: str | None = None
    source: str | None = None  # "is" | "oos" | "query"

    # Derived
    regime_tags: list[str] = field(default_factory=list)

    # Warnings from computation
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (NaN → null)."""
        return clean_nan_for_json(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> "RegimeSnapshot":
        """Create from dictionary (e.g., from JSON storage)."""
        if data is None:
            return None
        # Handle schema version compatibility
        version = data.get("schema_version", REGIME_SCHEMA_VERSION)
        if version != REGIME_SCHEMA_VERSION:
            # Future: handle migrations between versions
            pass
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# TrialDoc
# =============================================================================


@dataclass
class TrialDoc:
    """
    KB document for a single backtest trial.

    Contains all metadata needed for:
    - Vector storage (Qdrant payload)
    - Embedding generation (text template)
    - Retrieval filtering (performance floors, quality flags)
    - Aggregation (params, weights)
    """

    # Schema version
    schema_version: str = TRIAL_DOC_SCHEMA_VERSION

    # Identity
    doc_type: Literal["trial"] = "trial"
    tune_run_id: UUID | str = ""
    tune_id: UUID | str = ""

    # Dataset identity
    workspace_id: UUID | str = ""
    dataset_id: str | None = None
    instrument: str | None = None
    timeframe: str | None = None
    exchange: str | None = None

    # Strategy
    strategy_name: str = ""
    params: dict = field(default_factory=dict)

    # Performance (fractions 0-1, not percentages)
    sharpe_is: float | None = None
    sharpe_oos: float | None = None
    return_frac_is: float | None = None
    return_frac_oos: float | None = None
    max_dd_frac_is: float | None = None  # Positive value (0.10 = 10% drawdown)
    max_dd_frac_oos: float | None = None
    n_trades_is: int | None = None
    n_trades_oos: int | None = None

    # Overfit
    overfit_gap: float | None = None  # max(0, sharpe_is - sharpe_oos) or None

    # Regime (full snapshots preserved)
    regime_is: RegimeSnapshot | None = None
    regime_oos: RegimeSnapshot | None = None

    # Quality flags
    has_oos: bool = False
    is_valid: bool = True
    warnings: list[str] = field(default_factory=list)

    # Scoring
    objective_type: str = ""
    objective_score: float | None = None

    created_at: str = ""

    @property
    def regime_tags_is(self) -> list[str]:
        """Get regime tags from IS snapshot."""
        return self.regime_is.regime_tags if self.regime_is else []

    @property
    def regime_tags_oos(self) -> list[str]:
        """Get regime tags from OOS snapshot."""
        return self.regime_oos.regime_tags if self.regime_oos else []

    @property
    def regime_primary(self) -> Literal["is", "oos"]:
        """Which regime snapshot is primary (OOS preferred)."""
        return "oos" if self.regime_oos else "is"

    @property
    def regime_tags(self) -> list[str]:
        """Get tags from primary regime (OOS if available, else IS)."""
        return self.regime_tags_oos if self.regime_oos else self.regime_tags_is

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (NaN → null)."""
        data = asdict(self)
        # Convert UUIDs to strings
        for key in ["tune_run_id", "tune_id", "workspace_id"]:
            if isinstance(data.get(key), UUID):
                data[key] = str(data[key])
        return clean_nan_for_json(data)

    @classmethod
    def from_dict(cls, data: dict) -> "TrialDoc":
        """Create from dictionary."""
        if data is None:
            return None
        # Handle nested RegimeSnapshot
        for regime_key in ["regime_is", "regime_oos"]:
            if data.get(regime_key) and isinstance(data[regime_key], dict):
                data[regime_key] = RegimeSnapshot.from_dict(data[regime_key])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Helper Functions
# =============================================================================


def compute_trial_warnings(
    sharpe_is: float | None,
    sharpe_oos: float | None,
    overfit_gap: float | None,
    n_trades_oos: int | None,
    max_dd_frac_oos: float | None,
    is_valid: bool,
    regime_is: RegimeSnapshot | None = None,
    regime_oos: RegimeSnapshot | None = None,
    has_oos: bool = False,
) -> list[str]:
    """
    Compute warning flags for a trial document.

    Returns sorted list of warning strings.
    """
    warnings = []

    # Missing metrics
    if sharpe_is is None:
        warnings.append("missing_metrics")

    # Failed gates
    if not is_valid:
        warnings.append("failed_gates")

    # Overfit detection
    if overfit_gap is not None:
        if overfit_gap > OVERFIT_GAP_HIGH:
            warnings.append("high_overfit")
        elif overfit_gap > OVERFIT_GAP_MODERATE:
            warnings.append("moderate_overfit")

    # Low trade count
    if n_trades_oos is not None and n_trades_oos < LOW_TRADES_THRESHOLD:
        warnings.append("low_trades_oos")

    # High drawdown
    if max_dd_frac_oos is not None and max_dd_frac_oos > HIGH_DRAWDOWN_THRESHOLD:
        warnings.append("high_drawdown")

    # Missing regime data
    if regime_is is None:
        warnings.append("missing_regime_is")
    if has_oos and regime_oos is None:
        warnings.append("missing_regime_oos")

    return sorted(warnings)


def compute_overfit_gap(
    sharpe_is: float | None,
    sharpe_oos: float | None,
) -> float | None:
    """
    Compute overfit gap between IS and OOS Sharpe ratios.

    Returns max(0, sharpe_is - sharpe_oos) or None if data missing.
    """
    if sharpe_is is None or sharpe_oos is None:
        return None
    return max(0.0, sharpe_is - sharpe_oos)


def utc_now_iso() -> str:
    """Get current UTC time as ISO string."""
    from datetime import timezone as tz

    return datetime.now(tz.utc).isoformat().replace("+00:00", "Z")


def clean_nan_for_json(obj: any) -> any:
    """
    Recursively clean NaN/Inf values from an object for JSON serialization.

    NaN and Inf become None (null in JSON).

    Args:
        obj: Any object (dict, list, or scalar)

    Returns:
        Object with NaN/Inf replaced by None
    """
    import math

    if isinstance(obj, dict):
        return {k: clean_nan_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
