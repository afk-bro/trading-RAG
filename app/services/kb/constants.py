"""
Constants for the Trading Knowledge Base.

All thresholds, defaults, and timeouts for regime computation,
retrieval, aggregation, and confidence scoring.
"""

from typing import Final

# =============================================================================
# Regime Computation
# =============================================================================

# Window size for regime computation (number of bars)
REGIME_WINDOW_BARS: Final[int] = 200

# Minimum effective bars after NaN drop from rolling calculations
MIN_EFFECTIVE_BARS: Final[int] = 50

# Default feature computation parameters
DEFAULT_FEATURE_PARAMS: Final[dict] = {
    "atr_period": 14,
    "rsi_period": 14,
    "bb_period": 20,
    "bb_k": 2.0,
    "z_period": 20,
    "trend_lookback": 50,
}

# =============================================================================
# Tagging Thresholds
# =============================================================================

# Trend strength thresholds
TREND_STRONG_THRESHOLD: Final[float] = 0.6  # Above = trending
TREND_WEAK_THRESHOLD: Final[float] = 0.3  # Below = flat

# Volatility thresholds (ATR as fraction of close)
VOL_LOW_THRESHOLD: Final[float] = 0.005  # 0.5%
VOL_HIGH_THRESHOLD: Final[float] = 0.015  # 1.5%

# Regime type thresholds
BB_WIDTH_CHOPPY_THRESHOLD: Final[float] = 0.02  # 2%
ZSCORE_MEAN_REVERT_THRESHOLD: Final[float] = 1.0

# Efficiency ratio thresholds (Kaufman ER)
ER_NOISY_THRESHOLD: Final[float] = 0.3
ER_EFFICIENT_THRESHOLD: Final[float] = 0.7

# Oscillator extreme thresholds
ZSCORE_OVERSOLD: Final[float] = -1.5
ZSCORE_OVERBOUGHT: Final[float] = 1.5
RSI_OVERSOLD: Final[float] = 30.0
RSI_OVERBOUGHT: Final[float] = 70.0

# =============================================================================
# Retrieval Configuration
# =============================================================================

# Minimum candidates before trying relaxed filters
MIN_CANDIDATES_THRESHOLD: Final[int] = 10

# Default retrieval counts
DEFAULT_RETRIEVE_K: Final[int] = 100
DEFAULT_RERANK_KEEP: Final[int] = 30
DEFAULT_TOP_K: Final[int] = 15

# =============================================================================
# Performance Floors (defaults)
# =============================================================================

DEFAULT_MIN_OOS_SHARPE: Final[float] = 0.5
DEFAULT_MIN_TRADES: Final[int] = 20
DEFAULT_MAX_DRAWDOWN_FRAC: Final[float] = 0.20  # 20%
DEFAULT_MAX_OVERFIT_GAP: Final[float] = 0.50

# =============================================================================
# Timeouts
# =============================================================================

TIMEOUT_EMBED_S: Final[float] = 5.0
TIMEOUT_QDRANT_S: Final[float] = 10.0
TIMEOUT_BACKTEST_S: Final[float] = 60.0

# =============================================================================
# Weight Computation
# =============================================================================

# Penalty multiplier for relaxed filter candidates
RELAXED_WEIGHT_PENALTY: Final[float] = 0.25

# Reference trade count for weight normalization
TRADES_REF: Final[int] = 50

# Default sharpe scale for weight computation
DEFAULT_SHARPE_SCALE: Final[float] = 1.0

# =============================================================================
# Overfit Detection
# =============================================================================

OVERFIT_GAP_MODERATE: Final[float] = 0.3
OVERFIT_GAP_HIGH: Final[float] = 0.5

# =============================================================================
# Data Quality
# =============================================================================

# Low trade count warning threshold
LOW_TRADES_THRESHOLD: Final[int] = 10

# High drawdown warning threshold
HIGH_DRAWDOWN_THRESHOLD: Final[float] = 0.15  # 15%

# Outlier detection (percentage change)
OUTLIER_PCT_CHANGE_THRESHOLD: Final[float] = 0.5  # 50%

# =============================================================================
# Feature Scales (for distance normalization)
# =============================================================================

FEATURE_SCALES: Final[dict[str, float]] = {
    "atr_pct": 2.0,  # 0-2% typical range
    "bb_width_pct": 5.0,  # 0-5% typical
    "trend_strength": 1.0,  # Already 0-1
    "efficiency_ratio": 1.0,  # Already 0-1
    "zscore": 3.0,  # -3 to +3 typical
    "rsi": 100.0,  # 0-100
    "std_pct": 2.0,  # Similar to ATR
    "range_pct": 5.0,  # Similar to BB
    "return_pct": 1.0,  # -100% to +100% but usually smaller
    "drift_bps_per_bar": 10.0,  # -10 to +10 bps typical
}

# =============================================================================
# Qdrant Configuration
# =============================================================================

KB_TRIALS_COLLECTION_NAME: Final[str] = "trading_kb_trials"
KB_TRIALS_DOC_TYPE: Final[str] = "trial"

# =============================================================================
# Schema Versioning
# =============================================================================

REGIME_SCHEMA_VERSION: Final[str] = "regime_v1_1"
TRIAL_DOC_SCHEMA_VERSION: Final[str] = "trial_v1"

# =============================================================================
# Tag Exclusive Families (v1.1)
# =============================================================================

# Tags that are mutually exclusive (at most one can be assigned)
# Evaluator applies priority order within families; first passing tag wins.
# Order: uptrend > downtrend > trending > flat
EXCLUSIVE_FAMILIES: Final[dict[str, list[str]]] = {
    "trend": ["uptrend", "downtrend", "trending", "flat"],
}
