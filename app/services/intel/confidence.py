"""Confidence computation for strategy intelligence.

Pure functions for computing regime classification and confidence scores.
No I/O or database access - just computation logic.

v1.5 Step 2A - Confidence v0.1 + Regime v0.1
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID

import numpy as np
import pandas as pd

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Default component weights for confidence score
DEFAULT_WEIGHTS = {
    "performance": 0.30,
    "drawdown": 0.25,
    "stability": 0.20,
    "data_freshness": 0.10,
    "regime_fit": 0.15,
}

# Algorithm version for inputs_hash (bump when computation changes)
ALGO_VERSION = "confidence_v0.1"

# Thresholds for regime classification
VOLATILITY_LOW_PERCENTILE = 33
VOLATILITY_HIGH_PERCENTILE = 67
TREND_STRENGTH_THRESHOLD = 0.4  # R-squared above this = trending


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ConfidenceContext:
    """
    Input context for confidence computation.

    Holds all data needed to compute confidence score.
    """

    # Required: strategy version info
    version_id: UUID
    as_of_ts: datetime

    # OHLCV data (for regime detection)
    ohlcv: Optional[pd.DataFrame] = None  # columns: open, high, low, close, volume

    # Backtest metrics (from most recent run for this version)
    backtest_metrics: Optional[dict] = None
    # Expected keys: sharpe, return_pct, max_drawdown_pct, trades, win_rate

    # WFO metrics (preferred over backtest if available)
    wfo_metrics: Optional[dict] = None
    # Expected keys: oos_sharpe, oos_return_pct, fold_variance, num_folds

    # Data freshness
    latest_candle_ts: Optional[datetime] = None

    # Regime fit context (strategy's known good/bad regimes)
    strategy_regime_profile: Optional[dict] = None
    # Expected: {"good_regimes": ["trend_low_vol"], "bad_regimes": ["range_high_vol"]}


@dataclass
class ConfidenceResult:
    """
    Output from confidence computation.
    """

    regime: str
    confidence_score: float
    confidence_components: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    explain: dict = field(default_factory=dict)
    inputs_hash: str = ""


# =============================================================================
# Regime Classification (OHLCV-only, simplified)
# =============================================================================


def compute_regime(ohlcv: Optional[pd.DataFrame]) -> tuple[str, dict]:
    """
    Compute simplified regime classification from OHLCV data.

    Returns one of:
    - trend_low_vol
    - trend_high_vol
    - range_low_vol
    - range_high_vol
    - unknown (if insufficient data)

    Args:
        ohlcv: DataFrame with OHLCV columns (at least close required)

    Returns:
        Tuple of (regime_string, features_dict)
    """
    if ohlcv is None or len(ohlcv) < 20:
        return "unknown", {
            "reason": "insufficient_data",
            "bars": len(ohlcv) if ohlcv is not None else 0,
        }

    # Normalize column names
    df = ohlcv.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    if "close" not in df.columns:
        return "unknown", {"reason": "missing_close_column"}

    close = df["close"]

    # Compute volatility (rolling std as % of price)
    returns = close.pct_change().dropna()
    if len(returns) < 10:
        return "unknown", {"reason": "insufficient_returns"}

    volatility = returns.rolling(window=14).std().iloc[-1]
    vol_percentile = _compute_percentile(
        volatility, returns.rolling(window=14).std().dropna()
    )

    # Compute trend strength (R-squared of linear regression)
    trend_strength = _compute_trend_strength(close, lookback=min(50, len(close)))

    # Classify
    is_trending = trend_strength >= TREND_STRENGTH_THRESHOLD
    is_low_vol = vol_percentile < VOLATILITY_LOW_PERCENTILE
    is_high_vol = vol_percentile > VOLATILITY_HIGH_PERCENTILE

    # Build regime string
    trend_label = "trend" if is_trending else "range"
    if is_low_vol:
        vol_label = "low_vol"
    elif is_high_vol:
        vol_label = "high_vol"
    else:
        vol_label = "mid_vol"

    regime = f"{trend_label}_{vol_label}"

    features = {
        "trend_strength": round(trend_strength, 4),
        "volatility": round(volatility, 6) if not np.isnan(volatility) else 0.0,
        "vol_percentile": round(vol_percentile, 2),
        "is_trending": is_trending,
        "bars_used": len(close),
    }

    return regime, features


def _compute_percentile(value: float, series: pd.Series) -> float:
    """Compute percentile of value within series."""
    if pd.isna(value) or len(series) == 0:
        return 50.0
    return float((series < value).sum() / len(series) * 100)


def _compute_trend_strength(close: pd.Series, lookback: int = 50) -> float:
    """
    Compute trend strength as R-squared of linear regression.

    Returns value between 0 (no trend) and 1 (perfect trend).
    """
    if len(close) < lookback:
        lookback = len(close)

    if lookback < 5:
        return 0.0

    window = close.tail(lookback).values
    x = np.arange(len(window))

    # Check for valid data
    if np.isnan(window).any():
        return 0.0

    try:
        correlation = np.corrcoef(x, window)[0, 1]
        if np.isnan(correlation):
            return 0.0
        return float(correlation**2)
    except (ValueError, FloatingPointError):
        return 0.0


# =============================================================================
# Confidence Components
# =============================================================================


def compute_components(ctx: ConfidenceContext, regime: str) -> dict[str, float]:
    """
    Compute confidence component scores.

    Each component is normalized to [0, 1].

    Args:
        ctx: ConfidenceContext with metrics
        regime: Current regime classification

    Returns:
        Dict of component name -> score [0, 1]
    """
    components = {}

    # 1. Performance: normalized from backtest/WFO metrics
    components["performance"] = _compute_performance_component(ctx)

    # 2. Drawdown: penalty based on max DD
    components["drawdown"] = _compute_drawdown_component(ctx)

    # 3. Stability: variance across folds or time
    components["stability"] = _compute_stability_component(ctx)

    # 4. Data freshness: penalty if inputs are stale
    components["data_freshness"] = _compute_freshness_component(ctx)

    # 5. Regime fit: bonus/penalty based on strategy profile
    components["regime_fit"] = _compute_regime_fit_component(ctx, regime)

    return components


def _compute_performance_component(ctx: ConfidenceContext) -> float:
    """
    Compute performance component from backtest metrics.

    Uses Sharpe ratio as primary metric, normalized to [0, 1].
    Sharpe mapping: -1 → 0, 0 → 0.3, 1 → 0.6, 2 → 0.8, 3+ → 1.0
    """
    # Prefer WFO OOS metrics if available
    if ctx.wfo_metrics and ctx.wfo_metrics.get("oos_sharpe") is not None:
        sharpe = ctx.wfo_metrics["oos_sharpe"]
    elif ctx.backtest_metrics and ctx.backtest_metrics.get("sharpe") is not None:
        sharpe = ctx.backtest_metrics["sharpe"]
    else:
        return 0.5  # No data - neutral

    # Normalize Sharpe to [0, 1]
    # Using sigmoid-like mapping
    if sharpe <= -1:
        return 0.0
    elif sharpe <= 0:
        return 0.3 * (sharpe + 1)  # -1 to 0 maps to 0 to 0.3
    elif sharpe <= 1:
        return 0.3 + 0.3 * sharpe  # 0 to 1 maps to 0.3 to 0.6
    elif sharpe <= 2:
        return 0.6 + 0.2 * (sharpe - 1)  # 1 to 2 maps to 0.6 to 0.8
    elif sharpe <= 3:
        return 0.8 + 0.2 * (sharpe - 2)  # 2 to 3 maps to 0.8 to 1.0
    else:
        return 1.0


def _compute_drawdown_component(ctx: ConfidenceContext) -> float:
    """
    Compute drawdown component (higher = less drawdown = better).

    Max DD mapping: 0% → 1.0, 10% → 0.8, 25% → 0.5, 50%+ → 0.0
    """
    max_dd = None

    if ctx.wfo_metrics and ctx.wfo_metrics.get("max_drawdown_pct") is not None:
        max_dd = abs(ctx.wfo_metrics["max_drawdown_pct"])
    elif (
        ctx.backtest_metrics
        and ctx.backtest_metrics.get("max_drawdown_pct") is not None
    ):
        max_dd = abs(ctx.backtest_metrics["max_drawdown_pct"])

    if max_dd is None:
        return 0.5  # No data - neutral

    # Convert to percentage if needed (handle both 0.25 and 25 formats)
    if max_dd < 1:
        max_dd = max_dd * 100

    # Linear interpolation
    if max_dd <= 0:
        return 1.0
    elif max_dd <= 10:
        return 1.0 - (max_dd / 10) * 0.2  # 0-10% → 1.0-0.8
    elif max_dd <= 25:
        return 0.8 - ((max_dd - 10) / 15) * 0.3  # 10-25% → 0.8-0.5
    elif max_dd <= 50:
        return 0.5 - ((max_dd - 25) / 25) * 0.5  # 25-50% → 0.5-0.0
    else:
        return 0.0


def _compute_stability_component(ctx: ConfidenceContext) -> float:
    """
    Compute stability component from WFO fold variance.

    Low variance across folds = high stability.
    """
    if ctx.wfo_metrics and ctx.wfo_metrics.get("fold_variance") is not None:
        variance = ctx.wfo_metrics["fold_variance"]

        # Normalize: variance 0 → 1.0, variance 0.5 → 0.5, variance 1+ → 0.0
        if variance <= 0:
            return 1.0
        elif variance >= 1:
            return 0.0
        else:
            return 1.0 - variance

    # No WFO data - use backtest trade count as proxy
    if ctx.backtest_metrics and ctx.backtest_metrics.get("trades") is not None:
        trades = ctx.backtest_metrics["trades"]

        # More trades = more statistical significance = more stable
        # <10 trades → low confidence, 50+ trades → high confidence
        if trades < 10:
            return 0.3
        elif trades < 30:
            return 0.3 + (trades - 10) / 20 * 0.3  # 10-30 → 0.3-0.6
        elif trades < 50:
            return 0.6 + (trades - 30) / 20 * 0.2  # 30-50 → 0.6-0.8
        else:
            return 0.8 + min((trades - 50) / 100, 0.2)  # 50+ → 0.8-1.0

    return 0.5  # No data - neutral


def _compute_freshness_component(ctx: ConfidenceContext) -> float:
    """
    Compute data freshness component.

    Penalty if as_of_ts is too far from latest candle.
    """
    if ctx.latest_candle_ts is None:
        return 0.7  # No info - slight penalty

    # Calculate staleness in hours
    staleness = (ctx.as_of_ts - ctx.latest_candle_ts).total_seconds() / 3600

    if staleness < 0:
        # as_of_ts is in the future relative to candle - odd but not penalized
        return 1.0
    elif staleness <= 1:
        return 1.0  # Fresh
    elif staleness <= 4:
        return 0.9  # Slightly stale
    elif staleness <= 24:
        return 0.7  # Moderately stale
    elif staleness <= 168:  # 1 week
        return 0.5
    else:
        return 0.3  # Very stale


def _compute_regime_fit_component(ctx: ConfidenceContext, regime: str) -> float:
    """
    Compute regime fit component based on strategy profile.

    Bonus if current regime matches strategy's known good regimes.
    Penalty if it matches known bad regimes.
    """
    if ctx.strategy_regime_profile is None:
        return 0.5  # No profile - neutral

    good_regimes = ctx.strategy_regime_profile.get("good_regimes", [])
    bad_regimes = ctx.strategy_regime_profile.get("bad_regimes", [])

    # Check for exact or partial match
    regime_parts = set(regime.split("_"))

    # Good regime match
    for good in good_regimes:
        good_parts = set(good.split("_"))
        if regime == good or regime_parts & good_parts:
            return 0.85 if regime == good else 0.7

    # Bad regime match
    for bad in bad_regimes:
        bad_parts = set(bad.split("_"))
        if regime == bad:
            return 0.15
        if regime_parts & bad_parts:
            return 0.3

    return 0.5  # No match - neutral


# =============================================================================
# Main Computation
# =============================================================================


def compute_confidence(
    ctx: ConfidenceContext,
    weights: Optional[dict[str, float]] = None,
) -> ConfidenceResult:
    """
    Compute confidence score and regime for a strategy version.

    This is the main entry point for confidence computation.

    Args:
        ctx: ConfidenceContext with all input data
        weights: Optional custom weights (defaults to DEFAULT_WEIGHTS)

    Returns:
        ConfidenceResult with regime, score, components, and inputs_hash
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Step 1: Compute regime from OHLCV
    regime, regime_features = compute_regime(ctx.ohlcv)

    # Step 2: Compute confidence components
    components = compute_components(ctx, regime)

    # Step 3: Compute weighted score
    score = 0.0
    total_weight = 0.0

    for component_name, component_value in components.items():
        weight = weights.get(component_name, 0.0)
        score += weight * component_value
        total_weight += weight

    # Normalize if weights don't sum to 1
    if total_weight > 0:
        score = score / total_weight

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # Step 4: Build explanation
    explain = _build_explanation(components, weights, regime)

    # Step 5: Compute inputs hash for deduplication
    inputs_hash = _compute_inputs_hash(ctx, regime, components, weights)

    # Step 6: Build features dict
    features = {
        "regime": regime_features,
        "metrics_source": _get_metrics_source(ctx),
    }

    return ConfidenceResult(
        regime=regime,
        confidence_score=round(score, 4),
        confidence_components={k: round(v, 4) for k, v in components.items()},
        features=features,
        explain=explain,
        inputs_hash=inputs_hash,
    )


def _build_explanation(
    components: dict[str, float],
    weights: dict[str, float],
    regime: str,
) -> dict:
    """Build human-readable explanation of confidence score."""
    explanations = []

    # Sort by weighted contribution
    contributions = [
        (name, components[name], weights.get(name, 0.0)) for name in components
    ]
    contributions.sort(key=lambda x: x[1] * x[2], reverse=True)

    for name, value, weight in contributions:
        if weight > 0:
            level = _score_to_level(value)
            contribution = round(value * weight, 3)
            explanations.append(
                {
                    "component": name,
                    "score": round(value, 3),
                    "weight": weight,
                    "contribution": contribution,
                    "level": level,
                }
            )

    return {
        "regime": regime,
        "breakdown": explanations,
        "summary": _generate_summary(components, regime),
    }


def _score_to_level(score: float) -> str:
    """Convert score to human-readable level."""
    if score >= 0.8:
        return "high"
    elif score >= 0.6:
        return "good"
    elif score >= 0.4:
        return "moderate"
    elif score >= 0.2:
        return "low"
    else:
        return "poor"


def _generate_summary(components: dict[str, float], regime: str) -> str:
    """Generate one-line summary of confidence."""
    avg = sum(components.values()) / len(components) if components else 0.5

    if avg >= 0.7:
        return f"High confidence in {regime} regime"
    elif avg >= 0.5:
        return f"Moderate confidence in {regime} regime"
    elif avg >= 0.3:
        return f"Low confidence in {regime} regime"
    else:
        return f"Poor confidence in {regime} regime - caution advised"


def _get_metrics_source(ctx: ConfidenceContext) -> str:
    """Determine which metrics source was used."""
    if ctx.wfo_metrics and any(
        ctx.wfo_metrics.get(k) is not None for k in ["oos_sharpe", "oos_return_pct"]
    ):
        return "wfo"
    elif ctx.backtest_metrics and any(
        ctx.backtest_metrics.get(k) is not None for k in ["sharpe", "return_pct"]
    ):
        return "backtest"
    else:
        return "none"


def _compute_inputs_hash(
    ctx: ConfidenceContext,
    regime: str,
    components: dict[str, float],
    weights: dict[str, float],
) -> str:
    """
    Compute deterministic hash of all inputs for deduplication.

    Hash includes:
    - version_id
    - as_of_ts
    - regime
    - component values (rounded)
    - weights
    - algorithm version
    """
    hash_input = {
        "version_id": str(ctx.version_id),
        "as_of_ts": ctx.as_of_ts.isoformat(),
        "regime": regime,
        "components": {k: round(v, 4) for k, v in sorted(components.items())},
        "weights": {k: round(v, 4) for k, v in sorted(weights.items())},
        "algo_version": ALGO_VERSION,
    }

    # Deterministic JSON serialization
    json_str = json.dumps(hash_input, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(json_str.encode()).hexdigest()
