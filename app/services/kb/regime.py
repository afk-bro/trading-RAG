"""
Regime computation for the Trading Knowledge Base.

Implements market regime feature extraction from OHLCV data:
- ATR (Average True Range)
- RSI (Relative Strength Index)
- Bollinger Band width
- Kaufman Efficiency Ratio
- Trend strength and direction
- Z-score positioning

All indicators implemented in numpy/pandas for determinism and minimal dependencies.
"""

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from app.services.kb.types import RegimeSnapshot, TagRule, TagEvidence, Op, utc_now_iso
from app.services.kb.constants import (
    REGIME_SCHEMA_VERSION,
    DEFAULT_FEATURE_PARAMS,
    REGIME_WINDOW_BARS,
    MIN_EFFECTIVE_BARS,
    TREND_STRONG_THRESHOLD,
    TREND_WEAK_THRESHOLD,
    VOL_LOW_THRESHOLD,
    VOL_HIGH_THRESHOLD,
    BB_WIDTH_CHOPPY_THRESHOLD,
    ZSCORE_MEAN_REVERT_THRESHOLD,
    ER_NOISY_THRESHOLD,
    ER_EFFICIENT_THRESHOLD,
    ZSCORE_OVERSOLD,
    ZSCORE_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    OUTLIER_PCT_CHANGE_THRESHOLD,
    EXCLUSIVE_FAMILIES,
)


# =============================================================================
# Technical Indicator Implementations
# =============================================================================


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR).

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = EWM of True Range with span=period

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)

    Returns:
        ATR series
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Use EWM for standard ATR calculation
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    RS = avg_gain / avg_loss (EWM)

    Args:
        close: Close prices
        period: RSI period (default 14)

    Returns:
        RSI series (0-100)
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge case where loss is 0 (RSI = 100)
    rsi = rsi.fillna(100.0)

    return rsi


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    k: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.

    Middle = SMA(close, period)
    Upper = Middle + k * std
    Lower = Middle - k * std

    Args:
        close: Close prices
        period: Moving average period (default 20)
        k: Standard deviation multiplier (default 2.0)

    Returns:
        Tuple of (middle, upper, lower) bands
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + k * std
    lower = middle - k * std

    return middle, upper, lower


def compute_bb_width_pct(
    close: pd.Series,
    period: int = 20,
    k: float = 2.0,
) -> pd.Series:
    """
    Compute Bollinger Band width as percentage.

    BB Width % = (Upper - Lower) / Middle

    Args:
        close: Close prices
        period: BB period (default 20)
        k: Standard deviation multiplier (default 2.0)

    Returns:
        BB width as fraction (0.05 = 5%)
    """
    middle, upper, lower = compute_bollinger_bands(close, period, k)

    # Avoid division by zero
    width_pct = (upper - lower) / middle.replace(0, np.nan)

    return width_pct


def compute_efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Compute Kaufman Efficiency Ratio.

    ER = |net_change| / sum(|bar_changes|)

    Ranges from 0 (choppy/noisy) to 1 (trending/efficient).

    Args:
        close: Close prices
        period: Lookback period (default 10)

    Returns:
        Efficiency ratio series (0-1)
    """
    # Net change over period
    net_change = close.diff(period).abs()

    # Sum of absolute bar-to-bar changes
    abs_changes = close.diff().abs()
    sum_changes = abs_changes.rolling(window=period).sum()

    # Avoid division by zero
    er = net_change / sum_changes.replace(0, np.nan)

    # Clamp to [0, 1]
    er = er.clip(0.0, 1.0)

    return er


def compute_zscore(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute Z-score of price relative to rolling mean.

    Z = (price - rolling_mean) / rolling_std

    Args:
        close: Close prices
        period: Rolling period (default 20)

    Returns:
        Z-score series
    """
    rolling_mean = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()

    zscore = (close - rolling_mean) / rolling_std.replace(0, np.nan)

    return zscore


def compute_trend_strength(close: pd.Series, lookback: int = 50) -> float:
    """
    Compute trend strength as R-squared of linear regression.

    Measures how well price follows a linear trend (0-1).

    Args:
        close: Close prices
        lookback: Period for trend calculation

    Returns:
        Trend strength (0-1)
    """
    if len(close) < lookback:
        return 0.0

    window = close.tail(lookback).values
    x = np.arange(len(window))

    # Check for valid data
    if np.isnan(window).any() or len(window) < 2:
        return 0.0

    # Linear regression R-squared
    try:
        correlation = np.corrcoef(x, window)[0, 1]
        if np.isnan(correlation):
            return 0.0
        r_squared = correlation**2
        return float(r_squared)
    except (ValueError, FloatingPointError):
        return 0.0


def compute_trend_direction(close: pd.Series, lookback: int = 50) -> int:
    """
    Compute trend direction from linear regression slope.

    Args:
        close: Close prices
        lookback: Period for trend calculation

    Returns:
        -1 (downtrend), 0 (flat), +1 (uptrend)
    """
    if len(close) < lookback:
        return 0

    window = close.tail(lookback).values
    x = np.arange(len(window))

    # Check for valid data
    if np.isnan(window).any() or len(window) < 2:
        return 0

    try:
        # Simple linear regression slope
        n = len(window)
        slope = (n * np.sum(x * window) - np.sum(x) * np.sum(window)) / (
            n * np.sum(x**2) - np.sum(x) ** 2
        )

        if np.isnan(slope):
            return 0

        # Normalize by average price to get relative slope
        avg_price = np.mean(window)
        if avg_price == 0:
            return 0

        rel_slope = slope / avg_price

        # Threshold for direction determination (0.1% per bar)
        threshold = 0.001

        if rel_slope > threshold:
            return 1
        elif rel_slope < -threshold:
            return -1
        else:
            return 0

    except (ValueError, FloatingPointError, ZeroDivisionError):
        return 0


# =============================================================================
# Declarative Tagging Rules (v1.1)
# =============================================================================

# DEFAULT_RULESET: Declarative rules for all regime tags
# - Rules grouped by (tag, group): AND within group, OR across groups
# - is_headline=True: surface in near-miss extraction
# - transform="abs": apply abs() before comparison
DEFAULT_RULESET: list[TagRule] = [
    # Uptrend: strong trend + positive direction (AND within default group)
    TagRule(
        "uptrend",
        "uptrend_strength",
        "trend_strength",
        ">=",
        TREND_STRONG_THRESHOLD,
        is_headline=True,
    ),
    TagRule("uptrend", "uptrend_dir", "trend_dir", ">", 0),
    # Downtrend: strong trend + negative direction (AND within default group)
    TagRule(
        "downtrend",
        "downtrend_strength",
        "trend_strength",
        ">=",
        TREND_STRONG_THRESHOLD,
        is_headline=True,
    ),
    TagRule("downtrend", "downtrend_dir", "trend_dir", "<", 0),
    # Trending: strong trend but neutral direction (AND within default group)
    # This is a fallback when trend is strong but direction is 0
    TagRule(
        "trending",
        "trending_strength",
        "trend_strength",
        ">=",
        TREND_STRONG_THRESHOLD,
        is_headline=True,
    ),
    TagRule("trending", "trending_dir", "trend_dir", "==", 0),
    # Flat: weak trend (single rule)
    TagRule(
        "flat",
        "flat_weak_trend",
        "trend_strength",
        "<",
        TREND_WEAK_THRESHOLD,
        is_headline=True,
    ),
    # Volatility (mutually exclusive by threshold, no middle tag)
    TagRule(
        "low_vol",
        "low_vol_atr",
        "atr_pct",
        "<",
        VOL_LOW_THRESHOLD,
        units="%",
        is_headline=True,
    ),
    TagRule(
        "high_vol",
        "high_vol_atr",
        "atr_pct",
        ">",
        VOL_HIGH_THRESHOLD,
        units="%",
        is_headline=True,
    ),
    # Mean-reverting: requires flat + extreme zscore (abs transform)
    TagRule("mean_reverting", "mr_flat", "trend_strength", "<", TREND_WEAK_THRESHOLD),
    TagRule(
        "mean_reverting",
        "mr_zscore",
        "zscore",
        ">",
        ZSCORE_MEAN_REVERT_THRESHOLD,
        transform="abs",
        units="σ",
        is_headline=True,
    ),
    # Choppy: requires flat + narrow bands
    TagRule("choppy", "choppy_flat", "trend_strength", "<", TREND_WEAK_THRESHOLD),
    TagRule(
        "choppy",
        "choppy_bb",
        "bb_width_pct",
        "<",
        BB_WIDTH_CHOPPY_THRESHOLD,
        units="%",
        is_headline=True,
    ),
    # Efficiency (mutually exclusive by threshold, no middle tag)
    TagRule(
        "noisy",
        "noisy_er",
        "efficiency_ratio",
        "<",
        ER_NOISY_THRESHOLD,
        is_headline=True,
    ),
    TagRule(
        "efficient",
        "efficient_er",
        "efficiency_ratio",
        ">",
        ER_EFFICIENT_THRESHOLD,
        is_headline=True,
    ),
    # Oversold: zscore OR rsi (separate groups for OR semantics)
    TagRule(
        "oversold",
        "oversold_zscore",
        "zscore",
        "<",
        ZSCORE_OVERSOLD,
        units="σ",
        is_headline=True,
        group="zscore",
    ),
    TagRule(
        "oversold",
        "oversold_rsi",
        "rsi",
        "<",
        RSI_OVERSOLD,
        units="RSI",
        is_headline=True,
        group="rsi",
    ),
    # Overbought: zscore OR rsi (separate groups for OR semantics)
    TagRule(
        "overbought",
        "overbought_zscore",
        "zscore",
        ">",
        ZSCORE_OVERBOUGHT,
        units="σ",
        is_headline=True,
        group="zscore",
    ),
    TagRule(
        "overbought",
        "overbought_rsi",
        "rsi",
        ">",
        RSI_OVERBOUGHT,
        units="RSI",
        is_headline=True,
        group="rsi",
    ),
]

# Exclusive families: at most one tag per family can be assigned
# Order determines priority (first passing wins)


def _evaluate_op(value: float, op: Op, threshold: float) -> bool:
    """Evaluate comparison operation."""
    if op == ">=":
        return value >= threshold
    elif op == ">":
        return value > threshold
    elif op == "<=":
        return value <= threshold
    elif op == "<":
        return value < threshold
    elif op == "==":
        return value == threshold
    return False


def _compute_margin(value: float, op: Op, threshold: float) -> float:
    """
    Compute normalized margin (positive = passed).

    For >= and >: margin = value - threshold (positive if passed)
    For <= and <: margin = threshold - value (positive if passed)
    For ==: margin = -abs(value - threshold) (0 if exact match)
    """
    if op in (">=", ">"):
        return value - threshold
    elif op in ("<=", "<"):
        return threshold - value
    elif op == "==":
        return -abs(value - threshold)
    return 0.0


def evaluate_rules(
    features: dict[str, float],
    ruleset: list[TagRule] | None = None,
) -> tuple[list[str], list[TagEvidence]]:
    """
    Evaluate ruleset against features.

    Evaluator semantics:
    - Group rules by (tag, group)
    - AND within group: all rules in a group must pass
    - OR across groups: any group passing assigns the tag
    - Exclusive families: at most one tag per family

    Args:
        features: Dict mapping metric names to values
        ruleset: List of TagRule (defaults to DEFAULT_RULESET)

    Returns:
        Tuple of (assigned_tags, all_evidence)
    """
    if ruleset is None:
        ruleset = DEFAULT_RULESET

    evidence: list[TagEvidence] = []

    # Group rules by (tag, group)
    grouped: dict[str, dict[str, list[TagRule]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rule in ruleset:
        grouped[rule.tag][rule.group].append(rule)

    # Track which tags pass (before exclusive family filtering)
    passing_tags: set[str] = set()

    for tag, groups in grouped.items():
        tag_assigned = False

        for group_name, rules in groups.items():
            # AND within group: all rules must pass
            group_passed = True
            group_evidence = []

            for rule in rules:
                # Get raw value and apply transform
                raw_value = features.get(rule.metric, 0.0)
                if rule.transform == "abs":
                    computed = abs(raw_value)
                else:
                    computed = raw_value

                # Evaluate using computed value
                passed = _evaluate_op(computed, rule.op, rule.threshold)
                margin = _compute_margin(computed, rule.op, rule.threshold)

                ev = TagEvidence(
                    tag=rule.tag,
                    rule_id=rule.rule_id,
                    passed=passed,
                    metric=rule.metric,
                    value=raw_value,  # Always store raw
                    op=rule.op,
                    threshold=rule.threshold,
                    units=rule.units,
                    margin=margin,
                    transform=rule.transform,
                    computed_value=computed if rule.transform else None,
                )
                group_evidence.append(ev)

                if not passed:
                    group_passed = False

            evidence.extend(group_evidence)

            # OR across groups: any group passing assigns tag
            if group_passed:
                tag_assigned = True

        if tag_assigned:
            passing_tags.add(tag)

    # Apply exclusive family filtering
    assigned_tags: list[str] = []
    claimed_families: set[str] = set()

    for family_name, family_members in EXCLUSIVE_FAMILIES.items():
        # Find first passing tag in priority order
        for tag in family_members:
            if tag in passing_tags:
                assigned_tags.append(tag)
                claimed_families.add(family_name)
                break

    # Add all passing tags not in any exclusive family
    all_family_tags = {
        tag for members in EXCLUSIVE_FAMILIES.values() for tag in members
    }
    for tag in passing_tags:
        if tag not in all_family_tags:
            assigned_tags.append(tag)

    return sorted(assigned_tags), evidence


def snapshot_to_features(s: "RegimeSnapshot") -> dict[str, float]:
    """
    Extract feature dict from RegimeSnapshot for rule evaluation.

    Maps snapshot fields to feature names used in DEFAULT_RULESET.
    """
    return {
        "trend_strength": s.trend_strength,
        "trend_dir": float(s.trend_dir),  # int to float for consistent comparison
        "atr_pct": s.atr_pct,
        "zscore": s.zscore,
        "rsi": s.rsi,
        "efficiency_ratio": s.efficiency_ratio,
        "bb_width_pct": s.bb_width_pct,
    }


def compute_tags_with_evidence(
    s: "RegimeSnapshot",
    ruleset: list[TagRule] | None = None,
) -> tuple[list[str], list[TagEvidence]]:
    """
    Compute regime tags with full evidence trail.

    This is the v1.1 replacement for compute_tags().
    Returns both tags and evidence for explainability.

    Args:
        s: RegimeSnapshot with computed features
        ruleset: Optional custom ruleset (defaults to DEFAULT_RULESET)

    Returns:
        Tuple of (sorted tags, all evidence)
    """
    features = snapshot_to_features(s)
    return evaluate_rules(features, ruleset)


# =============================================================================
# Legacy Tagging (preserved for compatibility verification)
# =============================================================================


def compute_tags(s: RegimeSnapshot) -> list[str]:
    """
    Compute deterministic regime tags from snapshot features.

    Tags are alphabetically sorted for consistency.

    Args:
        s: RegimeSnapshot with computed features

    Returns:
        Sorted list of regime tags
    """
    tags = []

    # Trend
    if s.trend_strength > TREND_STRONG_THRESHOLD:
        if s.trend_dir > 0:
            tags.append("uptrend")
        elif s.trend_dir < 0:
            tags.append("downtrend")
        else:
            tags.append("trending")
    elif s.trend_strength < TREND_WEAK_THRESHOLD:
        tags.append("flat")

    # Volatility
    if s.atr_pct < VOL_LOW_THRESHOLD:
        tags.append("low_vol")
    elif s.atr_pct > VOL_HIGH_THRESHOLD:
        tags.append("high_vol")

    # Regime type (mutually exclusive with flat)
    if s.trend_strength < TREND_WEAK_THRESHOLD:
        if abs(s.zscore) > ZSCORE_MEAN_REVERT_THRESHOLD:
            tags.append("mean_reverting")
        elif s.bb_width_pct < BB_WIDTH_CHOPPY_THRESHOLD:
            tags.append("choppy")

    # Noise level
    if s.efficiency_ratio < ER_NOISY_THRESHOLD:
        tags.append("noisy")
    elif s.efficiency_ratio > ER_EFFICIENT_THRESHOLD:
        tags.append("efficient")

    # Oscillator extremes (independent, can co-occur)
    if s.zscore < ZSCORE_OVERSOLD or s.rsi < RSI_OVERSOLD:
        tags.append("oversold")
    if s.zscore > ZSCORE_OVERBOUGHT or s.rsi > RSI_OVERBOUGHT:
        tags.append("overbought")

    return sorted(tags)


# =============================================================================
# Text Template
# =============================================================================


def regime_snapshot_to_text(s: RegimeSnapshot) -> str:
    """
    Convert RegimeSnapshot to text for embedding.

    Deterministic format for stable embeddings.

    Args:
        s: RegimeSnapshot

    Returns:
        Text representation
    """
    tags_str = ", ".join(s.regime_tags) if s.regime_tags else "neutral"
    tf = f" ({s.timeframe})" if s.timeframe else ""

    return f"""Market regime{tf}: {tags_str}.
Volatility: ATR {s.atr_pct*100:.2f}%, BB width {s.bb_width_pct*100:.1f}%.
Trend: strength {s.trend_strength:.2f}, direction {s.trend_dir:+d}.
Position: z-score {s.zscore:.2f}, RSI {s.rsi:.0f}.
Efficiency: {s.efficiency_ratio:.2f}."""


# =============================================================================
# Main Computation Function
# =============================================================================


def compute_regime_snapshot(
    df: pd.DataFrame,
    source: str = "live",
    instrument: Optional[str] = None,
    timeframe: Optional[str] = None,
    feature_params: Optional[dict] = None,
) -> RegimeSnapshot:
    """
    Compute a RegimeSnapshot from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] or similar
            (will be normalized). Index should be datetime or have a 'date'/'ts' column.
        source: Computation source ("live", "backfill", "query")
        instrument: Optional instrument identifier
        timeframe: Optional timeframe string (e.g., "1h", "1d")
        feature_params: Override default feature parameters

    Returns:
        RegimeSnapshot with computed features, tags, and warnings
    """
    warnings = []
    params = {**DEFAULT_FEATURE_PARAMS, **(feature_params or {})}

    # Normalize column names
    df = _normalize_ohlcv_columns(df)

    # Get window
    window_size = min(REGIME_WINDOW_BARS, len(df))
    window = df.tail(window_size).copy()

    n_bars = len(window)
    if n_bars < MIN_EFFECTIVE_BARS:
        warnings.append("insufficient_bars")

    # Extract OHLCV
    close = window["close"]
    high = window["high"]
    low = window["low"]

    # Compute features
    # ATR
    atr = compute_atr(high, low, close, period=params["atr_period"])
    atr_last = atr.iloc[-1] if not atr.empty else 0.0
    close_last = close.iloc[-1] if not close.empty else 1.0
    atr_pct = atr_last / close_last if close_last != 0 else 0.0

    # RSI
    rsi_series = compute_rsi(close, period=params["rsi_period"])
    rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50.0

    # Bollinger Band width
    bb_width = compute_bb_width_pct(close, period=params["bb_period"], k=params["bb_k"])
    bb_width_pct = bb_width.iloc[-1] if not bb_width.empty else 0.0

    # Rolling std
    std_series = close.rolling(window=params["bb_period"]).std()
    std_pct = (
        (std_series.iloc[-1] / close_last)
        if close_last != 0 and not std_series.empty
        else 0.0
    )

    # Range over window
    range_pct = (high.max() - low.min()) / close_last if close_last != 0 else 0.0

    # Z-score
    zscore_series = compute_zscore(close, period=params["z_period"])
    zscore = zscore_series.iloc[-1] if not zscore_series.empty else 0.0

    # Trend
    trend_strength = compute_trend_strength(close, lookback=params["trend_lookback"])
    trend_dir = compute_trend_direction(close, lookback=params["trend_lookback"])

    # Efficiency ratio
    er_series = compute_efficiency_ratio(
        close, period=min(params["trend_lookback"], n_bars // 2)
    )
    efficiency_ratio = er_series.iloc[-1] if not er_series.empty else 0.5

    # Return metrics
    if len(close) >= 2:
        return_pct = (
            (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
            if close.iloc[0] != 0
            else 0.0
        )
        drift_bps_per_bar = (
            (return_pct * 10000) / (len(close) - 1) if len(close) > 1 else 0.0
        )
    else:
        return_pct = 0.0
        drift_bps_per_bar = 0.0

    # Effective bars after NaN drops
    effective_n_bars = int(close.dropna().shape[0])
    if effective_n_bars < MIN_EFFECTIVE_BARS:
        warnings.append("low_effective_bars")

    # Timestamps
    ts_start = _get_timestamp(window, 0)
    ts_end = _get_timestamp(window, -1)

    # Data quality checks
    pct_changes = close.pct_change().abs()
    if (pct_changes > OUTLIER_PCT_CHANGE_THRESHOLD).any():
        warnings.append("possible_bad_ticks")

    if atr_last == 0:
        warnings.append("zero_atr")

    if std_series.iloc[-1] == 0 if not std_series.empty else True:
        warnings.append("zero_variance")

    # Handle NaN values
    atr_pct = _safe_float(atr_pct)
    std_pct = _safe_float(std_pct)
    bb_width_pct = _safe_float(bb_width_pct)
    range_pct = _safe_float(range_pct)
    zscore = _safe_float(zscore)
    rsi = _safe_float(rsi, default=50.0)
    trend_strength = _safe_float(trend_strength)
    efficiency_ratio = _safe_float(efficiency_ratio, default=0.5)
    return_pct = _safe_float(return_pct)
    drift_bps_per_bar = _safe_float(drift_bps_per_bar)

    # Create snapshot
    snapshot = RegimeSnapshot(
        schema_version=REGIME_SCHEMA_VERSION,
        feature_params=params,
        timeframe=timeframe,
        computed_at=utc_now_iso(),
        computation_source=source,
        n_bars=n_bars,
        effective_n_bars=effective_n_bars,
        ts_start=ts_start,
        ts_end=ts_end,
        atr_pct=atr_pct,
        std_pct=std_pct,
        bb_width_pct=bb_width_pct,
        range_pct=range_pct,
        trend_strength=trend_strength,
        trend_dir=trend_dir,
        zscore=zscore,
        rsi=rsi,
        return_pct=return_pct,
        drift_bps_per_bar=drift_bps_per_bar,
        efficiency_ratio=efficiency_ratio,
        instrument=instrument,
        source=source,
        warnings=warnings,
    )

    # Compute tags with evidence (v1.1)
    tags, evidence = compute_tags_with_evidence(snapshot)
    snapshot.regime_tags = tags
    snapshot.tag_evidence = evidence

    return snapshot


# =============================================================================
# Helpers
# =============================================================================


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV column names to lowercase standard."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Common aliases
    aliases = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "adj_close": "close",
        "adj close": "close",
    }

    df = df.rename(columns=aliases)

    return df


def _get_timestamp(df: pd.DataFrame, idx: int) -> str:
    """Get timestamp as ISO string from dataframe."""
    try:
        if df.index.name and "date" in df.index.name.lower():
            ts = df.index[idx]
        elif hasattr(df.index, "to_pydatetime"):
            ts = df.index[idx]
        elif "date" in df.columns:
            ts = df["date"].iloc[idx]
        elif "ts" in df.columns:
            ts = df["ts"].iloc[idx]
        elif "timestamp" in df.columns:
            ts = df["timestamp"].iloc[idx]
        else:
            return ""

        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return str(ts)
    except (IndexError, KeyError):
        return ""


def _safe_float(val: float, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/inf with default."""
    if pd.isna(val) or np.isinf(val):
        return default
    return float(val)


def compute_regime_from_ohlcv(
    ohlcv: list[dict],
    timeframe: Optional[str] = None,
    source: str = "query",
) -> RegimeSnapshot:
    """
    Convenience wrapper to compute regime from OHLCV list of dicts.

    Args:
        ohlcv: List of dicts with OHLCV data (open, high, low, close, volume)
        timeframe: Optional timeframe string
        source: Computation source

    Returns:
        RegimeSnapshot with computed features and tags
    """
    df = pd.DataFrame(ohlcv)
    return compute_regime_snapshot(df, source=source, timeframe=timeframe)
