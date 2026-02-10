"""
ICT Unicorn Model Strategy — v2.1 "Intent" (Structure-Gated).

A comprehensive ICT (Inner Circle Trader) strategy implementation
that combines multiple confluence factors for high-probability entries.

Version history:
  v1.0 "Scan"   — Classic soft-scored with 8 equal-weight criteria
  v2.0 "Bias"   — MTF bias-gated with mandatory/scored split (3M+5S)
  v2.1 "Intent" — Structure-gated: MSS + displacement as mandatory (5M+4S)

Entry Criteria (9 total = 5 mandatory + 4 scored):
  Mandatory (all must pass):
    1. HTF Bias confirms direction (Daily/4H/1H alignment)
    2. Stop placement within ATR-based limit (risk management)
    3. Valid macro time window (session timing)
    4. MSS (Market Structure Shift) shows directional change
    5. Displacement meets conviction threshold (institutional intent)
  Scored (min N of 4, default 3):
    6. Liquidity sweep has occurred (stop hunt)
    7. FVG (Fair Value Gap) is present on HTF
    8. Breaker or Mitigation Block provides entry zone
    9. LTF FVG/DOL (Draw on Liquidity) confirms setup

Key features:
- Volatility-normalized thresholds (FVG size, stop distance use ATR)
- Guardrailed soft scoring: mandatory gates + scored threshold
- Configurable session profiles (STRICT, NORMAL, WIDE)

Exit Criteria:
- Target: Opposing liquidity pool or FVG fill
- Stop: Below/above entry FVG or swing point
- EOD exit if enabled
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from math import floor
from typing import Any, Callable, Optional, Union
from uuid import UUID

from app.utils.time import to_eastern_time

from app.schemas import IntentAction, PaperState, TradeIntent
from app.services.strategy.indicators.ict_patterns import (
    FairValueGap,
    FVGType,
    BreakerBlock,
    MitigationBlock,
    LiquiditySweep,
    MarketStructureShift,
    BlockType,
    detect_fvgs,
    detect_breaker_blocks,
    detect_mitigation_blocks,
    detect_liquidity_sweeps,
    detect_mss,
    _calculate_atr,
)
from app.services.strategy.indicators.tf_bias import (
    TimeframeBias,
    BiasDirection,
    compute_tf_bias,
)
from app.services.strategy.models import (
    ExecutionSpec,
    MarketSnapshot,
    OHLCVBar,
    StrategyEvaluation,
)


# =============================================================================
# Strategy Family Versioning
# =============================================================================

STRATEGY_FAMILY = "Unicorn"

# Version history — each entry carries the criteria schema so docs can't
# disagree with code.  Only v2.1 is fully populated today; older versions
# record the schema they shipped with for archaeological reference.
MODEL_VERSIONS: dict[str, dict] = {
    "1.0": {
        "codename": "Scan",
        "mandatory": 0,
        "scored": 8,
        "desc": "Classic soft-scored with 8 equal-weight criteria",
    },
    "2.0": {
        "codename": "Bias",
        "mandatory": 3,
        "scored": 5,
        "desc": "MTF bias-gated with mandatory/scored split (3M+5S)",
    },
    "2.1": {
        "codename": "Intent",
        "mandatory": 5,
        "scored": 4,
        "desc": "Structure-gated: MSS + displacement as mandatory (5M+4S)",
    },
}

MODEL_VERSION = "2.1"
MODEL_CODENAME = MODEL_VERSIONS[MODEL_VERSION]["codename"]

EXPECTED_SEGMENT_ORDER = (
    "ver",
    "bias",
    "side",
    "displ",
    "minscore",
    "window",
    "ts",
    "mode",
)


def _label_segments(
    config: "UnicornConfig",
    *,
    direction_filter: object = None,
    time_stop_minutes: object = None,
    bar_bundle: object = None,
    eval_mode: object = None,
) -> list[tuple[str, str]]:
    """Return (key, value) pairs that define a run's identity.

    Pure config choices only — no market-derived facts (dates, symbols, bar
    counts).  This keeps labels deterministic across identical configs.
    """
    segs: list[tuple[str, str]] = []
    segs.append(
        ("ver", f"{STRATEGY_FAMILY.lower()}_v{MODEL_VERSION.replace('.', '_')}")
    )

    bias = "mtf" if bar_bundle is not None else "single"
    segs.append(("bias", bias))

    if direction_filter is None:
        side = "bidir"
    else:
        side = "long" if str(direction_filter).endswith("BULLISH") else "short"
    segs.append(("side", side))

    displ = (
        str(config.min_displacement_atr)
        if config.min_displacement_atr is not None
        else "off"
    )
    segs.append(("displ", displ))

    segs.append(("minscore", f"{config.min_scored_criteria}of4"))
    segs.append(("window", config.session_profile.value))

    ts = f"{time_stop_minutes}m" if time_stop_minutes is not None else "none"
    segs.append(("ts", ts))

    segs.append(("mode", "eval" if eval_mode else "research"))

    return segs


def build_run_label(
    config: "UnicornConfig",
    *,
    direction_filter: object = None,  # BiasDirection or None
    time_stop_minutes: object = None,  # int or None
    bar_bundle: object = None,  # BarBundle or None
    eval_mode: object = None,  # bool or None
) -> str:
    """Build a self-describing one-line run label from config + runtime params.

    Format: Unicorn v2.1 | Bias=<profile> | Side=<dir> |
    Displ=<val> | MinScore=N/4 | Window=<profile> | TS=<val>

    Contains only config choices — no market-derived facts (dates, symbols,
    bar counts).  Deterministic for identical configs.
    """
    segs = _label_segments(
        config,
        direction_filter=direction_filter,
        time_stop_minutes=time_stop_minutes,
        bar_bundle=bar_bundle,
        eval_mode=eval_mode,
    )
    # Human-friendly display format
    display_map: dict[str, Union[str, Callable[[str], str]]] = {
        "ver": f"{STRATEGY_FAMILY} v{MODEL_VERSION}",
        "bias": lambda v: f"Bias={v.upper() if v == 'mtf' else v.capitalize()}",
        "side": lambda v: f"Side={'BiDir' if v == 'bidir' else v.capitalize()}",
        "displ": lambda v: f"Displ={v}",
        "minscore": lambda v: f"MinScore={v.replace('of', '/')}",
        "window": lambda v: f"Window={v}",
        "ts": lambda v: f"TS={v}",
        "mode": lambda v: f"Mode={v.capitalize()}",
    }
    parts = []
    for key, val in segs:
        fmt = display_map[key]
        parts.append(fmt if isinstance(fmt, str) else fmt(val))
    return " | ".join(parts)


MAX_KEY_LEN = 160


def build_run_key(
    config: "UnicornConfig",
    *,
    direction_filter: object = None,
    time_stop_minutes: object = None,
    bar_bundle: object = None,
    eval_mode: object = None,
) -> str:
    """Machine-stable slug for database indexing, caching, and artifact naming.

    Example: unicorn_v2_1_bias_mtf_side_long_displ_0_3_minscore_2of4_window_strict_ts_30m

    Deterministic: same config + runtime params always produce the same key.
    Safe for filenames and DB indexing: lowercase alphanumeric + underscores,
    capped at MAX_KEY_LEN characters.
    """
    import hashlib
    import re

    segs = _label_segments(
        config,
        direction_filter=direction_filter,
        time_stop_minutes=time_stop_minutes,
        bar_bundle=bar_bundle,
        eval_mode=eval_mode,
    )
    raw = "_".join(f"{k}_{v}" for k, v in segs)
    # Collapse dots and any non-alphanumeric chars to underscores
    sanitized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")

    if len(sanitized) > MAX_KEY_LEN:
        full_hash = hashlib.sha256(sanitized.encode()).hexdigest()[:8]
        sanitized = f"{sanitized[:MAX_KEY_LEN - 9]}_{full_hash}"

    assert re.fullmatch(r"[a-z0-9_]+", sanitized)
    assert len(sanitized) <= MAX_KEY_LEN

    return sanitized


# =============================================================================
# Configuration
# =============================================================================

# Quantity rounding (8 decimals for crypto, fewer for futures)
QUANTITY_DECIMALS = 2  # NQ/ES use 2 decimal places


class ScaleOutPreset(str, Enum):
    """Locked scale-out configurations.  TUNING PHASE CLOSED.

    Only two presets exist.  Do not add more.

    NONE:      Full position rides to target/stop.  No partial exits.
    PROP_SAFE: 33% off at +1R, 67% trails.

    Decision record (real Databento NQ, Q3 2024, NY AM strict, eval-mode):
      - PROP_SAFE reduces max trailing DD by ~$1,500 vs baseline.
      - One fewer halted day, 24 fewer skipped signals.
      - Profit factor 0.99 → 1.13, expectancy -0.65 → +6.31 pts/setup.
      - Avg win R drops 1.88 → 1.51 (capped upside on the 33% leg).
      - Dollar PnL is worse due to per-leg commission at low contract counts.

    Rejected alternatives:
      B (50% @ +1R)   — halves the runner, too aggressive.
      D (50% @ +0.75R) — locks in too little, chokes MFE capture.
    """

    NONE = "none"
    PROP_SAFE = "prop_safe"  # 33% at +1R


# Immutable.  Do not add entries — scale-out tuning phase is closed.
SCALE_OUT_PARAMS: dict[ScaleOutPreset, dict] = {
    ScaleOutPreset.NONE: {
        "partial_exit_r": None,
        "partial_exit_pct": 0.0,
    },
    ScaleOutPreset.PROP_SAFE: {
        "partial_exit_r": 1.0,
        "partial_exit_pct": 0.33,
    },
}


class SessionProfile(str, Enum):
    """Trading session window profiles."""

    NY_OPEN = "ny_open"  # First 60 min of NY session
    STRICT = "strict"  # NY AM only
    NORMAL = "normal"  # London + NY AM
    WIDE = "wide"  # London + NY AM + NY PM
    LONDON = "london"  # London session only


# Session windows by profile
SESSION_WINDOWS = {
    SessionProfile.NY_OPEN: [
        (time(9, 30), time(10, 30)),  # First 60 min of NY
    ],
    SessionProfile.STRICT: [
        (time(9, 30), time(11, 0)),  # NY AM only
    ],
    SessionProfile.NORMAL: [
        (time(3, 0), time(4, 0)),  # London
        (time(9, 30), time(11, 0)),  # NY AM
    ],
    SessionProfile.WIDE: [
        (time(3, 0), time(4, 0)),  # London
        (time(9, 30), time(11, 0)),  # NY AM
        (time(13, 30), time(15, 0)),  # NY PM
        (time(19, 0), time(20, 0)),  # Asia
    ],
    SessionProfile.LONDON: [
        (time(3, 0), time(4, 0)),  # London only
    ],
}

# Legacy compatibility
MACRO_WINDOWS = SESSION_WINDOWS[SessionProfile.WIDE]


@dataclass
class UnicornConfig:
    """Configuration for Unicorn Model strategy."""

    # Scoring: minimum scored criteria to enter (out of 4 scored items)
    # Mandatory criteria (htf_bias, stop_valid, macro_window, mss, displacement) always required.
    min_scored_criteria: int = 3

    # Confidence gate (None = metric-only, not used for entry filtering)
    min_confidence: Optional[float] = None

    # Session
    session_profile: SessionProfile = SessionProfile.NORMAL

    # ATR-based thresholds
    atr_period: int = 14
    fvg_min_atr_mult: float = 0.3  # FVG must be >= 0.3 * ATR
    stop_max_atr_mult: float = 3.0  # Stop must be <= 3.0 * ATR

    # FVG invalidation threshold (partial fill)
    max_fvg_fill_pct: float = 1.0  # 1.0 = only 100% fill invalidates (legacy)

    # Stop placement buffer (points beyond FVG/sweep edge)
    stop_buffer_points: float = 2.0

    # Legacy absolute thresholds (used if ATR not available)
    max_stop_points_nq: float = 30.0
    max_stop_points_es: float = 10.0

    # Point values
    point_value_nq: float = 20.0
    point_value_es: float = 50.0

    # Pre-entry bar-quality guards (None = disabled)
    max_wick_ratio: Optional[float] = None  # e.g., 0.6 — skip if signal bar wick > 60%
    max_range_atr_mult: Optional[float] = (
        None  # e.g., 3.0 — skip if signal bar range > 3x ATR
    )

    # Displacement conviction guard (None = disabled)
    min_displacement_atr: Optional[float] = (
        None  # e.g., 0.5 — skip if MSS displacement < 0.5x ATR
    )

    # Sweep closure confirmation gate
    max_sweep_age_bars: Optional[int] = (
        None  # e.g., 10 — only accept sweeps within N HTF bars
    )
    require_sweep_settlement: bool = (
        False  # next bar must close on correct side of swept level
    )

    # Backstop cleanup flag: when False, skip the redundant post-scoring displacement guard
    enable_displacement_backstop: bool = True

    # Confidence tiering (None = disabled, all confidence levels pass)
    # Tier A: >= confidence_tier_a → full size
    # Tier B: >= confidence_tier_b → half size (50% risk budget)
    # Tier C: < confidence_tier_b → blocked (rejected)
    confidence_tier_a: Optional[float] = None
    confidence_tier_b: Optional[float] = None

    # NY AM timebox tightening (None = use default session window)
    # Minutes from 9:30 ET — e.g. 60 → window becomes 9:30-10:30 instead of 9:30-11:00
    ny_am_cutoff_minutes: Optional[int] = None

    def __post_init__(self):
        if not (0 <= self.min_scored_criteria <= 4):
            raise ValueError(
                f"min_scored_criteria must be 0-4 (got {self.min_scored_criteria}). "
                f"There are only 4 scored criteria; mandatory gates are always enforced."
            )
        if self.min_confidence is not None and not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be 0.0-1.0 or None (got {self.min_confidence})"
            )
        if self.max_wick_ratio is not None and not (0.0 < self.max_wick_ratio <= 1.0):
            raise ValueError("max_wick_ratio must be in (0.0, 1.0] when set")
        if self.max_range_atr_mult is not None and self.max_range_atr_mult <= 0:
            raise ValueError("max_range_atr_mult must be > 0 when set")
        if self.min_displacement_atr is not None and self.min_displacement_atr <= 0:
            raise ValueError("min_displacement_atr must be > 0 when set")
        if self.max_sweep_age_bars is not None and self.max_sweep_age_bars < 1:
            raise ValueError("max_sweep_age_bars must be >= 1 when set")
        if self.confidence_tier_a is not None and not (
            0.0 < self.confidence_tier_a <= 1.0
        ):
            raise ValueError("confidence_tier_a must be in (0.0, 1.0] when set")
        if self.confidence_tier_b is not None and not (
            0.0 < self.confidence_tier_b <= 1.0
        ):
            raise ValueError("confidence_tier_b must be in (0.0, 1.0] when set")
        if (self.confidence_tier_a is None) != (self.confidence_tier_b is None):
            raise ValueError(
                "confidence_tier_a and confidence_tier_b must both be set or both be None"
            )
        if (
            self.confidence_tier_a is not None
            and self.confidence_tier_b is not None
            and self.confidence_tier_b >= self.confidence_tier_a
        ):
            raise ValueError("confidence_tier_b must be < confidence_tier_a")
        if self.ny_am_cutoff_minutes is not None and not (
            1 <= self.ny_am_cutoff_minutes <= 90
        ):
            raise ValueError("ny_am_cutoff_minutes must be 1-90 when set")


# Default config
DEFAULT_CONFIG = UnicornConfig()

# Risk parameters for NQ/ES (legacy, for backward compatibility)
MAX_STOP_POINTS_NQ = 30  # 30 points max stop for NQ
MAX_STOP_POINTS_ES = 10  # 10 points max stop for ES

# Re-export point values from shared module for backward compatibility
from app.utils.instruments import (  # noqa: E402, F401
    POINT_VALUE_NQ,
    POINT_VALUE_ES,
    get_point_value,
)


@dataclass
class CriteriaScore:
    """Detailed scoring of each criterion."""

    htf_bias: bool = False
    liquidity_sweep: bool = False
    htf_fvg: bool = False
    breaker_block: bool = False
    ltf_fvg: bool = False
    mss: bool = False
    stop_valid: bool = False
    macro_window: bool = False
    displacement_valid: bool = False

    @property
    def score(self) -> int:
        """Total criteria met (0-9)."""
        return sum(
            [
                self.htf_bias,
                self.liquidity_sweep,
                self.htf_fvg,
                self.breaker_block,
                self.ltf_fvg,
                self.mss,
                self.stop_valid,
                self.macro_window,
                self.displacement_valid,
            ]
        )

    @property
    def missing(self) -> list[str]:
        """List of criteria that failed."""
        missing = []
        if not self.htf_bias:
            missing.append("htf_bias")
        if not self.liquidity_sweep:
            missing.append("liquidity_sweep")
        if not self.htf_fvg:
            missing.append("htf_fvg")
        if not self.breaker_block:
            missing.append("breaker_block")
        if not self.ltf_fvg:
            missing.append("ltf_fvg")
        if not self.mss:
            missing.append("mss")
        if not self.stop_valid:
            missing.append("stop_valid")
        if not self.macro_window:
            missing.append("macro_window")
        if not self.displacement_valid:
            missing.append("displacement")
        return missing

    @property
    def mandatory_met(self) -> bool:
        """All 5 mandatory criteria must pass.

        Checks: htf_bias, stop_valid, macro_window, mss, displacement.
        """
        return (
            self.htf_bias
            and self.stop_valid
            and self.macro_window
            and self.mss
            and self.displacement_valid
        )

    @property
    def scored_count(self) -> int:
        """Count of passed scored criteria (out of 4)."""
        return sum(
            [
                self.liquidity_sweep,
                self.htf_fvg,
                self.breaker_block,
                self.ltf_fvg,
            ]
        )

    @property
    def scored_missing(self) -> list[str]:
        """Names of scored criteria that failed (subset of missing)."""
        _scored = {"liquidity_sweep", "htf_fvg", "breaker_block", "ltf_fvg"}
        return [name for name in self.missing if name in _scored]

    def decide_entry(self, min_scored: int = 3) -> bool:
        """
        Canonical entry gate used by both live evaluator and backtest.

        Requires all 5 mandatory criteria AND at least min_scored of 4 scored criteria.
        """
        return self.mandatory_met and self.scored_count >= min_scored

    def meets_threshold(self, threshold: int = 6) -> bool:
        """Check if score meets minimum threshold (legacy flat count)."""
        return self.score >= threshold


@dataclass
class UnicornSetup:
    """Complete Unicorn Model setup with all criteria."""

    direction: BiasDirection
    confidence: float
    entry_price: float  # Theoretical signal price (FVG midpoint)
    entry_price_model: float  # Same as entry_price; backtest overrides with fill_price
    stop_price: float
    target_price: float
    risk_points: float

    # Component confirmations
    htf_bias: TimeframeBias
    liquidity_sweep: Optional[LiquiditySweep]
    entry_fvg: Optional[FairValueGap]
    entry_block: Optional[BreakerBlock | MitigationBlock]
    ltf_fvg: Optional[FairValueGap]
    mss: Optional[MarketStructureShift]

    # Flags
    in_macro_window: bool
    stop_valid: bool

    # Scoring (new)
    criteria_score: CriteriaScore = field(default_factory=CriteriaScore)
    current_atr: float = 0.0  # For diagnostics

    @property
    def all_criteria_met(self) -> bool:
        """Check if all 9 criteria are satisfied (strict mode)."""
        return self.criteria_score.score == 9

    def meets_threshold(self, threshold: int = 6) -> bool:
        """Check if setup meets minimum criteria threshold."""
        return self.criteria_score.meets_threshold(threshold)


def round_quantity(qty: float, decimals: int = QUANTITY_DECIMALS) -> float:
    """Round DOWN to avoid over-allocation."""
    multiplier = 10**decimals
    return floor(qty * multiplier) / multiplier


def _apply_ny_am_cutoff(
    windows: list[tuple[time, time]],
    cutoff_minutes: int,
) -> list[tuple[time, time]]:
    """Replace the NY AM window end with 9:30 + cutoff_minutes."""
    ny_am_start = time(9, 30)
    new_end_total = 9 * 60 + 30 + cutoff_minutes
    new_end = time(new_end_total // 60, new_end_total % 60)
    result = []
    for start, end in windows:
        if start == ny_am_start:
            result.append((start, new_end))
        else:
            result.append((start, end))
    return result


def is_in_macro_window(
    ts: datetime,
    profile: SessionProfile = SessionProfile.WIDE,
    *,
    ny_am_cutoff_minutes: Optional[int] = None,
) -> bool:
    """
    Check if timestamp is within a valid macro trading window.

    Args:
        ts: Timezone-aware datetime to check (any timezone).
        profile: Session profile (STRICT, NORMAL, WIDE)
        ny_am_cutoff_minutes: Override NY AM window length (minutes from 9:30 ET).
                             e.g. 60 → 9:30-10:30 instead of 9:30-11:00.

    Raises:
        ValueError: If ts is a naive datetime.
    """
    current_time = to_eastern_time(ts)
    windows = SESSION_WINDOWS.get(profile, MACRO_WINDOWS)

    if ny_am_cutoff_minutes is not None:
        windows = _apply_ny_am_cutoff(windows, ny_am_cutoff_minutes)

    for start, end in windows:
        if start <= current_time < end:
            return True

    return False


def _calculate_atr_local(
    bars: list[OHLCVBar],
    period: int = 14,
) -> list[float]:
    """
    Calculate Average True Range (ATR) from OHLCVBar objects.

    Args:
        bars: List of OHLCV bars
        period: ATR period (default 14)

    Returns:
        List of ATR values (same length as input, with zeros for initial bars)
    """
    if len(bars) < 2:
        return [0.0] * len(bars)

    # Calculate True Range for each bar
    true_ranges: list[float] = [0.0]  # First bar has no TR
    for i in range(1, len(bars)):
        high = bars[i].high
        low = bars[i].low
        prev_close = bars[i - 1].close

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        true_ranges.append(tr)

    # Calculate ATR as exponential moving average of TR
    atr_values: list[float] = []
    for i in range(len(bars)):
        if i < period:
            # Use simple average for initial period
            atr_values.append(sum(true_ranges[: i + 1]) / (i + 1) if i > 0 else 0.0)
        else:
            # Use EMA-style smoothing
            prev_atr = atr_values[-1]
            current_tr = true_ranges[i]
            new_atr = (prev_atr * (period - 1) + current_tr) / period
            atr_values.append(new_atr)

    return atr_values


def get_max_stop_points(
    symbol: str,
    atr: Optional[float] = None,
    config: Optional[UnicornConfig] = None,
) -> float:
    """
    Get maximum allowed stop distance in points for the instrument.

    Args:
        symbol: Trading symbol
        atr: Current ATR value (if available, uses ATR-based limit)
        config: Strategy config (uses defaults if not provided)

    Returns:
        Maximum stop distance in price units
    """
    if config is None:
        config = DEFAULT_CONFIG

    # If ATR available, use volatility-normalized limit
    if atr is not None and atr > 0:
        return atr * config.stop_max_atr_mult

    # Fall back to absolute limits
    symbol_upper = symbol.upper()
    if "NQ" in symbol_upper or "MNQ" in symbol_upper:
        return config.max_stop_points_nq
    elif "ES" in symbol_upper or "MES" in symbol_upper:
        return config.max_stop_points_es
    else:
        return config.max_stop_points_nq  # Default to NQ


# get_point_value is imported from app.utils.instruments above


def find_entry_zone(
    fvgs: list[FairValueGap],
    breakers: list[BreakerBlock],
    mitigations: list[MitigationBlock],
    direction: BiasDirection,
    current_price: float,
    max_fvg_fill_pct: float = 1.0,
) -> tuple[Optional[FairValueGap], Optional[BreakerBlock | MitigationBlock], float]:
    """
    Find the best entry zone combining FVG with Breaker/Mitigation block.

    Returns:
        Tuple of (entry_fvg, entry_block, entry_price)
    """
    best_fvg: Optional[FairValueGap] = None
    best_block: Optional[BreakerBlock | MitigationBlock] = None
    entry_price = current_price

    # Filter FVGs by direction and invalidation status
    relevant_fvgs = [
        f
        for f in fvgs
        if not f.invalidated(max_fvg_fill_pct)
        and (
            (direction == BiasDirection.BULLISH and f.fvg_type == FVGType.BULLISH)
            or (direction == BiasDirection.BEARISH and f.fvg_type == FVGType.BEARISH)
        )
    ]

    if not relevant_fvgs:
        return None, None, entry_price

    # Find FVG closest to current price
    for fvg in relevant_fvgs:
        if direction == BiasDirection.BULLISH:
            # For bullish, look for FVG below current price
            if fvg.gap_high < current_price:
                if best_fvg is None or fvg.gap_high > best_fvg.gap_high:
                    best_fvg = fvg
        else:
            # For bearish, look for FVG above current price
            if fvg.gap_low > current_price:
                if best_fvg is None or fvg.gap_low < best_fvg.gap_low:
                    best_fvg = fvg

    if best_fvg is None:
        return None, None, entry_price

    # Look for overlapping breaker or mitigation block
    fvg_range = (best_fvg.gap_low, best_fvg.gap_high)

    # Check breakers
    for breaker in breakers:
        breaker_range = (breaker.low, breaker.high)
        if _ranges_overlap(fvg_range, breaker_range):
            if (
                direction == BiasDirection.BULLISH
                and breaker.block_type == BlockType.BULLISH
            ) or (
                direction == BiasDirection.BEARISH
                and breaker.block_type == BlockType.BEARISH
            ):
                best_block = breaker
                break

    # If no breaker, check mitigations
    if best_block is None:
        for mitigation in mitigations:
            mit_range = (mitigation.low, mitigation.high)
            if _ranges_overlap(fvg_range, mit_range):
                if (
                    direction == BiasDirection.BULLISH
                    and mitigation.block_type == BlockType.BULLISH
                ) or (
                    direction == BiasDirection.BEARISH
                    and mitigation.block_type == BlockType.BEARISH
                ):
                    best_block = mitigation
                    break

    # Entry at FVG midpoint or block edge
    if best_fvg:
        entry_price = best_fvg.midpoint

    return best_fvg, best_block, entry_price


def _ranges_overlap(r1: tuple[float, float], r2: tuple[float, float]) -> bool:
    """Check if two price ranges overlap."""
    return r1[0] <= r2[1] and r2[0] <= r1[1]


def calculate_stop_and_target(
    entry_price: float,
    direction: BiasDirection,
    entry_fvg: Optional[FairValueGap],
    liquidity_sweep: Optional[LiquiditySweep],
    symbol: str,
    atr: Optional[float] = None,
    config: Optional[UnicornConfig] = None,
) -> tuple[float, float, float, bool]:
    """
    Calculate stop loss and target prices.

    Args:
        entry_price: Entry price
        direction: Trade direction
        entry_fvg: Entry FVG (for stop placement)
        liquidity_sweep: Sweep (alternative stop placement)
        symbol: Trading symbol
        atr: Current ATR (for volatility-normalized stop limit)
        config: Strategy configuration

    Returns:
        Tuple of (stop_price, target_price, risk_points, stop_valid)
    """
    if config is None:
        config = DEFAULT_CONFIG

    max_points = get_max_stop_points(symbol, atr=atr, config=config)
    buf = config.stop_buffer_points

    if direction == BiasDirection.BULLISH:
        # Stop below FVG or sweep low
        if entry_fvg:
            stop_price = entry_fvg.gap_low - buf
        elif liquidity_sweep:
            stop_price = liquidity_sweep.sweep_low - buf
        else:
            stop_price = entry_price - max_points

        risk_points = entry_price - stop_price

        # Target at 2R or next liquidity
        target_price = entry_price + (risk_points * 2)

    else:  # BEARISH
        # Stop above FVG or sweep high
        if entry_fvg:
            stop_price = entry_fvg.gap_high + buf
        elif liquidity_sweep:
            stop_price = liquidity_sweep.sweep_high + buf
        else:
            stop_price = entry_price + max_points

        risk_points = stop_price - entry_price

        # Target at 2R
        target_price = entry_price - (risk_points * 2)

    stop_valid = risk_points <= max_points

    return stop_price, target_price, risk_points, stop_valid


def analyze_unicorn_setup(
    snapshot: MarketSnapshot,
    htf_bias: TimeframeBias,
    htf_bars: list[OHLCVBar],
    ltf_bars: list[OHLCVBar],
    config: Optional[UnicornConfig] = None,
) -> Optional[UnicornSetup]:
    """
    Analyze current market state for a Unicorn Model setup.

    Args:
        snapshot: Current market snapshot
        htf_bias: Pre-computed HTF bias
        htf_bars: Higher timeframe bars (15m-1H)
        ltf_bars: Lower timeframe bars (5m)
        config: Strategy configuration

    Returns:
        UnicornSetup with scoring (always returns if bias has direction)
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Initialize scoring
    score = CriteriaScore()

    direction = htf_bias.final_direction
    current_price = snapshot.bars[-1].close

    # Compute ATR for volatility normalization
    atr_values = _calculate_atr(htf_bars, config.atr_period)
    current_atr = atr_values[-1] if atr_values else 0.0

    # 1. HTF Bias check (+ optional confidence gate)
    confidence_ok = (
        config.min_confidence is None
        or htf_bias.final_confidence >= config.min_confidence
    )
    score.htf_bias = (
        htf_bias.is_tradeable and direction != BiasDirection.NEUTRAL and confidence_ok
    )

    if direction == BiasDirection.NEUTRAL:
        # Can't proceed without direction, return empty setup
        return UnicornSetup(
            direction=direction,
            confidence=htf_bias.final_confidence,
            entry_price=current_price,
            entry_price_model=current_price,
            stop_price=current_price,
            target_price=current_price,
            risk_points=0,
            htf_bias=htf_bias,
            liquidity_sweep=None,
            entry_fvg=None,
            entry_block=None,
            ltf_fvg=None,
            mss=None,
            in_macro_window=False,
            stop_valid=False,
            criteria_score=score,
            current_atr=current_atr,
        )

    # Detect ICT patterns on HTF (with ATR-normalized FVG threshold)
    htf_fvgs = detect_fvgs(
        htf_bars,
        lookback=50,
        min_gap_atr_mult=config.fvg_min_atr_mult,
        atr_period=config.atr_period,
    )
    htf_breakers = detect_breaker_blocks(htf_bars, lookback=50)
    htf_mitigations = detect_mitigation_blocks(htf_bars, lookback=50)
    htf_sweeps = detect_liquidity_sweeps(htf_bars, lookback=50)
    htf_mss = detect_mss(htf_bars, lookback=50)

    # Detect patterns on LTF (with ATR-normalized threshold)
    _calculate_atr(ltf_bars, config.atr_period)  # warmup ATR cache
    ltf_fvgs = detect_fvgs(
        ltf_bars,
        lookback=30,
        min_gap_atr_mult=config.fvg_min_atr_mult * 0.5,  # Lower threshold for LTF
        atr_period=config.atr_period,
    )

    # 2. Check for liquidity sweep in direction
    relevant_sweep: Optional[LiquiditySweep] = None
    for sweep in htf_sweeps:
        if direction == BiasDirection.BULLISH and sweep.sweep_type == "low":
            relevant_sweep = sweep
            break
        elif direction == BiasDirection.BEARISH and sweep.sweep_type == "high":
            relevant_sweep = sweep
            break
    score.liquidity_sweep = relevant_sweep is not None

    # 3. Find entry zone (HTF FVG)
    entry_fvg, entry_block, entry_price = find_entry_zone(
        htf_fvgs,
        htf_breakers,
        htf_mitigations,
        direction,
        current_price,
        max_fvg_fill_pct=config.max_fvg_fill_pct,
    )
    score.htf_fvg = entry_fvg is not None

    # 4. Breaker/Mitigation block
    score.breaker_block = entry_block is not None

    # 5. Check for MSS confirmation (prefer newest matching shift)
    relevant_mss: Optional[MarketStructureShift] = None
    for mss in reversed(htf_mss):
        if (direction == BiasDirection.BULLISH and mss.shift_type == "bullish") or (
            direction == BiasDirection.BEARISH and mss.shift_type == "bearish"
        ):
            relevant_mss = mss
            break
    score.mss = relevant_mss is not None

    # 5b. Displacement validation (mandatory context gate)
    score.displacement_valid = config.min_displacement_atr is None or (
        relevant_mss is not None
        and relevant_mss.displacement_size >= config.min_displacement_atr
    )

    # 6. Check for LTF FVG (DOL confirmation)
    relevant_ltf_fvg: Optional[FairValueGap] = None
    for fvg in ltf_fvgs:
        if not fvg.invalidated(config.max_fvg_fill_pct):
            if (
                direction == BiasDirection.BULLISH and fvg.fvg_type == FVGType.BULLISH
            ) or (
                direction == BiasDirection.BEARISH and fvg.fvg_type == FVGType.BEARISH
            ):
                relevant_ltf_fvg = fvg
                break
    score.ltf_fvg = relevant_ltf_fvg is not None

    # 7. Check macro window
    in_macro = is_in_macro_window(snapshot.ts, config.session_profile)
    score.macro_window = in_macro

    # 8. Calculate stop and target (ATR-based validation)
    stop_price, target_price, risk_points, stop_valid = calculate_stop_and_target(
        entry_price,
        direction,
        entry_fvg,
        relevant_sweep,
        snapshot.symbol,
        atr=current_atr,
        config=config,
    )
    score.stop_valid = stop_valid

    return UnicornSetup(
        direction=direction,
        confidence=htf_bias.final_confidence,
        entry_price=entry_price,
        entry_price_model=entry_price,  # Theoretical FVG midpoint
        stop_price=stop_price,
        target_price=target_price,
        risk_points=risk_points,
        htf_bias=htf_bias,
        liquidity_sweep=relevant_sweep,
        entry_fvg=entry_fvg,
        entry_block=entry_block,
        ltf_fvg=relevant_ltf_fvg,
        mss=relevant_mss,
        in_macro_window=in_macro,
        stop_valid=stop_valid,
        criteria_score=score,
        current_atr=current_atr,
    )


def evaluate_unicorn_model(
    spec: ExecutionSpec,
    snapshot: MarketSnapshot,
    paper_state: PaperState,
    evaluation_id: UUID,
    at_max_positions: bool,
    htf_bias: Optional[TimeframeBias] = None,
    htf_bars: Optional[list[OHLCVBar]] = None,
    ltf_bars: Optional[list[OHLCVBar]] = None,
    config: Optional[UnicornConfig] = None,
) -> StrategyEvaluation:
    """
    Evaluate ICT Unicorn Model strategy.

    This strategy requires:
    1. HTF Bias alignment (from multi-timeframe analysis)
    2. Liquidity sweep (stop hunt)
    3. FVG (Fair Value Gap) for entry zone
    4. Breaker or Mitigation block confluence
    5. LTF FVG confirmation (5-15m)
    6. MSS (Market Structure Shift)
    7. Valid stop placement (within max points)
    8. Valid macro time window

    Args:
        spec: Strategy configuration
        snapshot: Current market state with OHLCV bars
        paper_state: Current paper trading positions/cash
        evaluation_id: Shared ID for all intents
        at_max_positions: Whether position limit reached
        htf_bias: Pre-computed HTF bias (optional, will compute if not provided)
        htf_bars: Higher timeframe bars for pattern detection
        ltf_bars: Lower timeframe bars for confirmation

    Returns:
        StrategyEvaluation with intents, signals, and debug metadata
    """
    if config is None:
        config = DEFAULT_CONFIG

    symbol = snapshot.symbol
    position = paper_state.positions.get(symbol)
    has_position = position is not None and position.quantity > 0
    intents: list[TradeIntent] = []
    signals: list[str] = []

    # Get current price
    last_price = snapshot.last_price or snapshot.bars[-1].close

    # Guard: invalid price
    if last_price <= 0:
        signals.append("entry_skipped_invalid_price")
        return StrategyEvaluation(
            spec_id=str(spec.instance_id),
            symbol=symbol,
            ts=snapshot.ts,
            intents=[],
            signals=signals,
            metadata={"last_price": last_price},
            evaluation_id=evaluation_id,
        )

    # Use provided HTF bars or fall back to snapshot bars
    if htf_bars is None:
        htf_bars = snapshot.bars
    if ltf_bars is None:
        ltf_bars = snapshot.bars

    # Compute or use provided HTF bias
    if htf_bias is None:
        # In production, this would use actual multi-TF data
        # For now, use snapshot bars as proxy
        htf_bias = compute_tf_bias(
            m5_bars=ltf_bars,
            m15_bars=htf_bars,
        )

    # Helper to create intent
    def make_intent(**kwargs) -> TradeIntent:
        return TradeIntent(
            workspace_id=spec.workspace_id,
            correlation_id=str(evaluation_id),
            strategy_entity_id=spec.instance_id,
            symbol=symbol,
            timeframe=spec.timeframe,
            **kwargs,
        )

    # EXIT: EOD with position
    if snapshot.is_eod and has_position and position is not None:
        intents.append(
            make_intent(
                action=(
                    IntentAction.CLOSE_LONG
                    if position.side == "long"
                    else IntentAction.CLOSE_SHORT
                ),
                quantity=position.quantity,
                reason="EOD exit",
            )
        )
        signals.append("eod_exit_triggered")
        return StrategyEvaluation(
            spec_id=str(spec.instance_id),
            symbol=symbol,
            ts=snapshot.ts,
            intents=intents,
            signals=signals,
            metadata={"exit_type": "eod"},
            evaluation_id=evaluation_id,
        )

    # ENTRY: Analyze for Unicorn setup
    if not has_position:
        setup = analyze_unicorn_setup(
            snapshot, htf_bias, htf_bars, ltf_bars, config=config
        )

        if setup is None:
            signals.append("no_setup_detected")
        elif not setup.criteria_score.decide_entry(
            min_scored=config.min_scored_criteria
        ):
            # Log which criteria failed
            cs = setup.criteria_score
            if not cs.mandatory_met:
                if not cs.htf_bias:
                    if (
                        config.min_confidence is not None
                        and setup.confidence < config.min_confidence
                    ):
                        signals.append("htf_bias_confidence_below_threshold")
                    else:
                        signals.append("htf_bias_not_tradeable")
                if not cs.stop_valid:
                    signals.append(f"stop_too_wide_{setup.risk_points:.1f}_points")
                if not cs.macro_window:
                    signals.append("outside_macro_window")
            else:
                signals.append(
                    f"scored_{cs.scored_count}/4_below_{config.min_scored_criteria}"
                )
            for name in cs.missing:
                if name not in (
                    "htf_bias",
                    "stop_valid",
                    "macro_window",
                    "mss",
                    "displacement",
                ):
                    signals.append(f"no_{name}")
        else:
            # Entry criteria met - generate entry intent
            point_value = get_point_value(symbol)
            risk_per_contract = setup.risk_points * point_value

            # Position sizing based on risk
            if risk_per_contract > 0:
                max_contracts = spec.risk.dollars_per_trade / risk_per_contract
                qty = round_quantity(max_contracts)
            else:
                qty = 1.0

            if qty > 0:
                action = (
                    IntentAction.OPEN_LONG
                    if setup.direction == BiasDirection.BULLISH
                    else IntentAction.OPEN_SHORT
                )

                intents.append(
                    make_intent(
                        action=action,
                        quantity=qty,
                        price=setup.entry_price,
                        stop_loss=setup.stop_price,
                        take_profit=setup.target_price,
                        signal_strength=setup.confidence,
                        reason=f"Unicorn {setup.direction.value}: "
                        f"FVG@{setup.entry_price:.2f}, "
                        f"stop={setup.risk_points:.1f}pt, "
                        f"conf={setup.confidence:.2f}",
                    )
                )
                signals.append(f"unicorn_entry_{setup.direction.value}")
            else:
                signals.append("entry_skipped_zero_qty")

        # Build metadata
        metadata: dict[str, Any] = {
            "last_price": last_price,
            "htf_bias_direction": htf_bias.final_direction.value,
            "htf_bias_confidence": htf_bias.final_confidence,
            "htf_alignment_score": htf_bias.alignment_score,
            "at_max_positions": at_max_positions,
            "in_macro_window": is_in_macro_window(snapshot.ts, config.session_profile),
        }

        if setup:
            metadata.update(
                {
                    "setup_direction": setup.direction.value,
                    "setup_confidence": setup.confidence,
                    "entry_price": setup.entry_price,
                    "stop_price": setup.stop_price,
                    "target_price": setup.target_price,
                    "risk_points": setup.risk_points,
                    "all_criteria_met": setup.all_criteria_met,
                    "entry_decision": setup.criteria_score.decide_entry(
                        min_scored=config.min_scored_criteria
                    ),
                    "scored_count": setup.criteria_score.scored_count,
                    "mandatory_met": setup.criteria_score.mandatory_met,
                    "has_sweep": setup.liquidity_sweep is not None,
                    "has_fvg": setup.entry_fvg is not None,
                    "has_block": setup.entry_block is not None,
                    "has_ltf_fvg": setup.ltf_fvg is not None,
                    "has_mss": setup.mss is not None,
                    "confidence_threshold": config.min_confidence,
                    "scored_missing": setup.criteria_score.scored_missing,
                }
            )

        return StrategyEvaluation(
            spec_id=str(spec.instance_id),
            symbol=symbol,
            ts=snapshot.ts,
            intents=intents,
            signals=signals,
            metadata=metadata,
            evaluation_id=evaluation_id,
        )

    # Has position - check for exit conditions
    # In production, this would check for target/stop hits
    signals.append("position_held")
    return StrategyEvaluation(
        spec_id=str(spec.instance_id),
        symbol=symbol,
        ts=snapshot.ts,
        intents=[],
        signals=signals,
        metadata={
            "last_price": last_price,
            "position_qty": position.quantity if position else 0,
            "at_max_positions": at_max_positions,
        },
        evaluation_id=evaluation_id,
    )
