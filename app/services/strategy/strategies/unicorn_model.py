"""
ICT Unicorn Model Strategy.

A comprehensive ICT (Inner Circle Trader) strategy implementation
that combines multiple confluence factors for high-probability entries.

Entry Criteria (scored, configurable threshold):
1. HTF Bias confirms direction (Daily/4H/1H alignment)
2. Liquidity sweep has occurred (stop hunt)
3. FVG (Fair Value Gap) is present
4. Breaker or Mitigation Block provides entry zone
5. 5-15M FVG/DOL (Draw on Liquidity) confirms setup
6. Displacement / MSS (Market Structure Shift) shows intent
7. Stop placement within ATR-based limit (risk management)
8. Valid macro time window (session timing)

Key improvements:
- Volatility-normalized thresholds (FVG size, stop distance use ATR)
- Soft scoring: enter at score >= threshold (default 6/8)
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
from typing import Optional
from uuid import UUID

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
    detect_displacement,
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
# Configuration
# =============================================================================

# Quantity rounding (8 decimals for crypto, fewer for futures)
QUANTITY_DECIMALS = 2  # NQ/ES use 2 decimal places


class SessionProfile(str, Enum):
    """Trading session window profiles."""
    STRICT = "strict"   # NY AM only
    NORMAL = "normal"   # London + NY AM
    WIDE = "wide"       # London + NY AM + NY PM


# Session windows by profile
SESSION_WINDOWS = {
    SessionProfile.STRICT: [
        (time(9, 30), time(11, 0)),   # NY AM only
    ],
    SessionProfile.NORMAL: [
        (time(3, 0), time(4, 0)),     # London
        (time(9, 30), time(11, 0)),   # NY AM
    ],
    SessionProfile.WIDE: [
        (time(3, 0), time(4, 0)),     # London
        (time(9, 30), time(11, 0)),   # NY AM
        (time(13, 30), time(15, 0)),  # NY PM
        (time(19, 0), time(20, 0)),   # Asia
    ],
}

# Legacy compatibility
MACRO_WINDOWS = SESSION_WINDOWS[SessionProfile.WIDE]


@dataclass
class UnicornConfig:
    """Configuration for Unicorn Model strategy."""

    # Scoring: minimum scored criteria to enter (out of 5 scored items)
    # Mandatory criteria (htf_bias, stop_valid, macro_window) always required.
    min_scored_criteria: int = 3

    # Confidence gate (None = metric-only, not used for entry filtering)
    min_confidence: Optional[float] = None

    # Session
    session_profile: SessionProfile = SessionProfile.NORMAL

    # ATR-based thresholds
    atr_period: int = 14
    fvg_min_atr_mult: float = 0.3      # FVG must be >= 0.3 * ATR
    stop_max_atr_mult: float = 3.0     # Stop must be <= 3.0 * ATR

    # Stop placement buffer (handles beyond FVG/sweep edge)
    stop_buffer_handles: float = 2.0

    # Legacy absolute thresholds (used if ATR not available)
    max_stop_handles_nq: float = 30.0
    max_stop_handles_es: float = 10.0

    # Point values
    point_value_nq: float = 20.0
    point_value_es: float = 50.0

    def __post_init__(self):
        if not (0 <= self.min_scored_criteria <= 5):
            raise ValueError(
                f"min_scored_criteria must be 0-5 (got {self.min_scored_criteria}). "
                f"There are only 5 scored criteria; mandatory gates are always enforced."
            )
        if self.min_confidence is not None and not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be 0.0-1.0 or None (got {self.min_confidence})"
            )


# Default config
DEFAULT_CONFIG = UnicornConfig()

# Risk parameters for NQ/ES (legacy, for backward compatibility)
MAX_STOP_HANDLES_NQ = 30  # 30 points max stop for NQ
MAX_STOP_HANDLES_ES = 10  # 10 points max stop for ES
POINT_VALUE_NQ = 20.0  # $20 per point for NQ
POINT_VALUE_ES = 50.0  # $50 per point for ES


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

    @property
    def score(self) -> int:
        """Total criteria met (0-8)."""
        return sum([
            self.htf_bias,
            self.liquidity_sweep,
            self.htf_fvg,
            self.breaker_block,
            self.ltf_fvg,
            self.mss,
            self.stop_valid,
            self.macro_window,
        ])

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
        return missing

    @property
    def mandatory_met(self) -> bool:
        """All 3 mandatory criteria must pass: htf_bias, stop_valid, macro_window."""
        return self.htf_bias and self.stop_valid and self.macro_window

    @property
    def scored_count(self) -> int:
        """Count of passed scored criteria (out of 5)."""
        return sum([
            self.liquidity_sweep,
            self.htf_fvg,
            self.breaker_block,
            self.ltf_fvg,
            self.mss,
        ])

    def decide_entry(self, min_scored: int = 3) -> bool:
        """
        Canonical entry gate used by both live evaluator and backtest.

        Requires all 3 mandatory criteria AND at least min_scored of 5 scored criteria.
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
    entry_price: float           # Theoretical signal price (FVG midpoint)
    entry_price_model: float     # Same as entry_price; backtest overrides with fill_price
    stop_price: float
    target_price: float
    risk_handles: float

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
        """Check if all 8 criteria are satisfied (strict mode)."""
        return self.criteria_score.score == 8

    def meets_threshold(self, threshold: int = 6) -> bool:
        """Check if setup meets minimum criteria threshold."""
        return self.criteria_score.meets_threshold(threshold)


def round_quantity(qty: float, decimals: int = QUANTITY_DECIMALS) -> float:
    """Round DOWN to avoid over-allocation."""
    multiplier = 10**decimals
    return floor(qty * multiplier) / multiplier


def is_in_macro_window(
    ts: datetime,
    profile: SessionProfile = SessionProfile.WIDE,
) -> bool:
    """
    Check if timestamp is within a valid macro trading window.

    Args:
        ts: Timestamp to check
        profile: Session profile (STRICT, NORMAL, WIDE)

    Note: This assumes the timestamp is in ET timezone.
    In production, proper timezone conversion would be needed.
    """
    current_time = ts.time()
    windows = SESSION_WINDOWS.get(profile, MACRO_WINDOWS)

    for start, end in windows:
        if start <= current_time < end:
            return True

    return False


def _calculate_atr(
    bars: list[OHLCVBar],
    period: int = 14,
) -> list[float]:
    """
    Calculate Average True Range (ATR).

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


def get_max_stop_handles(
    symbol: str,
    atr: Optional[float] = None,
    config: Optional[UnicornConfig] = None,
) -> float:
    """
    Get maximum allowed stop distance in handles for the instrument.

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
        return config.max_stop_handles_nq
    elif "ES" in symbol_upper or "MES" in symbol_upper:
        return config.max_stop_handles_es
    else:
        return config.max_stop_handles_nq  # Default to NQ


def get_point_value(symbol: str) -> float:
    """Get point value for the instrument."""
    symbol_upper = symbol.upper()
    # Check micro contracts first (MNQ before NQ, MES before ES)
    if "MNQ" in symbol_upper:
        return POINT_VALUE_NQ / 10  # Micro NQ = $2 per point
    elif "NQ" in symbol_upper:
        return POINT_VALUE_NQ  # Full NQ = $20 per point
    elif "MES" in symbol_upper:
        return POINT_VALUE_ES / 10  # Micro ES = $5 per point
    elif "ES" in symbol_upper:
        return POINT_VALUE_ES  # Full ES = $50 per point
    else:
        return POINT_VALUE_NQ  # Default


def find_entry_zone(
    fvgs: list[FairValueGap],
    breakers: list[BreakerBlock],
    mitigations: list[MitigationBlock],
    direction: BiasDirection,
    current_price: float,
) -> tuple[Optional[FairValueGap], Optional[BreakerBlock | MitigationBlock], float]:
    """
    Find the best entry zone combining FVG with Breaker/Mitigation block.

    Returns:
        Tuple of (entry_fvg, entry_block, entry_price)
    """
    best_fvg: Optional[FairValueGap] = None
    best_block: Optional[BreakerBlock | MitigationBlock] = None
    entry_price = current_price

    # Filter FVGs by direction and unfilled status
    relevant_fvgs = [
        f
        for f in fvgs
        if not f.filled
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
        Tuple of (stop_price, target_price, risk_handles, stop_valid)
    """
    if config is None:
        config = DEFAULT_CONFIG

    max_handles = get_max_stop_handles(symbol, atr=atr, config=config)
    buf = config.stop_buffer_handles

    if direction == BiasDirection.BULLISH:
        # Stop below FVG or sweep low
        if entry_fvg:
            stop_price = entry_fvg.gap_low - buf
        elif liquidity_sweep:
            stop_price = liquidity_sweep.sweep_low - buf
        else:
            stop_price = entry_price - max_handles

        risk_handles = entry_price - stop_price

        # Target at 2R or next liquidity
        target_price = entry_price + (risk_handles * 2)

    else:  # BEARISH
        # Stop above FVG or sweep high
        if entry_fvg:
            stop_price = entry_fvg.gap_high + buf
        elif liquidity_sweep:
            stop_price = liquidity_sweep.sweep_high + buf
        else:
            stop_price = entry_price + max_handles

        risk_handles = stop_price - entry_price

        # Target at 2R
        target_price = entry_price - (risk_handles * 2)

    stop_valid = risk_handles <= max_handles

    return stop_price, target_price, risk_handles, stop_valid


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
        htf_bias.is_tradeable
        and direction != BiasDirection.NEUTRAL
        and confidence_ok
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
            risk_handles=0,
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
    ltf_atr = _calculate_atr(ltf_bars, config.atr_period)
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
        htf_fvgs, htf_breakers, htf_mitigations, direction, current_price
    )
    score.htf_fvg = entry_fvg is not None

    # 4. Breaker/Mitigation block
    score.breaker_block = entry_block is not None

    # 5. Check for MSS confirmation
    relevant_mss: Optional[MarketStructureShift] = None
    for mss in htf_mss:
        if (
            direction == BiasDirection.BULLISH and mss.shift_type == "bullish"
        ) or (
            direction == BiasDirection.BEARISH and mss.shift_type == "bearish"
        ):
            relevant_mss = mss
            break
    score.mss = relevant_mss is not None

    # 6. Check for LTF FVG (DOL confirmation)
    relevant_ltf_fvg: Optional[FairValueGap] = None
    for fvg in ltf_fvgs:
        if not fvg.filled:
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
    stop_price, target_price, risk_handles, stop_valid = calculate_stop_and_target(
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
        risk_handles=risk_handles,
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
    7. Valid stop placement (within max handles)
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
    if snapshot.is_eod and has_position:
        intents.append(
            make_intent(
                action=IntentAction.CLOSE_LONG if position.side == "long" else IntentAction.CLOSE_SHORT,
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
        setup = analyze_unicorn_setup(snapshot, htf_bias, htf_bars, ltf_bars, config=config)

        if setup is None:
            signals.append("no_setup_detected")
        elif not setup.criteria_score.decide_entry(min_scored=config.min_scored_criteria):
            # Log which criteria failed
            cs = setup.criteria_score
            if not cs.mandatory_met:
                if not cs.htf_bias:
                    if (
                        config.min_confidence is not None
                        and setup.confidence < config.min_confidence
                    ):
                        signals.append(
                            f"htf_bias_confidence_{setup.confidence:.2f}"
                            f"_below_{config.min_confidence:.2f}"
                        )
                    else:
                        signals.append("htf_bias_not_tradeable")
                if not cs.stop_valid:
                    signals.append(f"stop_too_wide_{setup.risk_handles:.1f}_handles")
                if not cs.macro_window:
                    signals.append("outside_macro_window")
            else:
                signals.append(
                    f"scored_{cs.scored_count}/5_below_{config.min_scored_criteria}"
                )
            for name in cs.missing:
                if name not in ("htf_bias", "stop_valid", "macro_window"):
                    signals.append(f"no_{name}")
        else:
            # Entry criteria met - generate entry intent
            point_value = get_point_value(symbol)
            risk_per_contract = setup.risk_handles * point_value

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
                        f"stop={setup.risk_handles:.1f}h, "
                        f"conf={setup.confidence:.2f}",
                    )
                )
                signals.append(f"unicorn_entry_{setup.direction.value}")
            else:
                signals.append("entry_skipped_zero_qty")

        # Build metadata
        metadata = {
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
                    "risk_handles": setup.risk_handles,
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
