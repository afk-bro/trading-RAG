"""
ICT (Inner Circle Trader) Pattern Detection.

Implements detection for:
- Fair Value Gaps (FVG) / Imbalances
- Breaker Blocks
- Mitigation Blocks
- Liquidity Sweeps (stop hunts)
- Market Structure Shift (MSS)
- Displacement moves
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from app.services.strategy.models import OHLCVBar


class FVGType(str, Enum):
    """Fair Value Gap direction."""

    BULLISH = "bullish"  # Gap up (price inefficiency to fill from below)
    BEARISH = "bearish"  # Gap down (price inefficiency to fill from above)


class BlockType(str, Enum):
    """Order block type."""

    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class FairValueGap:
    """
    Fair Value Gap (Imbalance) detection result.

    An FVG occurs when candle N's body doesn't overlap with candle N-2's body,
    creating a price inefficiency between candle N-1's high/low.
    """

    fvg_type: FVGType
    gap_high: float  # Upper bound of the gap
    gap_low: float  # Lower bound of the gap
    gap_size: float  # Size in price units
    bar_index: int  # Index of the middle candle (N-1)
    timestamp: datetime
    filled: bool = False  # Has price returned to fully (100%) fill this gap?
    fill_percent: float = 0.0  # How much of gap has been filled (0-1)

    @property
    def midpoint(self) -> float:
        """Midpoint of the gap (potential entry level)."""
        return (self.gap_high + self.gap_low) / 2

    def invalidated(self, max_fill_pct: float = 1.0) -> bool:
        """Whether the FVG is invalidated (fill exceeds threshold).

        Args:
            max_fill_pct: Maximum fill percentage before FVG is considered
                          invalidated. 1.0 means only 100% fill invalidates
                          (same as ``filled``). 0.75 means 75%+ fill invalidates.

        Returns:
            True if the FVG should no longer be used for entries.
        """
        return self.fill_percent >= max_fill_pct


@dataclass
class BreakerBlock:
    """
    Breaker Block detection result.

    A breaker block forms when price breaks through an order block,
    failing to hold. The failed order block becomes a breaker.
    """

    block_type: BlockType
    high: float
    low: float
    bar_index: int
    timestamp: datetime
    broken_at_index: Optional[int] = None
    broken_timestamp: Optional[datetime] = None
    mitigated: bool = False


@dataclass
class MitigationBlock:
    """
    Mitigation Block detection result.

    An area where institutional orders were filled but not completely.
    Price returning to this zone may find support/resistance.
    """

    block_type: BlockType
    high: float
    low: float
    bar_index: int
    timestamp: datetime
    strength: float = 1.0  # Relative strength based on volume/move


@dataclass
class LiquiditySweep:
    """
    Liquidity Sweep (Stop Hunt) detection result.

    Occurs when price briefly exceeds a swing high/low (taking stops)
    then quickly reverses, showing institutional accumulation/distribution.
    """

    sweep_type: str  # "high" or "low"
    swept_level: float  # The level that was swept
    sweep_high: float  # Highest point of sweep
    sweep_low: float  # Lowest point of sweep
    bar_index: int
    timestamp: datetime
    reversal_strength: float = 0.0  # How strong was the reversal


@dataclass
class MarketStructureShift:
    """
    Market Structure Shift (MSS) / Break of Structure (BOS) detection.

    Identifies when the market changes direction by breaking
    a significant swing high (for bullish MSS) or swing low (for bearish MSS).
    """

    shift_type: str  # "bullish" or "bearish"
    broken_level: float  # The swing level that was broken
    break_bar_index: int
    break_timestamp: datetime
    displacement: bool = False  # Was this a displacement move?
    displacement_size: float = 0.0  # Size of displacement in ATR multiples


@dataclass
class DisplacementMove:
    """
    Displacement move detection result.

    A large, impulsive move showing strong institutional participation.
    Typically 2-3x ATR in a single candle.
    """

    direction: str  # "bullish" or "bearish"
    bar_index: int
    timestamp: datetime
    move_size: float  # Size of the move
    atr_multiple: float  # Move size as multiple of ATR
    creates_fvg: bool = False  # Did this move create an FVG?


def detect_fvgs(
    bars: list[OHLCVBar],
    min_gap_size: float = 0.0,
    lookback: int = 50,
    min_gap_atr_mult: Optional[float] = None,
    atr_period: int = 14,
    as_of_ts: Optional[datetime] = None,
) -> list[FairValueGap]:
    """
    Detect Fair Value Gaps in price data.

    An FVG forms when there's no overlap between:
    - Bullish FVG: bar[i-2].high < bar[i].low (gap up)
    - Bearish FVG: bar[i-2].low > bar[i].high (gap down)

    Args:
        bars: OHLCV bars (oldest to newest)
        min_gap_size: Minimum gap size in absolute price units (legacy)
        lookback: Number of bars to analyze
        min_gap_atr_mult: Minimum gap size as ATR multiple (overrides min_gap_size)
        atr_period: ATR period for volatility normalization
        as_of_ts: Causal cutoff — only consider bars with ts <= as_of_ts.
                  None means use all bars (backward compatible).

    Returns:
        List of detected FVGs, newest first
    """
    if as_of_ts is not None:
        bars = [b for b in bars if b.ts <= as_of_ts]

    if len(bars) < 3:
        return []

    fvgs: list[FairValueGap] = []
    start_idx = max(2, len(bars) - lookback)

    # Calculate ATR if using volatility-normalized threshold
    atr_values: Optional[list[float]] = None
    if min_gap_atr_mult is not None:
        atr_values = _calculate_atr(bars, atr_period)

    for i in range(start_idx, len(bars)):
        bar_minus_2 = bars[i - 2]
        bar_minus_1 = bars[i - 1]  # The candle that creates the gap
        bar_current = bars[i]

        # Determine minimum gap size (ATR-based or absolute)
        if atr_values is not None and min_gap_atr_mult is not None:
            current_atr = atr_values[i] if i < len(atr_values) else atr_values[-1]
            effective_min_gap = current_atr * min_gap_atr_mult
        else:
            effective_min_gap = min_gap_size

        # Bullish FVG: gap up (bar_minus_2.high < bar_current.low)
        if bar_minus_2.high < bar_current.low:
            gap_high = bar_current.low
            gap_low = bar_minus_2.high
            gap_size = gap_high - gap_low

            if gap_size >= effective_min_gap:
                fvgs.append(
                    FairValueGap(
                        fvg_type=FVGType.BULLISH,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        gap_size=gap_size,
                        bar_index=i - 1,
                        timestamp=bar_minus_1.ts,
                    )
                )

        # Bearish FVG: gap down (bar_minus_2.low > bar_current.high)
        elif bar_minus_2.low > bar_current.high:
            gap_high = bar_minus_2.low
            gap_low = bar_current.high
            gap_size = gap_high - gap_low

            if gap_size >= effective_min_gap:
                fvgs.append(
                    FairValueGap(
                        fvg_type=FVGType.BEARISH,
                        gap_high=gap_high,
                        gap_low=gap_low,
                        gap_size=gap_size,
                        bar_index=i - 1,
                        timestamp=bar_minus_1.ts,
                    )
                )

    # Check if FVGs have been filled
    if fvgs and len(bars) > start_idx:
        _check_fvg_fills(fvgs, bars, as_of_ts=as_of_ts)

    return list(reversed(fvgs))  # Return newest first


def _check_fvg_fills(
    fvgs: list[FairValueGap],
    bars: list[OHLCVBar],
    as_of_ts: Optional[datetime] = None,
) -> None:
    """Check if FVGs have been filled by subsequent price action."""
    for fvg in fvgs:
        # Look at bars after the FVG formed
        for i in range(fvg.bar_index + 2, len(bars)):
            bar = bars[i]

            if as_of_ts is not None and bar.ts > as_of_ts:
                break

            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish FVG filled when price trades down into it
                if bar.low <= fvg.gap_high:
                    fill_depth = min(fvg.gap_high - bar.low, fvg.gap_size)
                    fvg.fill_percent = max(fvg.fill_percent, fill_depth / fvg.gap_size)
                    if bar.low <= fvg.gap_low:
                        fvg.filled = True
                        break
            else:
                # Bearish FVG filled when price trades up into it
                if bar.high >= fvg.gap_low:
                    fill_depth = min(bar.high - fvg.gap_low, fvg.gap_size)
                    fvg.fill_percent = max(fvg.fill_percent, fill_depth / fvg.gap_size)
                    if bar.high >= fvg.gap_high:
                        fvg.filled = True
                        break


def detect_breaker_blocks(
    bars: list[OHLCVBar],
    swing_lookback: int = 5,
    lookback: int = 50,
) -> list[BreakerBlock]:
    """
    Detect Breaker Blocks.

    A breaker block is a failed order block - an area where price:
    1. Found support/resistance (order block)
    2. Later broke through that level (invalidating the order block)
    3. The broken level becomes potential future S/R (breaker)

    Args:
        bars: OHLCV bars
        swing_lookback: Bars to look back for swing detection
        lookback: Total bars to analyze

    Returns:
        List of breaker blocks
    """
    if len(bars) < swing_lookback * 2:
        return []

    breakers: list[BreakerBlock] = []
    start_idx = max(swing_lookback, len(bars) - lookback)

    # Find swing points
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    for i in range(start_idx, len(bars) - swing_lookback):
        bar = bars[i]

        # Check if this is a swing high
        is_swing_high = all(
            bar.high >= bars[i - j].high and bar.high >= bars[i + j].high
            for j in range(1, swing_lookback + 1)
            if i - j >= 0 and i + j < len(bars)
        )
        if is_swing_high:
            swing_highs.append((i, bar.high))

        # Check if this is a swing low
        is_swing_low = all(
            bar.low <= bars[i - j].low and bar.low <= bars[i + j].low
            for j in range(1, swing_lookback + 1)
            if i - j >= 0 and i + j < len(bars)
        )
        if is_swing_low:
            swing_lows.append((i, bar.low))

    # Check for breaks of swing highs (bullish breakers)
    for sh_idx, sh_level in swing_highs:
        for i in range(sh_idx + 1, len(bars)):
            if bars[i].close > sh_level:
                # Swing high broken - becomes bullish breaker
                breakers.append(
                    BreakerBlock(
                        block_type=BlockType.BULLISH,
                        high=bars[sh_idx].high,
                        low=bars[sh_idx].low,
                        bar_index=sh_idx,
                        timestamp=bars[sh_idx].ts,
                        broken_at_index=i,
                        broken_timestamp=bars[i].ts,
                    )
                )
                break

    # Check for breaks of swing lows (bearish breakers)
    for sl_idx, sl_level in swing_lows:
        for i in range(sl_idx + 1, len(bars)):
            if bars[i].close < sl_level:
                # Swing low broken - becomes bearish breaker
                breakers.append(
                    BreakerBlock(
                        block_type=BlockType.BEARISH,
                        high=bars[sl_idx].high,
                        low=bars[sl_idx].low,
                        bar_index=sl_idx,
                        timestamp=bars[sl_idx].ts,
                        broken_at_index=i,
                        broken_timestamp=bars[i].ts,
                    )
                )
                break

    return breakers


def detect_mitigation_blocks(
    bars: list[OHLCVBar],
    lookback: int = 50,
    volume_threshold: float = 1.5,
) -> list[MitigationBlock]:
    """
    Detect Mitigation Blocks.

    Areas where significant volume occurred, indicating institutional
    order flow that may not have been completely filled.

    Args:
        bars: OHLCV bars
        lookback: Number of bars to analyze
        volume_threshold: Volume must be this multiple of average

    Returns:
        List of mitigation blocks
    """
    if len(bars) < 20:
        return []

    blocks: list[MitigationBlock] = []
    start_idx = max(0, len(bars) - lookback)

    # Calculate average volume
    volumes = [b.volume for b in bars[start_idx:] if b.volume > 0]
    if not volumes:
        return []
    avg_volume = sum(volumes) / len(volumes)

    for i in range(start_idx, len(bars)):
        bar = bars[i]

        # Look for high volume candles
        if bar.volume > avg_volume * volume_threshold:
            # Bullish candle with high volume
            if bar.close > bar.open:
                blocks.append(
                    MitigationBlock(
                        block_type=BlockType.BULLISH,
                        high=bar.high,
                        low=bar.low,
                        bar_index=i,
                        timestamp=bar.ts,
                        strength=bar.volume / avg_volume,
                    )
                )
            # Bearish candle with high volume
            elif bar.close < bar.open:
                blocks.append(
                    MitigationBlock(
                        block_type=BlockType.BEARISH,
                        high=bar.high,
                        low=bar.low,
                        bar_index=i,
                        timestamp=bar.ts,
                        strength=bar.volume / avg_volume,
                    )
                )

    return blocks


def detect_liquidity_sweeps(
    bars: list[OHLCVBar],
    swing_lookback: int = 5,
    lookback: int = 50,
    min_reversal_percent: float = 0.3,
) -> list[LiquiditySweep]:
    """
    Detect Liquidity Sweeps (Stop Hunts).

    Occurs when price briefly exceeds a swing high/low (taking out stops)
    then closes back inside the range, indicating institutional accumulation.

    Args:
        bars: OHLCV bars
        swing_lookback: Bars for swing detection
        lookback: Total bars to analyze
        min_reversal_percent: Minimum reversal as percent of candle range

    Returns:
        List of liquidity sweeps
    """
    if len(bars) < swing_lookback * 2:
        return []

    sweeps: list[LiquiditySweep] = []
    start_idx = max(swing_lookback, len(bars) - lookback)

    # Find swing highs and lows
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []

    for i in range(start_idx, len(bars) - 1):
        bar = bars[i]

        # Swing high detection
        is_swing_high = True
        for j in range(1, min(swing_lookback + 1, i + 1)):
            if i - j >= 0 and bars[i - j].high > bar.high:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append((i, bar.high))

        # Swing low detection
        is_swing_low = True
        for j in range(1, min(swing_lookback + 1, i + 1)):
            if i - j >= 0 and bars[i - j].low < bar.low:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append((i, bar.low))

    # Check for sweeps of swing highs
    for sh_idx, sh_level in swing_highs:
        for i in range(sh_idx + 1, len(bars)):
            bar = bars[i]
            candle_range = bar.high - bar.low

            # Sweep high: wick above swing high but close below
            if bar.high > sh_level and bar.close < sh_level and candle_range > 0:
                reversal = (bar.high - bar.close) / candle_range
                if reversal >= min_reversal_percent:
                    sweeps.append(
                        LiquiditySweep(
                            sweep_type="high",
                            swept_level=sh_level,
                            sweep_high=bar.high,
                            sweep_low=bar.low,
                            bar_index=i,
                            timestamp=bar.ts,
                            reversal_strength=reversal,
                        )
                    )
                    break

    # Check for sweeps of swing lows
    for sl_idx, sl_level in swing_lows:
        for i in range(sl_idx + 1, len(bars)):
            bar = bars[i]
            candle_range = bar.high - bar.low

            # Sweep low: wick below swing low but close above
            if bar.low < sl_level and bar.close > sl_level and candle_range > 0:
                reversal = (bar.close - bar.low) / candle_range
                if reversal >= min_reversal_percent:
                    sweeps.append(
                        LiquiditySweep(
                            sweep_type="low",
                            swept_level=sl_level,
                            sweep_high=bar.high,
                            sweep_low=bar.low,
                            bar_index=i,
                            timestamp=bar.ts,
                            reversal_strength=reversal,
                        )
                    )
                    break

    return sweeps


def detect_mss(
    bars: list[OHLCVBar],
    swing_lookback: int = 5,
    lookback: int = 50,
    atr_period: int = 14,
    displacement_atr_mult: float = 2.0,
) -> list[MarketStructureShift]:
    """
    Detect Market Structure Shifts (MSS) / Break of Structure (BOS).

    Identifies when market direction changes by breaking significant
    swing highs or lows with conviction.

    Args:
        bars: OHLCV bars
        swing_lookback: Bars for swing detection
        lookback: Total bars to analyze
        atr_period: Period for ATR calculation
        displacement_atr_mult: ATR multiple to qualify as displacement

    Returns:
        List of market structure shifts
    """
    if len(bars) < max(swing_lookback * 2, atr_period):
        return []

    shifts: list[MarketStructureShift] = []
    start_idx = max(swing_lookback, len(bars) - lookback)

    # Calculate ATR
    atr = _calculate_atr(bars, atr_period)

    # Track recent swing points
    recent_swing_high: Optional[tuple[int, float]] = None
    recent_swing_low: Optional[tuple[int, float]] = None
    trend = "neutral"  # Track current trend

    for i in range(start_idx, len(bars)):
        bar = bars[i]

        # Update swing points
        if i >= swing_lookback:
            # Check for new swing high
            is_swing_high = all(
                bars[i - swing_lookback].high >= bars[i - swing_lookback - j].high
                for j in range(1, min(swing_lookback, i - swing_lookback) + 1)
            )
            if is_swing_high:
                recent_swing_high = (i - swing_lookback, bars[i - swing_lookback].high)

            # Check for new swing low
            is_swing_low = all(
                bars[i - swing_lookback].low <= bars[i - swing_lookback - j].low
                for j in range(1, min(swing_lookback, i - swing_lookback) + 1)
            )
            if is_swing_low:
                recent_swing_low = (i - swing_lookback, bars[i - swing_lookback].low)

        # Check for bullish MSS (break of swing high in downtrend)
        if recent_swing_high and trend in ("bearish", "neutral"):
            if bar.close > recent_swing_high[1]:
                move_size = bar.close - bar.open
                atr_at_bar = atr[i] if i < len(atr) else atr[-1]
                is_displacement = abs(move_size) > atr_at_bar * displacement_atr_mult

                shifts.append(
                    MarketStructureShift(
                        shift_type="bullish",
                        broken_level=recent_swing_high[1],
                        break_bar_index=i,
                        break_timestamp=bar.ts,
                        displacement=is_displacement,
                        displacement_size=(
                            abs(move_size) / atr_at_bar if atr_at_bar > 0 else 0
                        ),
                    )
                )
                trend = "bullish"
                recent_swing_high = None

        # Check for bearish MSS (break of swing low in uptrend)
        if recent_swing_low and trend in ("bullish", "neutral"):
            if bar.close < recent_swing_low[1]:
                move_size = bar.open - bar.close
                atr_at_bar = atr[i] if i < len(atr) else atr[-1]
                is_displacement = abs(move_size) > atr_at_bar * displacement_atr_mult

                shifts.append(
                    MarketStructureShift(
                        shift_type="bearish",
                        broken_level=recent_swing_low[1],
                        break_bar_index=i,
                        break_timestamp=bar.ts,
                        displacement=is_displacement,
                        displacement_size=(
                            abs(move_size) / atr_at_bar if atr_at_bar > 0 else 0
                        ),
                    )
                )
                trend = "bearish"
                recent_swing_low = None

    return shifts


def detect_displacement(
    bars: list[OHLCVBar],
    atr_period: int = 14,
    atr_multiple: float = 2.0,
    lookback: int = 20,
) -> list[DisplacementMove]:
    """
    Detect Displacement moves (large impulsive candles).

    Displacement indicates strong institutional participation and
    often creates FVGs that can be used for entries.

    Args:
        bars: OHLCV bars
        atr_period: Period for ATR calculation
        atr_multiple: Minimum candle size in ATR multiples
        lookback: Number of bars to analyze

    Returns:
        List of displacement moves
    """
    if len(bars) < atr_period:
        return []

    displacements: list[DisplacementMove] = []
    atr = _calculate_atr(bars, atr_period)
    start_idx = max(atr_period, len(bars) - lookback)

    # Get FVGs for cross-reference
    fvgs = detect_fvgs(bars, lookback=lookback)
    fvg_indices = {f.bar_index for f in fvgs}

    for i in range(start_idx, len(bars)):
        bar = bars[i]
        atr_at_bar = atr[i] if i < len(atr) else atr[-1]

        if atr_at_bar <= 0:
            continue

        candle_body = abs(bar.close - bar.open)
        atr_mult = candle_body / atr_at_bar

        if atr_mult >= atr_multiple:
            direction = "bullish" if bar.close > bar.open else "bearish"

            # Check if this displacement created an FVG
            creates_fvg = i in fvg_indices or (i - 1) in fvg_indices or (i + 1) in fvg_indices

            displacements.append(
                DisplacementMove(
                    direction=direction,
                    bar_index=i,
                    timestamp=bar.ts,
                    move_size=candle_body,
                    atr_multiple=atr_mult,
                    creates_fvg=creates_fvg,
                )
            )

    return displacements


def _calculate_atr(bars: list[OHLCVBar], period: int = 14) -> list[float]:
    """
    Calculate Average True Range.

    Args:
        bars: OHLCV bars
        period: ATR period

    Returns:
        List of ATR values (same length as bars)
    """
    if len(bars) < 2:
        return [0.0] * len(bars)

    true_ranges: list[float] = [bars[0].high - bars[0].low]

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

    # Calculate ATR using EMA
    atr: list[float] = []
    multiplier = 2 / (period + 1)

    for i, tr in enumerate(true_ranges):
        if i < period:
            # Use SMA for initial values
            atr.append(sum(true_ranges[: i + 1]) / (i + 1))
        else:
            # EMA
            atr.append((tr - atr[-1]) * multiplier + atr[-1])

    return atr
