"""HTF (Daily) bias engine: swing detection, MSB, range, order blocks."""

from __future__ import annotations

from collections import deque
from typing import Optional

from .types import (
    Bias,
    HTFState,
    ICTBlueprintParams,
    OrderBlock,
    SwingPoint,
    TradingRange,
)


# ---------------------------------------------------------------------------
# SwingDetector — incremental, O(1) per bar
# ---------------------------------------------------------------------------


class SwingDetector:
    """Rolling-window swing detector using strict inequality.

    A swing high at index *c* is confirmed when the center bar's high
    is **strictly** greater than every other bar in the window.
    Flat bars (equal highs/lows) do NOT qualify.
    """

    def __init__(self, lookback: int = 1):
        self.lookback = lookback
        self.window_size = 2 * lookback + 1
        self._buf: deque[tuple[int, int, float, float, float, float]] = deque(
            maxlen=self.window_size
        )  # (index, ts, open, high, low, close)

    def push(
        self,
        index: int,
        timestamp: int,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> list[SwingPoint]:
        """Push a bar and return any confirmed swing points (0, 1, or 2)."""
        self._buf.append((index, timestamp, open_, high, low, close))
        if len(self._buf) < self.window_size:
            return []

        swings: list[SwingPoint] = []
        center = self.lookback
        c_idx, c_ts, _, c_high, c_low, _ = self._buf[center]

        # Check swing high — strict inequality on all surrounding bars
        is_sh = True
        for i in range(self.window_size):
            if i == center:
                continue
            if self._buf[i][3] >= c_high:  # [3] = high
                is_sh = False
                break
        if is_sh:
            swings.append(
                SwingPoint(index=c_idx, timestamp=c_ts, price=c_high, is_high=True)
            )

        # Check swing low — strict inequality
        is_sl = True
        for i in range(self.window_size):
            if i == center:
                continue
            if self._buf[i][4] <= c_low:  # [4] = low
                is_sl = False
                break
        if is_sl:
            swings.append(
                SwingPoint(index=c_idx, timestamp=c_ts, price=c_low, is_high=False)
            )

        return swings


# ---------------------------------------------------------------------------
# MSB detection
# ---------------------------------------------------------------------------


def check_msb(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    bar_close: float,
    current_bias: Bias,
) -> Optional[tuple[Bias, SwingPoint]]:
    """Check if *bar_close* breaks the most recent swing high or low.

    Returns (new_bias, broken_swing) or None if no MSB.
    Bullish MSB: close > most recent swing high price.
    Bearish MSB: close < most recent swing low price.
    If both would trigger (rare), prefer the one opposing current bias.
    """
    bullish_msb = None
    bearish_msb = None

    if swing_highs:
        sh = swing_highs[-1]
        if bar_close > sh.price:
            bullish_msb = (Bias.BULLISH, sh)

    if swing_lows:
        sl = swing_lows[-1]
        if bar_close < sl.price:
            bearish_msb = (Bias.BEARISH, sl)

    if bullish_msb and bearish_msb:
        # Both triggered — prefer opposing current bias
        if current_bias == Bias.BULLISH:
            return bearish_msb
        return bullish_msb

    return bullish_msb or bearish_msb


# ---------------------------------------------------------------------------
# Range definition
# ---------------------------------------------------------------------------


def define_range(
    msb_type: Bias,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    msb_swing: SwingPoint,
) -> Optional[TradingRange]:
    """Define trading range after an MSB.

    Bullish MSB: range_high = broken swing high, range_low = most immediate swing low preceding it.
    Bearish MSB: range_low = broken swing low, range_high = most immediate swing high preceding it.
    """
    if msb_type == Bias.BULLISH:
        range_high = msb_swing.price
        # Find most recent swing low before the MSB swing
        anchor = _find_preceding_swing(swing_lows, msb_swing.index)
        if anchor is None:
            return None
        range_low = anchor.price
    else:
        range_low = msb_swing.price
        anchor = _find_preceding_swing(swing_highs, msb_swing.index)
        if anchor is None:
            return None
        range_high = anchor.price

    if range_high <= range_low:
        return None

    midpoint = range_low + (range_high - range_low) * 0.5
    return TradingRange(
        high=range_high, low=range_low, midpoint=midpoint, bias=msb_type
    )


def _find_preceding_swing(
    swings: list[SwingPoint], before_index: int
) -> Optional[SwingPoint]:
    """Find the most recent swing point with index < before_index."""
    for sp in reversed(swings):
        if sp.index < before_index:
            return sp
    return None


# ---------------------------------------------------------------------------
# Order block identification
# ---------------------------------------------------------------------------


def identify_order_block(
    bars: list[tuple[int, int, float, float, float, float]],
    msb_bar_index: int,
    msb_type: Bias,
    ob_candles: int,
    anchor_swing: SwingPoint,
    daily_index: int,
) -> Optional[OrderBlock]:
    """Identify the order block before the impulse that caused the MSB.

    bars: list of (index, ts, open, high, low, close) up to and including msb_bar_index.
    For bullish MSB: find last N bearish (close < open) candles before the impulse.
    For bearish MSB: find last N bullish (close > open) candles before the impulse.
    """
    if not bars:
        return None

    # Find position of msb_bar in bars list
    msb_pos = None
    for i, b in enumerate(bars):
        if b[0] == msb_bar_index:
            msb_pos = i
            break
    if msb_pos is None or msb_pos == 0:
        return None

    # Scan backwards from the bar before MSB bar for opposing candles.
    # Skip impulse candles (same direction as MSB) to reach the OB candles.
    opposing: list[tuple[int, int, float, float, float, float]] = []
    found_opposing = False
    for i in range(msb_pos - 1, -1, -1):
        b = bars[i]
        b_open, b_close = b[2], b[5]
        if msb_type == Bias.BULLISH:
            is_opposing = b_close < b_open
        else:
            is_opposing = b_close > b_open

        if is_opposing:
            found_opposing = True
            opposing.append(b)
            if len(opposing) >= ob_candles:
                break
        elif found_opposing:
            # Already collecting opposing candles — stop at first non-opposing
            break
        # else: skip impulse candles before we find opposing
    if not opposing:
        return None

    ob_top = max(b[3] for b in opposing)  # max high
    ob_bottom = min(b[4] for b in opposing)  # min low
    ob_start = min(b[0] for b in opposing)
    ob_end = max(b[0] for b in opposing)
    side = "long" if msb_type == Bias.BULLISH else "short"
    ob_id = (msb_bar_index, ob_start, ob_end, side)

    return OrderBlock(
        top=ob_top,
        bottom=ob_bottom,
        bias=msb_type,
        ob_id=ob_id,
        anchor_swing=anchor_swing,
        msb_bar_index=msb_bar_index,
        created_at_daily_index=daily_index,
    )


# ---------------------------------------------------------------------------
# Premium / discount
# ---------------------------------------------------------------------------


def is_in_discount(
    price: float, trading_range: TradingRange, threshold: float = 0.5
) -> bool:
    """True if price is below the threshold level of the range (discount for longs)."""
    level = trading_range.low + (trading_range.high - trading_range.low) * threshold
    return price < level


def is_in_premium(
    price: float, trading_range: TradingRange, threshold: float = 0.5
) -> bool:
    """True if price is above the threshold level of the range (premium for shorts)."""
    level = trading_range.low + (trading_range.high - trading_range.low) * threshold
    return price > level


# ---------------------------------------------------------------------------
# OB invalidation
# ---------------------------------------------------------------------------


def invalidate_obs(
    state: HTFState,
    bar_close: float,
    current_daily_index: int,
    max_ob_age_bars: int,
) -> None:
    """Mark OBs as invalidated based on boundary breach, bias flip, or expiry."""
    for ob in state.active_obs:
        if ob.invalidated:
            continue
        # Boundary breach: close beyond the far boundary
        if ob.bias == Bias.BULLISH and bar_close < ob.bottom:
            ob.invalidated = True
        elif ob.bias == Bias.BEARISH and bar_close > ob.top:
            ob.invalidated = True
        # Bias flip: OB bias no longer matches HTF bias
        elif ob.bias != state.bias and state.bias != Bias.NEUTRAL:
            ob.invalidated = True
        # Expiry
        elif (current_daily_index - ob.created_at_daily_index) > max_ob_age_bars:
            ob.invalidated = True


# ---------------------------------------------------------------------------
# Main HTF update
# ---------------------------------------------------------------------------


def update_htf(
    state: HTFState,
    bar_index: int,
    bar: tuple[int, int, float, float, float, float],
    params: ICTBlueprintParams,
    swing_detector: SwingDetector,
    bars_history: list[tuple[int, int, float, float, float, float]],
) -> None:
    """Process one daily bar, mutating *state* in place.

    bar: (index, timestamp, open, high, low, close)
    """
    idx, ts, open_, high, low, close = bar

    # 1. Swing detection
    swings = swing_detector.push(idx, ts, open_, high, low, close)
    for sp in swings:
        if sp.is_high:
            state.swing_highs.append(sp)
        else:
            state.swing_lows.append(sp)

    # 2. MSB check
    msb_result = check_msb(state.swing_highs, state.swing_lows, close, state.bias)
    if msb_result is not None:
        new_bias, msb_swing = msb_result
        state.bias = new_bias
        state.last_msb_bar_index = bar_index

        # 3. Define range
        new_range = define_range(
            new_bias, state.swing_highs, state.swing_lows, msb_swing
        )
        if new_range is not None:
            state.current_range = new_range

        # 4. Identify order block
        ob = identify_order_block(
            bars=bars_history,
            msb_bar_index=bar_index,
            msb_type=new_bias,
            ob_candles=params.ob_candles,
            anchor_swing=msb_swing,
            daily_index=bar_index,
        )
        if ob is not None:
            state.active_obs.append(ob)

    # 5. Invalidate stale OBs
    invalidate_obs(state, close, bar_index, params.max_ob_age_bars)
