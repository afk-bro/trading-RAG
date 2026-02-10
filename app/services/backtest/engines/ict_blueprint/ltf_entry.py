"""LTF (H1) entry logic: OB zone activation, setup state machine, breaker, FVG."""

from __future__ import annotations

from typing import Optional

from .types import (
    Bias,
    BreakerZone,
    FVG,
    HTFStateSnapshot,
    ICTBlueprintParams,
    LTFSetup,
    OBId,
    OrderBlock,
    SetupPhase,
    Side,
    SwingPoint,
)


# ---------------------------------------------------------------------------
# OB zone entry check
# ---------------------------------------------------------------------------


def check_ob_zone_entry(
    h1_open: float,
    h1_high: float,
    h1_low: float,
    h1_close: float,
    ob: OrderBlock,
    requirement: str = "close_inside",
    overlap_pct: float = 0.10,
) -> bool:
    """Check whether an H1 bar enters the OB zone per the given requirement."""
    if requirement == "touch":
        # Any overlap between candle range and OB zone
        return h1_low <= ob.top and h1_high >= ob.bottom

    elif requirement == "close_inside":
        return ob.bottom <= h1_close <= ob.top

    elif requirement == "percent_inside":
        body_top = max(h1_open, h1_close)
        body_bottom = min(h1_open, h1_close)
        body_height = body_top - body_bottom
        if body_height <= 0:
            # Doji — treat as touch
            return ob.bottom <= h1_close <= ob.top
        overlap_top = min(body_top, ob.top)
        overlap_bottom = max(body_bottom, ob.bottom)
        overlap = max(0.0, overlap_top - overlap_bottom)
        return (overlap / body_height) >= overlap_pct

    return False


# ---------------------------------------------------------------------------
# Candidate setup selection
# ---------------------------------------------------------------------------


def select_candidate_setups(active_setups: list[LTFSetup]) -> list[LTFSetup]:
    """Return active (non-timed-out, non-inactive) setups ordered newest OB first."""
    valid = [
        s
        for s in active_setups
        if s.phase not in (SetupPhase.INACTIVE, SetupPhase.TIMED_OUT)
    ]
    # Newest OB first (by msb_bar_index descending)
    valid.sort(key=lambda s: s.ob.msb_bar_index, reverse=True)
    return valid


# ---------------------------------------------------------------------------
# L0 / H0 detection
# ---------------------------------------------------------------------------


def find_l0_h0(
    ltf_swing_lows: list[SwingPoint],
    ltf_swing_highs: list[SwingPoint],
    ob: OrderBlock,
    current_index: int,
    last_exit_bar_index: int = -1,
) -> Optional[tuple[SwingPoint, SwingPoint]]:
    """Find the most recent valid L0 (swing low inside/near OB) and its preceding H0.

    L0 must have center index > last_exit_bar_index for freshness.
    H0 is the most recent swing high with index < L0.index.
    """
    # Search from most recent swing low backwards
    for l0 in reversed(ltf_swing_lows):
        if l0.index > current_index:
            continue
        if l0.index <= last_exit_bar_index:
            continue
        # L0 should be inside or near the OB zone
        if l0.price > ob.top:
            continue
        # Find H0 = most recent swing high before L0
        h0 = None
        for sh in reversed(ltf_swing_highs):
            if sh.index < l0.index:
                h0 = sh
                break
        if h0 is not None:
            return (l0, h0)
    return None


# ---------------------------------------------------------------------------
# Sweep detection
# ---------------------------------------------------------------------------


def check_sweep(h1_low: float, l0_price: float) -> Optional[float]:
    """Check if the H1 bar sweeps below L0. Returns the sweep low or None."""
    if h1_low < l0_price:
        return h1_low
    return None


# ---------------------------------------------------------------------------
# LTF MSB
# ---------------------------------------------------------------------------


def check_ltf_msb(h1_close: float, h0_price: float) -> bool:
    """Bullish LTF MSB: H1 candle closes above H0."""
    return h1_close > h0_price


# ---------------------------------------------------------------------------
# Breaker zone
# ---------------------------------------------------------------------------


def define_breaker_zone(
    h1_bars: list[tuple[int, int, float, float, float, float]],
    msb_confirm_index: int,
    breaker_candles: int = 1,
) -> Optional[BreakerZone]:
    """Define breaker zone by looking backwards from the MSB confirmation bar.

    Find the last N bearish (close < open) H1 candles immediately before the confirmation bar.
    """
    # Find position of msb_confirm bar
    confirm_pos = None
    for i, b in enumerate(h1_bars):
        if b[0] == msb_confirm_index:
            confirm_pos = i
            break
    if confirm_pos is None or confirm_pos == 0:
        return None

    bearish: list[tuple[int, int, float, float, float, float]] = []
    for i in range(confirm_pos - 1, -1, -1):
        b = h1_bars[i]
        if b[5] < b[2]:  # close < open = bearish
            bearish.append(b)
            if len(bearish) >= breaker_candles:
                break
        else:
            break

    if not bearish:
        return None

    top = max(b[3] for b in bearish)
    bottom = min(b[4] for b in bearish)
    return BreakerZone(top=top, bottom=bottom, bar_index=bearish[0][0])


# ---------------------------------------------------------------------------
# FVG detection
# ---------------------------------------------------------------------------


def detect_fvg(
    h1_bars: list[tuple[int, int, float, float, float, float]],
    bar_index: int,
) -> Optional[FVG]:
    """Detect a fair value gap at bar_index (3-candle pattern).

    Bullish FVG: bar[i-2].high < bar[i].low (gap up).
    Bearish FVG: bar[i-2].low > bar[i].high (gap down).
    """
    # Find position of bar_index
    pos = None
    for i, b in enumerate(h1_bars):
        if b[0] == bar_index:
            pos = i
            break
    if pos is None or pos < 2:
        return None

    b0 = h1_bars[pos - 2]  # candle 1
    b2 = h1_bars[pos]  # candle 3

    # Bullish FVG
    if b0[3] < b2[4]:  # bar[-2].high < bar[0].low
        return FVG(top=b2[4], bottom=b0[3], bar_index=bar_index, bullish=True)

    # Bearish FVG
    if b0[4] > b2[3]:  # bar[-2].low > bar[0].high
        return FVG(top=b0[4], bottom=b2[3], bar_index=bar_index, bullish=False)

    return None


# ---------------------------------------------------------------------------
# Entry trigger
# ---------------------------------------------------------------------------


def check_entry_trigger(
    h1_open: float,
    h1_high: float,
    h1_low: float,
    h1_close: float,
    bar_index: int,
    setup: LTFSetup,
    entry_mode: str,
    h1_bars: list[tuple[int, int, float, float, float, float]],
) -> Optional[float]:
    """Check if entry conditions are met. Returns entry_price or None."""
    if entry_mode == "msb_close":
        # Enter on the MSB confirmation candle close
        if setup.phase == SetupPhase.ENTRY_PENDING and setup.msb_bar_index == bar_index:
            return h1_close
        return None

    elif entry_mode == "breaker_retest":
        if setup.breaker is None:
            return None
        # Price retests breaker zone (touches or closes inside)
        if h1_low <= setup.breaker.top and h1_high >= setup.breaker.bottom:
            # Entry at breaker top (conservative — fill at zone boundary)
            if setup.side == Side.LONG:
                return setup.breaker.top
            else:
                return setup.breaker.bottom
        return None

    elif entry_mode == "fvg_fill":
        if setup.fvg is None:
            # Try to detect FVG
            fvg = detect_fvg(h1_bars, setup.msb_bar_index)
            if fvg is not None:
                setup.fvg = fvg
            else:
                return None
        # Check if bar fills into FVG (must overlap the zone)
        if setup.side == Side.LONG and setup.fvg.bullish:
            if h1_low <= setup.fvg.top and h1_high >= setup.fvg.bottom:
                return setup.fvg.top
        elif setup.side == Side.SHORT and not setup.fvg.bullish:
            if h1_high >= setup.fvg.bottom and h1_low <= setup.fvg.top:
                return setup.fvg.bottom
        return None

    return None


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def should_timeout(setup: LTFSetup, bar_index: int, max_wait_bars: int) -> bool:
    """Check if setup has exceeded max wait bars after MSB confirmation."""
    if setup.phase not in (SetupPhase.ENTRY_PENDING, SetupPhase.MSB_PENDING):
        return False
    if setup.msb_bar_index < 0:
        return False
    return (bar_index - setup.msb_bar_index) > max_wait_bars


# ---------------------------------------------------------------------------
# Setup state machine
# ---------------------------------------------------------------------------


def advance_setup(
    setup: LTFSetup,
    h1_bar: tuple[int, int, float, float, float, float],
    ltf_swing_lows: list[SwingPoint],
    ltf_swing_highs: list[SwingPoint],
    h1_bars: list[tuple[int, int, float, float, float, float]],
    params: ICTBlueprintParams,
    htf_state: HTFStateSnapshot,
) -> Optional[float]:
    """Advance the setup state machine for one H1 bar.

    Returns entry_price if entry triggered, None otherwise.
    Mutates setup.phase and related fields.
    """
    bar_idx, bar_ts, bar_open, bar_high, bar_low, bar_close = h1_bar

    # SCANNING: look for L0/H0
    if setup.phase == SetupPhase.SCANNING:
        result = find_l0_h0(
            ltf_swing_lows,
            ltf_swing_highs,
            setup.ob,
            bar_idx,
            setup.last_exit_bar_index,
        )
        if result is not None:
            setup.l0, setup.h0 = result
            if params.require_sweep:
                setup.phase = SetupPhase.SWEEP_PENDING
            else:
                setup.phase = SetupPhase.MSB_PENDING
        return None

    # SWEEP_PENDING: wait for sweep below L0
    if setup.phase == SetupPhase.SWEEP_PENDING:
        # Check if L0/H0 needs update (newer L0 available)
        result = find_l0_h0(
            ltf_swing_lows,
            ltf_swing_highs,
            setup.ob,
            bar_idx,
            setup.last_exit_bar_index,
        )
        if result is not None:
            new_l0, new_h0 = result
            if setup.l0 is None or new_l0.index > setup.l0.index:
                setup.l0, setup.h0 = new_l0, new_h0

        if setup.l0 is None:
            return None
        sweep = check_sweep(bar_low, setup.l0.price)
        if sweep is not None:
            # Sweep must be strictly lower than prior sweep (or first sweep)
            if setup.sweep_low is None or sweep < setup.sweep_low:
                setup.sweep_low = sweep
                setup.phase = SetupPhase.MSB_PENDING
        return None

    # MSB_PENDING: wait for close above H0
    if setup.phase == SetupPhase.MSB_PENDING:
        if setup.h0 is None:
            return None
        if check_ltf_msb(bar_close, setup.h0.price):
            setup.msb_bar_index = bar_idx
            setup.bars_since_msb = 0

            # Define breaker zone
            setup.breaker = define_breaker_zone(
                h1_bars, bar_idx, params.breaker_candles
            )

            # Detect FVG
            setup.fvg = detect_fvg(h1_bars, bar_idx)

            entry_mode = params.entry_mode

            # For msb_close mode, enter immediately
            if entry_mode == "msb_close":
                setup.phase = SetupPhase.ENTRY_PENDING
                return bar_close

            setup.phase = SetupPhase.ENTRY_PENDING
        return None

    # ENTRY_PENDING: wait for entry trigger or timeout
    if setup.phase == SetupPhase.ENTRY_PENDING:
        setup.bars_since_msb = bar_idx - setup.msb_bar_index

        if should_timeout(setup, bar_idx, params.max_wait_bars_after_msb):
            setup.phase = SetupPhase.TIMED_OUT
            return None

        entry_price = check_entry_trigger(
            bar_open,
            bar_high,
            bar_low,
            bar_close,
            bar_idx,
            setup,
            params.entry_mode,
            h1_bars,
        )
        return entry_price

    return None


# ---------------------------------------------------------------------------
# Setup creation helpers
# ---------------------------------------------------------------------------


def create_setup_for_ob(ob: OrderBlock, htf_bias: Bias) -> LTFSetup:
    """Create a new LTF setup for an active order block.

    Carries forward ob.last_setup_bar_index so the setup won't reuse
    stale L0 swings from a prior timed-out or exited attempt.
    """
    side = Side.LONG if htf_bias == Bias.BULLISH else Side.SHORT
    return LTFSetup(
        ob=ob,
        side=side,
        phase=SetupPhase.SCANNING,
        last_exit_bar_index=ob.last_setup_bar_index,
    )


def get_active_setup_keys(setups: list[LTFSetup]) -> set[OBId]:
    """Return set of ob_ids that already have an active setup."""
    return {
        s.ob.ob_id
        for s in setups
        if s.phase not in (SetupPhase.INACTIVE, SetupPhase.TIMED_OUT)
    }
