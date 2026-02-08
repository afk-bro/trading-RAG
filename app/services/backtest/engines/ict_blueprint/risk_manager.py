"""Risk management: stop/TP placement, R:R gate, de-risk, position tracking."""

from __future__ import annotations

from typing import Optional

from .types import (
    Bias,
    ClosedTrade,
    HTFStateSnapshot,
    LTFSetup,
    Position,
    Side,
)


# ---------------------------------------------------------------------------
# Stop placement
# ---------------------------------------------------------------------------


def compute_stop_price(
    setup: LTFSetup,
    stop_mode: str,
    side: Side,
    buffer_ticks: float = 2.0,
) -> Optional[float]:
    """Compute stop price based on the stop mode.

    below_sweep: stop below the sweep low (+ buffer).
    below_breaker: stop below the breaker zone bottom (+ buffer).
    """
    if stop_mode == "below_sweep":
        if setup.sweep_low is None:
            return None
        if side == Side.LONG:
            return setup.sweep_low - buffer_ticks
        else:
            # For shorts, sweep would be a sweep high; stop above sweep
            # In the bullish-only framing, sweep_low is the liquidity grab below L0.
            # For shorts: stop above the sweep high (symmetrical)
            return setup.sweep_low + buffer_ticks

    elif stop_mode == "below_breaker":
        if setup.breaker is None:
            return None
        if side == Side.LONG:
            return setup.breaker.bottom - buffer_ticks
        else:
            return setup.breaker.top + buffer_ticks

    return None


# ---------------------------------------------------------------------------
# Target placement
# ---------------------------------------------------------------------------


def compute_target_price(
    entry_price: float,
    stop_price: float,
    side: Side,
    tp_mode: str,
    htf_state: HTFStateSnapshot,
    fixed_rr: float = 3.0,
) -> Optional[float]:
    """Compute take-profit target.

    external_range: prior swing high (longs) or swing low (shorts) from HTF.
    fixed_rr: fixed multiple of risk distance.
    """
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return None

    if tp_mode == "external_range":
        if htf_state.current_range is None:
            return None
        if side == Side.LONG:
            target = htf_state.current_range.high
            # Ensure target is above entry
            if target <= entry_price:
                # Fallback: use most recent swing high above entry
                for sh in reversed(htf_state.swing_highs):
                    if sh.price > entry_price:
                        target = sh.price
                        break
                else:
                    return entry_price + risk * fixed_rr
            return target
        else:
            target = htf_state.current_range.low
            if target >= entry_price:
                for sl in reversed(htf_state.swing_lows):
                    if sl.price < entry_price:
                        target = sl.price
                        break
                else:
                    return entry_price - risk * fixed_rr
            return target

    elif tp_mode == "fixed_rr":
        if side == Side.LONG:
            return entry_price + risk * fixed_rr
        else:
            return entry_price - risk * fixed_rr

    return None


# ---------------------------------------------------------------------------
# R:R gate
# ---------------------------------------------------------------------------


def check_rr_gate(
    entry_price: float,
    stop_price: float,
    target_price: float,
    min_rr: float,
    side: Side,
) -> bool:
    """Return True if the reward:risk ratio meets the minimum."""
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return False
    if side == Side.LONG:
        reward = target_price - entry_price
    else:
        reward = entry_price - target_price
    if reward <= 0:
        return False
    return (reward / risk) >= min_rr


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def compute_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    point_value: float = 50.0,
) -> float:
    """Compute position size (number of contracts/units).

    risk_dollars = equity * risk_pct
    risk_per_unit = |entry - stop| * point_value
    size = risk_dollars / risk_per_unit (floored to 1 minimum)
    """
    risk_dollars = equity * risk_pct
    risk_per_unit = abs(entry_price - stop_price) * point_value
    if risk_per_unit <= 0:
        return 0.0
    size = risk_dollars / risk_per_unit
    return max(1.0, size)


# ---------------------------------------------------------------------------
# De-risk
# ---------------------------------------------------------------------------


def process_derisk(
    position: Position,
    h1_high: float,
    h1_low: float,
    derisk_mode: str,
    trigger_rr: float,
) -> Optional[float]:
    """Evaluate de-risk condition on intrabar high/low.

    Returns partial PnL (points) if half_off triggered, or None.
    Mutates position (stop move or size reduction).
    """
    if position.derisk_triggered or derisk_mode == "none":
        return None

    risk = position.risk_points
    if risk <= 0:
        return None

    # Check if trigger reached on intrabar extremes
    if position.side == Side.LONG:
        unrealized_r = (h1_high - position.entry_price) / risk
    else:
        unrealized_r = (position.entry_price - h1_low) / risk

    if unrealized_r < trigger_rr:
        return None

    position.derisk_triggered = True

    if derisk_mode == "move_to_be":
        position.stop_price = position.entry_price
        return None  # No realized PnL yet

    elif derisk_mode == "half_off":
        half = position.remaining_size * 0.5
        if half <= 0:
            return None
        if position.side == Side.LONG:
            pnl_points = (h1_high - position.entry_price)
        else:
            pnl_points = (position.entry_price - h1_low)
        # Approximate: use the trigger level as fill (conservative)
        fill_price = position.entry_price + (risk * trigger_rr if position.side == Side.LONG
                                              else -(risk * trigger_rr))
        pnl_points = abs(fill_price - position.entry_price)
        position.remaining_size -= half
        return pnl_points * half  # Return PnL in point-units (multiply by point_value later)

    return None


# ---------------------------------------------------------------------------
# Exit checks
# ---------------------------------------------------------------------------


def check_exit(
    position: Position,
    h1_high: float,
    h1_low: float,
    h1_close: float,
    bar_ts: int,
    slippage_points: float = 0.0,
) -> Optional[tuple[float, str]]:
    """Check if position hits stop or target on this bar.

    Intrabar ordering: if both stop and target are reachable, stop wins (conservative).
    Returns (exit_price, reason) or None.
    """
    stop_hit = False
    target_hit = False

    if position.side == Side.LONG:
        stop_hit = h1_low <= position.stop_price
        target_hit = h1_high >= position.target_price
    else:
        stop_hit = h1_high >= position.stop_price
        target_hit = h1_low <= position.target_price

    # Conservative: stop wins if both hit
    if stop_hit:
        exit_price = position.stop_price
        if position.side == Side.LONG:
            exit_price -= slippage_points
        else:
            exit_price += slippage_points
        return (exit_price, "stop_loss")

    if target_hit:
        exit_price = position.target_price
        if position.side == Side.LONG:
            exit_price -= slippage_points
        else:
            exit_price += slippage_points
        return (exit_price, "take_profit")

    return None


# ---------------------------------------------------------------------------
# Entry-exit collision
# ---------------------------------------------------------------------------


def check_entry_exit_collision(
    entry_price: float,
    stop_price: float,
    h1_high: float,
    h1_low: float,
    side: Side,
) -> bool:
    """Check if the entry bar crosses both entry and stop -> reject the entry.

    If the bar range spans from entry to stop, the entry would be immediately
    stopped out. Return True to reject.
    """
    if side == Side.LONG:
        # For long: entry at or above some level, stop below.
        # Collision if bar reaches both entry (high >= entry) and stop (low <= stop)
        return h1_high >= entry_price and h1_low <= stop_price
    else:
        # For short: entry at or below some level, stop above.
        return h1_low <= entry_price and h1_high >= stop_price


# ---------------------------------------------------------------------------
# Commission / slippage helpers
# ---------------------------------------------------------------------------


def apply_commission(trade_value: float, commission_bps: float) -> float:
    """Return commission cost for a given trade value."""
    return trade_value * (commission_bps / 10000.0)


def bps_to_points(price: float, slippage_bps: float) -> float:
    """Convert slippage in bps to absolute points."""
    return price * (slippage_bps / 10000.0)


# ---------------------------------------------------------------------------
# Trade closing helper
# ---------------------------------------------------------------------------


def close_position(
    position: Position,
    exit_price: float,
    exit_time: int,
    exit_reason: str,
    point_value: float,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> ClosedTrade:
    """Create a ClosedTrade from a Position."""
    if position.side == Side.LONG:
        pnl_points = exit_price - position.entry_price
    else:
        pnl_points = position.entry_price - exit_price

    pnl_dollars = pnl_points * position.remaining_size * point_value

    # Apply commission on entry + exit
    entry_value = position.entry_price * position.remaining_size * point_value
    exit_value = exit_price * position.remaining_size * point_value
    total_commission = apply_commission(entry_value, commission_bps) + apply_commission(
        exit_value, commission_bps
    )
    pnl_dollars -= total_commission

    return ClosedTrade(
        entry_time=position.entry_time,
        exit_time=exit_time,
        entry_price=position.entry_price,
        exit_price=exit_price,
        side=position.side,
        size=position.remaining_size,
        pnl_points=pnl_points,
        pnl_dollars=pnl_dollars,
        exit_reason=exit_reason,
        ob_id=position.ob_id,
    )
