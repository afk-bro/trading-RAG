"""
Breakout 52-Week High Strategy.

Entry: When price breaks above 52-week (configurable lookback) high.
Exit: End-of-day (EOD) mandatory close.

This is a momentum strategy that enters positions when price
establishes new highs, indicating strong bullish momentum.
"""

from math import floor
from uuid import UUID

from app.schemas import IntentAction, PaperState, TradeIntent
from app.services.strategy.models import (
    ExecutionSpec,
    MarketSnapshot,
    StrategyEvaluation,
)

# Rounding for deterministic qty (8 decimals for crypto, adjustable)
QUANTITY_DECIMALS = 8


def round_quantity(qty: float, decimals: int = QUANTITY_DECIMALS) -> float:
    """
    Round DOWN to avoid over-allocation.

    Args:
        qty: Raw quantity to round
        decimals: Number of decimal places (default 8 for crypto)

    Returns:
        Quantity rounded down to specified decimals
    """
    multiplier = 10**decimals
    return floor(qty * multiplier) / multiplier


def evaluate_breakout_52w_high(
    spec: ExecutionSpec,
    snapshot: MarketSnapshot,
    paper_state: PaperState,
    evaluation_id: UUID,
    at_max_positions: bool,
) -> StrategyEvaluation:
    """
    Evaluate breakout above 52-week high strategy.

    Entry Logic:
    - No existing position for symbol
    - Current price > 52-week high (computed from PRIOR bars only)
    - max_positions gate is checked by runner (not blocked here)

    Exit Logic:
    - EOD flag set and has position -> mandatory close

    Args:
        spec: Strategy configuration (entry type, risk params, etc.)
        snapshot: Current market state with OHLCV bars
        paper_state: Current paper trading positions/cash
        evaluation_id: Shared ID for all intents in this evaluation
        at_max_positions: Whether position limit is reached (info only)

    Returns:
        StrategyEvaluation with intents, signals, and debug metadata
    """
    symbol = snapshot.symbol
    position = paper_state.positions.get(symbol)
    has_position = position is not None and position.quantity > 0
    intents: list[TradeIntent] = []
    signals: list[str] = []

    # Derive last_price from bars if not provided
    last_price = snapshot.last_price
    if last_price is None:
        last_price = snapshot.bars[-1].close

    # Guard: invalid price (division by zero risk)
    if last_price <= 0:
        signals.append("entry_skipped_invalid_price")
        return StrategyEvaluation(
            spec_id=str(spec.instance_id),
            symbol=symbol,
            ts=snapshot.ts,
            intents=[],
            signals=signals,
            metadata={"high_52w": None, "last_price": last_price, "at_max_positions": at_max_positions},
            evaluation_id=evaluation_id,
        )

    # Compute 52w high from PRIOR bars (exclude current bar!)
    lookback = spec.entry.lookback_days
    # snapshot.bars[-(lookback + 1):-1] gives us lookback bars BEFORE current
    prior_bars = snapshot.bars[-(lookback + 1) : -1]

    # Guard: no history to compute high_52w
    if not prior_bars and snapshot.high_52w is None:
        signals.append("entry_skipped_no_history")
        return StrategyEvaluation(
            spec_id=str(spec.instance_id),
            symbol=symbol,
            ts=snapshot.ts,
            intents=[],
            signals=signals,
            metadata={"high_52w": None, "last_price": last_price, "at_max_positions": at_max_positions},
            evaluation_id=evaluation_id,
        )

    # Signal partial lookback
    if prior_bars and len(prior_bars) < lookback:
        signals.append(f"lookback_partial_{len(prior_bars)}_of_{lookback}")

    if snapshot.high_52w is not None:
        high_52w = snapshot.high_52w
    else:
        high_52w = max(b.high for b in prior_bars)

    # Helper to create intent with shared correlation_id
    def make_intent(**kwargs) -> TradeIntent:
        return TradeIntent(
            workspace_id=spec.workspace_id,
            correlation_id=str(evaluation_id),  # Shared across evaluation
            strategy_entity_id=spec.instance_id,  # Runtime instance UUID
            symbol=symbol,
            timeframe=spec.timeframe,
            **kwargs,
        )

    # EXIT: EOD with position (always allowed, even at max_positions)
    if snapshot.is_eod and has_position:
        intents.append(
            make_intent(
                action=IntentAction.CLOSE_LONG,
                quantity=position.quantity,  # Full close
                reason="EOD exit",
            )
        )
        signals.append("eod_exit_triggered")

    # ENTRY: Breakout above 52w high (no position, not blocked by max_positions yet)
    elif not has_position and last_price > high_52w:
        raw_qty = spec.risk.dollars_per_trade / last_price
        qty = round_quantity(raw_qty)

        if qty > 0:
            intents.append(
                make_intent(
                    action=IntentAction.OPEN_LONG,
                    quantity=qty,
                    reason=f"Breakout: {last_price:.2f} > 52w high {high_52w:.2f}",
                )
            )
            signals.append("breakout_entry_triggered")
        else:
            signals.append("entry_skipped_zero_qty")

    return StrategyEvaluation(
        spec_id=str(spec.instance_id),
        symbol=symbol,
        ts=snapshot.ts,
        intents=intents,
        signals=signals,
        metadata={"high_52w": high_52w, "last_price": last_price, "at_max_positions": at_max_positions},
        evaluation_id=evaluation_id,
    )
