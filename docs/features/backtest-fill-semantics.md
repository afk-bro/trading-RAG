# Backtest Fill Semantics

Execution-model assumptions used by the Unicorn backtest runner
(`app/services/backtest/engines/unicorn_runner.py`).
All exit logic lives in `resolve_bar_exit()`.

---

## Entry timing

| Property | Value |
|----------|-------|
| Order type | Market-on-open |
| Fill price | `bar.open ± slippage_points` (+ for long, − for short) |
| Look-ahead | None — signal is computed from bars `[0 .. i-1]`, entry is at bar `i` open |

Entry slippage is always adverse: a buy fills above the open, a sell below.

---

## Stop orders

| Property | Value |
|----------|-------|
| Order type | Stop-market |
| Normal fill | `stop_price ± slippage_points` (− for long, + for short) |
| Gap-through fill | `min(stop_price, bar.open) − slippage` (long) |
| | `max(stop_price, bar.open) + slippage` (short) |
| Trigger condition | `bar.low <= stop` (long) or `bar.high >= stop` (short) |

Gap-through means the bar opens past the stop level.  The fill price
is the worse of `{stop, bar.open}` because a stop-market order cannot
fill better than the open when the open itself gaps through.

No worse-than-open scenario is modeled (e.g., further slippage into a
fast market beyond the open).

---

## Target orders

| Property | Value |
|----------|-------|
| Order type | Limit |
| Fill price | `target_price ± slippage_points` (− for long, + for short) |
| Trigger condition | `bar.high >= target` (long) or `bar.low <= target` (short) |
| Price improvement | None — limit fills at the limit price, not at the open |

Adverse slippage is applied even to limit fills as a conservative
assumption (partial fills, queue priority, etc.).  No gap-through
improvement is modeled for targets: if the bar opens beyond the target,
the fill price is still `target ± slippage`, not `open ± slippage`.

---

## Time-stop

| Property | Value |
|----------|-------|
| Order type | Market (at bar.close) |
| Trigger | Elapsed time ≥ `time_stop_minutes` AND unrealized R < threshold |
| Fill price | `bar.close ± slippage_points` |

Only fires when the trade has been open long enough AND is not
sufficiently profitable.  This is a conditional exit, not a trailing stop.

---

## EOD exit

| Property | Value |
|----------|-------|
| Order type | Market (at bar.close) |
| Trigger | Eastern time ≥ `eod_time` (default 15:45 ET) |
| Fill price | `bar.close ± slippage_points` |

---

## Entry-bar exit check

After a new trade is created and appended to `open_trades`, the
runner immediately calls `resolve_bar_exit()` on the same bar.
This handles the case where the entry bar itself breaches the stop
or target.

The general open-trades loop runs **before** new trade creation in
the same iteration, so there is no double-processing risk.

---

## IntrabarPolicy — same-bar ambiguity

When both stop and target are hit within the same bar (bar range spans
both levels), the `IntrabarPolicy` determines which exit fires first.

| Policy | Behavior | Use case |
|--------|----------|----------|
| `WORST` | Stop fires first | **Default.** Conservative. Use for validation. |
| `BEST` | Target fires first | Optimistic upper bound. Not realistic. |
| `RANDOM` | 50/50 coin flip | Monte Carlo stress testing. |
| `OHLC_PATH` | Deterministic path based on bar direction | See below. |

### OHLC_PATH details

The assumed intrabar price path depends on whether the bar is bullish
or bearish:

**Bullish bar** (`close >= open`): path is `O → H → L → C`

| Direction | First leg (O→H) | Second leg (H→L) | Result |
|-----------|-----------------|-------------------|--------|
| Long | Target hit | Stop hit | **Target wins** |
| Short | Stop hit | Target hit | **Stop wins** |

**Bearish bar** (`close < open`): path is `O → L → H → C`

| Direction | First leg (O→L) | Second leg (L→H) | Result |
|-----------|-----------------|-------------------|--------|
| Long | Stop hit | Target hit | **Stop wins** |
| Short | Target hit | Stop hit | **Target wins** |

This is a deterministic heuristic — no simulation or randomness.
It is more realistic than `WORST` or `BEST` for directional bars,
but still an approximation (real intrabar paths are not this simple).

---

## PnL calculation

```
pnl_points = exit_price − entry_price   (long)
pnl_points = entry_price − exit_price   (short)
```

Slippage is already embedded in `exit_price` (and `entry_price`).
Commission is applied separately after exit:

```
pnl_dollars = (pnl_points × point_value × quantity) − commission
```

---

## Pre-entry guards

Guards are applied after criteria scoring passes but before trade creation.
They reject setups that pass the 8-criteria checklist but lack signal conviction.
Diagnostics are recorded on every qualifying setup regardless of rejection.

| Guard | Config field | Rejects when | Default |
|-------|-------------|-------------|---------|
| Wick | `max_wick_ratio` | Signal bar adverse wick / range > threshold | None (disabled) |
| Range | `max_range_atr_mult` | Signal bar range / ATR > threshold | None (disabled) |
| Displacement | `min_displacement_atr` | MSS displacement < threshold × ATR | None (disabled) |

Evaluation order: direction filter → wick → range → displacement → trade creation.

All guards use causal data only (signal bar = last closed HTF bar, ATR from previous bar).

---

## What is NOT modeled

| Omission | Impact | Notes |
|----------|--------|-------|
| Gap-through target improvement | Conservative | Target fills at limit, not at better open |
| Probabilistic slippage | Minor | Fixed slippage per side, not volume/range-dependent |
| Volume-aware fills | Minor | No partial fills or liquidity checks |
| Worse-than-open stop fills | Conservative | Gap-through stops fill at open, not further |
| Bid-ask spread | Minor | Subsumed into slippage parameter |
