# ATR Trailing Stop — Exit Management Redesign

## Problem

MFE capture rate is ~40%. Trades routinely move +1R to +1.8R in our favor, then
reverse to hit the original structural stop at -1R. The fixed 2R target caps
upside on the few trades that run, while providing no protection on trades that
move favorably then reverse.

## Solution

Replace the fixed 2R target with an ATR-based trailing stop that activates after
the trade reaches +1R MFE. One new parameter: `trail_atr_mult`.

## Specification

### Trail distance (frozen at entry)

```
trail_distance = entry_atr * trail_atr_mult
```

Computed once from the ATR at the time of entry. Stored on `TradeRecord`. Never
recomputed. This avoids regime drift mid-trade and keeps behavior reproducible.

### Activation condition (+1R MFE, intrabar extremes)

Uses bar high/low, not closes. Activation is checked on every bar:

```python
# Long
favorable_extreme = bar.high
mfe_points = favorable_extreme - trade.entry_price
activated = mfe_points >= trade.risk_points  # risk_points = entry - initial_stop

# Short
favorable_extreme = bar.low
mfe_points = trade.entry_price - favorable_extreme
activated = mfe_points >= trade.risk_points
```

### Guarded activation with immediate trail

On the activation bar, BE floor is set first, then trail formula applies
immediately (same bar). No "next bar" delay:

```python
if not trade.trail_active and mfe_points >= trade.risk_points:
    trade.trail_active = True
    trade.stop_price = max(trade.stop_price, trade.entry_price)  # BE floor
    # Fall through to trail update below — same bar

if trade.trail_active:
    # Long
    new_stop = trade.trail_high - trade.trail_distance
    trade.stop_price = max(trade.stop_price, new_stop)  # ratchet only forward

    # Short
    new_stop = trade.trail_low + trade.trail_distance
    trade.stop_price = min(trade.stop_price, new_stop)
```

Before activation, `trade.stop_price` stays at the original structural stop.
No pre-arm, no partial trailing.

### Target removal

When trail is active, `target_price` is set to `float("inf")` for longs,
`float("-inf")` for shorts. Downstream comparisons (`bar.high >= target_price`)
naturally return `False` — target never triggers. No null checks needed.

The trace report formats inf as `"none (trailing)"`.

### R-multiple calculation

Always computed against `initial_stop`, not the trailed stop:

```
R = pnl_points / risk_points
```

Where `risk_points = abs(entry_price - initial_stop)`. A trade that enters at
100 with initial stop at 95 (risk=5), trails to 108, exits at 106 has
`R = 6/5 = +1.2R`.

### New TradeRecord fields

| Field | Type | Description |
|-------|------|-------------|
| `trail_distance` | `float` | Frozen `entry_atr * trail_mult`, set at entry. 0.0 when trail disabled. |
| `trail_active` | `bool` | Flipped once at +1R activation. |
| `trail_high` | `float` | Best favorable extreme (longs). Updated every bar. |
| `trail_low` | `float` | Best favorable extreme (shorts). Updated every bar. |
| `initial_stop` | `float` | Original stop before any mutation. For R-multiple calculation. |

### Exit reason taxonomy

| Reason | Meaning |
|--------|---------|
| `stop_loss` | Original stop hit (trade never reached +1R) |
| `trail_stop` | Trailing stop hit (trade reached +1R, captured partial) |
| `target` | Fixed target hit (only when trail is disabled) |
| `time_stop` | Time-based exit |
| `eod` | End-of-day flat |

### Integration with `resolve_bar_exit()`

```
resolve_bar_exit() evaluation order:
    0. Trail update (ratchet stop)     ← NEW, replaces breakeven block
    1. Stop/target hit detection       ← target_hit is dead when trailing
    2. Intrabar ambiguity              ← only stop matters when no target
    3. Time-stop                       ← unchanged
    4. EOD exit                        ← unchanged
```

Trail and breakeven are mutually exclusive. `trail_atr_mult` OR
`breakeven_at_r`, not both.

### CLI

One new flag:

```
--trail-atr-mult FLOAT    Trail distance as ATR multiple (default: None = disabled).
                          Activates at +1R MFE. Mutually exclusive with --breakeven-at-r.
```

Activation threshold is always +1R. One knob only.

### Report additions

After MFE/MAE section:

```
TRAILING STOP
Trail distance:        45.00 points (1.5x ATR)
Trades activated:      42 / 68  (61.8%)
Avg R at trail exit:   +0.85R
Avg R at stop exit:    -1.02R
MFE capture (trail):   67.3%
MFE capture (all):     58.1%
```

## What we are NOT doing

- No partial exits / scale-out
- No dynamic ATR recomputation mid-trade
- No configurable activation threshold (always +1R)
- No trail tightening over time
- Breakeven logic stays in codebase, mutually exclusive with trail

## Expected impact

- MFE capture: 40% → 65-70%
- Trades that reach +1R get protected (BE floor minimum)
- Winners that run past +2R are captured instead of capped
- Losers that never reach +1R are unchanged (same structural stop)

## Files to modify

| File | Change |
|------|--------|
| `app/services/backtest/engines/unicorn_runner.py` | TradeRecord fields, resolve_bar_exit trail block, trail_high/trail_low tracking, target sentinel, report section |
| `scripts/run_unicorn_backtest.py` | `--trail-atr-mult` CLI arg, mutual exclusion with `--breakeven-at-r` |
| `docs/features/backtests.md` | Document trailing stop feature |

## Validation

```bash
# A/B comparison: baseline (fixed 2R) vs trailing (1.5x ATR)
python scripts/run_unicorn_backtest.py --symbol MNQ \
  --databento-csv <data> --start-date 2024-01-01 --end-date 2025-07-01 \
  --multi-tf --eval-mode --eval-account-size 50000 --eval-max-drawdown 2000 \
  --write-baseline tmp/baseline.json

python scripts/run_unicorn_backtest.py --symbol MNQ \
  --databento-csv <data> --start-date 2024-01-01 --end-date 2025-07-01 \
  --multi-tf --eval-mode --eval-account-size 50000 --eval-max-drawdown 2000 \
  --trail-atr-mult 1.5 \
  --baseline-run tmp/baseline.json
```

Key metrics to compare: MFE capture rate, avg R, win rate, profit factor,
trailing drawdown, R_day range.
