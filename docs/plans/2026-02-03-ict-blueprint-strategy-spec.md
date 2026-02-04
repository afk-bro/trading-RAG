# ICT Blueprint Strategy Spec

**Source**: Trader Mine interview on Chart Fanatics — "The SIMPLE $10 Million ICT Blueprint"
**Asset**: ES / NQ Futures (universal — any liquid market)
**Timeframes**: Daily (HTF analysis) + H1 (LTF execution)
**Style**: Trend-following pullback into institutional order flow zones

---

## 1. High Timeframe — Directional Bias (Daily)

### 1.1 Swing Detection (3-Candle System)

A **swing high** is a candle whose high is higher than the high of the candle immediately before and after it.

A **swing low** is a candle whose low is lower than the low of the candle immediately before and after it.

Parameter: `swing_lookback` (default 1) — number of candles on each side. Value of 1 = the 3-candle system from the transcript. Value of 2 = 5-candle system.

### 1.2 Market Structure Break (MSB)

A **bullish MSB** occurs when a candle *closes* above a prior swing high.

A **bearish MSB** occurs when a candle *closes* below a prior swing low.

**MSB is a state transition, not a one-bar event.** After a bullish MSB is confirmed, the HTF bias is `BULLISH` and remains so until a bearish MSB is confirmed (and vice versa). This prevents bias thrashing during consolidation.

State machine:

```
NEUTRAL ──(bullish MSB)──▸ BULLISH
NEUTRAL ──(bearish MSB)──▸ BEARISH
BULLISH ──(bearish MSB)──▸ BEARISH
BEARISH ──(bullish MSB)──▸ BULLISH
```

Initial state: `NEUTRAL` (no trades until first MSB establishes direction).

### 1.3 Trading Range Definition

After a bullish MSB:
- **Range high** = the swing high that was just broken (the MSB level)
- **Range low** = the most immediate swing low preceding the MSB

After a bearish MSB:
- **Range low** = the swing low that was just broken
- **Range high** = the most immediate swing high preceding the MSB

**Anchor rule**: Always use the most recent/immediate swing point preceding the MSB. Do not reach back to "major" swings unless explicitly overridden by a higher-TF confluence parameter.

Parameter: `range_anchor_mode` — `immediate` (default) | `major_swing` (future extension)

### 1.4 Order Block Identification (Scope Zone)

**Bullish OB**: The last N bearish (down-close) candle(s) immediately before the impulse move that caused the bullish MSB.

**Bearish OB**: The last N bullish (up-close) candle(s) immediately before the impulse move that caused the bearish MSB.

The OB zone is defined by:
- **Top**: highest high of the OB candle(s)
- **Bottom**: lowest low of the OB candle(s)

**The HTF OB does not generate entries. It defines a zone of interest that activates LTF scanning.**

Parameter: `ob_candles` (default 1, sweep 1–3) — number of opposing candles to include in the OB zone.

### 1.5 Premium / Discount (Eligibility Gate)

Compute the 50% level of the trading range:

```
midpoint = range_low + (range_high - range_low) * 0.5
```

- **Discount zone** (below midpoint): Longs are eligible
- **Premium zone** (above midpoint): Shorts are eligible

This is a boolean gate only — it does not signal entry or reversal. LTF structure must independently confirm.

Parameter: `discount_threshold` (default 0.5, sweep 0.382 / 0.5 / 0.618) — how deep into the range price must be for eligibility.

---

## 2. Low Timeframe — Entry Logic (H1)

### 2.1 LTF Scanning Activation

LTF scanning activates when the HTF OB zone is entered. Activation requires meeting the `ob_zone_entry_requirement`:

- `touch` — any overlap between the H1 candle range and the OB zone
- `close_inside` — H1 candle closes within the OB zone
- `percent_inside` — at least X% of the H1 candle body overlaps with the OB zone

Parameter: `ob_zone_entry_requirement` — `close_inside` (default) | `touch` | `percent_inside`
Parameter: `ob_zone_overlap_pct` — 0.10 (default, used only with `percent_inside`)

### 2.2 Swing Structure Inside OB Zone

Once scanning is active, detect H1 swing points using the same 3-candle system:

- **L0** = a candidate swing low (3-candle) that forms while price is inside or has entered the HTF OB zone
- **H0** = the most recent swing high (3-candle) immediately preceding L0 — i.e., the swing high that produced the downleg into L0

These definitions are unambiguous: H0 is the last 3-candle swing high before L0, on the same (H1) timeframe.

Parameter: `ltf_swing_lookback` (default 1, sweep 1–2)

### 2.3 Liquidity Sweep

After L0 forms, wait for the sweep:

- Price trades **below** L0's low (taking out stops beneath the swing low)
- This is the "engineered liquidity" — early longs get stopped out

Parameter: `require_sweep` — `true` (default) | `false`

When `require_sweep=false`, skip directly to MSB confirmation (lower win rate but more signals).

### 2.4 LTF Market Structure Break (Confirmation)

After the sweep (or after L0 if sweep not required):

**A bullish LTF MSB is confirmed when an H1 candle closes above H0.**

This is the same "candle close through the high" rule applied at the LTF level. H0 is precisely defined in §2.2.

### 2.5 Breaker Zone Definition

After the LTF MSB-up is confirmed:

The **breaker zone** = the last N bearish (down-close) H1 candle(s) immediately before the MSB-up impulse.

This is the failed bearish order block that got reclaimed when price closed above H0. It is the "broken and reclaimed" level that becomes support.

Zone boundaries:
- **Top**: highest high of the breaker candle(s)
- **Bottom**: lowest low of the breaker candle(s)

Parameter: `breaker_candles` (default 1, sweep 1–2)

### 2.6 Entry Execution

Entry mode determines how aggressively we enter after the LTF MSB confirms:

| Mode | Entry Point | Description |
|------|------------|-------------|
| `breaker_retest` | Price retests (touches/closes into) the breaker zone after MSB | Most conservative, best R:R |
| `msb_close` | Immediately on the candle that closes above H0 | Aggressive, worse R:R but fewer missed trades |
| `fvg_fill` | On a fill of the fair value gap created by the MSB impulse | Middle ground |

Parameter: `entry_mode` — `breaker_retest` (default) | `msb_close` | `fvg_fill`

### 2.7 Setup Timeout

If no valid entry occurs within `max_wait_bars` H1 candles after the LTF MSB confirmation, the setup is invalidated. Wait for a fresh setup.

Parameter: `max_wait_bars_after_msb` (default 12, sweep 5–30)

This prevents chasing late retests that are no longer "the trade."

---

## 3. Risk Management

### 3.1 Stop Loss

| Mode | Stop Placement |
|------|---------------|
| `below_sweep` | Below the lowest point of the liquidity sweep (the wick low) |
| `below_breaker` | Below the bottom of the breaker zone |

Parameter: `stop_mode` — `below_sweep` (default) | `below_breaker`

`below_sweep` is wider but more robust (Trader Mine's preference for crypto due to wicks). `below_breaker` is tighter, better R:R, but more prone to stop hunts.

### 3.2 Take Profit

| Mode | Target |
|------|--------|
| `external_range` | HTF external range liquidity — the prior swing high (bullish) or swing low (bearish) that defines the range boundary |
| `fixed_rr` | Fixed multiple of risk distance from entry |

Parameter: `tp_mode` — `external_range` (default) | `fixed_rr`
Parameter: `min_rr` — 2.0 (default, sweep 2.0 / 2.5 / 3.0) — minimum reward:risk to accept the trade. If the distance to target / distance to stop < `min_rr`, skip the trade.

### 3.3 De-Risk Rule

At 2R profit:
- Move stop to breakeven, OR
- Close 50% of position

Parameter: `derisk_mode` — `move_to_be` (default) | `half_off` | `none`
Parameter: `derisk_trigger_rr` — 2.0 (default)

### 3.4 Re-Entry After Stop-Out

Multiple attempts are allowed within the same HTF OB zone, **but each attempt requires a fresh setup sequence**:

- A new L0 swing low must form
- A new sweep must occur (if `require_sweep=true`)
- A new H0 break must confirm
- A new breaker zone must be identified

Re-entering the same breaker level without fresh structure is not permitted.

Parameter: `max_attempts_per_ob` — 2 (default, sweep 1–3)

---

## 4. Complete Parameter Table

| Parameter | Type | Default | Sweep Range | Layer |
|-----------|------|---------|-------------|-------|
| `swing_lookback` | int | 1 | [1, 2, 3] | HTF |
| `ob_candles` | int | 1 | [1, 2, 3] | HTF |
| `discount_threshold` | float | 0.5 | [0.382, 0.5, 0.618] | HTF |
| `range_anchor_mode` | str | immediate | [immediate] | HTF |
| `ob_zone_entry_requirement` | str | close_inside | [touch, close_inside, percent_inside] | LTF |
| `ob_zone_overlap_pct` | float | 0.10 | [0.05, 0.10, 0.20] | LTF |
| `ltf_swing_lookback` | int | 1 | [1, 2] | LTF |
| `require_sweep` | bool | true | [true, false] | LTF |
| `entry_mode` | str | breaker_retest | [breaker_retest, msb_close, fvg_fill] | LTF |
| `max_wait_bars_after_msb` | int | 12 | [5, 12, 20, 30] | LTF |
| `breaker_candles` | int | 1 | [1, 2] | LTF |
| `stop_mode` | str | below_sweep | [below_sweep, below_breaker] | Risk |
| `tp_mode` | str | external_range | [external_range, fixed_rr] | Risk |
| `min_rr` | float | 2.0 | [2.0, 2.5, 3.0] | Risk |
| `derisk_mode` | str | move_to_be | [move_to_be, half_off, none] | Risk |
| `derisk_trigger_rr` | float | 2.0 | [1.5, 2.0, 2.5] | Risk |
| `max_attempts_per_ob` | int | 2 | [1, 2, 3] | Risk |

**Total parameter combinations** (full grid): 3×3×3×1×3×3×2×2×3×4×2×2×2×3×3×3×3 = ~2.5M

Recommended: Use random search with ~500–1000 trials + IS/OOS split.

---

## 5. Signal Flow Summary

```
                      ┌─────────────────────────────────┐
                      │     DAILY (HTF) — Bias Engine    │
                      │                                  │
                      │  1. Detect swings (3-candle)     │
                      │  2. Track MSB → set bias state   │
                      │  3. Define range (immediate SW)  │
                      │  4. Mark OB zone (scope)         │
                      │  5. Compute discount/premium     │
                      └──────────────┬──────────────────┘
                                     │
                          bias + OB zone + eligibility
                                     │
                      ┌──────────────▼──────────────────┐
                      │      H1 (LTF) — Entry Engine     │
                      │                                  │
                      │  1. Activate when OB zone hit    │
                      │  2. Detect L0 (swing low)        │
                      │  3. Identify H0 (preceding high) │
                      │  4. Wait for sweep below L0      │
                      │  5. Confirm MSB (close > H0)     │
                      │  6. Define breaker zone           │
                      │  7. Enter on retest / close / FVG│
                      │  8. Set stop + target             │
                      └──────────────┬──────────────────┘
                                     │
                              entry + stop + TP
                                     │
                      ┌──────────────▼──────────────────┐
                      │     Risk Management Layer        │
                      │                                  │
                      │  - Min R:R gate (skip if < 2R)   │
                      │  - De-risk at trigger R           │
                      │  - Re-entry: fresh setup only     │
                      │  - Max attempts per OB zone       │
                      └─────────────────────────────────┘
```

---

## 6. Evaluation Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Win rate | ≥ 35% | 33% breakeven at 2:1 R:R |
| Avg R:R | ≥ 2.0 | Enforced by `min_rr` gate |
| Profit factor | ≥ 1.3 | |
| Max drawdown | ≤ 20% | Standard gates policy |
| Min trades (IS) | ≥ 30 | Sufficient sample size |
| Sharpe ratio | ≥ 0.5 | Annualized |
| IS/OOS gap | ≤ 30% | Overfit detection |

---

## 7. Data Requirements

- **ES.FUT** 1-minute bars from Databento GLBX (2021-01 to 2026-01, ~5 years)
- Resample to **Daily** and **H1** bars
- Front-month continuous contract (roll-adjusted)
- RTH session filter recommended (09:30–16:00 ET for ES)

Source: `docs/historical_data/GLBX-20260129-JNB8PDSQ7C/glbx-mdp3-20210128-20260127.ohlcv-1m.csv`

---

## 8. Implementation Notes

- HTF bias engine runs once per daily bar close
- LTF entry engine runs on each H1 bar, gated by HTF state
- OB zones persist until price fully trades through them (invalidated on opposing MSB)
- Breaker zones are single-use per setup attempt
- All swing/MSB detection uses candle closes, not wicks, for confirmation
- Fair value gaps (for `fvg_fill` entry mode): 3-candle pattern where candle 1 high < candle 3 low (bullish) with gap unfilled
