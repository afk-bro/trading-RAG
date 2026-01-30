# ICT Unicorn Model: Strategy Overview, Forensic Analysis & Lessons Learned

## Table of Contents
1. [The Strategy in Plain English](#the-strategy-in-plain-english)
2. [Forensic Analysis Workflow](#forensic-analysis-workflow)
3. [What We Found and Fixed](#what-we-found-and-fixed)
4. [Lessons Learned](#lessons-learned)

---

## The Strategy in Plain English

### What is the ICT Unicorn Model?

The Unicorn Model is a day trading strategy for futures (like NQ - Nasdaq futures or ES - S&P futures). It's based on the teaching of a trader known as "ICT" (Inner Circle Trader) and looks for a specific sequence of market events before entering a trade.

Think of it like a checklist a pilot goes through before takeoff - every item must be checked before you're cleared to trade.

### The Core Idea

The strategy is built on one key insight: **big institutions (banks, hedge funds) move markets in predictable patterns**. They need to:
1. First "hunt" retail traders' stop losses to get liquidity (someone to trade against)
2. Then move the market in their intended direction

The Unicorn Model tries to identify when this hunt has happened and jump on board with the institutions.

### The 8 Criteria Checklist

Before taking any trade, the strategy checks 8 things:

#### Mandatory (Must ALL pass - these protect you from bad trades)

1. **Higher Timeframe Bias** - Is the bigger picture bullish or bearish?
   - Like checking if the tide is coming in or going out before swimming
   - We only trade WITH the trend, not against it

2. **Valid Stop Distance** - Is the stop loss a reasonable distance away?
   - Prevents taking trades where the stop would be too far (risking too much)
   - Uses ATR (Average True Range) to measure "normal" volatility

3. **Macro Window** - Is it the right time of day?
   - Markets have specific times when institutions are active
   - NY session open (9:30-11:00 AM ET) is prime time
   - Avoids lunch hours and overnight when moves are random

#### Scored (Need 3 of 5 - these confirm the setup quality)

4. **Liquidity Sweep** - Did price just grab stop losses?
   - Price makes a fake move to trigger retail stops
   - Then reverses - this is our signal that institutions are done hunting

5. **Higher Timeframe FVG (Fair Value Gap)** - Is there an imbalance in the market?
   - A gap where price moved so fast it left a "hole"
   - These gaps often get filled - we trade toward them

6. **Breaker Block** - Is there a key support/resistance level?
   - Old support becomes new resistance (and vice versa)
   - Institutions defend these levels

7. **Lower Timeframe FVG** - Confirmation on smaller timeframe
   - Same concept as #5 but on 5-minute charts
   - Fine-tunes our entry

8. **Market Structure Shift (MSS)** - Has the short-term trend changed?
   - Price breaks a recent high/low
   - Confirms the reversal is real, not just noise

### How a Trade Works

**Example: Long (buying) trade setup**

1. Morning, 9:45 AM - we're in the trading window ✓
2. 15-minute chart shows uptrend (higher highs, higher lows) ✓
3. Price drops and takes out yesterday's low (liquidity sweep) ✓
4. We see a bullish FVG on the 15-minute chart ✓
5. Price starts to reverse, breaks above a recent high (MSS) ✓
6. Stop would be 20 points below entry (within 3x ATR) ✓

**Result:** 6/8 criteria pass (3 mandatory + 3 scored) → Take the trade

**Entry:** Buy at market (current price)
**Stop:** Below the FVG low
**Target:** 2x the risk distance

### Risk Management

- Risk a fixed dollar amount per trade (e.g., $1,000)
- Position size adjusts based on stop distance
- Exit at end of day if neither stop nor target hit
- Only one trade at a time

---

## Forensic Analysis Workflow

### The Problem We Started With

The original backtest showed **Profit Factor 17.64** over 6 months. This is a massive red flag.

> **Reality check:** Professional quant funds target PF 1.3-1.5. A PF of 17 means either you've discovered something Nobel Prize-worthy, or (much more likely) there's a bug.

### Step 1: Identify "Too Good to Be True" Results

**Warning signs that triggered investigation:**
- PF > 3 over extended periods
- Win rate + average win that seems impossible
- Results that don't degrade with realistic friction (slippage, commission)

### Step 2: Systematically Check for Look-Ahead Bias

Look-ahead bias is when your backtest "cheats" by using information it couldn't have known at the time.

**What we checked:**

| Check | Question | How to Test |
|-------|----------|-------------|
| Signal timing | Are we using the current bar's data to make decisions about the current bar? | Shift all signals by 1 bar - if results collapse, you had bias |
| Entry price | Are we entering at prices that actually traded? | Enter at bar.open, not at theoretical "perfect" levels |
| ATR calculation | Is the ATR using future price data? | Use ATR from the PREVIOUS bar for sizing |
| Data contamination | Is future data leaking into past calculations? | Check all indicator windows |

### Step 3: Add Realistic Friction

**Before (fantasy land):**
- Enter at exact theoretical price
- Zero slippage
- Zero commission
- Best-case intrabar resolution

**After (real world):**
- Enter at bar.open + slippage (2 ticks = 0.50 points on NQ)
- $5.00 round-trip commission per contract
- When stop AND target both hit in same bar, assume stop hit first (worst case)

### Step 4: Fix Data Issues

**Problem found:** Multiple contract months in the data

CME futures roll quarterly. Our data had:
- NQH4 (March 2024)
- NQM4 (June 2024)
- NQU4 (September 2024)
- NQZ4 (December 2024)

These trade at different prices! Mixing them caused:
- False gaps between contracts
- Wildly inflated ATR (150 points instead of 15)
- Impossible trade setups

**Fix:** Filter to front-month contract only, switching at roll dates.

### Step 5: Re-run with Fixes

**Results after fixing look-ahead and data issues:**
- PF: 17.64 → **0.91** (now losing money!)
- This is actually GOOD - it means we found the bugs

### Step 6: Diagnose Why It's Losing

Now we have a realistic (if unprofitable) backtest. Time to understand WHY it loses.

**Expectancy Decomposition:**
```
Total Trades: 252
  Longs:  124 trades, WR=43.5%, PnL=+655h, AvgR=+0.07R
  Shorts: 128 trades, WR=27.3%, PnL=-1097h, AvgR=-0.20R
```

**Key insight:** Longs are profitable. Shorts are bleeding money.

This makes sense - we backtested Jan-Jun 2024, a strong bull market. Short setups kept getting run over.

### Step 7: Apply Simple Fixes

**Fix 1: Long-only filter**
- Only take bullish setups
- Rationale: Don't fight the trend

**Results:**
- Trades: 252 → 124
- PF: 0.91 → **1.34**
- PnL: -$13,400 → **+$9,250**

**Fix 2: Time-stop (tested but not used)**
- Exit if not at +0.25R within 30 minutes
- Rationale: Good ICT entries move quickly
- Result: Actually hurt performance (PF 1.34 → 1.18)
- Why: NQ needs more than 30 minutes to reach targets

---

## What We Found and Fixed

### Bug #1: Signal Look-Ahead

**Problem:** Pattern detection used `htf_bars[i-100:i+1]` which includes the current bar.

**Fix:** Changed to `htf_bars[i-100:i]` to only use PREVIOUS bars.

**Impact:** This alone dropped PF from ~17 to ~3.

### Bug #2: Entry at Fantasy Prices

**Problem:** Entering at theoretical FVG midpoint that may never have traded.

**Example:**
- Bar range: 16668-16702 (34 points)
- Original entry: 16737 (35 points ABOVE the high!)

**Fix:** Enter at `bar.open + slippage` like a real market order.

**Impact:** Combined with bug #1, dropped PF to ~1.

### Bug #3: ATR Using Current Bar

**Problem:** `htf_atr_values[i]` includes current bar's True Range.

**Fix:** Use `htf_atr_values[i-1]` for position sizing.

**Impact:** More realistic stop distances and position sizes.

### Bug #4: Contract Month Contamination

**Problem:** Back-month contracts mixed with front-month created false price levels.

**Example:**
- NQH4 trading at 17,500
- NQM4 trading at 17,580 (premium for later delivery)
- System sees 80-point "gap" that doesn't exist

**Fix:** Filter CSV to front-month contract only, switching at roll dates.

**Impact:** ATR returned to normal (~15 points vs 150).

### Bug #5: Intrabar Resolution Bias

**Problem:** Assuming target hit before stop when both levels are within same bar's range.

**Fix:** Added `intrabar_policy` parameter:
- `WORST`: Assume stop hit first (conservative, realistic)
- `BEST`: Assume target hit first (optimistic, unrealistic)
- `RANDOM`: 50/50 coin flip

**Impact:** Using WORST reduced win rate by ~5%.

---

## Lessons Learned

### On Backtesting

1. **"Too good" IS bad.** Any backtest showing PF > 3 should be treated as buggy until proven otherwise. Professional systems target PF 1.3-1.5.

2. **Look-ahead bias hides everywhere.** The most common sources:
   - Using `bar[i]` data to decide whether to trade on `bar[i]`
   - Entering at prices that require knowing the bar's full range
   - Indicators that include current bar in their calculation

3. **Test with friction FIRST, not last.** Add slippage, commission, and worst-case fills before you start optimizing. A strategy that only works with perfect fills isn't a strategy.

4. **Data quality matters more than strategy complexity.** Our contract contamination bug had more impact than any parameter we tuned.

### On Strategy Development

5. **Simple filters beat complex optimization.** "Long-only in a bull market" turned a loser into a winner. No parameter tuning required.

6. **Direction > Entry precision.** Being on the right side of the market matters more than getting the perfect entry price.

7. **Not every "improvement" improves things.** Time-stop sounded good in theory (cut losers early) but hurt in practice (cut winners early too).

8. **Expectancy decomposition reveals the truth.** Breaking down results by direction, exit type, and session showed exactly where the strategy bled money.

### On the ICT Unicorn Model Specifically

9. **The model has merit, but conditions matter.** In trending conditions with proper direction filtering, it produced PF 1.34 over 6 months.

10. **Macro timing is crucial.** The NY AM session (9:30-11:00) showed the best results. Off-hours trades dragged down performance.

11. **8 criteria may be too many.** Only 2% of scanned setups passed all 8. Relaxing to "3 mandatory + 3/5 scored" captured more trades without sacrificing edge.

12. **MFE/MAE analysis suggests fixed targets are suboptimal.** Average MFE was 30 handles but average captured profit was only 17 handles (55% capture rate). A trailing stop or scaled exit might improve this.

### On Process

13. **Forensics before optimization.** We spent more time finding bugs than tuning parameters. This is correct - you can't optimize a broken system.

14. **Document as you go.** This analysis would have been impossible to reconstruct from memory. Keeping notes on each change and its impact was essential.

15. **Walk-forward validation reveals overfitting.** Training on Jan-Apr and testing on May-Jun showed similar (poor) results. This confirmed we weren't curve-fitting - the strategy was genuinely marginal.

---

## Current State & Next Steps

### Where We Are Now

- **Realistic backtest:** Look-ahead fixed, friction applied, data cleaned
- **Profitable configuration:** Long-only filter, PF 1.34, +$9,250 over 6 months
- **Clear diagnostics:** Exit breakdown, direction analysis, session stats
- **Pre-entry guards shipped:**
  - **Wick guard** (`max_wick_ratio=0.6`): Filters stop-hunt candles. Reduced evil-profile bleed from -$957 to -$298.
  - **Displacement guard** (`min_displacement_atr=0.5`): Filters low-conviction MSS. On NQ, improved expectancy +52% (+3.04 → +4.61 pts/trade) while retaining 71% of trades. On ES, turned a breakeven instrument (+$0.22/trade) into profitable (+$1.48/trade). Validated across 5 market regimes (2021-2025), 90-day sub-window splits, and ATR(10/14/21) normalizations. Performance gains are regime-robust and scale-invariant.

### Potential Improvements to Test

1. **Regime filter:** Only trade when volatility (VIX) is in normal range
2. **Trailing stop:** Capture more MFE instead of fixed 2R target
3. **Session filter:** Only trade NY AM session
4. **Minimum ATR filter:** Avoid low-volatility chop days

### What NOT to Do

- Don't over-optimize parameters (curve-fitting risk)
- Don't remove friction to "see the true edge"
- Don't trade shorts until market regime analysis suggests it

---

## Appendix: Commands Reference

```bash
# Run baseline backtest (both directions)
python scripts/run_unicorn_backtest.py \
  --symbol NQ \
  --databento-csv path/to/data.csv \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --slippage-ticks 2 \
  --commission 5.00 \
  --intrabar-policy worst

# Run with long-only filter (recommended)
python scripts/run_unicorn_backtest.py \
  --symbol NQ \
  --databento-csv path/to/data.csv \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --slippage-ticks 2 \
  --commission 5.00 \
  --intrabar-policy worst \
  --long-only

# Production profile (recommended guards)
python scripts/run_unicorn_backtest.py \
  --symbol NQ \
  --databento-csv path/to/data.csv \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --slippage-ticks 1 \
  --commission 2.50 \
  --intrabar-policy worst \
  --max-wick-ratio 0.6 \
  --min-displacement-atr 0.5

# Run diagnostic analysis
python scripts/diagnose_unicorn.py \
  --csv path/to/data.csv \
  --symbol NQ \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --mode all
```

---

## Session Filter Validation

Sweep of 4 session profiles across 5 market regime windows (2021-2025, 180-day each), both NQ and ES. Production guards active: `max_wick_ratio=0.6`, `min_displacement_atr=0.5`, `intrabar_policy=WORST`.

### NQ — Expectancy (pts/trade)

| Window | Character | WIDE | NORMAL | STRICT | NY_OPEN |
|--------|-----------|-----:|-------:|-------:|--------:|
| W1 | Recovery | +0.93 | +5.85 | -0.15 | -1.98 |
| W2 | Bear | +9.07 | +2.67 | +21.13 | -0.28 |
| W3 | AI rally | +1.49 | -7.43 | -12.35 | -6.03 |
| W4 | Bull cont. | -0.83 | -3.95 | -6.39 | -9.55 |
| W5 | Recent | +5.06 | +1.63 | -11.19 | +4.82 |

### ES — Expectancy (pts/trade)

| Window | Character | WIDE | NORMAL | STRICT | NY_OPEN |
|--------|-----------|-----:|-------:|-------:|--------:|
| W1 | Recovery | +0.12 | -0.20 | +1.36 | +3.51 |
| W2 | Bear | +2.35 | +2.17 | +5.31 | +4.19 |
| W3 | AI rally | +2.01 | +2.21 | +1.22 | +4.22 |
| W4 | Bull cont. | +2.06 | +3.12 | +0.66 | +0.45 |
| W5 | Recent | +1.62 | +2.32 | -0.17 | -0.41 |

### WIDE per-session breakdown (NQ, taken trades)

NY AM is **not** the structural winner for NQ. It produces negative expectancy in 3 of 5 windows (W1, W3, W5). NY PM and London alternate as the strongest sessions. The "NY AM dominates" thesis holds for ES (positive in 4/5 windows) but not NQ.

### Decision

STRICT fails to beat NORMAL in >=4/5 windows for either instrument. NY_OPEN fails to beat STRICT. **NORMAL stays as default.** The session filter adds an `NY_OPEN` profile for future analysis but does not change production defaults.

Sweep script: `scripts/run_session_sweep.py`

### Next: Soft session weighting (not gating)

The sweep disproved hard session filtering but revealed a useful signal: session edge is non-stationary and instrument-specific. The correct next layer is a **soft PD-array modifier** — session acts as a contextual probability tilt on confidence, not a structural gate.

**Proposed config surface** (backward-compatible):

```python
# UnicornConfig
session_weight_map: dict[TradingSession, float]  # missing key → 1.0
```

- Per-symbol config (already true), so NQ vs ES divergence is naturally supported
- Apply as `confidence *= session_weight` late enough to affect trade selection but not contaminate raw pattern detection metrics
- Neutral defaults (1.0) produce bit-for-bit identical results vs current baseline

**Foundation already in place:**
- `setup_session` on every `SetupOccurrence` (taken + rejected) — session×regime table buildable from existing artifacts
- `signal_strength`/confidence path into intent generation
- Per-symbol config scoping via `UnicornConfig`

**Guardrails (add at implementation time):**
- Config validation: reject negative weights, clamp or reject NaN/inf
- Diagnostics echo: print effective weight used (even when 1.0), so "inert" is always explainable

**Definition of done for that PR:**
1. With neutral defaults, bit-for-bit identical results vs current baseline
2. With a non-neutral map, confidence distribution shifts as expected (unit test with fixed seed / canned setups)

---

*Document created: 2026-01-28*
*Session filter validation added: 2026-01-29*
*Based on analysis of NQ and ES futures, 2021-2025*
