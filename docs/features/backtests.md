# Backtest System

Research workflow for strategy optimization via parameter sweeps.

## Parameter Tuning

**Core Concepts**:
- `tune` - A parameter sweep session (grid/random search over param space)
- `tune_run` - A single trial within a tune (one param combination)
- IS/OOS split - In-Sample (training) and Out-of-Sample (validation) data split

**Objective Functions** (`app/services/backtest/tuner.py`):
- `sharpe` - Sharpe ratio
- `sharpe_dd_penalty` - `sharpe - λ × max_drawdown_pct`
- `return` - Return percentage
- `return_dd_penalty` - `return_pct - λ × max_drawdown_pct`
- `calmar` - `return_pct / abs(max_drawdown_pct)`

**Gates Policy**: Trials must pass gates (max drawdown ≤20%, min trades ≥5) to be considered valid. Gates are evaluated on OOS metrics when split is enabled.

**Overfit Detection**: `overfit_gap = score_is - score_oos`. Gap >0.3 indicates moderate overfit risk, >0.5 is high risk.

**Admin UI** (`app/admin/router.py`, `app/admin/templates/`):
- `/admin/backtests/tunes` - Filterable tune list with validity badges
- `/admin/backtests/tunes/{id}` - Tune detail with trial list
- `/admin/backtests/leaderboard` - Global ranking by objective score (CSV export)
- `/admin/backtests/compare?tune_id=A&tune_id=B` - N-way diff table (JSON export)

## Walk-Forward Optimization (WFO)

Validates strategy robustness by running parameter tuning across rolling time windows (folds), then aggregating results to find params that perform consistently across different market conditions.

**Architecture** (`app/services/backtest/wfo.py`, `app/jobs/handlers/wfo.py`):
```
WFO Request → Fold Planner → Child TuneJobs (parallel) → Aggregator → Best Candidate
```

**Core Concepts**:
- `Fold`: A train/test window pair using half-open intervals `[train_start, train_end) [test_start, test_end)`
- `Candidate`: A param set that appears in fold leaderboard (top-K results)
- `Coverage`: Percentage of folds where a candidate appears in top-K (must meet MIN_COVERAGE_RATIO = 60%)
- `Selection Score`: Ranking formula for eligible candidates

**WFO Configuration** (`WFOConfig`):
- `train_days`: Days in training window (IS period)
- `test_days`: Days in test window (OOS period)
- `step_days`: Days to step forward between folds
- `min_folds`: Minimum folds required (default 3)
- `leaderboard_top_k`: Top params per fold (default 10)
- `allow_partial`: Continue even if some folds fail (default False)

**Selection Score Formula**:
```python
score = (mean_oos - 0.5 * stddev_oos - 0.3 * max(0, threshold - worst_fold_oos)) * coverage
```

Higher is better. Penalizes high variance, poor worst-case performance, and low coverage.

**Database Schema** (`wfo_runs` table - Migration 070):
- `wfo_config` (JSONB): train_days, test_days, step_days, min_folds, etc.
- `param_space` (JSONB): Parameter search space
- `data_source` (JSONB): exchange_id, symbol, timeframe
- `status`: pending, running, completed, partial, failed, canceled
- `best_params` (JSONB): Winner params
- `best_candidate` (JSONB): Full `WFOCandidateMetrics` of winner
- `candidates` (JSONB): All eligible candidates sorted by selection score
- `child_tune_ids` (UUID[]): Child tune job references

**WFO Result Fields** (`WFOCandidateMetrics`):
- `params`, `params_hash`
- `mean_oos`, `median_oos`, `worst_fold_oos`, `stddev_oos`
- `pct_top_k`, `fold_count`, `total_folds`, `fold_scores`

**Prometheus Metrics** (`app/routers/metrics.py`):
- `wfo_runs_total{status}`, `wfo_run_duration_seconds`
- `wfo_folds_total{status}`, `wfo_candidates_eligible`, `wfo_best_score`

## Test Generator & Run Orchestrator

Parameter sweep framework for systematic strategy testing (`app/services/testing/`).

```
ExecutionSpec → Test Generator → RunPlan → Run Orchestrator → RunResult
```

**Key Design Decisions**:
- **Variant ID**: `sha256(canonical_json({base, overrides}))[:16]` - deterministic, stable
- **Overrides format**: Flat dotted-path dict only (e.g., `{"entry.lookback_days": 200}`)
- **Broker isolation**: Each variant uses uuid5 namespace - prevents cross-contamination
- **Equity tracking**: Trade-equity points (step function) for drawdown/sharpe

**Sweepable Parameters**:
- `entry.lookback_days`, `risk.dollars_per_trade`, `risk.max_positions`

**Generator Output**:
1. Baseline variant (empty overrides) - always first
2. Grid sweep variants (cartesian product)
3. Ablation variants (reset one param to default)
4. Deduplication + max_variants limit

**Metrics Calculated**:
- `return_pct`, `max_drawdown_pct`, `sharpe`, `win_rate`, `trade_count`, `profit_factor`

**Admin UI**:
- `/admin/testing/run-plans` - List page
- `/admin/testing/run-plans/{id}` - Detail page

## Strategy Backtest UI

Interactive backtesting interface for running and analyzing strategy performance.

**Route**: `/admin/backtests/strategy-test`

**Features**:
- Strategy selection (ICT Unicorn Model)
- Symbol selection (ES, NQ futures)
- Date range picker with data availability validation
- Direction filter (long-only, short-only, both)
- Min criteria score (3-5 of 5)
- Trading platform presets with realistic commissions:
  - Apex (Rithmic): $3.98/RT ($1.99/side)
  - Apex (Tradovate): $3.10/RT ($1.55/side)
  - Topstep: $2.80/RT (~$1.40/side)
  - Custom: user-defined
- Friction settings: slippage (ticks), intrabar policy

**R-Metrics Display** (prioritized over dollar-based metrics):
- **Avg R per Trade**: Edge metric, portable across contract sizes
- **Total R**: Sum of all R-multiples
- **Max Win/Loss**: Best and worst R-multiple trades
- **Expectancy Analysis**: Visual formula breakdown
  - `E = (Win% × Avg Win R) − (Loss% × |Avg Loss R|)`
- **R Distribution**: Bar chart showing trade bucketing by R-multiple
  - `< -1.0R`, `-1.0 to 0R`, `0 to +1R`, `+1 to +2R`, `+2R+`
  - P(≥+1R) and P(≥+2R) probability badges
- **Warning Badges**: "No +2R+ wins" when fat tails missing (capped exits)

**Additional Analytics**:
- Session breakdown (London, NY AM, NY PM, Asia) with win rate and PnL
- Exit reasons pie chart (target, stop_loss, time_stop, session_end)
- Criteria bottleneck analysis (why setups fail)
- MFE/MAE analysis (maximum favorable/adverse excursion)
- Confidence buckets correlation

**Implementation**:
- Backend: `app/admin/strategy_backtest.py`
- Templates: `app/admin/templates/strategy_backtest.html`, `strategy_backtest_results.html`
- Engine: `app/services/backtest/engines/unicorn_runner.py`

## ICT Unicorn Model Strategy

Discretionary trading strategy based on ICT methodology.

**8-Criteria Checklist**:

*Mandatory (must pass all 3)*:
1. Previous day closed in premium/discount (>60% or <40% of range)
2. Current price in opposite zone (discount if prev premium, vice versa)
3. FVG present in LTF for entry

*Scored (need 3+ of 5)*:
4. HTF trend alignment (SMA slope)
5. Session timing (London/NY AM preferred)
6. Liquidity sweep before entry
7. Displacement candle confirmation
8. Order block alignment

**Key Features**:
- Multi-timeframe analysis (HTF: 15m, LTF: 1m)
- Session filtering (London, NY AM, NY PM, Asia)
- Volatility-normalized sizing via ATR
- Direction filter support (long-only outperforms on NQ)
- Time-stop: Exit trades not hitting TP/SL within session
- Look-ahead bias prevention in entry logic
- Pre-entry guards (wick, range, displacement) — conviction filters on signal quality

**Configuration** (`UnicornConfig`):
- `min_scored_criteria`: Minimum scored criteria (default 3)
- `session_profile`: strict / normal / wide
- `slippage_ticks`: Entry/exit slippage
- `commission_per_contract`: Round-turn commission
- `intrabar_policy`: worst (conservative), best (optimistic), random, ohlc_path
- `direction_filter`: long_only, short_only, both
- `max_wick_ratio`: Skip entry if signal bar adverse wick ratio exceeds threshold (None=disabled)
- `max_range_atr_mult`: Skip entry if signal bar range exceeds ATR multiple (None=disabled)
- `min_displacement_atr`: Skip entry if MSS displacement < this ATR multiple (None=disabled, recommended: 0.5)

**Pre-Entry Guards** (applied after criteria scoring, before trade creation):

| Guard | Config | Rejects when | Recommended |
|-------|--------|-------------|-------------|
| Wick | `max_wick_ratio` | Adverse wick / range > threshold | 0.6 |
| Range | `max_range_atr_mult` | Bar range / ATR > threshold | None |
| Displacement | `min_displacement_atr` | MSS displacement < threshold ATR | 0.5 |

Guards are pre-entry filters, not scoring criteria. They gate on signal quality after the 8-criteria checklist passes. Diagnostics (`signal_wick_ratio`, `signal_range_atr_mult`, `signal_displacement_atr`) are recorded on every qualifying setup regardless of which guard rejects, enabling sweep analysis.

The displacement guard was validated across multiple historical regimes, sub-window splits, and ATR normalizations using deterministic worst-case intrabar execution. Performance gains are regime-robust and scale-invariant.

**Recommended production flags**:
```bash
--session-profile normal --max-wick-ratio 0.6 --min-displacement-atr 0.5 --intrabar-policy worst --trail-atr-mult 1.5
```

Session profile validated across 5 regime windows (2021-2025). STRICT and NY_OPEN failed decision criteria (expectancy >= baseline in >=4/5 windows). NORMAL remains default.

**Trailing Stop** (exit management):
- `--trail-atr-mult FLOAT`: ATR-based trailing stop distance (e.g. 1.5 = 1.5x ATR)
- `--trail-cap-mult FLOAT`: Cap trail distance at N × risk_points (default: 1.0 in eval mode, uncapped otherwise)
- `--trail-activate-r FLOAT`: MFE threshold to activate trail (default: 1.0 = +1R)
- Trail distance = min(entry_atr × mult, cap_mult × risk_points) — frozen at entry
- Activates at configured R-multiple MFE with breakeven floor
- Ratchets only forward behind favorable extreme (trail_high for longs, trail_low for shorts)
- Activation and ratchet happen on the same bar (no delay)
- Exit reason: `trail_stop` (vs `stop_loss` for original stop hits)
- Mutually exclusive with `--breakeven-at-r`
- Works with `--eval-mode` for prop-firm simulation (auto-caps at 1.0R)

Validated defaults (MNQ 18-month backtest, eval mode, 2024-01 to 2025-07):
- `--trail-activate-r 1.0` — lowering to 0.5R increases activation (38%→55%) but harvests structurally inferior trades (avg R drops +0.85R→+0.42R, PF drops 0.65→0.59). MFE distribution shows only 32% of trades reach +1R; the 21% gap between +0.5R and +1R doesn't carry enough excursion.
- `--trail-cap-mult 1.0` (eval auto-default) — tighter caps (0.75) trade smoothness for clipping; wider (1.25+) donate profit back. 1.0R balances eval safety with trend capture.
- Trail vs fixed 2R target: win rate +14.6%, expectancy +2.78 pts/trade, MFE capture 26%→50% for trail exits, best R uncapped to +3.41R.

**Lessons Learned**:
- Direction matters: Long-only on NQ improved PF from 0.91 to 1.34
- Time-stops reduce exposure without hurting edge
- Wick guard at 0.6 reduced evil-profile bleed from -$957 to -$298
- Displacement guard at 0.5x ATR improves NQ expectancy +52%, turns ES from breakeven to profitable
- Trailing stop resolved capped exits: 40% full-stop rate with 0% +2R+ wins → 38% activation with +0.85R avg trail exits
- Exit engineering is now data-validated; entry selectivity is the bottleneck (32% of trades reach +1R MFE)
- See `docs-archive` branch for forensic analysis details

## Coaching (Post-Run Experience)

Transforms the backtest post-run experience from results-first to experiment-oriented. Every run feels informative regardless of outcome by showing progress, process quality, and explained losses.

**Architecture** (`app/services/backtest/`):
```
Run Completed → Coaching Data Builder (500ms budget)
                ├── Process Score (process_score.py)
                ├── Loss Attribution (loss_attribution.py)
                └── Run Lineage + Deltas (run_lineage.py)
```

### Process Score

Composite 0–100 score measuring execution quality independent of outcome.

**Components** (weights redistribute proportionally when unavailable):

| Component | Weight | Source | Logic |
|-----------|--------|--------|-------|
| Rule Adherence | 0.25 | run_events | % of `entry_signal` preceded by `setup_valid` within bar window |
| Regime Alignment | 0.20 | regime + trades | % of trades aligned with regime direction (uptrend→long, downtrend→short) |
| Risk Discipline | 0.25 | trades | worst/median loss ratio + position sizing CV |
| Exit Quality | 0.15 | trades | avg winner / avg loser return ratio |
| Consistency | 0.15 | trades | CV of return_pct (N ≥ 5 required) |

**Grade mapping**: A (80+), B (65–79), C (50–64), D (35–49), F (0–34), `unavailable` (>50k trades or no data).

### Loss Attribution

Analyzes losing trades with clustering and actionable counterfactuals.

**Outputs**:
- **Time clusters**: Losses bucketed by hour-of-day from `t_entry`
- **Size clusters**: Losses bucketed by |pnl| into small/medium/large
- **Regime context**: Run-level regime tags with aggregate loss stats
- **Policy counterfactuals** (equity recomputed, compounding preserved):
  - "If you skipped trades during hour X, return would be Y%"
  - "If you enforced regime filter, drawdown would be X%"
  - "If you capped max trades/day at K, return would be Y%"
  - "If you skipped the first N minutes after ORB lock, return would be Y%"

**Compute ceiling**: >50k trades returns counts only (no counterfactual recompute).

### Run Lineage

Detects the previous completed run for the same strategy and computes deltas.

**Functions**:
- `compute_deltas()` — KPI deltas with `HIGHER_IS_BETTER` map and `improved` flag
- `compute_param_diffs()` — changed params with JSON-serialized `[old, new]` values
- `compute_comparison_warnings()` — symbol, timeframe, date range, run_kind mismatches
- `build_trajectory()` — last N runs ordered by `completed_at` for sparklines
- `find_previous_run()` / `find_lineage_candidates()` — async DB queries

All ordering uses `completed_at`, not `created_at`.

### API

**Extended endpoint** — `GET /dashboards/{ws}/backtests/{run_id}`:
- `?include_coaching=true` — adds `coaching` and `trajectory` keys to response
- `?baseline_run_id=UUID` — override auto-detected previous run

**New endpoint** — `GET /dashboards/{ws}/backtests/{run_id}/lineage`:
- Returns recent runs for the same strategy (powers baseline selector dropdown)

**Timeout budget**: 500ms total. I/O phase uses `asyncio.wait()` with partial recovery — completed tasks return even if others time out. `coaching_partial: true` flag signals degraded data.

### Dashboard Components

Seven components in `dashboard/src/components/coaching/`:

| Component | Purpose |
|-----------|---------|
| `WhatYouLearnedCard` | Improved/tradeoffs/unchanged columns, first-run "Baseline captured" state, comparison warnings, expandable param diff |
| `ProcessScoreCard` | SVG gauge arc (270°) with grade letter, component bars, focus area sentence |
| `DeltaKpiStrip` | 6-metric grid with delta arrows + sparklines (falls back to plain display without lineage) |
| `DeltaKpiCard` | Enhanced KPI card with delta value, color coding, optional sparkline |
| `BaselineSelector` | Dropdown for manual baseline override with "(auto)" badge |
| `LossAttributionPanel` | Collapsible: time-of-day bars, size buckets, regime context, counterfactual cards with "Hypothetical" badge |
| `TrajectorySparkline` | Pure SVG polyline mini chart |

**Results tab layout**: WhatYouLearned → ProcessScore → BaselineSelector + DeltaKpiStrip → EquityChart → LossAttribution → TradesTable.

All coaching sections guarded by optional chaining — non-completed runs and `include_coaching=false` show the existing layout unchanged.

### Tests

- `tests/unit/test_process_score.py` — 36 tests (components, weight redistribution, ceiling, grade boundaries, NaN/Inf)
- `tests/unit/test_loss_attribution.py` — 19 tests (clusters, counterfactuals, edge cases)
- `tests/unit/test_run_lineage.py` — 22 tests (deltas, param diffs, comparison warnings)
- `tests/e2e/test_coaching_api.py` — 8 Playwright API smoke tests (contract shape, backward compat)

## Eval Mode (Prop-Firm Simulation)

Simulates a prop-firm evaluation with trailing drawdown limits and R-native position sizing.

**Risk Model** (`app/services/backtest/engines/eval_profile.py`):

Per-trade risk is derived from remaining drawdown room rather than a fixed dollar budget:

```
R_day = clamp(room * risk_fraction, r_min, r_max)
room  = max_drawdown_dollars - (peak_equity - current_equity)
```

When `room <= 0`, the eval is blown and the backtest terminates.

**Components**:
- `EvalAccountProfile` - Frozen dataclass with account parameters (immutable during run)
- `compute_r_day()` - Pure function: `(profile, equity, peak) -> float`
- `DailyGovernor` - Intraday stepdown + halt (composes via `effective_R = R_day * risk_multiplier`)

**Composition**: The eval profile sets the base R_day at each day boundary. The DailyGovernor applies its intraday multiplier (1.0 → 0.5 after half-loss). They compose independently.

**CLI Arguments**:
| Flag | Description | Default |
|------|-------------|---------|
| `--eval-account-size DOLLARS` | Enables R-native sizing | (required to activate) |
| `--eval-max-drawdown DOLLARS` | Trailing DD limit | 4% of account size |
| `--eval-risk-fraction FLOAT` | Fraction of room per trade | 0.15 |
| `--eval-r-min DOLLARS` | R_day floor | 100 |
| `--eval-r-max DOLLARS` | R_day cap | 300 |
| `--eval-mode` | Also enables DailyGovernor | — |
| `--max-daily-loss DOLLARS` | DailyGovernor max loss/day | 50% of max drawdown |
| `--max-trades-per-day N` | DailyGovernor trade cap | 2 |

**Example trajectory** ($50k account, $2k DD, fraction=0.15):
- Day 1: room=$2000 → R=$300 (cap)
- After -$500 drawdown: room=$1500 → R=$225
- After -$1300 drawdown: room=$700 → R=$105
- After -$1400 drawdown: room=$600 → R=$100 (floor)
- After -$2000 drawdown: room=$0 → R=0 → **eval blown**, backtest stops

**Backtest report output**: Includes EVAL ACCOUNT section with starting/final/peak equity, trailing drawdown %, R_day range, and EVAL BLOWN indicator.

**CLI example**:
```bash
python scripts/run_unicorn_backtest.py --eval-mode --symbol MNQ \
  --eval-account-size 50000 --eval-max-drawdown 2000 \
  --eval-r-min 100 --eval-r-max 300 \
  --databento-csv data/mnq.csv --start-date 2024-01-01 --end-date 2024-06-30 --multi-tf
```
