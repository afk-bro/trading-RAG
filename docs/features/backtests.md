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
