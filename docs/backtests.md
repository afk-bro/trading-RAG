# Backtest Parameter Tuning

Research workflow for strategy optimization via parameter sweeps.

## Core Concepts

- `tune` - A parameter sweep session (grid/random search over param space)
- `tune_run` - A single trial within a tune (one param combination)
- IS/OOS split - In-Sample (training) and Out-of-Sample (validation) data split

## Objective Functions

Located in `app/services/backtest/tuner.py`:

| Function | Formula |
|----------|---------|
| `sharpe` | Sharpe ratio |
| `sharpe_dd_penalty` | `sharpe - λ × max_drawdown_pct` |
| `return` | Return percentage |
| `return_dd_penalty` | `return_pct - λ × max_drawdown_pct` |
| `calmar` | `return_pct / abs(max_drawdown_pct)` |

## Gates Policy

Trials must pass gates to be considered valid:
- Max drawdown ≤20%
- Min trades ≥5

Gates are evaluated on OOS metrics when split is enabled.

## Overfit Detection

`overfit_gap = score_is - score_oos`
- Gap >0.3: Moderate overfit risk
- Gap >0.5: High overfit risk

## Admin UI

Located in `app/admin/router.py`, `app/admin/templates/`:

| Route | Description |
|-------|-------------|
| `/admin/backtests/tunes` | Filterable tune list with validity badges |
| `/admin/backtests/tunes/{id}` | Tune detail with trial list |
| `/admin/backtests/leaderboard` | Global ranking by objective score (CSV export) |
| `/admin/backtests/compare?tune_id=A&tune_id=B` | N-way diff table (JSON export) |

## API Endpoints

```
POST /backtests/tune              - Start parameter sweep
GET  /backtests/tunes             - List tunes (filters: valid_only, objective_type, oos_enabled)
GET  /backtests/tunes/{id}        - Tune detail with trial list
POST /backtests/tunes/{id}/cancel - Cancel running tune
GET  /backtests/leaderboard       - Global ranking with best run metrics
```

---

# Walk-Forward Optimization (WFO)

Validates strategy robustness by running parameter tuning across rolling time windows (folds), then aggregating results to find params that perform consistently across different market conditions.

## Architecture

```
WFO Request
    │
    ▼
┌─────────────────┐
│   Fold Planner  │ ──► Generate train/test windows
└────────┬────────┘
         │
         ▼ (N folds)
┌─────────────────┐
│  Child TuneJobs │ ──► One tune per fold (parallel)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Aggregator    │ ──► Combine results across folds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Best Candidate │ ──► Params meeting coverage + selection score
└─────────────────┘
```

**Source**: `app/services/backtest/wfo.py`, `app/jobs/handlers/wfo.py`

## Core Concepts

| Term | Definition |
|------|------------|
| Fold | Train/test window pair: `[train_start, train_end) [test_start, test_end)` |
| Candidate | Param set appearing in fold leaderboard (top-K results) |
| Coverage | % of folds where candidate appears in top-K (min 60%) |
| Selection Score | Ranking formula for eligible candidates |

## WFO Configuration

`WFOConfig` fields:

| Field | Description | Default |
|-------|-------------|---------|
| `train_days` | Days in training window (IS period) | - |
| `test_days` | Days in test window (OOS period) | - |
| `step_days` | Days to step forward between folds | - |
| `min_folds` | Minimum folds required | 3 |
| `leaderboard_top_k` | Top params per fold | 10 |
| `allow_partial` | Continue even if some folds fail | False |

## Selection Score Formula

```python
score = (mean_oos - 0.5 * stddev_oos - 0.3 * max(0, threshold - worst_fold_oos)) * coverage
```

Higher is better. Penalizes:
- High variance (via stddev)
- Poor worst-case performance (via worst_fold penalty)
- Low coverage (multiplicative factor)

## API Endpoints

```
POST   /backtests/wfo              - Queue WFO job
GET    /backtests/wfo/{wfo_id}     - Get WFO details with candidates
GET    /backtests/wfo              - List WFO runs (filters: workspace_id, status)
POST   /backtests/wfo/{wfo_id}/cancel - Cancel pending/running WFO
DELETE /backtests/wfo/{wfo_id}     - Delete WFO record
```

## Database Schema

`wfo_runs` table (Migration 070):

| Column | Type | Description |
|--------|------|-------------|
| `wfo_config` | JSONB | train_days, test_days, step_days, min_folds, etc. |
| `param_space` | JSONB | Parameter search space |
| `data_source` | JSONB | exchange_id, symbol, timeframe |
| `status` | TEXT | pending, running, completed, partial, failed, canceled |
| `best_params` | JSONB | Winner params |
| `best_candidate` | JSONB | Full `WFOCandidateMetrics` of winner |
| `candidates` | JSONB | All eligible candidates sorted by selection score |
| `child_tune_ids` | UUID[] | Child tune job references |

## WFO Result Fields

`WFOCandidateMetrics`:
- `params`, `params_hash` - Parameter values and SHA256 for deduplication
- `mean_oos`, `median_oos`, `worst_fold_oos`, `stddev_oos` - OOS score statistics
- `pct_top_k` - Coverage percentage
- `fold_count` / `total_folds` - Coverage numerator/denominator
- `fold_scores` - Individual OOS scores per fold (diagnostics)

## Prometheus Metrics

Located in `app/routers/metrics.py`:

| Metric | Type | Description |
|--------|------|-------------|
| `wfo_runs_total{status}` | Counter | WFO runs by status |
| `wfo_run_duration_seconds` | Histogram | Duration (1m to 4h buckets) |
| `wfo_folds_total{status}` | Counter | Fold completions |
| `wfo_candidates_eligible` | Gauge | Eligible candidate count |
| `wfo_best_score` | Gauge | Best selection score |
