# Backtest Tuning Admin UI

Admin interface for parameter tuning workflows: run sweeps, compare results, export analysis.

## Routes

| Route | Purpose |
|-------|---------|
| `/admin/backtests/tunes?workspace_id=...` | List all tuning sessions |
| `/admin/backtests/tunes/{tune_id}` | Tune detail with trial list |
| `/admin/backtests/leaderboard?workspace_id=...` | Global leaderboard (ranked by objective) |
| `/admin/backtests/compare?tune_id=A&tune_id=B` | Side-by-side tune comparison |

## Leaderboard

### Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `valid_only` | `true` | Only tunes with a winning trial (best_run_id exists) |
| `include_canceled` | `false` | Include canceled tunes |
| `objective_type` | (all) | Filter: `sharpe`, `sharpe_dd_penalty`, `return`, `return_dd_penalty`, `calmar` |
| `oos_enabled` | (all) | Filter: `true` (with OOS split), `false` (IS only) |

### Semantics

- **Valid**: Tune has `best_run_id` (at least one trial passed gates)
- **Canceled + Valid**: Tune was canceled but some trials completed successfully before cancellation
- **Ordering**: `objective_score DESC` → `score_oos DESC` → `best_score DESC` → `created_at DESC` → `id ASC`

### Export

**CSV**: `?format=csv` appended to any leaderboard URL

Filename: `leaderboard_<workspace>_<objective>_<YYYYMMDD_HHMM>.csv`

Columns:
- Core: rank, tune_id, created_at, status, strategy_entity_id, strategy_name
- Config: objective_type, objective_params, oos_ratio, gates_*
- Winner: best_run_id, best_params, best_objective_score, best_score
- OOS metrics: return_pct, sharpe, max_drawdown_pct, trades, profit_factor
- Robustness: overfit_gap

## Compare

### URL Pattern

```
/admin/backtests/compare?tune_id=<A>&tune_id=<B>[&tune_id=<C>...]&workspace_id=<ws>
```

Supports 2+ tunes. N-way comparison renders as N columns.

### Diff Table Sections

1. **Identity**: Strategy, Status, Objective Type, λ (dd_lambda), OOS Ratio, Gates
2. **Winning Metrics**: Objective Score, Return %, Sharpe, Max DD %, Trades, Overfit Gap
3. **Best Params**: Union of all param keys across tunes

Rows with differing values are highlighted.

### Overfit Gap Thresholds

| Gap | Interpretation | UI |
|-----|----------------|-----|
| < 0 | OOS better than IS (rare, good) | Green |
| 0 - 0.3 | Normal | Default |
| 0.3 - 0.5 | Moderate overfit risk | Yellow |
| > 0.5 | High overfit risk | Red |

### Export

**JSON**: `?format=json` appended to compare URL

Filename: `compare_<tuneA>_<tuneB>_<YYYYMMDD_HHMM>.json`

Structure:
```json
{
  "generated_at": "...",
  "tunes": [
    {
      "tune_id": "...",
      "label": "A",
      "strategy": {...},
      "objective": {"type": "...", "params": {...}},
      "gates": {...},
      "best": {
        "run_id": "...",
        "objective_score": ...,
        "overfit_gap": ...,
        "metrics_oos": {...},
        "params": {...}
      }
    }
  ],
  "rows": [
    {"section": "Identity", "field": "Objective Type", "values": [...], "diff": true}
  ]
}
```

## Gates Policy

Gates are evaluated against winning trial metrics:

| Gate | Field | Threshold |
|------|-------|-----------|
| Max Drawdown | `max_drawdown_pct` | ≤ 20% (env: `GATE_MAX_DD_PCT`) |
| Min Trades | `trades` | ≥ 5 (env: `GATE_MIN_TRADES`) |

**Evaluated On**:
- `oos` when OOS split is enabled (`oos_ratio > 0`)
- `primary` when no split (IS-only tuning)

Gates snapshot is persisted at tune creation for audit trail.

## Objective Functions

| Type | Formula |
|------|---------|
| `sharpe` | Sharpe ratio |
| `sharpe_dd_penalty` | `sharpe - λ × max_drawdown_pct` |
| `return` | Return % |
| `return_dd_penalty` | `return_pct - λ × max_drawdown_pct` |
| `calmar` | `return_pct / abs(max_drawdown_pct)` |

Default λ: 0.02 (configurable via `objective_params.dd_lambda`)
