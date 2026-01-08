# Parameter Tuning Design

**Date:** 2026-01-08
**Status:** Approved for implementation

## Overview

Add parameter sweep functionality to the backtesting system. Users can run grid or random searches over strategy parameters to find optimal configurations.

## Goals

1. Zero-friction tuning: omit `param_space` to auto-derive from `param_schema`
2. Sync-first execution with async-compatible API shape
3. Deterministic results via seed support
4. Conservative guardrails against runaway searches

## Schema

### backtest_tunes

```sql
CREATE TABLE backtest_tunes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    strategy_entity_id UUID NOT NULL REFERENCES kb_entities(id) ON DELETE CASCADE,
    strategy_spec_id UUID REFERENCES kb_strategy_specs(id) ON DELETE SET NULL,

    -- Search configuration
    search_type TEXT NOT NULL CHECK (search_type IN ('grid', 'random')),
    n_trials INTEGER NOT NULL CHECK (n_trials > 0 AND n_trials <= 200),
    seed INTEGER,
    param_space JSONB NOT NULL,

    -- Objective configuration
    objective_metric TEXT NOT NULL DEFAULT 'sharpe',
    min_trades INTEGER NOT NULL DEFAULT 5,

    -- Execution state
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','completed','failed')),
    trials_completed INTEGER DEFAULT 0,

    -- Results (cached, derived from tune_runs)
    best_run_id UUID REFERENCES backtest_runs(id) ON DELETE SET NULL,
    leaderboard JSONB,

    -- Timing & errors
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT
);

CREATE INDEX idx_tunes_workspace ON backtest_tunes(workspace_id);
CREATE INDEX idx_tunes_strategy ON backtest_tunes(strategy_entity_id, created_at DESC);
```

### backtest_tune_runs

```sql
CREATE TABLE backtest_tune_runs (
    tune_id UUID NOT NULL REFERENCES backtest_tunes(id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
    trial_index INTEGER NOT NULL,
    params JSONB NOT NULL,
    score DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','completed','failed','skipped')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (tune_id, run_id),
    UNIQUE (tune_id, trial_index)
);

CREATE INDEX idx_tune_runs_tune_id ON backtest_tune_runs(tune_id);
CREATE INDEX idx_tune_runs_score ON backtest_tune_runs(tune_id, score DESC NULLS LAST);
CREATE INDEX idx_tune_runs_status ON backtest_tune_runs(tune_id, status);
```

## API Endpoints

### POST /backtests/tune

Create and run a parameter tuning session.

**Request (multipart form):**
- `file`: OHLCV CSV file
- `strategy_entity_id`: UUID
- `workspace_id`: UUID
- `search_type`: "grid" | "random" (default: random)
- `n_trials`: int (default: 50, max: 200)
- `seed`: int (optional, for reproducibility)
- `param_space`: JSON string (optional, auto-derived if omitted)
- `initial_cash`, `commission_bps`, `slippage_bps`: backtest settings
- `objective_metric`: "sharpe" | "return" | "calmar" (default: sharpe)
- `min_trades`: int (default: 5)

**Response:**
```json
{
  "tune_id": "uuid",
  "status": "completed",
  "search_type": "random",
  "n_trials": 50,
  "trials_completed": 50,
  "best_run_id": "uuid",
  "best_params": {"period": 15, "threshold": 1.8},
  "best_score": 1.82,
  "leaderboard": [
    {"rank": 1, "run_id": "...", "params": {...}, "score": 1.82, "summary": {...}},
    ...
  ]
}
```

### GET /backtests/tunes/{tune_id}

Get tune details and current leaderboard.

### GET /backtests/tunes/{tune_id}/runs

List trial runs with pagination and filtering.

**Query params:** `limit`, `offset`, `status`

### GET /backtests/tunes

List tunes for workspace/strategy.

**Query params:** `workspace_id`, `strategy_entity_id`, `limit`

### DELETE /backtests/tunes/{tune_id}

Delete tune and all associated tune_runs (cascade).

## Service Architecture

```
app/services/backtest/
├── scoring.py   # NEW: objective functions (pure)
├── tuner.py     # NEW: orchestration
├── runner.py    # existing single-run logic
├── data.py      # existing CSV parsing
└── validate.py  # existing param validation
```

### scoring.py

Pure functions for objective computation:

```python
def compute_score(
    summary: dict,
    objective: str = "sharpe",
    min_trades: int = 5,
) -> float | None:
    """
    Returns None if:
    - trades < min_trades (skipped)
    - objective metric is missing/NaN (skipped)
    """
```

### tuner.py

Orchestration class:

```python
class ParamTuner:
    max_concurrency = 4

    async def run(self, tune_id, file_content, ...) -> TuneResult:
        # 1. Parse CSV once, reuse DataFrame
        # 2. Generate param combinations
        # 3. Insert all tune_runs as 'queued'
        # 4. Run trials with semaphore-bounded concurrency
        # 5. Update leaderboard cache
        # 6. Return results
```

## Param Space Derivation

When `param_space` is omitted, derive from `param_schema`:

| Schema property | Derived space |
|-----------------|---------------|
| `enum` | Use enum values directly |
| `minimum` + `maximum` (integer) | 5 points around default: ±30%, ±15%, default |
| `minimum` + `maximum` (float, grid) | 5 points: default × [0.7, 0.85, 1.0, 1.15, 1.3] |
| `minimum` + `maximum` (float, random) | `{"min": X, "max": Y, "type": "float"}` |
| `default` only | Fixed value (not tuned) |

**Guardrails:**
- Grid combinations capped at 200
- If exceeds cap: auto-switch to random or reduce points per dimension

## Execution Flow

1. **Create tune row** with status='queued' (persists immediately)
2. **Parse CSV once** into DataFrame
3. **Generate param combinations** (grid or random with seed)
4. **Insert all tune_runs** with status='queued'
5. **Execute trials** with `asyncio.Semaphore(4)`:
   - Update tune_run to 'running'
   - Call existing `BacktestRunner.run()`
   - Compute score via `scoring.compute_score()`
   - Update tune_run to 'completed'/'failed'/'skipped'
   - Maintain in-memory top-10 heap
6. **Update tune row**: best_run_id, leaderboard cache, status='completed'
7. **Return response**

## Key Invariants

1. **Tune persists first** - Row created before any trials run
2. **CSV parsed once** - DataFrame reused across all trials
3. **Scoring is centralized** - `scoring.compute_score()` is the single source
4. **Skipped is first-class** - status='skipped', score=NULL for insufficient trades
5. **Leaderboard is derived** - Cached on tune row, rebuildable from tune_runs

## Testing Plan

| Test | Purpose |
|------|---------|
| Unit: `compute_score()` | All objectives, edge cases (0 trades, None sharpe, NaN) |
| Unit: `derive_param_space()` | Various schema shapes |
| Unit: param generation | Grid size estimation, random seed determinism |
| Unit: grid explosion guard | Auto-switch to random when exceeding cap |
| Integration: full tune | Small CSV, verify leaderboard ordering, best_run selection |

## Future Extensions (not in v1)

- Async execution for large searches
- Bayesian optimization
- Multi-objective Pareto frontiers
- Walk-forward analysis
- Early stopping for poor performers
