# Results Persistence vs Event-Only Design

**PR**: #5
**Status**: Ready for implementation
**Date**: 2026-01-10

## Overview

This design establishes what must be persisted, what remains event-only, and why. The goal is to enable replay, audit, and analysis of past runs while keeping the system streaming-friendly and avoiding premature over-modeling.

### Core Principle (North Star)

> **Events are for orchestration and UX. Results are for truth, analysis, and long-term value.**

Persist canonical results, not raw firehose events.

## The Three Data Layers

```
┌──────────────────────────┐
│  UI / Client             │
│  (SSE / WebSocket)       │
└──────────▲───────────────┘
           │ ephemeral events
┌──────────┴───────────────┐
│  Event Stream            │  ← Breadcrumbs only
│  - RUN_STARTED           │
│  - RUN_COMPLETED         │
│  - progress (ephemeral)  │
└──────────▲───────────────┘
           │ writes on completion
┌──────────┴───────────────┐
│  Results Persistence     │  ← SOURCE OF TRUTH
│  - run_plans             │
│  - backtest_runs         │
└──────────────────────────┘
```

## Key Design Decisions

### 1. Extend `backtest_runs` Rather Than Create New Tables

The "testing run" is conceptually a backtest run. The persisted unit of work is the same:
- A strategy version + params
- A dataset (or data selection)
- Metrics summary
- Optional artifacts (equity/trades)

**Benefits**:
- Fewer endpoints and UI components
- Easier RAG ingestion (one place to pull "trial results")
- Unified query patterns for leaderboard/compare/history

### 2. Add `run_plans` as Grouping Table

One run plan → many `backtest_runs` (variants).

```
run_plans (grouping/orchestration container)
    │
    │ 1:N
    ▼
backtest_runs (one row per variant attempt)
```

### 3. Inline JSONB for Artifacts (v1)

Keep `equity_curve` and `trades` as JSONB in `backtest_runs`. Add escape hatches for future external storage.

### 4. Dual-Write Coarse Lifecycle Events Only

Continue journaling `RUN_STARTED`/`RUN_COMPLETED` to `trade_events` as observability breadcrumbs. Tables are the source of truth.

## Schema Changes

### New Table: `run_plans`

```sql
CREATE TABLE run_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    strategy_entity_id UUID NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    objective_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL,
    n_variants INT NOT NULL DEFAULT 0,
    n_completed INT NOT NULL DEFAULT 0,
    n_failed INT NOT NULL DEFAULT 0,
    n_skipped INT NOT NULL DEFAULT 0,
    best_backtest_run_id UUID NULL,
    best_objective_score DOUBLE PRECISION NULL,
    error_summary TEXT NULL,
    plan JSONB NOT NULL
);

CREATE INDEX idx_run_plans_workspace ON run_plans(workspace_id, created_at DESC);
CREATE INDEX idx_run_plans_status ON run_plans(status, created_at DESC);
```

### `run_plans.plan` JSONB Structure

Three layers: inputs, resolved plan, provenance.

```json
{
  "inputs": {
    "base_spec": { "strategy_name": "breakout_52w_high", "params": {...} },
    "objective": { "name": "sharpe_dd_penalty", "direction": "maximize" },
    "constraints": { "max_variants": 25, "min_trades": 5 },
    "dataset_ref": "BTC_1h.csv",
    "generator_config": { "type": "grid", "include_ablations": true }
  },
  "resolved": {
    "n_variants": 15,
    "variants": [
      { "variant_index": 0, "params": {...}, "param_source": "baseline" },
      { "variant_index": 1, "params": {...}, "param_source": "grid" }
    ]
  },
  "provenance": {
    "generator": { "name": "grid_search_v1", "version": "1.0.0" },
    "created_at": "2026-01-10T12:00:00Z",
    "seed": null,
    "fingerprints": {
      "dataset": "sha256:abc123...",
      "strategy": "sha256:def456...",
      "plan": "sha256:789ghi..."
    }
  }
}
```

**Rules**:
- `plan` is immutable after creation
- Store resolved variants, not just ranges
- Keep large artifacts out of `plan`

### Extended: `backtest_runs`

```sql
-- New columns
ALTER TABLE backtest_runs
ADD COLUMN run_plan_id UUID NULL REFERENCES run_plans(id),
ADD COLUMN variant_index INT NULL,
ADD COLUMN variant_fingerprint TEXT NULL,
ADD COLUMN run_kind TEXT NOT NULL DEFAULT 'backtest'
    CHECK (run_kind IN ('backtest', 'tune_variant', 'test_variant')),
ADD COLUMN objective_score DOUBLE PRECISION NULL,
ADD COLUMN has_equity_curve BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN has_trades BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN equity_points INT NULL,
ADD COLUMN trade_count INT NULL,
ADD COLUMN artifacts_ref JSONB NULL,
ADD COLUMN skip_reason TEXT NULL;

-- Indexes for common queries
CREATE INDEX idx_backtest_runs_plan_variant
    ON backtest_runs(run_plan_id, variant_index);
CREATE INDEX idx_backtest_runs_plan_score
    ON backtest_runs(run_plan_id, objective_score DESC NULLS LAST);
CREATE INDEX idx_backtest_runs_kind_created
    ON backtest_runs(run_kind, created_at DESC);
```

### Column Semantics

| Column | Purpose |
|--------|---------|
| `run_plan_id` | Links to parent plan (NULL = standalone backtest) |
| `variant_index` | 0..N-1 ordering within plan |
| `variant_fingerprint` | `hash(canonical(params))` for verification |
| `run_kind` | Distinguishes standalone vs plan variants |
| `objective_score` | Extracted for fast ORDER BY (avoids JSONB cast) |
| `has_equity_curve` | Fast check without loading blob |
| `has_trades` | Fast check without loading blob |
| `equity_points` | Count for UI display |
| `trade_count` | Count for UI display |
| `artifacts_ref` | Future S3 refs: `{"equity_curve": "s3://..."}` |
| `skip_reason` | Why variant was skipped (if status = 'skipped') |

## Event Strategy

### What Gets Persisted to `trade_events` (Breadcrumbs)

Coarse lifecycle only:

| Event | When | Payload |
|-------|------|---------|
| `RUN_STARTED` | Plan begins | `run_plan_id`, `workspace_id`, `n_variants`, `objective` |
| `RUN_COMPLETED` | Plan finishes successfully | `run_plan_id`, counts, `best_backtest_run_id`, `best_objective_score` |
| `RUN_FAILED` | Plan fails | `run_plan_id`, `error_summary` |
| `RUN_CANCELLED` | Plan cancelled | `run_plan_id` |

### What Stays Ephemeral (SSE / In-Memory)

| Category | Examples |
|----------|----------|
| Progress | variant 3/20 started, bar 512/10000 |
| Heartbeats | runner alive, still computing |
| Variant lifecycle | VARIANT_STARTED, VARIANT_COMPLETED |
| Debug logs | internal prints, per-bar traces |

### Retention Policy

- Keep `trade_events` breadcrumbs for 30-90 days
- Implement cleanup job in future PR

## Query Patterns

```sql
-- All variants for a run plan
SELECT * FROM backtest_runs
WHERE run_plan_id = $1
ORDER BY variant_index;

-- Best variant (fast, uses index)
SELECT * FROM backtest_runs
WHERE run_plan_id = $1
ORDER BY objective_score DESC NULLS LAST
LIMIT 1;

-- Leaderboard across all test runs
SELECT br.*, rp.objective_name
FROM backtest_runs br
JOIN run_plans rp ON br.run_plan_id = rp.id
WHERE br.run_kind IN ('tune_variant', 'test_variant')
ORDER BY br.objective_score DESC NULLS LAST
LIMIT 100;

-- Run history for workspace
SELECT * FROM run_plans
WHERE workspace_id = $1
ORDER BY created_at DESC;
```

## API Changes

### New Verification Endpoints (This PR)

For validating the new persistence layer:

```
GET /admin/run-plans/{id}
  → Returns run_plans row (metadata + aggregates)

GET /admin/run-plans/{id}/runs
  → Returns backtest_runs for that plan (no equity/trades blobs)
```

### Existing Endpoints (Unchanged)

```
POST /testing/run-plans/generate
POST /testing/run-plans/generate-and-execute
```

These will be updated to write to the new tables instead of returning in-memory results.

## Write Path Changes

### RunOrchestrator Updates

```python
async def execute(self, run_plan: RunPlan, dataset: list[OHLCVBar]) -> list[RunResult]:
    # 1. Persist run_plan to DB (status = 'running')
    await self._persist_run_plan(run_plan, status='running')

    # 2. Dual-write RUN_STARTED event (breadcrumb)
    await self._journal_run_started(run_plan)

    results = []
    for variant in run_plan.variants:
        # 3. Execute variant
        result = await self._execute_variant(run_plan, variant, dataset)

        # 4. Persist variant result to backtest_runs
        await self._persist_variant_result(run_plan.id, variant, result)

        results.append(result)

    # 5. Update run_plan with aggregates (status = 'completed')
    await self._finalize_run_plan(run_plan, results)

    # 6. Dual-write RUN_COMPLETED event (breadcrumb)
    await self._journal_run_completed(run_plan, results)

    return results
```

## What's NOT in Scope

| Item | Reason |
|------|--------|
| External artifact storage (S3) | `artifacts_ref` escape hatch is sufficient for v1 |
| Variant-level event persistence | Captured structurally in `backtest_runs` |
| Admin UI query migration | Follow-up PR (keep on events for now) |
| Event retention cleanup job | Future PR |
| Per-bar PnL persistence | Recomputable from equity curve |

## Failure Semantics

| Scenario | Persist? | Notes |
|----------|----------|-------|
| Variant skipped | Yes | `status='skipped'`, `skip_reason` explains why |
| Variant failed | Yes | `status='failed'`, `error` field populated |
| Run aborted | Yes | `run_plans.status='cancelled'` |
| Runner crash mid-variant | No | No partial metrics; variant row not written |

**Rule**: Never persist partial metrics. A `backtest_runs` row is only written when the variant completes (success, fail, or skip).

## Migration Strategy

### This PR
1. Create `run_plans` table
2. Add columns to `backtest_runs`
3. Update `RunOrchestrator` write path
4. Add verification endpoints
5. Keep admin UI on events (unchanged)

### Follow-up PR (Admin Migration)
1. Replace CTE reconstruction with table queries
2. Keep optional "Events" debug tab
3. Remove dual-write after validation period

## Guardrails

1. **Do not remove `trade_events` journaling** until admin UI is fully migrated and validated
2. **Include `plan_fingerprint`** in RUN_COMPLETED event for cross-verification
3. **Treat `run_plans.plan` as immutable** - never update after creation

## Appendix: Existing Schema Reference

### Current `backtest_runs` Columns (pre-migration)

```sql
id UUID PRIMARY KEY
workspace_id UUID
strategy_entity_id UUID
strategy_spec_id UUID
status TEXT
params JSONB
engine TEXT
dataset_meta JSONB
summary JSONB
equity_curve JSONB
trades JSONB
warnings JSONB
error TEXT
started_at TIMESTAMPTZ
completed_at TIMESTAMPTZ
created_at TIMESTAMPTZ
```

### Current `trade_events` Event Types

```sql
CHECK (event_type IN (
    'intent_emitted', 'policy_evaluated', 'intent_approved', 'intent_rejected',
    'order_filled', 'position_opened', 'position_scaled', 'position_closed',
    'run_started', 'run_completed'
))
```

Will add: `run_failed`, `run_cancelled`
