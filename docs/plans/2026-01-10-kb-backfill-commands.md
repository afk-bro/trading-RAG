# KB Backfill Commands Implementation Plan

**Date**: 2026-01-10
**Author**: Claude (via brainstorming skill)
**Status**: Ready for implementation

## Overview

Two CLI commands for backfilling KB (Knowledge Base) status on historical backtest runs:

1. **backfill-candidacy**: Evaluate and mark eligible test_variant runs as `candidate`
2. **backfill-regime**: Compute missing regime snapshots from OHLCV data

## Command 1: backfill-candidacy

### Purpose
Sweep historical `backtest_runs` and mark eligible test_variant runs as `candidate` status based on the candidacy gate policy.

### CLI Interface
```bash
python -m app.services.kb.cli backfill-candidacy \
  --workspace-id <uuid> \
  --since 2025-01-01 \
  --limit 5000 \
  --dry-run
```

### Arguments
| Arg | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `--workspace-id` | UUID | Yes | - | Target workspace |
| `--since` | Date | No | None | Only process runs after this date |
| `--limit` | int | No | 1000 | Max runs to process |
| `--dry-run` | flag | No | False | Preview without writing |
| `--experiment-type` | str | No | "sweep" | Experiment type for gate check |

### Query
```sql
SELECT id, workspace_id, summary, regime_oos, trade_count, run_kind
FROM backtest_runs
WHERE run_kind = 'test_variant'
  AND status IN ('completed', 'success')
  AND kb_status = 'excluded'
  AND workspace_id = $1
  AND ($2 IS NULL OR created_at >= $2)
ORDER BY created_at
LIMIT $3
```

### Metrics Extraction
From `summary` JSONB:
- `n_trades_oos`: `summary->>'trades'` or `trade_count` column
- `max_dd_frac_oos`: `abs(summary->>'max_drawdown_pct') / 100`
- `sharpe_oos`: `summary->>'sharpe'`
- `overfit_gap`: None (IS metrics not stored for test_variants)

### Gate Evaluation
Use existing `is_candidate()` function from `app.services.kb.candidacy`:
```python
from app.services.kb.candidacy import is_candidate, VariantMetricsForCandidacy, CandidacyConfig

decision = is_candidate(
    metrics=VariantMetricsForCandidacy(
        n_trades_oos=row["trade_count"] or 0,
        max_dd_frac_oos=abs(row["summary"].get("max_drawdown_pct", 0)) / 100,
        sharpe_oos=row["summary"].get("sharpe"),
        overfit_gap=None,  # Not available for test_variants
    ),
    regime_oos=parse_regime_snapshot(row["regime_oos"]) if row["regime_oos"] else None,
    experiment_type="sweep",
    config=CandidacyConfig(require_regime=False),  # Skip regime check if not present
)
```

### Updates (non-dry-run)
```sql
UPDATE backtest_runs SET
  kb_status = 'candidate',
  kb_status_changed_at = NOW(),
  kb_status_changed_by = 'backfill',
  auto_candidate_gate = 'passed_all_gates'
WHERE id = $1
```

### Output Format
```
==================================================
BACKFILL CANDIDACY REPORT
==================================================
Workspace:      abc123
Since:          2025-01-01
Limit:          5000
Dry Run:        True
--------------------------------------------------
Total Scanned:  4821
Eligible:       312
Ineligible:     4509
Updated:        0 (dry run)
--------------------------------------------------
Rejection Reasons:
  insufficient_oos_trades:  2341
  dd_too_high:              1024
  sharpe_too_low:           891
  missing_regime_oos:       253
==================================================
```

---

## Command 2: backfill-regime

### Purpose
Compute missing `regime_is`/`regime_oos`/`regime_schema_version` for legacy runs.

### Challenge
OHLCV data is not stored with runs. Must be provided externally.

### CLI Interface
```bash
python -m app.services.kb.cli backfill-regime \
  --workspace-id <uuid> \
  --ohlcv-file /path/to/data.csv \
  --symbol AAPL \
  --timeframe 1h \
  --since 2025-01-01 \
  --limit 2000 \
  --dry-run
```

### Arguments
| Arg | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `--workspace-id` | UUID | Yes | - | Target workspace |
| `--ohlcv-file` | Path | Yes | - | CSV/JSON with OHLCV data |
| `--symbol` | str | No | None | Filter runs by symbol |
| `--timeframe` | str | No | "1d" | Timeframe for regime computation |
| `--since` | Date | No | None | Only process runs after this date |
| `--limit` | int | No | 500 | Max runs to process |
| `--dry-run` | flag | No | False | Preview without writing |

### Query (runs needing regime backfill)
```sql
SELECT id, workspace_id, dataset_meta, params, started_at, completed_at
FROM backtest_runs
WHERE run_kind = 'test_variant'
  AND status IN ('completed', 'success')
  AND regime_oos IS NULL
  AND workspace_id = $1
  AND ($2 IS NULL OR (dataset_meta->>'symbol' = $2))
  AND ($3 IS NULL OR created_at >= $3)
ORDER BY created_at
LIMIT $4
```

### Regime Computation
For each run:
1. Extract OOS date range from `dataset_meta` or `params`
2. Slice OHLCV data to OOS window
3. Call `compute_regime_snapshot(df, source="backfill", timeframe=timeframe)`

### Updates (non-dry-run)
```sql
UPDATE backtest_runs SET
  regime_oos = $2,
  regime_schema_version = 'regime_v1'
WHERE id = $1
```

### Output Format
```
==================================================
BACKFILL REGIME REPORT
==================================================
Workspace:      abc123
OHLCV File:     /path/to/AAPL_1h.csv
Symbol:         AAPL
Timeframe:      1h
OHLCV Rows:     10000
Date Range:     2023-01-01 to 2025-12-31
--------------------------------------------------
Total Scanned:  892
Computed:       743
Skipped:        149
  - insufficient_ohlcv_coverage:  89
  - missing_date_range:           60
Updated:        0 (dry run)
==================================================
```

---

## Implementation Tasks

### Task 1: Add backfill-candidacy command
**File**: `app/services/kb/cli.py`

1. Add argument parser for `backfill-candidacy` subcommand
2. Implement `cmd_backfill_candidacy(args)` async function
3. Query eligible runs with filters
4. Batch process with progress logging
5. Collect rejection reasons for summary
6. Update in batches (100 at a time)

### Task 2: Add backfill-regime command
**File**: `app/services/kb/cli.py`

1. Add argument parser for `backfill-regime` subcommand
2. Implement `cmd_backfill_regime(args)` async function
3. Load and parse OHLCV file
4. Query runs needing regime
5. Match run date ranges to OHLCV window
6. Compute regime snapshots
7. Update in batches

### Task 3: Unit tests
**File**: `tests/unit/services/kb/test_cli_backfill.py`

1. Test candidacy extraction from summary JSONB
2. Test gate evaluation with various metric combinations
3. Test dry-run produces correct counts
4. Test regime computation with OHLCV data
5. Test OHLCV date range matching

---

## Acceptance Criteria

1. ✅ `--dry-run` mode produces accurate counts + top rejection reasons
2. ✅ Non-dry-run updates rows in batches (chunked commits)
3. ✅ `pytest tests/` still green
4. ✅ After running `backfill-candidacy`, KB trial count increases
5. ✅ After running `backfill-regime`, runs have `regime_oos` populated
6. ✅ `recommend()` returns more candidates with backfilled data

---

## Design Decisions

1. **No ingestion in candidacy command**: Only updates `kb_status` to `candidate`. Existing `/kb/trials/ingest` picks them up.

2. **Batch updates**: Process 100 rows per commit to avoid long transactions.

3. **Regime file requirement**: OHLCV must be provided externally since it's not stored. This keeps storage costs down.

4. **CandidacyConfig.require_regime=False**: For backfill, we relax the regime requirement since legacy runs may not have it. The gate will still check other metrics.

5. **experiment_type default**: Default to "sweep" since most test_variants come from parameter sweeps.
