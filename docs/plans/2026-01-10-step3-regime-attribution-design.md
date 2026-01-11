# Step 3: Regime Attribution for Tune Results

## Overview

Connect regime tagging (Step 2) to tune performance outcomes. Enable queries like "what worked in similar regimes?" and compute regime-aware parameter uplift.

## Goals

1. **Attach regime fingerprint** to each tune session
2. **Query by regime**: Find best params for trend+vol combination
3. **Attribution**: Compare regime-selected params vs baseline

## Schema Changes

### Table: `backtest_tunes`

Add columns for regime identification and denormalized filtering:

```sql
-- Migration: 032_tune_regime_attribution.sql

-- Regime versioning (for fingerprint stability)
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS regime_schema_version TEXT,
  ADD COLUMN IF NOT EXISTS tag_ruleset_id TEXT;

-- Human-readable regime key: "regime_v1_1|default_v1|uptrend|high_vol|noisy"
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS regime_key TEXT;

-- SHA256 hash for indexing (derived from regime_key)
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS regime_fingerprint TEXT;

-- Denormalized tags for fast SQL filtering
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS trend_tag TEXT;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS vol_tag TEXT;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS efficiency_tag TEXT;

-- Explicit OOS best (existing best_score may be IS or ambiguous)
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS best_oos_score DOUBLE PRECISION;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS best_oos_params JSONB;
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS best_oos_run_id UUID REFERENCES backtest_runs(id);

-- Optional: baseline for uplift calculation
ALTER TABLE backtest_tunes
  ADD COLUMN IF NOT EXISTS baseline_oos_score DOUBLE PRECISION;
```

### Indexes

```sql
-- Fast regime-based lookups
CREATE INDEX IF NOT EXISTS idx_tunes_regime_key
  ON backtest_tunes(regime_key)
  WHERE regime_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tunes_trend_vol
  ON backtest_tunes(trend_tag, vol_tag)
  WHERE trend_tag IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tunes_best_oos_score
  ON backtest_tunes(best_oos_score DESC NULLS LAST)
  WHERE best_oos_score IS NOT NULL;

-- Multi-tenant regime lookup
CREATE INDEX IF NOT EXISTS idx_tunes_workspace_regime
  ON backtest_tunes(workspace_id, regime_key)
  WHERE regime_key IS NOT NULL;
```

## Regime Key Format

### Structure

```
{schema}|{ruleset}|{trend}|{vol}|{efficiency}
```

### Examples

```
regime_v1_1|default_v1|uptrend|high_vol|noisy
regime_v1_1|default_v1|flat|low_vol|efficient
regime_v1_1|default_v1|downtrend|_|_          # underscore for no tag
```

### Fingerprint Computation

```python
def compute_regime_key(
    snapshot: RegimeSnapshot,
    ruleset_id: str = "default_v1",
) -> str:
    """
    Compute human-readable regime key from snapshot.

    Format: {schema}|{ruleset}|{trend}|{vol}|{efficiency}
    Uses "_" for missing tags in each dimension.
    """
    tags = set(snapshot.regime_tags)

    # Extract one tag per dimension (or "_" if none)
    trend = next((t for t in ["uptrend", "downtrend", "trending", "flat"] if t in tags), "_")
    vol = next((t for t in ["high_vol", "low_vol"] if t in tags), "_")
    eff = next((t for t in ["efficient", "noisy"] if t in tags), "_")

    return f"{snapshot.schema_version}|{ruleset_id}|{trend}|{vol}|{eff}"


def compute_regime_fingerprint(regime_key: str) -> str:
    """SHA256 hash of regime key for indexing."""
    return hashlib.sha256(regime_key.encode()).hexdigest()
```

## Queries Unlocked

### Find best params for same regime

```sql
SELECT strategy_entity_id, best_oos_score, best_oos_params
FROM backtest_tunes
WHERE workspace_id = $1
  AND regime_key = $2
  AND best_oos_score IS NOT NULL
ORDER BY best_oos_score DESC
LIMIT 10;
```

### Find best params by trend+vol (ignoring efficiency)

```sql
SELECT strategy_entity_id, best_oos_score, best_oos_params, regime_key
FROM backtest_tunes
WHERE workspace_id = $1
  AND trend_tag = $2
  AND vol_tag = $3
  AND best_oos_score IS NOT NULL
ORDER BY best_oos_score DESC
LIMIT 10;
```

### Regime performance summary

```sql
SELECT
  regime_key,
  COUNT(*) as tune_count,
  AVG(best_oos_score) as avg_score,
  MAX(best_oos_score) as max_score
FROM backtest_tunes
WHERE workspace_id = $1
  AND regime_key IS NOT NULL
  AND best_oos_score IS NOT NULL
GROUP BY regime_key
ORDER BY avg_score DESC;
```

## Population Logic

On tune completion:

```python
async def populate_regime_attribution(tune_id: UUID, pool) -> None:
    """
    Populate regime columns after tune completes.

    Called from tune completion handler.
    """
    # 1. Get the OOS regime snapshot from the best run
    best_run = await get_best_oos_run(tune_id, pool)
    if not best_run or not best_run.regime_oos:
        return  # No OOS data, skip attribution

    snapshot = best_run.regime_oos

    # 2. Compute regime key and fingerprint
    regime_key = compute_regime_key(snapshot, ruleset_id="default_v1")
    fingerprint = compute_regime_fingerprint(regime_key)

    # 3. Extract denormalized tags
    tags = set(snapshot.regime_tags)
    trend_tag = next((t for t in ["uptrend", "downtrend", "trending", "flat"] if t in tags), None)
    vol_tag = next((t for t in ["high_vol", "low_vol"] if t in tags), None)
    eff_tag = next((t for t in ["efficient", "noisy"] if t in tags), None)

    # 4. Get best OOS score/params from leaderboard
    best_oos_score = best_run.metrics_oos.get("sharpe") if best_run.metrics_oos else None
    best_oos_params = best_run.params

    # 5. Update tune record
    await pool.execute("""
        UPDATE backtest_tunes SET
            regime_schema_version = $2,
            tag_ruleset_id = $3,
            regime_key = $4,
            regime_fingerprint = $5,
            trend_tag = $6,
            vol_tag = $7,
            efficiency_tag = $8,
            best_oos_score = $9,
            best_oos_params = $10,
            best_oos_run_id = $11
        WHERE id = $1
    """, tune_id, snapshot.schema_version, "default_v1", regime_key, fingerprint,
        trend_tag, vol_tag, eff_tag, best_oos_score, json.dumps(best_oos_params), best_run.id)
```

## File Changes

| File | Changes |
|------|---------|
| `migrations/032_tune_regime_attribution.sql` | Add columns and indexes |
| `app/services/kb/regime.py` | Add `compute_regime_key()`, `compute_regime_fingerprint()` |
| `app/services/backtest/tuner.py` | Call `populate_regime_attribution()` on completion |
| `tests/unit/kb/test_regime.py` | Add tests for regime key computation |

## Test Matrix (8-12 tests)

1. **regime_key format**: Correct structure with all tags
2. **regime_key with missing tags**: Uses "_" for dimensions without tags
3. **fingerprint stability**: Same key → same hash
4. **fingerprint changes**: Different schema/ruleset → different hash
5. **population on completion**: Columns populated after tune completes
6. **null-safe**: No crash when regime_oos is None
7. **denormalized tags match**: trend_tag/vol_tag match what's in regime_key
8. **query by regime_key**: Index used, correct results
9. **query by trend+vol**: Index used, correct results
10. **best_oos_score populated**: From leaderboard[0] OOS metrics

## Definition of Done

1. Migration applies cleanly
2. Tune completion populates all regime columns
3. Queries by regime_key and trend+vol use indexes
4. 10+ tests passing
5. Existing tunes unaffected (columns nullable)

## Future Enhancements (v2)

- **Backfill existing tunes**: Migration to populate regime columns for historical data
- **Attribution dashboard**: Show uplift of regime-selected vs default params
- **Similar regime search**: Fuzzy matching when exact regime has low sample count
- **Regime drift detection**: Alert when live regime diverges from training regime
