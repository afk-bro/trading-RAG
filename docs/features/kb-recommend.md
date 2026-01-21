# Trading KB Recommend

Strategy parameter recommendations based on historical backtest results.

## Pipeline (`/kb/trials/recommend`)

**Response Status**:
- `ok` - High confidence recommendations
- `degraded` - Recommendations with caveats (relaxed filters, low count)
- `none` - No suitable recommendations

**Key Features**:
- Strategy-specific quality floors (sharpe ≥0.3, return ≥5%, calmar ≥0.5)
- Single-axis relaxation suggestions when `status=none`
- Confidence scoring based on candidate count and score variance
- Regime-aware filtering (volatility, trend, momentum tags)
- Debug mode with full candidate inspection

**Request**:
```python
POST /kb/trials/recommend
{
    "workspace_id": "uuid",
    "strategy_name": "bb_reversal",
    "objective_type": "sharpe",
    "require_oos": true,
    "max_drawdown": 0.20,
    "min_trades": 5
}
```

## Regime Fingerprints

Materialized regime fingerprints for instant similarity queries (Migration 056).

**Problem**: Regime similarity queries recomputed vectors on every request.

**Solution**: Precompute and store regime hashes + vectors at tune time.

**Schema** (`regime_fingerprints` table):
- `fingerprint` (BYTEA) - 32-byte SHA256 for exact matching
- `regime_vector` (FLOAT8[]) - 6-dim: [atr_norm, rsi, bb_width, efficiency, trend_strength, zscore]
- `trend_tag`, `vol_tag`, `efficiency_tag` - Denormalized tags
- `regime_schema_version` - Schema version (default: `regime_v1_1`)

**SQL Functions**:
- `compute_regime_fingerprint(FLOAT8[])` - SHA256 from vector
- `regime_distance(FLOAT8[], FLOAT8[])` - Euclidean distance

**Indexes**:
- Hash on `fingerprint` for O(1) exact matching
- B-tree on `tune_id`
- Composite on tags for SQL filtering
- GIN on `regime_vector`

**Usage**:
```sql
-- Exact match
SELECT * FROM regime_fingerprints
WHERE fingerprint = compute_regime_fingerprint(ARRAY[0.014, 45.2, 0.023, 0.65, 0.78, -0.52]);

-- Similar regimes
SELECT *, regime_distance(regime_vector, ARRAY[...]) as dist
FROM regime_fingerprints
ORDER BY dist LIMIT 10;
```
