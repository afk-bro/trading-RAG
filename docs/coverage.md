# Coverage & Knowledge Base

Coverage gap triage workflow and KB recommendation pipeline.

## Coverage Triage Workflow

Admin endpoints for managing coverage gaps in the cockpit UI.

**Source**: `app/admin/coverage.py`, `app/services/coverage_gap/repository.py`

### Architecture

```
Match Run (weak_coverage=true)
       │
       ▼
┌─────────────────┐
│ Coverage Status │ ──► open → acknowledged → resolved
└─────────────────┘
       │
       ▼
   Priority Score (deterministic ranking)
```

### Status Lifecycle

| Status | Description |
|--------|-------------|
| `open` | New coverage gap, needs attention (default) |
| `acknowledged` | Someone is investigating |
| `resolved` | Gap addressed (strategy added, false positive, etc.) |

### Priority Score Formula

Higher = more urgent.

| Component | Value | Condition |
|-----------|-------|-----------|
| Base | `0.5 - best_score` | Clamped to [0, 0.5] |
| No results | +0.2 | `num_above_threshold == 0` |
| NO_MATCHES | +0.15 | Reason code present |
| NO_STRONG_MATCHES | +0.1 | Reason code present |
| Recency | +0.05 | Created in last 24h |

### Key Endpoints

**List weak coverage runs:**
```
GET /admin/coverage/weak?workspace_id=...&status=open
```

Parameters:
- `status`: `open` (default), `acknowledged`, `resolved`, `all`
- `include_candidate_cards=true` (default) - Hydrate strategy cards
- Results sorted by `priority_score` descending

**Update status:**
```
PATCH /admin/coverage/weak/{run_id}
Body: {"status": "acknowledged|resolved", "note": "optional resolution note"}
```

Tracks `acknowledged_at/by`, `resolved_at/by`, `resolution_note`.

### Resolution Guard

Cannot mark as `resolved` without at least one of:
- `candidate_strategy_ids` present (strategies were recommended)
- `resolution_note` provided (explains why resolved)

Returns 400 if guard fails.

### Auto-Resolve on Success

When `/youtube/match-pine` produces `weak_coverage=false`:
1. Find all `open`/`acknowledged` runs with same `intent_signature`
2. Auto-resolve them with `resolved_by='system'`
3. Set `resolution_note='Auto-resolved by successful match'`

### LLM-Powered Strategy Explanation

```
POST /admin/coverage/explain
Request: {run_id, strategy_id} + workspace_id query param
Response: {explanation, model, provider, latency_ms}
```

Builds prompts from:
- Intent: archetypes, indicators, timeframes, symbols, risk terms
- Strategy: name, description, tags, backtest summary
- Overlap: matched tags and similarity score

Returns 503 if LLM not configured, 404 if run/strategy not found.

### Cockpit UI

Route: `/admin/coverage/cockpit`

Features:
- Two-panel layout: queue (left) + detail (right)
- Status tabs: Open, Acknowledged, Resolved, All
- Priority badges: P1 (>=0.75), P2 (>=0.40), P3 (<0.40)
- Strategy cards with tags, backtest status, OOS score
- "Explain Match" button generates LLM explanation per candidate
- Deep link support: `/admin/coverage/cockpit/{run_id}`
- Triage controls: Acknowledge, Resolve, Reopen with optional notes

---

## Trading KB Recommend Pipeline

**Endpoint**: `POST /kb/trials/recommend`

Provides strategy parameter recommendations based on historical backtest results.

### Response Status

| Status | Description |
|--------|-------------|
| `ok` | High confidence recommendations available |
| `degraded` | Recommendations with caveats (relaxed filters, low count) |
| `none` | No suitable recommendations found |

### Key Features

- Strategy-specific quality floors (sharpe ≥0.3, return ≥5%, calmar ≥0.5)
- Single-axis relaxation suggestions when `status=none`
- Confidence scoring based on candidate count and score variance
- Regime-aware filtering (volatility, trend, momentum tags)
- Debug mode with full candidate inspection

### Request Example

```json
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

### Related Endpoints

```
POST /kb/trials/recommend?mode=debug - Debug mode with full candidates
POST /kb/trials/ingest               - Ingest trials from tune runs (admin-only)
POST /kb/trials/upload-ohlcv         - Upload OHLCV data for regime analysis
```

---

## Regime Fingerprints

Materialized regime fingerprints for instant similarity queries (Migration 056).

**Problem**: Regime similarity queries were recomputing vectors on every request.

**Solution**: Precompute and store regime hashes + vectors at tune time for O(1) lookup.

### Schema (`regime_fingerprints` table)

| Column | Type | Description |
|--------|------|-------------|
| `fingerprint` | BYTEA | 32-byte SHA256 hash for exact matching |
| `regime_vector` | FLOAT8[] | 6-dim: [atr_norm, rsi, bb_width, efficiency, trend_strength, zscore] |
| `trend_tag`, `vol_tag`, `efficiency_tag` | TEXT | Denormalized tags for SQL filtering |
| `regime_schema_version` | TEXT | Schema version (default: `regime_v1_1`) |

### SQL Functions

- `compute_regime_fingerprint(FLOAT8[])` - Compute SHA256 from vector (rounds to 4 decimals)
- `regime_distance(FLOAT8[], FLOAT8[])` - Euclidean distance between vectors

### Indexes

- Hash index on `fingerprint` for O(1) exact matching
- B-tree on `tune_id` for tune-based lookups
- Composite index on tags for SQL filtering
- GIN on `regime_vector` for array operators

### Usage

```sql
-- Find all trials with exact same regime
SELECT * FROM regime_fingerprints
WHERE fingerprint = compute_regime_fingerprint(ARRAY[0.014, 45.2, 0.023, 0.65, 0.78, -0.52]);

-- Find similar regimes by distance
SELECT *, regime_distance(regime_vector, ARRAY[...]) as dist
FROM regime_fingerprints
ORDER BY dist LIMIT 10;
```
