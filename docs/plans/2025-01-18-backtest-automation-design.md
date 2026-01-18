# Backtest Automation System Design

**Date**: 2025-01-18
**Status**: Approved
**Authors**: Human + Claude

## Overview

Job-based backtesting automation for crypto strategies with parameter sweeps and walk-forward optimization.

**Scope (v1)**:
- Asset class: Crypto (KuCoin primary, Binance v1.5+)
- Timeframes: 1m, 5m (execution), 15m, 1h (signal), 1d (regime)
- Features: Parameter sweeps, walk-forward optimization, artifact export

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Service                        │
│  Routers: /backtests/tune, /backtests/wfo, /admin/data/*   │
│  (accepts requests, enqueues jobs, returns status)          │
├─────────────────────────────────────────────────────────────┤
│                    Worker Process(es)                       │
│  Job runner loop: poll → claim → execute → complete         │
├──────────┬──────────┬──────────┬───────────────────────────┤
│ WFOJob   │ TuneJob  │DataSync  │ DataFetchJob              │
│ (orch:   │ (sweep   │(planner: │ (doer: CCXT fetch →       │
│  folds → │  via     │ enqueues │  upsert → data_revision)  │
│  tunes)  │ Engine)  │ fetches) │                           │
│          │    ↓     │          │                           │
│          │Backtest- │          │                           │
│          │Engine    │          │                           │
│          │(bt.py v1)│          │                           │
├──────────┴──────────┴──────────┴───────────────────────────┤
│              Postgres-backed Job Queue                      │
│    ┌─────────────────────────────────────────────┐         │
│    │ pg_cron → enqueue_core_data_sync() hourly/daily       │
│    └─────────────────────────────────────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ohlcv_candles │ jobs/job_events │ tunes/trials │core_symbols│
├────────────────┴─────────────────┴──────────────┴───────────┤
│  Artifacts: filesystem (data/artifacts/...)                 │
│             + artifact_index in Postgres (lineage, paths)   │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
         CCXT (KuCoin/Binance)
```

**Key separation**:
- FastAPI: thin request handler, enqueues jobs, serves status
- Worker: long-running job execution via BacktestEngine
- pg_cron: scheduled trigger for DataSyncJob (enqueue-only)
- DataSyncJob: planner (expands universe → enqueues DataFetchJobs)
- DataFetchJob: doer (CCXT → upsert → data_revision)

## Data Layer

### Market Data Strategy: Hybrid

- **Scheduled sync**: Keeps warm cache for core symbols (BTC-USDT, ETH-USDT + top alts)
- **Pull-through**: On-demand fetch for ad-hoc requests, then cached
- **Auto-promote** (future): Symbols with ≥3 requests in 7 days promoted to core

**History windows**:
- 1m, 5m: 180 days
- 15m, 1h: 2-3 years
- 1d: 5+ years

### OHLCV Storage

```sql
CREATE TABLE ohlcv_candles (
  exchange_id    TEXT,
  symbol         TEXT,           -- canonical: 'BTC-USDT'
  timeframe      TEXT CHECK (timeframe IN ('1m','5m','15m','1h','1d')),
  ts             TIMESTAMPTZ,    -- candle close, aligned to TF boundary, UTC
  open           DOUBLE PRECISION,
  high           DOUBLE PRECISION CHECK (high >= GREATEST(open, close, low)),
  low            DOUBLE PRECISION CHECK (low <= LEAST(open, close, high)),
  close          DOUBLE PRECISION,
  volume         DOUBLE PRECISION CHECK (volume >= 0),
  PRIMARY KEY (exchange_id, symbol, timeframe, ts)
);

CREATE INDEX idx_ohlcv_range ON ohlcv_candles
  (exchange_id, symbol, timeframe, ts DESC);
```

### Core Universe

```sql
CREATE TABLE core_symbols (
  exchange_id      TEXT,
  canonical_symbol TEXT,
  raw_symbol       TEXT,
  timeframes       TEXT[] DEFAULT ARRAY['1m','5m','15m','1h','1d'],
  is_enabled       BOOLEAN DEFAULT true,
  added_at         TIMESTAMPTZ DEFAULT now(),
  added_by         TEXT,
  UNIQUE (exchange_id, canonical_symbol)
);

-- Write-only log for future auto-promote
CREATE TABLE symbol_requests (
  exchange_id      TEXT,
  canonical_symbol TEXT,
  timeframe        TEXT,
  requested_at     TIMESTAMPTZ DEFAULT now()
);
```

**Starter list** (KuCoin):
```
BTC-USDT, ETH-USDT, SOL-USDT, BNB-USDT, XRP-USDT,
ADA-USDT, AVAX-USDT, DOGE-USDT, DOT-USDT, MATIC-USDT,
LINK-USDT, UNI-USDT, ATOM-USDT, LTC-USDT, ARB-USDT
```

### Data Revision (Drift Detection)

```sql
CREATE TABLE data_revisions (
  exchange_id  TEXT,
  symbol       TEXT,
  timeframe    TEXT,
  start_ts     TIMESTAMPTZ,
  end_ts       TIMESTAMPTZ,
  row_count    INT,
  checksum     TEXT,            -- deterministic sample hash
  computed_at  TIMESTAMPTZ DEFAULT now(),
  UNIQUE (exchange_id, symbol, timeframe, start_ts, end_ts)
);
```

**Checksum algorithm**: Sample first 10 + last 10 + every 1000th candle, canonical float formatting (10 decimals), UTC ISO timestamps, SHA256 truncated to 16 chars.

### Market Data Provider Interface

```python
class MarketDataProvider(Protocol):
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str,
        start_ts: datetime, end_ts: datetime
    ) -> list[Candle]

    def normalize_symbol(self, raw: str) -> str
    def exchange_symbol(self, canonical: str) -> str
    def canonical_timeframe(self, tf: str) -> str
```

**v1**: `CcxtMarketDataProvider(exchange_id)` for KuCoin/Binance.
**Later**: Native SDKs for execution (order placement, WebSocket streams).

## Job System

### Jobs Table

```sql
CREATE TABLE jobs (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  type            TEXT NOT NULL,    -- 'data_sync', 'data_fetch', 'tune', 'wfo'
  status          TEXT NOT NULL DEFAULT 'pending',
                  -- pending | running | succeeded | failed | canceled
  priority        INT DEFAULT 100,  -- lower = higher priority
  payload         JSONB NOT NULL,
  result          JSONB,            -- populated on completion

  attempt         INT DEFAULT 0,
  max_attempts    INT DEFAULT 3,
  run_after       TIMESTAMPTZ DEFAULT now(),

  locked_at       TIMESTAMPTZ,
  locked_by       TEXT,             -- worker_id

  parent_job_id   UUID REFERENCES jobs(id),
  workspace_id    UUID,
  dedupe_key      TEXT,             -- idempotency

  created_at      TIMESTAMPTZ DEFAULT now(),
  started_at      TIMESTAMPTZ,
  completed_at    TIMESTAMPTZ,

  UNIQUE (dedupe_key) WHERE dedupe_key IS NOT NULL
);

CREATE INDEX idx_jobs_claimable ON jobs (priority, created_at)
  WHERE status = 'pending';
CREATE INDEX idx_jobs_parent ON jobs (parent_job_id)
  WHERE parent_job_id IS NOT NULL;
```

### Job Events

```sql
CREATE TABLE job_events (
  id         BIGSERIAL PRIMARY KEY,
  job_id     UUID REFERENCES jobs(id) ON DELETE CASCADE,
  ts         TIMESTAMPTZ DEFAULT now(),
  level      TEXT,       -- 'info', 'warn', 'error'
  message    TEXT,
  meta       JSONB
);

CREATE INDEX idx_job_events_job ON job_events (job_id, ts);
```

### Worker Claim Pattern

```sql
WITH cte AS (
  SELECT id FROM jobs
  WHERE status = 'pending' AND run_after <= now()
  ORDER BY priority, created_at
  FOR UPDATE SKIP LOCKED
  LIMIT 1
)
UPDATE jobs j SET
  status = 'running',
  locked_at = now(),
  locked_by = $1,
  started_at = now(),
  attempt = j.attempt + 1
FROM cte
WHERE j.id = cte.id
RETURNING j.*;
```

**Polling**: 1 second interval.
**Retry**: On failure, if `attempt < max_attempts`: reset to `pending`, set `run_after = now() + backoff(attempt)`.
**Backoff**: `min(300, 2^attempt * 5) + jitter` seconds.

### Worker Heartbeat

```sql
CREATE TABLE worker_heartbeats (
  worker_id   TEXT PRIMARY KEY,
  version     TEXT,
  last_seen   TIMESTAMPTZ DEFAULT now()
);
```

Workers update every ~10s. `/ready` checks: DB ok + at least one heartbeat within 30s.

### Stale Job Reaper

Detect `status='running'` with `locked_at` older than N minutes → retry or fail. Implemented as periodic check in worker or separate reaper job.

### Job Type Behaviors

| Type | Enqueued By | Does |
|------|-------------|------|
| `data_sync` | pg_cron | Expands core universe → enqueues `data_fetch` jobs |
| `data_fetch` | data_sync / on-demand | CCXT fetch → upsert → data_revision |
| `tune` | API / WFO | `ensure_ohlcv_range()` → runs sweep → results + artifacts |
| `wfo` | API | Creates folds → enqueues child tunes → aggregates |

### pg_cron Schedules

```sql
-- Hourly: 1m + 5m for core symbols
SELECT cron.schedule('core-sync-hourly', '0 * * * *',
  $$SELECT enqueue_core_data_sync('core', '{"timeframes":["1m","5m"],"lookback_days":3}')$$);

-- Daily: 15m + 1h + 1d for core symbols
SELECT cron.schedule('core-sync-daily', '0 3 * * *',
  $$SELECT enqueue_core_data_sync('core', '{"timeframes":["15m","1h","1d"],"lookback_days":14}')$$);
```

## Walk-Forward Optimization

### WFO Config

```python
@dataclass
class WFOConfig:
    train_days: int
    test_days: int
    step_days: int
    min_folds: int
    start_ts: datetime | None
    end_ts: datetime | None
    leaderboard_top_k: int = 10
    allow_partial: bool = False
```

### Fold Generation (Half-Open Intervals)

```
[train_start, train_end) [test_start, test_end)

Data range: [start ────────────────────────────── end]

Fold 1:    [=====train=====)[test)
Fold 2:         [=====train=====)[test)
Fold 3:              [=====train=====)[test)
```

```python
def generate_folds(config: WFOConfig, available_range: tuple[datetime, datetime]) -> list[Fold]:
    effective_start = max(config.start_ts or available_range[0], available_range[0])
    effective_end = min(config.end_ts or available_range[1], available_range[1])

    folds = []
    cursor = effective_start
    while cursor + timedelta(days=config.train_days + config.test_days) <= effective_end:
        folds.append(Fold(
            train_start=cursor,
            train_end=cursor + timedelta(days=config.train_days),
            test_start=cursor + timedelta(days=config.train_days),
            test_end=cursor + timedelta(days=config.train_days + config.test_days),
        ))
        cursor += timedelta(days=config.step_days)

    if len(folds) < config.min_folds:
        raise InsufficientDataError(f"Only {len(folds)} folds, need {config.min_folds}")
    return folds
```

### WFO Job Flow

```
POST /backtests/wfo
        │
        ▼
   WFOJob created (status=pending)
        │
        ▼
   Worker claims WFOJob
        │
   ┌────┴────┐
   │ Generate folds (validate min_folds)
   │ Enqueue N TuneJobs (parent_job_id = wfo_id)
   │ Set WFOJob status = 'running'
   └────┬────┘
        │
        ▼
   Poll child tunes until all complete
   (respect cancellation, handle failures per allow_partial)
        │
        ▼
   Aggregate fold winners (top-K from each fold)
        │
        ▼
   Write WFO summary + artifacts
   Set WFOJob status = 'succeeded'
```

### Aggregation

```python
@dataclass
class WFOCandidateMetrics:
    params: dict
    mean_oos: float
    median_oos: float
    worst_fold_oos: float
    stddev_oos: float
    pct_top_k: float
    fold_count: int
    regime_tags: list[str]
```

**Coverage requirement**: Param must appear in ≥60% of folds to be eligible.

**OOS sign convention**: Higher is better. Metrics normalized so losses produce negative values.

**Selection rule**:
```python
coverage = fold_count / total_folds
score = (mean_oos - 0.5 * stddev_oos - 0.3 * max(0, threshold - worst_fold_oos)) * coverage
```

### Cancellation

1. Set WFO status → `canceled`
2. Set all `pending` child tunes → `canceled`
3. `running` children: cooperative check transitions to `canceled`

## Backtest Engine

### Interface

```python
class BacktestEngine(Protocol):
    def run_single(
        self, config: EngineConfig, candles: pd.DataFrame, params: dict
    ) -> SingleRunResult:
        """Returns metrics, equity_curve, trades for one param set."""

    def run_sweep(
        self, config: EngineConfig, candles: pd.DataFrame, param_grid: list[dict]
    ) -> list[SingleRunResult]:
        """Returns results for all param combinations."""
```

### v1: backtesting.py Adapter

- 24/7 continuous (no market hours)
- Bar-based execution: fills on next bar open
- Fee model: maker/taker percentages + slippage bps

### Upgrade Path

- **v1.5**: Swap `run_sweep` to vectorbt for speed
- **Later**: Custom event-driven engine for realistic execution

### ensure_ohlcv_range() Behavior

**v1 default**: TuneJob calls `ensure_ohlcv_range()` which directly fetches missing segments synchronously inside the job. Reduces orchestration complexity.

## Artifacts & Reproducibility

### Tune Artifacts

```
data/artifacts/tunes/{tune_id}/
├── tune.json
├── trials.csv
└── equity_best.csv
```

**tune.json**:
```json
{
  "identifiers": {
    "tune_id": "uuid",
    "workspace_id": "uuid",
    "service_git_sha": "abc123",
    "pine_git_sha": "def456",
    "config_schema_version": "tune_v1"
  },
  "engine": {
    "engine_id": "backtesting.py",
    "engine_version": "0.3.3",
    "adapter_version": "1.0.0"
  },
  "data_lineage": {
    "exchange_id": "kucoin",
    "raw_symbol": "BTC-USDT",
    "canonical_symbol": "BTC-USDT",
    "timeframe": "1h",
    "start_ts": "2024-01-01T00:00:00Z",
    "end_ts": "2024-07-01T00:00:00Z",
    "data_revision": {
      "row_count": 4320,
      "checksum": "a1b2c3d4e5f6",
      "computed_at": "2025-01-18T12:00:00Z"
    }
  },
  "config": {},
  "optimization": {},
  "outputs": {}
}
```

**trials.csv** (hybrid: JSON string + common scalars):
```csv
trial_idx,params_json,lookback_days,threshold,is_score,oos_score,...
```

**equity_best.csv**:
```csv
ts,equity,drawdown_pct,position_exposure
```

### WFO Artifacts

```
data/artifacts/wfo/{wfo_id}/
├── wfo.json
├── candidates.csv
└── folds/
    ├── 0/
    │   ├── tune.json
    │   ├── trials.csv
    │   └── equity_best.csv
    ├── 1/
    ...
```

### Artifact Index

```sql
CREATE TABLE artifact_index (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id  UUID NOT NULL,
  job_id        UUID REFERENCES jobs(id),
  run_id        UUID NOT NULL,
  job_type      TEXT NOT NULL,
  artifact_kind TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  data_revision JSONB,
  is_pinned     BOOLEAN DEFAULT false,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_artifact_workspace ON artifact_index (workspace_id, job_type, created_at DESC);
CREATE INDEX idx_artifact_run ON artifact_index (run_id);
```

### Retention

- Default: 90 days
- `is_pinned = true`: retained indefinitely
- Pinning via: `PATCH /admin/runs/{run_id}/pin`

## API Surface

**Naming**: `run_id == job_id` throughout.

**Idempotency**: `X-Idempotency-Key` dedupes against `(workspace_id, endpoint, key)`.

**Scoping**: `/backtests/*` enforces workspace scoping. `/admin/*` is privileged cross-workspace.

### Backtest Endpoints

```
POST /backtests/tune
  Body: {workspace_id, strategy_id, param_space, objective, gates,
         data_config: {exchange, symbol, timeframe, start_ts, end_ts},
         oos_ratio?, seed?}
  Headers: X-Idempotency-Key (optional)
  → 202 {run_id, status: "pending"}

POST /backtests/wfo
  Body: {workspace_id, strategy_id, param_space, objective, gates,
         data_config, wfo_config: {train_days, test_days, step_days, min_folds}}
  Headers: X-Idempotency-Key (optional)
  → 202 {run_id, status: "pending"}

GET /backtests/runs
  Query: workspace_id, type?, status?, limit?, offset?
  → {items: [{run_id, type, status, created_at, completed_at, summary}], total}

GET /backtests/runs/{run_id}
  → Full run detail (tune or wfo)

POST /backtests/runs/{run_id}/cancel
  → 200 {status: "canceled", children_canceled: N}

GET /backtests/leaderboard
  Query: workspace_id, objective?, limit?
  → {items: [{run_id, params, score, metrics, regime_tag}]}
  Accept: text/csv → CSV export
```

### Data Management (admin)

```
POST /admin/data/core-symbols
  Body: {action: "add"|"remove"|"enable"|"disable", exchange_id, symbols: [...]}

GET /admin/data/core-symbols
  Query: exchange_id?

POST /admin/data/sync/trigger
  Body: scope="core" → DataSyncJob, scope="symbol" → DataFetchJob

GET /admin/data/status
  Query: exchange_id, symbol, timeframe
  → {available_range, row_count, last_updated, gaps: [...], gaps_total}
```

### Artifacts (admin)

```
GET /admin/artifacts
  Query: workspace_id, job_type?, run_id?, limit?, offset?

PATCH /admin/runs/{run_id}/pin
  Body: {pinned: true|false}

GET /admin/artifacts/{id}/download
  → Single file download
```

### Jobs (admin)

```
GET /admin/jobs
  Query: type?, status?, workspace_id?, limit?

GET /admin/jobs/{run_id}
  → Full job + recent events

GET /admin/jobs/{run_id}/events
  Query: level?, limit?
```

### Health

```
GET /health    → 200 (liveness)
GET /ready     → 200/503 (DB + worker heartbeat)
GET /metrics   → Prometheus
```

## Implementation Phases

### Phase 1: Data Layer

```
Deliverables:
├── ohlcv_candles table + indexes + CHECK constraints
├── core_symbols table + admin endpoints
├── symbol_requests table (write-only)
├── data_revisions table
├── CcxtMarketDataProvider (KuCoin)
├── DataFetchJob implementation
└── POST /admin/data/sync/trigger (scope=symbol)

Done when: Can fetch BTC-USDT 1h from KuCoin, store in Postgres,
           query via SQL, compute data_revision checksum.
```

### Phase 2: Job System

```
Deliverables:
├── jobs + job_events tables
├── Worker process with claim loop
├── worker_heartbeats + /ready check
├── Retry/backoff logic
├── Stale job reaper (detect stuck running jobs)
├── DataSyncJob (expands core → enqueues fetches)
├── pg_cron schedule for hourly/daily sync
└── GET /admin/jobs/* endpoints

Done when: pg_cron triggers DataSyncJob, worker claims it,
           fetches missing ranges for core symbols, job_events logged.
```

### Phase 3: Tune Integration

```
Deliverables:
├── TuneJob wired to existing ParamTuner
├── BacktestEngine interface + backtesting.py adapter
├── ensure_ohlcv_range() (synchronous fetch inside job)
├── Artifact generation (tune.json, trials.csv, equity_best.csv)
├── artifact_index table
├── POST /backtests/tune endpoint
├── GET /backtests/runs/{run_id}
└── API idempotency keys (X-Idempotency-Key)

Done when: POST /backtests/tune with KuCoin data runs sweep,
           persists results, generates artifacts, queryable via API.
```

### Phase 4: WFO

```
Deliverables:
├── WFOJob implementation (fold generation, child orchestration)
├── Aggregation logic (candidates, coverage, selection)
├── WFO artifacts (wfo.json, candidates.csv, folds/)
├── POST /backtests/wfo endpoint
├── Cancellation (parent + children)
└── Leaderboard updates for WFO runs

Done when: POST /backtests/wfo runs 5-fold WFO, aggregates results,
           selects robust params, artifacts complete.
```

### Phase 5: Polish & Ops

```
Deliverables:
├── Pin/retention endpoints
├── Gap detection in /admin/data/status
├── Prometheus metrics (jobs, data freshness, tune duration)
├── Admin dashboard integration
├── Binance exchange support (config change)
└── Documentation

Done when: Can run WFO on Binance data, monitor via metrics,
           pin valuable runs, prune old artifacts.
```

### Dependency Graph

```
Phase 1 (Data) ──┬──► Phase 2 (Jobs)
                 │         │
                 │         ▼
                 └──► Phase 3 (Tune) ──► Phase 4 (WFO) ──► Phase 5 (Polish)
```

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Market data strategy | Hybrid (scheduled + pull-through) | Core symbols warm, ad-hoc without universe admin |
| OHLCV storage | Postgres per-candle rows | Simple, queryable, ops-boring, scale later |
| WFO mode | Fixed rolling windows | Cleanest mental model, comparable folds |
| WFO architecture | Separate job wrapping tunes | Modular, can run single tune or WFO |
| Exchange integration | CCXT for data, native for execution | Unified data now, precision execution later |
| Job queue | Postgres-backed | Restart-safe, observable, no new infra |
| Scheduler | pg_cron | DB is control plane, runs even if app down |
| Core universe | DB table, manual | Auditable, API-managed, auto-promote later |
| Backtest engine | backtesting.py behind interface | Already integrated, swap to vectorbt later |
| Artifact storage | Filesystem + Postgres index | Files for reproducibility, index for queries |
