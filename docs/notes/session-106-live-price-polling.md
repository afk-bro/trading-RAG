# Session 106: Live Price Polling Implementation

Date: 2026-01-21

## Summary

Completed full Live Price Polling feature across 3 PRs - a background service that periodically fetches OHLCV candles from exchanges with DB-backed scheduling and exponential backoff.

## What Was Accomplished

### Live Price Polling (LP1 + LP2 + LP3)

Implemented `LivePricePoller` service following the PineRepoPoller pattern.

| PR | Commit | Description |
|----|--------|-------------|
| LP1 | `6b260c0` | Poller skeleton, lifecycle, Prometheus metrics |
| LP2 | `cb9f686` | CCXT fetch, smart lookback, staleness metrics |
| LP3 | `e5a7c49` | DB-backed scheduling, exponential backoff |

### LP1: Poller Skeleton (Previously Completed)

- Created `app/services/market_data/poller.py` with lifecycle management
- Integrated with FastAPI lifespan in `app/core/lifespan.py`
- Added configuration settings in `app/config.py`
- Prometheus metrics: `POLL_ENABLED`, `POLL_INFLIGHT`, `POLL_RUNS_TOTAL`
- 19 unit tests covering lifecycle, selection, semaphore usage

### LP2: Fetch + Upsert

- Implemented `_fetch_pair()` with smart lookback logic
- CCXT provider integration via `_get_provider()` with caching
- Per-exchange semaphores for rate limiting
- `LATEST_CANDLE_AGE` staleness gauge per (exchange, symbol, timeframe)
- Candle upsert via OHLCV repository

**Smart Lookback Logic**:
- If `last_candle_ts` exists: fetch from `last_candle_ts - 1 candle buffer`
- Otherwise: fetch `lookback_candles × timeframe_seconds` worth

### LP3: DB-Backed Scheduling

**Migration** (`migrations/082_price_poll_state.sql`):
```sql
CREATE TABLE price_poll_state (
    exchange_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    next_poll_at TIMESTAMPTZ,
    failure_count INT NOT NULL DEFAULT 0,
    last_success_at TIMESTAMPTZ,
    last_candle_ts TIMESTAMPTZ,
    last_error TEXT,
    PRIMARY KEY (exchange_id, symbol, timeframe)
);
```

**Repository** (`app/repositories/price_poll_state.py`):
- `get_state()` / `list_due_pairs()` - query scheduling state
- `upsert_state_if_missing()` - seed new pairs (next_poll_at=NULL = due immediately)
- `mark_success()` - reset failure count, schedule at interval + jitter
- `mark_failure()` - exponential backoff: `min(interval * 2^(n-1), max)`
- Health helpers: `count_due_pairs()`, `count_never_polled()`, `get_worst_staleness()`

**DB-First Selection** in poller:
1. Query state table for due pairs (next_poll_at <= NOW or NULL)
2. Validate against enabled symbols
3. Filter to valid timeframes
4. Top up with newly-enabled symbols missing state rows

## Files Created/Modified

| File | Action | PR |
|------|--------|-----|
| `app/config.py` | Modified | LP1 |
| `app/services/market_data/poller.py` | Created | LP1-3 |
| `app/services/market_data/__init__.py` | Modified | LP1 |
| `app/core/lifespan.py` | Modified | LP1 |
| `app/repositories/price_poll_state.py` | Created | LP3 |
| `migrations/082_price_poll_state.sql` | Created | LP3 |
| `tests/unit/services/market_data/test_poller.py` | Created | LP1-2 |
| `tests/unit/repositories/test_price_poll_state.py` | Created | LP3 |

## Test Results

- **40 LP-specific tests** (25 poller + 15 repository)
- **3017 total unit tests passing**
- 3 pre-existing failures in `tests/unit/alerts/test_job.py` (unrelated)

## Key Technical Decisions

1. **PollKey = (exchange_id, symbol, timeframe)** - Core unit for all operations
2. **DB-first approach** - Query state table first, keeps tick cost proportional to work due
3. **Exponential backoff with cap** - 1×, 2×, 4×... up to `backoff_max_seconds` (900s default)
4. **Jitter on success** - Prevents thundering herd on interval boundaries
5. **Per-exchange semaphores** - Rate limiting respects exchange API limits

## Configuration Added

```python
live_price_poll_enabled: bool = False
live_price_poll_interval_seconds: int = 60
live_price_poll_tick_seconds: int = 15
live_price_poll_jitter_seconds: int = 5
live_price_poll_backoff_max_seconds: int = 900
live_price_poll_max_concurrency_per_exchange: int = 3
live_price_poll_max_pairs_per_tick: int = 50
live_price_poll_lookback_candles: int = 5
live_price_poll_timeframes: list[str] = ["1m"]
```

## Branch

`master` - Direct commits (LP1: 6b260c0, LP2: cb9f686, LP3: e5a7c49)

## Next Steps

1. Enable polling in staging environment
2. Monitor metrics and staleness
3. Add alerting rules for poll failures
4. Consider SSE streaming for real-time price updates (future)
