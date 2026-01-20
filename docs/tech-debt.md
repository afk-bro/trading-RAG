# Tech Debt & Follow-ups

Tracked technical debt and planned improvements.

---

## Completed

### Idempotency for run plan creation

**Status:** Done (migrations 039, 053)

- `idempotency_key` and `request_hash` columns on `run_plans`
- Generic `idempotency_keys` table for broader API use
- `X-Idempotency-Key` header support in `/backtests/tunes`
- Service: `app/services/testing/idempotency.py`
- Tests: `tests/unit/routers/test_testing_idempotency.py`

### Retention policy for trade_events

**Status:** Done (migrations 040, 054, 055)

- `retention_prune_trade_events()` function with batch deletes
- 90-day retention for `RUN_*` events, `ORDER_*` retained forever
- pg_cron schedule at 3:15 AM UTC daily
- Audit log: `retention_job_log` table
- Additional retention functions: `job_runs` (30d), `match_runs` (180d), `idempotency_keys` (7d)

### Connection resilience

**Status:** Done (app/core/resilience.py)

- Circuit breakers for DB and Qdrant
- Retry with exponential backoff
- Prometheus metrics: `resilience_retries_total`, `circuit_breaker_state`
- Health endpoint exposes circuit status

---

## Active Tech Debt

### Minor TODOs in code

| Location | Issue | Priority |
|----------|-------|----------|
| `app/deps/security.py:120` | Auth provider integration placeholder | Low |
| `app/services/kb/status_service.py:358` | Trigger re-ingestion not wired | Low |
| `app/routers/kb_trials.py:851` | Dataset loading from storage by ID | Low |
| `app/routers/query.py:212` | Workspace config fetch from DB | Low |

### Test coverage

**Current:** 57% (temporarily lowered from 70%)

**Target:** 65% post-v1.0

**Approach:** Surgical improvements as code is touched, not standalone test-writing sprints.

---

## Deferred Features

These are designed but intentionally deferred:

| Feature | Design Doc | Notes |
|---------|------------|-------|
| Alert rules table | 2026-01-19-parallel-minimal-alerts-design.md | Per-workspace thresholds |
| Full alerts admin UI | Same | List page, detail page, charts |
| Webhook sink | Same | Slack, PagerDuty delivery |
| v1.5 Live Intelligence | 2026-01-09-v1.5-*.md | Regime awareness, confidence decomposition |
