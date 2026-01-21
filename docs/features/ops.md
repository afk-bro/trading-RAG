# Operations & Security

## System Health Dashboard (`app/admin/system_health.py`)

Single-page operational health view for "what's broken?" diagnostics.

**Endpoints**:
- `GET /admin/system/health` - HTML dashboard
- `GET /admin/system/health.json` - JSON

**Component Checks**:
- **Database**: Pool size, available connections, acquire latency P95, errors (5m)
- **Qdrant**: Vector count, segments, collection status, last error
- **LLM**: Provider configured, degraded/error counts (1h), last success/error
- **SSE**: Subscriber count, events published (1h), queue drops
- **Retention**: pg_cron availability, last run time/status, rows deleted
- **Idempotency**: Total keys, expired pending, pending requests
- **Tunes**: Active, completed/failed (24h), avg duration
- **Ingestion**: Last success/failure per source type, pending jobs

**Status Values**: `ok`, `degraded`, `error`, `unknown`

## Security (`app/deps/security.py`)

- `require_admin_token()` - Admin endpoint protection (hmac.compare_digest)
- `require_workspace_access()` - Multi-tenant authorization stub
- `RateLimiter` - Sliding window rate limiting
- `WorkspaceSemaphore` - Per-workspace concurrency caps

**Production Environment**:
```bash
ADMIN_TOKEN=<secure-token>      # Required for /admin/*
DOCS_ENABLED=false              # Disable /docs
CORS_ORIGINS=https://app.com    # Explicit allowlist
CONFIG_PROFILE=production
GIT_SHA=abc123                  # Set by CI/CD
BUILD_TIME=2025-01-09T12:00:00Z
```

## Monitoring

**Sentry**:
- Tags: `kb_status`, `kb_confidence`, `strategy`, `workspace_id`, `collection`, `embed_model`
- Measurements: `kb.total_ms`, `kb.embed_ms`, `kb.qdrant_ms`, `kb.rerank_ms`
- 4xx errors filtered

**Prometheus** (`ops/prometheus/`):
- `rules/rag_core_alerts.yml` - 28 alerts across 10 subsystems
- Subsystems: platform, db, qdrant, llm, kb, backtests, wfo, retention, idempotency, sse, ingestion

## v1.0.0 - Operational Hardening (Jan 2026)

### Phase 1: Idempotency
- `X-Idempotency-Key` header for `/backtests/tune`
- Atomic claim pattern with `INSERT ON CONFLICT DO NOTHING`
- Migration: `053_idempotency_keys.sql`

### Phase 2: Retention
- SQL functions: `retention_prune_trade_events()`, `retention_prune_job_runs()`, `retention_prune_match_runs()`
- pg_cron scheduling at 3:15/3:20/3:25 AM
- Admin endpoints with `dry_run` support
- Migrations: `054_retention_functions.sql`, `055_retention_pgcron_schedule.sql`

### Phase 3: LLM Fallback
- Graceful degradation on timeout/error
- Reason codes: `llm_timeout`, `llm_unconfigured`, `llm_error`, `llm_rate_limit`

### Phase 4: SSE
- Real-time updates at `/admin/events/stream`
- `InMemoryEventBus` with abstract interface
- Topic filtering: `coverage`, `backtests`
- Workspace-scoped delivery
- `Last-Event-ID` reconnection support

### Canary Tests
`tests/integration/test_operational_hardening.py`:
- `TestIdempotencyConcurrentRetry`
- `TestRetentionDryRun`
- `TestLLMTimeoutFallback`
- `TestSSEEventDelivery`

## Tech Debt / Follow-ups

See `docs/tech-debt.md` for current items.

### Results Persistence Follow-ups

**1) Idempotency for run plan creation**
```sql
ALTER TABLE run_plans ADD COLUMN idempotency_key TEXT;
CREATE UNIQUE INDEX idx_run_plans_idempotency_key ON run_plans(idempotency_key) WHERE idempotency_key IS NOT NULL;
```

**2) Retention for run-level trade_events**
Retain 30-90 days for `RUN_*` breadcrumbs via pg_cron or scheduled endpoint.
