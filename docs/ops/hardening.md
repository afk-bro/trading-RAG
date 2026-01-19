# v1.0.0 - Operational Hardening (Jan 2026)

Tagged release implementing four phases of operational safety for production readiness.

## Phase 1: Idempotency

- `X-Idempotency-Key` header for `/backtests/tune` endpoint
- Prevents duplicate tunes on client retries (corrupts leaderboards)
- Atomic claim pattern with `INSERT ON CONFLICT DO NOTHING`
- Pending request polling with configurable timeout

**Migration required**: `053_idempotency_keys.sql`

## Phase 2: Retention

- SQL functions for batch deletes:
  - `retention_prune_trade_events()`
  - `retention_prune_job_runs()`
  - `retention_prune_match_runs()`
- pg_cron scheduling at 3:15/3:20/3:25 AM (when available)
- Admin endpoints with `dry_run` support for preview
- Retention job logging in `retention_job_log` table

**Migration required**: `054_retention_functions.sql`, `055_retention_pgcron_schedule.sql`

## Phase 3: LLM Fallback

- Graceful degradation when LLM times out or errors
- `StrategyExplanation` response includes `degraded`, `reason_code`, `model`, `provider`
- Reason codes: `llm_timeout`, `llm_unconfigured`, `llm_error`, `llm_rate_limit`
- Full stack traces logged internally, user-safe messages returned

## Phase 4: SSE (Server-Sent Events)

- Real-time updates for admin coverage cockpit at `/admin/events/stream`
- `InMemoryEventBus` with abstract `EventBus` interface for future Redis/PgNotify
- Topic-based filtering: `coverage`, `backtests`
- Workspace-scoped event delivery
- `Last-Event-ID` reconnection support with event buffer

## Canary Tests

Located in `tests/integration/test_operational_hardening.py`:

| Test | Purpose |
|------|---------|
| `TestIdempotencyConcurrentRetry` | Same key returns same tune_id |
| `TestRetentionDryRun` | Dry run returns count without deletion |
| `TestLLMTimeoutFallback` | Timeout triggers fallback response |
| `TestSSEEventDelivery` | Event publish/subscribe verification |

**Note**: DB tests gracefully skip if migrations not applied.

---

## System Health Dashboard

Single-page operational health view for "what's broken?" diagnostics.

**Source**: `app/admin/system_health.py`

### Endpoints

- `GET /admin/system/health` - HTML dashboard with status cards
- `GET /admin/system/health.json` - Machine-readable JSON

### Component Health Checks

| Component | Metrics |
|-----------|---------|
| **Database** | Pool size, available connections, acquire latency P95, connection errors (5m) |
| **Qdrant** | Vector count, segment count, collection status, last error |
| **LLM** | Provider configured, degraded count (1h), error count (1h), last success/error |
| **SSE** | Subscriber count, events published (1h), queue drops (1h) |
| **Retention** | pg_cron availability, last run time/status, rows deleted, consecutive failures |
| **Idempotency** | Total keys, expired pending prune, pending requests, oldest ages |
| **Tunes** | Active tunes, completed/failed (24h), avg duration |
| **Ingestion** | Last success/failure per source type, pending jobs |

### Status Values

`ok`, `degraded`, `error`, `unknown`

### Design Decisions

- No external dependencies (queries internal state only)
- Decision-grade metrics (actionable, not just informational)
- Sub-second response time (cached where appropriate)
- Answers "should I wake someone up?" without log access

---

## Security & Operations

### Security Dependencies

Located in `app/deps/security.py`:

| Function | Purpose |
|----------|---------|
| `require_admin_token()` | Admin endpoint protection (hmac.compare_digest) |
| `require_workspace_access()` | Multi-tenant authorization stub |
| `RateLimiter` | Sliding window rate limiting (per-IP, per-workspace) |
| `WorkspaceSemaphore` | Per-workspace concurrency caps |

### Production Environment Variables

```bash
ADMIN_TOKEN=<secure-token>      # Required for /admin/* endpoints
DOCS_ENABLED=false              # Disable /docs in production
CORS_ORIGINS=https://app.com    # Explicit allowlist
CONFIG_PROFILE=production       # Environment tag
GIT_SHA=abc123                  # Set by CI/CD
BUILD_TIME=2025-01-09T12:00:00Z # Set by CI/CD
```

### Monitoring

Sentry tags: `kb_status`, `kb_confidence`, `strategy`, `workspace_id`, `collection`, `embed_model`

Sentry measurements: `kb.total_ms`, `kb.embed_ms`, `kb.qdrant_ms`, `kb.rerank_ms`

4xx errors filtered (not captured as exceptions).

### Prometheus Alerting

Located in `ops/prometheus/`:
- `rules/rag_core_alerts.yml` - 28 production-ready alerts across 10 subsystems
- `README.md` - Required metrics, threshold tuning, loading instructions

Alert subsystems: `platform`, `db`, `qdrant`, `llm`, `kb`, `backtests`, `wfo`, `retention`, `idempotency`, `sse`, `ingestion`
