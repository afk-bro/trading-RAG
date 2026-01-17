# Idempotency Observability Design

**Date**: 2026-01-17
**Status**: Implemented
**Related Commits**: `76b0654`, `404938b`

## Problem Statement

The idempotency key system (migrations 053-055) provides duplicate request prevention and automatic cleanup via pg_cron. However, there was no visibility into:
- Whether the prune job is running successfully
- Accumulation of expired keys (indicates pg_cron failure)
- Stuck pending requests (indicates processing issues)

Operators had to query the database directly to detect hygiene issues.

## Goals

1. **Visibility**: Same metrics in health page, Prometheus, and alerts
2. **Automatic Action**: pg_cron prunes expired keys nightly
3. **Alerting**: Prometheus alerts fire when hygiene degrades
4. **Self-Maintaining**: System corrects itself without operator intervention

## Design

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Feedback Loop                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│   │ Health Page  │────▶│  Prometheus  │────▶│ Alertmanager │        │
│   │ /admin/      │     │  /metrics    │     │              │        │
│   │ system/health│     │              │     │              │        │
│   └──────┬───────┘     └──────────────┘     └──────────────┘        │
│          │                                                           │
│          │ set_idempotency_metrics()                                │
│          │                                                           │
│   ┌──────▼───────┐                                                   │
│   │ idempotency_ │                                                   │
│   │ keys table   │◀────── pg_cron (3:30 AM UTC daily)               │
│   └──────────────┘        retention_prune_idempotency_keys()        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Health Page Integration (`app/admin/system_health.py`)

New `IdempotencyHealth` model:
```python
class IdempotencyHealth(ComponentHealth):
    total_keys: int = 0           # Current table size
    expired_pending: int = 0      # Keys past expiration
    pending_requests: int = 0     # Active in-flight
    oldest_pending_age_minutes: Optional[float] = None
    oldest_expired_age_hours: Optional[float] = None
```

Status thresholds:
- `ok`: expired_pending < 100, oldest_expired < 48h
- `degraded`: expired_pending > 100 OR oldest_pending > 30 min
- `error`: expired_pending > 1000 OR oldest_expired > 48h

#### 2. Prometheus Metrics (`app/routers/metrics.py`)

| Metric | Type | Description |
|--------|------|-------------|
| `idempotency_keys_total` | Gauge | Total keys in table |
| `idempotency_expired_pending_total` | Gauge | Expired awaiting prune |
| `idempotency_pending_requests_total` | Gauge | Active in-flight |
| `idempotency_oldest_pending_age_minutes` | Gauge | Staleness indicator |
| `idempotency_oldest_expired_age_hours` | Gauge | pg_cron health |

Metrics updated during health check via `set_idempotency_metrics()`.

#### 3. Alert Rules (`ops/prometheus/rules/rag_core_alerts.yml`)

| Alert | Condition | Severity |
|-------|-----------|----------|
| `IdempotencyExpiredPending` | >100 for 30m | warning |
| `IdempotencyExpiredCritical` | >1000 for 1h | critical |
| `IdempotencyPruneStale` | oldest_expired >48h | critical |
| `IdempotencyPendingStuck` | oldest_pending >30min | warning |

#### 4. Retention (Pre-existing)

Migration 054: `retention_prune_idempotency_keys()` SQL function
Migration 055: pg_cron schedule at 3:30 AM UTC daily

## SQL Query

```sql
SELECT
    COUNT(*) as total_keys,
    COUNT(*) FILTER (WHERE expires_at < NOW()) as expired_pending,
    COUNT(*) FILTER (
        WHERE status = 'pending' AND expires_at >= NOW()
    ) as pending_requests,
    EXTRACT(EPOCH FROM (NOW() - MIN(created_at)))::FLOAT / 60.0
        FILTER (WHERE status = 'pending' AND expires_at >= NOW())
        as oldest_pending_age_minutes,
    EXTRACT(EPOCH FROM (NOW() - MIN(expires_at)))::FLOAT / 3600.0
        FILTER (WHERE expires_at < NOW())
        as oldest_expired_age_hours
FROM idempotency_keys
```

## Files Changed

- `app/admin/system_health.py` - IdempotencyHealth class, _check_idempotency()
- `app/routers/metrics.py` - Prometheus gauges, set_idempotency_metrics()
- `ops/prometheus/rules/rag_core_alerts.yml` - Alert rules
- `ops/prometheus/README.md` - Documentation

## Testing

- Unit tests pass (2091 passed)
- mypy clean
- Graceful fallback when idempotency_keys table doesn't exist

## Future Considerations

- Add retention job duration metrics for pg_cron monitoring
- Consider adding idempotency key creation rate for capacity planning
- Dashboard panel in Grafana for visual monitoring
