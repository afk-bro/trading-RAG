# Tech Debt & Follow-ups

Tracked technical debt and planned improvements.

## Post-merge: Results Persistence

### 1) Idempotency for run plan creation

**Problem:** Client retries (timeouts, network blips) can create duplicate `run_plans`.

**Preferred approach:** Option A - idempotency key (client-controlled)

**Proposed schema:**
```sql
ALTER TABLE run_plans ADD COLUMN idempotency_key TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_run_plans_idempotency_key
    ON run_plans(idempotency_key) WHERE idempotency_key IS NOT NULL;
```

**Endpoint behavior:**
- Accept `X-Idempotency-Key` header
- If key exists → return existing `run_plan_id` (200/201 semantics consistent)
- If not → create new plan

### 2) Retention policy for run-level trade_events

**Problem:** `trade_events` journal grows unbounded.

**Plan:** Retain 30–90 days for `RUN_*` breadcrumbs.

**Cleanup query:**
```sql
DELETE FROM trade_events
WHERE created_at < NOW() - INTERVAL '90 days'
  AND event_type IN ('run_started', 'run_completed', 'run_failed', 'run_cancelled');
```

**Implementation options:**
- pg_cron job (preferred if available)
- Scheduled endpoint invoked by external cron/GHA
