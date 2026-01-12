# PR11: Ops & Safety Layer Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make retention/rollup jobs production-grade with locking, tracking, dry-run, permissions, and alerting.

**Architecture:** Advisory locks for concurrency control + job_runs table for telemetry. Jobs run on lock-holding connection. Failures logged to trade_events for Sentry alerting.

**Tech Stack:** PostgreSQL advisory locks, asyncpg, FastAPI, Jinja2 templates

---

## 1. Job Runs Table Schema

```sql
CREATE TABLE job_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_name TEXT NOT NULL,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    duration_ms INTEGER,
    dry_run BOOLEAN NOT NULL DEFAULT FALSE,
    metrics JSONB NOT NULL DEFAULT '{}',
    error TEXT,
    triggered_by TEXT,
    correlation_id UUID,

    CONSTRAINT job_runs_status_check
        CHECK (status IN ('running', 'completed', 'failed', 'skipped'))
);

-- Admin UI: last N runs per job (global view)
CREATE INDEX idx_job_runs_job_started
    ON job_runs(job_name, started_at DESC);

-- Admin UI: filter by workspace
CREATE INDEX idx_job_runs_workspace_job_started
    ON job_runs(workspace_id, job_name, started_at DESC);

-- Stale detection
CREATE INDEX idx_job_runs_running
    ON job_runs(status, started_at) WHERE status = 'running';
```

---

## 2. Advisory Lock Strategy

```python
import hashlib
from uuid import UUID, uuid4

def job_lock_key(job_name: str, workspace_id: UUID) -> int:
    """Generate stable 64-bit unsigned lock key."""
    raw = f"{job_name}:{workspace_id}"
    h = hashlib.sha256(raw.encode()).digest()[:8]
    return int.from_bytes(h, byteorder="big", signed=False)
```

**Lock flow:**
1. Acquire `pg_try_advisory_lock(lock_key::bigint)` on dedicated connection
2. If not acquired → return 409 (no row written)
3. Create `job_runs` row with `status='running'`
4. Execute job on same connection (lock holder)
5. Update row to `completed` or `failed`
6. Release lock in finally block

**Crash behavior:** Advisory lock auto-releases. Stale `running` rows are cosmetic (UI shows "stale" badge if `updated_at < NOW() - 1 hour`).

**Duration calculation:** `(EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)::int`

---

## 3. API Contract

### Job Endpoints

```
POST /admin/jobs/rollup-events?workspace_id=<uuid>&dry_run=false
POST /admin/jobs/cleanup-events?workspace_id=<uuid>&dry_run=false
```

**Auth:** `X-Admin-Token` OR `Authorization: Bearer <JWT>` with workspace role `admin|owner`

**Parameters:**
| Param | Type | Required | Default |
|-------|------|----------|---------|
| workspace_id | UUID | Yes | - |
| dry_run | bool | No | false |

### Responses

**200 OK (completed):**
```json
{
  "job_name": "rollup_events",
  "workspace_id": "uuid",
  "lock_acquired": true,
  "status": "completed",
  "run_id": "uuid",
  "correlation_id": "uuid",
  "dry_run": false,
  "metrics": {
    "rows_rolled_up": 150,
    "days_affected": 7
  },
  "warnings": []
}
```

**409 Conflict (already running):**
```json
{
  "job_name": "rollup_events",
  "workspace_id": "uuid",
  "lock_acquired": false,
  "status": "already_running",
  "run_id": null
}
```

**500 Internal Server Error (failed, caught and returned as JSON):**
```json
{
  "job_name": "rollup_events",
  "workspace_id": "uuid",
  "lock_acquired": true,
  "status": "failed",
  "run_id": "uuid",
  "correlation_id": "uuid",
  "error": "truncated error message"
}
```

### Job Runs List/Detail Endpoints

```
GET /admin/jobs/runs?job_name=<optional>&workspace_id=<optional>&limit=20
GET /admin/jobs/runs/{run_id}
```

**Auth:** Same as job POSTs (`X-Admin-Token` OR JWT with `admin|owner`)

**List response:** Returns `metrics_preview` (truncated), click for full detail.

**Detail response:** Returns full `metrics` + `error` text.

---

## 4. Admin UI - Job Runs Page

**Route:** `GET /admin/jobs` (HTML)

**Features:**
- Filter by job_name dropdown
- Filter by workspace_id
- Last 20 runs, paginated
- Status badges:
  - `completed` (green)
  - `failed` (red)
  - `running` (blue)
  - `running + stale` (yellow) - when `updated_at < NOW() - 1 hour`
- Columns: job_name, workspace, status, started_at, duration_ms, dry_run, metrics_preview
- Click row → fetch `/admin/jobs/runs/{run_id}` for full detail modal

**Stale detection query:**
```sql
CASE
  WHEN status = 'running' AND updated_at < NOW() - INTERVAL '1 hour'
  THEN 'stale'
  ELSE status
END as display_status
```

---

## 5. Alerting on Failure

**v1 (minimal):**
1. Log at ERROR level with structlog (captured by Sentry)
2. Write `trade_events` row for visibility:

```python
await conn.execute("""
    INSERT INTO trade_events
        (workspace_id, event_type, severity, correlation_id, payload)
    VALUES ($1, 'JOB_FAILED', 'ERROR', $2, $3)
""", workspace_id, correlation_id, json.dumps({
    "job_name": job_name,
    "run_id": str(run_id),
    "error": str(e)[:500]
}))
```

**v2 (later):** Sentry alert rules on `event_type='JOB_FAILED'` tag, webhook/email notifications.

---

## 6. Stale Row Cleanup (Nice-to-Have)

Nightly cleanup for cosmetic hygiene:

```sql
UPDATE job_runs
SET status = 'failed',
    finished_at = NOW(),
    updated_at = NOW(),
    error = 'stale run (process terminated)'
WHERE status = 'running'
  AND updated_at < NOW() - INTERVAL '24 hours';
```

Run via same job framework or pg_cron.

---

## 7. Severity Convention

Standardize across codebase:
- `DEBUG` - verbose debugging
- `INFO` - normal operations
- `WARN` - recoverable issues
- `ERROR` - failures requiring attention

All uppercase in `trade_events.severity`.

---

## Implementation Tasks

1. Migration: Create `job_runs` table
2. Service: `JobRunner` with advisory lock + tracking
3. Update: `rollup_events` endpoint to use JobRunner
4. Update: `cleanup_events` endpoint to use JobRunner
5. Endpoint: `GET /admin/jobs/runs` (list)
6. Endpoint: `GET /admin/jobs/runs/{run_id}` (detail)
7. UI: Job runs HTML page with filters and status badges
8. Alert: Write `JOB_FAILED` event on failure
9. Test: Unit tests for lock acquisition, failure handling
10. Test: Integration test for concurrent job rejection
