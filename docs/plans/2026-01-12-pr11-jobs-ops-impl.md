# PR11: Jobs Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Production-grade job infrastructure with locking, tracking, dry-run, and alerting.

**Branch:** `feature/pr11-jobs-ops` (off `b631c68`)

---

## Task 1: Migration - job_runs table

**Files:**
- Create: `migrations/043_job_runs.sql`

**Step 1: Write migration**

```sql
-- migrations/043_job_runs.sql
-- Job runs tracking table for retention/rollup jobs

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

**Step 2: Apply migration via Supabase MCP**

**Step 3: Commit**

```bash
git add migrations/043_job_runs.sql
git commit -m "feat(db): add job_runs table for job tracking"
```

---

## Task 2: Advisory lock utility

**Files:**
- Create: `app/services/jobs/__init__.py`
- Create: `app/services/jobs/locks.py`
- Test: `tests/unit/test_job_locks.py`

**Step 1: Write the test**

```python
# tests/unit/test_job_locks.py
"""Tests for job advisory lock utilities."""

from uuid import uuid4

from app.services.jobs.locks import job_lock_key


class TestJobLockKey:
    """Tests for lock key generation."""

    def test_deterministic_key(self):
        """Same inputs produce same key."""
        workspace_id = uuid4()
        key1 = job_lock_key("rollup_events", workspace_id)
        key2 = job_lock_key("rollup_events", workspace_id)
        assert key1 == key2

    def test_different_jobs_different_keys(self):
        """Different job names produce different keys."""
        workspace_id = uuid4()
        key1 = job_lock_key("rollup_events", workspace_id)
        key2 = job_lock_key("cleanup_events", workspace_id)
        assert key1 != key2

    def test_different_workspaces_different_keys(self):
        """Different workspaces produce different keys."""
        key1 = job_lock_key("rollup_events", uuid4())
        key2 = job_lock_key("rollup_events", uuid4())
        assert key1 != key2

    def test_key_is_unsigned_int(self):
        """Key is unsigned 64-bit integer."""
        key = job_lock_key("test", uuid4())
        assert isinstance(key, int)
        assert key >= 0
```

**Step 2: Write implementation**

```python
# app/services/jobs/__init__.py
"""Job execution services."""

from app.services.jobs.locks import job_lock_key

__all__ = ["job_lock_key"]
```

```python
# app/services/jobs/locks.py
"""Advisory lock utilities for job execution."""

import hashlib
from uuid import UUID


def job_lock_key(job_name: str, workspace_id: UUID) -> int:
    """
    Generate stable 64-bit unsigned lock key for pg_try_advisory_lock.

    Args:
        job_name: Name of the job (e.g., "rollup_events")
        workspace_id: Workspace UUID for scoping

    Returns:
        Unsigned 64-bit integer suitable for advisory lock
    """
    raw = f"{job_name}:{workspace_id}"
    h = hashlib.sha256(raw.encode()).digest()[:8]
    return int.from_bytes(h, byteorder="big", signed=False)
```

**Step 3: Run test**

```bash
SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key \
    pytest tests/unit/test_job_locks.py -v
```

**Step 4: Commit**

```bash
git add app/services/jobs/ tests/unit/test_job_locks.py
git commit -m "feat(jobs): add advisory lock key utility"
```

---

## Task 3: Job runner with lock + tracking

**Files:**
- Create: `app/services/jobs/runner.py`
- Test: `tests/unit/test_job_runner.py`

**Step 1: Write the test**

```python
# tests/unit/test_job_runner.py
"""Tests for job runner with advisory lock."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.services.jobs.runner import JobRunner, JobResult


class TestJobRunner:
    """Tests for JobRunner execution."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        return pool, conn

    @pytest.mark.asyncio
    async def test_lock_acquired_job_succeeds(self, mock_pool):
        """Successful job creates completed run."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        run_id = uuid4()

        # Lock acquired
        conn.fetchval.side_effect = [True, run_id]
        conn.execute.return_value = None

        async def job_fn(c, dry_run, correlation_id):
            return {"rows_processed": 10}

        runner = JobRunner(pool)
        result = await runner.run(
            job_name="test_job",
            workspace_id=workspace_id,
            dry_run=False,
            triggered_by="admin_token",
            job_fn=job_fn,
        )

        assert result.lock_acquired is True
        assert result.status == "completed"
        assert result.metrics["rows_processed"] == 10

    @pytest.mark.asyncio
    async def test_lock_not_acquired_returns_already_running(self, mock_pool):
        """When lock held, returns already_running without writing row."""
        pool, conn = mock_pool
        workspace_id = uuid4()

        # Lock NOT acquired
        conn.fetchval.return_value = False

        async def job_fn(c, dry_run, correlation_id):
            return {}

        runner = JobRunner(pool)
        result = await runner.run(
            job_name="test_job",
            workspace_id=workspace_id,
            dry_run=False,
            triggered_by="admin_token",
            job_fn=job_fn,
        )

        assert result.lock_acquired is False
        assert result.status == "already_running"
        assert result.run_id is None
        # No INSERT should have been called
        assert "INSERT INTO job_runs" not in str(conn.execute.call_args_list)

    @pytest.mark.asyncio
    async def test_job_failure_writes_failed_row_and_event(self, mock_pool):
        """Failed job writes failed row + JOB_FAILED event."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        run_id = uuid4()

        conn.fetchval.side_effect = [True, run_id]
        conn.execute.return_value = None

        async def job_fn(c, dry_run, correlation_id):
            raise ValueError("Something broke")

        runner = JobRunner(pool)

        with pytest.raises(ValueError):
            await runner.run(
                job_name="test_job",
                workspace_id=workspace_id,
                dry_run=False,
                triggered_by="admin_token",
                job_fn=job_fn,
            )

        # Should have updated to failed + inserted trade_event
        execute_calls = [str(c) for c in conn.execute.call_args_list]
        assert any("status = 'failed'" in c for c in execute_calls)
        assert any("JOB_FAILED" in c for c in execute_calls)
```

**Step 2: Write implementation**

```python
# app/services/jobs/runner.py
"""Job runner with advisory locking and tracking."""

import json
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional
from uuid import UUID, uuid4

import structlog

from app.services.jobs.locks import job_lock_key

logger = structlog.get_logger(__name__)


@dataclass
class JobResult:
    """Result of a job execution."""

    job_name: str
    workspace_id: UUID
    lock_acquired: bool
    status: str  # running, completed, failed, already_running
    run_id: Optional[UUID] = None
    correlation_id: Optional[UUID] = None
    dry_run: bool = False
    metrics: dict = None
    warnings: list = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.warnings is None:
            self.warnings = []

    def to_dict(self) -> dict:
        """Convert to API response dict."""
        return {
            "job_name": self.job_name,
            "workspace_id": str(self.workspace_id),
            "lock_acquired": self.lock_acquired,
            "status": self.status,
            "run_id": str(self.run_id) if self.run_id else None,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "dry_run": self.dry_run,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "error": self.error,
        }


JobFn = Callable[[Any, bool, UUID], Coroutine[Any, Any, dict]]


class JobRunner:
    """Execute jobs with advisory lock protection and tracking."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def run(
        self,
        job_name: str,
        workspace_id: UUID,
        dry_run: bool,
        triggered_by: str,
        job_fn: JobFn,
    ) -> JobResult:
        """
        Execute job with advisory lock and tracking.

        The lock-holding connection runs the job. On crash, lock auto-releases.

        Args:
            job_name: Job identifier (e.g., "rollup_events")
            workspace_id: Workspace scope
            dry_run: If True, job returns counts without modifying data
            triggered_by: Auth context (e.g., "admin_token", "user:<uuid>")
            job_fn: Async function(conn, dry_run, correlation_id) -> metrics dict

        Returns:
            JobResult with status and metrics
        """
        lock_key = job_lock_key(job_name, workspace_id)
        correlation_id = uuid4()
        run_id = None

        async with self.pool.acquire() as conn:
            # Try to acquire advisory lock (non-blocking)
            locked = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1::bigint)", lock_key
            )

            if not locked:
                logger.info(
                    "Job lock not acquired",
                    job_name=job_name,
                    workspace_id=str(workspace_id),
                )
                return JobResult(
                    job_name=job_name,
                    workspace_id=workspace_id,
                    lock_acquired=False,
                    status="already_running",
                )

            try:
                # Create tracking row
                run_id = await conn.fetchval(
                    """
                    INSERT INTO job_runs
                        (job_name, workspace_id, dry_run, triggered_by, correlation_id)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    job_name,
                    workspace_id,
                    dry_run,
                    triggered_by,
                    correlation_id,
                )

                logger.info(
                    "Job started",
                    job_name=job_name,
                    run_id=str(run_id),
                    correlation_id=str(correlation_id),
                    dry_run=dry_run,
                )

                # Execute job on lock-holding connection
                metrics = await job_fn(conn, dry_run, correlation_id)

                # Mark completed
                await conn.execute(
                    """
                    UPDATE job_runs
                    SET status = 'completed',
                        finished_at = NOW(),
                        updated_at = NOW(),
                        duration_ms = (EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)::int,
                        metrics = $2
                    WHERE id = $1
                    """,
                    run_id,
                    json.dumps(metrics),
                )

                logger.info(
                    "Job completed",
                    job_name=job_name,
                    run_id=str(run_id),
                    metrics=metrics,
                )

                return JobResult(
                    job_name=job_name,
                    workspace_id=workspace_id,
                    lock_acquired=True,
                    status="completed",
                    run_id=run_id,
                    correlation_id=correlation_id,
                    dry_run=dry_run,
                    metrics=metrics,
                )

            except Exception as e:
                error_msg = str(e)[:1000]
                logger.error(
                    "Job failed",
                    job_name=job_name,
                    run_id=str(run_id) if run_id else None,
                    error=error_msg,
                )

                if run_id:
                    # Update job_runs to failed
                    await conn.execute(
                        """
                        UPDATE job_runs
                        SET status = 'failed',
                            finished_at = NOW(),
                            updated_at = NOW(),
                            duration_ms = (EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)::int,
                            error = $2
                        WHERE id = $1
                        """,
                        run_id,
                        error_msg,
                    )

                    # Write JOB_FAILED event for alerting
                    await conn.execute(
                        """
                        INSERT INTO trade_events
                            (workspace_id, event_type, severity, correlation_id, payload)
                        VALUES ($1, 'JOB_FAILED', 'ERROR', $2, $3)
                        """,
                        workspace_id,
                        correlation_id,
                        json.dumps({
                            "job_name": job_name,
                            "run_id": str(run_id),
                            "error": error_msg[:500],
                        }),
                    )

                raise

            finally:
                # Release lock (also auto-releases on connection close)
                await conn.execute(
                    "SELECT pg_advisory_unlock($1::bigint)", lock_key
                )
```

**Step 3: Run tests**

```bash
SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key \
    pytest tests/unit/test_job_runner.py -v
```

**Step 4: Update __init__.py exports**

```python
# app/services/jobs/__init__.py
"""Job execution services."""

from app.services.jobs.locks import job_lock_key
from app.services.jobs.runner import JobRunner, JobResult

__all__ = ["job_lock_key", "JobRunner", "JobResult"]
```

**Step 5: Commit**

```bash
git add app/services/jobs/ tests/unit/test_job_runner.py
git commit -m "feat(jobs): add JobRunner with advisory lock and tracking"
```

---

## Task 4: Wire existing endpoints through runner

**Files:**
- Modify: `app/admin/router.py`
- Test: `tests/unit/test_admin_jobs.py`

**Step 1: Write the test**

```python
# tests/unit/test_admin_jobs.py
"""Tests for admin job endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


class TestAdminJobEndpoints:
    """Tests for /admin/jobs/* endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked deps."""
        from app.main import app
        return TestClient(app)

    def test_rollup_requires_workspace_id(self, client):
        """Rollup endpoint requires workspace_id."""
        response = client.post(
            "/admin/jobs/rollup-events",
            headers={"X-Admin-Token": "test-token"},
        )
        assert response.status_code == 422  # Missing required param

    def test_cleanup_requires_workspace_id(self, client):
        """Cleanup endpoint requires workspace_id."""
        response = client.post(
            "/admin/jobs/cleanup-events",
            headers={"X-Admin-Token": "test-token"},
        )
        assert response.status_code == 422

    def test_unauthorized_without_token(self, client):
        """Endpoints require auth."""
        workspace_id = uuid4()
        response = client.post(
            f"/admin/jobs/rollup-events?workspace_id={workspace_id}",
        )
        assert response.status_code in (401, 403)
```

**Step 2: Update router**

Add to `app/admin/router.py`:

```python
from uuid import UUID
from app.services.jobs import JobRunner, JobResult
from app.repositories.event_rollups import EventRollupsRepository
from app.services.retention import RetentionService


def _get_job_runner():
    """Get JobRunner instance."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return JobRunner(_db_pool)


@router.post("/jobs/rollup-events")
async def trigger_rollup_events(
    workspace_id: UUID = Query(..., description="Workspace to rollup events for"),
    dry_run: bool = Query(False, description="Preview without modifying data"),
    _: str = Depends(require_admin_token),
):
    """Trigger event rollup job for workspace."""
    runner = _get_job_runner()

    async def rollup_job(conn, is_dry_run: bool, correlation_id: UUID) -> dict:
        from datetime import date, timedelta

        target_date = date.today() - timedelta(days=1)

        if is_dry_run:
            # Count what would be rolled up
            count = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT (workspace_id, strategy_entity_id, event_type))
                FROM trade_events
                WHERE workspace_id = $1
                  AND created_at >= $2::date
                  AND created_at < ($2::date + INTERVAL '1 day')
                """,
                workspace_id,
                target_date,
            )
            return {"rows_would_rollup": count, "target_date": str(target_date)}

        repo = EventRollupsRepository(conn)
        count = await repo.run_daily_rollup(target_date, workspace_id)
        return {"rows_rolled_up": count, "target_date": str(target_date)}

    try:
        result = await runner.run(
            job_name="rollup_events",
            workspace_id=workspace_id,
            dry_run=dry_run,
            triggered_by="admin_token",
            job_fn=rollup_job,
        )

        if not result.lock_acquired:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=result.to_dict(),
            )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        logger.exception("Rollup job failed", workspace_id=str(workspace_id))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "job_name": "rollup_events",
                "workspace_id": str(workspace_id),
                "lock_acquired": True,
                "status": "failed",
                "error": str(e)[:1000],
            },
        )


@router.post("/jobs/cleanup-events")
async def trigger_cleanup_events(
    workspace_id: UUID = Query(..., description="Workspace to cleanup events for"),
    dry_run: bool = Query(False, description="Preview without deleting"),
    _: str = Depends(require_admin_token),
):
    """Trigger event retention cleanup job for workspace."""
    runner = _get_job_runner()

    async def cleanup_job(conn, is_dry_run: bool, correlation_id: UUID) -> dict:
        service = RetentionService(conn)

        if is_dry_run:
            counts = await service.preview_cleanup(workspace_id)
            return {"would_delete": counts}

        result = await service.run_cleanup(workspace_id)
        return result

    try:
        result = await runner.run(
            job_name="cleanup_events",
            workspace_id=workspace_id,
            dry_run=dry_run,
            triggered_by="admin_token",
            job_fn=cleanup_job,
        )

        if not result.lock_acquired:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=result.to_dict(),
            )

        return JSONResponse(content=result.to_dict())

    except Exception as e:
        logger.exception("Cleanup job failed", workspace_id=str(workspace_id))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "job_name": "cleanup_events",
                "workspace_id": str(workspace_id),
                "lock_acquired": True,
                "status": "failed",
                "error": str(e)[:1000],
            },
        )
```

**Step 3: Update RetentionService and EventRollupsRepository**

Add `workspace_id` parameter and `preview_cleanup` method as needed.

**Step 4: Run tests**

**Step 5: Commit**

```bash
git add app/admin/router.py app/services/retention.py app/repositories/event_rollups.py tests/unit/test_admin_jobs.py
git commit -m "feat(admin): wire job endpoints through JobRunner"
```

---

## Task 5: Job runs list/detail endpoints

**Files:**
- Modify: `app/admin/router.py`
- Create: `app/repositories/job_runs.py`
- Test: `tests/unit/test_job_runs_repo.py`

**Step 1: Create repository**

```python
# app/repositories/job_runs.py
"""Repository for job runs tracking."""

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class JobRunsRepository:
    """Repository for job run queries."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def list_runs(
        self,
        job_name: Optional[str] = None,
        workspace_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """List job runs with filters and pagination."""
        conditions = []
        params = []
        param_idx = 1

        if job_name:
            conditions.append(f"job_name = ${param_idx}")
            params.append(job_name)
            param_idx += 1

        if workspace_id:
            conditions.append(f"workspace_id = ${param_idx}")
            params.append(workspace_id)
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT
                id, job_name, workspace_id, status, started_at, finished_at,
                updated_at, duration_ms, dry_run, triggered_by, correlation_id,
                LEFT(metrics::text, 200) as metrics_preview,
                CASE
                    WHEN status = 'running' AND updated_at < NOW() - INTERVAL '1 hour'
                    THEN 'stale'
                    ELSE status
                END as display_status
            FROM job_runs
            WHERE {where_clause}
            ORDER BY started_at DESC, id DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(r) for r in rows]

    async def get_run(self, run_id: UUID) -> Optional[dict]:
        """Get full job run details."""
        query = """
            SELECT
                id, job_name, workspace_id, status, started_at, finished_at,
                updated_at, duration_ms, dry_run, triggered_by, correlation_id,
                metrics, error,
                CASE
                    WHEN status = 'running' AND updated_at < NOW() - INTERVAL '1 hour'
                    THEN 'stale'
                    ELSE status
                END as display_status
            FROM job_runs
            WHERE id = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, run_id)

        return dict(row) if row else None
```

**Step 2: Add endpoints to router**

```python
@router.get("/jobs/runs")
async def list_job_runs(
    job_name: Optional[str] = Query(None),
    workspace_id: Optional[UUID] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: str = Depends(require_admin_token),
):
    """List job runs with filters."""
    from app.repositories.job_runs import JobRunsRepository

    repo = JobRunsRepository(_db_pool)
    runs = await repo.list_runs(
        job_name=job_name,
        workspace_id=workspace_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return {"runs": runs, "count": len(runs)}


@router.get("/jobs/runs/{run_id}")
async def get_job_run(
    run_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Get full job run details."""
    from app.repositories.job_runs import JobRunsRepository

    repo = JobRunsRepository(_db_pool)
    run = await repo.get_run(run_id)

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job run not found",
        )

    return run
```

**Step 3: Write tests and commit**

```bash
git add app/repositories/job_runs.py app/admin/router.py tests/unit/test_job_runs_repo.py
git commit -m "feat(admin): add job runs list and detail endpoints"
```

---

## Task 6: Admin UI page /admin/jobs

**Files:**
- Create: `app/admin/templates/jobs.html`
- Modify: `app/admin/router.py`

**Step 1: Create template**

Create a simple HTML page with:
- Job name dropdown filter
- Workspace ID filter input
- Table with status badges (completed=green, failed=red, running=blue, stale=yellow)
- Click row to expand and fetch full details via JS

**Step 2: Add HTML route**

```python
@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(
    request: Request,
    _: str = Depends(require_admin_token),
):
    """Admin job runs page."""
    return templates.TemplateResponse(
        "jobs.html",
        {"request": request},
    )
```

**Step 3: Commit**

```bash
git add app/admin/templates/jobs.html app/admin/router.py
git commit -m "feat(admin): add jobs UI page with filters and status badges"
```

---

## Task 7: Tests - auth and concurrency

**Files:**
- Modify: `tests/unit/test_admin_jobs.py`

**Step 1: Add auth tests**

```python
def test_admin_token_auth_works(self, client):
    """Admin token grants access."""
    # Mock the runner to avoid DB
    with patch("app.admin.router._get_job_runner") as mock:
        mock.return_value.run = AsyncMock(return_value=JobResult(...))
        response = client.post(
            f"/admin/jobs/rollup-events?workspace_id={uuid4()}",
            headers={"X-Admin-Token": "valid-token"},
        )
        assert response.status_code in (200, 409)


def test_jwt_admin_role_works(self, client):
    """JWT with admin role grants access."""
    # Similar test with mocked JWT validation
    pass


def test_member_role_forbidden(self, client):
    """JWT with member role is forbidden."""
    pass
```

**Step 2: Add concurrency test**

```python
@pytest.mark.asyncio
async def test_concurrent_job_returns_409():
    """Second job attempt while first running returns 409."""
    # Acquire lock manually, then call endpoint
    pass
```

**Step 3: Run all tests and commit**

```bash
SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key \
    pytest tests/unit/ -v --tb=short
git add tests/
git commit -m "test(jobs): add auth and concurrency tests"
```

---

## Task 8: Final verification and PR

**Step 1: Run full test suite**

```bash
SUPABASE_URL=https://test.supabase.co SUPABASE_SERVICE_ROLE_KEY=test-key \
    pytest tests/unit/ -v --tb=short
```

**Step 2: Run linting**

```bash
black --check app/ tests/
flake8 app/ tests/ --max-line-length=100
```

**Step 3: Push and create PR**

```bash
git push -u origin feature/pr11-jobs-ops
gh pr create --title "feat: PR11 Jobs Ops - locking, tracking, UI" --body "..."
```
