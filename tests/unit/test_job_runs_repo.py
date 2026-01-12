"""Tests for job runs repository."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.repositories.job_runs import JobRunsRepository


class TestJobRunsRepository:
    """Tests for JobRunsRepository operations."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        return pool, conn

    @pytest.mark.asyncio
    async def test_list_runs_no_filters(self, mock_pool):
        """Lists runs without filters."""
        pool, conn = mock_pool
        run_id = uuid4()
        workspace_id = uuid4()
        conn.fetch.return_value = [
            {
                "id": run_id,
                "job_name": "rollup_events",
                "workspace_id": workspace_id,
                "status": "completed",
                "started_at": datetime.now() - timedelta(hours=1),
                "finished_at": datetime.now(),
                "duration_ms": 150,
                "dry_run": False,
                "metrics_preview": '{"rows": 10}',
                "display_status": "completed",
            }
        ]

        repo = JobRunsRepository(pool)
        runs = await repo.list_runs()

        assert len(runs) == 1
        assert runs[0]["job_name"] == "rollup_events"
        assert runs[0]["status"] == "completed"
        conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_runs_with_job_name_filter(self, mock_pool):
        """Filters by job name."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        repo = JobRunsRepository(pool)
        await repo.list_runs(job_name="cleanup_events")

        # Verify job_name filter in query
        call_args = conn.fetch.call_args
        query = call_args[0][0]
        assert "job_name = $1" in query
        assert call_args[0][1] == "cleanup_events"

    @pytest.mark.asyncio
    async def test_list_runs_with_workspace_filter(self, mock_pool):
        """Filters by workspace ID."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        conn.fetch.return_value = []

        repo = JobRunsRepository(pool)
        await repo.list_runs(workspace_id=workspace_id)

        # Verify workspace_id filter in query
        call_args = conn.fetch.call_args
        query = call_args[0][0]
        assert "workspace_id = $1" in query
        assert call_args[0][1] == workspace_id

    @pytest.mark.asyncio
    async def test_list_runs_with_status_filter(self, mock_pool):
        """Filters by status."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        repo = JobRunsRepository(pool)
        await repo.list_runs(status="failed")

        # Verify status filter in query
        call_args = conn.fetch.call_args
        query = call_args[0][0]
        assert "status = $1" in query
        assert call_args[0][1] == "failed"

    @pytest.mark.asyncio
    async def test_list_runs_pagination(self, mock_pool):
        """Applies limit and offset."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        repo = JobRunsRepository(pool)
        await repo.list_runs(limit=50, offset=100)

        # Verify pagination params
        call_args = conn.fetch.call_args
        params = call_args[0][1:]
        # Last two params should be limit and offset
        assert 50 in params
        assert 100 in params

    @pytest.mark.asyncio
    async def test_list_runs_detects_stale(self, mock_pool):
        """Query includes stale detection logic."""
        pool, conn = mock_pool
        conn.fetch.return_value = []

        repo = JobRunsRepository(pool)
        await repo.list_runs()

        # Verify stale detection in query
        call_args = conn.fetch.call_args
        query = call_args[0][0]
        assert "INTERVAL '1 hour'" in query
        assert "'stale'" in query

    @pytest.mark.asyncio
    async def test_get_run_found(self, mock_pool):
        """Returns run when found."""
        pool, conn = mock_pool
        run_id = uuid4()
        workspace_id = uuid4()
        conn.fetchrow.return_value = {
            "id": run_id,
            "job_name": "rollup_events",
            "workspace_id": workspace_id,
            "status": "completed",
            "metrics": {"rows": 10, "target_date": "2026-01-10"},
            "error": None,
            "display_status": "completed",
        }

        repo = JobRunsRepository(pool)
        run = await repo.get_run(run_id)

        assert run is not None
        assert run["id"] == run_id
        assert run["job_name"] == "rollup_events"
        assert run["metrics"]["rows"] == 10

    @pytest.mark.asyncio
    async def test_get_run_not_found(self, mock_pool):
        """Returns None when not found."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = None

        repo = JobRunsRepository(pool)
        run = await repo.get_run(uuid4())

        assert run is None

    @pytest.mark.asyncio
    async def test_count_runs_no_filters(self, mock_pool):
        """Counts all runs without filters."""
        pool, conn = mock_pool
        conn.fetchval.return_value = 42

        repo = JobRunsRepository(pool)
        count = await repo.count_runs()

        assert count == 42
        call_args = conn.fetchval.call_args
        query = call_args[0][0]
        assert "COUNT(*)" in query
        assert "WHERE TRUE" in query

    @pytest.mark.asyncio
    async def test_count_runs_with_filters(self, mock_pool):
        """Counts runs with filters."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        conn.fetchval.return_value = 5

        repo = JobRunsRepository(pool)
        count = await repo.count_runs(
            job_name="rollup_events",
            workspace_id=workspace_id,
            status="completed",
        )

        assert count == 5
        call_args = conn.fetchval.call_args
        query = call_args[0][0]
        assert "job_name = $1" in query
        assert "workspace_id = $2" in query
        assert "status = $3" in query

    @pytest.mark.asyncio
    async def test_count_runs_handles_none(self, mock_pool):
        """Returns 0 when fetchval returns None."""
        pool, conn = mock_pool
        conn.fetchval.return_value = None

        repo = JobRunsRepository(pool)
        count = await repo.count_runs()

        assert count == 0
