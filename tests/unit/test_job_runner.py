"""Tests for job runner with advisory lock."""

import pytest
from unittest.mock import AsyncMock, MagicMock
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

        # Lock acquired, insert returns run_id, then duration_ms query
        conn.fetchval.side_effect = [True, run_id, 150]
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
        assert result.run_id == run_id
        assert result.duration_ms == 150

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

        # Verify failed update + JOB_FAILED event
        execute_calls = [str(c) for c in conn.execute.call_args_list]
        assert any("status = 'failed'" in c for c in execute_calls)
        assert any("JOB_FAILED" in c for c in execute_calls)

    @pytest.mark.asyncio
    async def test_lock_always_released_on_success(self, mock_pool):
        """Lock is released even on success."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        run_id = uuid4()

        # Lock acquired, insert returns run_id, then duration_ms query
        conn.fetchval.side_effect = [True, run_id, 100]
        conn.execute.return_value = None

        async def job_fn(c, dry_run, correlation_id):
            return {}

        runner = JobRunner(pool)
        await runner.run(
            job_name="test_job",
            workspace_id=workspace_id,
            dry_run=False,
            triggered_by="admin_token",
            job_fn=job_fn,
        )

        # Verify advisory lock release was called
        execute_calls = [str(c) for c in conn.execute.call_args_list]
        assert any("pg_advisory_unlock" in c for c in execute_calls)

    @pytest.mark.asyncio
    async def test_lock_released_on_failure(self, mock_pool):
        """Lock is released on job failure."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        run_id = uuid4()

        conn.fetchval.side_effect = [True, run_id]
        conn.execute.return_value = None

        async def job_fn(c, dry_run, correlation_id):
            raise RuntimeError("Boom")

        runner = JobRunner(pool)

        with pytest.raises(RuntimeError):
            await runner.run(
                job_name="test_job",
                workspace_id=workspace_id,
                dry_run=False,
                triggered_by="admin_token",
                job_fn=job_fn,
            )

        # Verify advisory lock release was called
        execute_calls = [str(c) for c in conn.execute.call_args_list]
        assert any("pg_advisory_unlock" in c for c in execute_calls)

    @pytest.mark.asyncio
    async def test_dry_run_passed_to_job_fn(self, mock_pool):
        """dry_run flag is passed through to job function."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        run_id = uuid4()

        # Lock acquired, insert returns run_id, then duration_ms query
        conn.fetchval.side_effect = [True, run_id, 50]
        conn.execute.return_value = None

        captured_dry_run = None

        async def job_fn(c, dry_run, correlation_id):
            nonlocal captured_dry_run
            captured_dry_run = dry_run
            return {}

        runner = JobRunner(pool)
        await runner.run(
            job_name="test_job",
            workspace_id=workspace_id,
            dry_run=True,
            triggered_by="admin_token",
            job_fn=job_fn,
        )

        assert captured_dry_run is True

    @pytest.mark.asyncio
    async def test_correlation_id_passed_to_job_fn(self, mock_pool):
        """correlation_id is passed through to job function."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        run_id = uuid4()

        # Lock acquired, insert returns run_id, then duration_ms query
        conn.fetchval.side_effect = [True, run_id, 75]
        conn.execute.return_value = None

        captured_correlation_id = None

        async def job_fn(c, dry_run, correlation_id):
            nonlocal captured_correlation_id
            captured_correlation_id = correlation_id
            return {}

        runner = JobRunner(pool)
        result = await runner.run(
            job_name="test_job",
            workspace_id=workspace_id,
            dry_run=False,
            triggered_by="admin_token",
            job_fn=job_fn,
        )

        assert captured_correlation_id is not None
        assert captured_correlation_id == result.correlation_id


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_to_dict_completed(self):
        """Completed result serializes correctly."""
        run_id = uuid4()
        result = JobResult(
            run_id=run_id,
            lock_acquired=True,
            status="completed",
            duration_ms=150,
            metrics={"rows": 10},
            correlation_id="test-123",
        )

        d = result.to_dict()
        assert d["run_id"] == str(run_id)
        assert d["lock_acquired"] is True
        assert d["status"] == "completed"
        assert d["duration_ms"] == 150
        assert d["metrics"] == {"rows": 10}
        assert d["correlation_id"] == "test-123"
        assert d["error"] is None

    def test_to_dict_failed(self):
        """Failed result includes error."""
        run_id = uuid4()
        result = JobResult(
            run_id=run_id,
            lock_acquired=True,
            status="failed",
            duration_ms=50,
            metrics={},
            correlation_id="test-456",
            error="Something went wrong",
        )

        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Something went wrong"

    def test_to_dict_already_running(self):
        """Already running result has no run_id."""
        result = JobResult(
            run_id=None,
            lock_acquired=False,
            status="already_running",
            duration_ms=0,
            metrics={},
            correlation_id="test-789",
        )

        d = result.to_dict()
        assert d["run_id"] is None
        assert d["lock_acquired"] is False
        assert d["status"] == "already_running"
