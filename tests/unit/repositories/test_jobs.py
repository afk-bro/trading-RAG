"""Tests for job repository."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import UUID, uuid4

import pytest

from app.jobs.models import Job
from app.jobs.types import JobStatus, JobType
from app.repositories.jobs import JobRepository


class TestJobRepository:
    def test_repository_creation(self):
        mock_pool = MagicMock()
        repo = JobRepository(mock_pool)
        assert repo._pool == mock_pool

    def test_backoff_calculation(self):
        repo = JobRepository(MagicMock())
        # Formula: min(300, 2^attempt * 5) + jitter
        # jitter = random.randint(0, min(10, base // 2))

        # First retry (attempt=1): 2^1 * 5 = 10, jitter up to 5
        b1 = repo._calculate_backoff(1)
        assert 10 <= b1 <= 15  # 10 + jitter(0-5)

        # Second retry (attempt=2): 2^2 * 5 = 20, jitter up to 10
        b2 = repo._calculate_backoff(2)
        assert 20 <= b2 <= 30  # 20 + jitter(0-10)

        # Max backoff (attempt=10): capped at 300, jitter up to 10
        b_max = repo._calculate_backoff(10)
        assert 300 <= b_max <= 310  # 300 + jitter(0-10)


class TestCreateJob:
    """Test job creation."""

    @pytest.mark.asyncio
    async def test_create_job_minimal(self):
        """Test creating a job with minimal parameters."""
        job_id = uuid4()
        workspace_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.PENDING.value,
            "payload": {"key": "value"},
            "attempt": 0,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": None,
            "locked_by": None,
            "parent_job_id": None,
            "workspace_id": workspace_id,
            "dedupe_key": None,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "priority": 100,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.create(
            job_type=JobType.DATA_SYNC,
            payload={"key": "value"},
            workspace_id=workspace_id,
        )

        assert job.id == job_id
        assert job.type == JobType.DATA_SYNC
        assert job.status == JobStatus.PENDING
        assert job.workspace_id == workspace_id

    @pytest.mark.asyncio
    async def test_create_job_with_dedupe_key(self):
        """Test creating a job with dedupe key for idempotency."""
        job_id = uuid4()
        workspace_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.PENDING.value,
            "payload": {"key": "value"},
            "attempt": 0,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": None,
            "locked_by": None,
            "parent_job_id": None,
            "workspace_id": workspace_id,
            "dedupe_key": "sync:daily:2025-01-20",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "priority": 100,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.create(
            job_type=JobType.DATA_SYNC,
            payload={"key": "value"},
            workspace_id=workspace_id,
            dedupe_key="sync:daily:2025-01-20",
        )

        assert job.dedupe_key == "sync:daily:2025-01-20"


class TestClaimJob:
    """Test job claiming logic."""

    @pytest.mark.asyncio
    async def test_claim_job_success(self):
        """Test successfully claiming a job."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.RUNNING.value,
            "payload": {"key": "value"},
            "attempt": 1,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": now,
            "locked_by": "worker-1",
            "parent_job_id": None,
            "workspace_id": uuid4(),
            "dedupe_key": None,
            "created_at": now,
            "started_at": now,
            "completed_at": None,
            "result": None,
            "priority": 100,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.claim(worker_id="worker-1")

        assert job is not None
        assert job.status == JobStatus.RUNNING
        assert job.locked_by == "worker-1"
        assert job.attempt == 1

    @pytest.mark.asyncio
    async def test_claim_job_none_available(self):
        """Test claiming when no jobs are available."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.claim(worker_id="worker-1")

        assert job is None

    @pytest.mark.asyncio
    async def test_claim_job_with_type_filter(self):
        """Test claiming with job type filter."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": job_id,
            "type": JobType.TUNE.value,
            "status": JobStatus.RUNNING.value,
            "payload": {},
            "attempt": 1,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": now,
            "locked_by": "worker-1",
            "parent_job_id": None,
            "workspace_id": uuid4(),
            "dedupe_key": None,
            "created_at": now,
            "started_at": now,
            "completed_at": None,
            "result": None,
            "priority": 100,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.claim(worker_id="worker-1", job_types=[JobType.TUNE, JobType.WFO])

        assert job is not None
        assert job.type == JobType.TUNE


class TestCompleteJob:
    """Test job completion."""

    @pytest.mark.asyncio
    async def test_complete_job_success(self):
        """Test marking a job as succeeded."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.SUCCEEDED.value,
            "payload": {},
            "attempt": 1,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": now,
            "locked_by": "worker-1",
            "parent_job_id": None,
            "workspace_id": uuid4(),
            "dedupe_key": None,
            "created_at": now,
            "started_at": now,
            "completed_at": now,
            "result": {"records_synced": 100},
            "priority": 100,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.complete(job_id, result={"records_synced": 100})

        assert job.status == JobStatus.SUCCEEDED
        assert job.result == {"records_synced": 100}
        assert job.completed_at is not None


class TestFailJob:
    """Test job failure handling."""

    @pytest.mark.asyncio
    async def test_fail_job_with_retry(self):
        """Test failing a job that should retry."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        # Initial state row
        initial_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.RUNNING.value,
            "payload": {},
            "attempt": 1,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": now,
            "locked_by": "worker-1",
            "parent_job_id": None,
            "workspace_id": uuid4(),
            "dedupe_key": None,
            "created_at": now,
            "started_at": now,
            "completed_at": None,
            "result": None,
            "priority": 100,
        }

        # Updated row after retry scheduling
        retry_row = {
            **initial_row,
            "status": JobStatus.PENDING.value,
            "locked_at": None,
            "locked_by": None,
            "result": {"last_error": "Connection timeout"},
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=[initial_row, retry_row])

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.fail(job_id, error="Connection timeout", should_retry=True)

        assert job.status == JobStatus.PENDING
        assert job.locked_at is None
        assert job.result == {"last_error": "Connection timeout"}

    @pytest.mark.asyncio
    async def test_fail_job_final(self):
        """Test failing a job that has exhausted retries."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        # Job at max attempts
        initial_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.RUNNING.value,
            "payload": {},
            "attempt": 3,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": now,
            "locked_by": "worker-1",
            "parent_job_id": None,
            "workspace_id": uuid4(),
            "dedupe_key": None,
            "created_at": now,
            "started_at": now,
            "completed_at": None,
            "result": None,
            "priority": 100,
        }

        # Final failure state
        failed_row = {
            **initial_row,
            "status": JobStatus.FAILED.value,
            "completed_at": now,
            "result": {"error": "Max retries exceeded"},
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=[initial_row, failed_row])

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.fail(job_id, error="Max retries exceeded", should_retry=True)

        assert job.status == JobStatus.FAILED
        assert job.completed_at is not None
        assert job.result == {"error": "Max retries exceeded"}

    @pytest.mark.asyncio
    async def test_fail_job_not_found(self):
        """Test failing a job that doesn't exist."""
        job_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)

        with pytest.raises(ValueError, match=f"Job {job_id} not found"):
            await repo.fail(job_id, error="Some error")


class TestCancelJob:
    """Test job cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_job_success(self):
        """Test canceling a job."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.CANCELED.value,
            "payload": {},
            "attempt": 0,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": None,
            "locked_by": None,
            "parent_job_id": None,
            "workspace_id": uuid4(),
            "dedupe_key": None,
            "created_at": now,
            "started_at": None,
            "completed_at": now,
            "result": None,
            "priority": 100,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.cancel(job_id)

        assert job.status == JobStatus.CANCELED
        assert job.completed_at is not None


class TestGetJob:
    """Test retrieving jobs."""

    @pytest.mark.asyncio
    async def test_get_job_found(self):
        """Test getting a job that exists."""
        job_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_row = {
            "id": job_id,
            "type": JobType.DATA_SYNC.value,
            "status": JobStatus.PENDING.value,
            "payload": {},
            "attempt": 0,
            "max_attempts": 3,
            "run_after": now,
            "locked_at": None,
            "locked_by": None,
            "parent_job_id": None,
            "workspace_id": uuid4(),
            "dedupe_key": None,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "priority": 100,
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.get(job_id)

        assert job is not None
        assert job.id == job_id

    @pytest.mark.asyncio
    async def test_get_job_not_found(self):
        """Test getting a job that doesn't exist."""
        job_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        job = await repo.get(job_id)

        assert job is None


class TestListByParent:
    """Test listing child jobs."""

    @pytest.mark.asyncio
    async def test_list_by_parent(self):
        """Test listing child jobs of a parent."""
        parent_id = uuid4()
        child1_id = uuid4()
        child2_id = uuid4()
        now = datetime.now(timezone.utc)

        mock_rows = [
            {
                "id": child1_id,
                "type": JobType.DATA_FETCH.value,
                "status": JobStatus.SUCCEEDED.value,
                "payload": {},
                "attempt": 1,
                "max_attempts": 3,
                "run_after": now,
                "locked_at": None,
                "locked_by": None,
                "parent_job_id": parent_id,
                "workspace_id": uuid4(),
                "dedupe_key": None,
                "created_at": now,
                "started_at": now,
                "completed_at": now,
                "result": {},
                "priority": 100,
            },
            {
                "id": child2_id,
                "type": JobType.DATA_FETCH.value,
                "status": JobStatus.PENDING.value,
                "payload": {},
                "attempt": 0,
                "max_attempts": 3,
                "run_after": now,
                "locked_at": None,
                "locked_by": None,
                "parent_job_id": parent_id,
                "workspace_id": uuid4(),
                "dedupe_key": None,
                "created_at": now,
                "started_at": None,
                "completed_at": None,
                "result": None,
                "priority": 100,
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = JobRepository(mock_pool)
        jobs = await repo.list_by_parent(parent_id)

        assert len(jobs) == 2
        assert all(j.parent_job_id == parent_id for j in jobs)
