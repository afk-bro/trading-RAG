"""Repository for job queue operations."""

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import structlog

from app.jobs.models import Job
from app.jobs.types import JobType, JobStatus

logger = structlog.get_logger(__name__)


class JobRepository:
    """Repository for job queue operations."""

    def __init__(self, pool):
        self._pool = pool

    def _calculate_backoff(self, attempt: int) -> int:
        """Calculate retry backoff: min(300, 2^attempt * 5) + jitter."""
        base = min(300, (2**attempt) * 5)
        jitter = random.randint(0, min(10, base // 2))
        return base + jitter

    async def create(
        self,
        job_type: JobType,
        payload: dict[str, Any],
        workspace_id: Optional[UUID] = None,
        parent_job_id: Optional[UUID] = None,
        dedupe_key: Optional[str] = None,
        priority: int = 100,
        max_attempts: int = 3,
        run_after: Optional[datetime] = None,
    ) -> Job:
        """Create a new job in the queue."""
        query = """
            INSERT INTO jobs (type, payload, workspace_id, parent_job_id,
                             dedupe_key, priority, max_attempts, run_after)
            VALUES ($1, $2, $3, $4, $5, $6, $7, COALESCE($8, now()))
            ON CONFLICT (dedupe_key) WHERE dedupe_key IS NOT NULL
            DO UPDATE SET id = jobs.id  -- no-op, just return existing
            RETURNING *
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                job_type.value,
                payload,
                workspace_id,
                parent_job_id,
                dedupe_key,
                priority,
                max_attempts,
                run_after,
            )
        return self._row_to_job(row)

    async def claim(
        self, worker_id: str, job_types: Optional[list[JobType]] = None
    ) -> Optional[Job]:
        """Claim the next available job using FOR UPDATE SKIP LOCKED.

        Returns None if no jobs available.
        """
        type_filter = ""
        params: list[Any] = [worker_id]

        if job_types:
            type_values = [jt.value for jt in job_types]
            type_filter = "AND type = ANY($2)"
            params.append(type_values)

        query = f"""
            WITH cte AS (
                SELECT id FROM jobs
                WHERE status = 'pending' AND run_after <= now()
                {type_filter}
                ORDER BY priority, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            UPDATE jobs j SET
                status = 'running',
                locked_at = now(),
                locked_by = $1,
                started_at = now(),
                attempt = j.attempt + 1
            FROM cte
            WHERE j.id = cte.id
            RETURNING j.*
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row:
            logger.info(
                "job_claimed",
                job_id=str(row["id"]),
                job_type=row["type"],
                worker_id=worker_id,
            )
            return self._row_to_job(row)
        return None

    async def complete(
        self, job_id: UUID, result: Optional[dict[str, Any]] = None
    ) -> Job:
        """Mark a job as succeeded."""
        query = """
            UPDATE jobs SET
                status = 'succeeded',
                result = $2,
                completed_at = now()
            WHERE id = $1
            RETURNING *
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, job_id, result or {})
        logger.info("job_completed", job_id=str(job_id))
        return self._row_to_job(row)

    async def fail(self, job_id: UUID, error: str, should_retry: bool = True) -> Job:
        """Mark a job as failed, optionally scheduling retry."""
        async with self._pool.acquire() as conn:
            # Get current job state
            row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
            if not row:
                raise ValueError(f"Job {job_id} not found")

            attempt = row["attempt"]
            max_attempts = row["max_attempts"]

            if should_retry and attempt < max_attempts:
                # Schedule retry
                backoff = self._calculate_backoff(attempt)
                run_after = datetime.now(timezone.utc) + timedelta(seconds=backoff)
                query = """
                    UPDATE jobs SET
                        status = 'pending',
                        locked_at = NULL,
                        locked_by = NULL,
                        run_after = $2,
                        result = jsonb_build_object('last_error', $3)
                    WHERE id = $1
                    RETURNING *
                """
                row = await conn.fetchrow(query, job_id, run_after, error)
                logger.info(
                    "job_retry_scheduled",
                    job_id=str(job_id),
                    attempt=attempt,
                    backoff=backoff,
                )
            else:
                # Final failure
                query = """
                    UPDATE jobs SET
                        status = 'failed',
                        completed_at = now(),
                        result = jsonb_build_object('error', $2)
                    WHERE id = $1
                    RETURNING *
                """
                row = await conn.fetchrow(query, job_id, error)
                logger.warning("job_failed", job_id=str(job_id), error=error)

        return self._row_to_job(row)

    async def cancel(self, job_id: UUID) -> Job:
        """Cancel a job."""
        query = """
            UPDATE jobs SET
                status = 'canceled',
                completed_at = now()
            WHERE id = $1
            RETURNING *
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, job_id)
        logger.info("job_canceled", job_id=str(job_id))
        return self._row_to_job(row)

    async def get(self, job_id: UUID) -> Optional[Job]:
        """Get a job by ID."""
        query = "SELECT * FROM jobs WHERE id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, job_id)
        return self._row_to_job(row) if row else None

    async def list_by_parent(self, parent_job_id: UUID) -> list[Job]:
        """List all child jobs of a parent job."""
        query = """
            SELECT * FROM jobs
            WHERE parent_job_id = $1
            ORDER BY created_at
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, parent_job_id)
        return [self._row_to_job(row) for row in rows]

    async def reap_stale(self, stale_minutes: int = 30) -> int:
        """Reset stale running jobs (stuck workers) to pending for retry."""
        query = """
            UPDATE jobs SET
                status = 'pending',
                locked_at = NULL,
                locked_by = NULL
            WHERE status = 'running'
              AND locked_at < now() - ($1 || ' minutes')::interval
              AND attempt < max_attempts
            RETURNING id
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, str(stale_minutes))
        count = len(rows)
        if count > 0:
            logger.warning("stale_jobs_reaped", count=count)
        return count

    async def list_jobs(
        self,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        workspace_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Job], int]:
        """List jobs with filters and pagination.

        Args:
            status: Filter by status (pending, running, succeeded, failed, canceled)
            job_type: Filter by job type (data_sync, data_fetch, tune, wfo)
            workspace_id: Filter by workspace
            limit: Max results (1-100)
            offset: Pagination offset

        Returns:
            Tuple of (jobs list, total count)
        """
        # Build WHERE clause dynamically
        conditions = []
        params: list[Any] = []
        param_idx = 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if job_type:
            conditions.append(f"type = ${param_idx}")
            params.append(job_type)
            param_idx += 1

        if workspace_id:
            conditions.append(f"workspace_id = ${param_idx}")
            params.append(workspace_id)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # Query for jobs
        query = f"""
            SELECT * FROM jobs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        # Query for count
        count_query = f"""
            SELECT COUNT(*) as total FROM jobs
            {where_clause}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            # For count, exclude limit/offset params
            count_row = await conn.fetchrow(count_query, *params[:-2])

        jobs = [self._row_to_job(row) for row in rows]
        total = count_row["total"] if count_row else 0

        return jobs, total

    async def cancel_job_tree(self, job_id: UUID) -> tuple[Optional[Job], int]:
        """Cancel a job and all its children.

        Args:
            job_id: The parent job ID to cancel

        Returns:
            Tuple of (canceled job or None if not found, count of children canceled)
        """
        async with self._pool.acquire() as conn:
            # First check if job exists
            row = await conn.fetchrow("SELECT * FROM jobs WHERE id = $1", job_id)
            if not row:
                return None, 0

            # Cancel the parent job
            parent_query = """
                UPDATE jobs SET
                    status = 'canceled',
                    completed_at = now()
                WHERE id = $1
                RETURNING *
            """
            parent_row = await conn.fetchrow(parent_query, job_id)

            # Cancel all children that are not already in terminal status
            children_query = """
                UPDATE jobs SET
                    status = 'canceled',
                    completed_at = now()
                WHERE parent_job_id = $1
                  AND status NOT IN ('succeeded', 'failed', 'canceled')
                RETURNING id
            """
            children_rows = await conn.fetch(children_query, job_id)

        canceled_job = self._row_to_job(parent_row) if parent_row else None
        children_count = len(children_rows)

        logger.info(
            "job_tree_canceled",
            job_id=str(job_id),
            children_canceled=children_count,
        )

        return canceled_job, children_count

    def _row_to_job(self, row) -> Job:
        """Convert a database row to a Job model."""
        return Job(
            id=row["id"],
            type=JobType(row["type"]),
            status=JobStatus(row["status"]),
            payload=row["payload"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            run_after=row["run_after"],
            locked_at=row["locked_at"],
            locked_by=row["locked_by"],
            parent_job_id=row["parent_job_id"],
            workspace_id=row["workspace_id"],
            dedupe_key=row["dedupe_key"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            result=row["result"],
            priority=row["priority"],
        )
