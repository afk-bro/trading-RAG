"""Repository for job runs tracking."""

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
        """
        List job runs with filters and pagination.

        Args:
            job_name: Filter by job name
            workspace_id: Filter by workspace
            status: Filter by status (running, completed, failed)
            limit: Max results
            offset: Pagination offset

        Returns:
            List of job run records with display_status
        """
        conditions = []
        params: list[Any] = []
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
        """
        Get full job run details.

        Args:
            run_id: Job run UUID

        Returns:
            Full job run record or None if not found
        """
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

    async def count_runs(
        self,
        job_name: Optional[str] = None,
        workspace_id: Optional[UUID] = None,
        status: Optional[str] = None,
    ) -> int:
        """
        Count job runs matching filters.

        Args:
            job_name: Filter by job name
            workspace_id: Filter by workspace
            status: Filter by status

        Returns:
            Total count of matching runs
        """
        conditions = []
        params: list[Any] = []
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
            SELECT COUNT(*) FROM job_runs WHERE {where_clause}
        """

        async with self.pool.acquire() as conn:
            count = await conn.fetchval(query, *params)

        return count or 0
