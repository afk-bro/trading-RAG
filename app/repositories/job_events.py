"""Repository for job event logging."""
from typing import Any, Optional
from uuid import UUID

import structlog

from app.jobs.models import JobEvent

logger = structlog.get_logger(__name__)


class JobEventsRepository:
    """Repository for job event operations."""

    def __init__(self, pool):
        self._pool = pool

    async def log(
        self,
        job_id: UUID,
        level: str,
        message: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> JobEvent:
        """Log an event for a job."""
        query = """
            INSERT INTO job_events (job_id, level, message, meta)
            VALUES ($1, $2, $3, $4)
            RETURNING *
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, job_id, level, message, meta)
        return self._row_to_event(row)

    async def info(self, job_id: UUID, message: str, **meta) -> JobEvent:
        """Log an info event."""
        return await self.log(job_id, "info", message, meta if meta else None)

    async def warn(self, job_id: UUID, message: str, **meta) -> JobEvent:
        """Log a warning event."""
        return await self.log(job_id, "warn", message, meta if meta else None)

    async def error(self, job_id: UUID, message: str, **meta) -> JobEvent:
        """Log an error event."""
        return await self.log(job_id, "error", message, meta if meta else None)

    async def list_for_job(
        self,
        job_id: UUID,
        level: Optional[str] = None,
        limit: int = 100,
    ) -> list[JobEvent]:
        """List events for a job, optionally filtered by level."""
        if level:
            query = """
                SELECT * FROM job_events
                WHERE job_id = $1 AND level = $2
                ORDER BY ts DESC
                LIMIT $3
            """
            params = [job_id, level, limit]
        else:
            query = """
                SELECT * FROM job_events
                WHERE job_id = $1
                ORDER BY ts DESC
                LIMIT $2
            """
            params = [job_id, limit]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [self._row_to_event(row) for row in rows]

    async def count_errors(self, job_id: UUID) -> int:
        """Count error events for a job."""
        query = """
            SELECT COUNT(*) as cnt FROM job_events
            WHERE job_id = $1 AND level = 'error'
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, job_id)
        return row["cnt"]

    def _row_to_event(self, row) -> JobEvent:
        """Convert a database row to a JobEvent model."""
        return JobEvent(
            id=row["id"],
            job_id=row["job_id"],
            ts=row["ts"],
            level=row["level"],
            message=row["message"],
            meta=row["meta"],
        )
