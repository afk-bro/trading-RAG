"""Job runner with advisory lock and tracking."""

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID, uuid4

import structlog

from app.services.jobs.locks import job_lock_key

logger = structlog.get_logger(__name__)


@dataclass
class JobResult:
    """Result of a job execution attempt."""

    run_id: Optional[UUID]
    lock_acquired: bool
    status: str  # "completed", "failed", "already_running"
    duration_ms: int
    metrics: dict[str, Any]
    correlation_id: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "run_id": str(self.run_id) if self.run_id else None,
            "lock_acquired": self.lock_acquired,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "metrics": self.metrics,
            "correlation_id": self.correlation_id,
            "error": self.error,
        }


# Type alias for job function signature
# job_fn(conn, dry_run, correlation_id) -> dict[str, Any]
JobFn = Callable[[Any, bool, str], Awaitable[dict[str, Any]]]


class JobRunner:
    """
    Runs jobs with advisory lock protection and tracking.

    Features:
    - Acquires advisory lock (non-blocking) to prevent concurrent runs
    - Creates job_runs tracking row before execution
    - Executes job on the SAME connection that holds the lock
    - Updates row to completed/failed on finish
    - On failure: writes JOB_FAILED event to trade_events
    - Always releases lock in finally block
    """

    def __init__(self, pool):
        """
        Initialize with asyncpg pool.

        Args:
            pool: asyncpg connection pool
        """
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
        Execute a job with advisory lock protection.

        Args:
            job_name: Name of the job (e.g., "rollup_events")
            workspace_id: Workspace scope for locking
            dry_run: If True, job should not make changes
            triggered_by: Who triggered the job (e.g., "cron", "admin_token")
            job_fn: Async function to execute: (conn, dry_run, correlation_id) -> metrics

        Returns:
            JobResult with execution details

        Raises:
            Original exception from job_fn on failure (after cleanup)
        """
        correlation_id = f"job-{job_name}-{uuid4().hex[:8]}"
        lock_key = job_lock_key(job_name, workspace_id)
        run_id: Optional[UUID] = None

        logger.info(
            "Job run starting",
            job_name=job_name,
            workspace_id=str(workspace_id),
            dry_run=dry_run,
            correlation_id=correlation_id,
        )

        async with self.pool.acquire() as conn:
            # Try to acquire advisory lock (non-blocking)
            lock_acquired = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", lock_key
            )

            if not lock_acquired:
                logger.info(
                    "Job skipped - already running",
                    job_name=job_name,
                    workspace_id=str(workspace_id),
                    correlation_id=correlation_id,
                )
                return JobResult(
                    run_id=None,
                    lock_acquired=False,
                    status="already_running",
                    duration_ms=0,
                    metrics={},
                    correlation_id=correlation_id,
                )

            try:
                # Create job_runs tracking row
                run_id = await conn.fetchval(
                    """
                    INSERT INTO job_runs (
                        job_name, workspace_id, status, triggered_by,
                        started_at, correlation_id
                    ) VALUES ($1, $2, 'running', $3, NOW(), $4)
                    RETURNING id
                    """,
                    job_name,
                    workspace_id,
                    triggered_by,
                    correlation_id,
                )

                logger.info(
                    "Job run created",
                    run_id=str(run_id),
                    job_name=job_name,
                    correlation_id=correlation_id,
                )

                # Execute job on the same connection that holds the lock
                metrics = await job_fn(conn, dry_run, correlation_id)

                # Update to completed with duration
                await conn.execute(
                    """
                    UPDATE job_runs
                    SET status = 'completed',
                        finished_at = NOW(),
                        duration_ms = (EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)::int,
                        metrics = $2
                    WHERE id = $1
                    """,
                    run_id,
                    json.dumps(metrics),
                )

                # Get actual duration for result
                duration_ms = await conn.fetchval(
                    "SELECT duration_ms FROM job_runs WHERE id = $1", run_id
                )

                logger.info(
                    "Job completed",
                    run_id=str(run_id),
                    job_name=job_name,
                    duration_ms=duration_ms,
                    metrics=metrics,
                    correlation_id=correlation_id,
                )

                return JobResult(
                    run_id=run_id,
                    lock_acquired=True,
                    status="completed",
                    duration_ms=duration_ms or 0,
                    metrics=metrics,
                    correlation_id=correlation_id,
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "Job failed",
                    run_id=str(run_id) if run_id else None,
                    job_name=job_name,
                    error=error_msg,
                    correlation_id=correlation_id,
                )

                if run_id:
                    # Update job_runs to failed
                    await conn.execute(
                        """
                        UPDATE job_runs
                        SET status = 'failed',
                            finished_at = NOW(),
                            duration_ms = (EXTRACT(EPOCH FROM (NOW() - started_at)) * 1000)::int,
                            error_message = $2
                        WHERE id = $1
                        """,
                        run_id,
                        error_msg,
                    )

                    # Insert JOB_FAILED event
                    await conn.execute(
                        """
                        INSERT INTO trade_events (
                            id, correlation_id, workspace_id, event_type,
                            created_at, payload, metadata
                        ) VALUES (
                            $1, $2, $3, 'JOB_FAILED', NOW(),
                            $4::jsonb, $5::jsonb
                        )
                        """,
                        uuid4(),
                        correlation_id,
                        workspace_id,
                        json.dumps(
                            {
                                "job_name": job_name,
                                "run_id": str(run_id),
                                "error": error_msg,
                            }
                        ),
                        json.dumps({"severity": "ERROR", "triggered_by": triggered_by}),
                    )

                # Re-raise original exception
                raise

            finally:
                # Always release lock
                await conn.execute("SELECT pg_advisory_unlock($1)", lock_key)
                logger.debug(
                    "Advisory lock released",
                    job_name=job_name,
                    lock_key=lock_key,
                    correlation_id=correlation_id,
                )
