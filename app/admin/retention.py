"""Admin endpoints for retention job management.

Provides fallback trigger mechanism when pg_cron is unavailable,
plus status/log viewing for all deployments.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token

router = APIRouter(prefix="/retention", tags=["admin-retention"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup via set_db_pool)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_pool():
    """Get the database pool."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return _db_pool


# ===========================================
# Response Models
# ===========================================


class PruneResult(BaseModel):
    """Result from a retention prune operation."""

    job_name: str = Field(..., description="Name of the retention job")
    deleted_count: int = Field(
        ..., description="Number of rows deleted (or would be deleted if dry_run)"
    )
    job_log_id: UUID = Field(..., description="ID of the job log entry")
    dry_run: bool = Field(..., description="Whether this was a dry run")
    cutoff_days: int = Field(..., description="Cutoff period in days")
    duration_ms: Optional[int] = Field(
        None, description="Execution duration in milliseconds"
    )


class RetentionLogEntry(BaseModel):
    """Entry from the retention job log."""

    id: UUID
    job_name: str
    started_at: datetime
    finished_at: Optional[datetime]
    rows_deleted: int
    cutoff_ts: datetime
    dry_run: bool
    ok: bool
    error: Optional[str]


class RetentionLogsResponse(BaseModel):
    """Response for listing retention logs."""

    items: list[RetentionLogEntry]
    total: int
    limit: int
    offset: int


class RetentionStatusResponse(BaseModel):
    """Overall retention system status."""

    pg_cron_available: bool = Field(
        ..., description="Whether pg_cron extension is installed"
    )
    scheduled_jobs: list[dict] = Field(
        default_factory=list, description="List of scheduled pg_cron jobs"
    )
    last_runs: dict[str, Optional[RetentionLogEntry]] = Field(
        default_factory=dict, description="Most recent run per job type"
    )
    table_estimates: dict[str, int] = Field(
        default_factory=dict, description="Estimated row counts for retention tables"
    )


# ===========================================
# Prune Endpoints
# ===========================================


@router.post(
    "/prune/trade-events",
    response_model=PruneResult,
    dependencies=[Depends(require_admin_token)],
    summary="Prune old trade events",
)
async def prune_trade_events(
    cutoff_days: int = Query(default=90, ge=1, le=365, description="Days to retain"),
    batch_size: int = Query(
        default=10000, ge=100, le=100000, description="Rows per batch"
    ),
    dry_run: bool = Query(
        default=True, description="If true, count only without deleting"
    ),
):
    """
    Prune old RUN_* trade events older than cutoff.

    ORDER_* events are never deleted (source of truth).
    Pinned events are never deleted.

    **Default: dry_run=true** - Always preview before actual deletion.
    """
    pool = _get_pool()
    start_time = datetime.utcnow()

    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT deleted_count, job_log_id
            FROM retention_prune_trade_events($1::interval, $2, $3)
            """,
            f"{cutoff_days} days",
            batch_size,
            dry_run,
        )

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        "retention_prune_trade_events",
        deleted_count=result["deleted_count"],
        cutoff_days=cutoff_days,
        dry_run=dry_run,
        duration_ms=duration_ms,
    )

    return PruneResult(
        job_name="trade_events",
        deleted_count=result["deleted_count"],
        job_log_id=result["job_log_id"],
        dry_run=dry_run,
        cutoff_days=cutoff_days,
        duration_ms=duration_ms,
    )


@router.post(
    "/prune/job-runs",
    response_model=PruneResult,
    dependencies=[Depends(require_admin_token)],
    summary="Prune old job runs",
)
async def prune_job_runs(
    cutoff_days: int = Query(default=30, ge=1, le=365, description="Days to retain"),
    batch_size: int = Query(
        default=10000, ge=100, le=100000, description="Rows per batch"
    ),
    dry_run: bool = Query(
        default=True, description="If true, count only without deleting"
    ),
):
    """
    Prune old job_runs records older than cutoff.

    **Default: dry_run=true** - Always preview before actual deletion.
    """
    pool = _get_pool()
    start_time = datetime.utcnow()

    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT deleted_count, job_log_id
            FROM retention_prune_job_runs($1::interval, $2, $3)
            """,
            f"{cutoff_days} days",
            batch_size,
            dry_run,
        )

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        "retention_prune_job_runs",
        deleted_count=result["deleted_count"],
        cutoff_days=cutoff_days,
        dry_run=dry_run,
        duration_ms=duration_ms,
    )

    return PruneResult(
        job_name="job_runs",
        deleted_count=result["deleted_count"],
        job_log_id=result["job_log_id"],
        dry_run=dry_run,
        cutoff_days=cutoff_days,
        duration_ms=duration_ms,
    )


@router.post(
    "/prune/match-runs",
    response_model=PruneResult,
    dependencies=[Depends(require_admin_token)],
    summary="Prune old resolved match runs",
)
async def prune_match_runs(
    cutoff_days: int = Query(default=180, ge=1, le=730, description="Days to retain"),
    batch_size: int = Query(
        default=10000, ge=100, le=100000, description="Rows per batch"
    ),
    dry_run: bool = Query(
        default=True, description="If true, count only without deleting"
    ),
):
    """
    Prune old RESOLVED match_runs records older than cutoff.

    Only resolved items are deleted. Open/acknowledged items retained for triage.

    **Default: dry_run=true** - Always preview before actual deletion.
    """
    pool = _get_pool()
    start_time = datetime.utcnow()

    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT deleted_count, job_log_id
            FROM retention_prune_match_runs($1::interval, $2, $3)
            """,
            f"{cutoff_days} days",
            batch_size,
            dry_run,
        )

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        "retention_prune_match_runs",
        deleted_count=result["deleted_count"],
        cutoff_days=cutoff_days,
        dry_run=dry_run,
        duration_ms=duration_ms,
    )

    return PruneResult(
        job_name="match_runs",
        deleted_count=result["deleted_count"],
        job_log_id=result["job_log_id"],
        dry_run=dry_run,
        cutoff_days=cutoff_days,
        duration_ms=duration_ms,
    )


@router.post(
    "/prune/idempotency-keys",
    response_model=PruneResult,
    dependencies=[Depends(require_admin_token)],
    summary="Prune expired idempotency keys",
)
async def prune_idempotency_keys(
    batch_size: int = Query(
        default=10000, ge=100, le=100000, description="Rows per batch"
    ),
    dry_run: bool = Query(
        default=True, description="If true, count only without deleting"
    ),
):
    """
    Prune expired idempotency keys (keys with expires_at < NOW()).

    Keys have a 7-day default expiry set at creation time.

    **Default: dry_run=true** - Always preview before actual deletion.
    """
    pool = _get_pool()
    start_time = datetime.utcnow()

    async with pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT deleted_count, job_log_id
            FROM retention_prune_idempotency_keys($1, $2)
            """,
            batch_size,
            dry_run,
        )

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        "retention_prune_idempotency_keys",
        deleted_count=result["deleted_count"],
        dry_run=dry_run,
        duration_ms=duration_ms,
    )

    return PruneResult(
        job_name="idempotency_keys",
        deleted_count=result["deleted_count"],
        job_log_id=result["job_log_id"],
        dry_run=dry_run,
        cutoff_days=7,  # Fixed 7-day expiry
        duration_ms=duration_ms,
    )


# ===========================================
# Status & Logs Endpoints
# ===========================================


@router.get(
    "/logs",
    response_model=RetentionLogsResponse,
    dependencies=[Depends(require_admin_token)],
    summary="List retention job logs",
)
async def list_retention_logs(
    job_name: Optional[str] = Query(None, description="Filter by job name"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List retention job execution logs with optional filtering."""
    pool = _get_pool()

    async with pool.acquire() as conn:
        # Build query with optional filter
        if job_name:
            rows = await conn.fetch(
                """
                SELECT * FROM retention_job_log
                WHERE job_name = $1
                ORDER BY started_at DESC
                LIMIT $2 OFFSET $3
                """,
                job_name,
                limit,
                offset,
            )
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM retention_job_log WHERE job_name = $1",
                job_name,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT * FROM retention_job_log
                ORDER BY started_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM retention_job_log")

    items = [
        RetentionLogEntry(
            id=row["id"],
            job_name=row["job_name"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            rows_deleted=row["rows_deleted"],
            cutoff_ts=row["cutoff_ts"],
            dry_run=row["dry_run"],
            ok=row["ok"],
            error=row["error"],
        )
        for row in rows
    ]

    return RetentionLogsResponse(
        items=items,
        total=total or 0,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/status",
    response_model=RetentionStatusResponse,
    dependencies=[Depends(require_admin_token)],
    summary="Get retention system status",
)
async def get_retention_status():
    """
    Get overall retention system status including:
    - pg_cron availability
    - Scheduled jobs (if pg_cron available)
    - Most recent run per job type
    - Estimated row counts for retention tables
    """
    pool = _get_pool()

    async with pool.acquire() as conn:
        # Check pg_cron availability
        pg_cron_available = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_cron')"
        )

        # Get scheduled jobs if pg_cron available
        scheduled_jobs = []
        if pg_cron_available:
            try:
                jobs = await conn.fetch(
                    """
                    SELECT jobid, jobname, schedule, command, active
                    FROM cron.job
                    WHERE jobname LIKE 'retention_prune_%'
                    ORDER BY jobname
                    """
                )
                scheduled_jobs = [
                    {
                        "jobid": row["jobid"],
                        "jobname": row["jobname"],
                        "schedule": row["schedule"],
                        "active": row["active"],
                    }
                    for row in jobs
                ]
            except Exception:
                # cron schema might not be accessible
                pass

        # Get last run per job type
        job_names = ["trade_events", "job_runs", "match_runs", "idempotency_keys"]
        last_runs = {}

        for job_name in job_names:
            row = await conn.fetchrow(
                """
                SELECT * FROM retention_job_log
                WHERE job_name = $1
                ORDER BY started_at DESC
                LIMIT 1
                """,
                job_name,
            )
            if row:
                last_runs[job_name] = RetentionLogEntry(
                    id=row["id"],
                    job_name=row["job_name"],
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    rows_deleted=row["rows_deleted"],
                    cutoff_ts=row["cutoff_ts"],
                    dry_run=row["dry_run"],
                    ok=row["ok"],
                    error=row["error"],
                )
            else:
                last_runs[job_name] = None

        # Get estimated row counts (use pg_class for fast estimates)
        table_estimates = {}
        tables = [
            ("trade_events", "trade_events"),
            ("job_runs", "job_runs"),
            ("match_runs", "match_runs"),
            ("idempotency_keys", "idempotency_keys"),
        ]

        for name, table in tables:
            try:
                count = await conn.fetchval(
                    """
                    SELECT reltuples::bigint
                    FROM pg_class
                    WHERE relname = $1
                    """,
                    table,
                )
                table_estimates[name] = count or 0
            except Exception:
                table_estimates[name] = -1  # Error getting estimate

    return RetentionStatusResponse(
        pg_cron_available=pg_cron_available,
        scheduled_jobs=scheduled_jobs,
        last_runs=last_runs,
        table_estimates=table_estimates,
    )
