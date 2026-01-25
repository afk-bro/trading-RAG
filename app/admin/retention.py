"""Admin endpoints for retention job management.

Provides fallback trigger mechanism when pg_cron is unavailable,
plus status/log viewing for all deployments.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.admin.utils import require_db_pool
from app.deps.security import require_admin_token

router = APIRouter(prefix="/retention", tags=["admin-retention"])
logger = structlog.get_logger(__name__)

# ===========================================
# Module Constants
# ===========================================

# Global connection pool (set during app startup via set_db_pool)
_db_pool = None

# Default retention periods (days)
DEFAULT_TRADE_EVENTS_DAYS = 90
DEFAULT_JOB_RUNS_DAYS = 30
DEFAULT_MATCH_RUNS_DAYS = 180
IDEMPOTENCY_KEYS_EXPIRY_DAYS = 7  # Fixed expiry, not configurable

# Query parameter limits
MIN_CUTOFF_DAYS = 1
MAX_CUTOFF_DAYS_STANDARD = 365
MAX_CUTOFF_DAYS_EXTENDED = 730  # For match_runs

# Batch size limits
DEFAULT_BATCH_SIZE = 10000
MIN_BATCH_SIZE = 100
MAX_BATCH_SIZE = 100000


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


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
# Helper Functions
# ===========================================


def _row_to_log_entry(row: Any) -> RetentionLogEntry:
    """Convert a database row to a RetentionLogEntry model."""
    return RetentionLogEntry(
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


async def _execute_prune(
    job_name: str,
    sql_function: str,
    cutoff_days: int,
    batch_size: int,
    dry_run: bool,
    use_interval: bool = True,
) -> PruneResult:
    """Execute a retention prune operation.

    Args:
        job_name: Name of the retention job for logging/response
        sql_function: Name of the SQL function to call
        cutoff_days: Days to retain (converted to interval if use_interval=True)
        batch_size: Number of rows to process per batch
        dry_run: If True, count only without deleting
        use_interval: If True, pass cutoff_days as interval; if False, omit interval param

    Returns:
        PruneResult with operation details
    """
    pool = require_db_pool(_db_pool)
    start_time = datetime.utcnow()

    async with pool.acquire() as conn:
        if use_interval:
            result = await conn.fetchrow(
                f"""
                SELECT deleted_count, job_log_id
                FROM {sql_function}($1::interval, $2, $3)
                """,
                f"{cutoff_days} days",
                batch_size,
                dry_run,
            )
        else:
            # For idempotency_keys which doesn't take a cutoff interval
            result = await conn.fetchrow(
                f"""
                SELECT deleted_count, job_log_id
                FROM {sql_function}($1, $2)
                """,
                batch_size,
                dry_run,
            )

    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        f"retention_{sql_function}",
        deleted_count=result["deleted_count"],
        cutoff_days=cutoff_days,
        dry_run=dry_run,
        duration_ms=duration_ms,
    )

    return PruneResult(
        job_name=job_name,
        deleted_count=result["deleted_count"],
        job_log_id=result["job_log_id"],
        dry_run=dry_run,
        cutoff_days=cutoff_days,
        duration_ms=duration_ms,
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
    cutoff_days: int = Query(
        default=DEFAULT_TRADE_EVENTS_DAYS,
        ge=MIN_CUTOFF_DAYS,
        le=MAX_CUTOFF_DAYS_STANDARD,
        description="Days to retain",
    ),
    batch_size: int = Query(
        default=DEFAULT_BATCH_SIZE,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
        description="Rows per batch",
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
    return await _execute_prune(
        job_name="trade_events",
        sql_function="retention_prune_trade_events",
        cutoff_days=cutoff_days,
        batch_size=batch_size,
        dry_run=dry_run,
    )


@router.post(
    "/prune/job-runs",
    response_model=PruneResult,
    dependencies=[Depends(require_admin_token)],
    summary="Prune old job runs",
)
async def prune_job_runs(
    cutoff_days: int = Query(
        default=DEFAULT_JOB_RUNS_DAYS,
        ge=MIN_CUTOFF_DAYS,
        le=MAX_CUTOFF_DAYS_STANDARD,
        description="Days to retain",
    ),
    batch_size: int = Query(
        default=DEFAULT_BATCH_SIZE,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
        description="Rows per batch",
    ),
    dry_run: bool = Query(
        default=True, description="If true, count only without deleting"
    ),
):
    """
    Prune old job_runs records older than cutoff.

    **Default: dry_run=true** - Always preview before actual deletion.
    """
    return await _execute_prune(
        job_name="job_runs",
        sql_function="retention_prune_job_runs",
        cutoff_days=cutoff_days,
        batch_size=batch_size,
        dry_run=dry_run,
    )


@router.post(
    "/prune/match-runs",
    response_model=PruneResult,
    dependencies=[Depends(require_admin_token)],
    summary="Prune old resolved match runs",
)
async def prune_match_runs(
    cutoff_days: int = Query(
        default=DEFAULT_MATCH_RUNS_DAYS,
        ge=MIN_CUTOFF_DAYS,
        le=MAX_CUTOFF_DAYS_EXTENDED,
        description="Days to retain",
    ),
    batch_size: int = Query(
        default=DEFAULT_BATCH_SIZE,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
        description="Rows per batch",
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
    return await _execute_prune(
        job_name="match_runs",
        sql_function="retention_prune_match_runs",
        cutoff_days=cutoff_days,
        batch_size=batch_size,
        dry_run=dry_run,
    )


@router.post(
    "/prune/idempotency-keys",
    response_model=PruneResult,
    dependencies=[Depends(require_admin_token)],
    summary="Prune expired idempotency keys",
)
async def prune_idempotency_keys(
    batch_size: int = Query(
        default=DEFAULT_BATCH_SIZE,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
        description="Rows per batch",
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
    return await _execute_prune(
        job_name="idempotency_keys",
        sql_function="retention_prune_idempotency_keys",
        cutoff_days=IDEMPOTENCY_KEYS_EXPIRY_DAYS,
        batch_size=batch_size,
        dry_run=dry_run,
        use_interval=False,
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
    pool = require_db_pool(_db_pool)

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

    items = [_row_to_log_entry(row) for row in rows]

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
    pool = require_db_pool(_db_pool)

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

        for jn in job_names:
            row = await conn.fetchrow(
                """
                SELECT * FROM retention_job_log
                WHERE job_name = $1
                ORDER BY started_at DESC
                LIMIT 1
                """,
                jn,
            )
            last_runs[jn] = _row_to_log_entry(row) if row else None

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
