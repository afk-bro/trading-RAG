"""Jobs admin endpoints (Retention Job Endpoints, Job Runs, Jobs Admin UI, Job Queue)."""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for jobs routes."""
    global _db_pool
    _db_pool = pool


def _json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)
    return obj


# ===========================================
# Retention Job Endpoints
# ===========================================


@router.post("/jobs/rollup-events")
async def run_rollup_job(
    workspace_id: UUID = Query(..., description="Workspace to scope the rollup"),
    target_date: Optional[date] = Query(
        None, description="Date to roll up (defaults to yesterday)"
    ),
    dry_run: bool = Query(False, description="Preview only, no changes"),
    _: bool = Depends(require_admin_token),
):
    """
    Run daily event rollup job.

    Aggregates trade_events into trade_event_rollups for the specified workspace.
    Defaults to yesterday if no date provided.
    Idempotent via ON CONFLICT - safe to run multiple times.

    Returns:
        200: Job completed successfully
        409: Job already running (lock not acquired)
        500: Job failed with error details
    """
    from app.repositories.event_rollups import EventRollupsRepository
    from app.services.jobs import JobRunner

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    repo = EventRollupsRepository()

    async def job_fn(conn, is_dry_run: bool, correlation_id: str) -> dict:
        """Job function for rollup."""
        if is_dry_run:
            preview = await repo.preview_daily_rollup(conn, workspace_id, target_date)
            return {
                "dry_run": True,
                "target_date": str(target_date),
                **preview,
            }
        else:
            count = await repo.run_daily_rollup(conn, workspace_id, target_date)
            return {
                "dry_run": False,
                "target_date": str(target_date),
                "rows_affected": count,
            }

    runner = JobRunner(_db_pool)
    try:
        result = await runner.run(
            job_name="rollup_events",
            workspace_id=workspace_id,
            dry_run=dry_run,
            triggered_by="admin_token",
            job_fn=job_fn,
        )

        if not result.lock_acquired:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=result.to_dict(),
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.to_dict(),
        )

    except Exception as e:
        logger.exception("rollup_events job failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "failed",
                "error": str(e),
                "workspace_id": str(workspace_id),
                "target_date": str(target_date),
            },
        )


@router.post("/jobs/cleanup-events")
async def run_cleanup_job(
    workspace_id: UUID = Query(..., description="Workspace to scope the cleanup"),
    dry_run: bool = Query(False, description="Preview only, no changes"),
    _: bool = Depends(require_admin_token),
):
    """
    Run event retention cleanup job.

    Deletes expired events based on severity tier for the specified workspace:
    - INFO/DEBUG: 30 days
    - WARN/ERROR: 90 days
    - Pinned events: Never deleted

    Returns:
        200: Job completed successfully
        409: Job already running (lock not acquired)
        500: Job failed with error details
    """
    from app.services.retention import RetentionService
    from app.services.jobs import JobRunner

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    service = RetentionService()

    async def job_fn(conn, is_dry_run: bool, correlation_id: str) -> dict:
        """Job function for cleanup."""
        if is_dry_run:
            preview = await service.preview_cleanup(conn, workspace_id)
            return {
                "dry_run": True,
                **preview,
            }
        else:
            result = await service.run_cleanup(conn, workspace_id)
            return {
                "dry_run": False,
                **result,
            }

    runner = JobRunner(_db_pool)
    try:
        result = await runner.run(
            job_name="cleanup_events",
            workspace_id=workspace_id,
            dry_run=dry_run,
            triggered_by="admin_token",
            job_fn=job_fn,
        )

        if not result.lock_acquired:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content=result.to_dict(),
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result.to_dict(),
        )

    except Exception as e:
        logger.exception("cleanup_events job failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "failed",
                "error": str(e),
                "workspace_id": str(workspace_id),
            },
        )


@router.post("/jobs/evaluate-alerts")
async def run_evaluate_alerts_job(
    workspace_id: UUID = Query(..., description="Workspace to evaluate alerts for"),
    dry_run: bool = Query(False, description="Preview only, no changes"),
    _: bool = Depends(require_admin_token),
):
    """
    Run alert evaluation job for workspace.

    Evaluates all enabled alert rules for the workspace, checking current
    regime drift and confidence metrics against configured thresholds.
    Creates or resolves alerts based on rule conditions.

    Returns:
        200: Job completed successfully with metrics
        409: Job already running (lock not acquired)
        500: Job failed with error details
    """
    from app.services.alerts.job import AlertEvaluatorJob

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    job = AlertEvaluatorJob(_db_pool)
    try:
        result = await job.run(workspace_id=workspace_id, dry_run=dry_run)

        if not result["lock_acquired"]:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "status": "already_running",
                    "workspace_id": str(workspace_id),
                    "metrics": result["metrics"],
                },
            )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": result["status"],
                "workspace_id": str(workspace_id),
                "dry_run": dry_run,
                "metrics": result["metrics"],
            },
        )

    except Exception as e:
        logger.exception("evaluate_alerts job failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "failed",
                "error": str(e),
                "workspace_id": str(workspace_id),
            },
        )


# ===========================================
# Job Runs List/Detail Endpoints
# ===========================================


@router.get("/jobs/runs")
async def list_job_runs(
    job_name: Optional[str] = Query(None, description="Filter by job name"),
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status (running, completed, failed)",
    ),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
):
    """
    List job runs with filters.

    Returns paginated list of job runs with filters for job name,
    workspace, and status. Includes display_status which marks
    running jobs older than 1 hour as 'stale'.
    """
    from app.repositories.job_runs import JobRunsRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    repo = JobRunsRepository(_db_pool)
    runs = await repo.list_runs(
        job_name=job_name,
        workspace_id=workspace_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    total = await repo.count_runs(
        job_name=job_name,
        workspace_id=workspace_id,
        status=status_filter,
    )

    # Convert to JSON-serializable format
    runs_serializable = [_json_serializable(r) for r in runs]

    return {
        "runs": runs_serializable,
        "count": len(runs),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/runs/{run_id}")
async def get_job_run(
    run_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Get full job run details.

    Returns complete job run record including full metrics JSON
    and error message if failed.
    """
    from app.repositories.job_runs import JobRunsRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    repo = JobRunsRepository(_db_pool)
    run = await repo.get_run(run_id)

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job run not found",
        )

    return _json_serializable(run)


# ===========================================
# Jobs Admin UI Pages
# ===========================================


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(
    request: Request,
    job_name: Optional[str] = Query(None, description="Filter by job name"),
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status",
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """Admin job runs page with filters and status badges."""
    from app.repositories.job_runs import JobRunsRepository

    if _db_pool is None:
        return templates.TemplateResponse(
            "jobs.html",
            {
                "request": request,
                "runs": [],
                "total": 0,
                "job_name": job_name,
                "workspace_id": workspace_id,
                "status_filter": status_filter,
                "limit": limit,
                "offset": offset,
                "job_names": ["rollup_events", "cleanup_events"],
                "error": "Database connection not available",
            },
        )

    repo = JobRunsRepository(_db_pool)
    runs = await repo.list_runs(
        job_name=job_name,
        workspace_id=workspace_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    total = await repo.count_runs(
        job_name=job_name,
        workspace_id=workspace_id,
        status=status_filter,
    )

    return templates.TemplateResponse(
        "jobs.html",
        {
            "request": request,
            "runs": runs,
            "total": total,
            "job_name": job_name,
            "workspace_id": str(workspace_id) if workspace_id else None,
            "status_filter": status_filter,
            "limit": limit,
            "offset": offset,
            "job_names": ["rollup_events", "cleanup_events"],
        },
    )


@router.get("/jobs/runs/{run_id}/detail", response_class=HTMLResponse)
async def job_run_detail_page(
    request: Request,
    run_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """Admin job run detail page."""
    from app.repositories.job_runs import JobRunsRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    repo = JobRunsRepository(_db_pool)
    run = await repo.get_run(run_id)

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job run not found",
        )

    return templates.TemplateResponse(
        "job_run_detail.html",
        {
            "request": request,
            "run": run,
        },
    )


# ===========================================
# Job Queue Management Endpoints
# ===========================================


def _job_to_dict(job) -> dict[str, Any]:
    """Convert a Job model to a JSON-serializable dict."""
    return {
        "id": str(job.id),
        "type": job.type.value,
        "status": job.status.value,
        "payload": job.payload,
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "run_after": job.run_after.isoformat() if job.run_after else None,
        "locked_at": job.locked_at.isoformat() if job.locked_at else None,
        "locked_by": job.locked_by,
        "parent_job_id": str(job.parent_job_id) if job.parent_job_id else None,
        "workspace_id": str(job.workspace_id) if job.workspace_id else None,
        "dedupe_key": job.dedupe_key,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "result": job.result,
        "priority": job.priority,
    }


def _job_event_to_dict(event) -> dict[str, Any]:
    """Convert a JobEvent model to a JSON-serializable dict."""
    return {
        "id": event.id,
        "job_id": str(event.job_id),
        "ts": event.ts.isoformat() if event.ts else None,
        "level": event.level,
        "message": event.message,
        "meta": event.meta,
    }


class TriggerSyncRequest(BaseModel):
    """Request body for triggering a data sync job."""

    exchange_id: Optional[str] = Field(
        default=None,
        description="Exchange ID to sync (optional, syncs all if not provided)",
    )
    mode: Literal["incremental", "full"] = Field(
        default="incremental",
        description="Sync mode: incremental (since last data) or full (history window)",
    )


@router.get("/jobs/queue")
async def list_jobs_queue(
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filter by status (pending, running, succeeded, failed, canceled)",
    ),
    type_filter: Optional[str] = Query(
        None,
        alias="type",
        description="Filter by job type (data_sync, data_fetch, tune, wfo)",
    ),
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
):
    """
    List jobs from the job queue with filters.

    Returns paginated list of queued jobs with filters for status,
    job type, and workspace.
    """
    from app.repositories.jobs import JobRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    repo = JobRepository(_db_pool)
    jobs, total = await repo.list_jobs(
        status=status_filter,
        job_type=type_filter,
        workspace_id=workspace_id,
        limit=limit,
        offset=offset,
    )

    return {
        "items": [_job_to_dict(job) for job in jobs],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/jobs/queue/{job_id}")
async def get_job_queue_detail(
    job_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Get full job details including events and children.

    Returns the job with its associated events (last 50) and
    any child jobs created by this job.
    """
    from app.repositories.jobs import JobRepository
    from app.repositories.job_events import JobEventsRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    job_repo = JobRepository(_db_pool)
    events_repo = JobEventsRepository(_db_pool)

    job = await job_repo.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    # Get events and children
    events = await events_repo.list_for_job(job_id, limit=50)
    children = await job_repo.list_by_parent(job_id)

    return {
        "job": _job_to_dict(job),
        "events": [_job_event_to_dict(e) for e in events],
        "children": [_job_to_dict(c) for c in children],
    }


@router.post("/jobs/queue/{job_id}/cancel")
async def cancel_job_queue(
    job_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Cancel a job and all its children.

    Cancels the specified job and any child jobs that are not
    already in a terminal state (succeeded, failed, canceled).
    """
    from app.repositories.jobs import JobRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    repo = JobRepository(_db_pool)
    job, children_count = await repo.cancel_job_tree(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    return {
        "job": _job_to_dict(job),
        "children_canceled": children_count,
    }


@router.post("/jobs/sync/trigger")
async def trigger_data_sync(
    request: TriggerSyncRequest = TriggerSyncRequest(),
    _: bool = Depends(require_admin_token),
):
    """
    Manually trigger a data sync job.

    Creates a DATA_SYNC job that will expand core symbols and
    enqueue DATA_FETCH jobs for each symbol/timeframe combination.
    """
    from app.repositories.jobs import JobRepository
    from app.jobs.types import JobType

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    repo = JobRepository(_db_pool)

    # Build payload
    payload: dict[str, Any] = {
        "mode": request.mode,
    }
    if request.exchange_id:
        payload["exchange_id"] = request.exchange_id

    job = await repo.create(
        job_type=JobType.DATA_SYNC,
        payload=payload,
    )

    logger.info(
        "data_sync_triggered",
        job_id=str(job.id),
        exchange_id=request.exchange_id or "all",
        mode=request.mode,
    )

    return {
        "job_id": str(job.id),
        "status": job.status.value,
    }
