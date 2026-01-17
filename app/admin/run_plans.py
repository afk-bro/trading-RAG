"""Run Plans admin endpoints (Test Generator / Orchestrator)."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for run_plans routes."""
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


def _get_run_plans_repo():
    """Get RunPlansRepository instance."""
    from app.repositories.run_plans import RunPlansRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return RunPlansRepository(_db_pool)


# ==============================================================================
# Run Plans Admin UI (Test Generator / Orchestrator)
# ==============================================================================


@router.get("/testing/run-plans", response_class=HTMLResponse)
async def admin_run_plans_list(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List run plans with event-driven summaries.

    Run plans are reconstructed from RUN_STARTED and RUN_COMPLETED events
    in the trade_events journal (v0 has no dedicated RunPlan table).
    """
    # If no workspace specified, get first available
    if not workspace_id and _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT id FROM workspaces LIMIT 1")
                if row:
                    workspace_id = row["id"]
        except Exception as e:
            logger.warning("Could not fetch default workspace", error=str(e))

    if not workspace_id:
        return templates.TemplateResponse(
            "run_plans_list.html",
            {
                "request": request,
                "run_plans": [],
                "total": 0,
                "workspace_id": None,
                "status_filter": status_filter,
                "hours": hours,
                "limit": limit,
                "offset": offset,
                "has_prev": False,
                "has_next": False,
                "prev_offset": 0,
                "next_offset": 0,
                "error": "No workspace found. Create a workspace first.",
            },
        )

    since = datetime.utcnow() - timedelta(hours=hours)

    # Query trade_events for run plan events
    # Group by correlation_id (which is run_plan_id)
    # Join RUN_STARTED and RUN_COMPLETED events
    query = """
        WITH run_events AS (
            SELECT
                correlation_id as run_plan_id,
                payload->>'run_event_type' as run_event_type,
                payload,
                created_at
            FROM trade_events
            WHERE workspace_id = $1
              AND created_at >= $2
              AND payload->>'run_event_type' IN ('RUN_STARTED', 'RUN_COMPLETED')
        ),
        started AS (
            SELECT
                run_plan_id,
                payload,
                created_at as started_at
            FROM run_events
            WHERE run_event_type = 'RUN_STARTED'
        ),
        completed AS (
            SELECT
                run_plan_id,
                payload,
                created_at as completed_at
            FROM run_events
            WHERE run_event_type = 'RUN_COMPLETED'
        ),
        run_plans AS (
            SELECT
                s.run_plan_id,
                s.started_at,
                c.completed_at,
                CASE
                    WHEN c.run_plan_id IS NOT NULL THEN 'completed'
                    ELSE 'running'
                END as status,
                (s.payload->>'n_variants')::int as n_variants,
                s.payload->>'objective' as objective,
                s.payload->>'dataset_ref' as dataset_ref,
                (s.payload->>'bar_count')::int as bar_count,
                (c.payload->>'n_successful')::int as n_successful,
                (c.payload->>'n_failed')::int as n_failed,
                c.payload->'summary' as summary
            FROM started s
            LEFT JOIN completed c ON s.run_plan_id = c.run_plan_id
        )
        SELECT * FROM run_plans
        WHERE ($3::text IS NULL OR status = $3)
        ORDER BY started_at DESC
        LIMIT $4 OFFSET $5
    """

    count_query = """
        WITH run_events AS (
            SELECT
                correlation_id as run_plan_id,
                payload->>'run_event_type' as run_event_type
            FROM trade_events
            WHERE workspace_id = $1
              AND created_at >= $2
              AND payload->>'run_event_type' IN ('RUN_STARTED', 'RUN_COMPLETED')
        ),
        started AS (
            SELECT run_plan_id FROM run_events WHERE run_event_type = 'RUN_STARTED'
        ),
        completed AS (
            SELECT run_plan_id FROM run_events WHERE run_event_type = 'RUN_COMPLETED'
        ),
        run_plans AS (
            SELECT
                s.run_plan_id,
                CASE WHEN c.run_plan_id IS NOT NULL THEN 'completed' ELSE 'running' END as status
            FROM started s
            LEFT JOIN completed c ON s.run_plan_id = c.run_plan_id
        )
        SELECT COUNT(*) as total FROM run_plans
        WHERE ($3::text IS NULL OR status = $3)
    """

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            query, workspace_id, since, status_filter, limit, offset
        )
        count_row = await conn.fetchrow(count_query, workspace_id, since, status_filter)
        total = count_row["total"] if count_row else 0

    run_plans = []
    for row in rows:
        run_plans.append(
            {
                "run_plan_id": row["run_plan_id"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "status": row["status"],
                "n_variants": row["n_variants"],
                "objective": row["objective"],
                "dataset_ref": row["dataset_ref"],
                "bar_count": row["bar_count"],
                "n_successful": row["n_successful"],
                "n_failed": row["n_failed"],
                "summary": row["summary"],
            }
        )

    return templates.TemplateResponse(
        "run_plans_list.html",
        {
            "request": request,
            "run_plans": run_plans,
            "total": total,
            "workspace_id": str(workspace_id),
            "status_filter": status_filter or "",
            "hours": hours,
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


@router.get("/testing/run-plans/{run_plan_id}", response_class=HTMLResponse)
async def admin_run_plan_detail(
    request: Request,
    run_plan_id: str,
    workspace_id: Optional[UUID] = Query(None, description="Workspace UUID"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """View run plan details with variant results."""
    # If no workspace specified, get first available
    if not workspace_id and _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT id FROM workspaces LIMIT 1")
                if row:
                    workspace_id = row["id"]
        except Exception as e:
            logger.warning("Could not fetch default workspace", error=str(e))

    if not workspace_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="workspace_id is required",
        )

    # Get run plan events
    query = """
        SELECT
            payload->>'run_event_type' as event_type,
            payload,
            created_at
        FROM trade_events
        WHERE workspace_id = $1
          AND correlation_id = $2
          AND payload->>'run_event_type' IN (
              'RUN_STARTED', 'RUN_COMPLETED',
              'VARIANT_STARTED', 'VARIANT_COMPLETED', 'VARIANT_FAILED'
          )
        ORDER BY created_at ASC
    """

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, workspace_id, run_plan_id)

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run plan {run_plan_id} not found",
        )

    # Parse events
    run_plan = {
        "run_plan_id": run_plan_id,
        "workspace_id": str(workspace_id),
        "status": "running",
        "variants": [],
    }

    variants_by_id = {}

    for row in rows:
        event_type = row["event_type"]
        payload = row["payload"]
        created_at = row["created_at"]

        if event_type == "RUN_STARTED":
            run_plan["started_at"] = created_at
            run_plan["n_variants"] = payload.get("n_variants")
            run_plan["objective"] = payload.get("objective")
            run_plan["dataset_ref"] = payload.get("dataset_ref")
            run_plan["bar_count"] = payload.get("bar_count")

        elif event_type == "RUN_COMPLETED":
            run_plan["status"] = "completed"
            run_plan["completed_at"] = created_at
            run_plan["n_successful"] = payload.get("n_successful")
            run_plan["n_failed"] = payload.get("n_failed")
            run_plan["summary"] = payload.get("summary")

        elif event_type == "VARIANT_STARTED":
            variant_id = payload.get("variant_id")
            variants_by_id[variant_id] = {
                "variant_id": variant_id,
                "status": "running",
                "started_at": created_at,
                "overrides": payload.get("overrides"),
            }

        elif event_type == "VARIANT_COMPLETED":
            variant_id = payload.get("variant_id")
            if variant_id in variants_by_id:
                variants_by_id[variant_id]["status"] = "completed"
                variants_by_id[variant_id]["completed_at"] = created_at
                variants_by_id[variant_id]["metrics"] = payload.get("metrics")

        elif event_type == "VARIANT_FAILED":
            variant_id = payload.get("variant_id")
            if variant_id in variants_by_id:
                variants_by_id[variant_id]["status"] = "failed"
                variants_by_id[variant_id]["completed_at"] = created_at
                variants_by_id[variant_id]["error"] = payload.get("error")

    # Convert variants dict to list
    run_plan["variants"] = list(variants_by_id.values())

    # Paginate variants
    total_variants = len(run_plan["variants"])
    run_plan["variants"] = run_plan["variants"][offset : offset + limit]  # noqa: E203

    # Convert to JSON-serializable for debug panel
    run_plan_json = _json_serializable(run_plan)

    return templates.TemplateResponse(
        "run_plan_detail.html",
        {
            "request": request,
            "run_plan": run_plan,
            "run_plan_json": json.dumps(run_plan_json, indent=2),
            "total_variants": total_variants,
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total_variants,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


# ===========================================
# Run Plans API Endpoints (Verification)
# ===========================================


@router.get("/run-plans/{plan_id}")
async def get_run_plan(
    plan_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """Get run plan details (API endpoint for verification)."""
    repo = _get_run_plans_repo()

    plan = await repo.get_run_plan(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run plan not found",
        )

    return plan


@router.get("/run-plans/{plan_id}/runs")
async def get_run_plan_runs(
    plan_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """Get runs for a run plan (API endpoint for verification)."""
    repo = _get_run_plans_repo()

    plan = await repo.get_run_plan(plan_id)
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run plan not found",
        )

    runs, total = await repo.list_runs_for_plan(
        plan_id=plan_id,
        limit=limit,
        offset=offset,
    )

    return {
        "runs": runs,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
