"""Admin endpoints for operational alerts management."""

from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.deps.security import require_admin_token
from app.services.alerts.models import AlertStatus, RuleType, Severity

router = APIRouter(prefix="/ops-alerts", tags=["ops-alerts"])

# Templates setup
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup via set_db_pool)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for ops_alerts routes."""
    global _db_pool
    _db_pool = pool


def _json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable form."""
    from datetime import datetime
    from uuid import UUID

    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)
    return obj


def _get_alerts_repo():
    """Get AlertsRepository instance."""
    from app.repositories.alerts import AlertsRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return AlertsRepository(_db_pool)


# =============================================================================
# Admin List Page
# =============================================================================


@router.get("", response_class=HTMLResponse)
async def ops_alerts_list_page(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace ID (required)"),
    status_filter: Optional[AlertStatus] = Query(
        None,
        alias="status",
        description="Filter by status (active, resolved)",
    ),
    severity: Optional[Severity] = Query(None, description="Filter by severity"),
    rule_type: Optional[RuleType] = Query(None, description="Filter by rule type"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    token: Optional[str] = Query(None, description="Admin token (dev convenience)"),
    _: str = Depends(require_admin_token),
):
    """Render ops alerts list page with filters and actions."""
    # Get admin token from header or query param (query param for dev convenience)
    admin_token = request.headers.get("X-Admin-Token", "") or token or ""

    repo = _get_alerts_repo()

    # Fetch alerts
    events, total = await repo.list_events(
        workspace_id=workspace_id,
        status=status_filter,
        severity=severity,
        rule_type=rule_type,
        limit=limit,
        offset=offset,
    )

    # Calculate pagination
    has_prev = offset > 0
    has_next = offset + limit < total
    prev_offset = max(0, offset - limit)
    next_offset = offset + limit

    return templates.TemplateResponse(
        "ops_alerts_list.html",
        {
            "request": request,
            "workspace_id": str(workspace_id),
            "admin_token": admin_token,
            "alerts": events,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_prev": has_prev,
            "has_next": has_next,
            "prev_offset": prev_offset,
            "next_offset": next_offset,
            "status_filter": status_filter.value if status_filter else None,
            "severity_filter": severity.value if severity else None,
            "rule_type_filter": rule_type.value if rule_type else None,
        },
    )


# =============================================================================
# Action Endpoints
# =============================================================================


@router.post("/{event_id}/acknowledge")
async def acknowledge_ops_alert(
    event_id: UUID,
    acknowledged_by: Optional[str] = None,
    _: str = Depends(require_admin_token),
):
    """
    Acknowledge an operational alert.

    Marks the alert as acknowledged, optionally recording who acknowledged it.
    """
    repo = _get_alerts_repo()
    success = await repo.acknowledge(event_id, acknowledged_by=acknowledged_by)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found or already acknowledged",
        )

    logger.info(
        "ops_alert_acknowledged",
        event_id=str(event_id),
        acknowledged_by=acknowledged_by,
    )

    return {"acknowledged": True, "event_id": str(event_id)}


@router.post("/{event_id}/resolve")
async def resolve_ops_alert(
    event_id: UUID,
    _: str = Depends(require_admin_token),
):
    """
    Resolve an operational alert.

    Marks the alert as resolved. Resolved alerts are hidden from active views.
    """
    repo = _get_alerts_repo()
    success = await repo.resolve(event_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found or already resolved",
        )

    logger.info("ops_alert_resolved", event_id=str(event_id))

    return {"resolved": True, "event_id": str(event_id)}


@router.post("/{event_id}/reopen")
async def reopen_ops_alert(
    event_id: UUID,
    _: str = Depends(require_admin_token),
):
    """
    Reopen a resolved operational alert.

    Reactivates a resolved alert, returning it to active status.
    """
    repo = _get_alerts_repo()

    # Get the existing event to extract necessary fields
    event = await repo.get_event(event_id)

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )

    if event["status"] != "resolved":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Alert is not resolved, cannot reopen",
        )

    # Reactivate by upserting with active status
    result = await repo.upsert_activate(
        workspace_id=event["workspace_id"],
        rule_id=event["rule_id"],
        strategy_entity_id=event["strategy_entity_id"],
        regime_key=event["regime_key"],
        timeframe=event["timeframe"],
        rule_type=RuleType(event["rule_type"]),
        severity=Severity(event["severity"]),
        context_json=event["context_json"],
        fingerprint=event["fingerprint"],
    )

    logger.info("ops_alert_reopened", event_id=str(event_id))

    return {"reopened": True, "event_id": str(event_id)}
