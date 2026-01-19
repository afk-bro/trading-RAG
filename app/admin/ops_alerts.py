"""Admin endpoints for operational alerts.

Provides read-only access to ops alerts with filtering,
plus acknowledge/resolve actions, and a management UI.
"""

from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token
from app.repositories.ops_alerts import OpsAlertsRepository, OpsAlert

# Templates
_template_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_template_dir))

router = APIRouter(prefix="/ops-alerts", tags=["admin-ops-alerts"])
logger = structlog.get_logger(__name__)

# Global pool reference (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def get_repo() -> OpsAlertsRepository:
    """Get repository instance."""
    if not _db_pool:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return OpsAlertsRepository(_db_pool)


# =============================================================================
# Response Models
# =============================================================================


class OpsAlertResponse(BaseModel):
    """Single ops alert response."""

    id: UUID
    workspace_id: UUID
    rule_type: str
    severity: str
    status: str
    rule_version: str
    dedupe_key: str
    payload: dict
    source: str
    job_run_id: Optional[UUID] = None
    created_at: str
    last_seen_at: str
    resolved_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    occurrence_count: int = 1

    @classmethod
    def from_model(cls, alert: OpsAlert) -> "OpsAlertResponse":
        """Create from OpsAlert model."""
        return cls(
            id=alert.id,
            workspace_id=alert.workspace_id,
            rule_type=alert.rule_type,
            severity=alert.severity,
            status=alert.status,
            rule_version=alert.rule_version,
            dedupe_key=alert.dedupe_key,
            payload=alert.payload,
            source=alert.source,
            job_run_id=alert.job_run_id,
            created_at=alert.created_at.isoformat(),
            last_seen_at=alert.last_seen_at.isoformat(),
            resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
            acknowledged_at=(
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
            ),
            acknowledged_by=alert.acknowledged_by,
            occurrence_count=alert.occurrence_count,
        )


class OpsAlertListResponse(BaseModel):
    """List of ops alerts with pagination."""

    items: list[OpsAlertResponse]
    total: int
    limit: int
    offset: int


class AcknowledgeResponse(BaseModel):
    """Response for acknowledge action."""

    id: UUID
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    was_already_acknowledged: bool


class ResolveResponse(BaseModel):
    """Response for resolve action."""

    id: UUID
    resolved_at: Optional[str] = None
    was_already_resolved: bool


class ReopenResponse(BaseModel):
    """Response for reopen action."""

    id: UUID
    status: str
    was_already_active: bool


# =============================================================================
# Endpoints
# =============================================================================


# =============================================================================
# UI Endpoint
# =============================================================================


@router.get("/ui", response_class=HTMLResponse)
async def ops_alerts_ui(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace ID (required)"),
    _: bool = Depends(require_admin_token),
) -> HTMLResponse:
    """
    Ops alerts management UI.

    Two-panel layout with queue list and detail panel.
    """
    repo = get_repo()

    # Get active alerts for initial load
    active_alerts, active_total = await repo.list_alerts(
        workspace_id=workspace_id,
        status=["active"],
        limit=50,
    )

    # Get resolved alerts count
    _resolved_alerts, resolved_total = await repo.list_alerts(
        workspace_id=workspace_id,
        status=["resolved"],
        limit=1,
    )

    # Build severity counts for badges
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for alert in active_alerts:
        if alert.severity in severity_counts:
            severity_counts[alert.severity] += 1

    # Build rule type counts
    rule_type_counts: dict[str, int] = {}
    for alert in active_alerts:
        rule_type_counts[alert.rule_type] = rule_type_counts.get(alert.rule_type, 0) + 1

    return templates.TemplateResponse(
        "ops_alerts.html",
        {
            "request": request,
            "workspace_id": str(workspace_id),
            "alerts": active_alerts,
            "active_count": active_total,
            "resolved_count": resolved_total,
            "severity_counts": severity_counts,
            "rule_type_counts": rule_type_counts,
            "selected_status": "active",
        },
    )


@router.get("", response_model=OpsAlertListResponse)
async def list_ops_alerts(
    workspace_id: UUID = Query(..., description="Workspace ID (required)"),
    status: Optional[str] = Query(
        None,
        description="Comma-separated status filter: active,resolved",
    ),
    severity: Optional[str] = Query(
        None,
        description="Comma-separated severity filter: critical,high,medium,low",
    ),
    rule_type: Optional[str] = Query(
        None,
        description="Comma-separated rule type filter",
    ),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
) -> OpsAlertListResponse:
    """
    List operational alerts with filters.

    Ordering:
    - If status=active only: ordered by last_seen_at DESC (still happening first)
    - Otherwise: ordered by created_at DESC
    """
    repo = get_repo()

    # Parse comma-separated filters
    status_list = [s.strip() for s in status.split(",")] if status else None
    severity_list = [s.strip() for s in severity.split(",")] if severity else None
    rule_type_list = [r.strip() for r in rule_type.split(",")] if rule_type else None

    alerts, total = await repo.list_alerts(
        workspace_id=workspace_id,
        status=status_list,
        severity=severity_list,
        rule_type=rule_type_list,
        limit=limit,
        offset=offset,
    )

    return OpsAlertListResponse(
        items=[OpsAlertResponse.from_model(a) for a in alerts],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{alert_id}", response_model=OpsAlertResponse)
async def get_ops_alert(
    alert_id: UUID,
    _: bool = Depends(require_admin_token),
) -> OpsAlertResponse:
    """Get a single ops alert by ID."""
    repo = get_repo()

    alert = await repo.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    return OpsAlertResponse.from_model(alert)


@router.post("/{alert_id}/acknowledge", response_model=AcknowledgeResponse)
async def acknowledge_ops_alert(
    alert_id: UUID,
    acknowledged_by: Optional[str] = Query(None, description="Who acknowledged"),
    _: bool = Depends(require_admin_token),
) -> AcknowledgeResponse:
    """
    Acknowledge an ops alert.

    Idempotent: returns 200 even if already acknowledged.
    """
    repo = get_repo()

    # First check if alert exists
    alert = await repo.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    success, was_already = await repo.acknowledge(alert_id, acknowledged_by)

    # Fetch updated alert
    alert = await repo.get(alert_id)

    return AcknowledgeResponse(
        id=alert_id,
        acknowledged_at=(
            alert.acknowledged_at.isoformat()
            if alert and alert.acknowledged_at
            else None
        ),
        acknowledged_by=alert.acknowledged_by if alert else acknowledged_by,
        was_already_acknowledged=was_already,
    )


@router.post("/{alert_id}/resolve", response_model=ResolveResponse)
async def resolve_ops_alert(
    alert_id: UUID,
    _: bool = Depends(require_admin_token),
) -> ResolveResponse:
    """
    Resolve an ops alert.

    Idempotent: returns 200 even if already resolved.
    """
    repo = get_repo()

    # First check if alert exists
    alert = await repo.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    was_already = alert.status == "resolved"

    if not was_already:
        alert = await repo.resolve(alert_id)

    return ResolveResponse(
        id=alert_id,
        resolved_at=(
            alert.resolved_at.isoformat() if alert and alert.resolved_at else None
        ),
        was_already_resolved=was_already,
    )


@router.post("/{alert_id}/reopen", response_model=ReopenResponse)
async def reopen_ops_alert(
    alert_id: UUID,
    _: bool = Depends(require_admin_token),
) -> ReopenResponse:
    """
    Reopen a resolved ops alert.

    Sets status back to 'active', clears resolved_at.
    Keeps acknowledged_at/acknowledged_by for history.

    Idempotent: returns 200 even if already active.
    """
    repo = get_repo()

    # First check if alert exists
    alert = await repo.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    was_already_active = alert.status == "active"

    if not was_already_active:
        alert = await repo.reopen(alert_id)

    return ReopenResponse(
        id=alert_id,
        status=alert.status if alert else "active",
        was_already_active=was_already_active,
    )


# =============================================================================
# Trigger endpoint (for manual/cron invocation)
# =============================================================================


class EvaluateRequest(BaseModel):
    """Request to trigger alert evaluation."""

    workspace_id: Optional[UUID] = Field(
        None, description="Evaluate single workspace, or all if not provided"
    )
    dry_run: bool = Field(False, description="Evaluate without writing alerts")


class EvaluateResponse(BaseModel):
    """Response from alert evaluation."""

    workspaces_evaluated: int
    total_conditions: int
    total_triggered: int
    total_new: int
    total_resolved: int
    total_escalated: int
    telegram_sent: int
    errors: list[str]


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_ops_alerts(
    request: EvaluateRequest,
    _: bool = Depends(require_admin_token),
) -> EvaluateResponse:
    """
    Trigger ops alert evaluation.

    This is the manual/cron entrypoint for alert evaluation.
    For production, use the job queue instead.

    If workspace_id is provided, evaluates just that workspace.
    Otherwise, iterates all active workspaces.
    """
    from app.services.ops_alerts.evaluator import OpsAlertEvaluator
    from app.services.ops_alerts.telegram import get_telegram_notifier

    if not _db_pool:
        raise HTTPException(status_code=503, detail="Database not initialized")

    repo = get_repo()
    evaluator = OpsAlertEvaluator(repo, _db_pool)
    notifier = get_telegram_notifier() if not request.dry_run else None

    # Get workspaces to evaluate
    if request.workspace_id:
        workspace_ids = [request.workspace_id]
    else:
        query = "SELECT id FROM workspaces WHERE is_active = true ORDER BY created_at"
        async with _db_pool.acquire() as conn:
            rows = await conn.fetch(query)
        workspace_ids = [r["id"] for r in rows]

    # Aggregate results
    result: dict[str, Any] = {
        "workspaces_evaluated": 0,
        "total_conditions": 0,
        "total_triggered": 0,
        "total_new": 0,
        "total_resolved": 0,
        "total_escalated": 0,
        "telegram_sent": 0,
        "errors": [],
    }

    for ws_id in workspace_ids:
        try:
            eval_result = await evaluator.evaluate(workspace_id=ws_id)

            result["workspaces_evaluated"] += 1
            result["total_conditions"] += eval_result.conditions_evaluated
            result["total_triggered"] += eval_result.alerts_triggered
            result["total_new"] += eval_result.alerts_new
            result["total_resolved"] += eval_result.alerts_resolved
            result["total_escalated"] += eval_result.alerts_escalated

            if eval_result.errors:
                result["errors"].extend([f"{ws_id}:{e}" for e in eval_result.errors])

            # Send notifications
            if notifier and not request.dry_run:
                alerts_to_notify = []
                for rule_type, details in eval_result.by_rule_type.items():
                    alert_id_str = details.get("alert_id")
                    if alert_id_str:
                        alert = await repo.get(UUID(alert_id_str))
                        if alert:
                            is_new = details.get("new", False)
                            is_escalated = details.get("escalated", False)
                            if is_new or is_escalated:
                                alerts_to_notify.append((alert, False, is_escalated))

                # Handle resolved alerts
                for rule_type, details in eval_result.by_rule_type.items():
                    if details.get("resolved"):
                        alerts, _total = await repo.list_alerts(
                            workspace_id=ws_id,
                            status=["resolved"],
                            rule_type=[rule_type],
                            limit=1,
                        )
                        if alerts:
                            alerts_to_notify.append((alerts[0], True, False))

                if alerts_to_notify:
                    sent = await notifier.send_batch(alerts_to_notify)
                    result["telegram_sent"] += sent

        except Exception as e:
            logger.error(
                "evaluate_workspace_error", workspace_id=str(ws_id), error=str(e)
            )
            result["errors"].append(f"{ws_id}:error:{str(e)}")

    return EvaluateResponse(**result)
