"""Admin endpoints for alert rules and events."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token
from app.services.alerts.models import AlertStatus, RuleType, Severity

router = APIRouter(prefix="/alerts", tags=["alerts"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup via set_db_pool)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for alerts routes."""
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
# Request/Response Models
# =============================================================================


class CreateRuleRequest(BaseModel):
    """Request model for creating an alert rule."""

    rule_type: RuleType
    strategy_entity_id: Optional[UUID] = None
    regime_key: Optional[str] = None
    timeframe: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    cooldown_minutes: int = Field(default=60, ge=1)


class UpdateRuleRequest(BaseModel):
    """Request model for updating an alert rule."""

    enabled: Optional[bool] = None
    config: Optional[dict[str, Any]] = None
    cooldown_minutes: Optional[int] = Field(default=None, ge=1)


class AcknowledgeRequest(BaseModel):
    """Request model for acknowledging an alert."""

    acknowledged_by: Optional[str] = None


# =============================================================================
# Alert Rules Endpoints
# =============================================================================


@router.get("/rules")
async def list_alert_rules(
    workspace_id: UUID = Query(..., description="Workspace ID (required)"),
    enabled_only: bool = Query(False, description="Filter to enabled rules only"),
    _: bool = Depends(require_admin_token),
):
    """
    List alert rules for a workspace.

    Returns all alert rules configured for the specified workspace,
    optionally filtered to only enabled rules.
    """
    repo = _get_alerts_repo()
    rules = await repo.list_rules(workspace_id=workspace_id, enabled_only=enabled_only)

    return {
        "rules": [_json_serializable(r) for r in rules],
        "count": len(rules),
    }


@router.post("/rules", status_code=status.HTTP_201_CREATED)
async def create_alert_rule(
    request: CreateRuleRequest,
    workspace_id: UUID = Query(..., description="Workspace ID (required)"),
    _: bool = Depends(require_admin_token),
):
    """
    Create a new alert rule.

    Creates an alert rule for the specified workspace with the given
    rule type and configuration. Rules are enabled by default.
    """
    repo = _get_alerts_repo()
    rule = await repo.create_rule(
        workspace_id=workspace_id,
        rule_type=request.rule_type,
        config=request.config,
        strategy_entity_id=request.strategy_entity_id,
        regime_key=request.regime_key,
        timeframe=request.timeframe,
        cooldown_minutes=request.cooldown_minutes,
    )

    logger.info(
        "alert_rule_created",
        rule_id=str(rule["id"]),
        workspace_id=str(workspace_id),
        rule_type=request.rule_type.value,
    )

    return _json_serializable(rule)


@router.get("/rules/{rule_id}")
async def get_alert_rule(
    rule_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Get alert rule details.

    Returns the full configuration and metadata for a specific alert rule.
    """
    repo = _get_alerts_repo()
    rule = await repo.get_rule(rule_id)

    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert rule not found",
        )

    return _json_serializable(rule)


@router.patch("/rules/{rule_id}")
async def update_alert_rule(
    rule_id: UUID,
    request: UpdateRuleRequest,
    _: bool = Depends(require_admin_token),
):
    """
    Update an alert rule.

    Updates the specified fields of an alert rule. Only provided fields
    are updated; others remain unchanged.
    """
    repo = _get_alerts_repo()
    rule = await repo.update_rule(
        rule_id=rule_id,
        enabled=request.enabled,
        config=request.config,
        cooldown_minutes=request.cooldown_minutes,
    )

    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert rule not found",
        )

    logger.info(
        "alert_rule_updated",
        rule_id=str(rule_id),
        enabled=request.enabled,
    )

    return _json_serializable(rule)


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert_rule(
    rule_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Delete an alert rule.

    Permanently removes an alert rule. Associated events are not deleted.
    """
    repo = _get_alerts_repo()
    deleted = await repo.delete_rule(rule_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert rule not found",
        )

    logger.info("alert_rule_deleted", rule_id=str(rule_id))

    return None


# =============================================================================
# Alert Events Endpoints
# =============================================================================


@router.get("")
async def list_alert_events(
    workspace_id: UUID = Query(..., description="Workspace ID (required)"),
    status_filter: Optional[AlertStatus] = Query(
        None,
        alias="status",
        description="Filter by status (active, resolved)",
    ),
    severity: Optional[Severity] = Query(None, description="Filter by severity"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledged"),
    rule_type: Optional[RuleType] = Query(None, description="Filter by rule type"),
    strategy_entity_id: Optional[UUID] = Query(
        None, description="Filter by strategy entity"
    ),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    regime_key: Optional[str] = Query(None, description="Filter by regime key"),
    from_ts: Optional[datetime] = Query(
        None, alias="from", description="Start timestamp filter (last_seen >= from)"
    ),
    to_ts: Optional[datetime] = Query(
        None, alias="to", description="End timestamp filter (last_seen <= to)"
    ),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
):
    """
    List alert events for a workspace.

    Returns paginated list of alert events with optional filters for
    status, severity, acknowledgment state, rule type, strategy, timeframe,
    regime key, and time range.
    """
    repo = _get_alerts_repo()
    events, total = await repo.list_events(
        workspace_id=workspace_id,
        status=status_filter,
        severity=severity,
        acknowledged=acknowledged,
        rule_type=rule_type,
        strategy_entity_id=strategy_entity_id,
        timeframe=timeframe,
        regime_key=regime_key,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=limit,
        offset=offset,
    )

    return {
        "items": [_json_serializable(e) for e in events],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{event_id}")
async def get_alert_event(
    event_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Get alert event details.

    Returns the full details and context for a specific alert event.
    """
    repo = _get_alerts_repo()
    event = await repo.get_event(event_id)

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert event not found",
        )

    return _json_serializable(event)


@router.post("/{event_id}/acknowledge")
async def acknowledge_alert_event(
    event_id: UUID,
    request: Optional[AcknowledgeRequest] = None,
    _: bool = Depends(require_admin_token),
):
    """
    Acknowledge an alert event.

    Marks the alert as acknowledged, optionally recording who acknowledged it.
    Acknowledgment does not resolve the alert but indicates awareness.
    """
    repo = _get_alerts_repo()
    acknowledged_by = request.acknowledged_by if request else None
    success = await repo.acknowledge(event_id, acknowledged_by=acknowledged_by)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert event not found or already acknowledged",
        )

    logger.info(
        "alert_acknowledged",
        event_id=str(event_id),
        acknowledged_by=acknowledged_by,
    )

    return {"acknowledged": True, "event_id": str(event_id)}


@router.post("/{event_id}/unacknowledge")
async def unacknowledge_alert_event(
    event_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """
    Unacknowledge an alert event.

    Clears the acknowledgment state, returning the alert to unacknowledged.
    """
    repo = _get_alerts_repo()
    success = await repo.unacknowledge(event_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert event not found or not acknowledged",
        )

    logger.info("alert_unacknowledged", event_id=str(event_id))

    return {"acknowledged": False, "event_id": str(event_id)}
