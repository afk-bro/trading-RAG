"""Trade Events Journal admin UI endpoints."""

import json
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for trade events routes."""
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


def _get_trade_events_repo():
    """Get TradeEventsRepository instance."""
    from app.repositories.trade_events import TradeEventsRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return TradeEventsRepository(_db_pool)


@router.get("/trade/events", response_class=HTMLResponse)
async def admin_trade_events(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    strategy_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List trade events with filters."""
    from app.repositories.trade_events import EventFilters
    from app.schemas import TradeEventType

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
            "trade_events.html",
            {
                "request": request,
                "events": [],
                "total": 0,
                "workspace_id": None,
                "event_type": event_type,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "correlation_id": correlation_id,
                "hours": hours,
                "limit": limit,
                "offset": offset,
                "event_types": [e.value for e in TradeEventType],
                "error": "No workspace found. Create a workspace first.",
            },
        )

    # Build filters
    since = datetime.utcnow() - timedelta(hours=hours)

    event_types_filter = None
    if event_type:
        try:
            event_types_filter = [TradeEventType(event_type)]
        except ValueError:
            pass

    filters = EventFilters(
        workspace_id=workspace_id,
        event_types=event_types_filter,
        strategy_entity_id=strategy_id,
        symbol=symbol,
        correlation_id=correlation_id,
        since=since,
    )

    repo = _get_trade_events_repo()
    events, total = await repo.list_events(filters, limit=limit, offset=offset)

    # Convert events to dicts for template
    events_data = []
    for event in events:
        events_data.append(
            {
                "id": str(event.id),
                "correlation_id": event.correlation_id,
                "event_type": event.event_type.value,
                "created_at": event.created_at,
                "strategy_entity_id": (
                    str(event.strategy_entity_id) if event.strategy_entity_id else None
                ),
                "symbol": event.symbol,
                "timeframe": event.timeframe,
                "intent_id": str(event.intent_id) if event.intent_id else None,
                "payload": event.payload,
            }
        )

    # Get event type counts for sidebar
    type_counts = await repo.count_by_type(workspace_id, since_hours=hours)

    return templates.TemplateResponse(
        "trade_events.html",
        {
            "request": request,
            "events": events_data,
            "total": total,
            "workspace_id": str(workspace_id),
            "event_type": event_type or "",
            "strategy_id": str(strategy_id) if strategy_id else "",
            "symbol": symbol or "",
            "correlation_id": correlation_id or "",
            "hours": hours,
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
            "event_types": [e.value for e in TradeEventType],
            "type_counts": type_counts,
        },
    )


@router.get("/trade/events/{event_id}", response_class=HTMLResponse)
async def admin_trade_event_detail(
    request: Request,
    event_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """View trade event details."""
    repo = _get_trade_events_repo()
    event = await repo.get_by_id(event_id)

    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    # Get related events (same correlation_id)
    related_events = await repo.get_by_correlation_id(event.correlation_id)

    # Convert to template-friendly format
    event_data = {
        "id": str(event.id),
        "correlation_id": event.correlation_id,
        "workspace_id": str(event.workspace_id),
        "event_type": event.event_type.value,
        "created_at": event.created_at,
        "strategy_entity_id": (
            str(event.strategy_entity_id) if event.strategy_entity_id else None
        ),
        "symbol": event.symbol,
        "timeframe": event.timeframe,
        "intent_id": str(event.intent_id) if event.intent_id else None,
        "order_id": event.order_id,
        "position_id": event.position_id,
        "payload": event.payload,
        "metadata": event.metadata,
    }

    related_data = []
    for rel in related_events:
        related_data.append(
            {
                "id": str(rel.id),
                "event_type": rel.event_type.value,
                "created_at": rel.created_at,
                "is_current": rel.id == event.id,
            }
        )

    return templates.TemplateResponse(
        "trade_event_detail.html",
        {
            "request": request,
            "event": event_data,
            "event_json": json.dumps(_json_serializable(event_data), indent=2),
            "related_events": related_data,
        },
    )
