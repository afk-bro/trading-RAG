"""Read-only dashboard endpoints for trust-building visualizations."""

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.lifespan import get_db_pool
from app.deps.security import check_workspace_consistency
from app.repositories.trade_events import EventFilters, TradeEventsRepository
from app.schemas import TradeEventType

router = APIRouter(
    prefix="/dashboards",
    tags=["dashboards"],
    dependencies=[Depends(check_workspace_consistency)],
)


# =============================================================================
# Equity Curve Endpoint
# =============================================================================


@router.get("/{workspace_id}/equity")
async def get_equity_curve(
    workspace_id: UUID,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """
    Get equity curve data with drawdown overlay.

    Returns time series of:
    - equity: Total equity value over time
    - cash: Cash component
    - positions_value: Positions value component
    - drawdown_pct: Calculated drawdown from rolling peak

    Use for equity curve chart with drawdown overlay.
    """
    query = """
        WITH snapshots AS (
            SELECT
                snapshot_ts,
                computed_at,
                equity,
                cash,
                positions_value,
                realized_pnl,
                -- Calculate rolling peak and drawdown
                MAX(equity) OVER (
                    ORDER BY snapshot_ts
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS peak_equity
            FROM paper_equity_snapshots
            WHERE workspace_id = $1
              AND snapshot_ts >= NOW() - make_interval(days => $2)
            ORDER BY snapshot_ts ASC
        )
        SELECT
            snapshot_ts,
            computed_at,
            equity,
            cash,
            positions_value,
            realized_pnl,
            peak_equity,
            CASE
                WHEN peak_equity > 0 THEN (peak_equity - equity) / peak_equity
                ELSE 0.0
            END AS drawdown_pct
        FROM snapshots
        ORDER BY snapshot_ts ASC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, workspace_id, days)

    data_points = [
        {
            "snapshot_ts": row["snapshot_ts"].isoformat(),
            "computed_at": row["computed_at"].isoformat(),
            "equity": row["equity"],
            "cash": row["cash"],
            "positions_value": row["positions_value"],
            "realized_pnl": row["realized_pnl"],
            "peak_equity": row["peak_equity"],
            "drawdown_pct": float(row["drawdown_pct"]) if row["drawdown_pct"] else 0.0,
        }
        for row in rows
    ]

    # Calculate summary stats
    if data_points:
        latest = data_points[-1]
        max_dd = max(p["drawdown_pct"] for p in data_points)
        starting_equity = data_points[0]["equity"]
        current_equity = latest["equity"]
        total_return_pct = (
            (current_equity - starting_equity) / starting_equity
            if starting_equity > 0
            else 0.0
        )
    else:
        latest = None
        max_dd = 0.0
        total_return_pct = 0.0

    return {
        "workspace_id": str(workspace_id),
        "window_days": days,
        "snapshot_count": len(data_points),
        "data": data_points,
        "summary": {
            "current_equity": latest["equity"] if latest else None,
            "current_drawdown_pct": latest["drawdown_pct"] if latest else None,
            "max_drawdown_pct": max_dd,
            "total_return_pct": total_return_pct,
            "latest_ts": latest["snapshot_ts"] if latest else None,
        },
    }


# =============================================================================
# Confidence & Regime Timeline Endpoint
# =============================================================================


@router.get("/{workspace_id}/intel-timeline")
async def get_intel_timeline(
    workspace_id: UUID,
    version_id: Optional[UUID] = None,
    days: Annotated[int, Query(ge=1, le=90)] = 14,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """
    Get strategy intel timeline with confidence and regime data.

    Returns time series of:
    - confidence_score: Overall confidence score
    - regime: Current regime classification
    - confidence_components: Breakdown by component

    If version_id is not specified, returns data for all active versions.
    """
    # Build query based on whether version_id is specified
    if version_id:
        query = """
            SELECT
                sis.id,
                sis.strategy_version_id,
                sv.version_number,
                sv.version_tag,
                s.name AS strategy_name,
                sis.as_of_ts,
                sis.computed_at,
                sis.regime,
                sis.confidence_score,
                sis.confidence_components,
                sis.source_snapshot_id
            FROM strategy_intel_snapshots sis
            JOIN strategy_versions sv ON sis.strategy_version_id = sv.id
            JOIN strategies s ON sv.strategy_id = s.id
            WHERE s.workspace_id = $1
              AND sis.strategy_version_id = $2
              AND sis.as_of_ts >= NOW() - make_interval(days => $3)
            ORDER BY sis.as_of_ts DESC
        """
        params = [workspace_id, version_id, days]
    else:
        query = """
            SELECT
                sis.id,
                sis.strategy_version_id,
                sv.version_number,
                sv.version_tag,
                s.name AS strategy_name,
                sis.as_of_ts,
                sis.computed_at,
                sis.regime,
                sis.confidence_score,
                sis.confidence_components,
                sis.source_snapshot_id
            FROM strategy_intel_snapshots sis
            JOIN strategy_versions sv ON sis.strategy_version_id = sv.id
            JOIN strategies s ON sv.strategy_id = s.id
            WHERE s.workspace_id = $1
              AND sv.state = 'active'
              AND sis.as_of_ts >= NOW() - make_interval(days => $2)
            ORDER BY sis.as_of_ts DESC
        """
        params = [workspace_id, days]

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    # Group by version
    by_version: dict[str, list[dict]] = {}
    for row in rows:
        vid = str(row["strategy_version_id"])
        if vid not in by_version:
            by_version[vid] = []

        by_version[vid].append(
            {
                "snapshot_id": str(row["id"]),
                "as_of_ts": row["as_of_ts"].isoformat(),
                "computed_at": row["computed_at"].isoformat(),
                "regime": row["regime"],
                "confidence_score": row["confidence_score"],
                "confidence_components": row["confidence_components"],
            }
        )

    # Build version metadata
    versions: list[dict[str, Any]] = []
    for row in rows:
        vid = str(row["strategy_version_id"])
        # Only add once per version
        if not any(v["version_id"] == vid for v in versions):
            snapshots = by_version.get(vid, [])
            versions.append(
                {
                    "version_id": vid,
                    "version_number": row["version_number"],
                    "version_tag": row["version_tag"],
                    "strategy_name": row["strategy_name"],
                    "snapshot_count": len(snapshots),
                    "latest_confidence": (
                        snapshots[0]["confidence_score"] if snapshots else None
                    ),
                    "latest_regime": snapshots[0]["regime"] if snapshots else None,
                    "snapshots": snapshots,
                }
            )

    return {
        "workspace_id": str(workspace_id),
        "version_filter": str(version_id) if version_id else "active",
        "window_days": days,
        "version_count": len(versions),
        "total_snapshots": sum(len(v["snapshots"]) for v in versions),
        "versions": versions,
    }


# =============================================================================
# Active Alerts Endpoint
# =============================================================================


@router.get("/{workspace_id}/alerts")
async def get_active_alerts(
    workspace_id: UUID,
    include_resolved: Annotated[bool, Query()] = False,
    days: Annotated[int, Query(ge=1, le=90)] = 7,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """
    Get active (and optionally resolved) alerts for a workspace.

    Returns alerts with:
    - rule_type, severity, status
    - dedupe_key, payload
    - timestamps (created, acknowledged, resolved)

    Use for alerts list/dashboard.
    """
    if include_resolved:
        status_filter = "1=1"  # No filter
    else:
        status_filter = "status = 'active'"

    query = f"""
        SELECT
            id,
            workspace_id,
            rule_type,
            severity,
            status,
            dedupe_key,
            payload,
            source,
            rule_version,
            occurrence_count,
            first_triggered_at,
            last_triggered_at,
            acknowledged_at,
            acknowledged_by,
            resolved_at,
            resolved_by,
            resolution_note,
            created_at
        FROM ops_alerts
        WHERE workspace_id = $1
          AND {status_filter}
          AND created_at >= NOW() - make_interval(days => $2)
        ORDER BY
            CASE status
                WHEN 'active' THEN 0
                WHEN 'acknowledged' THEN 1
                WHEN 'resolved' THEN 2
            END,
            CASE severity
                WHEN 'critical' THEN 0
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                WHEN 'low' THEN 3
            END,
            last_triggered_at DESC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, workspace_id, days)

    alerts = [
        {
            "id": str(row["id"]),
            "rule_type": row["rule_type"],
            "severity": row["severity"],
            "status": row["status"],
            "dedupe_key": row["dedupe_key"],
            "payload": row["payload"],
            "occurrence_count": row["occurrence_count"],
            "first_triggered_at": (
                row["first_triggered_at"].isoformat()
                if row["first_triggered_at"]
                else None
            ),
            "last_triggered_at": (
                row["last_triggered_at"].isoformat()
                if row["last_triggered_at"]
                else None
            ),
            "acknowledged_at": (
                row["acknowledged_at"].isoformat() if row["acknowledged_at"] else None
            ),
            "acknowledged_by": row["acknowledged_by"],
            "resolved_at": (
                row["resolved_at"].isoformat() if row["resolved_at"] else None
            ),
            "resolved_by": row["resolved_by"],
            "resolution_note": row["resolution_note"],
        }
        for row in rows
    ]

    # Count by status and severity
    by_status = {"active": 0, "acknowledged": 0, "resolved": 0}
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    by_rule_type: dict[str, int] = {}

    for alert in alerts:
        by_status[alert["status"]] = by_status.get(alert["status"], 0) + 1
        by_severity[alert["severity"]] = by_severity.get(alert["severity"], 0) + 1
        by_rule_type[alert["rule_type"]] = by_rule_type.get(alert["rule_type"], 0) + 1

    return {
        "workspace_id": str(workspace_id),
        "include_resolved": include_resolved,
        "window_days": days,
        "total_alerts": len(alerts),
        "summary": {
            "by_status": by_status,
            "by_severity": by_severity,
            "by_rule_type": by_rule_type,
        },
        "alerts": alerts,
    }


# =============================================================================
# Dashboard Summary Endpoint (combines all)
# =============================================================================


@router.get("/{workspace_id}/summary")
async def get_dashboard_summary(
    workspace_id: UUID,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """
    Get combined dashboard summary for a workspace.

    Returns:
    - Current equity and drawdown
    - Latest confidence scores for active versions
    - Active alert counts by severity

    Use as a single call for dashboard overview cards.
    """
    # Get latest equity snapshot
    equity_query = """
        WITH latest AS (
            SELECT
                equity,
                cash,
                positions_value,
                snapshot_ts,
                MAX(equity) OVER (
                    ORDER BY snapshot_ts
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS peak_equity
            FROM paper_equity_snapshots
            WHERE workspace_id = $1
              AND snapshot_ts >= NOW() - INTERVAL '30 days'
            ORDER BY snapshot_ts DESC
            LIMIT 1
        )
        SELECT
            equity,
            cash,
            positions_value,
            snapshot_ts,
            peak_equity,
            CASE
                WHEN peak_equity > 0 THEN (peak_equity - equity) / peak_equity
                ELSE 0.0
            END AS drawdown_pct
        FROM latest
    """

    # Get latest intel for active versions
    intel_query = """
        WITH latest_per_version AS (
            SELECT
                sis.strategy_version_id,
                sv.version_number,
                s.name AS strategy_name,
                sis.regime,
                sis.confidence_score,
                sis.as_of_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY sis.strategy_version_id
                    ORDER BY sis.as_of_ts DESC
                ) AS rn
            FROM strategy_intel_snapshots sis
            JOIN strategy_versions sv ON sis.strategy_version_id = sv.id
            JOIN strategies s ON sv.strategy_id = s.id
            WHERE s.workspace_id = $1
              AND sv.state = 'active'
        )
        SELECT *
        FROM latest_per_version
        WHERE rn = 1
    """

    # Get active alert counts
    alerts_query = """
        SELECT
            severity,
            COUNT(*) AS count
        FROM ops_alerts
        WHERE workspace_id = $1
          AND status = 'active'
        GROUP BY severity
    """

    async with pool.acquire() as conn:
        equity_row = await conn.fetchrow(equity_query, workspace_id)
        intel_rows = await conn.fetch(intel_query, workspace_id)
        alert_rows = await conn.fetch(alerts_query, workspace_id)

    # Build equity summary
    equity_summary = None
    if equity_row:
        equity_summary = {
            "equity": equity_row["equity"],
            "cash": equity_row["cash"],
            "positions_value": equity_row["positions_value"],
            "drawdown_pct": (
                float(equity_row["drawdown_pct"]) if equity_row["drawdown_pct"] else 0.0
            ),
            "peak_equity": equity_row["peak_equity"],
            "as_of": equity_row["snapshot_ts"].isoformat(),
        }

    # Build intel summary
    intel_summary = [
        {
            "version_id": str(row["strategy_version_id"]),
            "version_number": row["version_number"],
            "strategy_name": row["strategy_name"],
            "regime": row["regime"],
            "confidence_score": row["confidence_score"],
            "as_of": row["as_of_ts"].isoformat(),
        }
        for row in intel_rows
    ]

    # Build alert summary
    alert_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for row in alert_rows:
        alert_counts[row["severity"]] = row["count"]

    total_active_alerts = sum(alert_counts.values())

    return {
        "workspace_id": str(workspace_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "equity": equity_summary,
        "intel": {
            "active_versions": len(intel_summary),
            "versions": intel_summary,
        },
        "alerts": {
            "total_active": total_active_alerts,
            "by_severity": alert_counts,
        },
    }


# =============================================================================
# Trade Events Endpoints
# =============================================================================


def _normalize_event(event) -> dict:
    """Extract common fields from trade event payload for UI consumption."""
    payload = event.payload or {}
    metadata = event.metadata or {}

    # Derive side from payload or event_type
    side = payload.get("side")
    if not side:
        et = (
            event.event_type.value
            if hasattr(event.event_type, "value")
            else str(event.event_type)
        )
        if "long" in et.lower() or et in ("POSITION_OPENED",):
            side = payload.get("side", "long")
        elif "short" in et.lower():
            side = "short"

    # Derive prices
    entry_price = (
        payload.get("entry_price")
        or payload.get("fill_price")
        or payload.get("avg_price")
    )
    exit_price = None
    et_str = (
        event.event_type.value
        if hasattr(event.event_type, "value")
        else str(event.event_type)
    )
    if et_str == "position_closed":
        exit_price = payload.get("exit_price") or payload.get("fill_price")

    pnl = payload.get("pnl") or payload.get("realized_pnl")

    return {
        "id": str(event.id),
        "correlation_id": event.correlation_id,
        "event_type": et_str,
        "event_time": event.created_at.isoformat() if event.created_at else None,
        "symbol": event.symbol,
        "side": side,
        "entry_price": float(entry_price) if entry_price is not None else None,
        "exit_price": float(exit_price) if exit_price is not None else None,
        "pnl": float(pnl) if pnl is not None else None,
        "duration_s": payload.get("duration_s"),
        "strategy_entity_id": (
            str(event.strategy_entity_id) if event.strategy_entity_id else None
        ),
        "payload": payload,
        "metadata": metadata,
    }


@router.get("/{workspace_id}/trade-events")
async def list_trade_events(
    workspace_id: UUID,
    event_type: Optional[str] = None,
    symbol: Optional[str] = None,
    correlation_id: Optional[str] = None,
    days: Annotated[int, Query(ge=1, le=90)] = 30,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """List trade events with normalized fields for dashboard display."""
    since = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=days)

    event_types = None
    if event_type:
        try:
            event_types = [TradeEventType(event_type)]
        except ValueError:
            pass

    filters = EventFilters(
        workspace_id=workspace_id,
        event_types=event_types,
        symbol=symbol,
        correlation_id=correlation_id,
        since=since,
    )

    repo = TradeEventsRepository(pool)
    events, total = await repo.list_events(filters, limit=limit, offset=offset)

    return {
        "items": [_normalize_event(e) for e in events],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{workspace_id}/trade-events/{event_id}")
async def get_trade_event_detail(
    workspace_id: UUID,
    event_id: UUID,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """Get a single trade event with related events by correlation_id."""
    repo = TradeEventsRepository(pool)
    event = await repo.get_by_id(event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")

    related = await repo.get_by_correlation_id(event.correlation_id)
    related_normalized = [_normalize_event(e) for e in related if e.id != event_id]

    return {
        "event": _normalize_event(event),
        "related_events": related_normalized,
    }


# =============================================================================
# Backtest Run Detail (workspace-scoped, UI-shaped DTO)
# =============================================================================


def _parse_jsonb(raw: Any) -> Any:
    """Parse a JSONB field that might be a string."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw


def _build_drawdown_series(equity_points: list[dict]) -> list[dict]:
    """Compute drawdown series from equity points."""
    if not equity_points:
        return []
    peak = equity_points[0].get("equity", 0)
    result = []
    for p in equity_points:
        eq = p.get("equity", 0)
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        result.append({"t": p.get("t", ""), "drawdown_pct": round(dd, 6)})
    return result


def _slim_trades(trades_raw: Any, limit: int = 200) -> list[dict]:
    """Return slim trade records (no full payload)."""
    data = _parse_jsonb(trades_raw)
    if not data or not isinstance(data, list):
        return []
    out = []
    for t in data[:limit]:
        if not isinstance(t, dict):
            continue
        t_entry = t.get("t_entry") or t.get("entry_time") or t.get("EntryTime") or ""
        t_exit = t.get("t_exit") or t.get("exit_time") or t.get("ExitTime") or ""
        side = t.get("side") or t.get("Size", "long")
        if isinstance(side, (int, float)):
            side = "long" if side > 0 else "short"
        out.append(
            {
                "t_entry": str(t_entry),
                "t_exit": str(t_exit),
                "side": str(side).lower(),
                "entry_price": t.get("entry_price") or t.get("EntryPrice"),
                "exit_price": t.get("exit_price") or t.get("ExitPrice"),
                "pnl": float(t.get("pnl") or t.get("PnL") or 0),
                "return_pct": float(t.get("return_pct") or t.get("ReturnPct") or 0),
            }
        )
    return out


@router.get("/{workspace_id}/backtests/{run_id}")
async def get_backtest_run_detail(
    workspace_id: UUID,
    run_id: UUID,
    trades_limit: Annotated[int, Query(ge=1, le=500)] = 200,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """
    Workspace-scoped backtest run detail.

    Returns a UI-shaped DTO with:
    - Headline metrics (summary)
    - Equity curve series
    - Drawdown series (computed)
    - Trade list (slim, capped)
    - Dataset + strategy metadata
    - Regime tags
    """
    query = """
        SELECT
            br.id,
            br.status,
            br.created_at,
            br.started_at,
            br.completed_at,
            br.params,
            br.summary,
            br.dataset_meta,
            br.equity_curve,
            br.trades,
            br.warnings,
            br.run_kind,
            br.regime_is,
            br.regime_oos,
            br.trade_count,
            br.strategy_entity_id,
            br.strategy_version_id,
            -- Strategy name via kb_entities
            ke.title AS strategy_name
        FROM backtest_runs br
        LEFT JOIN kb_entities ke ON br.strategy_entity_id = ke.id
        WHERE br.id = $1
          AND br.workspace_id = $2
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, run_id, workspace_id)

    if not row:
        raise HTTPException(status_code=404, detail="Backtest run not found")

    # Parse equity curve
    equity_raw = _parse_jsonb(row["equity_curve"])
    equity_points: list[dict] = []
    if isinstance(equity_raw, list):
        for p in equity_raw:
            if isinstance(p, dict):
                t = p.get("t") or p.get("timestamp") or p.get("time")
                eq = p.get("equity") or p.get("value") or p.get("Equity")
                if t is not None and eq is not None:
                    equity_points.append({"t": str(t), "equity": float(eq)})

    # Compute drawdown series
    drawdown = _build_drawdown_series(equity_points)

    # Summary metrics
    summary = _parse_jsonb(row["summary"]) or {}

    # Dataset metadata
    dataset_meta = _parse_jsonb(row["dataset_meta"]) or {}

    # Slim trades
    trades = _slim_trades(row["trades"], limit=trades_limit)

    # Params
    params = _parse_jsonb(row["params"]) or {}

    # Regime tags
    def _regime_tags(raw: Any) -> Optional[dict]:
        data = _parse_jsonb(raw)
        if not data or not isinstance(data, dict):
            return None
        tags = data.get("regime_tags") or data.get("tags") or []
        return {
            "trend_tag": data.get("trend_tag"),
            "vol_tag": data.get("vol_tag"),
            "efficiency_tag": data.get("efficiency_tag"),
            "tags": tags,
        }

    return {
        "run_id": str(row["id"]),
        "workspace_id": str(workspace_id),
        "status": row["status"],
        "run_kind": row["run_kind"],
        "created_at": (row["created_at"].isoformat() if row["created_at"] else None),
        "started_at": (row["started_at"].isoformat() if row["started_at"] else None),
        "completed_at": (
            row["completed_at"].isoformat() if row["completed_at"] else None
        ),
        "strategy": {
            "entity_id": (
                str(row["strategy_entity_id"]) if row["strategy_entity_id"] else None
            ),
            "version_id": (
                str(row["strategy_version_id"]) if row["strategy_version_id"] else None
            ),
            "name": row["strategy_name"],
        },
        "dataset": dataset_meta,
        "params": params,
        "summary": summary,
        "equity": equity_points,
        "drawdown": drawdown,
        "trades": trades,
        "trade_count": row["trade_count"] or len(trades),
        "warnings": _parse_jsonb(row["warnings"]) or [],
        "regime_is": _regime_tags(row["regime_is"]),
        "regime_oos": _regime_tags(row["regime_oos"]),
    }
