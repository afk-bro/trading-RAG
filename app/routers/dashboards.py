"""Read-only dashboard endpoints for trust-building visualizations."""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.lifespan import get_db_pool
from app.deps.security import check_workspace_consistency
from app.repositories.dashboards import DashboardsRepository
from app.repositories.run_events import RunEventsRepository
from app.repositories.trade_events import EventFilters, TradeEventsRepository
from app.schemas import TradeEventType
from app.services.backtest.loss_attribution import compute_loss_attribution
from app.services.backtest.process_score import compute_process_score
from app.services.backtest.run_lineage import (
    build_trajectory,
    compute_comparison_warnings,
    compute_deltas,
    compute_param_diffs,
    find_lineage_candidates,
    find_previous_run,
    find_run_by_id,
)

_log = structlog.get_logger(__name__)

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
    repo = DashboardsRepository(pool)
    rows = await repo.get_equity_curve(workspace_id, days)

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
    repo = DashboardsRepository(pool)
    rows = await repo.get_intel_timeline(workspace_id, days, version_id=version_id)

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
    repo = DashboardsRepository(pool)
    rows = await repo.get_alerts(workspace_id, days, include_resolved=include_resolved)

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
    repo = DashboardsRepository(pool)
    equity_row = await repo.get_summary_equity(workspace_id)
    intel_rows = await repo.get_summary_intel(workspace_id)
    alert_rows = await repo.get_summary_alert_counts(workspace_id)

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
    include_coaching: bool = Query(False),
    baseline_run_id: Optional[UUID] = Query(None),
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
    repo = DashboardsRepository(pool)
    row = await repo.get_backtest_run(run_id, workspace_id)

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

    result = {
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

    # Coaching data (only for completed runs when requested)
    if include_coaching and row["status"] == "completed":
        coaching = await _build_coaching_data(
            pool=pool,
            workspace_id=workspace_id,
            run_id=run_id,
            row=row,
            trades=trades,
            summary=summary,
            params=params,
            dataset_meta=dataset_meta,
            regime_is=_parse_jsonb(row["regime_is"]),
            regime_oos=_parse_jsonb(row["regime_oos"]),
            baseline_run_id=baseline_run_id,
        )
        if coaching:
            result.update(coaching)

    return result


async def _build_coaching_data(
    *,
    pool: Any,
    workspace_id: UUID,
    run_id: UUID,
    row: Any,
    trades: list[dict],
    summary: dict,
    params: dict,
    dataset_meta: dict,
    regime_is: Any,
    regime_oos: Any,
    baseline_run_id: Optional[UUID],
) -> dict[str, Any]:
    """Build coaching + trajectory data with a 500ms timeout budget."""
    budget_start = time.monotonic()
    BUDGET_MS = 500

    strategy_entity_id = row["strategy_entity_id"]
    completed_at = row["completed_at"]

    if not strategy_entity_id or not completed_at:
        return {}

    # Phase 1: Parallel async I/O
    events_repo = RunEventsRepository(pool)

    async def _fetch_events():
        return await events_repo.get_events(run_id, workspace_id)

    async def _fetch_previous():
        if baseline_run_id:
            return await find_run_by_id(pool, baseline_run_id, workspace_id)
        return await find_previous_run(
            pool, workspace_id, strategy_entity_id, run_id, completed_at
        )

    async def _fetch_trajectory():
        return await build_trajectory(pool, workspace_id, strategy_entity_id, limit=10)

    # Use asyncio.wait so completed tasks return even if others timeout
    coaching_partial = False
    task_events = asyncio.create_task(_fetch_events())
    task_previous = asyncio.create_task(_fetch_previous())
    task_trajectory = asyncio.create_task(_fetch_trajectory())
    all_tasks = [task_events, task_previous, task_trajectory]

    done, pending = await asyncio.wait(all_tasks, timeout=0.4)
    for t in pending:
        t.cancel()
    if pending:
        coaching_partial = True
        _log.warning(
            "coaching_io_partial_timeout",
            run_id=str(run_id),
            pending_count=len(pending),
        )

    def _safe_result(task, default=None):
        if task in done:
            try:
                return task.result()
            except Exception:
                return default
        return default

    events_raw = _safe_result(task_events)
    previous_raw = _safe_result(task_previous)
    trajectory_runs = _safe_result(task_trajectory, default=[])

    # Phase 2: CPU-bound compute
    elapsed = (time.monotonic() - budget_start) * 1000

    # Build lineage
    lineage: dict[str, Any] = {
        "previous_run_id": None,
        "previous_completed_at": None,
        "deltas": [],
        "params_changed": False,
        "param_diffs": {},
        "comparison_warnings": [],
    }

    if previous_raw:
        prev_summary = (
            previous_raw.summary
            if hasattr(previous_raw, "summary")
            else previous_raw.get("summary", {})
        )
        prev_run_id = (
            previous_raw.run_id
            if hasattr(previous_raw, "run_id")
            else previous_raw.get("run_id")
        )
        prev_completed = (
            previous_raw.completed_at
            if hasattr(previous_raw, "completed_at")
            else previous_raw.get("completed_at")
        )
        prev_params = (
            previous_raw.get("params", {}) if isinstance(previous_raw, dict) else {}
        )
        prev_dataset = (
            previous_raw.get("dataset", {}) if isinstance(previous_raw, dict) else {}
        )
        prev_run_kind = (
            previous_raw.get("run_kind", "") if isinstance(previous_raw, dict) else ""
        )

        lineage["previous_run_id"] = prev_run_id
        lineage["previous_completed_at"] = prev_completed

        if prev_summary:
            deltas = compute_deltas(summary, prev_summary)
            lineage["deltas"] = [
                {
                    "metric": d.metric,
                    "current": d.current,
                    "previous": d.previous,
                    "delta": d.delta,
                    "improved": d.improved,
                    "higher_is_better": d.higher_is_better,
                }
                for d in deltas
            ]

        if prev_params:
            pdiffs = compute_param_diffs(params, prev_params)
            lineage["params_changed"] = bool(pdiffs)
            lineage["param_diffs"] = pdiffs

        # Comparison warnings
        current_run_info = {
            "dataset": dataset_meta,
            "run_kind": row["run_kind"],
        }
        prev_run_info = {
            "dataset": prev_dataset,
            "run_kind": prev_run_kind,
        }
        lineage["comparison_warnings"] = compute_comparison_warnings(
            current_run_info, prev_run_info
        )

    # Process score and loss attribution (skip if budget exceeded)
    process_score_data: Optional[dict] = None
    loss_attr_data: Optional[dict] = None

    elapsed = (time.monotonic() - budget_start) * 1000
    if elapsed >= BUDGET_MS:
        coaching_partial = True
    if elapsed < BUDGET_MS:
        try:
            ps = compute_process_score(
                trades=trades,
                events=events_raw,
                regime_is=regime_is,
                regime_oos=regime_oos,
                summary=summary,
            )
            process_score_data = {
                "total": ps.total,
                "grade": ps.grade,
                "components": [
                    {
                        "name": c.name,
                        "score": c.score,
                        "weight": c.weight,
                        "detail": c.detail,
                        "available": c.available,
                    }
                    for c in ps.components
                ],
            }
        except Exception:
            _log.exception("process_score_error", run_id=str(run_id))
            process_score_data = {
                "total": None,
                "grade": "unavailable",
                "components": [],
            }

    elapsed = (time.monotonic() - budget_start) * 1000
    if elapsed >= BUDGET_MS:
        coaching_partial = True
    if elapsed < BUDGET_MS:
        try:
            la = compute_loss_attribution(
                trades=trades,
                regime_is=regime_is,
                regime_oos=regime_oos,
                events=events_raw,
            )
            loss_attr_data = {
                "time_clusters": [
                    {
                        "label": c.label,
                        "trade_count": c.trade_count,
                        "total_loss": c.total_loss,
                        "pct_of_total_losses": c.pct_of_total_losses,
                    }
                    for c in la.time_clusters
                ],
                "size_clusters": [
                    {
                        "label": c.label,
                        "trade_count": c.trade_count,
                        "total_loss": c.total_loss,
                        "pct_of_total_losses": c.pct_of_total_losses,
                    }
                    for c in la.size_clusters
                ],
                "regime_summary": (
                    {
                        "regime_tags": la.regime_summary.regime_tags,
                        "loss_count": la.regime_summary.loss_count,
                        "context": la.regime_summary.context,
                    }
                    if la.regime_summary
                    else None
                ),
                "counterfactuals": [
                    {
                        "description": cf.description,
                        "metric_name": cf.metric_name,
                        "actual": cf.actual,
                        "hypothetical": cf.hypothetical,
                        "delta": cf.delta,
                    }
                    for cf in la.counterfactuals
                ],
                "total_losses": la.total_losses,
                "total_loss_amount": la.total_loss_amount,
            }
        except Exception:
            _log.exception("loss_attribution_error", run_id=str(run_id))

    # Trajectory data
    trajectory_data = {
        "runs": [
            {
                "run_id": r.run_id,
                "completed_at": r.completed_at,
                "sharpe": r.sharpe,
                "return_pct": r.return_pct,
                "max_drawdown_pct": r.max_drawdown_pct,
                "win_rate": r.win_rate,
                "trades": r.trades,
            }
            for r in trajectory_runs
        ]
    }

    coaching: dict[str, Any] = {"lineage": lineage}
    if process_score_data is not None:
        coaching["process_score"] = process_score_data
    else:
        coaching_partial = True
        coaching["process_score"] = {
            "total": None,
            "grade": "timed_out",
            "components": [],
        }
    if loss_attr_data is not None:
        coaching["loss_attribution"] = loss_attr_data
    else:
        coaching_partial = True
        coaching["loss_attribution"] = {"timed_out": True}

    coaching["coaching_partial"] = coaching_partial

    return {
        "coaching": coaching,
        "trajectory": trajectory_data,
    }


# =============================================================================
# Lineage Candidates Endpoint
# =============================================================================


@router.get("/{workspace_id}/backtests/{run_id}/lineage")
async def get_run_lineage(
    workspace_id: UUID,
    run_id: UUID,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """Return recent runs for the same strategy — powers baseline selector."""
    repo = DashboardsRepository(pool)
    row = await repo.get_backtest_run_strategy(run_id, workspace_id)

    if not row:
        raise HTTPException(status_code=404, detail="Backtest run not found")

    strategy_entity_id = row["strategy_entity_id"]
    if not strategy_entity_id:
        return {"candidates": []}

    candidates = await find_lineage_candidates(
        pool, workspace_id, strategy_entity_id, run_id, limit
    )

    return {
        "candidates": [
            {
                "run_id": c.run_id,
                "completed_at": c.completed_at,
                "sharpe": c.sharpe,
                "return_pct": c.return_pct,
                "is_auto_baseline": c.is_auto_baseline,
            }
            for c in candidates
        ]
    }
