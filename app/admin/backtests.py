"""Backtest Tune admin endpoints."""

import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for backtests routes."""
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


def _get_tune_repo():
    """Get TuneRepository instance."""
    from app.repositories.backtests import TuneRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return TuneRepository(_db_pool)


def _get_run_repo():
    """Get BacktestRepository instance."""
    from app.repositories.backtests import BacktestRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return BacktestRepository(_db_pool)


@router.get("/backtests/tunes", response_class=HTMLResponse)
async def admin_tunes_list(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    valid_only: bool = Query(False, description="Only show valid tunes"),
    objective_type: Optional[str] = Query(None, description="Filter by objective type"),
    oos_enabled: Optional[str] = Query(
        None, description="Filter by OOS: 'true', 'false', or empty"
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List parameter tuning sessions."""
    tune_repo = _get_tune_repo()

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
            "tunes_list.html",
            {
                "request": request,
                "tunes": [],
                "total": 0,
                "workspace_id": None,
                "error": "No workspace found.",
            },
        )

    # Convert oos_enabled string to bool (query params come as strings)
    oos_enabled_bool = None
    if oos_enabled == "true":
        oos_enabled_bool = True
    elif oos_enabled == "false":
        oos_enabled_bool = False

    tunes, total = await tune_repo.list_tunes(
        workspace_id=workspace_id,
        status=status,
        valid_only=valid_only,
        objective_type=objective_type if objective_type else None,
        oos_enabled=oos_enabled_bool,
        limit=limit,
        offset=offset,
    )

    # Get counts for each tune
    enriched_tunes = []
    for tune in tunes:
        counts = await tune_repo.get_tune_status_counts(tune["id"])

        # Parse best_params if needed
        best_params = tune.get("best_params")
        if isinstance(best_params, str):
            try:
                best_params = json.loads(best_params)
            except json.JSONDecodeError:
                best_params = None

        tune["counts"] = counts
        tune["best_params"] = best_params
        enriched_tunes.append(tune)

    return templates.TemplateResponse(
        "tunes_list.html",
        {
            "request": request,
            "tunes": enriched_tunes,
            "total": total,
            "workspace_id": str(workspace_id),
            "status_filter": status or "",
            "valid_only": valid_only,
            "objective_type_filter": objective_type or "",
            "oos_enabled_filter": oos_enabled or "",
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


@router.get("/backtests/leaderboard")
async def admin_leaderboard(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace UUID"),
    valid_only: bool = Query(
        True, description="Only tunes with valid results (default True)"
    ),
    include_canceled: bool = Query(False, description="Include canceled tunes"),
    objective_type: Optional[str] = Query(None, description="Filter by objective type"),
    oos_enabled: Optional[str] = Query(
        None, description="Filter by OOS: 'true' or 'false'"
    ),
    format: Optional[str] = Query(
        None, description="Output format: 'csv' for download"
    ),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """Global leaderboard: best tunes ranked by objective score."""
    tune_repo = _get_tune_repo()

    # Parse oos_enabled filter
    oos_enabled_bool = None
    if oos_enabled is not None:
        oos_enabled_bool = oos_enabled.lower() == "true"

    entries, total = await tune_repo.get_leaderboard(
        workspace_id=workspace_id,
        valid_only=valid_only,
        objective_type=objective_type,
        oos_enabled=oos_enabled_bool,
        include_canceled=include_canceled,
        limit=limit,
        offset=offset,
    )

    # CSV Export
    if format == "csv":
        return _generate_leaderboard_csv(
            entries,
            offset,
            workspace_id=str(workspace_id)[:8],
            objective_type=objective_type,
        )

    # Convert and enrich entries for template
    enriched_entries = []
    for entry in entries:
        e = dict(entry)
        e["tune_id"] = str(e["id"])
        e["strategy_entity_id"] = str(e["strategy_entity_id"])
        if e.get("best_run_id"):
            e["best_run_id"] = str(e["best_run_id"])

        # Parse gates snapshot if present
        if e.get("gates") and isinstance(e["gates"], dict):
            pass  # Already a dict
        elif e.get("gates") and isinstance(e["gates"], str):
            try:
                e["gates"] = json.loads(e["gates"])
            except json.JSONDecodeError:
                e["gates"] = None

        # Parse best_metrics_oos to object for template
        if e.get("best_metrics_oos"):

            class MetricsObj:
                def __init__(self, d):
                    self.return_pct = d.get("return_pct")
                    self.sharpe = d.get("sharpe")
                    self.max_drawdown_pct = d.get("max_drawdown_pct")
                    self.trades = d.get("trades")

            e["best_metrics_oos"] = MetricsObj(e["best_metrics_oos"])

        enriched_entries.append(e)

    return templates.TemplateResponse(
        "leaderboard.html",
        {
            "request": request,
            "entries": enriched_entries,
            "total": total,
            "workspace_id": str(workspace_id),
            "valid_only": valid_only,
            "include_canceled": include_canceled,
            "objective_type_filter": objective_type or "",
            "oos_enabled_filter": oos_enabled or "",
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


def _generate_leaderboard_csv(
    entries: list[dict],
    offset: int = 0,
    workspace_id: str = "",
    objective_type: Optional[str] = None,
) -> StreamingResponse:
    """Generate CSV export of leaderboard entries."""
    output = io.StringIO()
    writer = csv.writer(output)

    # CSV columns as specified
    headers = [
        # Core identifiers
        "rank",
        "tune_id",
        "created_at",
        "status",
        "strategy_entity_id",
        "strategy_name",
        # Config snapshot
        "objective_type",
        "objective_params",
        "oos_ratio",
        "gates_max_drawdown_pct",
        "gates_min_trades",
        "gates_evaluated_on",
        # Winner fields
        "best_run_id",
        "best_params",
        "best_objective_score",
        "best_score",
        # OOS metrics
        "return_pct",
        "sharpe",
        "max_drawdown_pct",
        "trades",
        "profit_factor",
        # Robustness
        "overfit_gap",
    ]
    writer.writerow(headers)

    for idx, entry in enumerate(entries):
        # Parse JSONB fields if needed
        gates = entry.get("gates") or {}
        if isinstance(gates, str):
            try:
                gates = json.loads(gates)
            except json.JSONDecodeError:
                gates = {}

        metrics_oos = entry.get("best_metrics_oos") or {}
        if isinstance(metrics_oos, str):
            try:
                metrics_oos = json.loads(metrics_oos)
            except json.JSONDecodeError:
                metrics_oos = {}

        objective_params = entry.get("objective_params")
        if isinstance(objective_params, dict):
            objective_params = json.dumps(objective_params)

        best_params = entry.get("best_params")
        if isinstance(best_params, dict):
            best_params = json.dumps(best_params)

        # Compute overfit gap
        score_is = entry.get("score_is")
        score_oos = entry.get("score_oos")
        overfit_gap = None
        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)

        row = [
            offset + idx + 1,  # rank (1-indexed)
            str(entry.get("id", "")),
            entry.get("created_at", "").isoformat() if entry.get("created_at") else "",
            entry.get("status", ""),
            str(entry.get("strategy_entity_id", "")),
            entry.get("strategy_name", ""),
            entry.get("objective_type", "sharpe"),
            objective_params or "",
            entry.get("oos_ratio", ""),
            gates.get("max_drawdown_pct", ""),
            gates.get("min_trades", ""),
            gates.get("evaluated_on", ""),
            str(entry.get("best_run_id", "")) if entry.get("best_run_id") else "",
            best_params or "",
            entry.get("best_objective_score", ""),
            entry.get("best_score", ""),
            metrics_oos.get("return_pct", ""),
            metrics_oos.get("sharpe", ""),
            metrics_oos.get("max_drawdown_pct", ""),
            metrics_oos.get("trades", ""),
            metrics_oos.get("profit_factor", ""),
            overfit_gap if overfit_gap is not None else "",
        ]
        writer.writerow(row)

    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Build descriptive filename
    parts = ["leaderboard"]
    if workspace_id:
        parts.append(workspace_id)
    if objective_type:
        parts.append(objective_type)
    parts.append(timestamp)
    filename = "_".join(parts) + ".csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# =============================================================================
# Tune Compare helpers
# =============================================================================


def _normalize_compare_value(value: Any, fmt: str = "default") -> str:
    """Normalize value for display and comparison."""
    if value is None:
        return "—"
    if fmt == "pct":
        return f"{value:+.2f}%" if isinstance(value, (int, float)) else str(value)
    if fmt == "pct_neg":
        # For drawdown (already negative or should show as negative)
        v = -abs(value) if isinstance(value, (int, float)) else value
        return f"{v:.1f}%"
    if fmt == "float2":
        return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    if fmt == "float4":
        return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
    if fmt == "int":
        return str(int(value)) if isinstance(value, (int, float)) else str(value)
    if fmt == "pct_ratio":
        return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else str(value)
    return str(value)


def _values_differ(values: list[str]) -> bool:
    """Check if normalized values differ across tunes."""
    non_missing = [v for v in values if v != "—"]
    if len(non_missing) <= 1:
        # All missing or only one has value = differ
        return len(set(values)) > 1
    return len(set(non_missing)) > 1


def _overfit_class(gap: Optional[float]) -> str:
    """CSS class for overfit gap severity."""
    if gap is None:
        return ""
    if gap < 0:
        return "overfit-good"  # OOS better than IS (rare but good)
    if gap <= 0.3:
        return ""  # Normal
    if gap <= 0.5:
        return "overfit-warning"
    return "overfit-danger"


class CompareField(BaseModel):
    """A single field in tune comparison."""

    label: str
    values: list[str]
    differs: bool = False


class CompareSection(BaseModel):
    """A section of fields in tune comparison."""

    title: str
    fields: list[CompareField]


async def _fetch_tune_for_compare(tune_id: UUID) -> Optional[dict[str, Any]]:
    """Fetch a tune's full detail for comparison."""
    tune_repo = _get_tune_repo()

    tune = await tune_repo.get_tune(tune_id)
    if not tune:
        return None

    result = dict(tune)

    # Parse JSONB fields
    for field in ["param_ranges", "gates", "best_params"]:
        if result.get(field) and isinstance(result[field], str):
            try:
                result[field] = json.loads(result[field])
            except json.JSONDecodeError:
                pass

    # Get best run info
    best_run = await tune_repo.get_best_run(tune_id)
    if best_run:
        result["best_run"] = dict(best_run)
        for field in ["params", "metrics_is", "metrics_oos"]:
            if result["best_run"].get(field) and isinstance(
                result["best_run"][field], str
            ):
                try:
                    result["best_run"][field] = json.loads(result["best_run"][field])
                except json.JSONDecodeError:
                    pass

    # Get counts
    result["counts"] = await tune_repo.get_tune_status_counts(tune_id)

    return result


@router.get("/backtests/compare")
async def admin_tune_compare(
    request: Request,
    tune_id: list[UUID] = Query(..., description="Tune IDs to compare (2-5)"),
    format: Optional[str] = Query(None, description="Output format: 'json' for export"),
    _: bool = Depends(require_admin_token),
):
    """Compare multiple tunes side-by-side (N-way diff)."""
    if len(tune_id) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 tune IDs required",
        )
    if len(tune_id) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 5 tunes can be compared",
        )

    # Fetch all tunes
    tunes = []
    for tid in tune_id:
        tune = await _fetch_tune_for_compare(tid)
        if tune:
            tunes.append(tune)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tune {tid} not found",
            )

    # Build comparison sections
    sections: list[CompareSection] = []

    # 1. Basic Info
    basic_fields = []

    # Strategy name
    strategy_names = [t.get("strategy_name", "—") for t in tunes]
    basic_fields.append(
        CompareField(
            label="Strategy",
            values=strategy_names,
            differs=_values_differ(strategy_names),
        )
    )

    # Status
    statuses = [t.get("status", "—") for t in tunes]
    basic_fields.append(
        CompareField(label="Status", values=statuses, differs=_values_differ(statuses))
    )

    # Created
    created = [_normalize_compare_value(t.get("created_at"), "datetime") for t in tunes]
    basic_fields.append(
        CompareField(label="Created", values=created, differs=_values_differ(created))
    )

    # Total runs
    total_runs = [str(t.get("counts", {}).get("total", 0)) for t in tunes]
    basic_fields.append(
        CompareField(
            label="Total Runs", values=total_runs, differs=_values_differ(total_runs)
        )
    )

    # Valid runs
    valid_runs = [str(t.get("counts", {}).get("valid", 0)) for t in tunes]
    basic_fields.append(
        CompareField(
            label="Valid Runs", values=valid_runs, differs=_values_differ(valid_runs)
        )
    )

    sections.append(CompareSection(title="Basic Info", fields=basic_fields))

    # 2. Configuration
    config_fields = []

    # Objective type
    obj_types = [t.get("objective_type", "sharpe") for t in tunes]
    config_fields.append(
        CompareField(
            label="Objective", values=obj_types, differs=_values_differ(obj_types)
        )
    )

    # OOS ratio
    oos_ratios = [_normalize_compare_value(t.get("oos_ratio"), "float") for t in tunes]
    config_fields.append(
        CompareField(
            label="OOS Ratio", values=oos_ratios, differs=_values_differ(oos_ratios)
        )
    )

    # Gates
    gates_dd = []
    gates_trades = []
    for t in tunes:
        gates = t.get("gates") or {}
        gates_dd.append(_normalize_compare_value(gates.get("max_drawdown_pct"), "pct"))
        gates_trades.append(str(gates.get("min_trades", "—")))
    config_fields.append(
        CompareField(
            label="Max Drawdown Gate",
            values=gates_dd,
            differs=_values_differ(gates_dd),
        )
    )
    config_fields.append(
        CompareField(
            label="Min Trades Gate",
            values=gates_trades,
            differs=_values_differ(gates_trades),
        )
    )

    sections.append(CompareSection(title="Configuration", fields=config_fields))

    # 3. Best Run Metrics
    metrics_fields = []

    # Return IS
    return_is = []
    for t in tunes:
        br = t.get("best_run") or {}
        mis = br.get("metrics_is") or {}
        return_is.append(_normalize_compare_value(mis.get("return_pct"), "pct"))
    metrics_fields.append(
        CompareField(
            label="Return % (IS)", values=return_is, differs=_values_differ(return_is)
        )
    )

    # Return OOS
    return_oos = []
    for t in tunes:
        br = t.get("best_run") or {}
        mos = br.get("metrics_oos") or {}
        return_oos.append(_normalize_compare_value(mos.get("return_pct"), "pct"))
    metrics_fields.append(
        CompareField(
            label="Return % (OOS)",
            values=return_oos,
            differs=_values_differ(return_oos),
        )
    )

    # Sharpe IS
    sharpe_is = []
    for t in tunes:
        br = t.get("best_run") or {}
        mis = br.get("metrics_is") or {}
        sharpe_is.append(_normalize_compare_value(mis.get("sharpe"), "float"))
    metrics_fields.append(
        CompareField(
            label="Sharpe (IS)", values=sharpe_is, differs=_values_differ(sharpe_is)
        )
    )

    # Sharpe OOS
    sharpe_oos = []
    for t in tunes:
        br = t.get("best_run") or {}
        mos = br.get("metrics_oos") or {}
        sharpe_oos.append(_normalize_compare_value(mos.get("sharpe"), "float"))
    metrics_fields.append(
        CompareField(
            label="Sharpe (OOS)", values=sharpe_oos, differs=_values_differ(sharpe_oos)
        )
    )

    # Max DD IS
    dd_is = []
    for t in tunes:
        br = t.get("best_run") or {}
        mis = br.get("metrics_is") or {}
        dd_is.append(_normalize_compare_value(mis.get("max_drawdown_pct"), "pct"))
    metrics_fields.append(
        CompareField(label="Max DD (IS)", values=dd_is, differs=_values_differ(dd_is))
    )

    # Max DD OOS
    dd_oos = []
    for t in tunes:
        br = t.get("best_run") or {}
        mos = br.get("metrics_oos") or {}
        dd_oos.append(_normalize_compare_value(mos.get("max_drawdown_pct"), "pct"))
    metrics_fields.append(
        CompareField(
            label="Max DD (OOS)", values=dd_oos, differs=_values_differ(dd_oos)
        )
    )

    # Trades
    trades_oos = []
    for t in tunes:
        br = t.get("best_run") or {}
        mos = br.get("metrics_oos") or {}
        trades_oos.append(str(mos.get("trades", "—")))
    metrics_fields.append(
        CompareField(
            label="Trades (OOS)", values=trades_oos, differs=_values_differ(trades_oos)
        )
    )

    sections.append(CompareSection(title="Best Run Metrics", fields=metrics_fields))

    # 4. Parameter comparison
    all_param_keys: set[str] = set()
    for t in tunes:
        br = t.get("best_run") or {}
        params = br.get("params") or {}
        all_param_keys.update(params.keys())

    if all_param_keys:
        param_fields = []
        for key in sorted(all_param_keys):
            values = []
            for t in tunes:
                br = t.get("best_run") or {}
                params = br.get("params") or {}
                values.append(_normalize_compare_value(params.get(key)))
            param_fields.append(
                CompareField(label=key, values=values, differs=_values_differ(values))
            )
        sections.append(CompareSection(title="Best Parameters", fields=param_fields))

    # 5. Overfit analysis
    overfit_fields = []

    # Score IS vs OOS
    score_is_vals = []
    score_oos_vals = []
    overfit_gaps = []
    for t in tunes:
        br = t.get("best_run") or {}
        s_is = br.get("objective_score_is")
        s_oos = br.get("objective_score_oos")
        score_is_vals.append(_normalize_compare_value(s_is, "float"))
        score_oos_vals.append(_normalize_compare_value(s_oos, "float"))
        if s_is is not None and s_oos is not None:
            gap = round(s_is - s_oos, 4)
            overfit_gaps.append(f"{gap:.4f}")
        else:
            overfit_gaps.append("—")

    overfit_fields.append(
        CompareField(
            label="Score (IS)",
            values=score_is_vals,
            differs=_values_differ(score_is_vals),
        )
    )
    overfit_fields.append(
        CompareField(
            label="Score (OOS)",
            values=score_oos_vals,
            differs=_values_differ(score_oos_vals),
        )
    )
    overfit_fields.append(
        CompareField(
            label="Overfit Gap",
            values=overfit_gaps,
            differs=_values_differ(overfit_gaps),
        )
    )

    sections.append(CompareSection(title="Overfit Analysis", fields=overfit_fields))

    # JSON export
    if format == "json":
        export_data = {
            "tune_ids": [str(tid) for tid in tune_id],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "sections": [s.model_dump() for s in sections],
        }
        return JSONResponse(content=export_data)

    # HTML response
    return templates.TemplateResponse(
        "tune_compare.html",
        {
            "request": request,
            "tunes": tunes,
            "tune_ids": [str(tid) for tid in tune_id],
            "sections": sections,
        },
    )


# ===========================================
# Regime Backfill Endpoint
# ===========================================


class BackfillRegimeRequest(BaseModel):
    """Request to trigger tune regime attribution backfill."""

    workspace_id: UUID
    dry_run: bool = True  # Default to dry run for safety
    limit: Optional[int] = None


class BackfillRegimeResponse(BaseModel):
    """Response from tune regime attribution backfill."""

    processed: int
    skipped: int
    errors: int
    dry_run: bool


@router.post(
    "/backtests/tunes/backfill-regime",
    response_model=BackfillRegimeResponse,
    responses={
        200: {"description": "Backfill completed successfully"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
        503: {"description": "Service unavailable"},
    },
    summary="Trigger tune regime attribution backfill",
    description="""
Backfill regime attribution for existing tune runs.

Computes and stores regime tags for tune runs that don't have them.
This is useful after adding new regime attribution logic.

**Admin-only endpoint.** Requires X-Admin-Token header.

Options:
- `dry_run` (default: true): Preview without writing changes
- `limit`: Maximum number of tunes to process (optional)
""",
)
async def backfill_tune_regime(
    request: BackfillRegimeRequest,
    _: bool = Depends(require_admin_token),
) -> BackfillRegimeResponse:
    """Trigger tune regime attribution backfill."""
    from app.jobs.backfill_tune_regime import BackfillTuneRegimeJob

    logger.info(
        "backfill_tune_regime_started",
        workspace_id=str(request.workspace_id),
        dry_run=request.dry_run,
        limit=request.limit,
    )

    try:
        job = BackfillTuneRegimeJob(pool=_db_pool)
        result = await job.run(
            workspace_id=request.workspace_id,
            dry_run=request.dry_run,
            limit=request.limit,
        )

        logger.info(
            "backfill_tune_regime_complete",
            workspace_id=str(request.workspace_id),
            processed=result.processed,
            skipped=len(result.skipped),
            errors=len(result.errors),
            dry_run=request.dry_run,
        )

        return BackfillRegimeResponse(
            processed=result.processed,
            skipped=len(result.skipped),
            errors=len(result.errors),
            dry_run=result.dry_run,
        )

    except Exception as e:
        logger.error(
            "backfill_tune_regime_failed",
            workspace_id=str(request.workspace_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Backfill failed: {str(e)}",
        )


@router.get("/backtests/tunes/{tune_id}", response_class=HTMLResponse)
async def admin_tune_detail(
    request: Request,
    tune_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """View tune details with runs."""
    tune_repo = _get_tune_repo()

    tune = await tune_repo.get_tune(tune_id)
    if not tune:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tune {tune_id} not found",
        )

    tune = dict(tune)

    # Parse JSONB fields
    for field in ["param_ranges", "gates", "best_params", "objective_params"]:
        if tune.get(field) and isinstance(tune[field], str):
            try:
                tune[field] = json.loads(tune[field])
            except json.JSONDecodeError:
                pass

    # Get runs
    runs, total = await tune_repo.list_tune_runs(
        tune_id=tune_id,
        limit=limit,
        offset=offset,
    )

    # Parse run JSONB fields
    enriched_runs = []
    for run in runs:
        run = dict(run)
        for field in ["params", "metrics_is", "metrics_oos"]:
            if run.get(field) and isinstance(run[field], str):
                try:
                    run[field] = json.loads(run[field])
                except json.JSONDecodeError:
                    pass
        enriched_runs.append(run)

    # Get counts
    counts = await tune_repo.get_tune_status_counts(tune_id)

    return templates.TemplateResponse(
        "tune_detail.html",
        {
            "request": request,
            "tune": tune,
            "runs": enriched_runs,
            "total": total,
            "counts": counts,
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


@router.get("/backtests/runs/{run_id}", response_class=HTMLResponse)
async def admin_backtest_run_detail(
    request: Request,
    run_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """View backtest run details."""
    run_repo = _get_run_repo()

    run = await run_repo.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id} not found",
        )

    run = dict(run)

    # Convert UUID fields to strings for template
    for field in [
        "id",
        "workspace_id",
        "strategy_entity_id",
        "strategy_spec_id",
        "tune_id",
    ]:
        if run.get(field) is not None:
            run[field] = str(run[field])

    # Parse JSONB fields
    for field in ["params", "metrics_is", "metrics_oos"]:
        if run.get(field) and isinstance(run[field], str):
            try:
                run[field] = json.loads(run[field])
            except json.JSONDecodeError:
                pass

    # Convert to JSON-serializable for debug panel
    run_json = _json_serializable(run)

    # Get admin token for API calls from template
    admin_token = request.headers.get("X-Admin-Token", "") or os.environ.get(
        "ADMIN_TOKEN", ""
    )

    return templates.TemplateResponse(
        "backtest_run_detail.html",
        {
            "request": request,
            "run": run,
            "run_json": json.dumps(run_json, indent=2),
            "admin_token": admin_token,
        },
    )
