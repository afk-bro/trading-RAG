"""Backtest Tune admin endpoints."""

import csv
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.admin.utils import (
    PaginationDefaults,
    json_serializable,
    parse_bool_param,
    parse_jsonb_fields,
    require_db_pool,
)
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


# =============================================================================
# MetricsObj - Module level class for metrics display
# =============================================================================


class MetricsObj:
    """Wrapper for metrics dict providing attribute access in templates."""

    def __init__(self, d: dict):
        self.return_pct = d.get("return_pct")
        self.sharpe = d.get("sharpe")
        self.max_drawdown_pct = d.get("max_drawdown_pct")
        self.trades = d.get("trades")
        self.profit_factor = d.get("profit_factor")


# =============================================================================
# Leaderboard CSV Formatter
# =============================================================================


class LeaderboardCSVFormatter:
    """Formats leaderboard data for CSV export."""

    HEADERS = [
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

    def _compute_overfit_gap(self, entry: dict) -> Optional[float]:
        """Compute overfit gap (IS - OOS score difference)."""
        score_is = entry.get("score_is")
        score_oos = entry.get("score_oos")
        if score_is is not None and score_oos is not None:
            return round(score_is - score_oos, 4)
        return None

    def format_row(self, entry: dict, rank: int) -> list:
        """Format a single leaderboard entry as a CSV row."""
        # Parse JSONB fields
        entry_copy = dict(entry)
        parse_jsonb_fields(entry_copy, ["gates", "best_metrics_oos"])

        gates = entry_copy.get("gates") or {}
        metrics_oos = entry_copy.get("best_metrics_oos") or {}

        # Format complex fields
        objective_params = entry.get("objective_params")
        if isinstance(objective_params, dict):
            objective_params = json.dumps(objective_params)

        best_params = entry.get("best_params")
        if isinstance(best_params, dict):
            best_params = json.dumps(best_params)

        overfit_gap = self._compute_overfit_gap(entry)

        return [
            rank,
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

    def generate(
        self,
        entries: list[dict],
        offset: int = 0,
        workspace_id: str = "",
        objective_type: Optional[str] = None,
    ) -> StreamingResponse:
        """Generate CSV export of leaderboard entries."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(self.HEADERS)

        for idx, entry in enumerate(entries):
            rank = offset + idx + 1
            writer.writerow(self.format_row(entry, rank))

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
# Compare Field Builder
# =============================================================================


class CompareField(BaseModel):
    """A single field in tune comparison."""

    label: str
    values: list[str]
    differs: bool = False


class CompareSection(BaseModel):
    """A section of fields in tune comparison."""

    title: str
    fields: list[CompareField]


class CompareFieldBuilder:
    """Builds comparison fields across multiple tunes."""

    def __init__(self, tunes: list[dict]):
        self.tunes = tunes
        self._fields: list[CompareField] = []

    def add_field(
        self,
        label: str,
        extractor: Callable[[dict], Any],
        format_spec: str = "default",
    ) -> "CompareFieldBuilder":
        """Add a comparison field.

        Args:
            label: Display label for the field
            extractor: Function to extract value from a tune dict
            format_spec: Format type (pct, float, float2, float4, int, pct_ratio, etc)

        Returns:
            Self for chaining
        """
        values = [
            _normalize_compare_value(extractor(t), format_spec) for t in self.tunes
        ]
        self._fields.append(
            CompareField(label=label, values=values, differs=_values_differ(values))
        )
        return self

    def add_metric_pair(
        self,
        label_prefix: str,
        metric_key: str,
        format_spec: str = "pct",
    ) -> "CompareFieldBuilder":
        """Add IS/OOS metric pair."""
        # IS metric
        self.add_field(
            f"{label_prefix} (IS)",
            lambda t, k=metric_key: (t.get("best_run") or {})  # type: ignore[misc]
            .get("metrics_is", {})
            .get(k),
            format_spec,
        )
        # OOS metric
        self.add_field(
            f"{label_prefix} (OOS)",
            lambda t, k=metric_key: (t.get("best_run") or {})  # type: ignore[misc]
            .get("metrics_oos", {})
            .get(k),
            format_spec,
        )
        return self

    def build(self) -> list[CompareField]:
        """Build and return the list of fields."""
        fields = self._fields
        self._fields = []
        return fields

    def build_section(self, title: str) -> CompareSection:
        """Build a section with current fields and reset."""
        return CompareSection(title=title, fields=self.build())


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_default_workspace_id() -> Optional[UUID]:
    """Get the first available workspace ID."""
    if _db_pool is None:
        return None
    try:
        async with _db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM workspaces LIMIT 1")
            if row:
                return row["id"]
    except Exception as e:
        logger.warning("Could not fetch default workspace", error=str(e))
    return None


def _get_tune_repo():
    """Get TuneRepository instance."""
    from app.repositories.backtests import TuneRepository

    require_db_pool(_db_pool)
    return TuneRepository(_db_pool)


def _get_run_repo():
    """Get BacktestRepository instance."""
    from app.repositories.backtests import BacktestRepository

    require_db_pool(_db_pool)
    return BacktestRepository(_db_pool)


def _get_wfo_repo():
    """Get WFO repository instance."""
    from app.repositories.backtests import WFORepository

    require_db_pool(_db_pool)
    return WFORepository(_db_pool)


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
    if fmt == "float":
        return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    if fmt == "float2":
        return f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
    if fmt == "float4":
        return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
    if fmt == "int":
        return str(int(value)) if isinstance(value, (int, float)) else str(value)
    if fmt == "pct_ratio":
        return f"{value * 100:.0f}%" if isinstance(value, (int, float)) else str(value)
    if fmt == "datetime":
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)
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


async def _fetch_tune_for_compare(tune_id: UUID) -> Optional[dict[str, Any]]:
    """Fetch a tune's full detail for comparison."""
    tune_repo = _get_tune_repo()

    tune = await tune_repo.get_tune(tune_id)
    if not tune:
        return None

    result = dict(tune)
    parse_jsonb_fields(result, ["param_ranges", "gates", "best_params"])

    # Get best run info
    best_run = await tune_repo.get_best_run(tune_id)
    if best_run:
        result["best_run"] = dict(best_run)
        parse_jsonb_fields(result["best_run"], ["params", "metrics_is", "metrics_oos"])

    # Get counts
    result["counts"] = await tune_repo.get_tune_status_counts(tune_id)

    return result


# =============================================================================
# Admin Endpoints
# =============================================================================


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
    limit: int = Query(
        PaginationDefaults.DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.MAX_LIMIT,
    ),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List parameter tuning sessions."""
    tune_repo = _get_tune_repo()

    # If no workspace specified, get first available
    if not workspace_id:
        workspace_id = await _get_default_workspace_id()

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

    oos_enabled_bool = parse_bool_param(oos_enabled)

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
        tune = dict(tune)
        counts = await tune_repo.get_tune_status_counts(tune["id"])
        parse_jsonb_fields(tune, ["best_params"])
        tune["counts"] = counts
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
    limit: int = Query(
        PaginationDefaults.LEADERBOARD_DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.LEADERBOARD_MAX_LIMIT,
    ),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """Global leaderboard: best tunes ranked by objective score."""
    tune_repo = _get_tune_repo()

    oos_enabled_bool = parse_bool_param(oos_enabled)

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
        formatter = LeaderboardCSVFormatter()
        return formatter.generate(
            entries,
            offset=offset,
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

        parse_jsonb_fields(e, ["gates"])

        # Parse best_metrics_oos to object for template
        if e.get("best_metrics_oos"):
            if isinstance(e["best_metrics_oos"], str):
                try:
                    e["best_metrics_oos"] = json.loads(e["best_metrics_oos"])
                except json.JSONDecodeError:
                    e["best_metrics_oos"] = {}
            e["best_metrics_oos"] = MetricsObj(e["best_metrics_oos"])

        enriched_entries.append(e)

    context = {
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
    }

    # HTMX partial response (filter/pagination without full page reload)
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("leaderboard_partial.html", context)

    return templates.TemplateResponse("leaderboard.html", context)


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

    # Build comparison sections using CompareFieldBuilder
    sections: list[CompareSection] = []
    builder = CompareFieldBuilder(tunes)

    # 1. Basic Info
    builder.add_field("Strategy", lambda t: t.get("strategy_name", "—"))
    builder.add_field("Status", lambda t: t.get("status", "—"))
    builder.add_field("Created", lambda t: t.get("created_at"), "datetime")
    builder.add_field(
        "Total Runs", lambda t: t.get("counts", {}).get("total", 0), "int"
    )
    builder.add_field(
        "Valid Runs", lambda t: t.get("counts", {}).get("valid", 0), "int"
    )
    sections.append(builder.build_section("Basic Info"))

    # 2. Configuration
    builder.add_field("Objective", lambda t: t.get("objective_type", "sharpe"))
    builder.add_field("OOS Ratio", lambda t: t.get("oos_ratio"), "float")
    builder.add_field(
        "Max Drawdown Gate",
        lambda t: (t.get("gates") or {}).get("max_drawdown_pct"),
        "pct",
    )
    builder.add_field(
        "Min Trades Gate",
        lambda t: (t.get("gates") or {}).get("min_trades"),
        "int",
    )
    sections.append(builder.build_section("Configuration"))

    # 3. Best Run Metrics
    builder.add_metric_pair("Return %", "return_pct", "pct")
    builder.add_metric_pair("Sharpe", "sharpe", "float")
    builder.add_metric_pair("Max DD", "max_drawdown_pct", "pct")
    builder.add_field(
        "Trades (OOS)",
        lambda t: (t.get("best_run") or {}).get("metrics_oos", {}).get("trades"),
        "int",
    )
    sections.append(builder.build_section("Best Run Metrics"))

    # 4. Parameter comparison
    all_param_keys: set[str] = set()
    for t in tunes:
        br = t.get("best_run") or {}
        params = br.get("params") or {}
        all_param_keys.update(params.keys())

    if all_param_keys:
        for key in sorted(all_param_keys):
            builder.add_field(
                key,
                lambda t, k=key: (t.get("best_run") or {})  # type: ignore[misc]
                .get("params", {})
                .get(k),
            )
        sections.append(builder.build_section("Best Parameters"))

    # 5. Overfit analysis
    builder.add_field(
        "Score (IS)",
        lambda t: (t.get("best_run") or {}).get("objective_score_is"),
        "float",
    )
    builder.add_field(
        "Score (OOS)",
        lambda t: (t.get("best_run") or {}).get("objective_score_oos"),
        "float",
    )

    # Compute overfit gaps
    overfit_gaps = []
    for t in tunes:
        br = t.get("best_run") or {}
        s_is = br.get("objective_score_is")
        s_oos = br.get("objective_score_oos")
        if s_is is not None and s_oos is not None:
            gap = round(s_is - s_oos, 4)
            overfit_gaps.append(f"{gap:.4f}")
        else:
            overfit_gaps.append("—")

    overfit_fields = builder.build()
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
# Run Compare Endpoint (for compare tray)
# ===========================================


@router.get("/backtests/compare-runs", response_class=HTMLResponse)
async def admin_run_compare(
    request: Request,
    run_ids: str = Query(..., description="Comma-separated run UUIDs"),
    workspace_id: Optional[UUID] = Query(None, description="Workspace UUID"),
    _: bool = Depends(require_admin_token),
):
    """Compare multiple backtest runs side-by-side."""
    # Parse run IDs
    try:
        run_id_list = [UUID(rid.strip()) for rid in run_ids.split(",") if rid.strip()]
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid run ID format: {e}",
        )

    if len(run_id_list) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 run IDs required",
        )
    if len(run_id_list) > 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 12 runs can be compared",
        )

    pool = require_db_pool(_db_pool)

    # Fetch run summaries
    runs = []
    async with pool.acquire() as conn:
        for rid in run_id_list:
            row = await conn.fetchrow(
                """
                SELECT
                    r.id, r.status, r.created_at,
                    r.strategy_entity_id,
                    r.params, r.summary, r.dataset_meta,
                    r.regime_is, r.regime_oos,
                    e.name as strategy_name
                FROM backtest_runs r
                LEFT JOIN kb_entities e ON r.strategy_entity_id = e.id
                WHERE r.id = $1
                """,
                rid,
            )
            if row:
                run = dict(row)
                # Parse JSONB fields
                parse_jsonb_fields(
                    run,
                    ["params", "summary", "dataset_meta", "regime_is", "regime_oos"],
                )
                runs.append(run)

    if len(runs) < 2:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Some runs were not found",
        )

    # Extract comparison data
    comparison_data = []
    for run in runs:
        summary = run.get("summary") or {}
        dataset = run.get("dataset_meta") or {}
        regime_oos = run.get("regime_oos") or {}
        regime_is = run.get("regime_is") or {}

        # Get regime tags
        regime = regime_oos or regime_is
        tags = regime.get("regime_tags", []) if regime else []
        trend_tag = next(
            (t for t in tags if t in ("uptrend", "downtrend", "flat", "ranging")), None
        )
        vol_tag = next((t for t in tags if t in ("high_vol", "low_vol")), None)

        comparison_data.append(
            {
                "id": str(run["id"]),
                "status": run.get("status", "unknown"),
                "strategy_name": run.get("strategy_name")
                or str(run["strategy_entity_id"])[:8],
                "symbol": dataset.get("symbol", "-"),
                "timeframe": dataset.get("timeframe", "-"),
                "return_pct": summary.get("return_pct"),
                "sharpe": summary.get("sharpe"),
                "max_drawdown_pct": summary.get("max_drawdown_pct"),
                "trades": summary.get("trades"),
                "profit_factor": summary.get("profit_factor"),
                "win_rate": summary.get("win_rate"),
                "trend_tag": trend_tag,
                "vol_tag": vol_tag,
                "created_at": run.get("created_at"),
            }
        )

    return templates.TemplateResponse(
        "backtests_compare.html",
        {
            "request": request,
            "runs": comparison_data,
            "run_ids": [str(rid) for rid in run_id_list],
            "workspace_id": str(workspace_id) if workspace_id else None,
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
    limit: int = Query(
        PaginationDefaults.DETAIL_DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.DETAIL_MAX_LIMIT,
    ),
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
    parse_jsonb_fields(
        tune, ["param_ranges", "gates", "best_params", "objective_params"]
    )

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
        parse_jsonb_fields(run, ["params", "metrics_is", "metrics_oos"])
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

    # Convert UUID fields to strings for template (but keep datetime as-is for strftime)
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
    parse_jsonb_fields(run, ["params", "metrics_is", "metrics_oos"])

    # Convert to JSON-serializable for debug panel
    run_json = json_serializable(run)

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


# ===========================================
# WFO Admin Endpoints
# ===========================================


@router.get("/backtests/wfo", response_class=HTMLResponse)
async def admin_wfo_list(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    status_filter: Optional[str] = Query(
        None, alias="status", description="Filter by status"
    ),
    limit: int = Query(
        PaginationDefaults.DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.MAX_LIMIT,
    ),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List Walk-Forward Optimization runs."""
    wfo_repo = _get_wfo_repo()

    # If no workspace specified, get first available
    if not workspace_id:
        workspace_id = await _get_default_workspace_id()

    if not workspace_id:
        return templates.TemplateResponse(
            "wfo_list.html",
            {
                "request": request,
                "wfos": [],
                "total": 0,
                "workspace_id": None,
                "error": "No workspace found.",
            },
        )

    wfos, total = await wfo_repo.list_wfos(
        workspace_id=workspace_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    # Enrich WFO data
    enriched_wfos = []
    for wfo in wfos:
        wfo = dict(wfo)
        parse_jsonb_fields(
            wfo, ["wfo_config", "data_source", "best_params", "best_candidate"]
        )
        enriched_wfos.append(wfo)

    return templates.TemplateResponse(
        "wfo_list.html",
        {
            "request": request,
            "wfos": enriched_wfos,
            "total": total,
            "workspace_id": str(workspace_id),
            "status_filter": status_filter or "",
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


@router.get("/backtests/wfo/{wfo_id}", response_class=HTMLResponse)
async def admin_wfo_detail(
    request: Request,
    wfo_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """View WFO run details with fold selector and candidate comparison."""
    wfo_repo = _get_wfo_repo()

    wfo = await wfo_repo.get_wfo(wfo_id)
    if not wfo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"WFO run {wfo_id} not found",
        )

    wfo = dict(wfo)

    # Convert UUID fields to strings (but keep datetime as-is for strftime)
    for field in ["id", "workspace_id", "strategy_entity_id", "job_id"]:
        if wfo.get(field) is not None:
            wfo[field] = str(wfo[field])

    # Parse JSONB fields
    parse_jsonb_fields(
        wfo,
        [
            "wfo_config",
            "data_source",
            "param_space",
            "best_params",
            "best_candidate",
            "candidates",
        ],
    )

    # Get child tune IDs
    child_tune_ids = [str(tid) for tid in (wfo.get("child_tune_ids") or [])]
    wfo["child_tune_ids"] = child_tune_ids

    # Get strategy name
    strategy_name = None
    if wfo.get("strategy_entity_id") and _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT name FROM strategies WHERE strategy_entity_id = $1",
                    UUID(wfo["strategy_entity_id"]),
                )
                if row:
                    strategy_name = row["name"]
        except Exception as e:
            logger.warning("Could not fetch strategy name", error=str(e))

    wfo["strategy_name"] = strategy_name

    # Convert to JSON-serializable for debug panel
    wfo_json = json_serializable(wfo)

    # Get admin token for API calls from template
    admin_token = request.headers.get("X-Admin-Token", "") or os.environ.get(
        "ADMIN_TOKEN", ""
    )

    return templates.TemplateResponse(
        "wfo_detail.html",
        {
            "request": request,
            "wfo": wfo,
            "wfo_json": json.dumps(wfo_json, indent=2),
            "admin_token": admin_token,
        },
    )
