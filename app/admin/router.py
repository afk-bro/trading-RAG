"""Admin UI router for KB inspection and curation."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog


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


import csv  # noqa: E402
import io  # noqa: E402

from fastapi import (  # noqa: E402
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)  # noqa: E402
from fastapi.responses import (  # noqa: E402
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.deps.security import require_admin_token  # noqa: E402
from app.schemas import KBEntityType, KBClaimType  # noqa: E402
from app.admin import analytics as analytics_router  # noqa: E402
from app.admin import alerts as alerts_router  # noqa: E402
from app.admin import coverage as coverage_router  # noqa: E402
from app.admin import retention as retention_router  # noqa: E402
from app.admin import events as events_router  # noqa: E402

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger(__name__)

# Include analytics sub-router
router.include_router(analytics_router.router)

# Include alerts sub-router
router.include_router(alerts_router.router)

# Include coverage sub-router
router.include_router(coverage_router.router)

# Include retention sub-router
router.include_router(retention_router.router)

# Include events (SSE) sub-router
router.include_router(events_router.router)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for admin routes."""
    global _db_pool
    _db_pool = pool
    # Also set pool for analytics router
    analytics_router.set_db_pool(pool)
    # Also set pool for alerts router
    alerts_router.set_db_pool(pool)
    # Also set pool for coverage router
    coverage_router.set_db_pool(pool)
    # Also set pool for retention router
    retention_router.set_db_pool(pool)


def _get_kb_repo():
    """Get KnowledgeBaseRepository instance."""
    from app.repositories.kb import KnowledgeBaseRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return KnowledgeBaseRepository(_db_pool)


def _get_run_plans_repo():
    """Get RunPlansRepository instance."""
    from app.repositories.run_plans import RunPlansRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return RunPlansRepository(_db_pool)


# ===========================================
# Admin Routes
# ===========================================
# Note: All routes use require_admin_token from app.deps.security
# which provides constant-time token comparison and no debug bypass.


@router.get("/", response_class=HTMLResponse)
async def admin_home(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """Admin home page - redirect to KB entities."""
    return RedirectResponse(url="/admin/kb/entities", status_code=302)


@router.get("/kb/entities", response_class=HTMLResponse)
async def admin_entities(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    q: Optional[str] = Query(None, description="Search query"),
    type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List KB entities with search and filters."""
    kb_repo = _get_kb_repo()

    # If no workspace specified, get first available
    if not workspace_id:
        # Try to get workspaces from DB
        try:
            async with _db_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT id FROM workspaces LIMIT 1")
                if row:
                    workspace_id = row["id"]
        except Exception as e:
            logger.warning("Could not fetch default workspace", error=str(e))

    if not workspace_id:
        return templates.TemplateResponse(
            "entities.html",
            {
                "request": request,
                "entities": [],
                "total": 0,
                "workspace_id": None,
                "q": q,
                "type": type,
                "limit": limit,
                "offset": offset,
                "entity_types": [e.value for e in KBEntityType],
                "error": "No workspace found. Create a workspace first.",
            },
        )

    # Convert type string to enum if provided
    entity_type = None
    if type:
        try:
            from app.services.kb_types import EntityType

            entity_type = EntityType(type)
        except ValueError:
            pass

    entities, total = await kb_repo.list_entities(
        workspace_id=workspace_id,
        q=q,
        entity_type=entity_type,
        limit=limit,
        offset=offset,
        include_counts=True,
    )

    # Parse aliases for display
    import json

    for entity in entities:
        if isinstance(entity.get("aliases"), str):
            try:
                entity["aliases"] = json.loads(entity["aliases"])
            except json.JSONDecodeError:
                entity["aliases"] = []

    return templates.TemplateResponse(
        "entities.html",
        {
            "request": request,
            "entities": entities,
            "total": total,
            "workspace_id": str(workspace_id),
            "q": q or "",
            "type": type or "",
            "limit": limit,
            "offset": offset,
            "entity_types": [e.value for e in KBEntityType],
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


@router.get("/kb/entities/{entity_id}", response_class=HTMLResponse)
async def admin_entity_detail(
    request: Request,
    entity_id: UUID,
    status_filter: Optional[str] = Query("verified", description="Claim status filter"),
    claim_type: Optional[str] = Query(None, description="Claim type filter"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """View entity details with claims."""
    kb_repo = _get_kb_repo()

    # Get entity
    entity = await kb_repo.get_entity_by_id(entity_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found",
        )

    # Find possible duplicates
    possible_duplicates = await kb_repo.find_possible_duplicates(
        entity_id=entity_id,
        workspace_id=entity["workspace_id"],
        limit=5,
    )

    # Parse aliases
    if isinstance(entity.get("aliases"), str):
        try:
            entity["aliases"] = json.loads(entity["aliases"])
        except json.JSONDecodeError:
            entity["aliases"] = []

    # Get claims for this entity
    from app.services.kb_types import ClaimType

    internal_claim_type = None
    if claim_type:
        try:
            internal_claim_type = ClaimType(claim_type)
        except ValueError:
            pass

    # Map "all" status to None (no filter)
    status_value = status_filter if status_filter != "all" else None

    claims, total_claims = await kb_repo.list_claims(
        workspace_id=entity["workspace_id"],
        entity_id=entity_id,
        status=status_value,
        claim_type=internal_claim_type,
        limit=limit,
        offset=offset,
    )

    # Build StrategySpec for strategy entities
    strategy_spec = None
    strategy_spec_status = None
    strategy_spec_version = None
    strategy_spec_approved_by = None
    strategy_spec_approved_at = None
    has_persisted_spec = False

    if entity.get("type") == "strategy":
        # First check for persisted spec
        persisted_spec = await kb_repo.get_strategy_spec(entity_id)

        if persisted_spec:
            has_persisted_spec = True
            # Use persisted spec
            spec_json = persisted_spec.get("spec_json", {})
            if isinstance(spec_json, str):
                spec_json = json.loads(spec_json)
            strategy_spec = spec_json
            strategy_spec_status = persisted_spec.get("status", "draft")
            strategy_spec_version = persisted_spec.get("version", 1)
            strategy_spec_approved_by = persisted_spec.get("approved_by")
            strategy_spec_approved_at = persisted_spec.get("approved_at")
        else:
            # Fall back to live preview from claims
            spec_claims, _ = await kb_repo.list_claims(
                workspace_id=entity["workspace_id"],
                entity_id=entity_id,
                status="verified",
                limit=100,
            )

            # Group by claim type
            spec_draft = {
                "name": entity["name"],
                "description": entity.get("description"),
                "rules": [],
                "parameters": [],
                "equations": [],
                "warnings": [],
                "assumptions": [],
            }

            for c in spec_claims:
                ctype = c.get("claim_type", "other")
                if ctype == "rule":
                    spec_draft["rules"].append(c["text"])
                elif ctype == "parameter":
                    spec_draft["parameters"].append(c["text"])
                elif ctype == "equation":
                    spec_draft["equations"].append(c["text"])
                elif ctype == "warning":
                    spec_draft["warnings"].append(c["text"])
                elif ctype == "assumption":
                    spec_draft["assumptions"].append(c["text"])

            # Remove empty sections
            strategy_spec = {k: v for k, v in spec_draft.items() if v}

    return templates.TemplateResponse(
        "entity_detail.html",
        {
            "request": request,
            "entity": entity,
            "claims": claims,
            "total_claims": total_claims,
            "status_filter": status_filter or "verified",
            "claim_type": claim_type or "",
            "limit": limit,
            "offset": offset,
            "claim_types": [c.value for c in KBClaimType],
            "claim_statuses": ["all", "verified", "weak", "pending", "rejected"],
            "has_prev": offset > 0,
            "has_next": offset + limit < total_claims,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
            "strategy_spec": (
                json.dumps(strategy_spec, indent=2) if strategy_spec else None
            ),
            "strategy_spec_status": strategy_spec_status,
            "strategy_spec_version": strategy_spec_version,
            "strategy_spec_approved_by": strategy_spec_approved_by,
            "strategy_spec_approved_at": strategy_spec_approved_at,
            "has_persisted_spec": has_persisted_spec,
            "possible_duplicates": possible_duplicates,
        },
    )


@router.get("/kb/claims/{claim_id}", response_class=HTMLResponse)
async def admin_claim_detail(
    request: Request,
    claim_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """View claim details with evidence."""
    kb_repo = _get_kb_repo()

    claim = await kb_repo.get_claim_by_id(claim_id, include_evidence=True)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim {claim_id} not found",
        )

    # Convert to JSON-serializable for debug panel
    claim_json = _json_serializable(claim)

    return templates.TemplateResponse(
        "claim_detail.html",
        {
            "request": request,
            "claim": claim,
            "claim_json": json.dumps(claim_json, indent=2),
        },
    )


@router.get("/kb/claims", response_class=HTMLResponse)
async def admin_claims_list(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    q: Optional[str] = Query(None, description="Search query"),
    status_filter: Optional[str] = Query("verified", description="Status filter"),
    claim_type: Optional[str] = Query(None, description="Claim type filter"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List all claims with search and filters."""
    kb_repo = _get_kb_repo()

    # If no workspace specified, get first available
    if not workspace_id:
        try:
            async with _db_pool.acquire() as conn:
                row = await conn.fetchrow("SELECT id FROM workspaces LIMIT 1")
                if row:
                    workspace_id = row["id"]
        except Exception as e:
            logger.warning("Could not fetch default workspace", error=str(e))

    if not workspace_id:
        return templates.TemplateResponse(
            "claims_list.html",
            {
                "request": request,
                "claims": [],
                "total": 0,
                "workspace_id": None,
                "error": "No workspace found.",
            },
        )

    from app.services.kb_types import ClaimType

    internal_claim_type = None
    if claim_type:
        try:
            internal_claim_type = ClaimType(claim_type)
        except ValueError:
            pass

    status_value = status_filter if status_filter != "all" else None

    claims, total = await kb_repo.list_claims(
        workspace_id=workspace_id,
        q=q,
        status=status_value,
        claim_type=internal_claim_type,
        limit=limit,
        offset=offset,
    )

    return templates.TemplateResponse(
        "claims_list.html",
        {
            "request": request,
            "claims": claims,
            "total": total,
            "workspace_id": str(workspace_id),
            "q": q or "",
            "status_filter": status_filter or "verified",
            "claim_type": claim_type or "",
            "limit": limit,
            "offset": offset,
            "claim_types": [c.value for c in KBClaimType],
            "claim_statuses": ["all", "verified", "weak", "pending", "rejected"],
            "has_prev": offset > 0,
            "has_next": offset + limit < total,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
        },
    )


# ===========================================
# Backtest Tune Admin Routes
# ===========================================


def _get_tune_repo():
    """Get TuneRepository instance."""
    from app.repositories.backtests import TuneRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return TuneRepository(_db_pool)


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
    if not workspace_id:
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


async def _fetch_tune_for_compare(tune_id: UUID) -> Optional[dict[str, Any]]:
    """Fetch tune with best run metrics for comparison."""
    if _db_pool is None:
        return None

    query = """
        SELECT t.id, t.created_at, t.status, t.workspace_id,
               t.strategy_entity_id, t.objective_metric, t.objective_type,
               t.objective_params, t.oos_ratio, t.gates,
               t.best_run_id, t.best_score, t.best_params,
               e.name as strategy_name,
               -- Best run metrics
               tr.objective_score as best_objective_score,
               tr.score_is, tr.score_oos,
               tr.metrics_oos as best_metrics_oos
        FROM backtest_tunes t
        LEFT JOIN kb_entities e ON t.strategy_entity_id = e.id
        LEFT JOIN backtest_tune_runs tr ON t.id = tr.tune_id AND tr.run_id = t.best_run_id
        WHERE t.id = $1
    """

    async with _db_pool.acquire() as conn:
        row = await conn.fetchrow(query, tune_id)
        if not row:
            return None

        result = dict(row)

        # Parse JSONB fields
        for field in ["objective_params", "gates", "best_params", "best_metrics_oos"]:
            if result.get(field) and isinstance(result[field], str):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = None

        # Compute overfit gap
        score_is = result.get("score_is")
        score_oos = result.get("score_oos")
        if score_is is not None and score_oos is not None:
            result["overfit_gap"] = round(score_is - score_oos, 4)
        else:
            result["overfit_gap"] = None

        # Convert UUIDs
        result["id"] = str(result["id"])
        result["workspace_id"] = str(result["workspace_id"])
        result["strategy_entity_id"] = str(result["strategy_entity_id"])
        if result.get("best_run_id"):
            result["best_run_id"] = str(result["best_run_id"])

        return result


@router.get("/backtests/compare")
async def admin_tune_compare(
    request: Request,
    tune_id: list[UUID] = Query(..., description="Tune IDs to compare (2+)"),
    workspace_id: Optional[UUID] = Query(None, description="Workspace UUID (optional)"),
    format: Optional[str] = Query(
        None, description="Output format: 'json' for download"
    ),
    _: bool = Depends(require_admin_token),
):
    """Compare two or more parameter tuning sessions side-by-side."""

    # Require at least 2 tunes
    if len(tune_id) < 2:
        return templates.TemplateResponse(
            "compare.html",
            {
                "request": request,
                "error": "Compare requires at least 2 tune IDs",
                "hint": "Example: /admin/backtests/compare?tune_id=<A>&tune_id=<B>",
                "tunes": [],
                "rows": [],
                "workspace_id": str(workspace_id) if workspace_id else "",
            },
        )

    # Fetch all tunes
    tunes = []
    for tid in tune_id:
        tune = await _fetch_tune_for_compare(tid)
        if tune:
            tunes.append(tune)
        else:
            return templates.TemplateResponse(
                "compare.html",
                {
                    "request": request,
                    "error": f"Tune {tid} not found",
                    "tunes": [],
                    "rows": [],
                    "workspace_id": str(workspace_id) if workspace_id else "",
                },
            )

    # Use workspace from first tune if not provided
    if not workspace_id:
        workspace_id = tunes[0]["workspace_id"]

    # Build comparison rows
    rows = []

    # --- Section: Identity ---
    rows.append({"section": "Identity", "is_header": True})

    rows.append(
        {
            "label": "Strategy",
            "values": [
                t.get("strategy_name") or t["strategy_entity_id"][:8] + "..."
                for t in tunes
            ],
            "fmt": "default",
        }
    )
    rows.append(
        {
            "label": "Status",
            "values": [t["status"] for t in tunes],
            "fmt": "default",
        }
    )
    rows.append(
        {
            "label": "Objective Type",
            "values": [t.get("objective_type") or "sharpe" for t in tunes],
            "fmt": "default",
        }
    )

    # Extract dd_lambda from objective_params
    dd_lambdas = []
    for t in tunes:
        params = t.get("objective_params") or {}
        dd_lambdas.append(params.get("dd_lambda"))
    rows.append(
        {
            "label": "λ (dd_lambda)",
            "values": [_normalize_compare_value(v, "float2") for v in dd_lambdas],
            "raw_values": dd_lambdas,
            "fmt": "float2",
        }
    )

    rows.append(
        {
            "label": "OOS Ratio",
            "values": [
                _normalize_compare_value(t.get("oos_ratio"), "pct_ratio") for t in tunes
            ],
            "raw_values": [t.get("oos_ratio") for t in tunes],
            "fmt": "pct_ratio",
        }
    )

    # Gates
    rows.append(
        {
            "label": "Gates: Max DD",
            "values": [
                (
                    _normalize_compare_value(
                        (t.get("gates") or {}).get("max_drawdown_pct"), "int"
                    )
                    + "%"
                    if (t.get("gates") or {}).get("max_drawdown_pct") is not None
                    else "—"
                )
                for t in tunes
            ],
            "raw_values": [
                (t.get("gates") or {}).get("max_drawdown_pct") for t in tunes
            ],
            "fmt": "default",
        }
    )
    rows.append(
        {
            "label": "Gates: Min Trades",
            "values": [
                _normalize_compare_value(
                    (t.get("gates") or {}).get("min_trades"), "int"
                )
                for t in tunes
            ],
            "raw_values": [(t.get("gates") or {}).get("min_trades") for t in tunes],
            "fmt": "int",
        }
    )
    rows.append(
        {
            "label": "Gates: Evaluated On",
            "values": [
                (t.get("gates") or {}).get("evaluated_on") or "—" for t in tunes
            ],
            "fmt": "default",
        }
    )

    # --- Section: Winning Metrics ---
    rows.append({"section": "Winning Metrics", "is_header": True})

    # Extract metrics from best_metrics_oos
    for t in tunes:
        t["_metrics"] = t.get("best_metrics_oos") or {}

    rows.append(
        {
            "label": "Objective Score",
            "values": [
                _normalize_compare_value(
                    t.get("best_objective_score")
                    or t.get("score_oos")
                    or t.get("best_score"),
                    "float4",
                )
                for t in tunes
            ],
            "fmt": "float4",
        }
    )
    rows.append(
        {
            "label": "Return %",
            "values": [
                _normalize_compare_value(t["_metrics"].get("return_pct"), "pct")
                for t in tunes
            ],
            "raw_values": [t["_metrics"].get("return_pct") for t in tunes],
            "fmt": "pct",
        }
    )
    rows.append(
        {
            "label": "Sharpe",
            "values": [
                _normalize_compare_value(t["_metrics"].get("sharpe"), "float2")
                for t in tunes
            ],
            "raw_values": [t["_metrics"].get("sharpe") for t in tunes],
            "fmt": "float2",
        }
    )
    rows.append(
        {
            "label": "Max DD %",
            "values": [
                _normalize_compare_value(
                    t["_metrics"].get("max_drawdown_pct"), "pct_neg"
                )
                for t in tunes
            ],
            "raw_values": [t["_metrics"].get("max_drawdown_pct") for t in tunes],
            "fmt": "pct_neg",
        }
    )
    rows.append(
        {
            "label": "Trades",
            "values": [
                _normalize_compare_value(t["_metrics"].get("trades"), "int")
                for t in tunes
            ],
            "raw_values": [t["_metrics"].get("trades") for t in tunes],
            "fmt": "int",
        }
    )

    # Overfit gap with special styling
    overfit_gaps = [t.get("overfit_gap") for t in tunes]
    rows.append(
        {
            "label": "Overfit Gap",
            "values": [_normalize_compare_value(g, "float4") for g in overfit_gaps],
            "raw_values": overfit_gaps,
            "classes": [_overfit_class(g) for g in overfit_gaps],
            "fmt": "float4",
        }
    )

    # --- Section: Best Params ---
    rows.append({"section": "Best Params", "is_header": True})

    # Union of all param keys
    all_param_keys = set()
    for t in tunes:
        params = t.get("best_params") or {}
        all_param_keys.update(params.keys())

    for key in sorted(all_param_keys):
        param_values = []
        for t in tunes:
            params = t.get("best_params") or {}
            val = params.get(key)
            if val is None:
                param_values.append("—")
            elif isinstance(val, float):
                param_values.append(f"{val:.4g}")
            else:
                param_values.append(str(val))
        rows.append(
            {
                "label": key,
                "values": param_values,
                "is_param": True,
                "fmt": "default",
            }
        )

    # Mark differing rows
    for row in rows:
        if row.get("is_header"):
            continue
        row["differs"] = _values_differ(row.get("values", []))

    # JSON Export
    if format == "json":
        return _generate_compare_json(tunes, rows)

    return templates.TemplateResponse(
        "compare.html",
        {
            "request": request,
            "tunes": tunes,
            "rows": rows,
            "tune_ids": [str(tid) for tid in tune_id],
            "workspace_id": (
                str(workspace_id) if workspace_id else tunes[0]["workspace_id"]
            ),
        },
    )


def _generate_compare_json(tunes: list[dict], rows: list[dict]) -> JSONResponse:
    """Generate JSON export of compare data."""
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]  # Support up to 8 tunes

    tune_exports = []
    for idx, tune in enumerate(tunes):
        metrics_oos = tune.get("best_metrics_oos") or {}
        if isinstance(metrics_oos, str):
            try:
                metrics_oos = json.loads(metrics_oos)
            except json.JSONDecodeError:
                metrics_oos = {}

        tune_export = {
            "tune_id": tune.get("id"),
            "label": labels[idx] if idx < len(labels) else str(idx + 1),
            "status": tune.get("status"),
            "strategy": {
                "entity_id": tune.get("strategy_entity_id"),
                "name": tune.get("strategy_name"),
            },
            "objective": {
                "type": tune.get("objective_type") or "sharpe",
                "params": tune.get("objective_params"),
            },
            "oos_ratio": tune.get("oos_ratio"),
            "gates": tune.get("gates"),
            "best": {
                "run_id": tune.get("best_run_id"),
                "objective_score": tune.get("best_objective_score"),
                "score": tune.get("best_score"),
                "score_is": tune.get("score_is"),
                "score_oos": tune.get("score_oos"),
                "overfit_gap": tune.get("overfit_gap"),
                "metrics_oos": metrics_oos,
                "params": tune.get("best_params"),
            },
            "created_at": (
                tune.get("created_at").isoformat() if tune.get("created_at") else None
            ),
        }
        tune_exports.append(tune_export)

    # Build simplified rows for JSON (skip is_header rows, convert to diff format)
    row_exports = []
    current_section = None
    for row in rows:
        if row.get("is_header"):
            current_section = row.get("section")
            continue

        row_export = {
            "section": current_section,
            "field": row.get("label"),
            "values": row.get("values", []),
            "diff": row.get("differs", False),
        }
        row_exports.append(row_export)

    export_data = {
        "generated_at": datetime.now().isoformat(),
        "tunes": tune_exports,
        "rows": row_exports,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Build filename with tune ID prefixes (e.g., compare_abc123_def456_20240115_1430.json)
    tune_id_parts = [
        t.get("id", "")[:8] for t in tunes[:3]
    ]  # First 3 tune IDs, 8 chars each
    filename = f"compare_{'_'.join(tune_id_parts)}_{timestamp}.json"

    return JSONResponse(
        content=_json_serializable(export_data),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ===========================================
# Backfill Regime Attribution
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
        job = BackfillTuneRegimeJob(db_pool=_db_pool)
        result = await job.run(
            workspace_id=request.workspace_id,
            dry_run=request.dry_run,
            limit=request.limit,
        )

        logger.info(
            "backfill_tune_regime_complete",
            workspace_id=str(request.workspace_id),
            processed=result.processed,
            skipped=result.skipped,
            errors=result.errors,
            dry_run=request.dry_run,
        )

        return BackfillRegimeResponse(
            processed=result.processed,
            skipped=result.skipped,
            errors=result.errors,
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
    run_status: Optional[str] = Query(None, description="Filter trials by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """View tune session details with trials."""
    tune_repo = _get_tune_repo()

    # Get tune
    tune = await tune_repo.get_tune(tune_id)
    if not tune:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tune {tune_id} not found",
        )

    # Convert UUIDs to strings for template
    tune["id"] = str(tune["id"])
    tune["workspace_id"] = str(tune["workspace_id"])
    tune["strategy_entity_id"] = str(tune["strategy_entity_id"])
    if tune.get("best_run_id"):
        tune["best_run_id"] = str(tune["best_run_id"])

    # Get counts
    counts = await tune_repo.get_tune_status_counts(tune_id)

    # Parse best_params if needed
    best_params = tune.get("best_params")
    if isinstance(best_params, str):
        try:
            best_params = json.loads(best_params)
        except json.JSONDecodeError:
            best_params = None
    tune["best_params"] = best_params

    # Get trials
    runs, total_runs = await tune_repo.list_tune_runs(
        tune_id=tune_id,
        status=run_status,
        limit=limit,
        offset=offset,
    )

    # Parse params, convert UUIDs, and compute derived fields for each run
    for run in runs:
        if run.get("run_id"):
            run["run_id"] = str(run["run_id"])
        if run.get("tune_id"):
            run["tune_id"] = str(run["tune_id"])
        if isinstance(run.get("params"), str):
            try:
                run["params"] = json.loads(run["params"])
            except json.JSONDecodeError:
                run["params"] = {}

        # Compute overfit_gap when both IS and OOS scores available
        score_is = run.get("score_is")
        score_oos = run.get("score_oos")
        if score_is is not None and score_oos is not None:
            run["overfit_gap"] = round(score_is - score_oos, 4)
        else:
            run["overfit_gap"] = None

    # Aggregate skip reasons for "Why trials skipped?" callout
    skip_reasons_summary = []
    if counts.get("skipped", 0) > 0:
        async with _db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT skip_reason, COUNT(*) as count
                FROM backtest_tune_runs
                WHERE tune_id = $1 AND skip_reason IS NOT NULL
                GROUP BY skip_reason
                ORDER BY count DESC
                """,
                tune_id,
            )
            skip_reasons_summary = [
                {"reason": row["skip_reason"], "count": row["count"]} for row in rows
            ]

    return templates.TemplateResponse(
        "tune_detail.html",
        {
            "request": request,
            "tune": tune,
            "counts": counts,
            "runs": runs,
            "total_runs": total_runs,
            "run_status": run_status or "",
            "limit": limit,
            "offset": offset,
            "has_prev": offset > 0,
            "has_next": offset + limit < total_runs,
            "prev_offset": max(0, offset - limit),
            "next_offset": offset + limit,
            "skip_reasons_summary": skip_reasons_summary,
        },
    )


@router.get("/backtests/runs/{run_id}", response_class=HTMLResponse)
async def admin_backtest_run_detail(
    request: Request,
    run_id: UUID,
    _: bool = Depends(require_admin_token),
):
    """View backtest run details."""
    from app.repositories.backtests import BacktestRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    backtest_repo = BacktestRepository(_db_pool)
    run = await backtest_repo.get_run(run_id)

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    # Convert UUIDs to strings for template
    run["id"] = str(run["id"])
    run["strategy_entity_id"] = str(run["strategy_entity_id"])
    if run.get("workspace_id"):
        run["workspace_id"] = str(run["workspace_id"])
    if run.get("strategy_spec_id"):
        run["strategy_spec_id"] = str(run["strategy_spec_id"])

    # For now, redirect to API endpoint or show JSON
    # TODO: Create proper backtest run detail template
    run_json = _json_serializable(run)

    return templates.TemplateResponse(
        "backtest_run_detail.html",
        {
            "request": request,
            "run": run,
            "run_json": json.dumps(run_json, indent=2),
        },
    )


# ==============================================================================
# Query Compare Evals (PR3: Evaluation Collector)
# ==============================================================================


def _get_eval_repo():
    """Get EvalRepository instance."""
    from app.repositories.evals import EvalRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return EvalRepository(_db_pool)


@router.get("/evals/query-compare/summary")
async def eval_summary(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace to get stats for"),
    since: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    _: None = Depends(require_admin_token),
):
    """
    Get summary stats for query compare evaluations.

    Returns impact rate, latency percentiles, and fallback rate.
    """
    # Parse time window
    since_hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}.get(since, 24)

    repo = _get_eval_repo()
    summary = await repo.get_summary(workspace_id, since_hours)

    return {
        "workspace_id": str(workspace_id),
        "since": since,
        "total": summary.total,
        "impacted_count": summary.impacted_count,
        "impact_rate": round(summary.impact_rate, 4),
        "p50_rerank_ms": (
            round(summary.p50_rerank_ms, 1) if summary.p50_rerank_ms else None
        ),
        "p95_rerank_ms": (
            round(summary.p95_rerank_ms, 1) if summary.p95_rerank_ms else None
        ),
        "fallback_count": summary.fallback_count,
        "fallback_rate": round(summary.fallback_rate, 4),
        "timeout_count": summary.timeout_count,
        "timeout_rate": round(summary.timeout_rate, 4),
    }


@router.get("/evals/query-compare/by-config")
async def eval_by_config(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace to get stats for"),
    since: str = Query("7d", description="Time window: 1h, 24h, 7d, 30d"),
    _: None = Depends(require_admin_token),
):
    """
    Get stats grouped by config fingerprint.

    Useful for comparing different retrieve_k/top_k combinations.
    """
    since_hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}.get(since, 168)

    repo = _get_eval_repo()
    configs = await repo.get_by_config(workspace_id, since_hours)

    return {
        "workspace_id": str(workspace_id),
        "since": since,
        "configs": [
            {
                "config_fingerprint": c.config_fingerprint,
                "rerank_method": c.rerank_method,
                "rerank_model": c.rerank_model,
                "candidates_k": c.candidates_k,
                "top_k": c.top_k,
                "total": c.total,
                "impact_rate": round(c.impact_rate, 4),
                "p50_rerank_ms": round(c.p50_rerank_ms, 1) if c.p50_rerank_ms else None,
                "p95_rerank_ms": round(c.p95_rerank_ms, 1) if c.p95_rerank_ms else None,
                "fallback_rate": round(c.fallback_rate, 4),
            }
            for c in configs
        ],
    }


@router.get("/evals/query-compare/most-impacted")
async def eval_most_impacted(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace to get stats for"),
    since: str = Query("7d", description="Time window: 1h, 24h, 7d, 30d"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    _: None = Depends(require_admin_token),
):
    """
    Get queries with highest reranking impact (lowest jaccard).

    Use for manual spot-checking of rerank quality.
    """
    since_hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}.get(since, 168)

    repo = _get_eval_repo()
    impacted = await repo.get_most_impacted(workspace_id, since_hours, limit)

    return {
        "workspace_id": str(workspace_id),
        "since": since,
        "count": len(impacted),
        "queries": [
            {
                "created_at": q.created_at.isoformat(),
                "question_hash": q.question_hash,
                "question_preview": q.question_preview,
                "jaccard": round(q.jaccard, 4),
                "spearman": round(q.spearman, 4) if q.spearman is not None else None,
                "rank_delta_mean": (
                    round(q.rank_delta_mean, 2) if q.rank_delta_mean else None
                ),
                "rank_delta_max": q.rank_delta_max,
                "vector_top5_ids": q.vector_top5_ids,
                "reranked_top5_ids": q.reranked_top5_ids,
            }
            for q in impacted
        ],
    }


@router.delete("/evals/query-compare/cleanup")
async def eval_cleanup(
    request: Request,
    days: int = Query(90, ge=1, le=365, description="Delete evals older than N days"),
    _: None = Depends(require_admin_token),
):
    """
    Delete evaluations older than N days.

    Use for retention management.
    """
    repo = _get_eval_repo()
    deleted = await repo.delete_older_than(days)

    return {
        "deleted": deleted,
        "retention_days": days,
    }


# =============================================================================
# KB Trials Admin Endpoints
# =============================================================================


def _get_kb_trial_repo():
    """Get KBTrialRepository instance."""
    from app.repositories.kb_trials import KBTrialRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return KBTrialRepository(_db_pool)


@router.get("/kb/trials/stats")
async def kb_trials_stats(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    since: Optional[datetime] = Query(
        None, description="Only count trials after this time"
    ),
    window_days: Optional[int] = Query(
        None, ge=1, le=365, description="Time window in days (7, 30, etc.)"
    ),
    _: None = Depends(require_admin_token),
):
    """
    Get KB trials statistics for a workspace.

    Returns point-in-time stats plus trend deltas when window_days is specified.

    **Query params:**
    - `since`: Only count trials created after this timestamp
    - `window_days`: Compare current vs previous window (e.g., 7 days)

    **Coverage metrics** explain why recommend might return none:
    - pct_with_regime_is: Has in-sample regime snapshot
    - pct_with_regime_oos: Has OOS regime snapshot (when has_oos=true)
    - pct_with_objective_score: Has objective_score set
    - pct_with_sharpe_oos: Has sharpe_oos metric (when has_oos=true)
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    from datetime import timedelta

    # Calculate time boundaries
    now = datetime.utcnow()
    if window_days:
        since = now - timedelta(days=window_days)
        prev_since = since - timedelta(days=window_days)
    else:
        prev_since = None

    async with _db_pool.acquire() as conn:
        # Build query conditions
        workspace_cond = "AND t.workspace_id = $1" if workspace_id else ""
        base_params = [workspace_id] if workspace_id else []

        async def get_stats(time_filter: str = "", time_params: list = []):
            """Get stats with optional time filter."""
            params = base_params + time_params

            # Total trials
            total = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE 1=1 {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            if total == 0:
                return {
                    "total": 0,
                    "with_oos": 0,
                    "valid": 0,
                    "stale": 0,
                    "with_regime_is": 0,
                    "with_regime_oos": 0,
                    "with_objective_score": 0,
                    "with_sharpe_oos": 0,
                }

            # Core metrics
            with_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            valid = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.is_valid = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            # Stale count
            try:
                stale = (
                    await conn.fetchval(
                        f"""
                    SELECT COUNT(*) FROM kb_trial_vectors t
                    WHERE t.needs_reembed = true {workspace_cond} {time_filter}
                """,
                        *params,
                    )
                    or 0
                )
            except Exception:
                stale = 0

            # Coverage metrics
            with_regime_is = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_is IS NOT NULL {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_regime_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_objective = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.objective_score IS NOT NULL {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_sharpe_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.sharpe_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            return {
                "total": total,
                "with_oos": with_oos,
                "valid": valid,
                "stale": stale,
                "with_regime_is": with_regime_is,
                "with_regime_oos": with_regime_oos,
                "with_objective_score": with_objective,
                "with_sharpe_oos": with_sharpe_oos,
            }

        # Get current stats
        if since:
            param_idx = len(base_params) + 1
            time_filter = f"AND t.created_at >= ${param_idx}"
            current = await get_stats(time_filter, [since])
        else:
            current = await get_stats()

        # Get previous window stats for deltas
        deltas = None
        if prev_since and window_days:
            param_idx = len(base_params) + 1
            time_filter = (
                f"AND t.created_at >= ${param_idx} AND t.created_at < ${param_idx + 1}"
            )
            previous = await get_stats(time_filter, [prev_since, since])

            # Calculate deltas
            deltas = {
                "trials_added": current["total"] - previous["total"],
                "valid_added": current["valid"] - previous["valid"],
                "stale_added": current["stale"] - previous["stale"],
                "pct_valid_delta": round(
                    (
                        current["valid"] / current["total"] * 100
                        if current["total"] > 0
                        else 0
                    )
                    - (
                        previous["valid"] / previous["total"] * 100
                        if previous["total"] > 0
                        else 0
                    ),
                    1,
                ),
                "window_days": window_days,
                "prev_window_start": prev_since.isoformat(),
                "prev_window_end": since.isoformat(),
            }

        # Last ingestion timestamp
        last_ts = await conn.fetchval(
            f"""
            SELECT MAX(t.created_at) FROM kb_trial_vectors t
            WHERE 1=1 {workspace_cond}
        """,
            *base_params,
        )

        # Workspace config for embedding info
        embed_model = "nomic-embed-text"
        embed_dim = 768
        collection_name = "trading_kb_trials__nomic-embed-text__768"

        if workspace_id:
            config_row = await conn.fetchrow(
                """
                SELECT config FROM workspaces WHERE id = $1
            """,
                workspace_id,
            )
            if config_row and config_row["config"]:
                config = config_row["config"]
                if isinstance(config, str):
                    config = json.loads(config)
                kb_config = config.get("kb", {})
                embed_model = kb_config.get("embed_model", embed_model)
                embed_dim = kb_config.get("embed_dim", embed_dim)
                collection_name = kb_config.get("collection_name", collection_name)

    # Calculate percentages
    total = current["total"]
    oos_count = current["with_oos"]

    def pct(num, denom):
        return round(num / denom * 100, 1) if denom > 0 else 0

    result = {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "total_trials": total,
        "trials_with_oos": oos_count,
        "trials_valid": current["valid"],
        "pct_with_oos": pct(oos_count, total),
        "pct_valid": pct(current["valid"], total),
        # Coverage metrics (explains why recommend might fail)
        "coverage": {
            "pct_with_regime_is": pct(current["with_regime_is"], total),
            "pct_with_regime_oos": pct(current["with_regime_oos"], oos_count),
            "pct_with_objective_score": pct(current["with_objective_score"], total),
            "pct_with_sharpe_oos": pct(current["with_sharpe_oos"], oos_count),
        },
        # Embedding config
        "embedding_model": embed_model,
        "embedding_dim": embed_dim,
        "collection_name": collection_name,
        "last_ingestion_ts": last_ts.isoformat() if last_ts else None,
        "stale_text_hash_count": current["stale"],
    }

    # Add time window info
    if since:
        result["since"] = since.isoformat()
    if deltas:
        result["deltas"] = deltas

    return result


@router.get("/kb/trials/ingestion-status")
async def kb_ingestion_status(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    _: None = Depends(require_admin_token),
):
    """
    Get KB ingestion health status.

    Returns:
    - trials_missing_vectors: Trials without embeddings
    - trials_missing_regime: Trials without regime snapshots
    - warning_counts: Top warning types and counts
    - recent_ingestion_runs: Last 10 ingestion runs
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        workspace_filter = "AND workspace_id = $1" if workspace_id else ""
        params = [workspace_id] if workspace_id else []

        # Missing vectors
        missing_vectors_query = f"""
            SELECT COUNT(*) as count
            FROM kb_trial_vectors
            WHERE vector IS NULL {workspace_filter}
        """
        try:
            missing_vectors = await conn.fetchval(missing_vectors_query, *params)
        except Exception:
            missing_vectors = 0

        # Missing regime
        missing_regime_query = f"""
            SELECT COUNT(*) as count
            FROM kb_trial_vectors
            WHERE regime_snapshot IS NULL {workspace_filter}
        """
        try:
            missing_regime = await conn.fetchval(missing_regime_query, *params)
        except Exception:
            missing_regime = 0

        # Warning counts
        warning_counts_query = f"""
            SELECT
                warning,
                COUNT(*) as count
            FROM kb_trial_vectors,
                 LATERAL unnest(warnings) as warning
            WHERE 1=1 {workspace_filter}
            GROUP BY warning
            ORDER BY count DESC
            LIMIT 20
        """
        try:
            warning_rows = await conn.fetch(warning_counts_query, *params)
            warning_counts = {row["warning"]: row["count"] for row in warning_rows}
        except Exception:
            warning_counts = {}

        # Recent ingestion runs (if tracked)
        recent_runs = []
        try:
            runs_query = f"""
                SELECT
                    id,
                    created_at,
                    ingested_count,
                    skipped_count,
                    error_count,
                    duration_ms
                FROM kb_ingestion_runs
                WHERE 1=1 {workspace_filter}
                ORDER BY created_at DESC
                LIMIT 10
            """
            run_rows = await conn.fetch(runs_query, *params)
            recent_runs = [
                {
                    "id": str(row["id"]),
                    "created_at": row["created_at"].isoformat(),
                    "ingested_count": row["ingested_count"],
                    "skipped_count": row["skipped_count"],
                    "error_count": row["error_count"],
                    "duration_ms": row["duration_ms"],
                }
                for row in run_rows
            ]
        except Exception:
            # Table might not exist yet
            pass

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "trials_missing_vectors": missing_vectors or 0,
        "trials_missing_regime": missing_regime or 0,
        "warning_counts": warning_counts,
        "recent_ingestion_runs": recent_runs,
    }


@router.get("/kb/collections")
async def kb_collections(
    request: Request,
    _: None = Depends(require_admin_token),
):
    """
    Get list of KB collections in Qdrant with health info.

    Returns:
    - Collection names and point counts
    - Vector dimension and distance metric
    - Payload index counts
    - Optimizer status
    """
    from qdrant_client import AsyncQdrantClient
    from app.config import get_settings

    settings = get_settings()

    try:
        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Get all collections
        collections_response = await client.get_collections()

        result = []
        for coll in collections_response.collections:
            try:
                info = await client.get_collection(coll.name)

                # Parse embedding model from collection name if encoded
                # Format: trading_kb_trials__{model}__{dim}
                parts = coll.name.split("__")
                embedding_model = parts[1] if len(parts) >= 2 else None
                embedding_dim = int(parts[2]) if len(parts) >= 3 else None

                # Vector config
                vec_cfg = info.config.params.vectors
                vector_size = vec_cfg.size if vec_cfg else None
                distance = (
                    vec_cfg.distance.value if vec_cfg and vec_cfg.distance else None
                )

                # Payload indexes count
                payload_indexes = 0
                if info.payload_schema:
                    payload_indexes = len(info.payload_schema)

                # Optimizer status
                optimizer_status = "unknown"
                if info.optimizer_status:
                    optimizer_status = (
                        info.optimizer_status.status.value
                        if hasattr(info.optimizer_status, "status")
                        else str(info.optimizer_status)
                    )

                result.append(
                    {
                        "name": coll.name,
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count,
                        "status": info.status.value if info.status else "unknown",
                        "vector_size": vector_size or embedding_dim,
                        "distance": distance,
                        "embedding_model_id": embedding_model,
                        "payload_indexes_count": payload_indexes,
                        "optimizer_status": optimizer_status,
                        "segments_count": (
                            len(info.segments or [])
                            if hasattr(info, "segments")
                            else None
                        ),
                    }
                )
            except Exception as e:
                result.append(
                    {
                        "name": coll.name,
                        "error": str(e),
                    }
                )

        await client.close()

        return {
            "collections": result,
            "qdrant_host": settings.qdrant_host,
            "qdrant_port": settings.qdrant_port,
            "total_collections": len(result),
        }

    except Exception as e:
        logger.error("Failed to get Qdrant collections", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Qdrant: {str(e)}",
        )


@router.get("/kb/warnings/top")
async def kb_top_warnings(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    limit: int = Query(20, ge=1, le=100, description="Number of warnings to return"),
    _: None = Depends(require_admin_token),
):
    """
    Get top warning types across KB trials.

    Useful for identifying systematic data quality issues.
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        workspace_filter = "AND workspace_id = $1" if workspace_id else ""
        params = [workspace_id, limit] if workspace_id else [limit]
        limit_param = "$2" if workspace_id else "$1"

        query = f"""
            SELECT
                warning,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM kb_trial_vectors WHERE 1=1 {workspace_filter}) as pct  # noqa: E501
            FROM kb_trial_vectors,
                 LATERAL unnest(warnings) as warning
            WHERE 1=1 {workspace_filter}
            GROUP BY warning
            ORDER BY count DESC
            LIMIT {limit_param}
        """

        try:
            rows = await conn.fetch(query, *params)
            warnings = [
                {
                    "warning": row["warning"],
                    "count": row["count"],
                    "pct": round(row["pct"], 2) if row["pct"] else 0,
                }
                for row in rows
            ]
        except Exception:
            warnings = []

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "warnings": warnings,
    }


@router.get("/kb/trials/sample")
async def kb_trials_sample(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    warning: Optional[str] = Query(
        None, description="Filter by warning type (e.g., high_overfit)"
    ),
    is_valid: Optional[bool] = Query(None, description="Filter by validity"),
    has_oos: Optional[bool] = Query(None, description="Filter by OOS availability"),
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(20, ge=1, le=100, description="Number of samples to return"),
    _: None = Depends(require_admin_token),
):
    """
    Get sample trials for debugging quality issues.

    Returns safe fields only - no sensitive internals.
    Useful for inspecting what's actually in the KB.

    **Example use cases:**
    - `?warning=high_overfit` - See trials flagged as overfit
    - `?is_valid=false` - See invalid trials
    - `?has_oos=true&strategy_name=mean_reversion` - See valid OOS trials for a strategy
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        # Build dynamic WHERE clause
        conditions = ["1=1"]
        params = []
        param_idx = 1

        if workspace_id:
            conditions.append(f"workspace_id = ${param_idx}")
            params.append(workspace_id)
            param_idx += 1

        if warning:
            conditions.append(f"${param_idx} = ANY(warnings)")
            params.append(warning)
            param_idx += 1

        if is_valid is not None:
            conditions.append(f"is_valid = ${param_idx}")
            params.append(is_valid)
            param_idx += 1

        if has_oos is not None:
            conditions.append(f"has_oos_metrics = ${param_idx}")
            params.append(has_oos)
            param_idx += 1

        if strategy_name:
            conditions.append(f"strategy_name = ${param_idx}")
            params.append(strategy_name)
            param_idx += 1

        # Add limit
        params.append(limit)
        limit_param = f"${param_idx}"

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                id,
                point_id,
                tune_run_id,
                strategy_name,
                objective_type,
                objective_score,
                is_valid,
                has_oos_metrics,
                overfit_gap,
                sharpe_is,
                sharpe_oos,
                trades_is,
                trades_oos,
                max_drawdown_is,
                max_drawdown_oos,
                regime_tags_is,
                regime_tags_oos,
                warnings,
                created_at
            FROM kb_trial_vectors
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit_param}
        """

        try:
            rows = await conn.fetch(query, *params)
            samples = [
                {
                    "id": str(row["id"]),
                    "point_id": row["point_id"],
                    "tune_run_id": (
                        str(row["tune_run_id"]) if row["tune_run_id"] else None
                    ),
                    "strategy_name": row["strategy_name"],
                    "objective_type": row["objective_type"],
                    "objective_score": row["objective_score"],
                    "is_valid": row["is_valid"],
                    "has_oos_metrics": row["has_oos_metrics"],
                    "overfit_gap": row["overfit_gap"],
                    "metrics": {
                        "sharpe_is": row["sharpe_is"],
                        "sharpe_oos": row["sharpe_oos"],
                        "trades_is": row["trades_is"],
                        "trades_oos": row["trades_oos"],
                        "max_drawdown_is": row["max_drawdown_is"],
                        "max_drawdown_oos": row["max_drawdown_oos"],
                    },
                    "regime_tags_is": row["regime_tags_is"],
                    "regime_tags_oos": row["regime_tags_oos"],
                    "warnings": row["warnings"],
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to fetch KB samples", error=str(e))
            samples = []

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "filters": {
            "warning": warning,
            "is_valid": is_valid,
            "has_oos": has_oos,
            "strategy_name": strategy_name,
        },
        "count": len(samples),
        "samples": samples,
    }


# ===========================================
# Ops Snapshot Endpoint
# ===========================================


@router.get("/ops/snapshot")
async def ops_snapshot(
    _: bool = Depends(require_admin_token),
    settings: Settings = Depends(get_settings),
):
    """
    Operational snapshot for go-live verification.

    Aggregates:
    - Service readiness + release metadata
    - Active collection and model configuration
    - Ingestion coverage metrics
    - Last smoke run timestamp (if available)

    Use this endpoint to verify deployment health before enabling features.
    """
    import asyncio
    from app import __version__
    from app.routers.health import (
        check_database_health,
        check_qdrant_collection,
        check_embed_service,
        _timed_check,
    )

    # Run readiness checks
    db_check, qdrant_check, embed_check = await asyncio.gather(
        _timed_check("database", check_database_health(settings)),
        _timed_check("qdrant_collection", check_qdrant_collection(settings)),
        _timed_check("embed_service", check_embed_service(settings)),
    )

    readiness = {
        "ready": all(c.status == "ok" for c in [db_check, qdrant_check, embed_check]),
        "checks": {
            "database": {
                "status": db_check.status,
                "latency_ms": db_check.latency_ms,
                "error": db_check.error,
            },
            "qdrant_collection": {
                "status": qdrant_check.status,
                "latency_ms": qdrant_check.latency_ms,
                "error": qdrant_check.error,
            },
            "embed_service": {
                "status": embed_check.status,
                "latency_ms": embed_check.latency_ms,
                "error": embed_check.error,
            },
        },
    }

    # Release metadata
    release = {
        "version": __version__,
        "git_sha": settings.git_sha or "unknown",
        "build_time": settings.build_time or "unknown",
        "config_profile": settings.config_profile,
        "environment": settings.sentry_environment,
    }

    # Configuration snapshot
    config = {
        "active_collection": settings.qdrant_collection_active,
        "embed_model": settings.embed_model,
        "embed_dim": settings.embed_dim,
        "docs_enabled": settings.docs_enabled,
        "rate_limit_enabled": settings.rate_limit_enabled,
        "llm_enabled": settings.llm_enabled,
    }

    # Ingestion coverage metrics (from database)
    coverage = {
        "total_kb_trials": 0,
        "trials_with_vectors": 0,
        "missing_vectors": 0,
        "by_strategy": {},
        "by_workspace": {},
        "stale_text_hash_count": 0,
    }

    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                # Total KB trials
                total = await conn.fetchval("SELECT COUNT(*) FROM kb_trials")
                coverage["total_kb_trials"] = total or 0

                # Trials with vectors (have point_id)
                with_vectors = await conn.fetchval(
                    "SELECT COUNT(*) FROM kb_trials WHERE point_id IS NOT NULL"
                )
                coverage["trials_with_vectors"] = with_vectors or 0
                coverage["missing_vectors"] = (total or 0) - (with_vectors or 0)

                # By strategy
                strategy_rows = await conn.fetch(
                    """
                    SELECT strategy_name, COUNT(*) as count,
                           COUNT(*) FILTER (WHERE point_id IS NOT NULL) as indexed
                    FROM kb_trials
                    GROUP BY strategy_name
                    ORDER BY count DESC
                    """
                )
                coverage["by_strategy"] = {
                    row["strategy_name"]: {
                        "total": row["count"],
                        "indexed": row["indexed"],
                    }
                    for row in strategy_rows
                }

                # By workspace
                workspace_rows = await conn.fetch(
                    """
                    SELECT workspace_id, COUNT(*) as count,
                           COUNT(*) FILTER (WHERE point_id IS NOT NULL) as indexed
                    FROM kb_trials
                    GROUP BY workspace_id
                    ORDER BY count DESC
                    """
                )
                coverage["by_workspace"] = {
                    str(row["workspace_id"]): {
                        "total": row["count"],
                        "indexed": row["indexed"],
                    }
                    for row in workspace_rows
                }

                # Check for stale text hashes (trials where text_hash doesn't match current embedding)  # noqa: E501
                # This is a simplified check - in production you'd compare against chunk_vectors
                stale = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM kb_trials
                    WHERE point_id IS NOT NULL
                    AND updated_at > indexed_at
                    """
                )
                coverage["stale_text_hash_count"] = stale or 0

        except Exception as e:
            logger.warning("Failed to fetch coverage metrics", error=str(e))
            coverage["error"] = str(e)

    # Last smoke run timestamp (check for smoke test results file)
    smoke_run = {
        "last_run": None,
        "status": "unknown",
        "results_available": False,
    }

    # Check common smoke test output locations
    smoke_paths = [
        Path("tests/smoke/results/latest.json"),
        Path("smoke_results.json"),
        Path("/tmp/smoke_results.json"),
    ]

    for smoke_path in smoke_paths:
        if smoke_path.exists():
            try:
                import json

                with open(smoke_path) as f:
                    smoke_data = json.load(f)
                smoke_run["last_run"] = smoke_data.get("timestamp") or smoke_data.get(
                    "run_at"
                )
                smoke_run["status"] = smoke_data.get("status", "unknown")
                smoke_run["results_available"] = True
                break
            except Exception:
                pass

    return {
        "snapshot_time": datetime.utcnow().isoformat() + "Z",
        "readiness": readiness,
        "release": release,
        "config": config,
        "coverage": coverage,
        "smoke_run": smoke_run,
    }


# ==============================================================================
# Trade Events Journal (Intent Contract + Policy Engine Audit)
# ==============================================================================


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
    if not workspace_id:
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
    from datetime import timedelta

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
    from datetime import timedelta

    # If no workspace specified, get first available
    if not workspace_id:
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
                (c.payload->>'n_completed')::int as n_completed,
                (c.payload->>'n_failed')::int as n_failed,
                (c.payload->>'duration_ms')::int as duration_ms
            FROM started s
            LEFT JOIN completed c ON s.run_plan_id = c.run_plan_id
        )
        SELECT * FROM run_plans
        WHERE 1=1
    """

    params = [workspace_id, since]
    param_idx = 3

    # Apply status filter
    if status_filter:
        query += f" AND status = ${param_idx}"
        params.append(status_filter)
        param_idx += 1

    # Count query
    count_query = f"""
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
        WHERE 1=1 {'AND status = $3' if status_filter else ''}
    """

    count_params = [workspace_id, since]
    if status_filter:
        count_params.append(status_filter)

    # Add ordering and pagination
    query += f" ORDER BY started_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
    params.extend([limit, offset])

    async with _db_pool.acquire() as conn:
        count_row = await conn.fetchrow(count_query, *count_params)
        total = count_row["total"] if count_row else 0

        rows = await conn.fetch(query, *params)

    run_plans = []
    for row in rows:
        run_plans.append(
            {
                "run_plan_id": row["run_plan_id"],
                "status": row["status"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "n_variants": row["n_variants"],
                "objective": row["objective"],
                "dataset_ref": row["dataset_ref"],
                "bar_count": row["bar_count"],
                "n_completed": row["n_completed"],
                "n_failed": row["n_failed"],
                "duration_ms": row["duration_ms"],
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
    _: bool = Depends(require_admin_token),
):
    """View run plan details from trade_events journal.

    Shows RUN_STARTED and RUN_COMPLETED event payloads.
    """
    # Query all events for this run_plan_id (correlation_id)
    query = """
        SELECT
            id,
            correlation_id,
            workspace_id,
            event_type,
            created_at,
            payload
        FROM trade_events
        WHERE correlation_id = $1
        ORDER BY created_at ASC
    """

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, run_plan_id)

    if not rows:
        # Graceful error - no 500
        return templates.TemplateResponse(
            "run_plan_detail.html",
            {
                "request": request,
                "run_plan_id": run_plan_id,
                "workspace_id": None,
                "status": "not_found",
                "started_at": None,
                "duration_ms": None,
                "n_variants": None,
                "n_completed": None,
                "n_failed": None,
                "started_event": None,
                "completed_event": None,
                "events": [],
                "error": f"Run plan {run_plan_id[:12]}... not found in trade events journal.",
            },
        )

    # Parse events
    events = []
    started_event = None
    completed_event = None
    workspace_id = None

    for row in rows:
        payload = row["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)

        event_data = {
            "id": str(row["id"]),
            "event_type": row["event_type"],
            "created_at": row["created_at"],
            "payload": payload,
        }
        events.append(event_data)

        if not workspace_id:
            workspace_id = str(row["workspace_id"])

        run_event_type = payload.get("run_event_type")
        if run_event_type == "RUN_STARTED":
            started_event = event_data
        elif run_event_type == "RUN_COMPLETED":
            completed_event = event_data

    # Compute status and metrics
    if completed_event:
        status = "completed"
    elif started_event:
        status = "running"
    else:
        status = "unknown"

    started_at = started_event["created_at"] if started_event else None
    n_variants = started_event["payload"].get("n_variants") if started_event else None
    n_completed = (
        completed_event["payload"].get("n_completed") if completed_event else None
    )
    n_failed = completed_event["payload"].get("n_failed") if completed_event else None
    duration_ms = (
        completed_event["payload"].get("duration_ms") if completed_event else None
    )

    return templates.TemplateResponse(
        "run_plan_detail.html",
        {
            "request": request,
            "run_plan_id": run_plan_id,
            "workspace_id": workspace_id,
            "status": status,
            "started_at": started_at,
            "duration_ms": duration_ms,
            "n_variants": n_variants,
            "n_completed": n_completed,
            "n_failed": n_failed,
            "started_event": started_event,
            "completed_event": completed_event,
            "events": events,
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
    """
    Get a run plan by ID (verification endpoint).

    Returns the full plan data including the plan JSONB.
    """
    repo = _get_run_plans_repo()
    plan = await repo.get_run_plan(plan_id)

    if not plan:
        raise HTTPException(status_code=404, detail="Run plan not found")

    return JSONResponse(content=_json_serializable(plan))


@router.get("/run-plans/{plan_id}/runs")
async def get_run_plan_runs(
    plan_id: UUID,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """
    List backtest_runs for a run plan (no large blobs).

    Does NOT include equity_curve or trades columns to keep response small.
    """
    repo = _get_run_plans_repo()

    # Verify plan exists
    plan = await repo.get_run_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Run plan not found")

    runs, total = await repo.list_runs_for_plan(plan_id, limit=limit, offset=offset)

    return JSONResponse(
        content=_json_serializable(
            {
                "runs": runs,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )
    )


# ===========================================
# KB Trials Promotion Endpoints
# ===========================================


def _get_status_service():
    """Get KBStatusService instance."""
    from app.services.kb.status_service import KBStatusService

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    # Create repository adapters
    from app.admin.kb_trials_repos import (
        AdminKBStatusRepository,
        AdminKBIndexRepository,
    )

    status_repo = AdminKBStatusRepository(_db_pool)
    index_repo = AdminKBIndexRepository(_db_pool)

    return KBStatusService(
        status_repo=status_repo,
        index_repo=index_repo,
    )


@router.get("/kb/trials/promotion-preview")
async def kb_trials_promotion_preview(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    group_id: Optional[UUID] = Query(
        None, description="Filter by group (tune_id or run_plan_id)"
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: str = Query("sharpe_oos", description="Sort field"),
    include_ineligible: bool = Query(False, description="Include ineligible trials"),
    _: bool = Depends(require_admin_token),
):
    """
    Preview trials for promotion consideration.

    Returns trials that could be promoted with eligibility analysis.
    Uses the same candidacy logic as auto-promotion to ensure consistency.
    """
    from app.admin.kb_trials_schemas import (
        PromotionPreviewResponse,
        PromotionPreviewSummary,
        TrialPreviewItem,
    )
    from app.services.kb.candidacy import (
        CandidacyConfig,
        is_candidate,
        VariantMetricsForCandidacy,
        KNOWN_EXPERIMENT_TYPES,
    )
    from app.services.kb.types import RegimeSnapshot

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    # Build query for eligible trials view
    query = """
        SELECT
            source_type,
            experiment_type,
            source_id,
            group_id,
            workspace_id,
            strategy_name,
            params,
            trial_status,
            regime_is,
            regime_oos,
            regime_schema_version,
            sharpe_oos,
            return_frac_oos,
            max_dd_frac_oos,
            n_trades_oos,
            sharpe_is,
            kb_status,
            kb_promoted_at,
            objective_type,
            objective_score,
            created_at
        FROM kb_eligible_trials
        WHERE workspace_id = $1
    """
    params = [workspace_id]
    param_idx = 2

    if source_type:
        query += f" AND source_type = ${param_idx}"
        params.append(source_type)
        param_idx += 1

    if group_id:
        query += f" AND group_id = ${param_idx}"
        params.append(group_id)
        param_idx += 1

    # Also include excluded/candidate trials if include_ineligible
    if include_ineligible:
        # Get trials not in eligible view too
        query = f"""
            WITH eligible AS ({query})
            SELECT * FROM eligible
            UNION ALL
            SELECT
                'test_variant'::TEXT AS source_type,
                COALESCE(r.summary->>'experiment_type', 'sweep')::TEXT AS experiment_type,
                r.id AS source_id,
                r.run_plan_id AS group_id,
                r.workspace_id,
                COALESCE(r.summary->>'strategy_name', e.name) AS strategy_name,
                r.params,
                r.status AS trial_status,
                r.regime_is,
                r.regime_oos,
                r.regime_schema_version,
                (r.summary->>'sharpe')::FLOAT AS sharpe_oos,
                (r.summary->>'return_pct')::FLOAT AS return_frac_oos,
                (r.summary->>'max_drawdown_pct')::FLOAT AS max_dd_frac_oos,
                (r.summary->>'trade_count')::INT AS n_trades_oos,
                NULL::FLOAT AS sharpe_is,
                r.kb_status,
                r.kb_promoted_at,
                COALESCE(rp.objective_name, 'sharpe') AS objective_type,
                r.objective_score,
                r.created_at
            FROM backtest_runs r
            LEFT JOIN kb_entities e ON r.strategy_entity_id = e.id
            LEFT JOIN run_plans rp ON r.run_plan_id = rp.id
            WHERE r.workspace_id = $1
              AND r.run_kind = 'test_variant'
              AND r.kb_status = 'excluded'
              AND r.status IN ('completed', 'success')
        """

    # Add sorting
    sort_column = "sharpe_oos"
    if sort in ("sharpe_oos", "return_frac_oos", "max_dd_frac_oos", "created_at"):
        sort_column = sort

    query += f" ORDER BY {sort_column} DESC NULLS LAST, created_at DESC, source_id"
    query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
    params.extend([limit, offset])

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        # Get total count
        count_query = """
            SELECT COUNT(*) FROM kb_eligible_trials
            WHERE workspace_id = $1
        """
        count_params = [workspace_id]
        if source_type:
            count_query += " AND source_type = $2"
            count_params.append(source_type)
        if group_id:
            count_query += f" AND group_id = ${len(count_params) + 1}"
            count_params.append(group_id)

        total = await conn.fetchval(count_query, *count_params)

    # Process rows with candidacy check
    trials = []
    summary = PromotionPreviewSummary()
    config = CandidacyConfig()

    for row in rows:
        regime_is = None
        regime_oos = None
        if row["regime_is"]:
            regime_is = RegimeSnapshot.from_dict(row["regime_is"])
        if row["regime_oos"]:
            regime_oos = RegimeSnapshot.from_dict(row["regime_oos"])

        # Check candidacy gates
        ineligibility_reasons = []
        passes_gates = False

        experiment_type = row["experiment_type"] or "sweep"

        if experiment_type not in KNOWN_EXPERIMENT_TYPES:
            ineligibility_reasons.append("unknown_experiment_type")
        elif experiment_type == "manual":
            ineligibility_reasons.append("manual_experiment_excluded")
        else:
            # Build metrics for candidacy check
            metrics = VariantMetricsForCandidacy(
                sharpe_oos=row["sharpe_oos"],
                return_frac_oos=row["return_frac_oos"],
                max_dd_frac_oos=row["max_dd_frac_oos"],
                n_trades_oos=row["n_trades_oos"],
                overfit_gap=(
                    max(0, (row["sharpe_is"] or 0) - (row["sharpe_oos"] or 0))
                    if row["sharpe_is"] is not None and row["sharpe_oos"] is not None
                    else None
                ),
            )

            decision = is_candidate(
                metrics=metrics,
                regime_oos=regime_oos,
                experiment_type=experiment_type,
                config=config,
            )

            passes_gates = decision.eligible
            if not decision.eligible:
                ineligibility_reasons.append(decision.reason)

        # Determine promotion eligibility
        kb_status = row["kb_status"]
        can_promote = kb_status not in ("promoted", "rejected")
        is_eligible = passes_gates and can_promote

        # Update summary counts
        if kb_status == "promoted":
            summary.already_promoted += 1
        elif not regime_oos and kb_status != "promoted":
            summary.missing_regime += 1
        elif is_eligible:
            summary.would_promote += 1
        else:
            summary.would_skip += 1

        trials.append(
            TrialPreviewItem(
                source_type=row["source_type"],
                source_id=row["source_id"],
                group_id=row["group_id"],
                experiment_type=experiment_type,
                strategy_name=row["strategy_name"],
                kb_status=kb_status,
                sharpe_oos=row["sharpe_oos"],
                return_frac_oos=row["return_frac_oos"],
                max_dd_frac_oos=row["max_dd_frac_oos"],
                n_trades_oos=row["n_trades_oos"],
                passes_auto_gates=passes_gates,
                can_promote=can_promote,
                is_eligible=is_eligible,
                ineligibility_reasons=ineligibility_reasons,
                has_regime_is=regime_is is not None,
                has_regime_oos=regime_oos is not None,
                regime_schema_version=row["regime_schema_version"],
                created_at=row["created_at"],
            )
        )

    return PromotionPreviewResponse(
        summary=summary,
        pagination={
            "limit": limit,
            "offset": offset,
            "total": total or 0,
        },
        trials=trials,
    )


@router.post("/kb/trials/promote")
async def kb_trials_promote(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """
    Bulk promote trials to 'promoted' status.

    Transitions trials from excluded/candidate to promoted.
    Optionally triggers ingestion for newly promoted trials.
    """
    from app.admin.kb_trials_schemas import (
        BulkStatusRequest,
        BulkStatusResponse,
        StatusChangeResult,
    )

    body = await request.json()
    req = BulkStatusRequest(**body)

    service = _get_status_service()
    results = []
    updated = 0
    skipped = 0
    ingested = 0
    errors = 0

    for source_id in req.source_ids:
        try:
            result = await service.transition(
                source_type=req.source_type,
                source_id=source_id,
                to_status="promoted",
                actor_type="admin",
                actor_id="admin",  # Could be extracted from auth token
                reason=req.reason,
                trigger_ingest=req.trigger_ingest,
            )

            if result.transitioned:
                updated += 1
                if req.trigger_ingest:
                    ingested += 1
            else:
                skipped += 1

            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    group_id=result.group_id if hasattr(result, "group_id") else None,
                    status="promoted",
                )
            )
        except Exception as e:
            errors += 1
            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="error",
                    error=str(e),
                )
            )
            logger.warning(
                "Failed to promote trial",
                source_type=req.source_type,
                source_id=str(source_id),
                error=str(e),
            )

    return BulkStatusResponse(
        updated=updated,
        skipped=skipped,
        ingested=ingested,
        errors=errors,
        results=results,
    )


@router.post("/kb/trials/reject")
async def kb_trials_reject(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """
    Bulk reject trials.

    Transitions trials to 'rejected' status. Requires a reason.
    Archives trials from Qdrant index.
    """
    from app.admin.kb_trials_schemas import (
        BulkStatusRequest,
        BulkStatusResponse,
        StatusChangeResult,
    )

    body = await request.json()
    req = BulkStatusRequest(**body)

    if not req.reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reason is required for rejection",
        )

    service = _get_status_service()
    results = []
    updated = 0
    skipped = 0
    errors = 0

    for source_id in req.source_ids:
        try:
            result = await service.transition(
                source_type=req.source_type,
                source_id=source_id,
                to_status="rejected",
                actor_type="admin",
                actor_id="admin",
                reason=req.reason,
            )

            if result.transitioned:
                updated += 1
            else:
                skipped += 1

            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="rejected",
                )
            )
        except Exception as e:
            errors += 1
            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="error",
                    error=str(e),
                )
            )
            logger.warning(
                "Failed to reject trial",
                source_type=req.source_type,
                source_id=str(source_id),
                error=str(e),
            )

    return BulkStatusResponse(
        updated=updated,
        skipped=skipped,
        ingested=0,
        errors=errors,
        results=results,
    )


@router.post("/kb/trials/mark-candidate")
async def kb_trials_mark_candidate(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """
    Mark trials as candidates.

    Transitions excluded trials to 'candidate' status.
    Does not trigger ingestion - use the ingestion endpoint for that.
    """
    from app.admin.kb_trials_schemas import (
        MarkCandidateRequest,
        BulkStatusResponse,
        StatusChangeResult,
    )

    body = await request.json()
    req = MarkCandidateRequest(**body)

    service = _get_status_service()
    results = []
    updated = 0
    skipped = 0
    errors = 0

    for source_id in req.source_ids:
        try:
            result = await service.transition(
                source_type=req.source_type,
                source_id=source_id,
                to_status="candidate",
                actor_type="admin",
                actor_id="admin",
            )

            if result.transitioned:
                updated += 1
            else:
                skipped += 1

            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="candidate",
                )
            )
        except Exception as e:
            errors += 1
            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="error",
                    error=str(e),
                )
            )
            logger.warning(
                "Failed to mark candidate",
                source_type=req.source_type,
                source_id=str(source_id),
                error=str(e),
            )

    return BulkStatusResponse(
        updated=updated,
        skipped=skipped,
        ingested=0,
        errors=errors,
        results=results,
    )


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
