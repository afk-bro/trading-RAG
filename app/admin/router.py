"""Admin UI router for KB inspection and curation."""

import json
import os
from datetime import datetime
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
import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings
from app.schemas import KBEntityType, KBClaimType, KBClaimStatus

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for admin routes."""
    global _db_pool
    _db_pool = pool


def _get_kb_repo():
    """Get KnowledgeBaseRepository instance."""
    from app.repositories.kb import KnowledgeBaseRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return KnowledgeBaseRepository(_db_pool)


async def verify_admin_access(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """
    Verify admin access.

    Security checks:
    1. If ADMIN_TOKEN is set, require it in header or query param
    2. In local development (localhost only), allow access
    3. Otherwise, deny access

    NOTE: LOG_LEVEL=DEBUG does NOT bypass auth. It only controls log verbosity.
    """
    admin_token = os.environ.get("ADMIN_TOKEN")

    if admin_token:
        # Check header first, then query param
        provided_token = request.headers.get("X-Admin-Token")
        if not provided_token:
            provided_token = request.query_params.get("token")

        if provided_token != admin_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid or missing admin token",
            )
        return True

    # Allow in local development only (not DEBUG mode - that's a verbosity setting)
    host = request.headers.get("host", "")
    if "localhost" in host or "127.0.0.1" in host:
        return True

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access not allowed. Set ADMIN_TOKEN or use localhost.",
    )


# ===========================================
# Admin Routes
# ===========================================


@router.get("/", response_class=HTMLResponse)
async def admin_home(
    request: Request,
    _: bool = Depends(verify_admin_access),
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
    _: bool = Depends(verify_admin_access),
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
    _: bool = Depends(verify_admin_access),
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
            "strategy_spec": json.dumps(strategy_spec, indent=2) if strategy_spec else None,
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
    _: bool = Depends(verify_admin_access),
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
    _: bool = Depends(verify_admin_access),
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
    oos_enabled: Optional[str] = Query(None, description="Filter by OOS: 'true', 'false', or empty"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: bool = Depends(verify_admin_access),
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
    valid_only: bool = Query(True, description="Only tunes with valid results (default True)"),
    include_canceled: bool = Query(False, description="Include canceled tunes"),
    objective_type: Optional[str] = Query(None, description="Filter by objective type"),
    oos_enabled: Optional[str] = Query(None, description="Filter by OOS: 'true' or 'false'"),
    format: Optional[str] = Query(None, description="Output format: 'csv' for download"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _: bool = Depends(verify_admin_access),
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
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
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
    format: Optional[str] = Query(None, description="Output format: 'json' for download"),
    _: bool = Depends(verify_admin_access),
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

    rows.append({
        "label": "Strategy",
        "values": [t.get("strategy_name") or t["strategy_entity_id"][:8] + "..." for t in tunes],
        "fmt": "default",
    })
    rows.append({
        "label": "Status",
        "values": [t["status"] for t in tunes],
        "fmt": "default",
    })
    rows.append({
        "label": "Objective Type",
        "values": [t.get("objective_type") or "sharpe" for t in tunes],
        "fmt": "default",
    })

    # Extract dd_lambda from objective_params
    dd_lambdas = []
    for t in tunes:
        params = t.get("objective_params") or {}
        dd_lambdas.append(params.get("dd_lambda"))
    rows.append({
        "label": "λ (dd_lambda)",
        "values": [_normalize_compare_value(v, "float2") for v in dd_lambdas],
        "raw_values": dd_lambdas,
        "fmt": "float2",
    })

    rows.append({
        "label": "OOS Ratio",
        "values": [_normalize_compare_value(t.get("oos_ratio"), "pct_ratio") for t in tunes],
        "raw_values": [t.get("oos_ratio") for t in tunes],
        "fmt": "pct_ratio",
    })

    # Gates
    rows.append({
        "label": "Gates: Max DD",
        "values": [_normalize_compare_value((t.get("gates") or {}).get("max_drawdown_pct"), "int") + "%"
                   if (t.get("gates") or {}).get("max_drawdown_pct") is not None else "—" for t in tunes],
        "raw_values": [(t.get("gates") or {}).get("max_drawdown_pct") for t in tunes],
        "fmt": "default",
    })
    rows.append({
        "label": "Gates: Min Trades",
        "values": [_normalize_compare_value((t.get("gates") or {}).get("min_trades"), "int") for t in tunes],
        "raw_values": [(t.get("gates") or {}).get("min_trades") for t in tunes],
        "fmt": "int",
    })
    rows.append({
        "label": "Gates: Evaluated On",
        "values": [(t.get("gates") or {}).get("evaluated_on") or "—" for t in tunes],
        "fmt": "default",
    })

    # --- Section: Winning Metrics ---
    rows.append({"section": "Winning Metrics", "is_header": True})

    # Extract metrics from best_metrics_oos
    for t in tunes:
        t["_metrics"] = t.get("best_metrics_oos") or {}

    rows.append({
        "label": "Objective Score",
        "values": [_normalize_compare_value(t.get("best_objective_score") or t.get("score_oos") or t.get("best_score"), "float4") for t in tunes],
        "fmt": "float4",
    })
    rows.append({
        "label": "Return %",
        "values": [_normalize_compare_value(t["_metrics"].get("return_pct"), "pct") for t in tunes],
        "raw_values": [t["_metrics"].get("return_pct") for t in tunes],
        "fmt": "pct",
    })
    rows.append({
        "label": "Sharpe",
        "values": [_normalize_compare_value(t["_metrics"].get("sharpe"), "float2") for t in tunes],
        "raw_values": [t["_metrics"].get("sharpe") for t in tunes],
        "fmt": "float2",
    })
    rows.append({
        "label": "Max DD %",
        "values": [_normalize_compare_value(t["_metrics"].get("max_drawdown_pct"), "pct_neg") for t in tunes],
        "raw_values": [t["_metrics"].get("max_drawdown_pct") for t in tunes],
        "fmt": "pct_neg",
    })
    rows.append({
        "label": "Trades",
        "values": [_normalize_compare_value(t["_metrics"].get("trades"), "int") for t in tunes],
        "raw_values": [t["_metrics"].get("trades") for t in tunes],
        "fmt": "int",
    })

    # Overfit gap with special styling
    overfit_gaps = [t.get("overfit_gap") for t in tunes]
    rows.append({
        "label": "Overfit Gap",
        "values": [_normalize_compare_value(g, "float4") for g in overfit_gaps],
        "raw_values": overfit_gaps,
        "classes": [_overfit_class(g) for g in overfit_gaps],
        "fmt": "float4",
    })

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
        rows.append({
            "label": key,
            "values": param_values,
            "is_param": True,
            "fmt": "default",
        })

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
            "workspace_id": str(workspace_id) if workspace_id else tunes[0]["workspace_id"],
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
            "created_at": tune.get("created_at").isoformat() if tune.get("created_at") else None,
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
    tune_id_parts = [t.get("id", "")[:8] for t in tunes[:3]]  # First 3 tune IDs, 8 chars each
    filename = f"compare_{'_'.join(tune_id_parts)}_{timestamp}.json"

    return JSONResponse(
        content=_json_serializable(export_data),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@router.get("/backtests/tunes/{tune_id}", response_class=HTMLResponse)
async def admin_tune_detail(
    request: Request,
    tune_id: UUID,
    run_status: Optional[str] = Query(None, description="Filter trials by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: bool = Depends(verify_admin_access),
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
                {"reason": row["skip_reason"], "count": row["count"]}
                for row in rows
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
    _: bool = Depends(verify_admin_access),
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
    _: None = Depends(verify_admin_access),
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
        "p50_rerank_ms": round(summary.p50_rerank_ms, 1) if summary.p50_rerank_ms else None,
        "p95_rerank_ms": round(summary.p95_rerank_ms, 1) if summary.p95_rerank_ms else None,
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
    _: None = Depends(verify_admin_access),
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
    _: None = Depends(verify_admin_access),
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
                "rank_delta_mean": round(q.rank_delta_mean, 2) if q.rank_delta_mean else None,
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
    _: None = Depends(verify_admin_access),
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
    since: Optional[datetime] = Query(None, description="Only count trials after this time"),
    window_days: Optional[int] = Query(None, ge=1, le=365, description="Time window in days (7, 30, etc.)"),
    _: None = Depends(verify_admin_access),
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
            total = await conn.fetchval(f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE 1=1 {workspace_cond} {time_filter}
            """, *params) or 0

            if total == 0:
                return {
                    "total": 0, "with_oos": 0, "valid": 0, "stale": 0,
                    "with_regime_is": 0, "with_regime_oos": 0,
                    "with_objective_score": 0, "with_sharpe_oos": 0,
                }

            # Core metrics
            with_oos = await conn.fetchval(f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.has_oos_metrics = true {workspace_cond} {time_filter}
            """, *params) or 0

            valid = await conn.fetchval(f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.is_valid = true {workspace_cond} {time_filter}
            """, *params) or 0

            # Stale count
            try:
                stale = await conn.fetchval(f"""
                    SELECT COUNT(*) FROM kb_trial_vectors t
                    WHERE t.needs_reembed = true {workspace_cond} {time_filter}
                """, *params) or 0
            except Exception:
                stale = 0

            # Coverage metrics
            with_regime_is = await conn.fetchval(f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_is IS NOT NULL {workspace_cond} {time_filter}
            """, *params) or 0

            with_regime_oos = await conn.fetchval(f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """, *params) or 0

            with_objective = await conn.fetchval(f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.objective_score IS NOT NULL {workspace_cond} {time_filter}
            """, *params) or 0

            with_sharpe_oos = await conn.fetchval(f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.sharpe_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """, *params) or 0

            return {
                "total": total, "with_oos": with_oos, "valid": valid, "stale": stale,
                "with_regime_is": with_regime_is, "with_regime_oos": with_regime_oos,
                "with_objective_score": with_objective, "with_sharpe_oos": with_sharpe_oos,
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
            time_filter = f"AND t.created_at >= ${param_idx} AND t.created_at < ${param_idx + 1}"
            previous = await get_stats(time_filter, [prev_since, since])

            # Calculate deltas
            deltas = {
                "trials_added": current["total"] - previous["total"],
                "valid_added": current["valid"] - previous["valid"],
                "stale_added": current["stale"] - previous["stale"],
                "pct_valid_delta": round(
                    (current["valid"] / current["total"] * 100 if current["total"] > 0 else 0) -
                    (previous["valid"] / previous["total"] * 100 if previous["total"] > 0 else 0),
                    1
                ),
                "window_days": window_days,
                "prev_window_start": prev_since.isoformat(),
                "prev_window_end": since.isoformat(),
            }

        # Last ingestion timestamp
        last_ts = await conn.fetchval(f"""
            SELECT MAX(t.created_at) FROM kb_trial_vectors t
            WHERE 1=1 {workspace_cond}
        """, *base_params)

        # Workspace config for embedding info
        embed_model = "nomic-embed-text"
        embed_dim = 768
        collection_name = "trading_kb_trials__nomic-embed-text__768"

        if workspace_id:
            config_row = await conn.fetchrow("""
                SELECT config FROM workspaces WHERE id = $1
            """, workspace_id)
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
    _: None = Depends(verify_admin_access),
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
    _: None = Depends(verify_admin_access),
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
                distance = vec_cfg.distance.value if vec_cfg and vec_cfg.distance else None

                # Payload indexes count
                payload_indexes = 0
                if info.payload_schema:
                    payload_indexes = len(info.payload_schema)

                # Optimizer status
                optimizer_status = "unknown"
                if info.optimizer_status:
                    optimizer_status = info.optimizer_status.status.value if hasattr(info.optimizer_status, 'status') else str(info.optimizer_status)

                result.append({
                    "name": coll.name,
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                    "status": info.status.value if info.status else "unknown",
                    "vector_size": vector_size or embedding_dim,
                    "distance": distance,
                    "embedding_model_id": embedding_model,
                    "payload_indexes_count": payload_indexes,
                    "optimizer_status": optimizer_status,
                    "segments_count": len(info.segments or []) if hasattr(info, 'segments') else None,
                })
            except Exception as e:
                result.append({
                    "name": coll.name,
                    "error": str(e),
                })

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
    _: None = Depends(verify_admin_access),
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
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM kb_trial_vectors WHERE 1=1 {workspace_filter}) as pct
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
    warning: Optional[str] = Query(None, description="Filter by warning type (e.g., high_overfit)"),
    is_valid: Optional[bool] = Query(None, description="Filter by validity"),
    has_oos: Optional[bool] = Query(None, description="Filter by OOS availability"),
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(20, ge=1, le=100, description="Number of samples to return"),
    _: None = Depends(verify_admin_access),
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
                    "tune_run_id": str(row["tune_run_id"]) if row["tune_run_id"] else None,
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
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
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
