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
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
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
    2. If LOG_LEVEL is DEBUG, allow access (dev mode)
    3. Otherwise, deny access
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

    # Allow in dev mode (DEBUG log level)
    if settings.log_level.upper() == "DEBUG":
        return True

    # Check if we're in development (localhost)
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

    # Build StrategySpec preview for strategy entities
    strategy_spec = None
    if entity.get("type") == "strategy":
        # Gather all verified claims with spec-relevant types
        spec_types = ["rule", "parameter", "equation", "warning", "assumption"]
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
