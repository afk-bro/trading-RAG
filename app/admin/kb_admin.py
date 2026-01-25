"""KB entities and claims admin UI endpoints."""

import json
from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.admin.utils import (
    PaginationDefaults,
    json_serializable,
    parse_json_field,
    require_db_pool,
)
from app.deps.security import require_admin_token
from app.schemas import KBClaimType, KBEntityType

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for KB admin routes."""
    global _db_pool
    _db_pool = pool


def _get_kb_repo():
    """Get KnowledgeBaseRepository instance."""
    from app.repositories.kb import KnowledgeBaseRepository

    pool = require_db_pool(_db_pool, "Database")
    return KnowledgeBaseRepository(pool)


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
    limit: int = Query(
        PaginationDefaults.DETAIL_DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.DETAIL_MAX_LIMIT,
    ),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List KB entities with search and filters."""
    kb_repo = _get_kb_repo()

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
    for entity in entities:
        entity["aliases"] = parse_json_field(entity.get("aliases")) or []

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
    limit: int = Query(
        PaginationDefaults.DETAIL_DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.DETAIL_MAX_LIMIT,
    ),
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
    entity["aliases"] = parse_json_field(entity.get("aliases")) or []

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
    claim_json = json_serializable(claim)

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
    limit: int = Query(
        PaginationDefaults.DETAIL_DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.DETAIL_MAX_LIMIT,
    ),
    offset: int = Query(0, ge=0),
    _: bool = Depends(require_admin_token),
):
    """List all claims with search and filters."""
    kb_repo = _get_kb_repo()

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
