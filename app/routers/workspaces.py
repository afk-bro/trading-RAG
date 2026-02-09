"""Workspace management endpoints."""

import json
import re
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.lifespan import get_db_pool
from app.deps.security import check_workspace_consistency

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WorkspaceOut(BaseModel):
    id: str
    name: str
    slug: str
    is_active: bool
    created_at: str


class WorkspaceListResponse(BaseModel):
    workspaces: list[WorkspaceOut]


class CreateWorkspaceBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    slug: Optional[str] = None
    description: Optional[str] = None


class UpdateWorkspaceBody(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert name to a URL-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")[:60]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=WorkspaceListResponse)
async def list_workspaces(pool: Any = Depends(get_db_pool)) -> dict:
    """List all active workspaces."""
    query = """
        SELECT id, name, slug, is_active, created_at
        FROM workspaces
        WHERE is_active = true
        ORDER BY created_at DESC
        LIMIT 50
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    return {
        "workspaces": [
            {
                "id": str(row["id"]),
                "name": row["name"],
                "slug": row["slug"] or "",
                "is_active": row["is_active"],
                "created_at": (
                    row["created_at"].isoformat() if row["created_at"] else ""
                ),
            }
            for row in rows
        ]
    }


@router.post("", response_model=WorkspaceOut, status_code=201)
async def create_workspace(
    body: CreateWorkspaceBody,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """Create a new workspace."""
    ws_id = uuid.uuid4()
    slug = body.slug or _slugify(body.name)

    if not slug:
        raise HTTPException(status_code=422, detail="Could not generate slug from name")

    config = {}
    if body.description:
        config["description"] = body.description

    query = """
        INSERT INTO workspaces (id, name, slug, is_active, ingestion_enabled,
                                default_collection, config)
        VALUES ($1, $2, $3, true, true, 'kb_nomic_embed_text_v1', $4::jsonb)
        RETURNING id, name, slug, is_active, created_at
    """
    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(query, ws_id, body.name, slug, json.dumps(config))
        except Exception as e:
            err = str(e)
            if "unique" in err.lower() or "duplicate" in err.lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"Workspace with slug '{slug}' already exists",
                )
            raise

    return {
        "id": str(row["id"]),
        "name": row["name"],
        "slug": row["slug"] or "",
        "is_active": row["is_active"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
    }


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceOut,
    dependencies=[Depends(check_workspace_consistency)],
)
async def update_workspace(
    workspace_id: str,
    body: UpdateWorkspaceBody,
    pool: Any = Depends(get_db_pool),
) -> dict:
    """Update workspace name and/or description."""
    # Build dynamic SET clauses
    sets: list[str] = []
    params: list[Any] = []
    idx = 1

    if body.name is not None:
        sets.append(f"name = ${idx}")
        params.append(body.name)
        idx += 1

    if body.description is not None:
        sets.append(
            f"config = COALESCE(config, '{{}}'::jsonb) || "
            f"jsonb_build_object('description', ${idx}::text)"
        )
        params.append(body.description)
        idx += 1

    if not sets:
        raise HTTPException(status_code=422, detail="No fields to update")

    params.append(workspace_id)
    query = f"""
        UPDATE workspaces
        SET {', '.join(sets)}
        WHERE id = ${idx}::uuid
        RETURNING id, name, slug, is_active, created_at
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)

    if not row:
        raise HTTPException(status_code=404, detail="Workspace not found")

    return {
        "id": str(row["id"]),
        "name": row["name"],
        "slug": row["slug"] or "",
        "is_active": row["is_active"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
    }
