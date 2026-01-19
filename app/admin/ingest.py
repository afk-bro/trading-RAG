"""Admin ingestion UI - unified interface for all content ingestion sources."""

from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for ingest routes."""
    global _db_pool
    _db_pool = pool


async def _fetch_workspaces() -> list[dict]:
    """Fetch active workspaces for the selector dropdown."""
    if _db_pool is None:
        return []
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, slug FROM workspaces WHERE is_active = true ORDER BY name"
        )
        return [
            {"id": str(row["id"]), "name": row["name"], "slug": row["slug"]}
            for row in rows
        ]


async def _fetch_last_ingests(workspace_id: str) -> dict[str, str]:
    """Fetch last successful ingest timestamp per source type for a workspace.

    Returns:
        Dict mapping source_type -> ISO timestamp string (or empty if none)
    """
    if _db_pool is None or not workspace_id:
        return {}
    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_type, MAX(created_at) as last_ingest
            FROM documents
            WHERE workspace_id = $1
              AND status = 'active'
            GROUP BY source_type
            """,
            UUID(workspace_id),
        )
        return {
            row["source_type"]: row["last_ingest"].isoformat() if row["last_ingest"] else None
            for row in rows
        }


@router.get("/ingest", response_class=HTMLResponse)
async def ingest_page(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Pre-selected workspace"),
    tab: str = Query("youtube", description="Active tab"),
    _: bool = Depends(require_admin_token),
):
    """
    Ingestion UI page with tabbed interface.

    Provides forms for:
    - YouTube: Paste URL to ingest video transcripts
    - PDF: Upload PDF files for extraction and chunking
    - Pine Script: Ingest from registry JSON files
    - Generic: Direct content ingestion with metadata
    """
    workspaces = await _fetch_workspaces()

    # Auto-select first workspace if none provided
    selected_workspace_id = None
    if workspace_id:
        selected_workspace_id = str(workspace_id)
    elif workspaces:
        selected_workspace_id = workspaces[0]["id"]

    # Fetch last ingest timestamps for selected workspace
    last_ingests = await _fetch_last_ingests(selected_workspace_id)

    return templates.TemplateResponse(
        "ingest.html",
        {
            "request": request,
            "workspaces": workspaces,
            "workspace_id": selected_workspace_id,
            "active_tab": tab,
            "last_ingests": last_ingests,
        },
    )
