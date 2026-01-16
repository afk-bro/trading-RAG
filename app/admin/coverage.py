"""Admin endpoints for coverage gap inspection."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token

router = APIRouter(prefix="/coverage", tags=["admin-coverage"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for coverage routes."""
    global _db_pool
    _db_pool = pool


def _get_pool():
    """Get database pool."""
    if _db_pool is None:
        raise HTTPException(503, "Database not available")
    return _db_pool


# Response schemas
class WeakCoverageItem(BaseModel):
    """Single weak coverage run for cockpit display."""

    run_id: UUID = Field(..., description="Match run ID")
    created_at: str = Field(..., description="ISO timestamp")
    intent_signature: str = Field(..., description="SHA256 intent hash")
    script_type: Optional[str] = Field(None, description="Filtered script type")
    weak_reason_codes: list[str] = Field(
        default_factory=list, description="Coverage gap reasons"
    )
    best_score: Optional[float] = Field(None, description="Best match score")
    num_above_threshold: int = Field(..., description="Results above threshold")
    candidate_strategy_ids: list[UUID] = Field(
        default_factory=list, description="Strategy IDs with tag overlap"
    )
    candidate_scores: Optional[dict] = Field(
        None, description="Detailed scores per strategy"
    )
    query_preview: str = Field(..., description="First ~120 chars of query")
    source_ref: Optional[str] = Field(None, description="Source reference for display")


class WeakCoverageResponse(BaseModel):
    """Response for weak coverage list endpoint."""

    items: list[WeakCoverageItem] = Field(default_factory=list)
    count: int = Field(..., description="Number of items returned")


@router.get(
    "/weak",
    response_model=WeakCoverageResponse,
    summary="List recent weak coverage runs",
    description="Returns match runs with weak coverage, shaped for cockpit UI.",
)
async def list_weak_coverage(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    since: Optional[str] = Query(
        None, description="ISO timestamp filter (e.g., 2026-01-01T00:00:00Z)"
    ),
    _: bool = Depends(require_admin_token),
) -> WeakCoverageResponse:
    """
    List recent weak coverage runs for cockpit display.

    Returns rows shaped for UI with:
    - run_id, created_at, intent_signature
    - script_type (from filters)
    - weak_reason_codes[]
    - best_score, num_above_threshold
    - candidate_strategy_ids[] (point-in-time snapshot)
    - query_preview (first ~120 chars)
    - source_ref (youtube:ID or doc:UUID)
    """
    pool = _get_pool()

    from app.services.coverage_gap import MatchRunRepository

    repo = MatchRunRepository(pool)

    try:
        items = await repo.list_weak_coverage_for_cockpit(
            workspace_id=workspace_id,
            limit=limit,
            since=since,
        )
    except Exception as e:
        logger.error("weak_coverage_list_failed", error=str(e))
        raise HTTPException(500, f"Failed to list weak coverage: {e}")

    return WeakCoverageResponse(
        items=[WeakCoverageItem(**item) for item in items],
        count=len(items),
    )
