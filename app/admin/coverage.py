"""Admin endpoints for coverage gap inspection."""

from enum import Enum
from typing import Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token
from app.schemas import StrategyCard

router = APIRouter(prefix="/coverage", tags=["admin-coverage"])
logger = structlog.get_logger(__name__)


class CoverageStatusEnum(str, Enum):
    """Coverage status for triage workflow."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


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
    coverage_status: CoverageStatusEnum = Field(
        default=CoverageStatusEnum.OPEN, description="Triage status"
    )
    priority_score: float = Field(
        default=0.0, description="Priority score for sorting (higher = more urgent)"
    )


class CoverageStatusUpdateRequest(BaseModel):
    """Request for PATCH /admin/coverage/weak/{run_id}."""

    status: CoverageStatusEnum = Field(..., description="New status")
    note: Optional[str] = Field(None, max_length=1000, description="Resolution note")


class CoverageStatusUpdateResponse(BaseModel):
    """Response for PATCH /admin/coverage/weak/{run_id}."""

    run_id: UUID = Field(..., description="Match run ID")
    coverage_status: CoverageStatusEnum = Field(..., description="New status")
    acknowledged_at: Optional[str] = Field(None, description="When acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged")
    resolved_at: Optional[str] = Field(None, description="When resolved")
    resolved_by: Optional[str] = Field(None, description="Who resolved")
    resolution_note: Optional[str] = Field(None, description="Resolution note")


class WeakCoverageResponse(BaseModel):
    """Response for weak coverage list endpoint."""

    items: list[WeakCoverageItem] = Field(default_factory=list)
    count: int = Field(..., description="Number of items returned")
    strategy_cards_by_id: Optional[dict[str, StrategyCard]] = Field(
        None,
        description="Hydrated strategy cards keyed by UUID (include_candidate_cards=true)",
    )
    missing_strategy_ids: list[UUID] = Field(
        default_factory=list,
        description="Strategy IDs referenced but not found (deleted/archived)",
    )


# Max unique strategy IDs to hydrate (prevents payload bloat)
MAX_HYDRATION_IDS = 300


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
    status: Optional[Literal["open", "acknowledged", "resolved", "all"]] = Query(
        None, description="Filter by status (default: open)"
    ),
    include_candidate_cards: bool = Query(
        True, description="Include hydrated strategy cards (default: true for cockpit)"
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
    - candidate_strategy_ids[] (point-in-time snapshot, ordered by recommendation)
    - query_preview (first ~120 chars)
    - source_ref (youtube:ID or doc:UUID)
    - coverage_status (open, acknowledged, resolved)
    - priority_score (higher = more urgent, sorted desc)

    By default (include_candidate_cards=true), also returns:
    - strategy_cards_by_id: {uuid: StrategyCard} for all candidates across items
    - missing_strategy_ids: IDs referenced but not found (deleted/archived)

    Filters:
    - status: open (default), acknowledged, resolved, or all
    - Hydration capped at 300 unique IDs to prevent payload bloat
    - Results sorted by priority_score descending (most actionable first)
    """
    pool = _get_pool()

    from app.services.coverage_gap import MatchRunRepository

    repo = MatchRunRepository(pool)

    try:
        items = await repo.list_weak_coverage_for_cockpit(
            workspace_id=workspace_id,
            limit=limit,
            since=since,
            status=status,
        )
    except Exception as e:
        logger.error("weak_coverage_list_failed", error=str(e))
        raise HTTPException(500, f"Failed to list weak coverage: {e}")

    response_items = [WeakCoverageItem(**item) for item in items]

    # Hydrate strategy cards (default on for cockpit)
    strategy_cards_by_id: Optional[dict[str, StrategyCard]] = None
    missing_strategy_ids: list[UUID] = []

    if include_candidate_cards and items:
        # Collect all unique candidate IDs across all items (preserve order for first N)
        all_candidate_ids: list[UUID] = []
        seen: set[UUID] = set()
        for item in items:
            for cid in item.get("candidate_strategy_ids", []):
                if cid and cid not in seen:
                    seen.add(cid)
                    all_candidate_ids.append(cid)
                    if len(all_candidate_ids) >= MAX_HYDRATION_IDS:
                        break
            if len(all_candidate_ids) >= MAX_HYDRATION_IDS:
                break

        if all_candidate_ids:
            try:
                from app.services.strategy import StrategyRepository
                from app.schemas import (
                    BacktestSummaryStatus,
                    StrategyEngine,
                    StrategyStatus,
                    StrategyTags,
                )

                strategy_repo = StrategyRepository(pool)
                cards_dict = await strategy_repo.get_cards_by_ids(
                    workspace_id, all_candidate_ids
                )

                # Convert to schema
                strategy_cards_by_id = {}
                for uuid_str, card in cards_dict.items():
                    tags_data = card.get("tags") or {}
                    strategy_cards_by_id[uuid_str] = StrategyCard(
                        id=card["id"],
                        name=card["name"],
                        slug=card["slug"],
                        engine=StrategyEngine(card["engine"]),
                        status=StrategyStatus(card["status"]),
                        tags=StrategyTags(**tags_data) if tags_data else StrategyTags(),
                        backtest_status=(
                            BacktestSummaryStatus(card["backtest_status"])
                            if card.get("backtest_status")
                            else None
                        ),
                        last_backtest_at=card.get("last_backtest_at"),
                        best_oos_score=card.get("best_oos_score"),
                        max_drawdown=card.get("max_drawdown"),
                    )

                # Track missing IDs (deleted/archived strategies)
                found_ids = set(cards_dict.keys())
                for cid in all_candidate_ids:
                    if str(cid) not in found_ids:
                        missing_strategy_ids.append(cid)

                logger.info(
                    "hydrated_candidate_cards",
                    requested=len(all_candidate_ids),
                    found=len(strategy_cards_by_id),
                    missing=len(missing_strategy_ids),
                )
            except Exception as e:
                logger.warning("candidate_card_hydration_failed", error=str(e))
                # Don't fail the whole request, just skip hydration

    return WeakCoverageResponse(
        items=response_items,
        count=len(response_items),
        strategy_cards_by_id=strategy_cards_by_id,
        missing_strategy_ids=missing_strategy_ids,
    )


@router.patch(
    "/weak/{run_id}",
    response_model=CoverageStatusUpdateResponse,
    summary="Update coverage status",
    description="Mark a weak coverage run as acknowledged or resolved.",
)
async def update_coverage_status(
    run_id: UUID,
    request: CoverageStatusUpdateRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> CoverageStatusUpdateResponse:
    """
    Update coverage status for a match run.

    Use this to triage weak coverage items:
    - acknowledged: Someone is looking at it
    - resolved: The coverage gap has been addressed
    - open: Reopen if needed
    """
    pool = _get_pool()

    from app.services.coverage_gap import MatchRunRepository

    repo = MatchRunRepository(pool)

    try:
        result = await repo.update_coverage_status(
            run_id=run_id,
            workspace_id=workspace_id,
            status=request.status.value,
            note=request.note,
            updated_by="admin",  # Could be extracted from token in future
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error("coverage_status_update_failed", run_id=str(run_id), error=str(e))
        raise HTTPException(500, f"Failed to update coverage status: {e}")

    if not result:
        raise HTTPException(404, "Match run not found")

    return CoverageStatusUpdateResponse(
        run_id=result["id"],
        coverage_status=CoverageStatusEnum(result["coverage_status"]),
        acknowledged_at=(
            result["acknowledged_at"].isoformat()
            if result.get("acknowledged_at")
            else None
        ),
        acknowledged_by=result.get("acknowledged_by"),
        resolved_at=(
            result["resolved_at"].isoformat() if result.get("resolved_at") else None
        ),
        resolved_by=result.get("resolved_by"),
        resolution_note=result.get("resolution_note"),
    )
