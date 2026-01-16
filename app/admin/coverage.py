"""Admin endpoints for coverage gap inspection."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token
from app.schemas import StrategyCard

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
    strategy_cards_by_id: Optional[dict[str, StrategyCard]] = Field(
        None,
        description="Hydrated strategy cards keyed by UUID (include_candidate_cards=true)",
    )


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
    include_candidate_cards: bool = Query(
        False, description="Include hydrated strategy cards in response"
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

    When include_candidate_cards=true, also returns:
    - strategy_cards_by_id: {uuid: StrategyCard} for all candidates across items
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

    response_items = [WeakCoverageItem(**item) for item in items]

    # Optionally hydrate strategy cards
    strategy_cards_by_id = None
    if include_candidate_cards and items:
        # Collect all unique candidate IDs across all items
        all_candidate_ids: set[UUID] = set()
        for item in items:
            for cid in item.get("candidate_strategy_ids", []):
                if cid:
                    all_candidate_ids.add(cid)

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
                    workspace_id, list(all_candidate_ids)
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

                logger.info(
                    "hydrated_candidate_cards",
                    requested=len(all_candidate_ids),
                    found=len(strategy_cards_by_id),
                )
            except Exception as e:
                logger.warning("candidate_card_hydration_failed", error=str(e))
                # Don't fail the whole request, just skip hydration

    return WeakCoverageResponse(
        items=response_items,
        count=len(response_items),
        strategy_cards_by_id=strategy_cards_by_id,
    )
