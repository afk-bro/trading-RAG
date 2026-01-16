"""Strategy registry API endpoints."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps.security import require_admin_token
from app.schemas import (
    BacktestSummary,
    CandidateStrategy,
    StrategyCreateRequest,
    StrategyDetailResponse,
    StrategyEngine,
    StrategyListItem,
    StrategyListResponse,
    StrategyReviewStatus,
    StrategyRiskLevel,
    StrategySourceRef,
    StrategyStatus,
    StrategyTags,
    StrategyUpdateRequest,
)
from app.services.strategy import StrategyRepository

router = APIRouter()
logger = structlog.get_logger(__name__)


def _get_pool():
    """Get database pool from ingest router."""
    from app.routers.ingest import _db_pool

    if _db_pool is None:
        raise HTTPException(503, "Database not available")
    return _db_pool


def _row_to_list_item(row: dict) -> StrategyListItem:
    """Convert repository row to list item schema."""
    tags_data = row.get("tags") or {}
    backtest_data = row.get("backtest_summary")

    return StrategyListItem(
        id=row["id"],
        name=row["name"],
        slug=row["slug"],
        engine=StrategyEngine(row["engine"]),
        status=StrategyStatus(row["status"]),
        review_status=StrategyReviewStatus(row["review_status"]),
        risk_level=(
            StrategyRiskLevel(row["risk_level"]) if row.get("risk_level") else None
        ),
        tags=StrategyTags(**tags_data) if tags_data else StrategyTags(),
        backtest_summary=BacktestSummary(**backtest_data) if backtest_data else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_detail(row: dict) -> StrategyDetailResponse:
    """Convert repository row to detail schema."""
    tags_data = row.get("tags") or {}
    backtest_data = row.get("backtest_summary")
    source_ref_data = row.get("source_ref") or {}

    return StrategyDetailResponse(
        id=row["id"],
        workspace_id=row["workspace_id"],
        name=row["name"],
        slug=row["slug"],
        description=row.get("description"),
        engine=StrategyEngine(row["engine"]),
        source_ref=StrategySourceRef(**source_ref_data) if source_ref_data else None,
        status=StrategyStatus(row["status"]),
        review_status=StrategyReviewStatus(row["review_status"]),
        risk_level=(
            StrategyRiskLevel(row["risk_level"]) if row.get("risk_level") else None
        ),
        tags=StrategyTags(**tags_data) if tags_data else StrategyTags(),
        backtest_summary=BacktestSummary(**backtest_data) if backtest_data else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get(
    "",
    response_model=StrategyListResponse,
    summary="List strategies",
    description="List strategies with filters for engine, status, review_status, search.",
)
async def list_strategies(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    engine: Optional[str] = Query(None, description="Filter by engine"),
    status: Optional[str] = Query(None, description="Filter by status"),
    review_status: Optional[str] = Query(None, description="Filter by review_status"),
    q: Optional[str] = Query(None, description="Search name/tags"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
) -> StrategyListResponse:
    """List strategies with filters."""
    pool = _get_pool()
    repo = StrategyRepository(pool)

    strategies, total = await repo.list_strategies(
        workspace_id=workspace_id,
        engine=engine,
        status=status,
        review_status=review_status,
        q=q,
        limit=limit,
        offset=offset,
    )

    items = [_row_to_list_item(row) for row in strategies]

    return StrategyListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(items) < total,
    )


@router.get(
    "/{strategy_id}",
    response_model=StrategyDetailResponse,
    summary="Get strategy details",
    description="Get full details for a single strategy including source_ref and backtest_summary.",
)
async def get_strategy(
    strategy_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> StrategyDetailResponse:
    """Get strategy by ID."""
    pool = _get_pool()
    repo = StrategyRepository(pool)

    row = await repo.get_by_id(strategy_id, workspace_id)
    if not row:
        raise HTTPException(404, "Strategy not found")

    return _row_to_detail(row)


@router.post(
    "",
    response_model=StrategyDetailResponse,
    status_code=201,
    summary="Create strategy",
    description="Create a new strategy in draft status.",
)
async def create_strategy(
    request: StrategyCreateRequest,
    _: bool = Depends(require_admin_token),
) -> StrategyDetailResponse:
    """Create a new strategy."""
    pool = _get_pool()
    repo = StrategyRepository(pool)

    row = await repo.create(
        workspace_id=request.workspace_id,
        name=request.name,
        engine=request.engine.value,
        description=request.description,
        source_ref=request.source_ref.model_dump() if request.source_ref else None,
        status=request.status.value,
        risk_level=request.risk_level.value if request.risk_level else None,
        tags=request.tags.model_dump() if request.tags else None,
    )

    logger.info(
        "strategy_created_via_api",
        strategy_id=str(row["id"]),
        workspace_id=str(request.workspace_id),
        name=request.name,
    )

    return _row_to_detail(row)


@router.patch(
    "/{strategy_id}",
    response_model=StrategyDetailResponse,
    summary="Update strategy",
    description="Update strategy fields: status, review_status, tags, backtest_summary.",
)
async def update_strategy(
    strategy_id: UUID,
    request: StrategyUpdateRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> StrategyDetailResponse:
    """Update strategy fields."""
    pool = _get_pool()
    repo = StrategyRepository(pool)

    # Check exists
    existing = await repo.get_by_id(strategy_id, workspace_id)
    if not existing:
        raise HTTPException(404, "Strategy not found")

    # Build updates dict
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.description is not None:
        updates["description"] = request.description
    if request.status is not None:
        updates["status"] = request.status.value
    if request.review_status is not None:
        updates["review_status"] = request.review_status.value
    if request.risk_level is not None:
        updates["risk_level"] = request.risk_level.value
    if request.source_ref is not None:
        updates["source_ref"] = request.source_ref.model_dump()
    if request.tags is not None:
        updates["tags"] = request.tags.model_dump()
    if request.backtest_summary is not None:
        updates["backtest_summary"] = request.backtest_summary.model_dump()

    row = await repo.update(strategy_id, workspace_id, **updates)

    logger.info(
        "strategy_updated_via_api",
        strategy_id=str(strategy_id),
        fields=list(updates.keys()),
    )

    return _row_to_detail(row)


@router.get(
    "/candidates/by-intent",
    response_model=list[CandidateStrategy],
    summary="Find candidate strategies by intent",
    description="Find strategies with tags that overlap the given intent signature or tags.",
)
async def get_candidates_by_intent(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    intent_signature: Optional[str] = Query(None, description="Intent signature hash"),
    archetypes: Optional[str] = Query(None, description="Comma-separated archetypes"),
    indicators: Optional[str] = Query(None, description="Comma-separated indicators"),
    timeframes: Optional[str] = Query(None, description="Comma-separated timeframes"),
    topics: Optional[str] = Query(None, description="Comma-separated topics"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
    _: bool = Depends(require_admin_token),
) -> list[CandidateStrategy]:
    """Find candidate strategies by intent tags."""
    pool = _get_pool()
    repo = StrategyRepository(pool)

    # Build tags from query params
    intent_tags = {
        "strategy_archetypes": archetypes.split(",") if archetypes else [],
        "indicators": indicators.split(",") if indicators else [],
        "timeframe_buckets": timeframes.split(",") if timeframes else [],
        "topics": topics.split(",") if topics else [],
        "risk_terms": [],
    }

    # If intent_signature provided, could look up cached intent from match_runs
    # For now, use direct tags

    candidates = await repo.find_candidates_by_tags(
        workspace_id=workspace_id,
        intent_tags=intent_tags,
        limit=limit,
    )

    return [
        CandidateStrategy(
            strategy_id=c["strategy_id"],
            name=c["name"],
            score=c["score"],
            matched_tags=c["matched_tags"],
        )
        for c in candidates
    ]
