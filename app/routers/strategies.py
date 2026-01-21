"""Strategy registry API endpoints."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps.security import require_admin_token
from app.repositories.strategy_intel import StrategyIntelRepository, IntelSnapshot
from app.repositories.strategy_versions import StrategyVersionsRepository
from app.services.intel import IntelRunner
from app.schemas import (
    BacktestSummary,
    BacktestSummaryStatus,
    CandidateStrategy,
    StrategyCard,
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
    # Strategy Versions
    StrategyVersionState,
    StrategyVersionCreateRequest,
    StrategyVersionResponse,
    StrategyVersionListItem,
    StrategyVersionListResponse,
    VersionTransitionRequest,
    VersionTransitionResponse,
    # Intel Snapshots
    IntelSnapshotResponse,
    IntelSnapshotListItem,
    IntelSnapshotListResponse,
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


def _card_dict_to_schema(card: dict) -> StrategyCard:
    """Convert repository card dict to StrategyCard schema."""
    tags_data = card.get("tags") or {}

    return StrategyCard(
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
    "/cards",
    response_model=dict[str, StrategyCard],
    summary="Get strategy cards by IDs",
    description="Bulk fetch lightweight strategy cards by IDs. Returns {uuid: card} map.",
)
async def get_strategy_cards(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    ids: str = Query(..., description="Comma-separated strategy UUIDs"),
    _: bool = Depends(require_admin_token),
) -> dict[str, StrategyCard]:
    """
    Bulk fetch strategy cards by IDs.

    Optimized for cockpit UI to avoid N+1 queries when hydrating
    candidate strategies from weak coverage items.

    Returns a dict mapping UUID strings to StrategyCard objects.
    Missing IDs are silently omitted from response.
    """
    pool = _get_pool()
    repo = StrategyRepository(pool)

    # Parse comma-separated UUIDs
    try:
        id_list = [UUID(id_str.strip()) for id_str in ids.split(",") if id_str.strip()]
    except ValueError as e:
        raise HTTPException(422, f"Invalid UUID in ids parameter: {e}")

    if not id_list:
        return {}

    if len(id_list) > 100:
        raise HTTPException(422, "Maximum 100 IDs per request")

    cards_dict = await repo.get_cards_by_ids(workspace_id, id_list)

    # Convert to schema
    return {
        uuid_str: _card_dict_to_schema(card) for uuid_str, card in cards_dict.items()
    }


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
    updates: dict[str, Any] = {}
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
    if row is None:
        raise HTTPException(404, "Strategy not found after update")

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


# =============================================================================
# Strategy Versions (Lifecycle v0.5)
# =============================================================================


def _version_to_response(version) -> StrategyVersionResponse:
    """Convert StrategyVersion dataclass to response schema."""
    return StrategyVersionResponse(
        id=version.id,
        strategy_id=version.strategy_id,
        strategy_entity_id=version.strategy_entity_id,
        version_number=version.version_number,
        version_tag=version.version_tag,
        config_snapshot=version.config_snapshot,
        config_hash=version.config_hash,
        state=StrategyVersionState(version.state),
        regime_awareness=version.regime_awareness,
        created_at=version.created_at,
        created_by=version.created_by,
        activated_at=version.activated_at,
        paused_at=version.paused_at,
        retired_at=version.retired_at,
        kb_strategy_spec_id=version.kb_strategy_spec_id,
    )


def _version_to_list_item(version) -> StrategyVersionListItem:
    """Convert StrategyVersion to list item (lighter response)."""
    return StrategyVersionListItem(
        id=version.id,
        version_number=version.version_number,
        version_tag=version.version_tag,
        state=StrategyVersionState(version.state),
        config_hash=version.config_hash[:16],  # Truncate for list view
        created_at=version.created_at,
        created_by=version.created_by,
        activated_at=version.activated_at,
    )


def _transition_to_response(transition) -> VersionTransitionResponse:
    """Convert VersionTransition dataclass to response schema."""
    return VersionTransitionResponse(
        id=transition.id,
        version_id=transition.version_id,
        from_state=(
            StrategyVersionState(transition.from_state)
            if transition.from_state
            else None
        ),
        to_state=StrategyVersionState(transition.to_state),
        triggered_by=transition.triggered_by,
        triggered_at=transition.triggered_at,
        reason=transition.reason,
    )


@router.post(
    "/{strategy_id}/versions",
    response_model=StrategyVersionResponse,
    status_code=201,
    summary="Create strategy version",
    description="Create a new draft version with immutable config snapshot.",
)
async def create_version(
    strategy_id: UUID,
    request: StrategyVersionCreateRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> StrategyVersionResponse:
    """Create a new strategy version in draft state."""
    pool = _get_pool()

    # Verify strategy exists and belongs to workspace
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    if not strategy.get("strategy_entity_id"):
        raise HTTPException(
            400, "Strategy has no entity_id mapping; cannot create versions"
        )

    version_repo = StrategyVersionsRepository(pool)

    try:
        version = await version_repo.create_version(
            strategy_id=strategy_id,
            config_snapshot=request.config_snapshot,
            created_by=request.created_by or "system",
            version_tag=request.version_tag,
            regime_awareness=request.regime_awareness or {},
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    logger.info(
        "strategy_version_created",
        strategy_id=str(strategy_id),
        version_id=str(version.id),
        version_number=version.version_number,
    )

    return _version_to_response(version)


@router.get(
    "/{strategy_id}/versions",
    response_model=StrategyVersionListResponse,
    summary="List strategy versions",
    description="List all versions for a strategy, newest first.",
)
async def list_versions(
    strategy_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
) -> StrategyVersionListResponse:
    """List versions for a strategy."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    version_repo = StrategyVersionsRepository(pool)
    versions, total = await version_repo.list_versions(
        strategy_id=strategy_id,
        state=state,
        limit=limit,
        offset=offset,
    )

    items = [_version_to_list_item(v) for v in versions]

    return StrategyVersionListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + len(items) < total,
    )


@router.get(
    "/{strategy_id}/versions/{version_id}",
    response_model=StrategyVersionResponse,
    summary="Get strategy version",
    description="Get full details for a specific version.",
)
async def get_version(
    strategy_id: UUID,
    version_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> StrategyVersionResponse:
    """Get a specific version by ID."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    version_repo = StrategyVersionsRepository(pool)
    version = await version_repo.get_version(version_id, strategy_id=strategy_id)
    if not version:
        raise HTTPException(404, "Version not found")

    return _version_to_response(version)


@router.post(
    "/{strategy_id}/versions/{version_id}/activate",
    response_model=StrategyVersionResponse,
    summary="Activate version",
    description="Activate a draft or paused version. Pauses any currently active version.",
)
async def activate_version(
    strategy_id: UUID,
    version_id: UUID,
    request: VersionTransitionRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> StrategyVersionResponse:
    """Activate a version (transitions from draft/paused to active)."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    version_repo = StrategyVersionsRepository(pool)

    try:
        version = await version_repo.activate(
            version_id=version_id,
            triggered_by=request.triggered_by,
            reason=request.reason,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    logger.info(
        "strategy_version_activated",
        strategy_id=str(strategy_id),
        version_id=str(version_id),
        triggered_by=request.triggered_by,
    )

    return _version_to_response(version)


@router.post(
    "/{strategy_id}/versions/{version_id}/pause",
    response_model=StrategyVersionResponse,
    summary="Pause version",
    description="Pause an active version. Clears the active_version_id pointer.",
)
async def pause_version(
    strategy_id: UUID,
    version_id: UUID,
    request: VersionTransitionRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> StrategyVersionResponse:
    """Pause an active version."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    version_repo = StrategyVersionsRepository(pool)

    try:
        version = await version_repo.pause(
            version_id=version_id,
            triggered_by=request.triggered_by,
            reason=request.reason,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    logger.info(
        "strategy_version_paused",
        strategy_id=str(strategy_id),
        version_id=str(version_id),
        triggered_by=request.triggered_by,
    )

    return _version_to_response(version)


@router.post(
    "/{strategy_id}/versions/{version_id}/retire",
    response_model=StrategyVersionResponse,
    summary="Retire version",
    description="Retire a version (terminal state). Cannot be undone.",
)
async def retire_version(
    strategy_id: UUID,
    version_id: UUID,
    request: VersionTransitionRequest,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> StrategyVersionResponse:
    """Retire a version (terminal state)."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    version_repo = StrategyVersionsRepository(pool)

    try:
        version = await version_repo.retire(
            version_id=version_id,
            triggered_by=request.triggered_by,
            reason=request.reason,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    logger.info(
        "strategy_version_retired",
        strategy_id=str(strategy_id),
        version_id=str(version_id),
        triggered_by=request.triggered_by,
    )

    return _version_to_response(version)


@router.get(
    "/{strategy_id}/versions/{version_id}/transitions",
    response_model=list[VersionTransitionResponse],
    summary="Get version transitions",
    description="Get audit trail of state transitions for a version.",
)
async def get_version_transitions(
    strategy_id: UUID,
    version_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    _: bool = Depends(require_admin_token),
) -> list[VersionTransitionResponse]:
    """Get state transition history for a version."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    version_repo = StrategyVersionsRepository(pool)

    # Verify version exists and belongs to this strategy
    version = await version_repo.get_version(version_id, strategy_id=strategy_id)
    if not version:
        raise HTTPException(404, "Version not found")

    transitions = await version_repo.get_transitions(version_id, limit=limit)

    return [_transition_to_response(t) for t in transitions]


# =============================================================================
# Strategy Intel Snapshots (v1.5)
# =============================================================================


def _snapshot_to_response(snapshot: IntelSnapshot) -> IntelSnapshotResponse:
    """Convert IntelSnapshot dataclass to response schema."""
    return IntelSnapshotResponse(
        id=snapshot.id,
        workspace_id=snapshot.workspace_id,
        strategy_version_id=snapshot.strategy_version_id,
        as_of_ts=snapshot.as_of_ts,
        computed_at=snapshot.computed_at,
        regime=snapshot.regime,
        confidence_score=snapshot.confidence_score,
        confidence_components=snapshot.confidence_components,
        features=snapshot.features,
        explain=snapshot.explain,
        engine_version=snapshot.engine_version,
        inputs_hash=snapshot.inputs_hash,
        run_id=snapshot.run_id,
    )


def _snapshot_to_list_item(snapshot: IntelSnapshot) -> IntelSnapshotListItem:
    """Convert IntelSnapshot to list item (lighter response)."""
    return IntelSnapshotListItem(
        id=snapshot.id,
        strategy_version_id=snapshot.strategy_version_id,
        as_of_ts=snapshot.as_of_ts,
        computed_at=snapshot.computed_at,
        regime=snapshot.regime,
        confidence_score=snapshot.confidence_score,
    )


@router.get(
    "/{strategy_id}/versions/{version_id}/intel/latest",
    response_model=IntelSnapshotResponse,
    summary="Get latest intel snapshot",
    description="Get the most recent intelligence snapshot for a strategy version.",
)
async def get_latest_intel(
    strategy_id: UUID,
    version_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    _: bool = Depends(require_admin_token),
) -> IntelSnapshotResponse:
    """Get the latest intel snapshot for a strategy version."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    # Verify version exists and belongs to this strategy
    version_repo = StrategyVersionsRepository(pool)
    version = await version_repo.get_version(version_id, strategy_id=strategy_id)
    if not version:
        raise HTTPException(404, "Version not found")

    # Get latest snapshot
    intel_repo = StrategyIntelRepository(pool)
    snapshot = await intel_repo.get_latest_snapshot(version_id)
    if not snapshot:
        raise HTTPException(404, "No intel snapshot found for this version")

    return _snapshot_to_response(snapshot)


@router.get(
    "/{strategy_id}/versions/{version_id}/intel",
    response_model=IntelSnapshotListResponse,
    summary="List intel snapshots",
    description="List intel snapshots for a version with cursor-based pagination.",
)
async def list_intel_snapshots(
    strategy_id: UUID,
    version_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    cursor: Optional[datetime] = Query(
        None, description="Cursor (as_of_ts) for pagination"
    ),
    _: bool = Depends(require_admin_token),
) -> IntelSnapshotListResponse:
    """List intel snapshots for a strategy version (newest first)."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    # Verify version exists and belongs to this strategy
    version_repo = StrategyVersionsRepository(pool)
    version = await version_repo.get_version(version_id, strategy_id=strategy_id)
    if not version:
        raise HTTPException(404, "Version not found")

    # List snapshots
    intel_repo = StrategyIntelRepository(pool)
    snapshots = await intel_repo.list_snapshots(
        strategy_version_id=version_id,
        limit=limit + 1,  # Fetch one extra to check for more pages
        cursor=cursor,
    )

    # Determine if there are more results
    has_more = len(snapshots) > limit
    if has_more:
        snapshots = snapshots[:limit]

    items = [_snapshot_to_list_item(s) for s in snapshots]

    # Set next cursor to oldest item's as_of_ts
    next_cursor = snapshots[-1].as_of_ts if has_more else None

    return IntelSnapshotListResponse(
        items=items,
        total=len(items),  # Note: this is page size, not total count
        limit=limit,
        next_cursor=next_cursor,
    )


@router.post(
    "/{strategy_id}/versions/{version_id}/intel/recompute",
    response_model=IntelSnapshotResponse,
    summary="Recompute intel snapshot",
    description="Trigger recomputation of intel for a strategy version at current time.",
)
async def recompute_intel(
    strategy_id: UUID,
    version_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    force: bool = Query(False, description="Force even if inputs unchanged"),
    _: bool = Depends(require_admin_token),
) -> IntelSnapshotResponse:
    """Recompute intel for a strategy version."""
    pool = _get_pool()

    # Verify strategy exists
    strategy_repo = StrategyRepository(pool)
    strategy = await strategy_repo.get_by_id(strategy_id, workspace_id)
    if not strategy:
        raise HTTPException(404, "Strategy not found")

    # Verify version exists and belongs to this strategy
    version_repo = StrategyVersionsRepository(pool)
    version = await version_repo.get_version(version_id, strategy_id=strategy_id)
    if not version:
        raise HTTPException(404, "Version not found")

    # Run intel computation
    as_of_ts = datetime.now(timezone.utc)
    runner = IntelRunner(pool)

    try:
        snapshot = await runner.run_for_version(
            version_id=version_id,
            as_of_ts=as_of_ts,
            workspace_id=workspace_id,
            force=force,
        )
    except Exception as e:
        logger.error(
            "intel_recompute_failed",
            version_id=str(version_id),
            error=str(e),
        )
        raise HTTPException(500, f"Intel computation failed: {e}")

    if snapshot is None:
        # Deduplication kicked in - return the existing latest
        intel_repo = StrategyIntelRepository(pool)
        existing = await intel_repo.get_latest_snapshot(version_id)
        if existing:
            return _snapshot_to_response(existing)
        raise HTTPException(
            500, "Computation returned None but no existing snapshot found"
        )

    logger.info(
        "intel_recompute_success",
        version_id=str(version_id),
        snapshot_id=str(snapshot.id),
        regime=snapshot.regime,
        confidence=snapshot.confidence_score,
    )

    return _snapshot_to_response(snapshot)
