"""Walk-Forward Optimization (WFO) endpoints."""

import json
from typing import Optional
from uuid import UUID

import structlog
from fastapi import (
    APIRouter,
    Form,
    HTTPException,
    Query,
    status,
)

from .schemas import (
    CancelWFOResponse,
    WFOCandidateModel,
    WFOConfigModel,
    WFOListItem,
    WFOListResponse,
    WFOResponse,
)

router = APIRouter(tags=["backtests"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_repos():
    """Get repository instances."""
    from app.repositories.backtests import BacktestRepository, TuneRepository
    from app.repositories.kb import KnowledgeBaseRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return (
        KnowledgeBaseRepository(_db_pool),
        BacktestRepository(_db_pool),
        TuneRepository(_db_pool),
    )


def _get_wfo_repo():
    """Get WFO repository instance."""
    from app.repositories.backtests import WFORepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return WFORepository(_db_pool)


@router.post(
    "/wfo",
    response_model=WFOResponse,
    responses={
        202: {"description": "WFO job queued successfully"},
        404: {"description": "Strategy spec not found"},
        422: {"description": "Invalid parameters"},
        503: {"description": "Database unavailable"},
    },
    summary="Start Walk-Forward Optimization",
    description="Queue a WFO job to validate strategy robustness across rolling time windows.",
)
async def create_wfo(
    strategy_entity_id: UUID = Form(..., description="Strategy entity UUID"),
    workspace_id: UUID = Form(..., description="Workspace UUID"),
    train_days: int = Form(default=90, ge=1, description="Days in training window"),
    test_days: int = Form(default=30, ge=1, description="Days in test window"),
    step_days: int = Form(default=30, ge=1, description="Days to step between folds"),
    min_folds: int = Form(default=3, ge=1, description="Minimum folds required"),
    leaderboard_top_k: int = Form(
        default=10, ge=1, description="Top-K params per fold"
    ),
    allow_partial: bool = Form(
        default=False, description="Continue if some folds fail"
    ),
    param_space: Optional[str] = Form(
        default=None, description="Parameter space JSON (auto-derived if omitted)"
    ),
    search_type: str = Form(default="grid", description="Search type: grid or random"),
    n_trials: int = Form(default=50, ge=1, le=200, description="Trials per fold"),
    objective_type: str = Form(default="sharpe", description="Objective function type"),
    exchange_id: str = Form(..., description="Exchange ID for OHLCV data"),
    symbol: str = Form(..., description="Symbol for OHLCV data"),
    timeframe: str = Form(default="1h", description="Timeframe for OHLCV data"),
    start_ts: Optional[str] = Form(default=None, description="Start timestamp (ISO)"),
    end_ts: Optional[str] = Form(default=None, description="End timestamp (ISO)"),
    initial_cash: float = Form(default=10000, ge=100),
    commission_bps: float = Form(default=10, ge=0, le=1000),
    slippage_bps: float = Form(default=0, ge=0, le=1000),
    min_trades: int = Form(default=5, ge=1),
    seed: Optional[int] = Form(default=None),
):
    """
    Start a Walk-Forward Optimization session.

    WFO validates strategy robustness by:
    1. Generating rolling train/test folds across the data range
    2. Running parameter tuning on each fold's training period
    3. Aggregating results to find params that work across all market conditions

    Returns immediately with wfo_id. Use GET /backtests/wfo/{wfo_id} to poll status.
    """
    from app.jobs.types import JobType
    from app.repositories.jobs import JobRepository

    kb_repo, _, _ = _get_repos()
    wfo_repo = _get_wfo_repo()
    job_repo = JobRepository(_db_pool)

    # Validate search type
    if search_type not in ("grid", "random"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "detail": "search_type must be 'grid' or 'random'",
                "code": "INVALID_SEARCH_TYPE",
            },
        )

    # Parse param_space if provided
    parsed_param_space = None
    if param_space:
        try:
            parsed_param_space = json.loads(param_space)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "detail": f"Invalid param_space JSON: {e}",
                    "code": "INVALID_PARAM_SPACE",
                },
            )

    # Verify strategy exists
    spec = await kb_repo.get_strategy_spec(strategy_entity_id)
    if not spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy spec found for entity {strategy_entity_id}",
        )

    # Build WFO config
    wfo_config = {
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "min_folds": min_folds,
        "leaderboard_top_k": leaderboard_top_k,
        "allow_partial": allow_partial,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }

    # Build data source
    data_source = {
        "exchange_id": exchange_id,
        "symbol": symbol,
        "timeframe": timeframe,
    }

    # Create WFO record
    wfo_id = await wfo_repo.create_wfo(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        wfo_config=wfo_config,
        param_space=parsed_param_space,
        data_source=data_source,
    )

    # Build job payload
    job_payload = {
        "workspace_id": str(workspace_id),
        "wfo_id": str(wfo_id),
        "strategy_entity_id": str(strategy_entity_id),
        "data_source": data_source,
        "wfo_config": wfo_config,
        "param_space": parsed_param_space,
        "search_type": search_type,
        "n_trials": n_trials,
        "objective_type": objective_type,
        "objective_metric": objective_type.split("_")[0],  # sharpe_dd_penalty -> sharpe
        "initial_cash": initial_cash,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
        "min_trades": min_trades,
        "seed": seed,
    }

    # Enqueue WFO job
    job = await job_repo.enqueue(
        job_type=JobType.WFO,
        payload=job_payload,
        workspace_id=workspace_id,
    )

    # Link job to WFO record
    await wfo_repo.update_wfo_started(wfo_id, n_folds=0, job_id=job.id)

    logger.info(
        "WFO job queued",
        wfo_id=str(wfo_id),
        job_id=str(job.id),
        strategy_entity_id=str(strategy_entity_id),
    )

    return WFOResponse(
        wfo_id=str(wfo_id),
        status="pending",
        n_folds=0,
        folds_completed=0,
        folds_failed=0,
        warnings=[],
    )


@router.get(
    "/wfo/{wfo_id}",
    response_model=WFOResponse,
    summary="Get WFO run details",
)
async def get_wfo(wfo_id: UUID):
    """Get details of a WFO run including candidates and status."""
    wfo_repo = _get_wfo_repo()

    wfo = await wfo_repo.get_wfo(wfo_id)
    if not wfo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"WFO run {wfo_id} not found",
        )

    # Parse candidates
    candidates = []
    for c in wfo.get("candidates") or []:
        candidates.append(
            WFOCandidateModel(
                params=c.get("params", {}),
                params_hash=c.get("params_hash", ""),
                mean_oos=c.get("mean_oos", 0),
                median_oos=c.get("median_oos", 0),
                worst_fold_oos=c.get("worst_fold_oos", 0),
                stddev_oos=c.get("stddev_oos", 0),
                pct_top_k=c.get("pct_top_k", 0),
                fold_count=c.get("fold_count", 0),
                total_folds=c.get("total_folds", 0),
                coverage=c.get("coverage", 0),
                regime_tags=c.get("regime_tags", []),
            )
        )

    # Parse best_candidate
    best_candidate = None
    if wfo.get("best_candidate"):
        bc = wfo["best_candidate"]
        best_candidate = WFOCandidateModel(
            params=bc.get("params", {}),
            params_hash=bc.get("params_hash", ""),
            mean_oos=bc.get("mean_oos", 0),
            median_oos=bc.get("median_oos", 0),
            worst_fold_oos=bc.get("worst_fold_oos", 0),
            stddev_oos=bc.get("stddev_oos", 0),
            pct_top_k=bc.get("pct_top_k", 0),
            fold_count=bc.get("fold_count", 0),
            total_folds=bc.get("total_folds", 0),
            coverage=bc.get(
                "coverage", bc.get("fold_count", 0) / max(bc.get("total_folds", 1), 1)
            ),
            regime_tags=bc.get("regime_tags", []),
        )

    child_tune_ids = [str(tid) for tid in (wfo.get("child_tune_ids") or [])]

    return WFOResponse(
        wfo_id=str(wfo["id"]),
        status=wfo["status"],
        n_folds=wfo.get("n_folds", 0),
        folds_completed=wfo.get("folds_completed", 0),
        folds_failed=wfo.get("folds_failed", 0),
        best_params=wfo.get("best_params"),
        best_candidate=best_candidate,
        candidates=candidates,
        child_tune_ids=child_tune_ids,
        warnings=wfo.get("warnings") or [],
        created_at=wfo.get("created_at"),
        started_at=wfo.get("started_at"),
        completed_at=wfo.get("completed_at"),
    )


@router.get(
    "/wfo",
    response_model=WFOListResponse,
    summary="List WFO runs",
)
async def list_wfos(
    workspace_id: UUID = Query(..., description="Workspace UUID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List WFO runs with optional filtering."""
    wfo_repo = _get_wfo_repo()

    wfos, total = await wfo_repo.list_wfos(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    items = []
    for wfo in wfos:
        config = wfo.get("wfo_config") or {}
        items.append(
            WFOListItem(
                id=str(wfo["id"]),
                created_at=wfo["created_at"],
                strategy_entity_id=str(wfo["strategy_entity_id"]),
                strategy_name=wfo.get("strategy_name"),
                status=wfo["status"],
                n_folds=wfo.get("n_folds", 0),
                folds_completed=wfo.get("folds_completed", 0),
                folds_failed=wfo.get("folds_failed", 0),
                wfo_config=WFOConfigModel(
                    train_days=config.get("train_days", 90),
                    test_days=config.get("test_days", 30),
                    step_days=config.get("step_days", 30),
                    min_folds=config.get("min_folds", 3),
                    leaderboard_top_k=config.get("leaderboard_top_k", 10),
                    allow_partial=config.get("allow_partial", False),
                ),
                best_params=wfo.get("best_params"),
            )
        )

    return WFOListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/wfo/{wfo_id}/cancel",
    response_model=CancelWFOResponse,
    summary="Cancel a WFO run",
)
async def cancel_wfo(wfo_id: UUID):
    """
    Cancel a running or pending WFO run.

    - Sets WFO status to 'canceled'
    - Running child jobs will complete but new folds are skipped
    """
    wfo_repo = _get_wfo_repo()

    # Check WFO exists and is cancelable
    wfo = await wfo_repo.get_wfo(wfo_id)
    if not wfo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"WFO run {wfo_id} not found",
        )

    if wfo["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel WFO in '{wfo['status']}' state",
        )

    canceled = await wfo_repo.cancel_wfo(wfo_id)
    if not canceled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="WFO could not be canceled (may have completed)",
        )

    logger.info("WFO canceled via API", wfo_id=str(wfo_id))

    return CancelWFOResponse(
        wfo_id=str(wfo_id),
        status="canceled",
        message="WFO canceled. Running child jobs may complete.",
    )


@router.delete(
    "/wfo/{wfo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a WFO run",
)
async def delete_wfo(wfo_id: UUID):
    """Delete a WFO run."""
    wfo_repo = _get_wfo_repo()

    deleted = await wfo_repo.delete_wfo(wfo_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"WFO run {wfo_id} not found",
        )
