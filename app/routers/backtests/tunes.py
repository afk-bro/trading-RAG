"""Parameter tuning endpoints."""

import json
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog
from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from .schemas import (
    BacktestError,
    BacktestSummary,
    BestRunMetrics,
    CancelTuneResponse,
    GatesSnapshot,
    GlobalLeaderboardEntry,
    GlobalLeaderboardResponse,
    LeaderboardEntry,
    StatusCounts,
    TuneListItem,
    TuneListResponse,
    TuneResponse,
    TuneRunListItem,
    TuneRunListResponse,
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


def _get_idempotency_repo():
    """Get idempotency repository instance."""
    from app.repositories.idempotency import IdempotencyRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return IdempotencyRepository(_db_pool)


@router.post(
    "/tune",
    response_model=TuneResponse,
    responses={
        200: {"description": "Tuning completed successfully"},
        404: {"description": "Strategy spec not found"},
        409: {
            "description": "Idempotency key conflict (different payload or in progress)"
        },
        422: {"description": "Invalid parameters", "model": BacktestError},
        503: {"description": "Database unavailable"},
    },
    summary="Run parameter tuning",
    description="Execute grid or random search over strategy parameters.",
)
async def create_tune(
    file: UploadFile = File(..., description="OHLCV CSV file"),
    strategy_entity_id: UUID = Form(..., description="Strategy entity UUID"),
    workspace_id: UUID = Form(..., description="Workspace UUID"),
    search_type: str = Form(
        default="random", description="Search type: grid or random"
    ),
    n_trials: int = Form(default=50, ge=1, le=200, description="Number of trials"),
    seed: Optional[int] = Form(
        default=None, description="Random seed for reproducibility"
    ),
    param_space: Optional[str] = Form(
        default=None, description="Parameter space JSON (auto-derived if omitted)"
    ),
    initial_cash: float = Form(default=10000, ge=100),
    commission_bps: float = Form(default=10, ge=0, le=1000),
    slippage_bps: float = Form(default=0, ge=0, le=1000),
    objective_metric: str = Form(
        default="sharpe", description="Objective: sharpe, return, or calmar"
    ),
    min_trades: int = Form(
        default=5, ge=1, description="Minimum trades for valid trial"
    ),
    oos_ratio: Optional[float] = Form(
        default=None,
        ge=0.01,
        le=0.5,
        description="Out-of-sample split ratio (0.01-0.5). When set, score=score_oos.",
    ),
    objective_type: str = Form(
        default="sharpe",
        description="Objective function type: sharpe, sharpe_dd_penalty, return, return_dd_penalty, calmar",  # noqa: E501
    ),
    objective_params: Optional[str] = Form(
        default=None, description='Objective params JSON (e.g., {"dd_lambda": 0.02})'
    ),
    date_from: Optional[str] = Form(default=None),
    date_to: Optional[str] = Form(default=None),
    x_idempotency_key: Optional[str] = Header(
        default=None,
        alias="X-Idempotency-Key",
        description="Idempotency key for preventing duplicate tunes (max 200 chars)",
    ),
):
    """
    Run parameter tuning session.

    **Search types:**
    - `grid`: Exhaustive search over all combinations (capped at 200)
    - `random`: Random sampling from parameter space

    **Parameter space:**
    - If omitted, auto-derived from strategy's param_schema
    - If provided, use as-is (JSON string)

    **Example param_space:**
    ```json
    {
      "period": [10, 15, 20, 25, 30],
      "threshold": {"min": 1.0, "max": 3.0, "type": "float"}
    }
    ```
    """
    from app.services.backtest.tuner import ParamTuner, derive_param_space

    kb_repo, backtest_repo, tune_repo = _get_repos()

    # Validate search type
    if search_type not in ("grid", "random"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "detail": "search_type must be 'grid' or 'random'",
                "code": "INVALID_SEARCH_TYPE",
            },
        )

    # Validate objective
    if objective_metric not in ("sharpe", "return", "calmar"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "detail": "objective_metric must be 'sharpe', 'return', or 'calmar'",
                "code": "INVALID_OBJECTIVE",
            },
        )

    # Validate objective_type
    valid_objective_types = (
        "sharpe",
        "sharpe_dd_penalty",
        "return",
        "return_dd_penalty",
        "calmar",
    )
    if objective_type not in valid_objective_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "detail": f"objective_type must be one of: {', '.join(valid_objective_types)}",
                "code": "INVALID_OBJECTIVE_TYPE",
            },
        )

    # Parse objective_params if provided
    parsed_objective_params = None
    if objective_params:
        try:
            parsed_objective_params = json.loads(objective_params)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "detail": f"Invalid objective_params JSON: {e}",
                    "code": "INVALID_OBJECTIVE_PARAMS",
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

    # Parse date filters
    parsed_date_from = None
    parsed_date_to = None
    try:
        if date_from:
            parsed_date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        if date_to:
            parsed_date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "detail": f"Invalid date format: {e}",
                "code": "INVALID_DATE_FORMAT",
            },
        )

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"detail": f"Failed to read file: {e}", "code": "FILE_READ_ERROR"},
        )

    filename = file.filename or "data.csv"

    # Get strategy spec to derive param_space if not provided
    spec = await kb_repo.get_strategy_spec(strategy_entity_id)
    if not spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy spec found for entity {strategy_entity_id}",
        )

    strategy_spec_id = spec.get("id")

    # Derive param_space if not provided
    if parsed_param_space is None:
        param_schema = spec.get("compiled_param_schema") or {}
        if isinstance(param_schema, str):
            param_schema = json.loads(param_schema)
        parsed_param_space = derive_param_space(param_schema, search_type)

    if not parsed_param_space:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "detail": "No tunable parameters found in param_schema",
                "code": "NO_TUNABLE_PARAMS",
            },
        )

    # ===========================================
    # Idempotency handling
    # ===========================================
    idempotency_record = None
    idempotency_repo = None

    if x_idempotency_key:
        import hashlib

        from app.services.idempotency import (
            IdempotencyKeyReusedError,
            IdempotencyKeyTooLongError,
            compute_request_hash,
            validate_idempotency_key,
            verify_hash_match,
        )

        idempotency_repo = _get_idempotency_repo()

        # Validate key length
        try:
            validate_idempotency_key(x_idempotency_key)
        except IdempotencyKeyTooLongError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "detail": str(e),
                    "code": "IDEMPOTENCY_KEY_TOO_LONG",
                },
            )

        # Compute content hash for file (separate from request hash for efficiency)
        file_content_hash = hashlib.sha256(file_content).hexdigest()

        # Build canonical payload for hash computation
        # Include all parameters that define the tune outcome
        canonical_payload = {
            "strategy_entity_id": str(strategy_entity_id),
            "search_type": search_type,
            "n_trials": n_trials,
            "seed": seed,
            "param_space": parsed_param_space,
            "initial_cash": initial_cash,
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "objective_metric": objective_metric,
            "objective_type": objective_type,
            "objective_params": parsed_objective_params,
            "min_trades": min_trades,
            "oos_ratio": oos_ratio,
            "date_from": date_from,
            "date_to": date_to,
            "file_content_hash": file_content_hash,
        }

        # Compute request hash with float normalization for known fields
        request_hash = compute_request_hash(
            canonical_payload,
            float_fields=[
                "initial_cash",
                "commission_bps",
                "slippage_bps",
                "oos_ratio",
            ],
        )

        # Try to claim the idempotency key
        is_new, existing_record = await idempotency_repo.claim_or_get(
            workspace_id=workspace_id,
            idempotency_key=x_idempotency_key,
            request_hash=request_hash,
            endpoint="backtests.tune",
            http_method="POST",
        )

        if not is_new:
            # Key already exists - handle based on status
            if existing_record.status == "pending":
                # Wait for original request to complete
                completed = await idempotency_repo.wait_for_completion_or_timeout(
                    workspace_id=workspace_id,
                    endpoint="backtests.tune",
                    idempotency_key=x_idempotency_key,
                    max_wait_seconds=5.0,
                )
                if completed:
                    existing_record = completed
                else:
                    # Still pending after timeout
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "detail": "Idempotency key in progress, retry after 5s",
                            "code": "IDEMPOTENCY_IN_PROGRESS",
                            "retry_after_seconds": 5,
                        },
                    )

            # Verify hash matches (detect key reuse with different payload)
            try:
                verify_hash_match(existing_record, request_hash)
            except IdempotencyKeyReusedError:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "detail": "Idempotency key reused with different payload",
                        "code": "IDEMPOTENCY_KEY_REUSED",
                    },
                )

            # Replay response based on status
            if existing_record.status == "completed":
                logger.info(
                    "idempotency_replay",
                    idempotency_key=x_idempotency_key,
                    tune_id=str(existing_record.resource_id),
                )
                # Return cached response with 200 status
                return JSONResponse(
                    content=existing_record.response_json,
                    status_code=200,
                )
            elif existing_record.status == "failed":
                logger.info(
                    "idempotency_replay_failed",
                    idempotency_key=x_idempotency_key,
                    error_code=existing_record.error_code,
                )
                # Replay the original error
                error_json = existing_record.error_json or {}
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "detail": error_json.get("detail", "Tuning failed"),
                        "code": existing_record.error_code or "TUNE_FAILED",
                    },
                )

        # New claim - store the record for completion tracking
        idempotency_record = existing_record

    logger.info(
        "Creating parameter tune",
        strategy_entity_id=str(strategy_entity_id),
        workspace_id=str(workspace_id),
        search_type=search_type,
        n_trials=n_trials,
        seed=seed,
        param_space=parsed_param_space,
        idempotency_key=x_idempotency_key,
    )

    # Build gates snapshot (audit trail for gate policy at tune creation)
    from app.services.backtest.tuner import GATE_MAX_DD_PCT, GATE_MIN_TRADES

    gates_snapshot = {
        "max_drawdown_pct": GATE_MAX_DD_PCT,
        "min_trades": GATE_MIN_TRADES,
        "evaluated_on": "oos" if oos_ratio else "primary",
    }

    # Create tune record
    tune_id = await tune_repo.create_tune(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        strategy_spec_id=strategy_spec_id,
        search_type=search_type,
        n_trials=n_trials,
        seed=seed,
        param_space=parsed_param_space,
        objective_metric=objective_metric,
        min_trades=min_trades,
        oos_ratio=oos_ratio,
        objective_type=objective_type,
        objective_params=parsed_objective_params,
        gates=gates_snapshot,
    )

    # Run tuning
    tuner = ParamTuner(kb_repo, backtest_repo, tune_repo)

    try:
        result = await tuner.run(
            tune_id=tune_id,
            strategy_entity_id=strategy_entity_id,
            workspace_id=workspace_id,
            file_content=file_content,
            filename=filename,
            param_space=parsed_param_space,
            search_type=search_type,
            n_trials=n_trials,
            seed=seed,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            objective_metric=objective_metric,
            min_trades=min_trades,
            oos_ratio=oos_ratio,
            objective_type=objective_type,
            objective_params=parsed_objective_params,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
        )
    except Exception as e:
        logger.error("Tuning failed", tune_id=str(tune_id), error=str(e))

        # Mark idempotency record as failed if present
        if idempotency_record and idempotency_repo:
            await idempotency_repo.mark_failed(
                idempotency_id=idempotency_record.id,
                error_code="TUNE_FAILED",
                error_json={"detail": f"Tuning failed: {e}"},
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"detail": f"Tuning failed: {e}", "code": "TUNE_FAILED"},
        )

    # Build response
    leaderboard = []
    for entry in result.leaderboard:
        summary = entry.get("summary")
        leaderboard.append(
            LeaderboardEntry(
                rank=entry["rank"],
                run_id=entry["run_id"],
                params=entry["params"],
                score=entry["score"],
                summary=BacktestSummary(**summary) if summary else None,
            )
        )

    response = TuneResponse(
        tune_id=str(result.tune_id),
        status=result.status,
        search_type=search_type,
        n_trials=result.n_trials,
        trials_completed=result.trials_completed,
        best_run_id=str(result.best_run_id) if result.best_run_id else None,
        best_params=result.best_params,
        best_score=result.best_score,
        leaderboard=leaderboard,
        warnings=result.warnings,
    )

    # Mark idempotency record as completed with response for replay
    if idempotency_record and idempotency_repo:
        await idempotency_repo.complete(
            idempotency_id=idempotency_record.id,
            resource_id=result.tune_id,
            response_json=response.model_dump(mode="json"),
        )

    return response


@router.get(
    "/tunes/{tune_id}",
    response_model=TuneResponse,
    summary="Get tune details",
)
async def get_tune(tune_id: UUID):
    """Get details of a parameter tuning session including status counts."""
    _, _, tune_repo = _get_repos()

    tune = await tune_repo.get_tune(tune_id)
    if not tune:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tune {tune_id} not found",
        )

    # Get status counts
    counts = await tune_repo.get_tune_status_counts(tune_id)

    # Build leaderboard
    leaderboard = []
    for entry in tune.get("leaderboard") or []:
        summary = entry.get("summary")
        leaderboard.append(
            LeaderboardEntry(
                rank=entry.get("rank", 0),
                run_id=entry.get("run_id", ""),
                params=entry.get("params", {}),
                score=entry.get("score", 0),
                summary=BacktestSummary(**summary) if summary else None,
            )
        )

    # Parse gates snapshot if present
    gates = None
    if tune.get("gates"):
        gates = GatesSnapshot(**tune["gates"])

    return TuneResponse(
        tune_id=str(tune["id"]),
        status=tune["status"],
        search_type=tune["search_type"],
        n_trials=tune["n_trials"],
        trials_completed=tune.get("trials_completed", 0),
        best_run_id=str(tune["best_run_id"]) if tune.get("best_run_id") else None,
        best_params=leaderboard[0].params if leaderboard else None,
        best_score=leaderboard[0].score if leaderboard else None,
        leaderboard=leaderboard,
        counts=StatusCounts(**counts),
        gates=gates,
    )


@router.post(
    "/tunes/{tune_id}/cancel",
    response_model=CancelTuneResponse,
    summary="Cancel a tuning session",
)
async def cancel_tune(tune_id: UUID):
    """
    Cancel a running or queued tuning session.

    - Sets tune status to 'canceled'
    - Marks all queued trials as skipped with skip_reason='canceled'
    - Running trials will complete but subsequent trials are skipped
    """
    _, _, tune_repo = _get_repos()

    # Check tune exists and is in cancelable state
    tune = await tune_repo.get_tune(tune_id)
    if not tune:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tune {tune_id} not found",
        )

    if tune["status"] not in ("queued", "running"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel tune in '{tune['status']}' state. Only queued or running tunes can be canceled.",  # noqa: E501
        )

    # Attempt cancellation
    canceled = await tune_repo.cancel_tune(tune_id)

    if not canceled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tune could not be canceled (may have completed)",
        )

    logger.info("Tune canceled via API", tune_id=str(tune_id))

    return CancelTuneResponse(
        tune_id=str(tune_id),
        status="canceled",
        message="Tune canceled. Running trials may complete, remaining trials skipped.",
    )


@router.get(
    "/tunes/{tune_id}/runs",
    response_model=TuneRunListResponse,
    summary="List tune trial runs",
)
async def list_tune_runs(
    tune_id: UUID,
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List trial runs for a tuning session."""
    _, _, tune_repo = _get_repos()

    runs, total = await tune_repo.list_tune_runs(
        tune_id=tune_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    items = []
    for run in runs:
        score_is = run.get("score_is")
        score_oos = run.get("score_oos")

        # Compute overfit_gap when both scores available
        overfit_gap = None
        if score_is is not None and score_oos is not None:
            overfit_gap = round(score_is - score_oos, 4)

        items.append(
            TuneRunListItem(
                trial_index=run["trial_index"],
                run_id=str(run["run_id"]) if run.get("run_id") else None,
                params=run.get("params", {}),
                score=run.get("score"),
                score_is=score_is,
                score_oos=score_oos,
                objective_score=run.get("objective_score"),
                overfit_gap=overfit_gap,
                metrics_is=run.get("metrics_is"),
                metrics_oos=run.get("metrics_oos"),
                status=run["status"],
                skip_reason=run.get("skip_reason"),
                failed_reason=run.get("failed_reason"),
            )
        )

    return TuneRunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/tunes",
    response_model=TuneListResponse,
    summary="List tuning sessions",
)
async def list_tunes(
    workspace_id: UUID = Query(..., description="Workspace UUID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    valid_only: bool = Query(
        False, description="Only show tunes with valid results (best_run_id not null)"
    ),
    objective_type: Optional[str] = Query(
        None, description="Filter by objective type (sharpe, sharpe_dd_penalty, etc.)"
    ),
    oos_enabled: Optional[bool] = Query(
        None,
        description="Filter by OOS split: true=with OOS, false=without OOS, null=all",
    ),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List parameter tuning sessions."""
    _, _, tune_repo = _get_repos()

    tunes, total = await tune_repo.list_tunes(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        status=status_filter,
        valid_only=valid_only,
        objective_type=objective_type,
        oos_enabled=oos_enabled,
        limit=limit,
        offset=offset,
    )

    items = []
    for tune in tunes:
        tune_id = tune["id"]

        # Get counts for this tune
        counts = await tune_repo.get_tune_status_counts(tune_id)

        # Get best_* from persisted fields (not derived from leaderboard)
        best_score = tune.get("best_score")
        best_run_id = tune.get("best_run_id")
        best_params = tune.get("best_params")

        # Parse best_params if stored as JSON string
        if isinstance(best_params, str):
            try:
                best_params = json.loads(best_params)
            except json.JSONDecodeError:
                best_params = None

        # Parse gates snapshot if present
        gates = None
        if tune.get("gates"):
            gates = GatesSnapshot(**tune["gates"])

        items.append(
            TuneListItem(
                id=str(tune_id),
                created_at=tune["created_at"],
                strategy_entity_id=str(tune["strategy_entity_id"]),
                strategy_name=tune.get("strategy_name"),
                search_type=tune["search_type"],
                n_trials=tune["n_trials"],
                status=tune["status"],
                trials_completed=tune.get("trials_completed", 0),
                best_score=best_score,
                best_run_id=str(best_run_id) if best_run_id else None,
                best_params=best_params,
                objective_metric=tune["objective_metric"],
                objective_type=tune.get("objective_type"),
                oos_ratio=tune.get("oos_ratio"),
                gates=gates,
                counts=StatusCounts(**counts),
            )
        )

    return TuneListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/leaderboard",
    response_model=GlobalLeaderboardResponse,
    summary="Get global leaderboard",
)
async def get_leaderboard(
    workspace_id: UUID = Query(..., description="Workspace UUID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    valid_only: bool = Query(
        True,
        description="Only tunes with valid winning trial (default True for leaderboard)",
    ),
    objective_type: Optional[str] = Query(
        None, description="Filter by objective type: sharpe, sharpe_dd_penalty, etc."
    ),
    oos_enabled: Optional[str] = Query(
        None, description="Filter by OOS enabled: 'true' or 'false'"
    ),
    include_canceled: bool = Query(False, description="Include canceled tunes"),
    limit: int = Query(default=50, ge=1, le=100, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
):
    """
    Get global leaderboard: best tunes ranked by objective score.

    Returns tunes joined with their winning trial's metrics, ordered by
    objective_score (or fallback chain: score_oos, best_score).

    By default filters to valid_only=True (tunes with a winning trial).
    Includes overfit_gap (score_is - score_oos) for robustness assessment.
    """
    _, _, tune_repo = _get_repos()

    # Parse oos_enabled filter
    oos_enabled_bool = None
    if oos_enabled is not None:
        oos_enabled_bool = oos_enabled.lower() == "true"

    entries, total = await tune_repo.get_leaderboard(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        valid_only=valid_only,
        objective_type=objective_type,
        oos_enabled=oos_enabled_bool,
        include_canceled=include_canceled,
        limit=limit,
        offset=offset,
    )

    items = []
    for entry in entries:
        # Parse gates snapshot
        gates = None
        if entry.get("gates"):
            gates = GatesSnapshot(**entry["gates"])

        # Parse best_metrics_oos
        best_metrics_oos = None
        if entry.get("best_metrics_oos"):
            metrics = entry["best_metrics_oos"]
            best_metrics_oos = BestRunMetrics(
                return_pct=metrics.get("return_pct"),
                sharpe=metrics.get("sharpe"),
                max_drawdown_pct=metrics.get("max_drawdown_pct"),
                trades=metrics.get("trades"),
            )

        items.append(
            GlobalLeaderboardEntry(
                tune_id=str(entry["id"]),
                created_at=entry["created_at"],
                strategy_entity_id=str(entry["strategy_entity_id"]),
                strategy_name=entry.get("strategy_name"),
                objective_type=entry.get("objective_type"),
                objective_params=entry.get("objective_params"),
                oos_ratio=entry.get("oos_ratio"),
                gates=gates,
                status=entry["status"],
                best_run_id=(
                    str(entry["best_run_id"]) if entry.get("best_run_id") else None
                ),
                best_score=entry.get("best_score"),
                best_objective_score=entry.get("best_objective_score"),
                score_is=entry.get("score_is"),
                score_oos=entry.get("score_oos"),
                overfit_gap=entry.get("overfit_gap"),
                best_metrics_oos=best_metrics_oos,
            )
        )

    return GlobalLeaderboardResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/tunes/{tune_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a tuning session",
)
async def delete_tune(tune_id: UUID):
    """Delete a tuning session and all its trial runs."""
    _, _, tune_repo = _get_repos()

    deleted = await tune_repo.delete_tune(tune_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tune {tune_id} not found",
        )
