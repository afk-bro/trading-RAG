"""Backtest API endpoints."""

import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.config import Settings, get_settings

router = APIRouter(prefix="/backtests", tags=["backtests"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_repos():
    """Get repository instances."""
    from app.repositories.kb import KnowledgeBaseRepository
    from app.repositories.backtests import BacktestRepository, TuneRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return KnowledgeBaseRepository(_db_pool), BacktestRepository(_db_pool), TuneRepository(_db_pool)


# ===========================================
# Request/Response Models
# ===========================================


class BacktestSummary(BaseModel):
    """Summary metrics from a backtest run."""

    return_pct: float = Field(..., description="Total return percentage")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    sharpe: Optional[float] = Field(None, description="Sharpe ratio")
    win_rate: float = Field(..., description="Win rate (0-1)")
    trades: int = Field(..., description="Number of trades")
    buy_hold_return_pct: Optional[float] = Field(None, description="Buy & hold return for comparison")
    avg_trade_pct: Optional[float] = Field(None, description="Average trade return percentage")
    profit_factor: Optional[float] = Field(None, description="Profit factor (gross profit / gross loss)")


class EquityPoint(BaseModel):
    """Single point on equity curve."""

    t: str = Field(..., description="ISO timestamp")
    equity: float = Field(..., description="Equity value")


class TradeRecord(BaseModel):
    """Single trade record."""

    entry_time: str = Field(..., description="Entry timestamp")
    exit_time: str = Field(..., description="Exit timestamp")
    side: str = Field(..., description="Trade direction: long or short")
    entry_price: float = Field(..., description="Entry price")
    exit_price: float = Field(..., description="Exit price")
    size: float = Field(..., description="Position size")
    pnl: float = Field(..., description="Profit/loss in currency")
    return_pct: float = Field(..., description="Return percentage")
    duration_bars: int = Field(..., description="Duration in bars")


class BacktestRunResponse(BaseModel):
    """Response from running a backtest."""

    run_id: str = Field(..., description="Unique run identifier")
    status: str = Field(..., description="Run status: completed or failed")
    summary: BacktestSummary = Field(..., description="Summary metrics")
    equity_curve: list[dict[str, Any]] = Field(..., description="Equity curve points")
    trades: list[dict[str, Any]] = Field(..., description="Trade records")
    warnings: list[str] = Field(default_factory=list, description="Warnings generated during run")


class BacktestRunListItem(BaseModel):
    """Summary item for listing backtest runs."""

    id: str
    created_at: datetime
    strategy_entity_id: str
    strategy_name: Optional[str]
    status: str
    summary: Optional[BacktestSummary]
    dataset_meta: dict[str, Any]


class BacktestRunListResponse(BaseModel):
    """Response for listing backtest runs."""

    items: list[BacktestRunListItem]
    total: int
    limit: int
    offset: int


class BacktestError(BaseModel):
    """Error response for backtest failures."""

    detail: str
    code: str
    errors: Optional[list[dict[str, Any]]] = None


# ===========================================
# Tune Response Models
# ===========================================


class LeaderboardEntry(BaseModel):
    """Entry in tune leaderboard."""

    rank: int
    run_id: str
    params: dict[str, Any]
    score: float
    summary: Optional[BacktestSummary] = None


class StatusCounts(BaseModel):
    """Status counts for tune runs."""

    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0


class GatesSnapshot(BaseModel):
    """Gate policy snapshot persisted with tune."""

    max_drawdown_pct: float = Field(..., description="Max allowed drawdown percent (gate threshold)")
    min_trades: int = Field(..., description="Min required trades (gate threshold)")
    evaluated_on: str = Field(..., description="Which metrics gates were evaluated on: 'oos' or 'primary'")


class TuneResponse(BaseModel):
    """Response from running parameter tuning."""

    tune_id: str
    status: str
    search_type: str
    n_trials: int
    trials_completed: int
    best_run_id: Optional[str] = None
    best_params: Optional[dict[str, Any]] = None
    best_score: Optional[float] = None
    leaderboard: list[LeaderboardEntry]
    counts: Optional[StatusCounts] = None
    gates: Optional[GatesSnapshot] = Field(None, description="Gate policy snapshot at tune creation")
    warnings: list[str] = Field(default_factory=list)


class TuneListItem(BaseModel):
    """Summary item for listing tunes."""

    id: str
    created_at: datetime
    strategy_entity_id: str
    strategy_name: Optional[str]
    search_type: str
    n_trials: int
    status: str
    trials_completed: int
    best_score: Optional[float] = None
    best_run_id: Optional[str] = None
    best_params: Optional[dict[str, Any]] = None
    objective_metric: str
    objective_type: Optional[str] = None
    oos_ratio: Optional[float] = None
    gates: Optional[GatesSnapshot] = None
    counts: Optional[StatusCounts] = None


class TuneListResponse(BaseModel):
    """Response for listing tunes."""

    items: list[TuneListItem]
    total: int
    limit: int
    offset: int


class BestRunMetrics(BaseModel):
    """Metrics from the best run's OOS period."""

    return_pct: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    trades: Optional[int] = None


class GlobalLeaderboardEntry(BaseModel):
    """Entry in the global leaderboard showing best tunes across strategies."""

    tune_id: str
    created_at: datetime
    strategy_entity_id: str
    strategy_name: Optional[str] = None
    objective_type: Optional[str] = None
    objective_params: Optional[dict[str, Any]] = None
    oos_ratio: Optional[float] = None
    gates: Optional[GatesSnapshot] = None
    status: str
    best_run_id: Optional[str] = None
    best_score: Optional[float] = None
    best_objective_score: Optional[float] = Field(None, description="Objective score from winning trial")
    score_is: Optional[float] = Field(None, description="In-sample score from winning trial")
    score_oos: Optional[float] = Field(None, description="Out-of-sample score from winning trial")
    overfit_gap: Optional[float] = Field(None, description="Computed: score_is - score_oos")
    best_metrics_oos: Optional[BestRunMetrics] = Field(None, description="OOS metrics from winning trial")


class GlobalLeaderboardResponse(BaseModel):
    """Response for global leaderboard endpoint."""

    items: list[GlobalLeaderboardEntry]
    total: int
    limit: int
    offset: int


class TuneRunListItem(BaseModel):
    """Summary item for listing tune runs."""

    trial_index: int
    run_id: Optional[str]
    params: dict[str, Any]
    score: Optional[float]
    score_is: Optional[float] = None
    score_oos: Optional[float] = None
    objective_score: Optional[float] = None  # Composite objective score (when using dd_penalty, etc.)
    overfit_gap: Optional[float] = None  # Computed: score_is - score_oos (when split enabled)
    metrics_is: Optional[dict[str, Any]] = None
    metrics_oos: Optional[dict[str, Any]] = None
    status: str
    skip_reason: Optional[str] = None
    failed_reason: Optional[str] = None


class TuneRunListResponse(BaseModel):
    """Response for listing tune runs."""

    items: list[TuneRunListItem]
    total: int
    limit: int
    offset: int


# ===========================================
# Endpoints
# ===========================================


@router.post(
    "/run",
    response_model=BacktestRunResponse,
    responses={
        200: {"description": "Backtest completed successfully"},
        404: {"description": "Strategy spec not found"},
        409: {"description": "Strategy spec not approved"},
        422: {"description": "Invalid data or parameters", "model": BacktestError},
        503: {"description": "Database unavailable"},
    },
    summary="Run a backtest",
    description="Execute a backtest using a strategy spec and uploaded OHLCV CSV data.",
)
async def run_backtest(
    file: UploadFile = File(..., description="OHLCV CSV file (columns: date, open, high, low, close, volume)"),
    strategy_entity_id: UUID = Form(..., description="Strategy entity UUID"),
    workspace_id: UUID = Form(..., description="Workspace UUID"),
    params: str = Form(default="{}", description="Strategy parameters as JSON string"),
    initial_cash: float = Form(default=10000, ge=100, description="Starting capital"),
    commission_bps: float = Form(default=10, ge=0, le=1000, description="Commission in basis points (10 = 0.1%)"),
    slippage_bps: float = Form(default=0, ge=0, le=1000, description="Slippage in basis points"),
    date_from: Optional[str] = Form(default=None, description="Filter: start date (ISO format)"),
    date_to: Optional[str] = Form(default=None, description="Filter: end date (ISO format)"),
    allow_draft: bool = Form(default=False, description="Allow running on draft (unapproved) specs"),
):
    """
    Run a backtest with uploaded OHLCV data.

    **CSV Requirements:**
    - Required columns: date, open, high, low, close, volume
    - Common aliases supported: timestamp→date, adj_close→close
    - Data will be sorted by date and deduplicated
    - Maximum file size: 25MB
    - Maximum rows: 2,000,000

    **Parameters:**
    - Pass strategy parameters as JSON in the `params` field
    - Parameters are validated against the strategy's param_schema
    - Missing parameters use defaults from the schema

    **Example curl:**
    ```bash
    curl -X POST http://localhost:8000/backtests/run \\
      -F "file=@AAPL_1h.csv" \\
      -F "strategy_entity_id=<uuid>" \\
      -F "workspace_id=<uuid>" \\
      -F "params={\"period\": 20, \"threshold\": 2.0}" \\
      -F "initial_cash=10000"
    ```
    """
    from app.services.backtest.runner import BacktestRunner, BacktestRunError

    kb_repo, backtest_repo, _ = _get_repos()

    # Parse params JSON
    try:
        params_dict = json.loads(params) if params else {}
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"detail": f"Invalid params JSON: {e}", "code": "INVALID_PARAMS_JSON"},
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
            detail={"detail": f"Invalid date format: {e}", "code": "INVALID_DATE_FORMAT"},
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

    logger.info(
        "Backtest run requested",
        strategy_entity_id=str(strategy_entity_id),
        workspace_id=str(workspace_id),
        filename=filename,
        file_size_kb=len(file_content) / 1024,
        params=params_dict,
    )

    # Run backtest
    runner = BacktestRunner(kb_repo, backtest_repo)

    try:
        result = await runner.run(
            strategy_entity_id=strategy_entity_id,
            file_content=file_content,
            filename=filename,
            params=params_dict,
            workspace_id=workspace_id,
            initial_cash=initial_cash,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            date_from=parsed_date_from,
            date_to=parsed_date_to,
            allow_draft=allow_draft,
        )
    except BacktestRunError as e:
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        if e.code == "SPEC_NOT_FOUND":
            status_code = status.HTTP_404_NOT_FOUND
        elif e.code == "SPEC_NOT_APPROVED":
            status_code = status.HTTP_409_CONFLICT

        raise HTTPException(
            status_code=status_code,
            detail={
                "detail": e.message,
                "code": e.code,
                "errors": e.details.get("errors") if e.details else None,
            },
        )

    return BacktestRunResponse(
        run_id=result["run_id"],
        status=result["status"],
        summary=BacktestSummary(**result["summary"]),
        equity_curve=result["equity_curve"],
        trades=result["trades"],
        warnings=result["warnings"],
    )


@router.get(
    "/{run_id}",
    response_model=BacktestRunResponse,
    responses={
        200: {"description": "Backtest run retrieved"},
        404: {"description": "Run not found"},
    },
    summary="Get backtest run details",
)
async def get_backtest_run(run_id: UUID):
    """Get detailed results of a backtest run."""
    _, backtest_repo, _ = _get_repos()

    run = await backtest_repo.get_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    summary = run.get("summary") or {}

    return BacktestRunResponse(
        run_id=str(run["id"]),
        status=run["status"],
        summary=BacktestSummary(
            return_pct=summary.get("return_pct", 0),
            max_drawdown_pct=summary.get("max_drawdown_pct", 0),
            sharpe=summary.get("sharpe"),
            win_rate=summary.get("win_rate", 0),
            trades=summary.get("trades", 0),
            buy_hold_return_pct=summary.get("buy_hold_return_pct"),
            avg_trade_pct=summary.get("avg_trade_pct"),
            profit_factor=summary.get("profit_factor"),
        ),
        equity_curve=run.get("equity_curve") or [],
        trades=run.get("trades") or [],
        warnings=run.get("warnings") or [],
    )


@router.get(
    "",
    response_model=BacktestRunListResponse,
    summary="List backtest runs",
)
async def list_backtest_runs(
    workspace_id: UUID = Query(..., description="Workspace UUID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List backtest runs with optional filtering."""
    _, backtest_repo, _ = _get_repos()

    runs, total = await backtest_repo.list_runs(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    items = []
    for run in runs:
        summary = run.get("summary")
        items.append(
            BacktestRunListItem(
                id=str(run["id"]),
                created_at=run["created_at"],
                strategy_entity_id=str(run["strategy_entity_id"]),
                strategy_name=run.get("strategy_name"),
                status=run["status"],
                summary=BacktestSummary(**summary) if summary else None,
                dataset_meta=run.get("dataset_meta") or {},
            )
        )

    return BacktestRunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a backtest run",
)
async def delete_backtest_run(run_id: UUID):
    """Delete a backtest run and its results."""
    _, backtest_repo, _ = _get_repos()

    deleted = await backtest_repo.delete_run(run_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )


# ===========================================
# Parameter Tuning Endpoints
# ===========================================


@router.post(
    "/tune",
    response_model=TuneResponse,
    responses={
        200: {"description": "Tuning completed successfully"},
        404: {"description": "Strategy spec not found"},
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
    search_type: str = Form(default="random", description="Search type: grid or random"),
    n_trials: int = Form(default=50, ge=1, le=200, description="Number of trials"),
    seed: Optional[int] = Form(default=None, description="Random seed for reproducibility"),
    param_space: Optional[str] = Form(default=None, description="Parameter space JSON (auto-derived if omitted)"),
    initial_cash: float = Form(default=10000, ge=100),
    commission_bps: float = Form(default=10, ge=0, le=1000),
    slippage_bps: float = Form(default=0, ge=0, le=1000),
    objective_metric: str = Form(default="sharpe", description="Objective: sharpe, return, or calmar"),
    min_trades: int = Form(default=5, ge=1, description="Minimum trades for valid trial"),
    oos_ratio: Optional[float] = Form(default=None, ge=0.01, le=0.5, description="Out-of-sample split ratio (0.01-0.5). When set, score=score_oos."),
    objective_type: str = Form(default="sharpe", description="Objective function type: sharpe, sharpe_dd_penalty, return, return_dd_penalty, calmar"),
    objective_params: Optional[str] = Form(default=None, description="Objective params JSON (e.g., {\"dd_lambda\": 0.02})"),
    date_from: Optional[str] = Form(default=None),
    date_to: Optional[str] = Form(default=None),
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
            detail={"detail": "search_type must be 'grid' or 'random'", "code": "INVALID_SEARCH_TYPE"},
        )

    # Validate objective
    if objective_metric not in ("sharpe", "return", "calmar"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"detail": "objective_metric must be 'sharpe', 'return', or 'calmar'", "code": "INVALID_OBJECTIVE"},
        )

    # Validate objective_type
    valid_objective_types = ("sharpe", "sharpe_dd_penalty", "return", "return_dd_penalty", "calmar")
    if objective_type not in valid_objective_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"detail": f"objective_type must be one of: {', '.join(valid_objective_types)}", "code": "INVALID_OBJECTIVE_TYPE"},
        )

    # Parse objective_params if provided
    parsed_objective_params = None
    if objective_params:
        try:
            parsed_objective_params = json.loads(objective_params)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"detail": f"Invalid objective_params JSON: {e}", "code": "INVALID_OBJECTIVE_PARAMS"},
            )

    # Parse param_space if provided
    parsed_param_space = None
    if param_space:
        try:
            parsed_param_space = json.loads(param_space)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"detail": f"Invalid param_space JSON: {e}", "code": "INVALID_PARAM_SPACE"},
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
            detail={"detail": f"Invalid date format: {e}", "code": "INVALID_DATE_FORMAT"},
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
            detail={"detail": "No tunable parameters found in param_schema", "code": "NO_TUNABLE_PARAMS"},
        )

    logger.info(
        "Creating parameter tune",
        strategy_entity_id=str(strategy_entity_id),
        workspace_id=str(workspace_id),
        search_type=search_type,
        n_trials=n_trials,
        seed=seed,
        param_space=parsed_param_space,
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

    return TuneResponse(
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
    for entry in (tune.get("leaderboard") or []):
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


class CancelTuneResponse(BaseModel):
    """Response from canceling a tune."""

    tune_id: str
    status: str
    message: str


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
            detail=f"Cannot cancel tune in '{tune['status']}' state. Only queued or running tunes can be canceled.",
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
    valid_only: bool = Query(False, description="Only show tunes with valid results (best_run_id not null)"),
    objective_type: Optional[str] = Query(None, description="Filter by objective type (sharpe, sharpe_dd_penalty, etc.)"),
    oos_enabled: Optional[bool] = Query(None, description="Filter by OOS split: true=with OOS, false=without OOS, null=all"),
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
    valid_only: bool = Query(True, description="Only tunes with valid winning trial (default True for leaderboard)"),
    objective_type: Optional[str] = Query(None, description="Filter by objective type: sharpe, sharpe_dd_penalty, etc."),
    oos_enabled: Optional[str] = Query(None, description="Filter by OOS enabled: 'true' or 'false'"),
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
                best_run_id=str(entry["best_run_id"]) if entry.get("best_run_id") else None,
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
