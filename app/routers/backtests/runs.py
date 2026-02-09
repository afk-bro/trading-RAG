"""Backtest run endpoints."""

import json
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from app.deps.security import WorkspaceContext, get_workspace_ctx

from .schemas import (
    BacktestError,
    BacktestRunListItem,
    BacktestRunListResponse,
    BacktestRunResponse,
    BacktestSummary,
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
    from app.repositories.backtests import BacktestRepository
    from app.repositories.kb import KnowledgeBaseRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return (
        KnowledgeBaseRepository(_db_pool),
        BacktestRepository(_db_pool),
    )


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
    file: UploadFile = File(
        ...,
        description="OHLCV CSV file (columns: date, open, high, low, close, volume)",
    ),
    strategy_entity_id: UUID = Form(..., description="Strategy entity UUID"),
    workspace_id: UUID = Form(..., description="Workspace UUID"),
    params: str = Form(default="{}", description="Strategy parameters as JSON string"),
    initial_cash: float = Form(default=10000, ge=100, description="Starting capital"),
    commission_bps: float = Form(
        default=10, ge=0, le=1000, description="Commission in basis points (10 = 0.1%)"
    ),
    slippage_bps: float = Form(
        default=0, ge=0, le=1000, description="Slippage in basis points"
    ),
    date_from: Optional[str] = Form(
        default=None, description="Filter: start date (ISO format)"
    ),
    date_to: Optional[str] = Form(
        default=None, description="Filter: end date (ISO format)"
    ),
    allow_draft: bool = Form(
        default=False, description="Allow running on draft (unapproved) specs"
    ),
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
    from app.services.backtest.runner import BacktestRunError, BacktestRunner

    kb_repo, backtest_repo = _get_repos()

    # Parse params JSON
    try:
        params_dict = json.loads(params) if params else {}
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "detail": f"Invalid params JSON: {e}",
                "code": "INVALID_PARAMS_JSON",
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
async def get_backtest_run(
    run_id: UUID, ws: WorkspaceContext = Depends(get_workspace_ctx)
):
    """Get detailed results of a backtest run."""
    _, backtest_repo = _get_repos()

    run = await backtest_repo.get_run(run_id)
    if not run or str(run.get("workspace_id")) != str(ws.workspace_id):
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
    "/",
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
    _, backtest_repo = _get_repos()

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
async def delete_backtest_run(
    run_id: UUID, ws: WorkspaceContext = Depends(get_workspace_ctx)
):
    """Delete a backtest run and its results."""
    _, backtest_repo = _get_repos()

    run = await backtest_repo.get_run(run_id)
    if not run or str(run.get("workspace_id")) != str(ws.workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest run {run_id} not found",
        )

    await backtest_repo.delete_run(run_id)
