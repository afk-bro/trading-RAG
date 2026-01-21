"""WFO chart data endpoints."""

import json
from typing import Any, Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["backtests"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


# =============================================================================
# Pydantic Models
# =============================================================================


class EquityPoint(BaseModel):
    """Single point on equity curve."""

    t: str = Field(..., description="ISO timestamp with Z suffix")
    equity: float = Field(..., description="Equity value")


class FoldSummary(BaseModel):
    """Summary of a single WFO fold."""

    fold_index: int
    tune_id: str
    status: str
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    best_params: Optional[dict[str, Any]] = None
    best_score: Optional[float] = None
    metrics_oos: Optional[dict[str, Any]] = None


class CandidateMetrics(BaseModel):
    """WFO candidate aggregate metrics."""

    params_hash: str
    params: dict[str, Any]
    mean_oos: float
    median_oos: float
    worst_fold_oos: float
    stddev_oos: float
    pct_top_k: float
    fold_count: int
    total_folds: int
    coverage: float
    regime_tags: list[str] = Field(default_factory=list)


class FoldEquityData(BaseModel):
    """Equity data for a specific fold."""

    fold_index: int
    tune_id: str
    equity: list[EquityPoint]
    equity_source: Literal["jsonb", "csv", "missing"]
    summary: Optional[dict[str, Any]] = None


class WFOChartData(BaseModel):
    """Full WFO chart data response."""

    wfo_id: str
    status: str
    n_folds: int
    folds_completed: int
    folds_failed: int
    wfo_config: dict[str, Any]
    data_source: Optional[dict[str, Any]] = None
    strategy_name: Optional[str] = None
    folds: list[FoldSummary]
    best_candidate: Optional[CandidateMetrics] = None
    candidates: list[CandidateMetrics]
    # Selected fold equity (when fold_index provided)
    selected_fold: Optional[FoldEquityData] = None
    notes: list[str] = Field(default_factory=list)


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_timestamp(ts: Any) -> str:
    """Normalize timestamp to ISO format with Z suffix."""
    from datetime import datetime, timezone

    if ts is None:
        return ""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat().replace("+00:00", "Z")
    s = str(ts)
    if not s.endswith("Z") and "+" not in s:
        s = s + "Z"
    return s


def _parse_jsonb(raw: Any) -> Any:
    """Parse JSONB field that might be stored as string or dict."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def _parse_equity_curve(raw: Any) -> tuple[list[EquityPoint], Literal["jsonb", "csv", "missing"]]:
    """Parse equity curve from JSONB, return (points, source)."""
    if not raw:
        return [], "missing"

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [], "missing"

    if isinstance(raw, list):
        points = []
        for p in raw:
            if isinstance(p, dict):
                t = p.get("t") or p.get("timestamp") or p.get("time")
                equity = p.get("equity") or p.get("value") or p.get("Equity")
                if t is not None and equity is not None:
                    points.append(
                        EquityPoint(t=_normalize_timestamp(t), equity=float(equity))
                    )
        return points, "jsonb" if points else "missing"

    return [], "missing"


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/wfo/{wfo_id}/chart-data",
    response_model=WFOChartData,
    responses={
        200: {"description": "WFO chart data retrieved"},
        404: {"description": "WFO not found"},
    },
    summary="Get WFO chart data",
    description="Returns chart-ready data for WFO visualization: folds, candidates, equity curves.",
)
async def get_wfo_chart_data(
    wfo_id: UUID,
    fold_index: Optional[int] = Query(None, ge=0, description="Fold to get equity curve for"),
):
    """Get chart data for a WFO run."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    notes: list[str] = []

    # Fetch WFO record
    async with _db_pool.acquire() as conn:
        wfo = await conn.fetchrow(
            """
            SELECT
                w.id, w.status, w.n_folds, w.folds_completed, w.folds_failed,
                w.wfo_config, w.data_source, w.candidates, w.best_candidate,
                w.child_tune_ids, w.strategy_entity_id,
                s.name as strategy_name
            FROM wfo_runs w
            LEFT JOIN strategies s ON s.strategy_entity_id = w.strategy_entity_id
            WHERE w.id = $1
            """,
            wfo_id,
        )

    if not wfo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"WFO run {wfo_id} not found",
        )

    wfo_config = _parse_jsonb(wfo["wfo_config"]) or {}
    data_source = _parse_jsonb(wfo["data_source"])
    candidates_raw = _parse_jsonb(wfo["candidates"]) or []
    best_candidate_raw = _parse_jsonb(wfo["best_candidate"])
    child_tune_ids = wfo["child_tune_ids"] or []

    # Build candidates list
    candidates = []
    for c in candidates_raw:
        candidates.append(
            CandidateMetrics(
                params_hash=c.get("params_hash", ""),
                params=c.get("params", {}),
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

    # Build best candidate
    best_candidate = None
    if best_candidate_raw:
        bc = best_candidate_raw
        best_candidate = CandidateMetrics(
            params_hash=bc.get("params_hash", ""),
            params=bc.get("params", {}),
            mean_oos=bc.get("mean_oos", 0),
            median_oos=bc.get("median_oos", 0),
            worst_fold_oos=bc.get("worst_fold_oos", 0),
            stddev_oos=bc.get("stddev_oos", 0),
            pct_top_k=bc.get("pct_top_k", 0),
            fold_count=bc.get("fold_count", 0),
            total_folds=bc.get("total_folds", wfo["n_folds"] or 0),
            coverage=bc.get("coverage", 0),
            regime_tags=bc.get("regime_tags", []),
        )

    # Fetch fold details from child tunes
    folds: list[FoldSummary] = []
    if child_tune_ids:
        async with _db_pool.acquire() as conn:
            tune_rows = await conn.fetch(
                """
                SELECT
                    t.id, t.status, t.best_params, t.best_score,
                    t.data_split, t.dataset_meta
                FROM tunes t
                WHERE t.id = ANY($1)
                ORDER BY t.created_at
                """,
                child_tune_ids,
            )

        # Map tunes by ID for ordered access
        tune_map = {str(r["id"]): dict(r) for r in tune_rows}

        for idx, tune_id in enumerate(child_tune_ids):
            tune = tune_map.get(str(tune_id))
            if not tune:
                folds.append(
                    FoldSummary(
                        fold_index=idx,
                        tune_id=str(tune_id),
                        status="missing",
                    )
                )
                continue

            data_split = _parse_jsonb(tune.get("data_split")) or {}
            dataset_meta = _parse_jsonb(tune.get("dataset_meta")) or {}

            # Get best run OOS metrics for this fold
            best_run_oos = None
            if tune["status"] == "completed" and tune.get("best_score") is not None:
                async with _db_pool.acquire() as conn:
                    best_run = await conn.fetchrow(
                        """
                        SELECT metrics_oos
                        FROM tune_runs
                        WHERE tune_id = $1 AND is_best = true
                        LIMIT 1
                        """,
                        UUID(str(tune_id)),
                    )
                    if best_run:
                        best_run_oos = _parse_jsonb(best_run.get("metrics_oos"))

            folds.append(
                FoldSummary(
                    fold_index=idx,
                    tune_id=str(tune_id),
                    status=tune["status"],
                    train_start=data_split.get("train_start") or dataset_meta.get("date_min"),
                    train_end=data_split.get("train_end"),
                    test_start=data_split.get("test_start"),
                    test_end=data_split.get("test_end") or dataset_meta.get("date_max"),
                    best_params=_parse_jsonb(tune.get("best_params")),
                    best_score=tune.get("best_score"),
                    metrics_oos=best_run_oos,
                )
            )

    # Fetch selected fold equity if requested
    selected_fold = None
    # Handle fold_index - ensure it's a valid int (Query params pass through as-is in tests)
    fold_idx = fold_index if isinstance(fold_index, int) else None
    if fold_idx is not None:
        if fold_idx >= len(child_tune_ids):
            notes.append(f"Fold {fold_idx} does not exist (only {len(child_tune_ids)} folds)")
        else:
            tune_id = child_tune_ids[fold_idx]
            async with _db_pool.acquire() as conn:
                # Get best run from this fold's tune
                best_run = await conn.fetchrow(
                    """
                    SELECT tr.id, tr.metrics_oos, br.equity_curve, br.summary
                    FROM tune_runs tr
                    JOIN backtest_runs br ON br.id = tr.backtest_run_id
                    WHERE tr.tune_id = $1 AND tr.is_best = true
                    LIMIT 1
                    """,
                    tune_id,
                )

            if best_run:
                equity, source = _parse_equity_curve(best_run.get("equity_curve"))
                summary = _parse_jsonb(best_run.get("summary"))

                selected_fold = FoldEquityData(
                    fold_index=fold_idx,
                    tune_id=str(tune_id),
                    equity=equity,
                    equity_source=source,
                    summary=summary,
                )

                if not equity:
                    notes.append(f"No equity curve stored for fold {fold_idx}")
            else:
                notes.append(f"No best run found for fold {fold_idx}")

    return WFOChartData(
        wfo_id=str(wfo_id),
        status=wfo["status"],
        n_folds=wfo["n_folds"] or 0,
        folds_completed=wfo["folds_completed"] or 0,
        folds_failed=wfo["folds_failed"] or 0,
        wfo_config=wfo_config,
        data_source=data_source,
        strategy_name=wfo["strategy_name"],
        folds=folds,
        best_candidate=best_candidate,
        candidates=candidates,
        selected_fold=selected_fold,
        notes=notes,
    )
