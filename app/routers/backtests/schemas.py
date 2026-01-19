"""Backtest API schemas (request/response models)."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ===========================================
# Backtest Run Models
# ===========================================


class BacktestSummary(BaseModel):
    """Summary metrics from a backtest run."""

    return_pct: float = Field(..., description="Total return percentage")
    max_drawdown_pct: float = Field(..., description="Maximum drawdown percentage")
    sharpe: Optional[float] = Field(None, description="Sharpe ratio")
    win_rate: float = Field(..., description="Win rate (0-1)")
    trades: int = Field(..., description="Number of trades")
    buy_hold_return_pct: Optional[float] = Field(
        None, description="Buy & hold return for comparison"
    )
    avg_trade_pct: Optional[float] = Field(
        None, description="Average trade return percentage"
    )
    profit_factor: Optional[float] = Field(
        None, description="Profit factor (gross profit / gross loss)"
    )


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
    warnings: list[str] = Field(
        default_factory=list, description="Warnings generated during run"
    )


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

    max_drawdown_pct: float = Field(
        ..., description="Max allowed drawdown percent (gate threshold)"
    )
    min_trades: int = Field(..., description="Min required trades (gate threshold)")
    evaluated_on: str = Field(
        ..., description="Which metrics gates were evaluated on: 'oos' or 'primary'"
    )


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
    gates: Optional[GatesSnapshot] = Field(
        None, description="Gate policy snapshot at tune creation"
    )
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
    best_objective_score: Optional[float] = Field(
        None, description="Objective score from winning trial"
    )
    score_is: Optional[float] = Field(
        None, description="In-sample score from winning trial"
    )
    score_oos: Optional[float] = Field(
        None, description="Out-of-sample score from winning trial"
    )
    overfit_gap: Optional[float] = Field(
        None, description="Computed: score_is - score_oos"
    )
    best_metrics_oos: Optional[BestRunMetrics] = Field(
        None, description="OOS metrics from winning trial"
    )


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
    objective_score: Optional[float] = (
        None  # Composite objective score (when using dd_penalty, etc.)
    )
    overfit_gap: Optional[float] = (
        None  # Computed: score_is - score_oos (when split enabled)
    )
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


class CancelTuneResponse(BaseModel):
    """Response from canceling a tune."""

    tune_id: str
    status: str
    message: str


# ===========================================
# Walk-Forward Optimization (WFO) Models
# ===========================================


class WFOConfigModel(BaseModel):
    """WFO configuration."""

    train_days: int = Field(..., ge=1, description="Days in training window")
    test_days: int = Field(..., ge=1, description="Days in test window")
    step_days: int = Field(..., ge=1, description="Days to step forward between folds")
    min_folds: int = Field(
        default=3, ge=1, description="Minimum number of folds required"
    )
    leaderboard_top_k: int = Field(
        default=10, ge=1, description="Top params per fold to track"
    )
    allow_partial: bool = Field(
        default=False, description="Continue even if some folds fail"
    )


class WFOCandidateModel(BaseModel):
    """WFO candidate metrics."""

    params: dict[str, Any]
    params_hash: str
    mean_oos: float
    median_oos: float
    worst_fold_oos: float
    stddev_oos: float
    pct_top_k: float
    fold_count: int
    total_folds: int
    coverage: float
    regime_tags: list[str] = Field(default_factory=list)


class WFOResponse(BaseModel):
    """Response for WFO run."""

    wfo_id: str
    status: str
    n_folds: int
    folds_completed: int
    folds_failed: int
    best_params: Optional[dict[str, Any]] = None
    best_candidate: Optional[WFOCandidateModel] = None
    candidates: list[WFOCandidateModel] = Field(default_factory=list)
    child_tune_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WFOListItem(BaseModel):
    """Summary item for listing WFO runs."""

    id: str
    created_at: datetime
    strategy_entity_id: str
    strategy_name: Optional[str]
    status: str
    n_folds: int
    folds_completed: int
    folds_failed: int
    wfo_config: WFOConfigModel
    best_params: Optional[dict[str, Any]] = None


class WFOListResponse(BaseModel):
    """Response for listing WFO runs."""

    items: list[WFOListItem]
    total: int
    limit: int
    offset: int


class CancelWFOResponse(BaseModel):
    """Response from canceling a WFO run."""

    wfo_id: str
    status: str
    message: str
