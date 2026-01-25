"""Admin analytics endpoints for KB attribution dashboard.

Provides analytics endpoints for:
1. Regime Coverage: "Do we have enough data per regime?"
2. Tier Usage: "How often are we falling back?"
3. Value Add (Uplift): "Does regime selection outperform baseline?"

Refactored to use repository pattern - SQL queries moved to analytics_repository.py.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.admin.services.analytics_models import (
    DriftBucketItem,
    DriftDriverItem,
    DriftDriversResponse,
    RegimeBacklogItem,
    RegimeBacklogRequest,
    RegimeBacklogResponse,
    RegimeCoverageItem,
    RegimeCoverageResponse,
    RegimeDriftResponse,
    SuggestedActions,
    TierDistribution,
    TierUsageItem,
    TierUsageResponse,
    UpliftItem,
    UpliftResponse,
)
from app.admin.services.analytics_queries import get_tier_usage_time_series
from app.admin.services.analytics_repository import (
    CONFIDENCE_TREND_THRESHOLD,
    DEFAULT_BACKLOG_SYMBOLS,
    DEFAULT_BACKLOG_TIMEFRAMES,
    DRIFT_TREND_THRESHOLD_PP,
    MIN_BUCKETS_FOR_TREND,
    PRIORITY_HIGH_THRESHOLD,
    PRIORITY_MEDIUM_THRESHOLD,
    AnalyticsRepository,
    QueryBuilder,
)
from app.deps.security import require_admin_token
from app.repositories.alerts import AlertsRepository

router = APIRouter(prefix="/analytics", tags=["admin-analytics"])
logger = structlog.get_logger(__name__)

# Setup Jinja2 templates (same dir as main admin templates)
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup via set_db_pool)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for analytics routes."""
    global _db_pool
    _db_pool = pool


def _require_db_pool():
    """Raise 503 if database pool is not available."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return _db_pool


def _get_repository() -> AnalyticsRepository:
    """Get analytics repository with active pool."""
    return AnalyticsRepository(_require_db_pool())


# =============================================================================
# HTML Page
# =============================================================================


@router.get(
    "/regimes",
    response_class=HTMLResponse,
    summary="Regime Analytics Dashboard",
    description="Interactive dashboard for regime coverage, tier usage, and uplift analysis.",
)
async def analytics_regimes_page(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    token: Optional[str] = Query(None, description="Admin token (dev convenience)"),
    _: bool = Depends(require_admin_token),
) -> HTMLResponse:
    """Render the regime analytics dashboard page."""
    admin_token = request.headers.get("X-Admin-Token", "") or token or ""

    recent_alerts = []
    if _db_pool is not None:
        try:
            alerts_repo = AlertsRepository(_db_pool)
            alerts, _count = await alerts_repo.list_events(
                workspace_id=workspace_id,
                limit=7,
                offset=0,
            )
            recent_alerts = alerts
        except Exception as e:
            logger.warning("failed_to_fetch_recent_alerts", error=str(e))

    return templates.TemplateResponse(
        "analytics_regimes.html",
        {
            "request": request,
            "workspace_id": str(workspace_id),
            "admin_token": admin_token,
            "recent_alerts": recent_alerts,
        },
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/regime-coverage",
    response_model=RegimeCoverageResponse,
    summary="Get regime coverage stats",
    description="""
Analyze regime coverage: "Do we have enough data per regime?"

Returns per-regime stats:
- n_tunes, n_runs: volume metrics
- avg/p50/p90 best_oos_score: quality metrics
- min_samples_met: whether regime has enough data for reliable recommendations
""",
)
async def get_regime_coverage(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    strategy_entity_id: Optional[UUID] = Query(
        None, description="Filter by strategy (recommended)"
    ),
    min_samples: int = Query(5, ge=1, le=100, description="Threshold for coverage"),
    since: Optional[datetime] = Query(None, description="Filter tunes created after"),
    _: bool = Depends(require_admin_token),
) -> RegimeCoverageResponse:
    """Get regime coverage statistics."""
    repo = _get_repository()
    rows = await repo.get_regime_coverage(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
        since=since,
    )

    items = []
    regimes_meeting = 0

    for row in rows:
        meets_threshold = row.n_tunes >= min_samples
        if meets_threshold:
            regimes_meeting += 1

        items.append(
            RegimeCoverageItem(
                regime_key=row.regime_key,
                trend_tag=row.trend_tag,
                vol_tag=row.vol_tag,
                n_tunes=row.n_tunes,
                n_runs=row.n_runs,
                avg_best_oos=row.avg_best_oos,
                p50_best_oos=row.p50_best_oos,
                p90_best_oos=row.p90_best_oos,
                min_samples_met=meets_threshold,
            )
        )

    total = len(items)
    coverage_pct = (regimes_meeting / total * 100) if total > 0 else 0.0

    return RegimeCoverageResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        min_samples=min_samples,
        total_regimes=total,
        regimes_meeting_threshold=regimes_meeting,
        coverage_pct=round(coverage_pct, 1),
        items=items,
    )


@router.get(
    "/tier-usage",
    response_model=None,  # Union type, handled manually
    summary="Get tier usage distribution",
    description="""
Analyze tier usage: "How often are we falling back?"

Shows distribution of recommendations across tiers:
- exact: Best case, regime-specific recommendations
- partial_trend/partial_vol: Partial regime match
- distance: Feature similarity fallback
- global_best: No regime match, using global best

**Time-series mode:** Add `bucket=day` or `bucket=week` to get trends over time.
""",
)
async def get_tier_usage(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    period_days: int = Query(30, ge=1, le=365, description="Analysis period"),
    regime_key: Optional[str] = Query(
        None, description="Filter to specific regime (e.g., 'trend=flat|vol=high_vol')"
    ),
    bucket: Optional[str] = Query(
        None,
        description="Time bucket: 'day' or 'week'. Omit for totals only.",
        pattern="^(day|week)$",
    ),
    _: bool = Depends(require_admin_token),
):
    """Get tier usage distribution from recommend_events."""
    pool = _require_db_pool()
    since = datetime.utcnow() - timedelta(days=period_days)

    # Build query conditions for time-series mode (uses legacy helper)
    qb = QueryBuilder()
    qb.add("workspace_id = ?", workspace_id)
    qb.add("created_at >= ?", since)
    qb.add_if("strategy_entity_id = ?", strategy_entity_id)
    qb.add_if("query_regime_key = ?", regime_key)

    # Time-series mode delegates to existing helper
    if bucket:
        return await get_tier_usage_time_series(
            pool=pool,
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            period_days=period_days,
            bucket=bucket,
            where_clause=qb.where_clause,
            params=qb.params,
        )

    # Totals mode uses repository
    repo = AnalyticsRepository(pool)
    rows = await repo.get_tier_usage_totals(
        workspace_id=workspace_id,
        since=since,
        strategy_entity_id=strategy_entity_id,
        regime_key=regime_key,
    )

    total = sum(r.count for r in rows)
    items = [
        TierUsageItem(
            tier=r.tier_used,
            count=r.count,
            pct=round((r.count / total * 100) if total > 0 else 0.0, 1),
            avg_confidence=round(r.avg_confidence, 2),
            avg_candidate_count=round(r.avg_candidate_count, 1),
        )
        for r in rows
    ]

    return TierUsageResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        total_recommendations=total,
        items=items,
    )


@router.get(
    "/uplift",
    response_model=UpliftResponse,
    summary="Get regime selection uplift",
    description="""
Analyze value add: "Does regime selection outperform baseline?"

Computes:
- baseline = best params overall (top OOS score across all regimes)
- regime_selected = top candidate returned by tiered matcher
- uplift = score(regime_selected) - score(baseline)

Positive uplift indicates regime-aware selection adds value.
""",
)
async def get_uplift(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    period_days: int = Query(30, ge=1, le=365, description="Analysis period"),
    regime_key: Optional[str] = Query(
        None, description="Filter to specific regime (e.g., 'trend=flat|vol=high_vol')"
    ),
    _: bool = Depends(require_admin_token),
) -> UpliftResponse:
    """Get uplift analysis comparing regime selection to baseline."""
    repo = _get_repository()
    since = datetime.utcnow() - timedelta(days=period_days)

    # Get baseline and uplift data
    baseline_score = await repo.get_baseline_score(
        workspace_id=workspace_id,
        strategy_entity_id=strategy_entity_id,
    )

    regime_rows = await repo.get_uplift_by_regime(
        workspace_id=workspace_id,
        since=since,
        strategy_entity_id=strategy_entity_id,
        regime_key=regime_key,
    )

    tier_rows = await repo.get_uplift_by_tier(
        workspace_id=workspace_id,
        since=since,
        strategy_entity_id=strategy_entity_id,
        regime_key=regime_key,
    )

    # Transform to response models
    def make_uplift_item(row, group_type: str) -> UpliftItem:
        uplift = row.avg_selected_score - baseline_score
        uplift_pct = (uplift / baseline_score * 100) if baseline_score != 0 else 0.0
        return UpliftItem(
            group_key=row.group_key,
            group_type=group_type,
            n_recommendations=row.n_recommendations,
            avg_selected_score=round(row.avg_selected_score, 4),
            avg_baseline_score=round(baseline_score, 4),
            uplift=round(uplift, 4),
            uplift_pct=round(uplift_pct, 2),
        )

    by_regime = [make_uplift_item(r, "regime") for r in regime_rows]
    by_tier = [make_uplift_item(r, "tier") for r in tier_rows]

    # Compute overall uplift (weighted by n_recommendations)
    total_recs = sum(item.n_recommendations for item in by_tier)
    overall_uplift = (
        sum(item.uplift * item.n_recommendations for item in by_tier) / total_recs
        if total_recs > 0
        else 0.0
    )

    return UpliftResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        baseline_score=round(baseline_score, 4),
        overall_uplift=round(overall_uplift, 4),
        by_regime=by_regime,
        by_tier=by_tier,
    )


@router.post(
    "/regime-backlog",
    response_model=RegimeBacklogResponse,
    summary="Generate tuning backlog from coverage gaps",
    description="""
Generate actionable tuning tasks from regime coverage gaps.

For each regime below min_samples threshold, returns:
- missing_samples: how many more tunes are needed
- suggested_actions: n_tunes, timeframes, symbols, priority

Priority levels:
- high: missing >= 3 samples
- medium: missing == 2 samples
- low: missing == 1 sample

This closes the analytics → execution loop by turning insights into tasks.
""",
)
async def generate_regime_backlog(
    request: RegimeBacklogRequest,
    _: bool = Depends(require_admin_token),
) -> RegimeBacklogResponse:
    """Generate tuning backlog from regime coverage gaps."""
    repo = _get_repository()

    # Handle focus mode: auto-select drift-driving regimes
    only_regimes = request.only_regimes
    if request.focus == "drift_drivers" and not only_regimes:
        only_regimes = await repo.get_drift_driver_regime_keys(
            workspace_id=request.workspace_id,
            strategy_entity_id=request.strategy_entity_id,
            limit=request.max_items,
        )

    rows = await repo.get_regime_backlog(
        workspace_id=request.workspace_id,
        min_samples=request.min_samples,
        max_items=request.max_items,
        strategy_entity_id=request.strategy_entity_id,
        only_regimes=only_regimes,
    )

    regimes = []
    total_missing = 0

    for row in rows:
        missing = request.min_samples - row.n_tunes
        total_missing += missing

        # Determine priority based on gap size
        if missing >= PRIORITY_HIGH_THRESHOLD:
            priority = "high"
        elif missing >= PRIORITY_MEDIUM_THRESHOLD:
            priority = "medium"
        else:
            priority = "low"

        regimes.append(
            RegimeBacklogItem(
                regime_key=row.regime_key,
                trend_tag=row.trend_tag,
                vol_tag=row.vol_tag,
                current_tunes=row.n_tunes,
                missing_samples=missing,
                suggested_actions=SuggestedActions(
                    n_tunes=missing,
                    timeframes=DEFAULT_BACKLOG_TIMEFRAMES,
                    symbols=DEFAULT_BACKLOG_SYMBOLS,
                    priority=priority,
                ),
            )
        )

    return RegimeBacklogResponse(
        workspace_id=str(request.workspace_id),
        strategy_entity_id=(
            str(request.strategy_entity_id) if request.strategy_entity_id else None
        ),
        min_samples=request.min_samples,
        total_gaps=len(regimes),
        total_missing=total_missing,
        regimes=regimes,
    )


def _compute_trend(
    values: list[float], threshold: float, increasing_label: str, decreasing_label: str
) -> str:
    """Compute trend by comparing first half vs second half averages.

    Args:
        values: List of numeric values (chronologically ordered)
        threshold: Minimum difference to consider significant
        increasing_label: Label for upward trend
        decreasing_label: Label for downward trend

    Returns:
        Trend label: increasing_label, decreasing_label, 'stable', or 'insufficient_data'
    """
    if len(values) < MIN_BUCKETS_FOR_TREND:
        return "insufficient_data" if len(values) < 2 else "stable"

    mid = len(values) // 2
    first_half_avg = sum(values[:mid]) / mid
    second_half_avg = sum(values[mid:]) / (len(values) - mid)
    diff = second_half_avg - first_half_avg

    if diff > threshold:
        return increasing_label
    elif diff < -threshold:
        return decreasing_label
    return "stable"


@router.get(
    "/regime-drift",
    response_model=RegimeDriftResponse,
    summary="Get regime drift indicators",
    description="""
Analyze regime stability over time.

Returns metrics for detecting regime drift:
- non_exact_pct: % of recommendations falling back from exact tier
- unique_regimes: Number of distinct regimes queried
- top_regime_pct: Concentration in most common regime (lower = more diverse)
- drift_trend: 'increasing', 'decreasing', 'stable', or 'insufficient_data'

High non_exact_pct or decreasing top_regime_pct may indicate:
- Market regime shifts
- Need for more training data
- Model staleness

This is a precursor to automated drift alerts.
""",
)
async def get_regime_drift(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    period_days: int = Query(30, ge=7, le=365, description="Analysis period"),
    regime_key: Optional[str] = Query(
        None, description="Filter to specific regime (e.g., 'trend=flat|vol=high_vol')"
    ),
    bucket: str = Query("day", pattern="^(day|week)$", description="Time bucket"),
    _: bool = Depends(require_admin_token),
) -> RegimeDriftResponse:
    """Get regime drift indicators from recommend_events."""
    repo = _get_repository()
    since = datetime.utcnow() - timedelta(days=period_days)

    # Fetch data
    bucket_rows = await repo.get_drift_buckets(
        workspace_id=workspace_id,
        since=since,
        bucket=bucket,
        strategy_entity_id=strategy_entity_id,
        regime_key=regime_key,
    )

    if not bucket_rows:
        return RegimeDriftResponse(
            workspace_id=str(workspace_id),
            strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
            period_days=period_days,
            bucket=bucket,
            drift_trend="insufficient_data",
            total_recommendations=0,
            overall_non_exact_pct=0.0,
            overall_unique_regimes=0,
            overall_avg_confidence=0.0,
            confidence_trend="insufficient_data",
            avg_daily_churn=0.0,
        )

    overall_unique = await repo.get_overall_unique_regimes(
        workspace_id=workspace_id,
        since=since,
        strategy_entity_id=strategy_entity_id,
        regime_key=regime_key,
    )

    # Build series and compute aggregates
    series = []
    total_recs = 0
    total_non_exact = 0
    non_exact_pcts = []
    confidence_scores = []
    weighted_confidence_sum = 0.0

    for row in bucket_rows:
        non_exact_pct = (
            (row.non_exact_count / row.total * 100) if row.total > 0 else 0.0
        )
        top_regime_pct = (
            (row.top_regime_count / row.total * 100) if row.total > 0 else 0.0
        )

        total_recs += row.total
        total_non_exact += row.non_exact_count
        non_exact_pcts.append(non_exact_pct)
        confidence_scores.append(row.avg_confidence)
        weighted_confidence_sum += row.avg_confidence * row.total

        series.append(
            DriftBucketItem(
                bucket_start=row.bucket_start.isoformat(),
                total=row.total,
                non_exact_count=row.non_exact_count,
                non_exact_pct=round(non_exact_pct, 1),
                unique_regimes=row.unique_regimes,
                top_regime_pct=round(top_regime_pct, 1),
                avg_confidence=round(row.avg_confidence, 3),
            )
        )

    # Calculate overall metrics
    overall_non_exact_pct = (
        (total_non_exact / total_recs * 100) if total_recs > 0 else 0.0
    )
    overall_avg_conf = weighted_confidence_sum / total_recs if total_recs > 0 else 0.0

    # Compute trends
    drift_trend = _compute_trend(
        non_exact_pcts, DRIFT_TREND_THRESHOLD_PP, "increasing", "decreasing"
    )
    confidence_trend = _compute_trend(
        confidence_scores, CONFIDENCE_TREND_THRESHOLD, "improving", "degrading"
    )

    # Calculate avg daily churn
    avg_daily_churn = overall_unique / len(series) if series else 0.0

    return RegimeDriftResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        bucket=bucket,
        total_recommendations=total_recs,
        overall_non_exact_pct=round(overall_non_exact_pct, 1),
        overall_unique_regimes=overall_unique,
        overall_avg_confidence=round(overall_avg_conf, 3),
        drift_trend=drift_trend,
        confidence_trend=confidence_trend,
        avg_daily_churn=round(avg_daily_churn, 2),
        series=series,
    )


@router.get(
    "/drift-drivers",
    response_model=DriftDriversResponse,
    summary="Get top regimes driving non-exact fallback",
    description="""
Identify which regimes are driving drift (non-exact tier usage).

Returns top regimes ranked by:
- non_exact_count: absolute count of fallbacks
- non_exact_pct: percentage of requests falling back
- wow_change: week-over-week change in fallback rate

For each regime, includes:
- tier_distribution: breakdown of which tiers were used
- coverage_gap: how many more tunes needed (from backtest_tunes)

Use this to prioritize tuning efforts on the highest-impact regimes.
""",
)
async def get_drift_drivers(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    strategy_entity_id: Optional[UUID] = Query(None, description="Filter by strategy"),
    period_days: int = Query(7, ge=1, le=90, description="Analysis period (default 7)"),
    min_samples: int = Query(5, ge=1, le=100, description="Target threshold for gap"),
    limit: int = Query(5, ge=1, le=20, description="Max regimes to return"),
    _: bool = Depends(require_admin_token),
) -> DriftDriversResponse:
    """Get top regimes driving non-exact fallback."""
    repo = _get_repository()
    since = datetime.utcnow() - timedelta(days=period_days)
    prior_week_start = since - timedelta(days=7)

    # Fetch current period data
    driver_rows = await repo.get_drift_drivers(
        workspace_id=workspace_id,
        since=since,
        limit=limit,
        strategy_entity_id=strategy_entity_id,
    )

    # Fetch prior week for WoW comparison
    prior_lookup = await repo.get_prior_week_drift(
        workspace_id=workspace_id,
        prior_start=prior_week_start,
        prior_end=since,
        strategy_entity_id=strategy_entity_id,
    )

    # Build response
    drivers = []
    total_non_exact = 0

    for row in driver_rows:
        non_exact_pct = (
            (row.non_exact_count / row.total_requests * 100)
            if row.total_requests > 0
            else 0.0
        )
        total_non_exact += row.non_exact_count

        # Calculate WoW change
        wow_change = None
        prior_pct = prior_lookup.get(row.regime_key)
        if prior_pct is not None:
            wow_change = round(non_exact_pct - prior_pct, 1)

        # Calculate coverage gap
        coverage_gap = max(0, min_samples - row.current_tunes)

        drivers.append(
            DriftDriverItem(
                regime_key=row.regime_key,
                trend_tag=row.trend_tag,
                vol_tag=row.vol_tag,
                total_requests=row.total_requests,
                non_exact_count=row.non_exact_count,
                non_exact_pct=round(non_exact_pct, 1),
                wow_change=wow_change,
                tier_distribution=TierDistribution(
                    exact=row.exact_count,
                    partial_trend=row.partial_trend_count,
                    partial_vol=row.partial_vol_count,
                    distance=row.distance_count,
                    global_best=row.global_count,
                ),
                current_tunes=row.current_tunes,
                coverage_gap=coverage_gap,
            )
        )

    return DriftDriversResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        min_samples=min_samples,
        total_non_exact=total_non_exact,
        drivers=drivers,
    )
