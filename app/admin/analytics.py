"""Admin analytics endpoints for KB attribution dashboard.

Provides analytics endpoints for:
1. Regime Coverage: "Do we have enough data per regime?"
2. Tier Usage: "How often are we falling back?"
3. Value Add (Uplift): "Does regime selection outperform baseline?"
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.deps.security import require_admin_token

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


# =============================================================================
# Response Models
# =============================================================================


class RegimeCoverageItem(BaseModel):
    """Coverage stats for a single regime."""

    regime_key: str
    trend_tag: Optional[str] = None
    vol_tag: Optional[str] = None
    n_tunes: int = Field(..., description="Number of tunes with this regime")
    n_runs: int = Field(..., description="Total runs across tunes")
    avg_best_oos: Optional[float] = Field(None, description="Mean best_oos_score")
    p50_best_oos: Optional[float] = Field(None, description="Median best_oos_score")
    p90_best_oos: Optional[float] = Field(None, description="90th percentile score")
    min_samples_met: bool = Field(
        ..., description="Whether n_tunes >= min_samples threshold"
    )


class RegimeCoverageResponse(BaseModel):
    """Response from regime coverage endpoint."""

    workspace_id: str
    strategy_entity_id: Optional[str] = None
    min_samples: int = Field(5, description="Threshold for min_samples_met")
    total_regimes: int = Field(0, description="Total unique regimes")
    regimes_meeting_threshold: int = Field(
        0, description="Regimes with n_tunes >= min_samples"
    )
    coverage_pct: float = Field(
        0.0, description="Percentage of regimes meeting threshold"
    )
    items: list[RegimeCoverageItem] = Field(default_factory=list)


class TierUsageItem(BaseModel):
    """Usage stats for a single tier."""

    tier: str
    count: int = Field(..., description="Number of recommendations using this tier")
    pct: float = Field(..., description="Percentage of total recommendations")
    avg_confidence: float = Field(..., description="Average confidence score")
    avg_candidate_count: float = Field(..., description="Average candidates returned")


class TierUsageResponse(BaseModel):
    """Response from tier usage endpoint."""

    workspace_id: str
    strategy_entity_id: Optional[str] = None
    period_days: int = Field(30, description="Analysis period in days")
    total_recommendations: int = Field(0, description="Total recommend calls")
    items: list[TierUsageItem] = Field(default_factory=list)


class TierUsageBucketItem(BaseModel):
    """Time-series data point for tier usage."""

    bucket_start: str = Field(..., description="Bucket start timestamp (ISO format)")
    tier: str = Field(..., description="Tier name")
    count: int = Field(..., description="Number of recommendations in this bucket")
    pct: float = Field(..., description="Percentage within this bucket")
    avg_confidence: float = Field(..., description="Average confidence in this bucket")


class BucketConfidence(BaseModel):
    """Overall confidence stats for a time bucket."""

    bucket_start: str = Field(..., description="Bucket start timestamp (ISO format)")
    avg_confidence: float = Field(..., description="Average confidence across all tiers")
    n: int = Field(..., description="Number of recommendations in this bucket")


class TierUsageTimeSeriesResponse(BaseModel):
    """Time-series response from tier usage endpoint."""

    workspace_id: str
    strategy_entity_id: Optional[str] = None
    period_days: int = Field(30, description="Analysis period in days")
    bucket: str = Field(..., description="Bucket size: 'day' or 'week'")
    total_recommendations: int = Field(0, description="Total recommend calls")
    buckets: list[str] = Field(
        default_factory=list, description="Ordered list of bucket timestamps"
    )
    series: list[TierUsageBucketItem] = Field(
        default_factory=list, description="Time-series data points"
    )
    confidence_series: list[BucketConfidence] = Field(
        default_factory=list, description="Per-bucket confidence for trend overlay"
    )


class UpliftItem(BaseModel):
    """Uplift stats for a grouping (regime or tier)."""

    group_key: str = Field(..., description="Regime key or tier name")
    group_type: str = Field(..., description="'regime' or 'tier'")
    n_recommendations: int = Field(..., description="Number of recommendations")
    avg_selected_score: float = Field(
        ..., description="Avg score of regime-selected candidates"
    )
    avg_baseline_score: float = Field(
        ..., description="Avg score of global baseline (top overall)"
    )
    uplift: float = Field(
        ..., description="selected - baseline (positive = regime adds value)"
    )
    uplift_pct: float = Field(..., description="Relative uplift percentage")


class UpliftResponse(BaseModel):
    """Response from uplift endpoint."""

    workspace_id: str
    strategy_entity_id: Optional[str] = None
    period_days: int = Field(30, description="Analysis period in days")
    baseline_score: float = Field(
        ..., description="Global baseline (top OOS score across all regimes)"
    )
    overall_uplift: float = Field(..., description="Mean uplift across all groups")
    by_regime: list[UpliftItem] = Field(default_factory=list)
    by_tier: list[UpliftItem] = Field(default_factory=list)


class SuggestedActions(BaseModel):
    """Suggested tuning actions for a regime gap."""

    n_tunes: int = Field(..., description="Number of tunes to run")
    timeframes: list[str] = Field(
        default_factory=list, description="Suggested timeframes"
    )
    symbols: list[str] = Field(default_factory=list, description="Suggested symbols")
    priority: str = Field(..., description="Priority: 'high', 'medium', 'low'")


class RegimeBacklogItem(BaseModel):
    """A regime with coverage gap and suggested actions."""

    regime_key: str
    trend_tag: Optional[str] = None
    vol_tag: Optional[str] = None
    current_tunes: int = Field(..., description="Current tune count")
    missing_samples: int = Field(..., description="Tunes needed to reach min_samples")
    suggested_actions: SuggestedActions


class RegimeBacklogRequest(BaseModel):
    """Request body for generating regime backlog."""

    workspace_id: UUID
    strategy_entity_id: Optional[UUID] = None
    min_samples: int = Field(5, ge=1, le=100, description="Target sample threshold")
    max_items: int = Field(10, ge=1, le=50, description="Max regimes to return")
    # Optional filters for focused backlog generation
    only_regimes: Optional[list[str]] = Field(
        None, description="Filter to specific regime_keys (e.g., from drift drivers)"
    )
    focus: Optional[str] = Field(
        None,
        description="Focus mode: 'drift_drivers' to auto-select top drift-driving regimes",
    )


class RegimeBacklogResponse(BaseModel):
    """Response from regime backlog generator."""

    workspace_id: str
    strategy_entity_id: Optional[str] = None
    min_samples: int = Field(5, description="Target threshold")
    total_gaps: int = Field(0, description="Total regimes below threshold")
    total_missing: int = Field(0, description="Sum of all missing samples")
    regimes: list[RegimeBacklogItem] = Field(default_factory=list)


class DriftBucketItem(BaseModel):
    """Drift metrics for a time bucket."""

    bucket_start: str = Field(..., description="Bucket start timestamp (ISO format)")
    total: int = Field(..., description="Total recommendations in bucket")
    non_exact_count: int = Field(..., description="Recommendations not using exact tier")
    non_exact_pct: float = Field(..., description="Percentage not using exact tier")
    unique_regimes: int = Field(..., description="Distinct query_regime_key values")
    top_regime_pct: float = Field(
        ..., description="Percentage of recommendations in most common regime"
    )


class RegimeDriftResponse(BaseModel):
    """Response from regime drift endpoint."""

    workspace_id: str
    strategy_entity_id: Optional[str] = None
    period_days: int = Field(30, description="Analysis period in days")
    bucket: str = Field("day", description="Bucket size: 'day' or 'week'")
    # Overall metrics
    total_recommendations: int = Field(0, description="Total recommend calls")
    overall_non_exact_pct: float = Field(
        0.0, description="Overall % not using exact tier"
    )
    overall_unique_regimes: int = Field(0, description="Total distinct regimes seen")
    # Trend indicators
    drift_trend: str = Field(
        "stable",
        description="Trend: 'increasing', 'decreasing', 'stable', 'insufficient_data'",
    )
    avg_daily_churn: float = Field(
        0.0, description="Average daily regime churn (new regimes per day)"
    )
    # Time-series
    series: list[DriftBucketItem] = Field(default_factory=list)


class TierDistribution(BaseModel):
    """Distribution of tier usage for a regime."""

    exact: int = Field(0, description="Count using exact tier")
    partial_trend: int = Field(0, description="Count using partial_trend tier")
    partial_vol: int = Field(0, description="Count using partial_vol tier")
    distance: int = Field(0, description="Count using distance tier")
    global_best: int = Field(0, description="Count using global_best tier")


class DriftDriverItem(BaseModel):
    """A regime driving non-exact fallback."""

    regime_key: str
    trend_tag: Optional[str] = None
    vol_tag: Optional[str] = None
    # Drift metrics
    total_requests: int = Field(..., description="Total recommendations for this regime")
    non_exact_count: int = Field(..., description="Recommendations not using exact tier")
    non_exact_pct: float = Field(..., description="% not using exact tier")
    # Week-over-week change
    wow_change: Optional[float] = Field(
        None, description="Week-over-week change in non_exact_pct (pp)"
    )
    # Tier distribution
    tier_distribution: TierDistribution
    # Coverage gap (joined from backtest_tunes)
    current_tunes: int = Field(0, description="Current tune count for this regime")
    coverage_gap: int = Field(0, description="Tunes needed to reach min_samples")


class DriftDriversResponse(BaseModel):
    """Response from drift-drivers endpoint."""

    workspace_id: str
    strategy_entity_id: Optional[str] = None
    period_days: int = Field(7, description="Analysis period in days")
    min_samples: int = Field(5, description="Target sample threshold for coverage gap")
    total_non_exact: int = Field(0, description="Total non-exact recommendations")
    drivers: list[DriftDriverItem] = Field(default_factory=list)


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
    # Get admin token from header or query param (query param for dev convenience)
    admin_token = request.headers.get("X-Admin-Token", "") or token or ""

    return templates.TemplateResponse(
        "analytics_regimes.html",
        {
            "request": request,
            "workspace_id": str(workspace_id),
            "admin_token": admin_token,
        },
    )


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_drift_driver_regimes(
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID] = None,
    limit: int = 5,
    period_days: int = 7,
) -> list[str]:
    """Get top drift-driving regime keys for backlog focus mode."""
    if _db_pool is None:
        return []

    since = datetime.utcnow() - timedelta(days=period_days)

    conditions = ["workspace_id = $1", "created_at >= $2", "query_regime_key IS NOT NULL"]
    params: list = [workspace_id, since]
    param_idx = 3

    if strategy_entity_id:
        conditions.append(f"strategy_entity_id = ${param_idx}")
        params.append(strategy_entity_id)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            query_regime_key,
            SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact_count
        FROM recommend_events
        WHERE {where_clause}
        GROUP BY query_regime_key
        HAVING SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) > 0
        ORDER BY non_exact_count DESC
        LIMIT ${param_idx}
    """
    params.append(limit)

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [row["query_regime_key"] for row in rows]


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
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    # Build query conditions
    conditions = ["t.workspace_id = $1", "t.regime_key IS NOT NULL"]
    params: list = [workspace_id]
    param_idx = 2

    if strategy_entity_id:
        conditions.append(f"t.strategy_entity_id = ${param_idx}")
        params.append(strategy_entity_id)
        param_idx += 1

    if since:
        conditions.append(f"t.created_at >= ${param_idx}")
        params.append(since)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            t.regime_key,
            t.trend_tag,
            t.vol_tag,
            COUNT(DISTINCT t.id) as n_tunes,
            COUNT(DISTINCT tr.run_id) as n_runs,
            AVG(t.best_oos_score) as avg_best_oos,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.best_oos_score) as p50_best_oos,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY t.best_oos_score) as p90_best_oos
        FROM backtest_tunes t
        LEFT JOIN backtest_tune_runs tr ON tr.tune_id = t.id AND tr.status = 'completed'
        WHERE {where_clause}
          AND t.status = 'completed'
          AND t.best_oos_score IS NOT NULL
        GROUP BY t.regime_key, t.trend_tag, t.vol_tag
        ORDER BY n_tunes DESC, avg_best_oos DESC NULLS LAST
    """

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    items = []
    regimes_meeting = 0

    for row in rows:
        n_tunes = row["n_tunes"]
        meets_threshold = n_tunes >= min_samples
        if meets_threshold:
            regimes_meeting += 1

        items.append(
            RegimeCoverageItem(
                regime_key=row["regime_key"],
                trend_tag=row["trend_tag"],
                vol_tag=row["vol_tag"],
                n_tunes=n_tunes,
                n_runs=row["n_runs"],
                avg_best_oos=(
                    float(row["avg_best_oos"]) if row["avg_best_oos"] else None
                ),
                p50_best_oos=(
                    float(row["p50_best_oos"]) if row["p50_best_oos"] else None
                ),
                p90_best_oos=(
                    float(row["p90_best_oos"]) if row["p90_best_oos"] else None
                ),
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
    bucket: Optional[str] = Query(
        None,
        description="Time bucket: 'day' or 'week'. Omit for totals only.",
        regex="^(day|week)$",
    ),
    _: bool = Depends(require_admin_token),
):
    """Get tier usage distribution from recommend_events."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    since = datetime.utcnow() - timedelta(days=period_days)

    # Build query conditions
    conditions = ["workspace_id = $1", "created_at >= $2"]
    params: list = [workspace_id, since]
    param_idx = 3

    if strategy_entity_id:
        conditions.append(f"strategy_entity_id = ${param_idx}")
        params.append(strategy_entity_id)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    # Time-series mode
    if bucket:
        return await _get_tier_usage_time_series(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            period_days=period_days,
            bucket=bucket,
            where_clause=where_clause,
            params=params,
        )

    # Totals mode (default)
    query = f"""
        SELECT
            tier_used,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            AVG(candidate_count) as avg_candidate_count
        FROM recommend_events
        WHERE {where_clause}
        GROUP BY tier_used
        ORDER BY count DESC
    """

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    total = sum(row["count"] for row in rows)
    items = []

    for row in rows:
        count = row["count"]
        pct = (count / total * 100) if total > 0 else 0.0

        items.append(
            TierUsageItem(
                tier=row["tier_used"],
                count=count,
                pct=round(pct, 1),
                avg_confidence=round(float(row["avg_confidence"]), 2),
                avg_candidate_count=round(float(row["avg_candidate_count"]), 1),
            )
        )

    return TierUsageResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        total_recommendations=total,
        items=items,
    )


async def _get_tier_usage_time_series(
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID],
    period_days: int,
    bucket: str,
    where_clause: str,
    params: list,
) -> TierUsageTimeSeriesResponse:
    """Get tier usage as time-series data."""
    # Determine bucket truncation
    trunc_unit = "day" if bucket == "day" else "week"

    query = f"""
        SELECT
            date_trunc('{trunc_unit}', created_at) as bucket_start,
            tier_used,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM recommend_events
        WHERE {where_clause}
        GROUP BY bucket_start, tier_used
        ORDER BY bucket_start ASC, tier_used
    """

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    # Collect unique buckets and build series
    bucket_set: set[str] = set()
    bucket_totals: dict[str, int] = {}  # For computing percentages within each bucket
    # Track weighted confidence for overall bucket avg
    bucket_conf_sum: dict[str, float] = {}  # sum(count * avg_conf)
    series = []

    for row in rows:
        bucket_start = row["bucket_start"].isoformat()
        bucket_set.add(bucket_start)
        count = row["count"]
        avg_conf = float(row["avg_confidence"])
        bucket_totals[bucket_start] = bucket_totals.get(bucket_start, 0) + count
        bucket_conf_sum[bucket_start] = (
            bucket_conf_sum.get(bucket_start, 0.0) + count * avg_conf
        )

    # Build series with percentages
    for row in rows:
        bucket_start = row["bucket_start"].isoformat()
        count = row["count"]
        bucket_total = bucket_totals[bucket_start]
        pct = (count / bucket_total * 100) if bucket_total > 0 else 0.0

        series.append(
            TierUsageBucketItem(
                bucket_start=bucket_start,
                tier=row["tier_used"],
                count=count,
                pct=round(pct, 1),
                avg_confidence=round(float(row["avg_confidence"]), 2),
            )
        )

    # Sort buckets chronologically
    buckets = sorted(bucket_set)
    total = sum(bucket_totals.values())

    # Build confidence series (overall avg per bucket)
    confidence_series = []
    for b in buckets:
        n = bucket_totals[b]
        avg_conf = bucket_conf_sum[b] / n if n > 0 else 0.0
        confidence_series.append(
            BucketConfidence(
                bucket_start=b,
                avg_confidence=round(avg_conf, 3),
                n=n,
            )
        )

    return TierUsageTimeSeriesResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        bucket=bucket,
        total_recommendations=total,
        buckets=buckets,
        series=series,
        confidence_series=confidence_series,
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
    _: bool = Depends(require_admin_token),
) -> UpliftResponse:
    """Get uplift analysis comparing regime selection to baseline."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    since = datetime.utcnow() - timedelta(days=period_days)

    # Build base conditions
    base_conditions = ["workspace_id = $1", "created_at >= $2"]
    params: list = [workspace_id, since]
    param_idx = 3

    if strategy_entity_id:
        base_conditions.append(f"strategy_entity_id = ${param_idx}")
        params.append(strategy_entity_id)
        param_idx += 1

    where_clause = " AND ".join(base_conditions)

    # Get global baseline (top OOS score)
    baseline_query = f"""
        SELECT MAX(t.best_oos_score) as baseline
        FROM backtest_tunes t
        WHERE t.workspace_id = $1
          AND t.status = 'completed'
          AND t.best_oos_score IS NOT NULL
          {"AND t.strategy_entity_id = $" + str(param_idx - 1) if strategy_entity_id else ""}
    """
    baseline_params = [workspace_id]
    if strategy_entity_id:
        baseline_params.append(strategy_entity_id)

    async with _db_pool.acquire() as conn:
        baseline_row = await conn.fetchrow(baseline_query, *baseline_params)
        baseline_score = (
            float(baseline_row["baseline"]) if baseline_row["baseline"] else 0.0
        )

        # Get uplift by regime
        regime_query = f"""
            SELECT
                query_regime_key as group_key,
                COUNT(*) as n_recommendations,
                AVG(top_candidate_score) as avg_selected_score
            FROM recommend_events
            WHERE {where_clause}
              AND query_regime_key IS NOT NULL
              AND top_candidate_score IS NOT NULL
            GROUP BY query_regime_key
            ORDER BY n_recommendations DESC
        """
        regime_rows = await conn.fetch(regime_query, *params)

        # Get uplift by tier
        tier_query = f"""
            SELECT
                tier_used as group_key,
                COUNT(*) as n_recommendations,
                AVG(top_candidate_score) as avg_selected_score
            FROM recommend_events
            WHERE {where_clause}
              AND top_candidate_score IS NOT NULL
            GROUP BY tier_used
            ORDER BY n_recommendations DESC
        """
        tier_rows = await conn.fetch(tier_query, *params)

    by_regime = []
    for row in regime_rows:
        avg_selected = float(row["avg_selected_score"])
        uplift = avg_selected - baseline_score
        uplift_pct = (uplift / baseline_score * 100) if baseline_score != 0 else 0.0

        by_regime.append(
            UpliftItem(
                group_key=row["group_key"],
                group_type="regime",
                n_recommendations=row["n_recommendations"],
                avg_selected_score=round(avg_selected, 4),
                avg_baseline_score=round(baseline_score, 4),
                uplift=round(uplift, 4),
                uplift_pct=round(uplift_pct, 2),
            )
        )

    by_tier = []
    for row in tier_rows:
        avg_selected = float(row["avg_selected_score"])
        uplift = avg_selected - baseline_score
        uplift_pct = (uplift / baseline_score * 100) if baseline_score != 0 else 0.0

        by_tier.append(
            UpliftItem(
                group_key=row["group_key"],
                group_type="tier",
                n_recommendations=row["n_recommendations"],
                avg_selected_score=round(avg_selected, 4),
                avg_baseline_score=round(baseline_score, 4),
                uplift=round(uplift, 4),
                uplift_pct=round(uplift_pct, 2),
            )
        )

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
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    # Handle focus mode: auto-select drift-driving regimes
    only_regimes = request.only_regimes
    if request.focus == "drift_drivers" and not only_regimes:
        # Fetch top drift drivers and use their regime keys
        drift_drivers = await _get_drift_driver_regimes(
            workspace_id=request.workspace_id,
            strategy_entity_id=request.strategy_entity_id,
            limit=request.max_items,
        )
        only_regimes = drift_drivers

    # Build query conditions
    conditions = ["t.workspace_id = $1", "t.regime_key IS NOT NULL"]
    params: list = [request.workspace_id]
    param_idx = 2

    if request.strategy_entity_id:
        conditions.append(f"t.strategy_entity_id = ${param_idx}")
        params.append(request.strategy_entity_id)
        param_idx += 1

    # Add only_regimes filter if specified
    if only_regimes:
        placeholders = ", ".join(f"${param_idx + i}" for i in range(len(only_regimes)))
        conditions.append(f"t.regime_key IN ({placeholders})")
        params.extend(only_regimes)
        param_idx += len(only_regimes)

    where_clause = " AND ".join(conditions)

    # Query regimes with tune counts
    query = f"""
        SELECT
            t.regime_key,
            t.trend_tag,
            t.vol_tag,
            COUNT(DISTINCT t.id) as n_tunes
        FROM backtest_tunes t
        WHERE {where_clause}
          AND t.status = 'completed'
        GROUP BY t.regime_key, t.trend_tag, t.vol_tag
        HAVING COUNT(DISTINCT t.id) < ${param_idx}
        ORDER BY COUNT(DISTINCT t.id) ASC
        LIMIT ${param_idx + 1}
    """
    params.extend([request.min_samples, request.max_items])

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    # Default suggestions (can be made smarter later)
    default_timeframes = ["5m", "15m"]
    default_symbols = ["BTC-USDT", "ETH-USDT"]

    regimes = []
    total_missing = 0

    for row in rows:
        current = row["n_tunes"]
        missing = request.min_samples - current
        total_missing += missing

        # Determine priority based on gap size
        if missing >= 3:
            priority = "high"
        elif missing >= 2:
            priority = "medium"
        else:
            priority = "low"

        regimes.append(
            RegimeBacklogItem(
                regime_key=row["regime_key"],
                trend_tag=row["trend_tag"],
                vol_tag=row["vol_tag"],
                current_tunes=current,
                missing_samples=missing,
                suggested_actions=SuggestedActions(
                    n_tunes=missing,
                    timeframes=default_timeframes,
                    symbols=default_symbols,
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
    bucket: str = Query("day", regex="^(day|week)$", description="Time bucket"),
    _: bool = Depends(require_admin_token),
) -> RegimeDriftResponse:
    """Get regime drift indicators from recommend_events."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    since = datetime.utcnow() - timedelta(days=period_days)
    trunc_unit = "day" if bucket == "day" else "week"

    # Build query conditions
    conditions = ["workspace_id = $1", "created_at >= $2"]
    params: list = [workspace_id, since]
    param_idx = 3

    if strategy_entity_id:
        conditions.append(f"strategy_entity_id = ${param_idx}")
        params.append(strategy_entity_id)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    # Query per-bucket metrics
    query = f"""
        WITH bucket_stats AS (
            SELECT
                date_trunc('{trunc_unit}', created_at) as bucket_start,
                COUNT(*) as total,
                SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact_count,
                COUNT(DISTINCT query_regime_key) as unique_regimes
            FROM recommend_events
            WHERE {where_clause}
            GROUP BY bucket_start
        ),
        top_regime_per_bucket AS (
            SELECT
                date_trunc('{trunc_unit}', created_at) as bucket_start,
                query_regime_key,
                COUNT(*) as regime_count,
                ROW_NUMBER() OVER (
                    PARTITION BY date_trunc('{trunc_unit}', created_at)
                    ORDER BY COUNT(*) DESC
                ) as rn
            FROM recommend_events
            WHERE {where_clause}
              AND query_regime_key IS NOT NULL
            GROUP BY bucket_start, query_regime_key
        )
        SELECT
            bs.bucket_start,
            bs.total,
            bs.non_exact_count,
            bs.unique_regimes,
            COALESCE(tr.regime_count, 0) as top_regime_count
        FROM bucket_stats bs
        LEFT JOIN top_regime_per_bucket tr
            ON tr.bucket_start = bs.bucket_start AND tr.rn = 1
        ORDER BY bs.bucket_start ASC
    """

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        # Get overall unique regimes
        overall_query = f"""
            SELECT COUNT(DISTINCT query_regime_key) as unique_regimes
            FROM recommend_events
            WHERE {where_clause}
              AND query_regime_key IS NOT NULL
        """
        overall_row = await conn.fetchrow(overall_query, *params)

    if not rows:
        return RegimeDriftResponse(
            workspace_id=str(workspace_id),
            strategy_entity_id=(
                str(strategy_entity_id) if strategy_entity_id else None
            ),
            period_days=period_days,
            bucket=bucket,
            drift_trend="insufficient_data",
        )

    # Build series
    series = []
    total_recs = 0
    total_non_exact = 0
    non_exact_pcts = []

    for row in rows:
        total = row["total"]
        non_exact = row["non_exact_count"]
        non_exact_pct = (non_exact / total * 100) if total > 0 else 0.0
        top_regime_pct = (
            (row["top_regime_count"] / total * 100) if total > 0 else 0.0
        )

        total_recs += total
        total_non_exact += non_exact
        non_exact_pcts.append(non_exact_pct)

        series.append(
            DriftBucketItem(
                bucket_start=row["bucket_start"].isoformat(),
                total=total,
                non_exact_count=non_exact,
                non_exact_pct=round(non_exact_pct, 1),
                unique_regimes=row["unique_regimes"],
                top_regime_pct=round(top_regime_pct, 1),
            )
        )

    # Calculate overall metrics
    overall_non_exact_pct = (
        (total_non_exact / total_recs * 100) if total_recs > 0 else 0.0
    )
    overall_unique = overall_row["unique_regimes"] if overall_row else 0

    # Calculate drift trend (compare first half vs second half of period)
    drift_trend = "stable"
    if len(non_exact_pcts) >= 4:
        mid = len(non_exact_pcts) // 2
        first_half_avg = sum(non_exact_pcts[:mid]) / mid
        second_half_avg = sum(non_exact_pcts[mid:]) / (len(non_exact_pcts) - mid)
        diff = second_half_avg - first_half_avg

        if diff > 10:  # More than 10pp increase
            drift_trend = "increasing"
        elif diff < -10:  # More than 10pp decrease
            drift_trend = "decreasing"
        else:
            drift_trend = "stable"
    elif len(non_exact_pcts) < 2:
        drift_trend = "insufficient_data"

    # Calculate avg daily churn (new unique regimes appearing)
    # Simple heuristic: unique_regimes / days with data
    avg_daily_churn = overall_unique / len(series) if series else 0.0

    return RegimeDriftResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        bucket=bucket,
        total_recommendations=total_recs,
        overall_non_exact_pct=round(overall_non_exact_pct, 1),
        overall_unique_regimes=overall_unique,
        drift_trend=drift_trend,
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
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    since = datetime.utcnow() - timedelta(days=period_days)
    # For WoW comparison, we need prior week data
    prior_week_start = since - timedelta(days=7)

    # Build query conditions
    conditions = ["workspace_id = $1", "created_at >= $2"]
    params: list = [workspace_id, since]
    param_idx = 3

    if strategy_entity_id:
        conditions.append(f"strategy_entity_id = ${param_idx}")
        params.append(strategy_entity_id)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    # Build prior week conditions (for WoW)
    prior_conditions = ["workspace_id = $1", "created_at >= $2", "created_at < $3"]
    prior_params: list = [workspace_id, prior_week_start, since]
    prior_param_idx = 4

    if strategy_entity_id:
        prior_conditions.append(f"strategy_entity_id = ${prior_param_idx}")
        prior_params.append(strategy_entity_id)
        prior_param_idx += 1

    prior_where_clause = " AND ".join(prior_conditions)

    # Query per-regime drift stats with tier distribution
    query = f"""
        WITH regime_stats AS (
            SELECT
                query_regime_key,
                query_trend_tag,
                query_vol_tag,
                COUNT(*) as total_requests,
                SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact_count,
                SUM(CASE WHEN tier_used = 'exact' THEN 1 ELSE 0 END) as exact_count,
                SUM(CASE WHEN tier_used = 'partial_trend' THEN 1 ELSE 0 END) as partial_trend_count,
                SUM(CASE WHEN tier_used = 'partial_vol' THEN 1 ELSE 0 END) as partial_vol_count,
                SUM(CASE WHEN tier_used = 'distance' THEN 1 ELSE 0 END) as distance_count,
                SUM(CASE WHEN tier_used = 'global_best' THEN 1 ELSE 0 END) as global_count
            FROM recommend_events
            WHERE {where_clause}
              AND query_regime_key IS NOT NULL
            GROUP BY query_regime_key, query_trend_tag, query_vol_tag
        ),
        coverage AS (
            SELECT
                regime_key,
                COUNT(DISTINCT id) as n_tunes
            FROM backtest_tunes
            WHERE workspace_id = $1
              AND regime_key IS NOT NULL
              AND status = 'completed'
            GROUP BY regime_key
        )
        SELECT
            rs.query_regime_key as regime_key,
            rs.query_trend_tag as trend_tag,
            rs.query_vol_tag as vol_tag,
            rs.total_requests,
            rs.non_exact_count,
            rs.exact_count,
            rs.partial_trend_count,
            rs.partial_vol_count,
            rs.distance_count,
            rs.global_count,
            COALESCE(c.n_tunes, 0) as current_tunes
        FROM regime_stats rs
        LEFT JOIN coverage c ON c.regime_key = rs.query_regime_key
        WHERE rs.non_exact_count > 0
        ORDER BY rs.non_exact_count DESC
        LIMIT ${param_idx}
    """
    params.append(limit)

    # Query prior week stats for WoW comparison
    prior_query = f"""
        SELECT
            query_regime_key,
            COUNT(*) as total,
            SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact
        FROM recommend_events
        WHERE {prior_where_clause}
          AND query_regime_key IS NOT NULL
        GROUP BY query_regime_key
    """

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        prior_rows = await conn.fetch(prior_query, *prior_params)

    # Build prior week lookup for WoW calculation
    prior_lookup = {}
    for row in prior_rows:
        total = row["total"]
        non_exact = row["non_exact"]
        pct = (non_exact / total * 100) if total > 0 else 0.0
        prior_lookup[row["query_regime_key"]] = pct

    # Build response
    drivers = []
    total_non_exact = 0

    for row in rows:
        total = row["total_requests"]
        non_exact = row["non_exact_count"]
        non_exact_pct = (non_exact / total * 100) if total > 0 else 0.0
        total_non_exact += non_exact

        # Calculate WoW change
        wow_change = None
        prior_pct = prior_lookup.get(row["regime_key"])
        if prior_pct is not None:
            wow_change = round(non_exact_pct - prior_pct, 1)

        # Calculate coverage gap
        current_tunes = row["current_tunes"]
        coverage_gap = max(0, min_samples - current_tunes)

        drivers.append(
            DriftDriverItem(
                regime_key=row["regime_key"],
                trend_tag=row["trend_tag"],
                vol_tag=row["vol_tag"],
                total_requests=total,
                non_exact_count=non_exact,
                non_exact_pct=round(non_exact_pct, 1),
                wow_change=wow_change,
                tier_distribution=TierDistribution(
                    exact=row["exact_count"],
                    partial_trend=row["partial_trend_count"],
                    partial_vol=row["partial_vol_count"],
                    distance=row["distance_count"],
                    global_best=row["global_count"],
                ),
                current_tunes=current_tunes,
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
