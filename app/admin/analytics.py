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
    _: bool = Depends(require_admin_token),
) -> HTMLResponse:
    """Render the regime analytics dashboard page."""
    # Get admin token from header for JS API calls
    admin_token = request.headers.get("X-Admin-Token", "")

    return templates.TemplateResponse(
        "analytics_regimes.html",
        {
            "request": request,
            "workspace_id": str(workspace_id),
            "admin_token": admin_token,
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
    series = []

    for row in rows:
        bucket_start = row["bucket_start"].isoformat()
        bucket_set.add(bucket_start)
        bucket_totals[bucket_start] = bucket_totals.get(bucket_start, 0) + row["count"]

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

    return TierUsageTimeSeriesResponse(
        workspace_id=str(workspace_id),
        strategy_entity_id=str(strategy_entity_id) if strategy_entity_id else None,
        period_days=period_days,
        bucket=bucket,
        total_recommendations=total,
        buckets=buckets,
        series=series,
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
