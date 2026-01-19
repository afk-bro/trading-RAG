"""Analytics response models.

Pydantic models for regime coverage, tier usage, uplift, and drift analytics.
Extracted from analytics.py to keep router thin.
"""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Regime Coverage Models
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


# =============================================================================
# Tier Usage Models
# =============================================================================


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
    avg_confidence: float = Field(
        ..., description="Average confidence across all tiers"
    )
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


# =============================================================================
# Uplift Models
# =============================================================================


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
# Regime Backlog Models
# =============================================================================


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


# =============================================================================
# Drift Models
# =============================================================================


class DriftBucketItem(BaseModel):
    """Drift metrics for a time bucket."""

    bucket_start: str = Field(..., description="Bucket start timestamp (ISO format)")
    total: int = Field(..., description="Total recommendations in bucket")
    non_exact_count: int = Field(
        ..., description="Recommendations not using exact tier"
    )
    non_exact_pct: float = Field(..., description="Percentage not using exact tier")
    unique_regimes: int = Field(..., description="Distinct query_regime_key values")
    top_regime_pct: float = Field(
        ..., description="Percentage of recommendations in most common regime"
    )
    avg_confidence: float = Field(
        0.0, description="Average confidence score for this bucket"
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
    overall_avg_confidence: float = Field(
        0.0, description="Overall average confidence score"
    )
    # Trend indicators
    drift_trend: str = Field(
        "stable",
        description="Trend: 'increasing', 'decreasing', 'stable', 'insufficient_data'",
    )
    confidence_trend: str = Field(
        "stable",
        description="Confidence trend: 'improving', 'degrading', 'stable', 'insufficient_data'",
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
    total_requests: int = Field(
        ..., description="Total recommendations for this regime"
    )
    non_exact_count: int = Field(
        ..., description="Recommendations not using exact tier"
    )
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


__all__ = [
    # Regime Coverage
    "RegimeCoverageItem",
    "RegimeCoverageResponse",
    # Tier Usage
    "TierUsageItem",
    "TierUsageResponse",
    "TierUsageBucketItem",
    "BucketConfidence",
    "TierUsageTimeSeriesResponse",
    # Uplift
    "UpliftItem",
    "UpliftResponse",
    # Regime Backlog
    "SuggestedActions",
    "RegimeBacklogItem",
    "RegimeBacklogRequest",
    "RegimeBacklogResponse",
    # Drift
    "DriftBucketItem",
    "RegimeDriftResponse",
    "TierDistribution",
    "DriftDriverItem",
    "DriftDriversResponse",
]
