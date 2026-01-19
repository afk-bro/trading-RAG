"""Analytics query helpers.

Extracted from analytics.py to keep router thin.
All functions accept pool explicitly (no globals).
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import HTTPException, status

from app.admin.services.analytics_models import (
    BucketConfidence,
    TierUsageBucketItem,
    TierUsageTimeSeriesResponse,
)

logger = structlog.get_logger(__name__)


async def get_drift_driver_regimes(
    pool: Any,
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID] = None,
    limit: int = 5,
    period_days: int = 7,
) -> list[str]:
    """
    Get top drift-driving regime keys for backlog focus mode.

    Identifies regimes that most frequently fall back to non-exact tiers,
    indicating they need more tuning data.

    Args:
        pool: Database connection pool
        workspace_id: Workspace UUID
        strategy_entity_id: Optional strategy filter
        limit: Max regimes to return
        period_days: Analysis period in days

    Returns:
        List of regime keys sorted by non-exact count descending
    """
    if pool is None:
        return []

    since = datetime.utcnow() - timedelta(days=period_days)

    conditions = [
        "workspace_id = $1",
        "created_at >= $2",
        "query_regime_key IS NOT NULL",
    ]
    params: list[Any] = [workspace_id, since]
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

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [row["query_regime_key"] for row in rows]


async def get_tier_usage_time_series(
    pool: Any,
    workspace_id: UUID,
    strategy_entity_id: Optional[UUID],
    period_days: int,
    bucket: str,
    where_clause: str,
    params: list[Any],
) -> TierUsageTimeSeriesResponse:
    """
    Get tier usage as time-series data.

    Aggregates recommendation tier usage into daily or weekly buckets,
    computing counts, percentages, and confidence trends.

    Args:
        pool: Database connection pool
        workspace_id: Workspace UUID
        strategy_entity_id: Optional strategy filter (for response building)
        period_days: Analysis period in days (for response building)
        bucket: Bucket size: 'day' or 'week'
        where_clause: SQL WHERE clause (already built by caller)
        params: SQL parameters for where_clause

    Returns:
        TierUsageTimeSeriesResponse with series data
    """
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

    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database pool not initialized",
        )

    async with pool.acquire() as conn:
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


__all__ = [
    "get_drift_driver_regimes",
    "get_tier_usage_time_series",
]
