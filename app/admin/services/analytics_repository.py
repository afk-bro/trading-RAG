"""Analytics repository for regime coverage, tier usage, uplift, and drift queries.

Extracts all SQL queries from analytics.py into a repository class.
Follows the existing repository pattern (see app/repositories/alerts.py).
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Drift trend thresholds (percentage points)
DRIFT_TREND_THRESHOLD_PP = 10  # >10pp change = increasing/decreasing
CONFIDENCE_TREND_THRESHOLD = 0.05  # >5% change = improving/degrading
MIN_BUCKETS_FOR_TREND = 4  # Need at least 4 data points for trend analysis

# Default backlog suggestions
DEFAULT_BACKLOG_TIMEFRAMES = ["5m", "15m"]
DEFAULT_BACKLOG_SYMBOLS = ["BTC-USDT", "ETH-USDT"]

# Priority thresholds for backlog items
PRIORITY_HIGH_THRESHOLD = 3  # missing >= 3 samples
PRIORITY_MEDIUM_THRESHOLD = 2  # missing >= 2 samples


# =============================================================================
# Query Builder Helper
# =============================================================================


@dataclass
class QueryBuilder:
    """Helper for building parameterized SQL WHERE clauses.

    Reduces boilerplate for the repeated conditions/params/param_idx pattern.
    """

    conditions: list[str] = field(default_factory=list)
    params: list[Any] = field(default_factory=list)
    _param_idx: int = 1

    def add(self, condition: str, value: Any) -> "QueryBuilder":
        """Add a condition with a parameter placeholder."""
        self.conditions.append(condition.replace("?", f"${self._param_idx}"))
        self.params.append(value)
        self._param_idx += 1
        return self

    def add_if(
        self, condition: str, value: Optional[Any], predicate: bool = True
    ) -> "QueryBuilder":
        """Add a condition only if predicate is True and value is not None."""
        if predicate and value is not None:
            self.add(condition, value)
        return self

    def add_in(self, column: str, values: list[Any]) -> "QueryBuilder":
        """Add an IN clause with multiple parameters."""
        if not values:
            return self
        placeholders = ", ".join(f"${self._param_idx + i}" for i in range(len(values)))
        self.conditions.append(f"{column} IN ({placeholders})")
        self.params.extend(values)
        self._param_idx += len(values)
        return self

    @property
    def next_param(self) -> str:
        """Get the next parameter placeholder (e.g., '$3')."""
        return f"${self._param_idx}"

    def add_param(self, value: Any) -> str:
        """Add a parameter and return its placeholder."""
        placeholder = self.next_param
        self.params.append(value)
        self._param_idx += 1
        return placeholder

    @property
    def where_clause(self) -> str:
        """Build the WHERE clause from conditions."""
        if not self.conditions:
            return "TRUE"
        return " AND ".join(self.conditions)


# =============================================================================
# Data Classes for Query Results
# =============================================================================


@dataclass
class RegimeCoverageRow:
    """Raw row from regime coverage query."""

    regime_key: str
    trend_tag: Optional[str]
    vol_tag: Optional[str]
    n_tunes: int
    n_runs: int
    avg_best_oos: Optional[float]
    p50_best_oos: Optional[float]
    p90_best_oos: Optional[float]


@dataclass
class TierUsageRow:
    """Raw row from tier usage query."""

    tier_used: str
    count: int
    avg_confidence: float
    avg_candidate_count: float


@dataclass
class UpliftGroupRow:
    """Raw row from uplift group query."""

    group_key: str
    n_recommendations: int
    avg_selected_score: float


@dataclass
class DriftBucketRow:
    """Raw row from drift bucket query."""

    bucket_start: datetime
    total: int
    non_exact_count: int
    unique_regimes: int
    avg_confidence: float
    top_regime_count: int


@dataclass
class DriftDriverRow:
    """Raw row from drift drivers query."""

    regime_key: str
    trend_tag: Optional[str]
    vol_tag: Optional[str]
    total_requests: int
    non_exact_count: int
    exact_count: int
    partial_trend_count: int
    partial_vol_count: int
    distance_count: int
    global_count: int
    current_tunes: int


@dataclass
class RegimeBacklogRow:
    """Raw row from regime backlog query."""

    regime_key: str
    trend_tag: Optional[str]
    vol_tag: Optional[str]
    n_tunes: int


# =============================================================================
# Repository Class
# =============================================================================


class AnalyticsRepository:
    """Repository for analytics queries.

    All methods accept the pool explicitly via __init__ (no globals).
    """

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    # -------------------------------------------------------------------------
    # Regime Coverage
    # -------------------------------------------------------------------------

    async def get_regime_coverage(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
        since: Optional[datetime] = None,
    ) -> list[RegimeCoverageRow]:
        """Get regime coverage statistics.

        Args:
            workspace_id: Workspace UUID
            strategy_entity_id: Optional strategy filter
            since: Optional filter for tunes created after this time

        Returns:
            List of coverage rows ordered by n_tunes desc
        """
        qb = QueryBuilder()
        qb.add("t.workspace_id = ?", workspace_id)
        qb.conditions.append("t.regime_key IS NOT NULL")
        qb.add_if("t.strategy_entity_id = ?", strategy_entity_id)
        qb.add_if("t.created_at >= ?", since)

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
            WHERE {qb.where_clause}
              AND t.status = 'completed'
              AND t.best_oos_score IS NOT NULL
            GROUP BY t.regime_key, t.trend_tag, t.vol_tag
            ORDER BY n_tunes DESC, avg_best_oos DESC NULLS LAST
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        return [
            RegimeCoverageRow(
                regime_key=r["regime_key"],
                trend_tag=r["trend_tag"],
                vol_tag=r["vol_tag"],
                n_tunes=r["n_tunes"],
                n_runs=r["n_runs"],
                avg_best_oos=float(r["avg_best_oos"]) if r["avg_best_oos"] else None,
                p50_best_oos=float(r["p50_best_oos"]) if r["p50_best_oos"] else None,
                p90_best_oos=float(r["p90_best_oos"]) if r["p90_best_oos"] else None,
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Tier Usage
    # -------------------------------------------------------------------------

    async def get_tier_usage_totals(
        self,
        workspace_id: UUID,
        since: datetime,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
    ) -> list[TierUsageRow]:
        """Get tier usage distribution totals.

        Args:
            workspace_id: Workspace UUID
            since: Start of analysis period
            strategy_entity_id: Optional strategy filter
            regime_key: Optional regime filter

        Returns:
            List of tier usage rows ordered by count desc
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", since)
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)
        qb.add_if("query_regime_key = ?", regime_key)

        query = f"""
            SELECT
                tier_used,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(candidate_count) as avg_candidate_count
            FROM recommend_events
            WHERE {qb.where_clause}
            GROUP BY tier_used
            ORDER BY count DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        return [
            TierUsageRow(
                tier_used=r["tier_used"],
                count=r["count"],
                avg_confidence=float(r["avg_confidence"]),
                avg_candidate_count=float(r["avg_candidate_count"]),
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Uplift
    # -------------------------------------------------------------------------

    async def get_baseline_score(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
    ) -> float:
        """Get global baseline (top OOS score across all regimes).

        Args:
            workspace_id: Workspace UUID
            strategy_entity_id: Optional strategy filter

        Returns:
            Maximum best_oos_score or 0.0 if none
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.conditions.append("status = 'completed'")
        qb.conditions.append("best_oos_score IS NOT NULL")
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)

        query = f"""
            SELECT MAX(best_oos_score) as baseline
            FROM backtest_tunes
            WHERE {qb.where_clause}
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *qb.params)

        return float(row["baseline"]) if row and row["baseline"] else 0.0

    async def get_uplift_by_regime(
        self,
        workspace_id: UUID,
        since: datetime,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
    ) -> list[UpliftGroupRow]:
        """Get uplift stats grouped by regime.

        Args:
            workspace_id: Workspace UUID
            since: Start of analysis period
            strategy_entity_id: Optional strategy filter
            regime_key: Optional regime filter

        Returns:
            List of uplift rows by regime
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", since)
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)
        qb.add_if("query_regime_key = ?", regime_key)

        query = f"""
            SELECT
                query_regime_key as group_key,
                COUNT(*) as n_recommendations,
                AVG(top_candidate_score) as avg_selected_score
            FROM recommend_events
            WHERE {qb.where_clause}
              AND query_regime_key IS NOT NULL
              AND top_candidate_score IS NOT NULL
            GROUP BY query_regime_key
            ORDER BY n_recommendations DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        return [
            UpliftGroupRow(
                group_key=r["group_key"],
                n_recommendations=r["n_recommendations"],
                avg_selected_score=float(r["avg_selected_score"]),
            )
            for r in rows
        ]

    async def get_uplift_by_tier(
        self,
        workspace_id: UUID,
        since: datetime,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
    ) -> list[UpliftGroupRow]:
        """Get uplift stats grouped by tier.

        Args:
            workspace_id: Workspace UUID
            since: Start of analysis period
            strategy_entity_id: Optional strategy filter
            regime_key: Optional regime filter

        Returns:
            List of uplift rows by tier
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", since)
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)
        qb.add_if("query_regime_key = ?", regime_key)

        query = f"""
            SELECT
                tier_used as group_key,
                COUNT(*) as n_recommendations,
                AVG(top_candidate_score) as avg_selected_score
            FROM recommend_events
            WHERE {qb.where_clause}
              AND top_candidate_score IS NOT NULL
            GROUP BY tier_used
            ORDER BY n_recommendations DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        return [
            UpliftGroupRow(
                group_key=r["group_key"],
                n_recommendations=r["n_recommendations"],
                avg_selected_score=float(r["avg_selected_score"]),
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Regime Backlog
    # -------------------------------------------------------------------------

    async def get_regime_backlog(
        self,
        workspace_id: UUID,
        min_samples: int,
        max_items: int,
        strategy_entity_id: Optional[UUID] = None,
        only_regimes: Optional[list[str]] = None,
    ) -> list[RegimeBacklogRow]:
        """Get regimes below min_samples threshold for backlog generation.

        Args:
            workspace_id: Workspace UUID
            min_samples: Threshold for coverage
            max_items: Maximum items to return
            strategy_entity_id: Optional strategy filter
            only_regimes: Optional list of regime keys to filter

        Returns:
            List of backlog rows ordered by n_tunes asc
        """
        qb = QueryBuilder()
        qb.add("t.workspace_id = ?", workspace_id)
        qb.conditions.append("t.regime_key IS NOT NULL")
        qb.add_if("t.strategy_entity_id = ?", strategy_entity_id)
        if only_regimes:
            qb.add_in("t.regime_key", only_regimes)

        min_samples_param = qb.add_param(min_samples)
        max_items_param = qb.add_param(max_items)

        query = f"""
            SELECT
                t.regime_key,
                t.trend_tag,
                t.vol_tag,
                COUNT(DISTINCT t.id) as n_tunes
            FROM backtest_tunes t
            WHERE {qb.where_clause}
              AND t.status = 'completed'
            GROUP BY t.regime_key, t.trend_tag, t.vol_tag
            HAVING COUNT(DISTINCT t.id) < {min_samples_param}
            ORDER BY COUNT(DISTINCT t.id) ASC
            LIMIT {max_items_param}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        return [
            RegimeBacklogRow(
                regime_key=r["regime_key"],
                trend_tag=r["trend_tag"],
                vol_tag=r["vol_tag"],
                n_tunes=r["n_tunes"],
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # Regime Drift
    # -------------------------------------------------------------------------

    async def get_drift_buckets(
        self,
        workspace_id: UUID,
        since: datetime,
        bucket: str,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
    ) -> list[DriftBucketRow]:
        """Get drift metrics per time bucket.

        Args:
            workspace_id: Workspace UUID
            since: Start of analysis period
            bucket: Time bucket ('day' or 'week')
            strategy_entity_id: Optional strategy filter
            regime_key: Optional regime filter

        Returns:
            List of drift bucket rows ordered chronologically
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", since)
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)
        qb.add_if("query_regime_key = ?", regime_key)

        trunc_unit = "day" if bucket == "day" else "week"

        query = f"""
            WITH bucket_stats AS (
                SELECT
                    date_trunc('{trunc_unit}', created_at) as bucket_start,
                    COUNT(*) as total,
                    SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact_count,
                    COUNT(DISTINCT query_regime_key) as unique_regimes,
                    AVG(COALESCE(confidence, 0.5)) as avg_confidence
                FROM recommend_events
                WHERE {qb.where_clause}
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
                WHERE {qb.where_clause}
                  AND query_regime_key IS NOT NULL
                GROUP BY bucket_start, query_regime_key
            )
            SELECT
                bs.bucket_start,
                bs.total,
                bs.non_exact_count,
                bs.unique_regimes,
                bs.avg_confidence,
                COALESCE(tr.regime_count, 0) as top_regime_count
            FROM bucket_stats bs
            LEFT JOIN top_regime_per_bucket tr
                ON tr.bucket_start = bs.bucket_start AND tr.rn = 1
            ORDER BY bs.bucket_start ASC
        """

        async with self.pool.acquire() as conn:
            # Note: params used twice in CTEs, need to double them
            all_params = qb.params + qb.params
            rows = await conn.fetch(query, *all_params)

        return [
            DriftBucketRow(
                bucket_start=r["bucket_start"],
                total=r["total"],
                non_exact_count=r["non_exact_count"],
                unique_regimes=r["unique_regimes"],
                avg_confidence=(
                    float(r["avg_confidence"]) if r["avg_confidence"] else 0.5
                ),
                top_regime_count=r["top_regime_count"],
            )
            for r in rows
        ]

    async def get_overall_unique_regimes(
        self,
        workspace_id: UUID,
        since: datetime,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
    ) -> int:
        """Get total unique regimes in period.

        Args:
            workspace_id: Workspace UUID
            since: Start of analysis period
            strategy_entity_id: Optional strategy filter
            regime_key: Optional regime filter

        Returns:
            Count of unique regimes
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", since)
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)
        qb.add_if("query_regime_key = ?", regime_key)

        query = f"""
            SELECT COUNT(DISTINCT query_regime_key) as unique_regimes
            FROM recommend_events
            WHERE {qb.where_clause}
              AND query_regime_key IS NOT NULL
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *qb.params)

        return row["unique_regimes"] if row else 0

    # -------------------------------------------------------------------------
    # Drift Drivers
    # -------------------------------------------------------------------------

    async def get_drift_drivers(
        self,
        workspace_id: UUID,
        since: datetime,
        limit: int,
        strategy_entity_id: Optional[UUID] = None,
    ) -> list[DriftDriverRow]:
        """Get top regimes driving non-exact fallback.

        Args:
            workspace_id: Workspace UUID
            since: Start of analysis period
            limit: Maximum drivers to return
            strategy_entity_id: Optional strategy filter

        Returns:
            List of drift driver rows ordered by non_exact_count desc
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", since)
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)
        limit_param = qb.add_param(limit)

        query = f"""
            WITH regime_stats AS (
                SELECT
                    query_regime_key,
                    query_trend_tag,
                    query_vol_tag,
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact_count,
                    SUM(CASE WHEN tier_used = 'exact' THEN 1 ELSE 0 END) as exact_count,
                    SUM(CASE WHEN tier_used = 'partial_trend' THEN 1 ELSE 0 END)
                        as partial_trend_count,
                    SUM(CASE WHEN tier_used = 'partial_vol' THEN 1 ELSE 0 END) as partial_vol_count,
                    SUM(CASE WHEN tier_used = 'distance' THEN 1 ELSE 0 END) as distance_count,
                    SUM(CASE WHEN tier_used = 'global_best' THEN 1 ELSE 0 END) as global_count
                FROM recommend_events
                WHERE {qb.where_clause}
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
            LIMIT {limit_param}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        return [
            DriftDriverRow(
                regime_key=r["regime_key"],
                trend_tag=r["trend_tag"],
                vol_tag=r["vol_tag"],
                total_requests=r["total_requests"],
                non_exact_count=r["non_exact_count"],
                exact_count=r["exact_count"],
                partial_trend_count=r["partial_trend_count"],
                partial_vol_count=r["partial_vol_count"],
                distance_count=r["distance_count"],
                global_count=r["global_count"],
                current_tunes=r["current_tunes"],
            )
            for r in rows
        ]

    async def get_prior_week_drift(
        self,
        workspace_id: UUID,
        prior_start: datetime,
        prior_end: datetime,
        strategy_entity_id: Optional[UUID] = None,
    ) -> dict[str, float]:
        """Get non-exact percentage by regime for prior week (WoW comparison).

        Args:
            workspace_id: Workspace UUID
            prior_start: Start of prior period
            prior_end: End of prior period
            strategy_entity_id: Optional strategy filter

        Returns:
            Dict mapping regime_key to non_exact_pct
        """
        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", prior_start)
        qb.add("created_at < ?", prior_end)
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)

        query = f"""
            SELECT
                query_regime_key,
                COUNT(*) as total,
                SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact
            FROM recommend_events
            WHERE {qb.where_clause}
              AND query_regime_key IS NOT NULL
            GROUP BY query_regime_key
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        result = {}
        for r in rows:
            total = r["total"]
            non_exact = r["non_exact"]
            pct = (non_exact / total * 100) if total > 0 else 0.0
            result[r["query_regime_key"]] = pct

        return result

    async def get_drift_driver_regime_keys(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
        limit: int = 5,
        period_days: int = 7,
    ) -> list[str]:
        """Get top drift-driving regime keys for backlog focus mode.

        This is a simpler version that just returns regime keys.

        Args:
            workspace_id: Workspace UUID
            strategy_entity_id: Optional strategy filter
            limit: Maximum regimes to return
            period_days: Analysis period in days

        Returns:
            List of regime keys sorted by non-exact count descending
        """
        since = datetime.utcnow() - timedelta(days=period_days)

        qb = QueryBuilder()
        qb.add("workspace_id = ?", workspace_id)
        qb.add("created_at >= ?", since)
        qb.conditions.append("query_regime_key IS NOT NULL")
        qb.add_if("strategy_entity_id = ?", strategy_entity_id)
        limit_param = qb.add_param(limit)

        query = f"""
            SELECT
                query_regime_key,
                SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) as non_exact_count
            FROM recommend_events
            WHERE {qb.where_clause}
            GROUP BY query_regime_key
            HAVING SUM(CASE WHEN tier_used != 'exact' THEN 1 ELSE 0 END) > 0
            ORDER BY non_exact_count DESC
            LIMIT {limit_param}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *qb.params)

        return [r["query_regime_key"] for r in rows]


__all__ = [
    # Constants
    "DRIFT_TREND_THRESHOLD_PP",
    "CONFIDENCE_TREND_THRESHOLD",
    "MIN_BUCKETS_FOR_TREND",
    "DEFAULT_BACKLOG_TIMEFRAMES",
    "DEFAULT_BACKLOG_SYMBOLS",
    "PRIORITY_HIGH_THRESHOLD",
    "PRIORITY_MEDIUM_THRESHOLD",
    # Query Builder
    "QueryBuilder",
    # Data Classes
    "RegimeCoverageRow",
    "TierUsageRow",
    "UpliftGroupRow",
    "DriftBucketRow",
    "DriftDriverRow",
    "RegimeBacklogRow",
    # Repository
    "AnalyticsRepository",
]
