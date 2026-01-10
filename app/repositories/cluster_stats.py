"""
Repository for regime cluster statistics.

Provides CRUD operations and backoff queries for cluster stats.
Used for distance z-score scaling in v1.5 live intelligence.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog

from app.services.kb.regime_key import extract_marginal_keys

logger = structlog.get_logger(__name__)


@dataclass
class ClusterStats:
    """
    Cluster statistics for a regime key.

    Stores feature centroids and variances for standardized distance computation.
    """

    strategy_entity_id: UUID
    timeframe: str
    regime_key: str
    regime_dims: dict
    n: int
    feature_mean: dict[str, float]
    feature_var: dict[str, float]
    feature_min: Optional[dict[str, float]] = None
    feature_max: Optional[dict[str, float]] = None
    feature_schema_version: int = 1
    updated_at: Optional[datetime] = None
    baseline: str = "composite"  # composite | marginal | neighbors_only


class ClusterStatsRepository:
    """
    Repository for regime_cluster_stats table.

    Provides:
    - get_stats: Direct lookup by (strategy_entity_id, timeframe, regime_key)
    - get_stats_with_backoff: Tries composite first, then marginals
    - upsert_stats: Insert or update stats
    """

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def get_stats(
        self,
        strategy_entity_id: UUID,
        timeframe: str,
        regime_key: str,
    ) -> Optional[ClusterStats]:
        """
        Get cluster stats for exact regime key.

        Args:
            strategy_entity_id: Strategy entity ID
            timeframe: Timeframe string (e.g., "5m", "1h")
            regime_key: Canonical regime key (e.g., "trend=uptrend|vol=high_vol")

        Returns:
            ClusterStats or None if not found
        """
        query = """
            SELECT
                strategy_entity_id, timeframe, regime_key, regime_dims,
                n, feature_schema_version, feature_mean, feature_var,
                feature_min, feature_max, updated_at
            FROM regime_cluster_stats
            WHERE strategy_entity_id = $1
              AND timeframe = $2
              AND regime_key = $3
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, strategy_entity_id, timeframe, regime_key)

        if row is None:
            return None

        return self._row_to_stats(row, baseline="composite")

    async def get_stats_with_backoff(
        self,
        strategy_entity_id: UUID,
        timeframe: str,
        regime_key: str,
        min_n: int = 20,
    ) -> Optional[ClusterStats]:
        """
        Get cluster stats with backoff chain.

        Backoff order:
        1. Exact composite key (e.g., "trend=uptrend|vol=high_vol")
        2. Marginal keys (e.g., "trend=uptrend", "vol=high_vol")
           - Combines marginals by averaging means and taking max variance
        3. None

        Args:
            strategy_entity_id: Strategy entity ID
            timeframe: Timeframe string
            regime_key: Canonical composite regime key
            min_n: Minimum sample count to accept (default: 20)

        Returns:
            ClusterStats with baseline indicator, or None
        """
        # Try exact composite first
        stats = await self.get_stats(strategy_entity_id, timeframe, regime_key)
        if stats is not None and stats.n >= min_n:
            return stats

        # Try marginals
        marginal_keys = extract_marginal_keys(regime_key)
        marginal_stats = []

        for marginal in marginal_keys:
            ms = await self.get_stats(strategy_entity_id, timeframe, marginal)
            if ms is not None:
                marginal_stats.append(ms)

        if marginal_stats:
            # Combine marginals: take max variance per feature for safety
            combined = self._combine_marginals(marginal_stats)
            combined.baseline = "marginal"
            logger.debug(
                "cluster_stats_backoff_to_marginal",
                strategy_entity_id=str(strategy_entity_id),
                timeframe=timeframe,
                regime_key=regime_key,
                marginal_count=len(marginal_stats),
                combined_n=combined.n,
            )
            return combined

        return None

    async def upsert_stats(self, stats: ClusterStats) -> None:
        """
        Insert or update cluster stats.

        Uses PostgreSQL UPSERT (INSERT ... ON CONFLICT DO UPDATE).

        Args:
            stats: ClusterStats to upsert
        """
        query = """
            INSERT INTO regime_cluster_stats (
                strategy_entity_id, timeframe, regime_key, regime_dims,
                n, feature_schema_version, feature_mean, feature_var,
                feature_min, feature_max, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, now())
            ON CONFLICT (strategy_entity_id, timeframe, regime_key)
            DO UPDATE SET
                regime_dims = EXCLUDED.regime_dims,
                n = EXCLUDED.n,
                feature_schema_version = EXCLUDED.feature_schema_version,
                feature_mean = EXCLUDED.feature_mean,
                feature_var = EXCLUDED.feature_var,
                feature_min = EXCLUDED.feature_min,
                feature_max = EXCLUDED.feature_max,
                updated_at = now()
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                stats.strategy_entity_id,
                stats.timeframe,
                stats.regime_key,
                json.dumps(stats.regime_dims),
                stats.n,
                stats.feature_schema_version,
                json.dumps(stats.feature_mean),
                json.dumps(stats.feature_var),
                json.dumps(stats.feature_min) if stats.feature_min else None,
                json.dumps(stats.feature_max) if stats.feature_max else None,
            )

        logger.info(
            "cluster_stats_upserted",
            strategy_entity_id=str(stats.strategy_entity_id),
            timeframe=stats.timeframe,
            regime_key=stats.regime_key,
            n=stats.n,
            baseline=stats.baseline,
        )

    def _row_to_stats(self, row: dict, baseline: str = "composite") -> ClusterStats:
        """
        Convert database row to ClusterStats.

        Handles JSON deserialization for dict fields.

        Args:
            row: Database row dict
            baseline: Baseline type indicator

        Returns:
            ClusterStats instance
        """
        return ClusterStats(
            strategy_entity_id=row["strategy_entity_id"],
            timeframe=row["timeframe"],
            regime_key=row["regime_key"],
            regime_dims=(
                row["regime_dims"]
                if isinstance(row["regime_dims"], dict)
                else json.loads(row["regime_dims"])
            ),
            n=row["n"],
            feature_schema_version=row.get("feature_schema_version", 1),
            feature_mean=(
                row["feature_mean"]
                if isinstance(row["feature_mean"], dict)
                else json.loads(row["feature_mean"])
            ),
            feature_var=(
                row["feature_var"]
                if isinstance(row["feature_var"], dict)
                else json.loads(row["feature_var"])
            ),
            feature_min=row.get("feature_min"),
            feature_max=row.get("feature_max"),
            updated_at=row.get("updated_at"),
            baseline=baseline,
        )

    def _combine_marginals(self, marginals: list[ClusterStats]) -> ClusterStats:
        """
        Combine marginal stats conservatively.

        Takes max variance per feature to avoid underestimating spread.
        Averages means across marginals.
        Sums n for total sample count.

        Args:
            marginals: List of marginal ClusterStats

        Returns:
            Combined ClusterStats with baseline="marginal"
        """
        if len(marginals) == 1:
            result = ClusterStats(
                strategy_entity_id=marginals[0].strategy_entity_id,
                timeframe=marginals[0].timeframe,
                regime_key=marginals[0].regime_key,
                regime_dims=marginals[0].regime_dims,
                n=marginals[0].n,
                feature_mean=marginals[0].feature_mean.copy(),
                feature_var=marginals[0].feature_var.copy(),
                baseline="marginal",
            )
            return result

        # Collect all features across marginals
        all_features = set()
        for m in marginals:
            all_features.update(m.feature_mean.keys())

        combined_mean = {}
        combined_var = {}

        for feat in all_features:
            means = [
                m.feature_mean.get(feat) for m in marginals if feat in m.feature_mean
            ]
            vars_ = [
                m.feature_var.get(feat) for m in marginals if feat in m.feature_var
            ]

            if means:
                combined_mean[feat] = sum(means) / len(means)
            if vars_:
                combined_var[feat] = max(vars_)  # Conservative: max variance

        total_n = sum(m.n for m in marginals)

        return ClusterStats(
            strategy_entity_id=marginals[0].strategy_entity_id,
            timeframe=marginals[0].timeframe,
            regime_key=marginals[0].regime_key,
            regime_dims=marginals[0].regime_dims,
            n=total_n,
            feature_mean=combined_mean,
            feature_var=combined_var,
            baseline="marginal",
        )
