"""
Cluster stats backfill job.

Aggregates existing ingested trials into cluster stats by
(strategy_entity_id, timeframe, regime_key) for use in v1.5 distance z-score
computation.

Uses Welford's online algorithm for numerically stable variance computation.
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.repositories.cluster_stats import ClusterStats, ClusterStatsRepository

logger = structlog.get_logger(__name__)

# Default batch size for Qdrant scroll pagination
DEFAULT_SCROLL_BATCH_SIZE = 1000


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BackfillResult:
    """Result of cluster stats backfill job."""

    trials_processed: int = 0
    trials_skipped: int = 0  # Skipped due to missing regime_features
    stats_written: int = 0
    stats_would_write: int = 0  # For dry run
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Welford's Online Algorithm
# =============================================================================


class WelfordAccumulator:
    """
    Welford's online algorithm for computing mean and variance in single pass.

    Numerically stable even for large values. Maintains running statistics
    that can be updated incrementally.

    Reference: Welford, B. P. (1962). "Note on a method for calculating
    corrected sums of squares and products".
    """

    def __init__(self):
        """Initialize empty accumulator."""
        self.n: int = 0
        self._mean: dict[str, float] = {}
        self._m2: dict[str, float] = {}  # Sum of squared differences from mean
        self._min: dict[str, float] = {}
        self._max: dict[str, float] = {}

    def update(self, features: dict[str, float]) -> None:
        """
        Update accumulator with new sample.

        Args:
            features: Dict of feature_name -> value
        """
        self.n += 1

        for key, value in features.items():
            if not isinstance(value, (int, float)):
                continue

            if key not in self._mean:
                # First value for this feature
                self._mean[key] = value
                self._m2[key] = 0.0
                self._min[key] = value
                self._max[key] = value
            else:
                # Welford's update
                delta = value - self._mean[key]
                self._mean[key] += delta / self.n
                delta2 = value - self._mean[key]
                self._m2[key] += delta * delta2

                # Min/max tracking
                if value < self._min[key]:
                    self._min[key] = value
                if value > self._max[key]:
                    self._max[key] = value

    @property
    def mean(self) -> dict[str, float]:
        """Get current mean for all features."""
        return self._mean.copy()

    @property
    def variance(self) -> dict[str, float]:
        """
        Get sample variance (Bessel-corrected) for all features.

        Returns zero for n < 2.
        """
        if self.n < 2:
            return {k: 0.0 for k in self._mean}

        return {k: self._m2[k] / (self.n - 1) for k in self._m2}

    @property
    def min(self) -> dict[str, float]:
        """Get minimum values for all features."""
        return self._min.copy()

    @property
    def max(self) -> dict[str, float]:
        """Get maximum values for all features."""
        return self._max.copy()


# =============================================================================
# Aggregation Helper
# =============================================================================


def aggregate_trial_features(
    trials: list[dict],
) -> dict[tuple[str, str, str], ClusterStats]:
    """
    Aggregate trial features by (strategy_entity_id, timeframe, regime_key).

    Args:
        trials: List of trial dicts with 'payload' containing trial metadata

    Returns:
        Dict mapping (strategy_entity_id, timeframe, regime_key) -> ClusterStats
    """
    if not trials:
        return {}

    # Accumulators keyed by (strategy_entity_id, timeframe, regime_key)
    accumulators: dict[tuple[str, str, str], WelfordAccumulator] = {}
    regime_dims_cache: dict[tuple[str, str, str], dict] = {}

    for trial in trials:
        payload = trial.get("payload", {})

        # Extract required fields
        strategy_entity_id = payload.get("strategy_entity_id")
        timeframe = payload.get("timeframe")
        regime_key = payload.get("regime_key")
        regime_dims = payload.get("regime_dims", {})
        regime_features = payload.get("regime_features")

        # Skip if missing required fields
        if not all([strategy_entity_id, timeframe, regime_key]):
            continue

        # Skip if no regime_features or empty
        if not regime_features:
            continue

        key = (str(strategy_entity_id), timeframe, regime_key)

        if key not in accumulators:
            accumulators[key] = WelfordAccumulator()
            regime_dims_cache[key] = regime_dims

        accumulators[key].update(regime_features)

    # Convert accumulators to ClusterStats
    result = {}
    for key, acc in accumulators.items():
        if acc.n == 0:
            continue

        strategy_entity_id, timeframe, regime_key = key

        stats = ClusterStats(
            strategy_entity_id=UUID(strategy_entity_id),
            timeframe=timeframe,
            regime_key=regime_key,
            regime_dims=regime_dims_cache[key],
            n=acc.n,
            feature_mean=acc.mean,
            feature_var=acc.variance,
            feature_min=acc.min if acc.min else None,
            feature_max=acc.max if acc.max else None,
        )
        result[key] = stats

    return result


# =============================================================================
# Main Backfill Job
# =============================================================================


async def run_cluster_stats_backfill(
    qdrant_client: AsyncQdrantClient,
    db_pool,
    collection_name: str,
    strategy_entity_id: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = DEFAULT_SCROLL_BATCH_SIZE,
) -> BackfillResult:
    """
    Backfill cluster stats from existing KB trials.

    Reads all trials from Qdrant, aggregates regime features by
    (strategy_entity_id, timeframe, regime_key), and upserts stats
    to regime_cluster_stats table.

    Args:
        qdrant_client: Qdrant async client
        db_pool: asyncpg connection pool
        collection_name: Qdrant collection name (e.g., "kb_trials")
        strategy_entity_id: Optional filter for specific strategy
        dry_run: If True, don't write to DB (just compute stats)
        batch_size: Qdrant scroll batch size

    Returns:
        BackfillResult with counts and errors
    """
    result = BackfillResult(dry_run=dry_run)

    logger.info(
        "cluster_stats_backfill_started",
        collection=collection_name,
        strategy_entity_id=strategy_entity_id,
        dry_run=dry_run,
        batch_size=batch_size,
    )

    # Build optional filter for strategy
    scroll_filter = None
    if strategy_entity_id:
        scroll_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="strategy_entity_id",
                    match=qmodels.MatchValue(value=strategy_entity_id),
                )
            ]
        )

    # Accumulate all trials across pages
    all_trials: list[dict] = []
    offset = None
    page_count = 0

    try:
        while True:
            page_count += 1

            # Scroll through Qdrant
            points, next_offset = await qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                break

            # Convert Qdrant points to trial dicts
            for point in points:
                trial = {
                    "id": point.id,
                    "payload": point.payload,
                }
                all_trials.append(trial)

                # Count trials with missing regime_features
                if not point.payload.get("regime_features"):
                    result.trials_skipped += 1

            result.trials_processed += len(points)

            logger.debug(
                "cluster_stats_backfill_page",
                page=page_count,
                points_in_page=len(points),
                total_trials=result.trials_processed,
            )

            # Check for more pages
            if next_offset is None:
                break
            offset = next_offset

    except Exception as e:
        error_msg = f"Failed to read trials from Qdrant: {str(e)}"
        logger.error("cluster_stats_backfill_qdrant_error", error=str(e))
        result.errors.append(error_msg)
        return result

    # Aggregate features
    aggregated_stats = aggregate_trial_features(all_trials)

    logger.info(
        "cluster_stats_backfill_aggregated",
        total_trials=result.trials_processed,
        trials_skipped=result.trials_skipped,
        unique_keys=len(aggregated_stats),
    )

    # Write stats to database
    if dry_run:
        result.stats_would_write = len(aggregated_stats)
        logger.info(
            "cluster_stats_backfill_dry_run_complete",
            would_write=result.stats_would_write,
        )
    else:
        repo = ClusterStatsRepository(db_pool)

        for key, stats in aggregated_stats.items():
            try:
                await repo.upsert_stats(stats)
                result.stats_written += 1
            except Exception as e:
                error_msg = f"Failed to upsert stats for {key}: {str(e)}"
                logger.error(
                    "cluster_stats_backfill_upsert_error",
                    key=key,
                    error=str(e),
                )
                result.errors.append(error_msg)

        logger.info(
            "cluster_stats_backfill_complete",
            stats_written=result.stats_written,
            errors=len(result.errors),
        )

    return result
