"""
Repository for regime duration statistics.

Provides CRUD operations and backoff queries for duration stats.
Used for regime persistence estimation in v1.5 live intelligence.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog

from app.services.kb.regime_key import extract_marginal_keys

logger = structlog.get_logger(__name__)


@dataclass
class RemainingEstimate:
    """Estimated remaining duration for a regime."""

    expected_remaining_bars: int
    remaining_iqr_bars: list[int]  # [p25_remaining, p75_remaining]


@dataclass
class DurationStats:
    """
    Duration statistics for a regime key.

    Stores historical duration distributions for regime persistence estimation.
    Keyed by (symbol, timeframe, regime_key) - NOT workspace scoped,
    as duration is market behavior.
    """

    symbol: str
    timeframe: str
    regime_key: str
    n_segments: int
    median_duration_bars: int
    p25_duration_bars: int
    p75_duration_bars: int
    updated_at: Optional[datetime] = None
    baseline: str = "composite_symbol"  # composite_symbol | marginal | global_timeframe

    def compute_expected_remaining(self, regime_age_bars: int) -> RemainingEstimate:
        """
        Compute expected remaining duration.

        Args:
            regime_age_bars: Current regime age in bars

        Returns:
            RemainingEstimate with expected remaining and IQR
        """
        expected = max(0, self.median_duration_bars - regime_age_bars)
        p25_remaining = max(0, self.p25_duration_bars - regime_age_bars)
        p75_remaining = max(0, self.p75_duration_bars - regime_age_bars)

        return RemainingEstimate(
            expected_remaining_bars=expected,
            remaining_iqr_bars=[p25_remaining, p75_remaining],
        )

    @property
    def duration_iqr_bars(self) -> list[int]:
        """Get duration IQR as list."""
        return [self.p25_duration_bars, self.p75_duration_bars]


class DurationStatsRepository:
    """
    Repository for regime_duration_stats table.

    Provides:
    - get_stats: Direct lookup by (symbol, timeframe, regime_key)
    - get_stats_with_backoff: Tries composite first, then marginals, then global
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
        symbol: str,
        timeframe: str,
        regime_key: str,
    ) -> Optional[DurationStats]:
        """
        Get duration stats for exact regime key.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            timeframe: Timeframe string (e.g., "5m", "1h")
            regime_key: Canonical regime key (e.g., "trend=uptrend|vol=high_vol")

        Returns:
            DurationStats or None if not found
        """
        query = """
            SELECT
                symbol, timeframe, regime_key,
                n_segments, median_duration_bars,
                p25_duration_bars, p75_duration_bars,
                updated_at
            FROM regime_duration_stats
            WHERE symbol = $1
              AND timeframe = $2
              AND regime_key = $3
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, symbol, timeframe, regime_key)

        if row is None:
            return None

        return self._row_to_stats(row, baseline="composite_symbol")

    async def get_stats_with_backoff(
        self,
        symbol: str,
        timeframe: str,
        regime_key: str,
        min_segments: int = 10,
    ) -> Optional[DurationStats]:
        """
        Get duration stats with backoff chain.

        Backoff order:
        1. Exact composite for symbol+timeframe
        2. Marginal keys for symbol+timeframe
        3. Global timeframe (aggregate across symbols)
        4. None

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            regime_key: Canonical composite regime key
            min_segments: Minimum segment count to accept (default: 10)

        Returns:
            DurationStats with baseline indicator, or None
        """
        # Try exact composite first
        stats = await self.get_stats(symbol, timeframe, regime_key)
        if stats is not None and stats.n_segments >= min_segments:
            logger.debug(
                "duration_stats_hit_composite",
                symbol=symbol,
                timeframe=timeframe,
                regime_key=regime_key,
                n_segments=stats.n_segments,
            )
            return stats

        # Try marginals for this symbol
        marginal_keys = extract_marginal_keys(regime_key)
        for marginal in marginal_keys:
            ms = await self.get_stats(symbol, timeframe, marginal)
            if ms is not None and ms.n_segments >= min_segments:
                ms.baseline = "marginal"
                logger.debug(
                    "duration_stats_backoff_to_marginal",
                    symbol=symbol,
                    timeframe=timeframe,
                    regime_key=regime_key,
                    marginal_key=marginal,
                    n_segments=ms.n_segments,
                )
                return ms

        # Try global timeframe (aggregate across symbols)
        global_stats = await self._get_global_timeframe_stats(timeframe, regime_key)
        if global_stats is not None and global_stats.n_segments >= min_segments:
            global_stats.baseline = "global_timeframe"
            logger.debug(
                "duration_stats_backoff_to_global",
                symbol=symbol,
                timeframe=timeframe,
                regime_key=regime_key,
                n_segments=global_stats.n_segments,
            )
            return global_stats

        logger.debug(
            "duration_stats_miss",
            symbol=symbol,
            timeframe=timeframe,
            regime_key=regime_key,
        )
        return None

    async def _get_global_timeframe_stats(
        self,
        timeframe: str,
        regime_key: str,
    ) -> Optional[DurationStats]:
        """
        Get aggregated stats across all symbols for a timeframe.

        Uses percentile aggregation to combine duration distributions
        from multiple symbols.

        Args:
            timeframe: Timeframe string
            regime_key: Canonical regime key

        Returns:
            DurationStats with baseline="global_timeframe", or None
        """
        query = """
            SELECT
                'global' as symbol,
                timeframe,
                regime_key,
                SUM(n_segments) as n_segments,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY median_duration_bars) as median_duration_bars,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY median_duration_bars) as p25_duration_bars,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY median_duration_bars) as p75_duration_bars,
                MAX(updated_at) as updated_at
            FROM regime_duration_stats
            WHERE timeframe = $1
              AND regime_key = $2
            GROUP BY timeframe, regime_key
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, timeframe, regime_key)

        if row is None or row["n_segments"] is None:
            return None

        return self._row_to_stats(row, baseline="global_timeframe")

    async def upsert_stats(self, stats: DurationStats) -> None:
        """
        Insert or update duration stats.

        Uses PostgreSQL UPSERT (INSERT ... ON CONFLICT DO UPDATE).

        Args:
            stats: DurationStats to upsert
        """
        query = """
            INSERT INTO regime_duration_stats (
                symbol, timeframe, regime_key,
                n_segments, median_duration_bars,
                p25_duration_bars, p75_duration_bars,
                updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, now())
            ON CONFLICT (symbol, timeframe, regime_key)
            DO UPDATE SET
                n_segments = EXCLUDED.n_segments,
                median_duration_bars = EXCLUDED.median_duration_bars,
                p25_duration_bars = EXCLUDED.p25_duration_bars,
                p75_duration_bars = EXCLUDED.p75_duration_bars,
                updated_at = now()
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                stats.symbol,
                stats.timeframe,
                stats.regime_key,
                stats.n_segments,
                stats.median_duration_bars,
                stats.p25_duration_bars,
                stats.p75_duration_bars,
            )

        logger.info(
            "duration_stats_upserted",
            symbol=stats.symbol,
            timeframe=stats.timeframe,
            regime_key=stats.regime_key,
            n_segments=stats.n_segments,
            baseline=stats.baseline,
        )

    def _row_to_stats(
        self, row: dict, baseline: str = "composite_symbol"
    ) -> DurationStats:
        """
        Convert database row to DurationStats.

        Args:
            row: Database row dict
            baseline: Baseline type indicator

        Returns:
            DurationStats instance
        """
        return DurationStats(
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            regime_key=row["regime_key"],
            n_segments=int(row["n_segments"]),
            median_duration_bars=int(row["median_duration_bars"]),
            p25_duration_bars=int(row["p25_duration_bars"]),
            p75_duration_bars=int(row["p75_duration_bars"]),
            updated_at=row.get("updated_at"),
            baseline=baseline,
        )
