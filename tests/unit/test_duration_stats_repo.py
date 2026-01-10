"""Tests for duration stats repository."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.repositories.duration_stats import (
    DurationStatsRepository,
    DurationStats,
    RemainingEstimate,
)


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


class TestDurationStatsRepository:
    """Tests for DurationStatsRepository."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_none_when_missing(self, mock_pool):
        """Returns None when no stats exist."""
        repo = DurationStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_stats_returns_duration_stats_when_found(self, mock_pool):
        """Returns DurationStats when row exists."""
        repo = DurationStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "regime_key": "trend=uptrend|vol=high_vol",
                "n_segments": 50,
                "median_duration_bars": 240,
                "p25_duration_bars": 180,
                "p75_duration_bars": 310,
                "updated_at": None,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.symbol == "BTC/USDT"
        assert result.timeframe == "5m"
        assert result.n_segments == 50
        assert result.median_duration_bars == 240
        assert result.baseline == "composite_symbol"

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_returns_composite_first(self, mock_pool):
        """Returns composite stats when available with sufficient segments."""
        repo = DurationStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "regime_key": "trend=uptrend|vol=high_vol",
                "n_segments": 50,
                "median_duration_bars": 240,
                "p25_duration_bars": 180,
                "p75_duration_bars": 310,
                "updated_at": None,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.baseline == "composite_symbol"

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_tries_marginals(self, mock_pool):
        """Backoff tries marginal keys when composite missing."""
        repo = DurationStatsRepository(mock_pool)

        call_count = 0

        async def mock_fetchrow(query, *args):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Composite and first marginal
                return None
            else:
                # Second marginal found
                return {
                    "symbol": "BTC/USDT",
                    "timeframe": "5m",
                    "regime_key": "vol=high_vol",
                    "n_segments": 25,
                    "median_duration_bars": 200,
                    "p25_duration_bars": 100,
                    "p75_duration_bars": 350,
                    "updated_at": None,
                }

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.baseline == "marginal"
        assert result.median_duration_bars == 200

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_tries_global_timeframe(self, mock_pool):
        """Backoff falls to global timeframe when marginals missing."""
        repo = DurationStatsRepository(mock_pool)

        call_count = 0

        async def mock_fetchrow(query, *args):
            nonlocal call_count
            call_count += 1
            # Composite and all marginals fail (3 calls)
            if call_count <= 3:
                return None
            else:
                # Global timeframe found
                return {
                    "symbol": "global",
                    "timeframe": "5m",
                    "regime_key": "trend=uptrend|vol=high_vol",
                    "n_segments": 100,
                    "median_duration_bars": 180,
                    "p25_duration_bars": 120,
                    "p75_duration_bars": 250,
                    "updated_at": None,
                }

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.baseline == "global_timeframe"

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_returns_none_when_all_missing(
        self, mock_pool
    ):
        """Returns None when composite, marginals, and global all missing."""
        repo = DurationStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_respects_min_segments(self, mock_pool):
        """Falls back when composite n_segments is below min_segments threshold."""
        repo = DurationStatsRepository(mock_pool)

        call_count = 0

        async def mock_fetchrow(query, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Composite found but n_segments too low
                return {
                    "symbol": "BTC/USDT",
                    "timeframe": "5m",
                    "regime_key": "trend=uptrend|vol=high_vol",
                    "n_segments": 5,  # Below default min_segments of 10
                    "median_duration_bars": 240,
                    "p25_duration_bars": 180,
                    "p75_duration_bars": 310,
                    "updated_at": None,
                }
            else:
                # Marginal with sufficient n_segments
                return {
                    "symbol": "BTC/USDT",
                    "timeframe": "5m",
                    "regime_key": "trend=uptrend",
                    "n_segments": 50,
                    "median_duration_bars": 200,
                    "p25_duration_bars": 150,
                    "p75_duration_bars": 280,
                    "updated_at": None,
                }

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            min_segments=10,
        )

        assert result is not None
        assert result.baseline == "marginal"

    @pytest.mark.asyncio
    async def test_expected_remaining_computed(self, mock_pool):
        """Expected remaining is computed from median - age."""
        repo = DurationStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "regime_key": "trend=uptrend|vol=high_vol",
                "n_segments": 50,
                "median_duration_bars": 240,
                "p25_duration_bars": 180,
                "p75_duration_bars": 310,
                "updated_at": None,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        # Compute expected remaining for age=120
        remaining = result.compute_expected_remaining(regime_age_bars=120)
        assert remaining.expected_remaining_bars == 120  # 240 - 120
        assert remaining.remaining_iqr_bars == [60, 190]  # [180-120, 310-120]

    @pytest.mark.asyncio
    async def test_upsert_stats(self, mock_pool):
        """Upsert creates or updates stats."""
        repo = DurationStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        stats = DurationStats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            n_segments=50,
            median_duration_bars=240,
            p25_duration_bars=180,
            p75_duration_bars=310,
        )

        await repo.upsert_stats(stats)

        conn.execute.assert_called_once()


class TestDurationStats:
    """Tests for DurationStats dataclass."""

    def test_duration_stats_creation(self):
        """DurationStats can be created with required fields."""
        stats = DurationStats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            n_segments=50,
            median_duration_bars=240,
            p25_duration_bars=180,
            p75_duration_bars=310,
        )

        assert stats.n_segments == 50
        assert stats.baseline == "composite_symbol"  # default

    def test_duration_stats_duration_iqr_property(self):
        """DurationStats provides duration_iqr_bars property."""
        stats = DurationStats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            n_segments=50,
            median_duration_bars=240,
            p25_duration_bars=180,
            p75_duration_bars=310,
        )

        assert stats.duration_iqr_bars == [180, 310]


class TestRemainingEstimate:
    """Tests for RemainingEstimate dataclass."""

    def test_remaining_estimate_creation(self):
        """RemainingEstimate can be created."""
        remaining = RemainingEstimate(
            expected_remaining_bars=120,
            remaining_iqr_bars=[60, 190],
        )

        assert remaining.expected_remaining_bars == 120
        assert remaining.remaining_iqr_bars == [60, 190]


class TestComputeExpectedRemaining:
    """Tests for compute_expected_remaining method."""

    def test_expected_remaining_positive(self):
        """Expected remaining is positive when age < median."""
        stats = DurationStats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            n_segments=50,
            median_duration_bars=240,
            p25_duration_bars=180,
            p75_duration_bars=310,
        )

        remaining = stats.compute_expected_remaining(regime_age_bars=100)

        assert remaining.expected_remaining_bars == 140  # 240 - 100
        assert remaining.remaining_iqr_bars == [80, 210]  # [180-100, 310-100]

    def test_expected_remaining_zero_when_age_exceeds_median(self):
        """Expected remaining is zero when age >= median."""
        stats = DurationStats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            n_segments=50,
            median_duration_bars=240,
            p25_duration_bars=180,
            p75_duration_bars=310,
        )

        remaining = stats.compute_expected_remaining(regime_age_bars=300)

        assert remaining.expected_remaining_bars == 0  # max(0, 240 - 300)
        assert remaining.remaining_iqr_bars == [
            0,
            10,
        ]  # [max(0, 180-300), max(0, 310-300)]

    def test_expected_remaining_all_zero_when_very_old(self):
        """All remaining estimates are zero when regime is very old."""
        stats = DurationStats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            n_segments=50,
            median_duration_bars=240,
            p25_duration_bars=180,
            p75_duration_bars=310,
        )

        remaining = stats.compute_expected_remaining(regime_age_bars=500)

        assert remaining.expected_remaining_bars == 0
        assert remaining.remaining_iqr_bars == [0, 0]

    def test_expected_remaining_at_zero_age(self):
        """Expected remaining equals median when age is zero."""
        stats = DurationStats(
            symbol="BTC/USDT",
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            n_segments=50,
            median_duration_bars=240,
            p25_duration_bars=180,
            p75_duration_bars=310,
        )

        remaining = stats.compute_expected_remaining(regime_age_bars=0)

        assert remaining.expected_remaining_bars == 240
        assert remaining.remaining_iqr_bars == [180, 310]
