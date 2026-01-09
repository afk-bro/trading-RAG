"""Tests for cluster stats repository."""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from app.repositories.cluster_stats import (
    ClusterStatsRepository,
    ClusterStats,
)


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


class TestClusterStatsRepository:
    """Tests for ClusterStatsRepository."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_none_when_missing(self, mock_pool):
        """Returns None when no stats exist for key."""
        repo = ClusterStatsRepository(mock_pool)

        # Mock empty result
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats(
            strategy_entity_id=uuid4(),
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_stats_returns_cluster_stats_when_found(self, mock_pool):
        """Returns ClusterStats when row exists."""
        repo = ClusterStatsRepository(mock_pool)
        strategy_id = uuid4()

        # Mock found result
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={
            "strategy_entity_id": strategy_id,
            "timeframe": "5m",
            "regime_key": "trend=uptrend|vol=high_vol",
            "regime_dims": {"trend": "uptrend", "vol": "high_vol"},
            "n": 50,
            "feature_schema_version": 1,
            "feature_mean": {"atr_pct": 0.02, "rsi": 50.0},
            "feature_var": {"atr_pct": 0.001, "rsi": 100.0},
            "feature_min": None,
            "feature_max": None,
            "updated_at": None,
        })
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats(
            strategy_entity_id=strategy_id,
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.strategy_entity_id == strategy_id
        assert result.n == 50
        assert result.feature_mean == {"atr_pct": 0.02, "rsi": 50.0}
        assert result.baseline == "composite"

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_returns_composite_first(self, mock_pool):
        """Returns composite stats when available."""
        repo = ClusterStatsRepository(mock_pool)
        strategy_id = uuid4()

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={
            "strategy_entity_id": strategy_id,
            "timeframe": "5m",
            "regime_key": "trend=uptrend|vol=high_vol",
            "regime_dims": {"trend": "uptrend", "vol": "high_vol"},
            "n": 50,
            "feature_schema_version": 1,
            "feature_mean": {"atr_pct": 0.02},
            "feature_var": {"atr_pct": 0.001},
            "feature_min": None,
            "feature_max": None,
            "updated_at": None,
        })
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            strategy_entity_id=strategy_id,
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.baseline == "composite"

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_falls_back_to_marginal(self, mock_pool):
        """Falls back to marginal when composite missing."""
        repo = ClusterStatsRepository(mock_pool)
        strategy_id = uuid4()

        call_count = 0

        async def mock_fetchrow(query, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Composite not found
                return None
            else:
                # Marginal found
                return {
                    "strategy_entity_id": strategy_id,
                    "timeframe": "5m",
                    "regime_key": "trend=uptrend",
                    "regime_dims": {"trend": "uptrend"},
                    "n": 10,
                    "feature_schema_version": 1,
                    "feature_mean": {"atr_pct": 0.02},
                    "feature_var": {"atr_pct": 0.001},
                    "feature_min": None,
                    "feature_max": None,
                    "updated_at": None,
                }

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            strategy_entity_id=strategy_id,
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.baseline == "marginal"

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_returns_none_when_all_missing(self, mock_pool):
        """Returns None when composite and all marginals are missing."""
        repo = ClusterStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            strategy_entity_id=uuid4(),
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_stats_with_backoff_respects_min_n(self, mock_pool):
        """Falls back when composite n is below min_n threshold."""
        repo = ClusterStatsRepository(mock_pool)
        strategy_id = uuid4()

        call_count = 0

        async def mock_fetchrow(query, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Composite found but n too low
                return {
                    "strategy_entity_id": strategy_id,
                    "timeframe": "5m",
                    "regime_key": "trend=uptrend|vol=high_vol",
                    "regime_dims": {"trend": "uptrend", "vol": "high_vol"},
                    "n": 5,  # Below default min_n of 20
                    "feature_schema_version": 1,
                    "feature_mean": {"atr_pct": 0.02},
                    "feature_var": {"atr_pct": 0.001},
                    "feature_min": None,
                    "feature_max": None,
                    "updated_at": None,
                }
            else:
                # Marginal with sufficient n
                return {
                    "strategy_entity_id": strategy_id,
                    "timeframe": "5m",
                    "regime_key": "trend=uptrend",
                    "regime_dims": {"trend": "uptrend"},
                    "n": 100,
                    "feature_schema_version": 1,
                    "feature_mean": {"atr_pct": 0.025},
                    "feature_var": {"atr_pct": 0.002},
                    "feature_min": None,
                    "feature_max": None,
                    "updated_at": None,
                }

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            strategy_entity_id=strategy_id,
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            min_n=20,
        )

        assert result is not None
        assert result.baseline == "marginal"

    @pytest.mark.asyncio
    async def test_upsert_stats(self, mock_pool):
        """Upsert creates or updates stats."""
        repo = ClusterStatsRepository(mock_pool)

        conn = AsyncMock()
        conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        stats = ClusterStats(
            strategy_entity_id=uuid4(),
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            regime_dims={"trend": "uptrend", "vol": "high_vol"},
            n=50,
            feature_mean={"atr_pct": 0.02, "rsi": 50.0},
            feature_var={"atr_pct": 0.001, "rsi": 100.0},
        )

        await repo.upsert_stats(stats)

        conn.execute.assert_called_once()


class TestClusterStats:
    """Tests for ClusterStats dataclass."""

    def test_cluster_stats_creation(self):
        """ClusterStats can be created with required fields."""
        stats = ClusterStats(
            strategy_entity_id=uuid4(),
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            regime_dims={"trend": "uptrend", "vol": "high_vol"},
            n=50,
            feature_mean={"atr_pct": 0.02},
            feature_var={"atr_pct": 0.001},
        )

        assert stats.n == 50
        assert stats.baseline == "composite"  # default
        assert stats.feature_schema_version == 1  # default

    def test_cluster_stats_optional_fields(self):
        """ClusterStats can have optional min/max fields."""
        stats = ClusterStats(
            strategy_entity_id=uuid4(),
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            regime_dims={"trend": "uptrend", "vol": "high_vol"},
            n=50,
            feature_mean={"atr_pct": 0.02},
            feature_var={"atr_pct": 0.001},
            feature_min={"atr_pct": 0.01},
            feature_max={"atr_pct": 0.05},
        )

        assert stats.feature_min == {"atr_pct": 0.01}
        assert stats.feature_max == {"atr_pct": 0.05}


class TestCombineMarginals:
    """Tests for marginal combination logic."""

    @pytest.mark.asyncio
    async def test_combine_marginals_takes_max_variance(self, mock_pool):
        """Combined marginals use max variance per feature (conservative)."""
        repo = ClusterStatsRepository(mock_pool)
        strategy_id = uuid4()

        call_count = 0

        async def mock_fetchrow(query, *args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Composite not found
                return None
            elif call_count == 2:
                # First marginal: trend=uptrend
                return {
                    "strategy_entity_id": strategy_id,
                    "timeframe": "5m",
                    "regime_key": "trend=uptrend",
                    "regime_dims": {"trend": "uptrend"},
                    "n": 30,
                    "feature_schema_version": 1,
                    "feature_mean": {"atr_pct": 0.02, "rsi": 50.0},
                    "feature_var": {"atr_pct": 0.001, "rsi": 50.0},  # lower var
                    "feature_min": None,
                    "feature_max": None,
                    "updated_at": None,
                }
            else:
                # Second marginal: vol=high_vol
                return {
                    "strategy_entity_id": strategy_id,
                    "timeframe": "5m",
                    "regime_key": "vol=high_vol",
                    "regime_dims": {"vol": "high_vol"},
                    "n": 40,
                    "feature_schema_version": 1,
                    "feature_mean": {"atr_pct": 0.025, "rsi": 55.0},
                    "feature_var": {"atr_pct": 0.002, "rsi": 100.0},  # higher var
                    "feature_min": None,
                    "feature_max": None,
                    "updated_at": None,
                }

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_stats_with_backoff(
            strategy_entity_id=strategy_id,
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
        )

        assert result is not None
        assert result.baseline == "marginal"
        # Should take max variance per feature
        assert result.feature_var["atr_pct"] == 0.002  # max of 0.001, 0.002
        assert result.feature_var["rsi"] == 100.0  # max of 50.0, 100.0
        # Mean should be averaged
        assert result.feature_mean["atr_pct"] == pytest.approx(0.0225)  # avg of 0.02, 0.025
        # n should be sum
        assert result.n == 70  # 30 + 40
