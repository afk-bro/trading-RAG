"""Tests for analytics_queries service module.

Tests the query helpers with mocked database pool.
No database required - all I/O is stubbed.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.admin.services.analytics_queries import (
    get_drift_driver_regimes,
    get_tier_usage_time_series,
)


# =============================================================================
# Test helpers
# =============================================================================


class AsyncContextManager:
    """Helper to mock async context managers like pool.acquire()."""

    def __init__(self, return_value, raise_error=False):
        self.return_value = return_value
        self.raise_error = raise_error

    async def __aenter__(self):
        if self.raise_error:
            raise Exception("Database connection failed")
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# =============================================================================
# get_drift_driver_regimes tests
# =============================================================================


class TestGetDriftDriverRegimes:
    """Tests for get_drift_driver_regimes helper."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_pool_is_none(self):
        """Returns empty list when pool is None."""
        workspace_id = uuid4()

        result = await get_drift_driver_regimes(
            pool=None,
            workspace_id=workspace_id,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_regime_keys_sorted_by_non_exact_count(self):
        """Returns regime keys ordered by non_exact_count descending."""
        workspace_id = uuid4()

        # Mock database response
        mock_rows = [
            {"query_regime_key": "trend=uptrend|vol=high_vol"},
            {"query_regime_key": "trend=flat|vol=low_vol"},
            {"query_regime_key": "trend=downtrend|vol=mid_vol"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_drift_driver_regimes(
            pool=mock_pool,
            workspace_id=workspace_id,
            limit=5,
        )

        assert result == [
            "trend=uptrend|vol=high_vol",
            "trend=flat|vol=low_vol",
            "trend=downtrend|vol=mid_vol",
        ]

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self):
        """Returns at most limit regime keys."""
        workspace_id = uuid4()

        # Mock database returns limited results (limit is applied in SQL)
        mock_rows = [
            {"query_regime_key": "trend=uptrend|vol=high_vol"},
            {"query_regime_key": "trend=flat|vol=low_vol"},
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_drift_driver_regimes(
            pool=mock_pool,
            workspace_id=workspace_id,
            limit=2,
        )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_filters_by_strategy_entity_id(self):
        """Includes strategy filter in query when provided."""
        workspace_id = uuid4()
        strategy_id = uuid4()

        mock_rows = []
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        await get_drift_driver_regimes(
            pool=mock_pool,
            workspace_id=workspace_id,
            strategy_entity_id=strategy_id,
        )

        # Verify fetch was called (strategy filter included in query)
        mock_conn.fetch.assert_called_once()
        call_args = mock_conn.fetch.call_args
        # The query should include strategy_entity_id parameter
        assert strategy_id in call_args[0]

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_drift_drivers(self):
        """Returns empty list when no regimes have non-exact fallback."""
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_drift_driver_regimes(
            pool=mock_pool,
            workspace_id=workspace_id,
        )

        assert result == []


# =============================================================================
# get_tier_usage_time_series tests
# =============================================================================


class TestGetTierUsageTimeSeries:
    """Tests for get_tier_usage_time_series helper."""

    @pytest.mark.asyncio
    async def test_raises_503_when_pool_is_none(self):
        """Raises HTTPException 503 when pool is None."""
        from fastapi import HTTPException

        workspace_id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await get_tier_usage_time_series(
                pool=None,
                workspace_id=workspace_id,
                strategy_entity_id=None,
                period_days=30,
                bucket="day",
                where_clause="workspace_id = $1",
                params=[workspace_id],
            )

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_returns_time_series_response(self):
        """Returns TierUsageTimeSeriesResponse with series data."""
        workspace_id = uuid4()
        bucket_time = datetime(2025, 1, 15, 0, 0, 0)

        mock_rows = [
            {
                "bucket_start": bucket_time,
                "tier_used": "exact",
                "count": 10,
                "avg_confidence": 0.85,
            },
            {
                "bucket_start": bucket_time,
                "tier_used": "distance",
                "count": 5,
                "avg_confidence": 0.65,
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_tier_usage_time_series(
            pool=mock_pool,
            workspace_id=workspace_id,
            strategy_entity_id=None,
            period_days=30,
            bucket="day",
            where_clause="workspace_id = $1",
            params=[workspace_id],
        )

        assert result.workspace_id == str(workspace_id)
        assert result.period_days == 30
        assert result.bucket == "day"
        assert result.total_recommendations == 15
        assert len(result.series) == 2
        assert len(result.buckets) == 1

    @pytest.mark.asyncio
    async def test_calculates_percentages_per_bucket(self):
        """Calculates tier percentages within each bucket."""
        workspace_id = uuid4()
        bucket_time = datetime(2025, 1, 15, 0, 0, 0)

        mock_rows = [
            {
                "bucket_start": bucket_time,
                "tier_used": "exact",
                "count": 80,
                "avg_confidence": 0.9,
            },
            {
                "bucket_start": bucket_time,
                "tier_used": "distance",
                "count": 20,
                "avg_confidence": 0.7,
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_tier_usage_time_series(
            pool=mock_pool,
            workspace_id=workspace_id,
            strategy_entity_id=None,
            period_days=30,
            bucket="day",
            where_clause="workspace_id = $1",
            params=[workspace_id],
        )

        # Find the series items
        exact_item = next(s for s in result.series if s.tier == "exact")
        distance_item = next(s for s in result.series if s.tier == "distance")

        assert exact_item.pct == 80.0  # 80/100 * 100
        assert distance_item.pct == 20.0  # 20/100 * 100

    @pytest.mark.asyncio
    async def test_builds_confidence_series(self):
        """Builds per-bucket confidence averages."""
        workspace_id = uuid4()
        bucket_time = datetime(2025, 1, 15, 0, 0, 0)

        mock_rows = [
            {
                "bucket_start": bucket_time,
                "tier_used": "exact",
                "count": 60,
                "avg_confidence": 0.9,
            },
            {
                "bucket_start": bucket_time,
                "tier_used": "distance",
                "count": 40,
                "avg_confidence": 0.7,
            },
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_tier_usage_time_series(
            pool=mock_pool,
            workspace_id=workspace_id,
            strategy_entity_id=None,
            period_days=30,
            bucket="day",
            where_clause="workspace_id = $1",
            params=[workspace_id],
        )

        assert len(result.confidence_series) == 1
        conf = result.confidence_series[0]
        assert conf.n == 100
        # Weighted avg: (60*0.9 + 40*0.7) / 100 = 0.82
        assert conf.avg_confidence == pytest.approx(0.82, abs=0.01)

    @pytest.mark.asyncio
    async def test_handles_empty_results(self):
        """Returns empty series when no data."""
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_tier_usage_time_series(
            pool=mock_pool,
            workspace_id=workspace_id,
            strategy_entity_id=None,
            period_days=30,
            bucket="day",
            where_clause="workspace_id = $1",
            params=[workspace_id],
        )

        assert result.total_recommendations == 0
        assert result.series == []
        assert result.buckets == []
        assert result.confidence_series == []

    @pytest.mark.asyncio
    async def test_includes_strategy_in_response(self):
        """Includes strategy_entity_id in response when provided."""
        workspace_id = uuid4()
        strategy_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool = MagicMock()
        mock_pool.acquire = MagicMock(return_value=AsyncContextManager(mock_conn))

        result = await get_tier_usage_time_series(
            pool=mock_pool,
            workspace_id=workspace_id,
            strategy_entity_id=strategy_id,
            period_days=30,
            bucket="week",
            where_clause="workspace_id = $1",
            params=[workspace_id],
        )

        assert result.strategy_entity_id == str(strategy_id)
        assert result.bucket == "week"
