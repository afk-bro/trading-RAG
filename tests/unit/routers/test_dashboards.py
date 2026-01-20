"""Unit tests for dashboard endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_workspace_id():
    """Sample workspace ID."""
    return uuid4()


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = MagicMock()
    conn = AsyncMock()

    # Setup acquire as async context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm

    return pool, conn


# =============================================================================
# Equity Curve Tests
# =============================================================================


class TestEquityCurveEndpoint:
    """Tests for GET /dashboards/{workspace_id}/equity endpoint."""

    @pytest.mark.asyncio
    async def test_equity_curve_returns_data(self, sample_workspace_id, mock_pool):
        """Test equity curve returns time series data."""
        from app.routers.dashboards import get_equity_curve

        pool, conn = mock_pool
        now = datetime.now(timezone.utc)

        # Mock equity data
        conn.fetch.return_value = [
            {
                "snapshot_ts": now,
                "computed_at": now,
                "equity": 10000.0,
                "cash": 8000.0,
                "positions_value": 2000.0,
                "realized_pnl": 500.0,
                "peak_equity": 10000.0,
                "drawdown_pct": 0.0,
            },
            {
                "snapshot_ts": now,
                "computed_at": now,
                "equity": 9500.0,
                "cash": 7500.0,
                "positions_value": 2000.0,
                "realized_pnl": 400.0,
                "peak_equity": 10000.0,
                "drawdown_pct": 0.05,
            },
        ]

        result = await get_equity_curve(sample_workspace_id, days=30, pool=pool)

        assert result["workspace_id"] == str(sample_workspace_id)
        assert result["window_days"] == 30
        assert result["snapshot_count"] == 2
        assert len(result["data"]) == 2
        assert "summary" in result
        assert result["summary"]["max_drawdown_pct"] == 0.05

    @pytest.mark.asyncio
    async def test_equity_curve_handles_empty_data(
        self, sample_workspace_id, mock_pool
    ):
        """Test equity curve handles empty data gracefully."""
        from app.routers.dashboards import get_equity_curve

        pool, conn = mock_pool
        conn.fetch.return_value = []

        result = await get_equity_curve(sample_workspace_id, days=30, pool=pool)

        assert result["snapshot_count"] == 0
        assert result["data"] == []
        assert result["summary"]["current_equity"] is None


# =============================================================================
# Intel Timeline Tests
# =============================================================================


class TestIntelTimelineEndpoint:
    """Tests for GET /dashboards/{workspace_id}/intel-timeline endpoint."""

    @pytest.mark.asyncio
    async def test_intel_timeline_returns_data(self, sample_workspace_id, mock_pool):
        """Test intel timeline returns version data."""
        from app.routers.dashboards import get_intel_timeline

        pool, conn = mock_pool
        now = datetime.now(timezone.utc)
        version_id = uuid4()

        # Mock intel data
        conn.fetch.return_value = [
            {
                "id": uuid4(),
                "strategy_version_id": version_id,
                "version_number": 1,
                "version_tag": "v1.0",
                "strategy_name": "Test Strategy",
                "as_of_ts": now,
                "computed_at": now,
                "regime": "bullish",
                "confidence_score": 0.75,
                "confidence_components": {"backtest": 0.8, "wfo": 0.7},
                "source_snapshot_id": None,
            },
        ]

        result = await get_intel_timeline(sample_workspace_id, days=14, pool=pool)

        assert result["workspace_id"] == str(sample_workspace_id)
        assert result["window_days"] == 14
        assert result["version_count"] == 1
        assert len(result["versions"]) == 1
        assert result["versions"][0]["strategy_name"] == "Test Strategy"

    @pytest.mark.asyncio
    async def test_intel_timeline_filters_by_version(
        self, sample_workspace_id, mock_pool
    ):
        """Test intel timeline can filter by version_id."""
        from app.routers.dashboards import get_intel_timeline

        pool, conn = mock_pool
        version_id = uuid4()
        conn.fetch.return_value = []

        result = await get_intel_timeline(
            sample_workspace_id, version_id=version_id, days=14, pool=pool
        )

        assert result["version_filter"] == str(version_id)
        # Verify query was called with version_id parameter
        call_args = conn.fetch.call_args[0]
        assert version_id in call_args


# =============================================================================
# Active Alerts Tests
# =============================================================================


class TestActiveAlertsEndpoint:
    """Tests for GET /dashboards/{workspace_id}/alerts endpoint."""

    @pytest.mark.asyncio
    async def test_alerts_returns_active_alerts(self, sample_workspace_id, mock_pool):
        """Test alerts endpoint returns active alerts."""
        from app.routers.dashboards import get_active_alerts

        pool, conn = mock_pool
        now = datetime.now(timezone.utc)
        alert_id = uuid4()

        # Mock alert data
        conn.fetch.return_value = [
            {
                "id": alert_id,
                "workspace_id": sample_workspace_id,
                "rule_type": "workspace_drawdown_high",
                "severity": "high",
                "status": "active",
                "dedupe_key": "workspace_drawdown_high:critical:2024-01-15",
                "payload": {"drawdown_pct": 0.22},
                "source": "alert_evaluator",
                "rule_version": None,
                "occurrence_count": 3,
                "first_triggered_at": now,
                "last_triggered_at": now,
                "acknowledged_at": None,
                "acknowledged_by": None,
                "resolved_at": None,
                "resolved_by": None,
                "resolution_note": None,
                "created_at": now,
            },
        ]

        result = await get_active_alerts(sample_workspace_id, days=7, pool=pool)

        assert result["workspace_id"] == str(sample_workspace_id)
        assert result["total_alerts"] == 1
        assert result["alerts"][0]["rule_type"] == "workspace_drawdown_high"
        assert result["summary"]["by_severity"]["high"] == 1

    @pytest.mark.asyncio
    async def test_alerts_includes_resolved_when_requested(
        self, sample_workspace_id, mock_pool
    ):
        """Test alerts can include resolved alerts."""
        from app.routers.dashboards import get_active_alerts

        pool, conn = mock_pool
        conn.fetch.return_value = []

        result = await get_active_alerts(
            sample_workspace_id, include_resolved=True, days=7, pool=pool
        )

        assert result["include_resolved"] is True


# =============================================================================
# Dashboard Summary Tests
# =============================================================================


class TestDashboardSummaryEndpoint:
    """Tests for GET /dashboards/{workspace_id}/summary endpoint."""

    @pytest.mark.asyncio
    async def test_summary_returns_combined_data(self, sample_workspace_id, mock_pool):
        """Test summary returns combined equity, intel, and alert data."""
        from app.routers.dashboards import get_dashboard_summary

        pool, conn = mock_pool
        now = datetime.now(timezone.utc)
        version_id = uuid4()

        # Mock equity data
        conn.fetchrow.return_value = {
            "equity": 10000.0,
            "cash": 8000.0,
            "positions_value": 2000.0,
            "snapshot_ts": now,
            "peak_equity": 10500.0,
            "drawdown_pct": 0.0476,
        }

        # Mock intel data (first fetch call)
        # Mock alert counts (second fetch call)
        conn.fetch.side_effect = [
            # Intel rows
            [
                {
                    "strategy_version_id": version_id,
                    "version_number": 1,
                    "strategy_name": "Test Strategy",
                    "regime": "bullish",
                    "confidence_score": 0.75,
                    "as_of_ts": now,
                    "rn": 1,
                },
            ],
            # Alert counts
            [
                {"severity": "high", "count": 1},
                {"severity": "medium", "count": 2},
            ],
        ]

        result = await get_dashboard_summary(sample_workspace_id, pool=pool)

        assert result["workspace_id"] == str(sample_workspace_id)
        assert "generated_at" in result
        assert result["equity"]["equity"] == 10000.0
        assert result["intel"]["active_versions"] == 1
        assert result["alerts"]["total_active"] == 3

    @pytest.mark.asyncio
    async def test_summary_handles_no_data(self, sample_workspace_id, mock_pool):
        """Test summary handles missing data gracefully."""
        from app.routers.dashboards import get_dashboard_summary

        pool, conn = mock_pool

        # No data
        conn.fetchrow.return_value = None
        conn.fetch.side_effect = [[], []]

        result = await get_dashboard_summary(sample_workspace_id, pool=pool)

        assert result["equity"] is None
        assert result["intel"]["active_versions"] == 0
        assert result["alerts"]["total_active"] == 0
