"""Unit tests for backtest chart endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.deps.security import WorkspaceContext


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_WS_ID = uuid4()


@pytest.fixture
def sample_ws():
    """Sample workspace context."""
    return WorkspaceContext(workspace_id=SAMPLE_WS_ID)


@pytest.fixture
def sample_run_id():
    """Sample run ID."""
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


@pytest.fixture
def sample_backtest_row():
    """Sample backtest_runs row with full data."""
    return {
        "id": uuid4(),
        "workspace_id": SAMPLE_WS_ID,
        "status": "completed",
        "params": {"lookback": 20, "threshold": 0.02},
        "summary": {
            "return_pct": 45.2,
            "max_drawdown_pct": 12.3,
            "sharpe": 1.8,
            "trades": 142,
            "win_rate": 0.585,
            "profit_factor": 1.65,
        },
        "dataset_meta": {
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-06-01T00:00:00Z",
            "row_count": 3624,
        },
        "equity_curve": [
            {"t": "2024-01-01T00:00:00Z", "equity": 10000.0},
            {"t": "2024-01-02T00:00:00Z", "equity": 10500.0},
            {"t": "2024-01-03T00:00:00Z", "equity": 10200.0},
        ],
        "trades": [
            {
                "t_entry": "2024-01-02T14:00:00Z",
                "t_exit": "2024-01-02T18:00:00Z",
                "side": "long",
                "size": 0.1,
                "entry_price": 42000.0,
                "exit_price": 42500.0,
                "pnl": 125.50,
                "return_pct": 1.25,
            },
            {
                "t_entry": "2024-01-03T10:00:00Z",
                "t_exit": "2024-01-03T14:00:00Z",
                "side": "short",
                "size": 0.1,
                "entry_price": 42500.0,
                "exit_price": 42200.0,
                "pnl": 75.00,
                "return_pct": 0.75,
            },
        ],
        "run_kind": None,
    }


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for chart.py helper functions."""

    def test_normalize_timestamp_with_datetime(self):
        """Normalize datetime to ISO with Z suffix."""
        from app.routers.backtests.chart import _normalize_timestamp

        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = _normalize_timestamp(dt)
        assert result == "2024-01-15T12:30:00Z"

    def test_normalize_timestamp_with_string_no_z(self):
        """Add Z suffix if missing from string."""
        from app.routers.backtests.chart import _normalize_timestamp

        result = _normalize_timestamp("2024-01-15T12:30:00")
        assert result == "2024-01-15T12:30:00Z"

    def test_normalize_timestamp_with_z(self):
        """Keep Z suffix if present."""
        from app.routers.backtests.chart import _normalize_timestamp

        result = _normalize_timestamp("2024-01-15T12:30:00Z")
        assert result == "2024-01-15T12:30:00Z"

    def test_normalize_timestamp_none(self):
        """Return empty string for None."""
        from app.routers.backtests.chart import _normalize_timestamp

        result = _normalize_timestamp(None)
        assert result == ""

    def test_parse_equity_curve_valid(self):
        """Parse valid equity curve."""
        from app.routers.backtests.chart import _parse_equity_curve

        raw = [
            {"t": "2024-01-01T00:00:00Z", "equity": 10000.0},
            {"t": "2024-01-02T00:00:00Z", "equity": 10500.0},
        ]
        points, source = _parse_equity_curve(raw)

        assert source == "jsonb"
        assert len(points) == 2
        assert points[0].equity == 10000.0
        assert points[1].equity == 10500.0

    def test_parse_equity_curve_empty(self):
        """Parse empty equity curve."""
        from app.routers.backtests.chart import _parse_equity_curve

        points, source = _parse_equity_curve([])
        assert source == "missing"
        assert len(points) == 0

    def test_parse_equity_curve_none(self):
        """Parse None equity curve."""
        from app.routers.backtests.chart import _parse_equity_curve

        points, source = _parse_equity_curve(None)
        assert source == "missing"
        assert len(points) == 0

    def test_parse_trades_pagination(self):
        """Test trades pagination."""
        from app.routers.backtests.chart import _parse_trades

        raw = [
            {
                "t_entry": f"2024-01-0{i}T00:00:00Z",
                "t_exit": f"2024-01-0{i}T04:00:00Z",
                "side": "long",
                "pnl": 100.0,
                "return_pct": 1.0,
            }
            for i in range(1, 6)
        ]

        # Page 1, size 2
        trades, total = _parse_trades(raw, page=1, page_size=2)
        assert total == 5
        assert len(trades) == 2

        # Page 2, size 2
        trades, total = _parse_trades(raw, page=2, page_size=2)
        assert total == 5
        assert len(trades) == 2

        # Page 3, size 2 (partial)
        trades, total = _parse_trades(raw, page=3, page_size=2)
        assert total == 5
        assert len(trades) == 1

    def test_parse_trades_empty(self):
        """Test empty trades."""
        from app.routers.backtests.chart import _parse_trades

        trades, total = _parse_trades(None, page=1, page_size=50)
        assert total == 0
        assert len(trades) == 0

    def test_parse_summary_valid(self):
        """Parse valid summary."""
        from app.routers.backtests.chart import _parse_summary

        raw = {
            "return_pct": 45.2,
            "max_drawdown_pct": 12.3,
            "sharpe": 1.8,
            "trades": 142,
            "win_rate": 0.585,
        }
        summary = _parse_summary(raw)

        assert summary.return_pct == 45.2
        assert summary.sharpe == 1.8
        assert summary.trades == 142

    def test_parse_summary_missing_keys(self):
        """Parse summary with missing keys returns None values."""
        from app.routers.backtests.chart import _parse_summary

        summary = _parse_summary({})
        assert summary.return_pct is None
        assert summary.sharpe is None

    def test_parse_dataset_meta_valid(self):
        """Parse valid dataset meta."""
        from app.routers.backtests.chart import _parse_dataset_meta

        raw = {
            "symbol": "BTC-USDT",
            "timeframe": "1h",
            "date_min": "2024-01-01T00:00:00Z",
            "date_max": "2024-06-01T00:00:00Z",
            "row_count": 3624,
        }
        meta = _parse_dataset_meta(raw)

        assert meta.symbol == "BTC-USDT"
        assert meta.timeframe == "1h"
        assert meta.row_count == 3624


# =============================================================================
# Chart Data Endpoint Tests
# =============================================================================


class TestChartDataEndpoint:
    """Tests for GET /backtests/runs/{run_id}/chart-data endpoint."""

    @pytest.mark.asyncio
    async def test_chart_data_returns_full_response(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test chart data returns full response with equity, summary, trades."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.get_chart_data(sample_run_id, ws=sample_ws, page=1, page_size=50)

        assert result.status == "completed"
        assert len(result.equity) == 3
        assert result.equity_source == "jsonb"
        assert result.summary.return_pct == 45.2
        assert result.summary.sharpe == 1.8
        assert result.trades_pagination.total == 2
        assert len(result.trades_page) == 2
        assert result.notes == []

    @pytest.mark.asyncio
    async def test_chart_data_404_on_missing_run(self, sample_run_id, sample_ws, mock_pool):
        """Test 404 for non-existent run."""
        from fastapi import HTTPException

        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)
        conn.fetchrow.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await chart.get_chart_data(sample_run_id, ws=sample_ws, page=1, page_size=50)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_chart_data_empty_equity_adds_note(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test empty equity curve adds note."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)

        # Set empty equity curve
        sample_backtest_row["equity_curve"] = []
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.get_chart_data(sample_run_id, ws=sample_ws, page=1, page_size=50)

        assert result.equity_source == "missing"
        assert len(result.equity) == 0
        assert "Equity curve not available" in result.notes[0]

    @pytest.mark.asyncio
    async def test_chart_data_tune_variant_adds_note(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test tune variant without equity adds specific note."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)

        # Set tune variant with no equity
        sample_backtest_row["equity_curve"] = None
        sample_backtest_row["run_kind"] = "tune_variant"
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.get_chart_data(sample_run_id, ws=sample_ws, page=1, page_size=50)

        assert "tune variants" in result.notes[0].lower()

    @pytest.mark.asyncio
    async def test_chart_data_no_trades_adds_note(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test no trades adds note."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)

        # Set no trades
        sample_backtest_row["trades"] = None
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.get_chart_data(sample_run_id, ws=sample_ws, page=1, page_size=50)

        assert result.trades_pagination.total == 0
        assert result.exports.trades_csv is None
        assert any("Trades not stored" in n for n in result.notes)

    @pytest.mark.asyncio
    async def test_chart_data_pagination_works(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test pagination returns correct page."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)

        # Add more trades
        sample_backtest_row["trades"] = [
            {
                "t_entry": f"2024-01-0{i}T00:00:00Z",
                "t_exit": f"2024-01-0{i}T04:00:00Z",
                "side": "long",
                "pnl": 100.0,
                "return_pct": 1.0,
            }
            for i in range(1, 8)
        ]
        conn.fetchrow.return_value = sample_backtest_row

        # Page 1
        result1 = await chart.get_chart_data(sample_run_id, ws=sample_ws, page=1, page_size=3)
        assert result1.trades_pagination.total == 7
        assert result1.trades_pagination.page == 1
        assert len(result1.trades_page) == 3

        # Page 2
        result2 = await chart.get_chart_data(sample_run_id, ws=sample_ws, page=2, page_size=3)
        assert result2.trades_pagination.page == 2
        assert len(result2.trades_page) == 3


# =============================================================================
# Export Endpoint Tests
# =============================================================================


class TestExportEndpoints:
    """Tests for export endpoints."""

    @pytest.mark.asyncio
    async def test_trades_csv_export(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test trades CSV export returns CSV."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)
        conn.fetchrow.return_value = {"trades": sample_backtest_row["trades"]}

        response = await chart.export_trades_csv(sample_run_id, ws=sample_ws)

        assert response.media_type == "text/csv"
        # Check headers
        headers = dict(response.headers)
        assert "content-disposition" in headers
        assert "attachment" in headers["content-disposition"]
        assert ".csv" in headers["content-disposition"]

    @pytest.mark.asyncio
    async def test_trades_csv_404_no_trades(self, sample_run_id, sample_ws, mock_pool):
        """Test trades CSV export returns 404 when no trades."""
        from fastapi import HTTPException

        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)
        conn.fetchrow.return_value = {"trades": []}

        with pytest.raises(HTTPException) as exc_info:
            await chart.export_trades_csv(sample_run_id, ws=sample_ws)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_json_snapshot_export(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test JSON snapshot export returns full data."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.export_json_snapshot(sample_run_id, ws=sample_ws)

        assert result["run_id"] == str(sample_run_id)
        assert result["status"] == "completed"
        assert len(result["equity"]) == 3
        assert len(result["trades"]) == 2
        assert "exported_at" in result


# =============================================================================
# Sparkline Endpoint Tests
# =============================================================================


class TestSparklineEndpoint:
    """Tests for GET /backtests/runs/{run_id}/sparkline endpoint."""

    @pytest.mark.asyncio
    async def test_sparkline_returns_downsampled_data(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test sparkline returns downsampled equity values."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.get_sparkline(sample_run_id, ws=sample_ws, max_points=96)

        assert result.status == "ok"
        assert len(result.y) == 3  # Original has 3 points, no downsampling needed
        assert result.y[0] == 10000.0
        assert result.y[1] == 10500.0
        assert result.y[2] == 10200.0

    @pytest.mark.asyncio
    async def test_sparkline_404_on_missing_run(self, sample_run_id, sample_ws, mock_pool):
        """Test 404 for non-existent run."""
        from fastapi import HTTPException

        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)
        conn.fetchrow.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await chart.get_sparkline(sample_run_id, ws=sample_ws)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_sparkline_pending_run_returns_empty(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test pending run returns empty sparkline with pending status."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)

        sample_backtest_row["status"] = "running"
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.get_sparkline(sample_run_id, ws=sample_ws)

        assert result.status == "pending"
        assert len(result.y) == 0

    @pytest.mark.asyncio
    async def test_sparkline_empty_equity_returns_empty(
        self, sample_run_id, sample_ws, mock_pool, sample_backtest_row
    ):
        """Test run without equity returns empty sparkline."""
        from app.routers.backtests import chart

        pool, conn = mock_pool
        chart.set_db_pool(pool)

        sample_backtest_row["equity_curve"] = []
        conn.fetchrow.return_value = sample_backtest_row

        result = await chart.get_sparkline(sample_run_id, ws=sample_ws)

        assert result.status == "empty"
        assert len(result.y) == 0


class TestDownsampleEquity:
    """Tests for _downsample_equity helper function."""

    def test_downsample_returns_all_if_under_max(self):
        """Test returns all points when under max."""
        from app.routers.backtests.chart import EquityPoint, _downsample_equity

        points = [
            EquityPoint(t="2024-01-01", equity=10000.0),
            EquityPoint(t="2024-01-02", equity=10500.0),
            EquityPoint(t="2024-01-03", equity=10200.0),
        ]

        result = _downsample_equity(points, max_points=10)

        assert len(result) == 3
        assert result == [10000.0, 10500.0, 10200.0]

    def test_downsample_reduces_to_max(self):
        """Test downsamples to max points."""
        from app.routers.backtests.chart import EquityPoint, _downsample_equity

        # Create 100 points
        points = [
            EquityPoint(t=f"2024-01-{i:02d}", equity=10000.0 + i * 10)
            for i in range(1, 101)
        ]

        result = _downsample_equity(points, max_points=10)

        assert len(result) == 10
        # First and last should be included
        assert result[0] == 10010.0  # First point
        assert result[-1] == 11000.0  # Last point

    def test_downsample_empty_returns_empty(self):
        """Test empty input returns empty."""
        from app.routers.backtests.chart import _downsample_equity

        result = _downsample_equity([], max_points=10)

        assert result == []

    def test_downsample_preserves_endpoints(self):
        """Test downsampling always includes first and last points."""
        from app.routers.backtests.chart import EquityPoint, _downsample_equity

        # Create 50 points with distinctive endpoints
        points = [
            EquityPoint(t=f"2024-01-{i:02d}", equity=10000.0 + i) for i in range(50)
        ]
        points[0] = EquityPoint(t="2024-01-00", equity=99999.0)  # Distinctive first
        points[-1] = EquityPoint(t="2024-02-18", equity=88888.0)  # Distinctive last

        result = _downsample_equity(points, max_points=10)

        assert result[0] == 99999.0  # First preserved
        assert result[-1] == 88888.0  # Last preserved
