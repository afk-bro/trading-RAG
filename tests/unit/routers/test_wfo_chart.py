"""Unit tests for WFO chart endpoints."""

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
def sample_wfo_id():
    """Sample WFO ID."""
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
def sample_wfo_row():
    """Sample wfo_runs row with full data."""
    tune_id_1 = uuid4()
    tune_id_2 = uuid4()
    return {
        "id": uuid4(),
        "workspace_id": SAMPLE_WS_ID,
        "status": "completed",
        "n_folds": 3,
        "folds_completed": 3,
        "folds_failed": 0,
        "wfo_config": {
            "train_days": 90,
            "test_days": 30,
            "step_days": 30,
            "min_folds": 3,
            "leaderboard_top_k": 10,
            "allow_partial": False,
        },
        "data_source": {
            "exchange_id": "binance",
            "symbol": "BTC-USDT",
            "timeframe": "1h",
        },
        "candidates": [
            {
                "params_hash": "abc123",
                "params": {"lookback": 20, "threshold": 0.02},
                "mean_oos": 1.5,
                "median_oos": 1.4,
                "worst_fold_oos": 0.8,
                "stddev_oos": 0.3,
                "pct_top_k": 0.9,
                "fold_count": 3,
                "total_folds": 3,
                "coverage": 1.0,
                "regime_tags": ["bullish"],
            },
            {
                "params_hash": "def456",
                "params": {"lookback": 30, "threshold": 0.03},
                "mean_oos": 1.2,
                "median_oos": 1.1,
                "worst_fold_oos": 0.5,
                "stddev_oos": 0.4,
                "pct_top_k": 0.7,
                "fold_count": 3,
                "total_folds": 3,
                "coverage": 1.0,
                "regime_tags": [],
            },
        ],
        "best_candidate": {
            "params_hash": "abc123",
            "params": {"lookback": 20, "threshold": 0.02},
            "mean_oos": 1.5,
            "median_oos": 1.4,
            "worst_fold_oos": 0.8,
            "stddev_oos": 0.3,
            "pct_top_k": 0.9,
            "fold_count": 3,
            "total_folds": 3,
            "coverage": 1.0,
            "regime_tags": ["bullish"],
        },
        "child_tune_ids": [tune_id_1, tune_id_2],
        "strategy_entity_id": uuid4(),
        "strategy_name": "test_strategy",
    }


@pytest.fixture
def sample_tune_rows(sample_wfo_row):
    """Sample tune rows for child tunes."""
    tune_ids = sample_wfo_row["child_tune_ids"]
    return [
        {
            "id": tune_ids[0],
            "status": "completed",
            "best_params": {"lookback": 20, "threshold": 0.02},
            "best_score": 1.5,
            "data_split": {
                "train_start": "2024-01-01T00:00:00Z",
                "train_end": "2024-04-01T00:00:00Z",
                "test_start": "2024-04-01T00:00:00Z",
                "test_end": "2024-05-01T00:00:00Z",
            },
            "dataset_meta": {
                "symbol": "BTC-USDT",
                "date_min": "2024-01-01T00:00:00Z",
                "date_max": "2024-05-01T00:00:00Z",
            },
        },
        {
            "id": tune_ids[1],
            "status": "completed",
            "best_params": {"lookback": 20, "threshold": 0.02},
            "best_score": 1.3,
            "data_split": {
                "train_start": "2024-02-01T00:00:00Z",
                "train_end": "2024-05-01T00:00:00Z",
                "test_start": "2024-05-01T00:00:00Z",
                "test_end": "2024-06-01T00:00:00Z",
            },
            "dataset_meta": None,
        },
    ]


@pytest.fixture
def sample_equity_curve():
    """Sample equity curve data."""
    return [
        {"t": "2024-04-01T00:00:00Z", "equity": 10000.0},
        {"t": "2024-04-15T00:00:00Z", "equity": 10500.0},
        {"t": "2024-05-01T00:00:00Z", "equity": 11000.0},
    ]


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestWFOChartHelpers:
    """Tests for wfo_chart.py helper functions."""

    def test_normalize_timestamp_with_datetime(self):
        """Normalize datetime to ISO with Z suffix."""
        from app.routers.backtests.wfo_chart import _normalize_timestamp

        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = _normalize_timestamp(dt)
        assert result == "2024-01-15T12:30:00Z"

    def test_normalize_timestamp_with_string_no_z(self):
        """Add Z suffix if missing from string."""
        from app.routers.backtests.wfo_chart import _normalize_timestamp

        result = _normalize_timestamp("2024-01-15T12:30:00")
        assert result == "2024-01-15T12:30:00Z"

    def test_normalize_timestamp_none(self):
        """Return empty string for None."""
        from app.routers.backtests.wfo_chart import _normalize_timestamp

        result = _normalize_timestamp(None)
        assert result == ""

    def test_parse_jsonb_dict(self):
        """Parse dict JSONB."""
        from app.routers.backtests.wfo_chart import _parse_jsonb

        raw = {"key": "value"}
        result = _parse_jsonb(raw)
        assert result == {"key": "value"}

    def test_parse_jsonb_string(self):
        """Parse string JSONB."""
        from app.routers.backtests.wfo_chart import _parse_jsonb

        raw = '{"key": "value"}'
        result = _parse_jsonb(raw)
        assert result == {"key": "value"}

    def test_parse_jsonb_none(self):
        """Parse None returns None."""
        from app.routers.backtests.wfo_chart import _parse_jsonb

        result = _parse_jsonb(None)
        assert result is None

    def test_parse_equity_curve_valid(self):
        """Parse valid equity curve."""
        from app.routers.backtests.wfo_chart import _parse_equity_curve

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
        from app.routers.backtests.wfo_chart import _parse_equity_curve

        points, source = _parse_equity_curve([])
        assert source == "missing"
        assert len(points) == 0

    def test_parse_equity_curve_none(self):
        """Parse None equity curve."""
        from app.routers.backtests.wfo_chart import _parse_equity_curve

        points, source = _parse_equity_curve(None)
        assert source == "missing"
        assert len(points) == 0


# =============================================================================
# WFO Chart Data Endpoint Tests
# =============================================================================


class TestWFOChartDataEndpoint:
    """Tests for GET /backtests/wfo/{wfo_id}/chart-data endpoint."""

    @pytest.mark.asyncio
    async def test_wfo_chart_data_returns_full_response(
        self, sample_wfo_id, sample_ws, mock_pool, sample_wfo_row, sample_tune_rows
    ):
        """Test WFO chart data returns full response with folds and candidates."""
        from app.routers.backtests import wfo_chart

        pool, conn = mock_pool
        wfo_chart.set_db_pool(pool)

        # Setup mock responses
        conn.fetchrow.return_value = sample_wfo_row
        conn.fetch.return_value = sample_tune_rows

        result = await wfo_chart.get_wfo_chart_data(sample_wfo_id, ws=sample_ws)

        assert result.status == "completed"
        assert result.n_folds == 3
        assert result.folds_completed == 3
        assert len(result.candidates) == 2
        assert result.best_candidate is not None
        assert result.best_candidate.mean_oos == 1.5
        assert len(result.folds) == 2
        assert result.selected_fold is None  # No fold_index provided
        assert result.notes == []

    @pytest.mark.asyncio
    async def test_wfo_chart_data_404_on_missing_wfo(
        self, sample_wfo_id, sample_ws, mock_pool
    ):
        """Test 404 for non-existent WFO."""
        from fastapi import HTTPException

        from app.routers.backtests import wfo_chart

        pool, conn = mock_pool
        wfo_chart.set_db_pool(pool)
        conn.fetchrow.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            await wfo_chart.get_wfo_chart_data(sample_wfo_id, ws=sample_ws)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_wfo_chart_data_with_fold_selection(
        self,
        sample_wfo_id,
        sample_ws,
        mock_pool,
        sample_wfo_row,
        sample_tune_rows,
        sample_equity_curve,
    ):
        """Test WFO chart data returns selected fold equity."""
        from app.routers.backtests import wfo_chart

        pool, conn = mock_pool
        wfo_chart.set_db_pool(pool)

        # The function makes multiple calls:
        # 1. fetchrow for WFO
        # 2. fetch for child tunes
        # 3. fetchrow for each fold's OOS metrics (inside loop)
        # 4. fetchrow for selected fold's best run equity
        fetchrow_results = [
            sample_wfo_row,  # WFO query
            {"metrics_oos": {"return_pct": 15.0, "sharpe": 1.2}},  # Fold 0 OOS
            {"metrics_oos": {"return_pct": 12.0, "sharpe": 1.0}},  # Fold 1 OOS
            # Selected fold best run equity
            {
                "id": uuid4(),
                "metrics_oos": {"return_pct": 15.0, "sharpe": 1.2},
                "equity_curve": sample_equity_curve,
                "summary": {"return_pct": 15.0, "sharpe": 1.2, "trades": 50},
            },
        ]

        conn.fetchrow.side_effect = fetchrow_results
        conn.fetch.return_value = sample_tune_rows

        result = await wfo_chart.get_wfo_chart_data(
            sample_wfo_id, ws=sample_ws, fold_index=0
        )

        assert result.selected_fold is not None
        assert result.selected_fold.fold_index == 0
        assert len(result.selected_fold.equity) == 3
        assert result.selected_fold.equity_source == "jsonb"

    @pytest.mark.asyncio
    async def test_wfo_chart_data_fold_out_of_range(
        self, sample_wfo_id, sample_ws, mock_pool, sample_wfo_row, sample_tune_rows
    ):
        """Test WFO chart data adds note when fold index out of range."""
        from app.routers.backtests import wfo_chart

        pool, conn = mock_pool
        wfo_chart.set_db_pool(pool)

        conn.fetchrow.return_value = sample_wfo_row
        conn.fetch.return_value = sample_tune_rows

        result = await wfo_chart.get_wfo_chart_data(
            sample_wfo_id, ws=sample_ws, fold_index=99
        )

        assert result.selected_fold is None
        assert any("does not exist" in n for n in result.notes)

    @pytest.mark.asyncio
    async def test_wfo_chart_data_no_candidates(
        self, sample_wfo_id, sample_ws, mock_pool, sample_wfo_row
    ):
        """Test WFO chart data handles no candidates."""
        from app.routers.backtests import wfo_chart

        pool, conn = mock_pool
        wfo_chart.set_db_pool(pool)

        # Remove candidates
        sample_wfo_row["candidates"] = []
        sample_wfo_row["best_candidate"] = None
        sample_wfo_row["child_tune_ids"] = []
        conn.fetchrow.return_value = sample_wfo_row

        result = await wfo_chart.get_wfo_chart_data(sample_wfo_id, ws=sample_ws)

        assert len(result.candidates) == 0
        assert result.best_candidate is None
        assert len(result.folds) == 0

    @pytest.mark.asyncio
    async def test_wfo_chart_data_parses_config(
        self, sample_wfo_id, sample_ws, mock_pool, sample_wfo_row
    ):
        """Test WFO chart data correctly parses config."""
        from app.routers.backtests import wfo_chart

        pool, conn = mock_pool
        wfo_chart.set_db_pool(pool)

        sample_wfo_row["child_tune_ids"] = []
        conn.fetchrow.return_value = sample_wfo_row

        result = await wfo_chart.get_wfo_chart_data(sample_wfo_id, ws=sample_ws)

        assert result.wfo_config["train_days"] == 90
        assert result.wfo_config["test_days"] == 30
        assert result.wfo_config["step_days"] == 30
        assert result.data_source["symbol"] == "BTC-USDT"
        assert result.strategy_name == "test_strategy"


# =============================================================================
# Candidate Model Tests
# =============================================================================


class TestCandidateMetrics:
    """Tests for CandidateMetrics model."""

    def test_candidate_metrics_validation(self):
        """Test CandidateMetrics model validation."""
        from app.routers.backtests.wfo_chart import CandidateMetrics

        candidate = CandidateMetrics(
            params_hash="abc123",
            params={"lookback": 20},
            mean_oos=1.5,
            median_oos=1.4,
            worst_fold_oos=0.8,
            stddev_oos=0.3,
            pct_top_k=0.9,
            fold_count=3,
            total_folds=3,
            coverage=1.0,
            regime_tags=["bullish"],
        )

        assert candidate.mean_oos == 1.5
        assert candidate.coverage == 1.0
        assert "bullish" in candidate.regime_tags

    def test_candidate_metrics_default_regime_tags(self):
        """Test CandidateMetrics default regime_tags."""
        from app.routers.backtests.wfo_chart import CandidateMetrics

        candidate = CandidateMetrics(
            params_hash="abc123",
            params={},
            mean_oos=1.0,
            median_oos=1.0,
            worst_fold_oos=0.5,
            stddev_oos=0.2,
            pct_top_k=0.5,
            fold_count=2,
            total_folds=3,
            coverage=0.67,
        )

        assert candidate.regime_tags == []


# =============================================================================
# Fold Summary Tests
# =============================================================================


class TestFoldSummary:
    """Tests for FoldSummary model."""

    def test_fold_summary_with_dates(self):
        """Test FoldSummary model with date range."""
        from app.routers.backtests.wfo_chart import FoldSummary

        fold = FoldSummary(
            fold_index=0,
            tune_id=str(uuid4()),
            status="completed",
            train_start="2024-01-01T00:00:00Z",
            train_end="2024-04-01T00:00:00Z",
            test_start="2024-04-01T00:00:00Z",
            test_end="2024-05-01T00:00:00Z",
            best_params={"lookback": 20},
            best_score=1.5,
            metrics_oos={"return_pct": 15.0, "sharpe": 1.2},
        )

        assert fold.fold_index == 0
        assert fold.status == "completed"
        assert fold.best_score == 1.5

    def test_fold_summary_missing_tune(self):
        """Test FoldSummary for missing tune."""
        from app.routers.backtests.wfo_chart import FoldSummary

        fold = FoldSummary(
            fold_index=2,
            tune_id=str(uuid4()),
            status="missing",
        )

        assert fold.status == "missing"
        assert fold.best_params is None
        assert fold.metrics_oos is None
