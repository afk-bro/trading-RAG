"""Tests for WFOJob handler."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.jobs.handlers.wfo import (
    handle_wfo,
    parse_iso_timestamp,
    _build_child_tune_payload,
)
from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType, JobStatus


class TestParseIsoTimestamp:
    """Tests for parse_iso_timestamp."""

    def test_parse_z_suffix(self):
        """Should parse Z suffix as UTC."""
        result = parse_iso_timestamp("2024-01-01T00:00:00Z")
        assert result == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_parse_offset(self):
        """Should parse timezone offset."""
        result = parse_iso_timestamp("2024-01-01T00:00:00+00:00")
        assert result == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_parse_naive_assumes_utc(self):
        """Should assume UTC for naive timestamps."""
        result = parse_iso_timestamp("2024-01-01T00:00:00")
        assert result == datetime(2024, 1, 1, tzinfo=timezone.utc)


class TestBuildChildTunePayload:
    """Tests for _build_child_tune_payload."""

    def test_builds_payload_with_fold_dates(self):
        """Should override data_source dates with fold dates."""
        from app.services.backtest.wfo import Fold

        fold = Fold(
            index=0,
            train_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            train_end=datetime(2024, 4, 1, tzinfo=timezone.utc),
            test_start=datetime(2024, 4, 1, tzinfo=timezone.utc),
            test_end=datetime(2024, 5, 1, tzinfo=timezone.utc),
        )
        base_payload = {
            "workspace_id": "ws-123",
            "data_source": {
                "exchange_id": "kucoin",
                "symbol": "BTC-USDT",
                "timeframe": "1h",
                "start_ts": "2024-01-01T00:00:00Z",  # Will be overridden
                "end_ts": "2024-12-31T00:00:00Z",  # Will be overridden
            },
            "param_space": {"lookback": [10, 20, 30]},
        }
        wfo_id = uuid4()

        result = _build_child_tune_payload(fold, base_payload, wfo_id, 0)

        # Should have train dates in data_source
        assert result["data_source"]["start_ts"] == fold.train_start.isoformat()
        assert result["data_source"]["end_ts"] == fold.train_end.isoformat()

        # Should add fold metadata
        assert result["wfo_fold_index"] == 0
        assert result["wfo_parent_id"] == str(wfo_id)

        # Should calculate OOS ratio
        expected_oos = fold.test_days / (fold.train_days + fold.test_days)
        assert result["oos_ratio"] == pytest.approx(expected_oos)

    def test_preserves_base_payload(self):
        """Should preserve other base payload fields."""
        from app.services.backtest.wfo import Fold

        fold = Fold(
            index=1,
            train_start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            train_end=datetime(2024, 5, 1, tzinfo=timezone.utc),
            test_start=datetime(2024, 5, 1, tzinfo=timezone.utc),
            test_end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        base_payload = {
            "workspace_id": "ws-456",
            "data_source": {"exchange_id": "binance"},
            "param_space": {"threshold": [0.5, 0.7]},
            "search_type": "random",
            "objective_type": "sharpe_dd_penalty",
        }

        result = _build_child_tune_payload(fold, base_payload, uuid4(), 1)

        assert result["workspace_id"] == "ws-456"
        assert result["param_space"] == {"threshold": [0.5, 0.7]}
        assert result["search_type"] == "random"
        assert result["objective_type"] == "sharpe_dd_penalty"


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_wfo_handler_registered(self):
        """Should register WFO handler with registry."""
        handler = default_registry.get_handler(JobType.WFO)
        assert handler is not None
        assert handler.__name__ == "handle_wfo"


class TestHandleWfo:
    """Tests for handle_wfo function."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        return MagicMock()

    @pytest.fixture
    def mock_events_repo(self):
        """Create mock events repository."""
        repo = MagicMock()
        repo.info = AsyncMock()
        repo.error = AsyncMock()
        repo.warning = AsyncMock()
        return repo

    @pytest.fixture
    def mock_job_repo(self):
        """Create mock job repository."""
        repo = MagicMock()
        repo.enqueue = AsyncMock()
        repo.list_children = AsyncMock(return_value=[])
        repo.get = AsyncMock(return_value=None)
        repo.update_status = AsyncMock()
        return repo

    @pytest.fixture
    def mock_ohlcv_repo(self):
        """Create mock OHLCV repository."""
        repo = MagicMock()
        repo.get_available_range = AsyncMock(return_value={
            "min_ts": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "max_ts": datetime(2024, 12, 31, tzinfo=timezone.utc),
            "row_count": 17520,
        })
        return repo

    @pytest.fixture
    def mock_tune_repo(self):
        """Create mock tune repository."""
        repo = MagicMock()
        repo.create_tune = AsyncMock(return_value={"id": uuid4()})
        return repo

    @pytest.fixture
    def base_job(self):
        """Create base job for testing."""
        return Job(
            id=uuid4(),
            type=JobType.WFO,
            status=JobStatus.RUNNING,
            payload={
                "workspace_id": str(uuid4()),
                "wfo_id": str(uuid4()),
                "strategy_entity_id": str(uuid4()),
                "data_source": {
                    "exchange_id": "kucoin",
                    "symbol": "BTC-USDT",
                    "timeframe": "1h",
                },
                "wfo_config": {
                    "train_days": 90,
                    "test_days": 30,
                    "step_days": 30,
                    "min_folds": 3,
                },
                "param_space": {"lookback": [10, 20, 30]},
            },
        )

    @pytest.mark.asyncio
    async def test_missing_workspace_id_raises(
        self, mock_pool, mock_events_repo
    ):
        """Should raise ValueError if workspace_id missing."""
        job = Job(
            id=uuid4(),
            type=JobType.WFO,
            status=JobStatus.RUNNING,
            payload={"wfo_id": str(uuid4())},
        )
        ctx = {"pool": mock_pool, "events_repo": mock_events_repo}

        with pytest.raises(ValueError, match="workspace_id"):
            await handle_wfo(job, ctx)

    @pytest.mark.asyncio
    async def test_missing_wfo_id_raises(
        self, mock_pool, mock_events_repo
    ):
        """Should raise ValueError if wfo_id missing."""
        job = Job(
            id=uuid4(),
            type=JobType.WFO,
            status=JobStatus.RUNNING,
            payload={"workspace_id": str(uuid4())},
        )
        ctx = {"pool": mock_pool, "events_repo": mock_events_repo}

        with pytest.raises(ValueError, match="wfo_id"):
            await handle_wfo(job, ctx)

    @pytest.mark.asyncio
    async def test_missing_data_source_raises(
        self, mock_pool, mock_events_repo
    ):
        """Should raise ValueError if data_source missing."""
        job = Job(
            id=uuid4(),
            type=JobType.WFO,
            status=JobStatus.RUNNING,
            payload={
                "workspace_id": str(uuid4()),
                "wfo_id": str(uuid4()),
                "strategy_entity_id": str(uuid4()),
                "wfo_config": {"train_days": 90, "test_days": 30, "step_days": 30},
            },
        )
        ctx = {"pool": mock_pool, "events_repo": mock_events_repo}

        with pytest.raises(ValueError, match="data_source"):
            await handle_wfo(job, ctx)

    @pytest.mark.asyncio
    async def test_enqueues_child_tune_jobs(
        self,
        mock_pool,
        mock_events_repo,
        mock_job_repo,
        mock_ohlcv_repo,
        mock_tune_repo,
        base_job,
    ):
        """Should enqueue child tune jobs for each fold."""
        # Use shorter data range to reduce fold count
        mock_ohlcv_repo.get_available_range = AsyncMock(return_value={
            "min_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "max_ts": datetime(2024, 6, 30, tzinfo=timezone.utc),  # ~180 days
            "row_count": 4320,
        })

        # Setup: Return succeeded jobs when polling (3 folds)
        succeeded_jobs = [
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.SUCCEEDED, payload={}),
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.SUCCEEDED, payload={}),
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.SUCCEEDED, payload={}),
        ]
        mock_job_repo.list_children = AsyncMock(return_value=succeeded_jobs)

        # Setup: Return job when enqueuing (use function to create new job each time)
        async def create_enqueued_job(*args, **kwargs):
            return Job(
                id=uuid4(), type=JobType.TUNE, status=JobStatus.PENDING, payload={}
            )
        mock_job_repo.enqueue = AsyncMock(side_effect=create_enqueued_job)

        ctx = {"pool": mock_pool, "events_repo": mock_events_repo}

        with patch(
            "app.jobs.handlers.wfo.JobRepository", return_value=mock_job_repo
        ), patch(
            "app.jobs.handlers.wfo.OHLCVRepository", return_value=mock_ohlcv_repo
        ), patch(
            "app.jobs.handlers.wfo.TuneRepository", return_value=mock_tune_repo
        ):
            result = await handle_wfo(base_job, ctx)

        assert result["status"] == "completed"
        assert result["n_folds"] >= 3
        assert result["folds_completed"] == 3
        assert mock_job_repo.enqueue.call_count >= 3

    @pytest.mark.asyncio
    async def test_returns_failed_when_all_folds_fail(
        self,
        mock_pool,
        mock_events_repo,
        mock_job_repo,
        mock_ohlcv_repo,
        mock_tune_repo,
        base_job,
    ):
        """Should return failed status when all folds fail."""
        # Use shorter data range to reduce fold count
        mock_ohlcv_repo.get_available_range = AsyncMock(return_value={
            "min_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "max_ts": datetime(2024, 6, 30, tzinfo=timezone.utc),
            "row_count": 4320,
        })

        # Setup: Return failed jobs when polling
        failed_jobs = [
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.FAILED, payload={}),
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.FAILED, payload={}),
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.FAILED, payload={}),
        ]
        mock_job_repo.list_children = AsyncMock(return_value=failed_jobs)

        async def create_enqueued_job(*args, **kwargs):
            return Job(
                id=uuid4(), type=JobType.TUNE, status=JobStatus.PENDING, payload={}
            )
        mock_job_repo.enqueue = AsyncMock(side_effect=create_enqueued_job)

        ctx = {"pool": mock_pool, "events_repo": mock_events_repo}

        with patch(
            "app.jobs.handlers.wfo.JobRepository", return_value=mock_job_repo
        ), patch(
            "app.jobs.handlers.wfo.OHLCVRepository", return_value=mock_ohlcv_repo
        ), patch(
            "app.jobs.handlers.wfo.TuneRepository", return_value=mock_tune_repo
        ):
            result = await handle_wfo(base_job, ctx)

        assert result["status"] == "failed"
        assert "All folds failed" in result["warnings"]

    @pytest.mark.asyncio
    async def test_returns_partial_when_allow_partial_true(
        self,
        mock_pool,
        mock_events_repo,
        mock_job_repo,
        mock_ohlcv_repo,
        mock_tune_repo,
    ):
        """Should return partial status when some folds fail and allow_partial=True."""
        # Use shorter data range to reduce fold count
        mock_ohlcv_repo.get_available_range = AsyncMock(return_value={
            "min_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "max_ts": datetime(2024, 6, 30, tzinfo=timezone.utc),
            "row_count": 4320,
        })

        # Create job with allow_partial=True
        job = Job(
            id=uuid4(),
            type=JobType.WFO,
            status=JobStatus.RUNNING,
            payload={
                "workspace_id": str(uuid4()),
                "wfo_id": str(uuid4()),
                "strategy_entity_id": str(uuid4()),
                "data_source": {
                    "exchange_id": "kucoin",
                    "symbol": "BTC-USDT",
                    "timeframe": "1h",
                },
                "wfo_config": {
                    "train_days": 90,
                    "test_days": 30,
                    "step_days": 30,
                    "min_folds": 3,
                    "allow_partial": True,
                },
                "param_space": {"lookback": [10, 20, 30]},
            },
        )

        # Setup: Mix of succeeded and failed jobs
        mixed_jobs = [
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.SUCCEEDED, payload={}),
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.FAILED, payload={}),
            Job(id=uuid4(), type=JobType.TUNE, status=JobStatus.SUCCEEDED, payload={}),
        ]
        mock_job_repo.list_children = AsyncMock(return_value=mixed_jobs)

        async def create_enqueued_job(*args, **kwargs):
            return Job(
                id=uuid4(), type=JobType.TUNE, status=JobStatus.PENDING, payload={}
            )
        mock_job_repo.enqueue = AsyncMock(side_effect=create_enqueued_job)

        ctx = {"pool": mock_pool, "events_repo": mock_events_repo}

        with patch(
            "app.jobs.handlers.wfo.JobRepository", return_value=mock_job_repo
        ), patch(
            "app.jobs.handlers.wfo.OHLCVRepository", return_value=mock_ohlcv_repo
        ), patch(
            "app.jobs.handlers.wfo.TuneRepository", return_value=mock_tune_repo
        ):
            result = await handle_wfo(job, ctx)

        assert result["status"] == "partial"
        assert result["folds_completed"] == 2
        assert result["folds_failed"] == 1

    @pytest.mark.asyncio
    async def test_insufficient_data_raises(
        self,
        mock_pool,
        mock_events_repo,
        mock_ohlcv_repo,
    ):
        """Should raise InsufficientDataError when data range too short."""
        # Setup: Return short data range
        mock_ohlcv_repo.get_available_range = AsyncMock(return_value={
            "min_ts": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "max_ts": datetime(2024, 3, 1, tzinfo=timezone.utc),  # Only 60 days
            "row_count": 1440,
        })

        job = Job(
            id=uuid4(),
            type=JobType.WFO,
            status=JobStatus.RUNNING,
            payload={
                "workspace_id": str(uuid4()),
                "wfo_id": str(uuid4()),
                "strategy_entity_id": str(uuid4()),
                "data_source": {
                    "exchange_id": "kucoin",
                    "symbol": "BTC-USDT",
                    "timeframe": "1h",
                },
                "wfo_config": {
                    "train_days": 90,  # Need 120 days per fold
                    "test_days": 30,
                    "step_days": 30,
                    "min_folds": 3,
                },
            },
        )

        ctx = {"pool": mock_pool, "events_repo": mock_events_repo}

        with patch(
            "app.jobs.handlers.wfo.OHLCVRepository", return_value=mock_ohlcv_repo
        ), patch(
            "app.jobs.handlers.wfo.JobRepository"
        ), patch(
            "app.jobs.handlers.wfo.TuneRepository"
        ):
            from app.services.backtest.wfo import InsufficientDataError

            with pytest.raises(InsufficientDataError):
                await handle_wfo(job, ctx)
