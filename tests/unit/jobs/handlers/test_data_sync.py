"""Tests for DataSyncJob handler."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

import pytest

from app.jobs.models import Job
from app.jobs.types import JobType, JobStatus
from app.repositories.core_symbols import CoreSymbol


class TestHandleDataSync:
    """Tests for handle_data_sync handler."""

    @pytest.fixture
    def sample_sync_job(self):
        """Create a sample DATA_SYNC job."""
        return Job(
            id=uuid4(),
            type=JobType.DATA_SYNC,
            status=JobStatus.RUNNING,
            payload={
                "exchange_id": "kucoin",
                "mode": "incremental",
            },
        )

    @pytest.fixture
    def sample_full_sync_job(self):
        """Create a sample DATA_SYNC job in full mode."""
        return Job(
            id=uuid4(),
            type=JobType.DATA_SYNC,
            status=JobStatus.RUNNING,
            payload={
                "mode": "full",  # No exchange_id - sync all exchanges
            },
        )

    @pytest.fixture
    def sample_symbols(self):
        """Create sample core symbols."""
        return [
            CoreSymbol(
                exchange_id="kucoin",
                canonical_symbol="BTC-USDT",
                raw_symbol="BTC-USDT",
                timeframes=["1h", "1d"],
                is_enabled=True,
            ),
            CoreSymbol(
                exchange_id="kucoin",
                canonical_symbol="ETH-USDT",
                raw_symbol="ETH-USDT",
                timeframes=["1h", "1d"],
                is_enabled=True,
            ),
        ]

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        mock_pool = MagicMock()
        mock_events_repo = AsyncMock()
        mock_events_repo.info = AsyncMock()
        mock_events_repo.error = AsyncMock()
        mock_job_repo = AsyncMock()
        mock_job_repo.create = AsyncMock()
        return {
            "pool": mock_pool,
            "events_repo": mock_events_repo,
            "job_repo": mock_job_repo,
            "worker_id": "test-worker",
        }

    @pytest.mark.asyncio
    async def test_handler_success_incremental(
        self, sample_sync_job, mock_context, sample_symbols
    ):
        """Test successful incremental data sync."""
        from app.jobs.handlers.data_sync import handle_data_sync

        # Mock CoreSymbolsRepository
        with patch(
            "app.jobs.handlers.data_sync.CoreSymbolsRepository"
        ) as MockCoreSymbolsRepo:
            mock_symbols_repo = AsyncMock()
            mock_symbols_repo.list_symbols = AsyncMock(return_value=sample_symbols)
            MockCoreSymbolsRepo.return_value = mock_symbols_repo

            # Mock OHLCVRepository for latest timestamps
            with patch("app.jobs.handlers.data_sync.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                # Return some existing data for incremental sync
                mock_ohlcv_repo.get_available_range = AsyncMock(
                    return_value=(
                        datetime(2024, 1, 1, tzinfo=timezone.utc),
                        datetime(2024, 6, 1, tzinfo=timezone.utc),
                    )
                )
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                # Mock settings
                with patch("app.jobs.handlers.data_sync.get_settings") as mock_settings:
                    mock_settings_instance = MagicMock()
                    mock_settings_instance.get_data_sync_history_days.return_value = 730
                    mock_settings.return_value = mock_settings_instance

                    # Create mock jobs for each enqueue call
                    created_jobs = []
                    for i in range(4):  # 2 symbols x 2 timeframes
                        created_job = Job(
                            id=uuid4(),
                            type=JobType.DATA_FETCH,
                            status=JobStatus.PENDING,
                            payload={},
                        )
                        created_jobs.append(created_job)
                    mock_context["job_repo"].create = AsyncMock(
                        side_effect=created_jobs
                    )

                    result = await handle_data_sync(sample_sync_job, mock_context)

                    # Verify result
                    assert result["symbols_processed"] == 2
                    assert result["jobs_enqueued"] == 4  # 2 symbols x 2 timeframes
                    assert result["exchange_id"] == "kucoin"

                    # Verify symbols were listed with exchange filter
                    mock_symbols_repo.list_symbols.assert_called_once_with(
                        exchange_id="kucoin", enabled_only=True
                    )

                    # Verify jobs were created with parent_job_id
                    assert mock_context["job_repo"].create.call_count == 4
                    for call in mock_context["job_repo"].create.call_args_list:
                        kwargs = call[1]
                        assert kwargs["job_type"] == JobType.DATA_FETCH
                        assert kwargs["parent_job_id"] == sample_sync_job.id
                        assert "dedupe_key" in kwargs

    @pytest.mark.asyncio
    async def test_handler_success_full_mode(
        self, sample_full_sync_job, mock_context, sample_symbols
    ):
        """Test successful full data sync (all exchanges)."""
        from app.jobs.handlers.data_sync import handle_data_sync

        with patch(
            "app.jobs.handlers.data_sync.CoreSymbolsRepository"
        ) as MockCoreSymbolsRepo:
            mock_symbols_repo = AsyncMock()
            mock_symbols_repo.list_symbols = AsyncMock(return_value=sample_symbols)
            MockCoreSymbolsRepo.return_value = mock_symbols_repo

            with patch("app.jobs.handlers.data_sync.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                # Return None for full sync (no existing data)
                mock_ohlcv_repo.get_available_range = AsyncMock(return_value=None)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch("app.jobs.handlers.data_sync.get_settings") as mock_settings:
                    mock_settings_instance = MagicMock()
                    mock_settings_instance.get_data_sync_history_days.return_value = 730
                    mock_settings.return_value = mock_settings_instance

                    # Create mock jobs
                    created_jobs = []
                    for i in range(4):
                        created_job = Job(
                            id=uuid4(),
                            type=JobType.DATA_FETCH,
                            status=JobStatus.PENDING,
                            payload={},
                        )
                        created_jobs.append(created_job)
                    mock_context["job_repo"].create = AsyncMock(
                        side_effect=created_jobs
                    )

                    result = await handle_data_sync(sample_full_sync_job, mock_context)

                    # Verify result
                    assert result["symbols_processed"] == 2
                    assert result["jobs_enqueued"] == 4
                    assert result["exchange_id"] == "all"

                    # Verify symbols were listed without exchange filter
                    mock_symbols_repo.list_symbols.assert_called_once_with(
                        exchange_id=None, enabled_only=True
                    )

    @pytest.mark.asyncio
    async def test_handler_no_symbols(self, sample_sync_job, mock_context):
        """Test handler with no enabled symbols."""
        from app.jobs.handlers.data_sync import handle_data_sync

        with patch(
            "app.jobs.handlers.data_sync.CoreSymbolsRepository"
        ) as MockCoreSymbolsRepo:
            mock_symbols_repo = AsyncMock()
            mock_symbols_repo.list_symbols = AsyncMock(return_value=[])
            MockCoreSymbolsRepo.return_value = mock_symbols_repo

            with patch("app.jobs.handlers.data_sync.get_settings") as mock_settings:
                mock_settings_instance = MagicMock()
                mock_settings.return_value = mock_settings_instance

                result = await handle_data_sync(sample_sync_job, mock_context)

                assert result["symbols_processed"] == 0
                assert result["jobs_enqueued"] == 0
                mock_context["job_repo"].create.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_incremental_uses_latest_timestamp(
        self, sample_sync_job, mock_context
    ):
        """Test that incremental mode uses latest data timestamp as start."""
        from app.jobs.handlers.data_sync import handle_data_sync

        single_symbol = [
            CoreSymbol(
                exchange_id="kucoin",
                canonical_symbol="BTC-USDT",
                raw_symbol="BTC-USDT",
                timeframes=["1h"],
                is_enabled=True,
            ),
        ]

        with patch(
            "app.jobs.handlers.data_sync.CoreSymbolsRepository"
        ) as MockCoreSymbolsRepo:
            mock_symbols_repo = AsyncMock()
            mock_symbols_repo.list_symbols = AsyncMock(return_value=single_symbol)
            MockCoreSymbolsRepo.return_value = mock_symbols_repo

            with patch("app.jobs.handlers.data_sync.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                latest_ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
                mock_ohlcv_repo.get_available_range = AsyncMock(
                    return_value=(
                        datetime(2024, 1, 1, tzinfo=timezone.utc),
                        latest_ts,
                    )
                )
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch("app.jobs.handlers.data_sync.get_settings") as mock_settings:
                    mock_settings_instance = MagicMock()
                    mock_settings_instance.get_data_sync_history_days.return_value = 730
                    mock_settings.return_value = mock_settings_instance

                    created_job = Job(
                        id=uuid4(),
                        type=JobType.DATA_FETCH,
                        status=JobStatus.PENDING,
                        payload={},
                    )
                    mock_context["job_repo"].create = AsyncMock(
                        return_value=created_job
                    )

                    await handle_data_sync(sample_sync_job, mock_context)

                    # Verify the job payload uses latest_ts as start
                    call_kwargs = mock_context["job_repo"].create.call_args[1]
                    payload = call_kwargs["payload"]
                    assert payload["start_ts"] == latest_ts.isoformat()

    @pytest.mark.asyncio
    async def test_handler_full_mode_uses_history_window(
        self, sample_full_sync_job, mock_context
    ):
        """Test that full mode uses history window from settings."""
        from app.jobs.handlers.data_sync import handle_data_sync

        single_symbol = [
            CoreSymbol(
                exchange_id="kucoin",
                canonical_symbol="BTC-USDT",
                raw_symbol="BTC-USDT",
                timeframes=["1d"],
                is_enabled=True,
            ),
        ]

        with patch(
            "app.jobs.handlers.data_sync.CoreSymbolsRepository"
        ) as MockCoreSymbolsRepo:
            mock_symbols_repo = AsyncMock()
            mock_symbols_repo.list_symbols = AsyncMock(return_value=single_symbol)
            MockCoreSymbolsRepo.return_value = mock_symbols_repo

            with patch("app.jobs.handlers.data_sync.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.get_available_range = AsyncMock(return_value=None)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch("app.jobs.handlers.data_sync.get_settings") as mock_settings:
                    mock_settings_instance = MagicMock()
                    # 1825 days for 1d timeframe (5 years)
                    mock_settings_instance.get_data_sync_history_days.return_value = (
                        1825
                    )
                    mock_settings.return_value = mock_settings_instance

                    created_job = Job(
                        id=uuid4(),
                        type=JobType.DATA_FETCH,
                        status=JobStatus.PENDING,
                        payload={},
                    )
                    mock_context["job_repo"].create = AsyncMock(
                        return_value=created_job
                    )

                    with patch("app.jobs.handlers.data_sync.datetime") as mock_datetime:
                        # Fix "now" for deterministic test
                        fixed_now = datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
                        mock_datetime.now.return_value = fixed_now
                        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

                        await handle_data_sync(sample_full_sync_job, mock_context)

                        # Verify settings method was called with correct timeframe
                        mock_settings_instance.get_data_sync_history_days.assert_called_with(
                            "1d"
                        )

                        # Verify the job payload
                        call_kwargs = mock_context["job_repo"].create.call_args[1]
                        payload = call_kwargs["payload"]
                        assert payload["timeframe"] == "1d"

    @pytest.mark.asyncio
    async def test_handler_dedupe_key_format(self, sample_sync_job, mock_context):
        """Test that dedupe keys have correct format."""
        from app.jobs.handlers.data_sync import handle_data_sync

        single_symbol = [
            CoreSymbol(
                exchange_id="kucoin",
                canonical_symbol="BTC-USDT",
                raw_symbol="BTC-USDT",
                timeframes=["1h"],
                is_enabled=True,
            ),
        ]

        with patch(
            "app.jobs.handlers.data_sync.CoreSymbolsRepository"
        ) as MockCoreSymbolsRepo:
            mock_symbols_repo = AsyncMock()
            mock_symbols_repo.list_symbols = AsyncMock(return_value=single_symbol)
            MockCoreSymbolsRepo.return_value = mock_symbols_repo

            with patch("app.jobs.handlers.data_sync.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.get_available_range = AsyncMock(return_value=None)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch("app.jobs.handlers.data_sync.get_settings") as mock_settings:
                    mock_settings_instance = MagicMock()
                    mock_settings_instance.get_data_sync_history_days.return_value = 730
                    mock_settings.return_value = mock_settings_instance

                    created_job = Job(
                        id=uuid4(),
                        type=JobType.DATA_FETCH,
                        status=JobStatus.PENDING,
                        payload={},
                    )
                    mock_context["job_repo"].create = AsyncMock(
                        return_value=created_job
                    )

                    await handle_data_sync(sample_sync_job, mock_context)

                    # Verify dedupe key format: data_fetch:{exchange}:{symbol}:{tf}:{date}
                    call_kwargs = mock_context["job_repo"].create.call_args[1]
                    dedupe_key = call_kwargs["dedupe_key"]
                    assert dedupe_key.startswith("data_fetch:kucoin:BTC-USDT:1h:")
                    # Should end with YYYY-MM-DD date format
                    date_part = dedupe_key.split(":")[-1]
                    assert len(date_part) == 10  # YYYY-MM-DD

    @pytest.mark.asyncio
    async def test_handler_logs_events(
        self, sample_sync_job, mock_context, sample_symbols
    ):
        """Test that handler logs events via events_repo."""
        from app.jobs.handlers.data_sync import handle_data_sync

        with patch(
            "app.jobs.handlers.data_sync.CoreSymbolsRepository"
        ) as MockCoreSymbolsRepo:
            mock_symbols_repo = AsyncMock()
            mock_symbols_repo.list_symbols = AsyncMock(return_value=sample_symbols)
            MockCoreSymbolsRepo.return_value = mock_symbols_repo

            with patch("app.jobs.handlers.data_sync.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.get_available_range = AsyncMock(return_value=None)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch("app.jobs.handlers.data_sync.get_settings") as mock_settings:
                    mock_settings_instance = MagicMock()
                    mock_settings_instance.get_data_sync_history_days.return_value = 730
                    mock_settings.return_value = mock_settings_instance

                    created_jobs = []
                    for _ in range(4):
                        created_jobs.append(
                            Job(
                                id=uuid4(),
                                type=JobType.DATA_FETCH,
                                status=JobStatus.PENDING,
                                payload={},
                            )
                        )
                    mock_context["job_repo"].create = AsyncMock(
                        side_effect=created_jobs
                    )

                    await handle_data_sync(sample_sync_job, mock_context)

                    # Should have logged info events
                    events_repo = mock_context["events_repo"]
                    assert events_repo.info.call_count >= 2  # Start and completion

                    # Check event messages
                    call_args_list = [
                        call[0][1] for call in events_repo.info.call_args_list
                    ]
                    assert any("sync" in msg.lower() for msg in call_args_list)
                    assert any(
                        "enqueued" in msg.lower() or "jobs" in msg.lower()
                        for msg in call_args_list
                    )


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_handler_is_registered(self):
        """Test that handle_data_sync is registered with the registry."""
        from app.jobs.registry import default_registry
        from app.jobs.types import JobType
        from app.jobs.handlers.data_sync import handle_data_sync

        handler = default_registry.get_handler(JobType.DATA_SYNC)
        assert handler is handle_data_sync


class TestDedupeKeyGeneration:
    """Tests for dedupe key generation utility."""

    def test_generate_dedupe_key(self):
        """Test dedupe key generation."""
        from app.jobs.handlers.data_sync import generate_dedupe_key

        key = generate_dedupe_key(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            date=datetime(2024, 7, 15, tzinfo=timezone.utc),
        )
        assert key == "data_fetch:kucoin:BTC-USDT:1h:2024-07-15"

    def test_generate_dedupe_key_different_dates(self):
        """Test that different dates produce different keys."""
        from app.jobs.handlers.data_sync import generate_dedupe_key

        key1 = generate_dedupe_key(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            date=datetime(2024, 7, 15, tzinfo=timezone.utc),
        )
        key2 = generate_dedupe_key(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            date=datetime(2024, 7, 16, tzinfo=timezone.utc),
        )
        assert key1 != key2
