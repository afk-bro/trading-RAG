"""Tests for DataFetchJob handler."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

import pytest

from app.jobs.models import Job
from app.jobs.types import JobType, JobStatus
from app.jobs.handlers.data_fetch import handle_data_fetch, parse_iso_timestamp
from app.services.market_data.base import MarketDataCandle


class TestParseIsoTimestamp:
    """Tests for ISO timestamp parsing."""

    def test_parse_z_suffix(self):
        """Test parsing ISO timestamp with Z suffix."""
        result = parse_iso_timestamp("2024-01-01T00:00:00Z")
        assert result == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_parse_utc_offset(self):
        """Test parsing ISO timestamp with +00:00 offset."""
        result = parse_iso_timestamp("2024-01-01T00:00:00+00:00")
        assert result == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def test_parse_no_timezone(self):
        """Test parsing ISO timestamp without timezone (assumes UTC)."""
        result = parse_iso_timestamp("2024-01-01T00:00:00")
        assert result == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


class TestHandleDataFetch:
    """Tests for handle_data_fetch handler."""

    @pytest.fixture
    def sample_job(self):
        """Create a sample DATA_FETCH job."""
        return Job(
            id=uuid4(),
            type=JobType.DATA_FETCH,
            status=JobStatus.RUNNING,
            payload={
                "exchange_id": "kucoin",
                "symbol": "BTC-USDT",
                "timeframe": "1h",
                "start_ts": "2024-01-01T00:00:00Z",
                "end_ts": "2024-07-01T00:00:00Z",
            },
        )

    @pytest.fixture
    def mock_context(self):
        """Create a mock execution context."""
        mock_pool = MagicMock()
        mock_events_repo = AsyncMock()
        mock_events_repo.info = AsyncMock()
        mock_events_repo.error = AsyncMock()
        return {
            "pool": mock_pool,
            "events_repo": mock_events_repo,
            "worker_id": "test-worker",
        }

    @pytest.fixture
    def sample_candles(self):
        """Create sample MarketDataCandles."""
        return [
            MarketDataCandle(
                ts=datetime(2024, 1, 1, i + 1, 0, 0, tzinfo=timezone.utc),
                open=42000.0 + i * 10,
                high=42500.0 + i * 10,
                low=41800.0 + i * 10,
                close=42200.0 + i * 10,
                volume=100.0 + i,
            )
            for i in range(10)
        ]

    @pytest.mark.asyncio
    async def test_handler_success(self, sample_job, mock_context, sample_candles):
        """Test successful data fetch execution."""
        # Mock CCXT provider
        with patch(
            "app.jobs.handlers.data_fetch.CcxtMarketDataProvider"
        ) as MockProvider:
            mock_provider = AsyncMock()
            mock_provider.fetch_ohlcv = AsyncMock(return_value=sample_candles)
            mock_provider.close = AsyncMock()
            MockProvider.return_value = mock_provider

            # Mock OHLCV repository
            with patch("app.jobs.handlers.data_fetch.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.upsert_candles = AsyncMock(return_value=10)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                # Mock DataRevision repository
                with patch(
                    "app.jobs.handlers.data_fetch.DataRevisionRepository"
                ) as MockRevisionRepo:
                    mock_revision_repo = AsyncMock()
                    mock_revision_repo.upsert = AsyncMock()
                    MockRevisionRepo.return_value = mock_revision_repo

                    # Execute handler
                    result = await handle_data_fetch(sample_job, mock_context)

                    # Verify result
                    assert result["candles_fetched"] == 10
                    assert result["candles_upserted"] == 10
                    assert "checksum" in result
                    assert len(result["checksum"]) == 16  # SHA256 truncated to 16 chars

                    # Verify provider was called correctly
                    MockProvider.assert_called_once_with("kucoin")
                    mock_provider.fetch_ohlcv.assert_called_once_with(
                        symbol="BTC-USDT",
                        timeframe="1h",
                        start_ts=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                        end_ts=datetime(2024, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
                    )
                    mock_provider.close.assert_called_once()

                    # Verify candles were upserted
                    mock_ohlcv_repo.upsert_candles.assert_called_once()
                    upserted_candles = mock_ohlcv_repo.upsert_candles.call_args[0][0]
                    assert len(upserted_candles) == 10
                    assert upserted_candles[0].exchange_id == "kucoin"
                    assert upserted_candles[0].symbol == "BTC-USDT"
                    assert upserted_candles[0].timeframe == "1h"

                    # Verify revision was saved
                    mock_revision_repo.upsert.assert_called_once()
                    call_kwargs = mock_revision_repo.upsert.call_args[1]
                    assert call_kwargs["exchange_id"] == "kucoin"
                    assert call_kwargs["symbol"] == "BTC-USDT"
                    assert call_kwargs["timeframe"] == "1h"
                    assert call_kwargs["row_count"] == 10
                    assert call_kwargs["job_id"] == str(sample_job.id)

    @pytest.mark.asyncio
    async def test_handler_missing_payload_fields(self, mock_context):
        """Test handler raises ValueError for missing payload fields."""
        job = Job(
            id=uuid4(),
            type=JobType.DATA_FETCH,
            status=JobStatus.RUNNING,
            payload={
                "exchange_id": "kucoin",
                # Missing symbol, timeframe, timestamps
            },
        )

        with pytest.raises(ValueError, match="Missing required payload fields"):
            await handle_data_fetch(job, mock_context)

    @pytest.mark.asyncio
    async def test_handler_empty_result(self, sample_job, mock_context):
        """Test handler with no candles returned."""
        with patch(
            "app.jobs.handlers.data_fetch.CcxtMarketDataProvider"
        ) as MockProvider:
            mock_provider = AsyncMock()
            mock_provider.fetch_ohlcv = AsyncMock(return_value=[])
            mock_provider.close = AsyncMock()
            MockProvider.return_value = mock_provider

            with patch("app.jobs.handlers.data_fetch.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.upsert_candles = AsyncMock(return_value=0)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch(
                    "app.jobs.handlers.data_fetch.DataRevisionRepository"
                ) as MockRevisionRepo:
                    mock_revision_repo = AsyncMock()
                    mock_revision_repo.upsert = AsyncMock()
                    MockRevisionRepo.return_value = mock_revision_repo

                    result = await handle_data_fetch(sample_job, mock_context)

                    assert result["candles_fetched"] == 0
                    assert result["candles_upserted"] == 0
                    assert result["checksum"] == "empty"  # Empty checksum

    @pytest.mark.asyncio
    async def test_handler_provider_close_on_exception(self, sample_job, mock_context):
        """Test that provider is closed even when fetch fails."""
        with patch(
            "app.jobs.handlers.data_fetch.CcxtMarketDataProvider"
        ) as MockProvider:
            mock_provider = AsyncMock()
            mock_provider.fetch_ohlcv = AsyncMock(side_effect=Exception("API error"))
            mock_provider.close = AsyncMock()
            MockProvider.return_value = mock_provider

            with pytest.raises(Exception, match="API error"):
                await handle_data_fetch(sample_job, mock_context)

            # Provider should still be closed
            mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_logs_events(self, sample_job, mock_context, sample_candles):
        """Test that handler logs events via events_repo."""
        with patch(
            "app.jobs.handlers.data_fetch.CcxtMarketDataProvider"
        ) as MockProvider:
            mock_provider = AsyncMock()
            mock_provider.fetch_ohlcv = AsyncMock(return_value=sample_candles)
            mock_provider.close = AsyncMock()
            MockProvider.return_value = mock_provider

            with patch("app.jobs.handlers.data_fetch.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.upsert_candles = AsyncMock(return_value=10)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch(
                    "app.jobs.handlers.data_fetch.DataRevisionRepository"
                ) as MockRevisionRepo:
                    mock_revision_repo = AsyncMock()
                    mock_revision_repo.upsert = AsyncMock()
                    MockRevisionRepo.return_value = mock_revision_repo

                    await handle_data_fetch(sample_job, mock_context)

                    # Should have logged multiple info events
                    events_repo = mock_context["events_repo"]
                    assert events_repo.info.call_count >= 4

                    # Check event messages
                    call_args_list = [
                        call[0][1] for call in events_repo.info.call_args_list
                    ]
                    assert any("Fetching" in msg for msg in call_args_list)
                    assert any("Received" in msg for msg in call_args_list)
                    assert any("Upserted" in msg for msg in call_args_list)
                    assert any("revision" in msg.lower() for msg in call_args_list)

    @pytest.mark.asyncio
    async def test_handler_converts_candles_correctly(
        self, sample_job, mock_context, sample_candles
    ):
        """Test that MarketDataCandle is correctly converted to repository Candle."""
        with patch(
            "app.jobs.handlers.data_fetch.CcxtMarketDataProvider"
        ) as MockProvider:
            mock_provider = AsyncMock()
            mock_provider.fetch_ohlcv = AsyncMock(return_value=sample_candles)
            mock_provider.close = AsyncMock()
            MockProvider.return_value = mock_provider

            with patch("app.jobs.handlers.data_fetch.OHLCVRepository") as MockOHLCVRepo:
                mock_ohlcv_repo = AsyncMock()
                mock_ohlcv_repo.upsert_candles = AsyncMock(return_value=10)
                MockOHLCVRepo.return_value = mock_ohlcv_repo

                with patch(
                    "app.jobs.handlers.data_fetch.DataRevisionRepository"
                ) as MockRevisionRepo:
                    mock_revision_repo = AsyncMock()
                    mock_revision_repo.upsert = AsyncMock()
                    MockRevisionRepo.return_value = mock_revision_repo

                    await handle_data_fetch(sample_job, mock_context)

                    # Check converted candles
                    upserted_candles = mock_ohlcv_repo.upsert_candles.call_args[0][0]

                    # First candle should match first MarketDataCandle
                    first_candle = upserted_candles[0]
                    first_market = sample_candles[0]

                    assert first_candle.exchange_id == "kucoin"
                    assert first_candle.symbol == "BTC-USDT"
                    assert first_candle.timeframe == "1h"
                    assert first_candle.ts == first_market.ts
                    assert first_candle.open == first_market.open
                    assert first_candle.high == first_market.high
                    assert first_candle.low == first_market.low
                    assert first_candle.close == first_market.close
                    assert first_candle.volume == first_market.volume


class TestHandlerRegistration:
    """Tests for handler registration."""

    def test_handler_is_registered(self):
        """Test that handle_data_fetch is registered with the registry."""
        from app.jobs.registry import default_registry
        from app.jobs.types import JobType

        handler = default_registry.get_handler(JobType.DATA_FETCH)
        assert handler is handle_data_fetch
