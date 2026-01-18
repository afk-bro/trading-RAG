"""Tests for ensure_ohlcv_range function."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.ohlcv import Candle
from app.services.market_data.base import MarketDataCandle
from app.services.market_data.ensure_range import (
    EnsureRangeResult,
    ensure_ohlcv_range,
)


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    """Create UTC datetime helper."""
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _make_candle(
    ts: datetime, price: float = 100.0, exchange_id: str = "kucoin"
) -> Candle:
    """Create test candle."""
    return Candle(
        exchange_id=exchange_id,
        symbol="BTC-USDT",
        timeframe="1d",
        ts=ts,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1000.0,
    )


def _make_market_candle(ts: datetime, price: float = 100.0) -> MarketDataCandle:
    """Create test market data candle."""
    return MarketDataCandle(
        ts=ts,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1000.0,
    )


class TestEnsureOhlcvRangeDataFullyCached:
    """Test case: data is fully cached (no fetch needed)."""

    @pytest.mark.asyncio
    async def test_data_fully_cached_returns_was_cached_true(self):
        """When all requested data is cached, return was_cached=True."""
        # Setup: existing range covers requested range
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        mock_repo.get_available_range.return_value = (
            _utc(2024, 1, 1),  # min_ts
            _utc(2024, 12, 31),  # max_ts
        )
        mock_repo.count_in_range.return_value = 365

        mock_provider_class = MagicMock()
        mock_provider = AsyncMock()
        mock_provider_class.return_value = mock_provider

        # Request a subset of existing data
        result = await ensure_ohlcv_range(
            pool=mock_pool,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1d",
            start_ts=_utc(2024, 6, 1),
            end_ts=_utc(2024, 6, 30),
            _repo=mock_repo,
            _provider_class=mock_provider_class,
        )

        assert result.was_cached is True
        assert result.fetched_candles == 0
        assert result.gaps_filled == []
        # Provider should not be instantiated
        mock_provider_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_data_fully_cached_returns_total_candle_count(self):
        """When cached, total_candles reflects count from repo."""
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        mock_repo.get_available_range.return_value = (
            _utc(2024, 1, 1),
            _utc(2024, 12, 31),
        )
        mock_repo.count_in_range.return_value = 30  # 30 days of data

        mock_provider_class = MagicMock()

        result = await ensure_ohlcv_range(
            pool=mock_pool,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1d",
            start_ts=_utc(2024, 6, 1),
            end_ts=_utc(2024, 7, 1),
            _repo=mock_repo,
            _provider_class=mock_provider_class,
        )

        assert result.total_candles == 30


class TestEnsureOhlcvRangeNoExistingData:
    """Test case: no existing data (fetch entire range)."""

    @pytest.mark.asyncio
    async def test_no_existing_data_fetches_entire_range(self):
        """When no data exists, fetch the entire requested range."""
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        mock_repo.get_available_range.return_value = None  # No existing data
        mock_repo.upsert_candles.return_value = 10
        mock_repo.count_in_range.return_value = 10

        # Provider returns candles
        mock_provider_class = MagicMock()
        mock_provider = AsyncMock()
        mock_provider.fetch_ohlcv.return_value = [
            _make_market_candle(_utc(2024, 6, d + 1)) for d in range(10)
        ]
        mock_provider_class.return_value = mock_provider

        start = _utc(2024, 6, 1)
        end = _utc(2024, 6, 11)

        result = await ensure_ohlcv_range(
            pool=mock_pool,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1d",
            start_ts=start,
            end_ts=end,
            _repo=mock_repo,
            _provider_class=mock_provider_class,
        )

        assert result.was_cached is False
        assert result.fetched_candles == 10
        assert result.gaps_filled == [(start, end)]
        assert result.total_candles == 10
        # Provider should be called once for the entire range
        mock_provider.fetch_ohlcv.assert_called_once_with("BTC-USDT", "1d", start, end)
        # Provider should be closed
        mock_provider.close.assert_called_once()


class TestEnsureOhlcvRangeMissingHeadOnly:
    """Test case: missing head only (data starts after requested start)."""

    @pytest.mark.asyncio
    async def test_missing_head_fetches_head_gap(self):
        """When existing data starts after requested start, fetch the head gap."""
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        # Existing data from June 15 to June 30
        mock_repo.get_available_range.return_value = (
            _utc(2024, 6, 15),
            _utc(2024, 6, 30),
        )
        mock_repo.upsert_candles.return_value = 14
        mock_repo.count_in_range.return_value = 30  # After fetch

        # Provider returns candles for the head gap
        mock_provider_class = MagicMock()
        mock_provider = AsyncMock()
        mock_provider.fetch_ohlcv.return_value = [
            _make_market_candle(_utc(2024, 6, d + 1)) for d in range(14)
        ]
        mock_provider_class.return_value = mock_provider

        # Request June 1 to June 30 (head missing: June 1-14)
        result = await ensure_ohlcv_range(
            pool=mock_pool,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1d",
            start_ts=_utc(2024, 6, 1),
            end_ts=_utc(2024, 6, 30),
            _repo=mock_repo,
            _provider_class=mock_provider_class,
        )

        assert result.was_cached is False
        assert result.fetched_candles == 14
        # Gap should be from requested start to existing min
        assert result.gaps_filled == [(_utc(2024, 6, 1), _utc(2024, 6, 15))]
        mock_provider.fetch_ohlcv.assert_called_once_with(
            "BTC-USDT", "1d", _utc(2024, 6, 1), _utc(2024, 6, 15)
        )


class TestEnsureOhlcvRangeMissingTailOnly:
    """Test case: missing tail only (data ends before requested end)."""

    @pytest.mark.asyncio
    async def test_missing_tail_fetches_tail_gap(self):
        """When existing data ends before requested end, fetch the tail gap."""
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        # Existing data from June 1 to June 15
        mock_repo.get_available_range.return_value = (
            _utc(2024, 6, 1),
            _utc(2024, 6, 15),
        )
        mock_repo.upsert_candles.return_value = 15
        mock_repo.count_in_range.return_value = 30  # After fetch

        # Provider returns candles for the tail gap
        mock_provider_class = MagicMock()
        mock_provider = AsyncMock()
        mock_provider.fetch_ohlcv.return_value = [
            _make_market_candle(_utc(2024, 6, d + 16)) for d in range(15)
        ]
        mock_provider_class.return_value = mock_provider

        # Request June 1 to June 30 (tail missing: June 16-30)
        result = await ensure_ohlcv_range(
            pool=mock_pool,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1d",
            start_ts=_utc(2024, 6, 1),
            end_ts=_utc(2024, 6, 30),
            _repo=mock_repo,
            _provider_class=mock_provider_class,
        )

        assert result.was_cached is False
        assert result.fetched_candles == 15
        # Gap should be from existing max to requested end
        # Note: We use max_ts + 1 timeframe unit as the gap start to avoid overlap
        # For 1d, the gap starts at max_ts + 1 day
        assert len(result.gaps_filled) == 1
        gap_start, gap_end = result.gaps_filled[0]
        assert gap_end == _utc(2024, 6, 30)


class TestEnsureOhlcvRangeMissingBothHeadAndTail:
    """Test case: missing both head and tail."""

    @pytest.mark.asyncio
    async def test_missing_both_head_and_tail(self):
        """When existing data is a subset, fetch both head and tail gaps."""
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        # Existing data from June 10 to June 20
        mock_repo.get_available_range.return_value = (
            _utc(2024, 6, 10),
            _utc(2024, 6, 20),
        )
        mock_repo.upsert_candles.return_value = 9  # per call
        mock_repo.count_in_range.return_value = 30  # After all fetches

        # Provider returns candles
        mock_provider_class = MagicMock()
        mock_provider = AsyncMock()
        # First call (head), second call (tail)
        mock_provider.fetch_ohlcv.side_effect = [
            [_make_market_candle(_utc(2024, 6, d + 1)) for d in range(9)],  # head
            [_make_market_candle(_utc(2024, 6, d + 21)) for d in range(10)],  # tail
        ]
        mock_provider_class.return_value = mock_provider

        # Request June 1 to June 30
        result = await ensure_ohlcv_range(
            pool=mock_pool,
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1d",
            start_ts=_utc(2024, 6, 1),
            end_ts=_utc(2024, 6, 30),
            _repo=mock_repo,
            _provider_class=mock_provider_class,
        )

        assert result.was_cached is False
        assert result.fetched_candles == 19  # 9 head + 10 tail
        assert len(result.gaps_filled) == 2


class TestEnsureOhlcvRangeProviderFailure:
    """Test case: provider failure handling."""

    @pytest.mark.asyncio
    async def test_provider_failure_raises_exception(self):
        """When provider fails, the exception should propagate."""
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        mock_repo.get_available_range.return_value = None  # No existing data

        # Provider raises an exception
        mock_provider_class = MagicMock()
        mock_provider = AsyncMock()
        mock_provider.fetch_ohlcv.side_effect = Exception("CCXT API Error")
        mock_provider_class.return_value = mock_provider

        with pytest.raises(Exception, match="CCXT API Error"):
            await ensure_ohlcv_range(
                pool=mock_pool,
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1d",
                start_ts=_utc(2024, 6, 1),
                end_ts=_utc(2024, 6, 30),
                _repo=mock_repo,
                _provider_class=mock_provider_class,
            )

        # Provider should still be closed even on failure
        mock_provider.close.assert_called_once()


class TestEnsureRangeResult:
    """Tests for EnsureRangeResult dataclass."""

    def test_result_attributes(self):
        """Test result dataclass has all expected attributes."""
        result = EnsureRangeResult(
            total_candles=100,
            fetched_candles=50,
            gaps_filled=[(_utc(2024, 1, 1), _utc(2024, 1, 15))],
            was_cached=False,
        )

        assert result.total_candles == 100
        assert result.fetched_candles == 50
        assert result.gaps_filled == [(_utc(2024, 1, 1), _utc(2024, 1, 15))]
        assert result.was_cached is False

    def test_result_was_cached_true(self):
        """Test result when data was fully cached."""
        result = EnsureRangeResult(
            total_candles=100,
            fetched_candles=0,
            gaps_filled=[],
            was_cached=True,
        )

        assert result.was_cached is True
        assert result.fetched_candles == 0
        assert result.gaps_filled == []


class TestEnsureOhlcvRangeCandleConversion:
    """Test case: MarketDataCandle to Candle conversion."""

    @pytest.mark.asyncio
    async def test_candles_converted_with_exchange_symbol_timeframe(self):
        """Fetched candles should be converted with exchange/symbol/timeframe."""
        mock_pool = MagicMock()
        mock_repo = AsyncMock()
        mock_repo.get_available_range.return_value = None

        captured_candles = []

        async def capture_upsert(candles):
            captured_candles.extend(candles)
            return len(candles)

        mock_repo.upsert_candles.side_effect = capture_upsert
        mock_repo.count_in_range.return_value = 2

        mock_provider_class = MagicMock()
        mock_provider = AsyncMock()
        mock_provider.fetch_ohlcv.return_value = [
            _make_market_candle(_utc(2024, 6, 1), 50000.0),
            _make_market_candle(_utc(2024, 6, 2), 51000.0),
        ]
        mock_provider_class.return_value = mock_provider

        await ensure_ohlcv_range(
            pool=mock_pool,
            exchange_id="binance",
            symbol="ETH-USDT",
            timeframe="1h",
            start_ts=_utc(2024, 6, 1),
            end_ts=_utc(2024, 6, 3),
            _repo=mock_repo,
            _provider_class=mock_provider_class,
        )

        # Verify candles were converted correctly
        assert len(captured_candles) == 2
        for candle in captured_candles:
            assert candle.exchange_id == "binance"
            assert candle.symbol == "ETH-USDT"
            assert candle.timeframe == "1h"
