"""Tests for OHLCV repository."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.repositories.ohlcv import OHLCVRepository, Candle


class TestCandle:
    def test_candle_creation(self):
        candle = Candle(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=100.5,
        )
        assert candle.symbol == "BTC-USDT"
        assert candle.close == 42200.0

    def test_candle_ohlc_validation(self):
        # high >= all others
        with pytest.raises(ValueError):
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime.now(timezone.utc),
                open=42000.0,
                high=41000.0,  # Invalid: high < open
                low=41800.0,
                close=42200.0,
                volume=100.0,
            )

    def test_candle_low_validation(self):
        # low <= all others
        with pytest.raises(ValueError):
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime.now(timezone.utc),
                open=42000.0,
                high=43000.0,
                low=42500.0,  # Invalid: low > open and close
                close=42200.0,
                volume=100.0,
            )

    def test_candle_volume_validation(self):
        # volume >= 0
        with pytest.raises(ValueError):
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime.now(timezone.utc),
                open=42000.0,
                high=43000.0,
                low=41000.0,
                close=42200.0,
                volume=-1.0,  # Invalid: negative volume
            )

    def test_candle_zero_volume_valid(self):
        # Zero volume is valid (e.g., for illiquid markets)
        candle = Candle(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            ts=datetime.now(timezone.utc),
            open=42000.0,
            high=42000.0,
            low=42000.0,
            close=42000.0,
            volume=0.0,
        )
        assert candle.volume == 0.0


class TestOHLCVRepository:
    @pytest.fixture
    def mock_pool(self):
        return MagicMock()

    def test_repository_creation(self, mock_pool):
        repo = OHLCVRepository(mock_pool)
        assert repo._pool == mock_pool

    @pytest.mark.asyncio
    async def test_upsert_candles_empty_list(self, mock_pool):
        """Test upserting empty list returns 0."""
        repo = OHLCVRepository(mock_pool)
        result = await repo.upsert_candles([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_upsert_candles_single(self, mock_pool):
        """Test upserting a single candle."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        candles = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                open=42000.0,
                high=42500.0,
                low=41800.0,
                close=42200.0,
                volume=100.5,
            )
        ]

        result = await repo.upsert_candles(candles)

        assert result == 1
        mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_candles_multiple(self, mock_pool):
        """Test upserting multiple candles."""
        mock_conn = AsyncMock()
        mock_conn.executemany = AsyncMock()

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        candles = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                open=42000.0 + i * 100,
                high=42500.0 + i * 100,
                low=41800.0 + i * 100,
                close=42200.0 + i * 100,
                volume=100.5,
            )
            for i in range(5)
        ]

        result = await repo.upsert_candles(candles)

        assert result == 5
        mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_range(self, mock_pool):
        """Test getting candles in a time range."""
        start_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_ts = datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc)

        mock_rows = [
            {
                "exchange_id": "kucoin",
                "symbol": "BTC-USDT",
                "timeframe": "1h",
                "ts": datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                "open": 42000.0,
                "high": 42500.0,
                "low": 41800.0,
                "close": 42200.0,
                "volume": 100.5,
            }
            for i in range(5)
        ]

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_rows)

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        candles = await repo.get_range("kucoin", "BTC-USDT", "1h", start_ts, end_ts)

        assert len(candles) == 5
        assert all(isinstance(c, Candle) for c in candles)
        assert candles[0].symbol == "BTC-USDT"

    @pytest.mark.asyncio
    async def test_get_range_empty(self, mock_pool):
        """Test getting candles when range is empty."""
        start_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_ts = datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc)

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        candles = await repo.get_range("kucoin", "BTC-USDT", "1h", start_ts, end_ts)

        assert candles == []

    @pytest.mark.asyncio
    async def test_get_available_range(self, mock_pool):
        """Test getting min/max timestamp range."""
        min_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        max_ts = datetime(2024, 1, 31, 23, 0, tzinfo=timezone.utc)

        mock_row = {"min_ts": min_ts, "max_ts": max_ts}

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        result = await repo.get_available_range("kucoin", "BTC-USDT", "1h")

        assert result is not None
        assert result[0] == min_ts
        assert result[1] == max_ts

    @pytest.mark.asyncio
    async def test_get_available_range_none(self, mock_pool):
        """Test getting range when no data exists."""
        mock_row = {"min_ts": None, "max_ts": None}

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        result = await repo.get_available_range("kucoin", "BTC-USDT", "1h")

        assert result is None

    @pytest.mark.asyncio
    async def test_count_in_range(self, mock_pool):
        """Test counting candles in a range."""
        start_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_ts = datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc)

        mock_row = {"cnt": 42}

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        count = await repo.count_in_range("kucoin", "BTC-USDT", "1h", start_ts, end_ts)

        assert count == 42

    @pytest.mark.asyncio
    async def test_count_in_range_zero(self, mock_pool):
        """Test counting when range has no candles."""
        start_ts = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_ts = datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc)

        mock_row = {"cnt": 0}

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)

        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        repo = OHLCVRepository(mock_pool)
        count = await repo.count_in_range("kucoin", "BTC-USDT", "1h", start_ts, end_ts)

        assert count == 0
