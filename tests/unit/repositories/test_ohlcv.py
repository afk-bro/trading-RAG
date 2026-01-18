"""Tests for OHLCV repository."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

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
