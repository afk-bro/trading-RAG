"""Tests for data revision computation."""

from datetime import datetime, timezone

from app.services.market_data.revision import compute_checksum
from app.repositories.ohlcv import Candle


class TestComputeChecksum:
    def test_empty_candles(self):
        checksum = compute_checksum([])
        assert checksum == "empty"

    def test_single_candle(self):
        candle = Candle(
            exchange_id="kucoin",
            symbol="BTC-USDT",
            timeframe="1h",
            ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            open=42000.0,
            high=42500.0,
            low=41800.0,
            close=42200.0,
            volume=100.0,
        )
        checksum = compute_checksum([candle])
        assert len(checksum) == 16
        assert checksum.isalnum()

    def test_deterministic(self):
        candles = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                open=42000.0 + i,
                high=42500.0 + i,
                low=41800.0 + i,
                close=42200.0 + i,
                volume=100.0 + i,
            )
            for i in range(5)
        ]
        checksum1 = compute_checksum(candles)
        checksum2 = compute_checksum(candles)
        assert checksum1 == checksum2

    def test_different_data_different_checksum(self):
        candles1 = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                open=42000.0,
                high=42500.0,
                low=41800.0,
                close=42200.0,
                volume=100.0,
            )
        ]
        candles2 = [
            Candle(
                exchange_id="kucoin",
                symbol="BTC-USDT",
                timeframe="1h",
                ts=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                open=42000.0,
                high=42500.0,
                low=41800.0,
                close=42201.0,  # Different close
                volume=100.0,
            )
        ]
        assert compute_checksum(candles1) != compute_checksum(candles2)
