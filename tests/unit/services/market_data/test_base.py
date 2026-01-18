"""Tests for market data provider interface."""
from datetime import datetime, timezone

from app.services.market_data.base import (
    MarketDataCandle,
    normalize_timeframe,
)


class TestMarketDataCandle:
    def test_candle_creation(self):
        candle = MarketDataCandle(
            ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000.0,
        )
        assert candle.close == 103.0


class TestNormalizeTimeframe:
    def test_standard_timeframes(self):
        assert normalize_timeframe("1m") == "1m"
        assert normalize_timeframe("5m") == "5m"
        assert normalize_timeframe("1h") == "1h"
        assert normalize_timeframe("1d") == "1d"

    def test_alternate_formats(self):
        assert normalize_timeframe("1min") == "1m"
        assert normalize_timeframe("1hour") == "1h"
        assert normalize_timeframe("1day") == "1d"
