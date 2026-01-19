"""Base interface for market data providers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class MarketDataCandle:
    """Single candle from market data provider."""

    ts: datetime  # Candle close time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


# Timeframe normalization map
TIMEFRAME_MAP = {
    "1m": "1m",
    "1min": "1m",
    "5m": "5m",
    "5min": "5m",
    "15m": "15m",
    "15min": "15m",
    "1h": "1h",
    "1hour": "1h",
    "60m": "1h",
    "1d": "1d",
    "1day": "1d",
    "24h": "1d",
}


def normalize_timeframe(tf: str) -> str:
    """Normalize timeframe to canonical format."""
    return TIMEFRAME_MAP.get(tf.lower(), tf)


class MarketDataProvider(Protocol):
    """Protocol for market data providers."""

    @property
    def exchange_id(self) -> str:
        """Get the exchange identifier."""
        ...

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[MarketDataCandle]:
        """Fetch OHLCV data for a symbol and time range."""
        ...

    def normalize_symbol(self, raw: str) -> str:
        """Convert exchange-specific symbol to canonical format."""
        ...

    def exchange_symbol(self, canonical: str) -> str:
        """Convert canonical symbol to exchange-specific format."""
        ...
