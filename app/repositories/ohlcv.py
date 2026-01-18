"""Repository for OHLCV candle data."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Candle:
    """Single OHLCV candle."""

    exchange_id: str
    symbol: str
    timeframe: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        """Validate OHLCV constraints."""
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"high ({self.high}) must be >= open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"low ({self.low}) must be <= open, close, and high")
        if self.volume < 0:
            raise ValueError(f"volume ({self.volume}) must be >= 0")


class OHLCVRepository:
    """Repository for OHLCV candle operations."""

    def __init__(self, pool):
        self._pool = pool

    async def upsert_candles(self, candles: list[Candle]) -> int:
        """Upsert candles into ohlcv_candles table. Returns number of rows affected."""
        if not candles:
            return 0

        query = """
            INSERT INTO ohlcv_candles
                (exchange_id, symbol, timeframe, ts, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (exchange_id, symbol, timeframe, ts)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """

        async with self._pool.acquire() as conn:
            await conn.executemany(
                query,
                [
                    (
                        c.exchange_id,
                        c.symbol,
                        c.timeframe,
                        c.ts,
                        c.open,
                        c.high,
                        c.low,
                        c.close,
                        c.volume,
                    )
                    for c in candles
                ],
            )
        return len(candles)

    async def get_range(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[Candle]:
        """Get candles in a time range [start_ts, end_ts)."""
        query = """
            SELECT exchange_id, symbol, timeframe, ts, open, high, low, close, volume
            FROM ohlcv_candles
            WHERE exchange_id = $1 AND symbol = $2 AND timeframe = $3
              AND ts >= $4 AND ts < $5
            ORDER BY ts ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                query, exchange_id, symbol, timeframe, start_ts, end_ts
            )
        return [
            Candle(
                exchange_id=row["exchange_id"],
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                ts=row["ts"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            for row in rows
        ]

    async def get_available_range(
        self, exchange_id: str, symbol: str, timeframe: str
    ) -> Optional[tuple[datetime, datetime]]:
        """Get the min and max timestamps for a symbol/timeframe."""
        query = """
            SELECT MIN(ts) as min_ts, MAX(ts) as max_ts
            FROM ohlcv_candles
            WHERE exchange_id = $1 AND symbol = $2 AND timeframe = $3
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, exchange_id, symbol, timeframe)
        if row and row["min_ts"] is not None:
            return (row["min_ts"], row["max_ts"])
        return None

    async def count_in_range(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> int:
        """Count candles in a range."""
        query = """
            SELECT COUNT(*) as cnt FROM ohlcv_candles
            WHERE exchange_id = $1 AND symbol = $2 AND timeframe = $3
              AND ts >= $4 AND ts < $5
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query, exchange_id, symbol, timeframe, start_ts, end_ts
            )
        return row["cnt"] if row else 0
