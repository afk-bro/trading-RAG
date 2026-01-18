"""CCXT-based market data provider."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt.async_support as ccxt
import structlog

from app.config import get_settings
from app.services.market_data.base import MarketDataCandle, normalize_timeframe

logger = structlog.get_logger(__name__)

# Timeframe to milliseconds
TIMEFRAME_MS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


class CcxtMarketDataProvider:
    """Market data provider using CCXT library."""

    # Default rate limit when not using settings
    DEFAULT_RATE_LIMIT_MS = 100

    def __init__(
        self,
        exchange_id: str,
        rate_limit_ms: Optional[int] = None,
    ):
        """Initialize the CCXT provider.

        Args:
            exchange_id: CCXT exchange identifier (e.g., 'kucoin', 'binance')
            rate_limit_ms: Minimum milliseconds between API calls.
                          If None, uses settings.ccxt_rate_limit_ms (lazy loaded).
        """
        self._exchange_id = exchange_id
        self._rate_limit_ms_override = rate_limit_ms
        self._exchange: Optional[ccxt.Exchange] = None
        self._last_request_time: float = 0

    @property
    def _rate_limit_ms(self) -> int:
        """Get rate limit in milliseconds (lazy settings load)."""
        if self._rate_limit_ms_override is not None:
            return self._rate_limit_ms_override
        try:
            return get_settings().ccxt_rate_limit_ms
        except Exception:
            # Fallback for unit tests without settings
            return self.DEFAULT_RATE_LIMIT_MS

    @property
    def exchange_id(self) -> str:
        """Get the exchange identifier."""
        return self._exchange_id

    def _get_exchange(self) -> ccxt.Exchange:
        """Get or create the CCXT exchange instance."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_class(
                {
                    "enableRateLimit": True,
                    "rateLimit": self._rate_limit_ms,
                }
            )
        return self._exchange

    async def close(self) -> None:
        """Close the exchange connection and release resources."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    def normalize_symbol(self, raw: str) -> str:
        """Convert exchange symbol (BTC/USDT) to canonical (BTC-USDT).

        Args:
            raw: Exchange-specific symbol format

        Returns:
            Canonical symbol format with dash separator
        """
        return raw.replace("/", "-")

    def exchange_symbol(self, canonical: str) -> str:
        """Convert canonical symbol (BTC-USDT) to exchange (BTC/USDT).

        Args:
            canonical: Canonical symbol format with dash separator

        Returns:
            Exchange-specific symbol format with slash separator
        """
        return canonical.replace("-", "/")

    def canonical_timeframe(self, tf: str) -> str:
        """Convert any timeframe format to canonical format.

        Args:
            tf: Input timeframe (e.g., '1m', '1min', '1hour')

        Returns:
            Canonical timeframe format
        """
        return normalize_timeframe(tf)

    def ccxt_timeframe(self, canonical_tf: str) -> str:
        """Convert canonical timeframe to CCXT format.

        Args:
            canonical_tf: Canonical timeframe (e.g., '1m', '1h', '1d')

        Returns:
            CCXT-compatible timeframe string
        """
        # CCXT uses the same format as our canonical
        return canonical_tf

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        now = time.time() * 1000
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_ms:
            await asyncio.sleep((self._rate_limit_ms - elapsed) / 1000)
        self._last_request_time = time.time() * 1000

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[MarketDataCandle]:
        """Fetch OHLCV data with pagination.

        Args:
            symbol: Canonical symbol (e.g., 'BTC-USDT')
            timeframe: Canonical timeframe (e.g., '1h', '1d')
            start_ts: Start timestamp (UTC)
            end_ts: End timestamp (UTC)

        Returns:
            List of MarketDataCandle with close timestamps

        Raises:
            Exception: If CCXT API call fails
        """
        exchange = self._get_exchange()
        ccxt_symbol = self.exchange_symbol(symbol)
        ccxt_tf = self.ccxt_timeframe(timeframe)

        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        tf_ms = TIMEFRAME_MS.get(timeframe, 60000)
        limit = 1000  # Max candles per request for most exchanges

        all_candles: list[MarketDataCandle] = []
        since = start_ms

        log = logger.bind(
            exchange=self._exchange_id, symbol=symbol, timeframe=timeframe
        )

        while since < end_ms:
            await self._rate_limit()
            try:
                ohlcv = await exchange.fetch_ohlcv(
                    ccxt_symbol, ccxt_tf, since=since, limit=limit
                )
            except Exception as e:
                log.error("ccxt_fetch_failed", error=str(e), since=since)
                raise

            if not ohlcv:
                break

            for candle in ohlcv:
                ts_ms, o, h, l, c, v = candle
                close_ts_ms = ts_ms + tf_ms
                if close_ts_ms > end_ms:
                    break
                all_candles.append(
                    MarketDataCandle(
                        ts=datetime.fromtimestamp(close_ts_ms / 1000, tz=timezone.utc),
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=float(v) if v else 0.0,
                    )
                )

            last_ts = ohlcv[-1][0]
            if last_ts <= since:
                # No progress, avoid infinite loop
                break
            since = last_ts + tf_ms

            log.debug(
                "ccxt_batch_fetched",
                batch_size=len(ohlcv),
                total_candles=len(all_candles),
            )

        log.info(
            "ccxt_fetch_complete",
            total_candles=len(all_candles),
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
        )
        return all_candles
