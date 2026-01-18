"""Ensure OHLCV data is available for a given range."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Type

import structlog

from app.repositories.ohlcv import Candle, OHLCVRepository
from app.services.market_data.base import MarketDataCandle
from app.services.market_data.ccxt_provider import CcxtMarketDataProvider

logger = structlog.get_logger(__name__)


@dataclass
class EnsureRangeResult:
    """Result of ensure_ohlcv_range operation."""

    total_candles: int  # Total candles in the requested range after operation
    fetched_candles: int  # Number of candles fetched (0 if data was cached)
    gaps_filled: list[tuple[datetime, datetime]]  # Ranges that were fetched
    was_cached: bool  # True if no fetching was needed


def _convert_market_candle(
    market_candle: MarketDataCandle,
    exchange_id: str,
    symbol: str,
    timeframe: str,
) -> Candle:
    """Convert MarketDataCandle to repository Candle."""
    return Candle(
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        ts=market_candle.ts,
        open=market_candle.open,
        high=market_candle.high,
        low=market_candle.low,
        close=market_candle.close,
        volume=market_candle.volume,
    )


async def ensure_ohlcv_range(
    pool,
    exchange_id: str,
    symbol: str,
    timeframe: str,
    start_ts: datetime,
    end_ts: datetime,
    _repo: Optional[OHLCVRepository] = None,
    _provider_class: Optional[Type[CcxtMarketDataProvider]] = None,
) -> EnsureRangeResult:
    """
    Ensure OHLCV data is available for the given range.

    Checks existing data in ohlcv_candles table.
    Fetches missing segments via CCXT provider.

    Args:
        pool: Database connection pool
        exchange_id: Exchange identifier (e.g., 'kucoin', 'binance')
        symbol: Canonical symbol (e.g., 'BTC-USDT')
        timeframe: Canonical timeframe (e.g., '1d', '1h')
        start_ts: Start of requested range (inclusive)
        end_ts: End of requested range (exclusive)
        _repo: Optional repository override for testing
        _provider_class: Optional provider class override for testing

    Returns:
        EnsureRangeResult with candle count and fetch details
    """
    log = logger.bind(
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        start_ts=start_ts.isoformat(),
        end_ts=end_ts.isoformat(),
    )

    # Use injected dependencies or create real ones
    repo = _repo if _repo is not None else OHLCVRepository(pool)
    provider_class = (
        _provider_class if _provider_class is not None else CcxtMarketDataProvider
    )

    # Check existing data range
    existing_range = await repo.get_available_range(exchange_id, symbol, timeframe)

    # Identify gaps
    gaps: list[tuple[datetime, datetime]] = []

    if existing_range is None:
        # No existing data - fetch entire range
        gaps.append((start_ts, end_ts))
        log.info("ensure_range_no_existing_data", gap_count=1)
    else:
        min_ts, max_ts = existing_range
        log.debug(
            "ensure_range_existing_data",
            min_ts=min_ts.isoformat(),
            max_ts=max_ts.isoformat(),
        )

        # Check for head gap (requested start is before existing min)
        if start_ts < min_ts:
            gaps.append((start_ts, min_ts))

        # Check for tail gap (requested end is after existing max)
        if end_ts > max_ts:
            # Use max_ts as gap start (provider will handle overlap deduplication)
            gaps.append((max_ts, end_ts))

    # If no gaps, data is fully cached
    if not gaps:
        total_count = await repo.count_in_range(
            exchange_id, symbol, timeframe, start_ts, end_ts
        )
        log.info("ensure_range_fully_cached", total_candles=total_count)
        return EnsureRangeResult(
            total_candles=total_count,
            fetched_candles=0,
            gaps_filled=[],
            was_cached=True,
        )

    # Fetch missing data
    provider = provider_class(exchange_id)
    fetched_count = 0
    gaps_filled: list[tuple[datetime, datetime]] = []

    try:
        for gap_start, gap_end in gaps:
            log.info(
                "ensure_range_fetching_gap",
                gap_start=gap_start.isoformat(),
                gap_end=gap_end.isoformat(),
            )

            # Fetch candles from provider
            market_candles = await provider.fetch_ohlcv(
                symbol, timeframe, gap_start, gap_end
            )

            if market_candles:
                # Convert to repository candles
                candles = [
                    _convert_market_candle(mc, exchange_id, symbol, timeframe)
                    for mc in market_candles
                ]

                # Upsert to database
                await repo.upsert_candles(candles)
                fetched_count += len(candles)
                gaps_filled.append((gap_start, gap_end))

                log.info(
                    "ensure_range_gap_filled",
                    gap_start=gap_start.isoformat(),
                    gap_end=gap_end.isoformat(),
                    candles_fetched=len(candles),
                )
    finally:
        # Always close the provider
        await provider.close()

    # Get final count
    total_count = await repo.count_in_range(
        exchange_id, symbol, timeframe, start_ts, end_ts
    )

    log.info(
        "ensure_range_complete",
        total_candles=total_count,
        fetched_candles=fetched_count,
        gaps_count=len(gaps_filled),
    )

    return EnsureRangeResult(
        total_candles=total_count,
        fetched_candles=fetched_count,
        gaps_filled=gaps_filled,
        was_cached=False,
    )
