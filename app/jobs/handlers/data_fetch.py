"""DataFetchJob handler - fetches OHLCV data via CCXT and stores it.

This handler:
1. Parses job payload (exchange, symbol, timeframe, date range)
2. Fetches OHLCV data via CcxtMarketDataProvider
3. Converts MarketDataCandle to repository Candle format
4. Upserts candles to ohlcv_candles table
5. Computes checksum and stores data revision
6. Returns result with counts and checksum
"""

from datetime import datetime, timezone
from typing import Any

import structlog

from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType
from app.repositories.ohlcv import OHLCVRepository, Candle
from app.repositories.data_revisions import DataRevisionRepository
from app.repositories.job_events import JobEventsRepository
from app.services.market_data.ccxt_provider import CcxtMarketDataProvider
from app.services.market_data.revision import compute_checksum

logger = structlog.get_logger(__name__)


def parse_iso_timestamp(value: str) -> datetime:
    """Parse ISO format timestamp string to datetime.

    Args:
        value: ISO format string (e.g., '2024-01-01T00:00:00Z')

    Returns:
        datetime with UTC timezone
    """
    # Handle both 'Z' suffix and '+00:00' timezone formats
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    # Ensure UTC timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@default_registry.handler(JobType.DATA_FETCH)
async def handle_data_fetch(job: Job, ctx: dict[str, Any]) -> dict[str, Any]:
    """Handle a DATA_FETCH job.

    Fetches OHLCV data from an exchange and stores it in the database.

    Job Payload:
        exchange_id: str - CCXT exchange ID (e.g., 'kucoin')
        symbol: str - Canonical symbol (e.g., 'BTC-USDT')
        timeframe: str - Canonical timeframe (e.g., '1h')
        start_ts: str - ISO format start timestamp
        end_ts: str - ISO format end timestamp

    Context:
        pool: Database connection pool
        events_repo: JobEventsRepository for logging

    Returns:
        dict with:
            candles_fetched: int - Number of candles from exchange
            candles_upserted: int - Number of candles written to DB
            checksum: str - Data revision checksum

    Raises:
        ValueError: If required payload fields are missing
        Exception: If CCXT fetch or database operations fail
    """
    pool = ctx["pool"]
    events_repo: JobEventsRepository = ctx["events_repo"]

    # Parse payload
    payload = job.payload
    exchange_id = payload.get("exchange_id")
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe")
    start_ts_str = payload.get("start_ts")
    end_ts_str = payload.get("end_ts")

    # Validate required fields
    if not all([exchange_id, symbol, timeframe, start_ts_str, end_ts_str]):
        raise ValueError(
            "Missing required payload fields: exchange_id, symbol, timeframe, start_ts, end_ts"
        )

    start_ts = parse_iso_timestamp(start_ts_str)
    end_ts = parse_iso_timestamp(end_ts_str)

    log = logger.bind(
        job_id=str(job.id),
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
    )
    log.info(
        "data_fetch_started", start_ts=start_ts.isoformat(), end_ts=end_ts.isoformat()
    )

    await events_repo.info(
        job.id,
        f"Fetching {symbol} {timeframe} data from {exchange_id}",
        start_ts=start_ts.isoformat(),
        end_ts=end_ts.isoformat(),
    )

    # Fetch data from exchange
    provider = CcxtMarketDataProvider(exchange_id)
    try:
        market_candles = await provider.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )
    finally:
        await provider.close()

    candles_fetched = len(market_candles)
    log.info("data_fetch_received", candles_fetched=candles_fetched)

    await events_repo.info(
        job.id,
        f"Received {candles_fetched} candles from exchange",
    )

    # Convert MarketDataCandle -> repository Candle
    candles = [
        Candle(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            ts=mc.ts,
            open=mc.open,
            high=mc.high,
            low=mc.low,
            close=mc.close,
            volume=mc.volume,
        )
        for mc in market_candles
    ]

    # Upsert candles to database
    ohlcv_repo = OHLCVRepository(pool)
    candles_upserted = await ohlcv_repo.upsert_candles(candles)

    log.info("data_fetch_upserted", candles_upserted=candles_upserted)

    await events_repo.info(
        job.id,
        f"Upserted {candles_upserted} candles to database",
    )

    # Compute and store data revision
    checksum = compute_checksum(candles)

    revision_repo = DataRevisionRepository(pool)
    await revision_repo.upsert(
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        row_count=candles_upserted,
        checksum=checksum,
        job_id=str(job.id),
    )

    log.info("data_fetch_completed", checksum=checksum)

    await events_repo.info(
        job.id,
        f"Data revision saved with checksum {checksum}",
        row_count=candles_upserted,
    )

    return {
        "candles_fetched": candles_fetched,
        "candles_upserted": candles_upserted,
        "checksum": checksum,
    }
