"""DataSyncJob handler - expands core symbols and enqueues DataFetchJobs.

This handler is the "planner" in the data pipeline:
1. Lists enabled core symbols (optionally filtered by exchange)
2. For each symbol/timeframe combination:
   - Calculates date range (incremental: since last data, full: history window)
   - Enqueues a DataFetchJob with dedupe key
3. Links child jobs via parent_job_id for tracking
"""

from datetime import datetime, timezone, timedelta
from typing import Any

import structlog

from app.config import get_settings
from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType
from app.repositories.core_symbols import CoreSymbolsRepository
from app.repositories.job_events import JobEventsRepository
from app.repositories.jobs import JobRepository
from app.repositories.ohlcv import OHLCVRepository

logger = structlog.get_logger(__name__)


def generate_dedupe_key(
    exchange_id: str, symbol: str, timeframe: str, date: datetime
) -> str:
    """Generate dedupe key for a DataFetchJob.

    Format: data_fetch:{exchange_id}:{symbol}:{timeframe}:{YYYY-MM-DD}

    This prevents duplicate fetch jobs for the same symbol/timeframe on the same day.
    """
    date_str = date.strftime("%Y-%m-%d")
    return f"data_fetch:{exchange_id}:{symbol}:{timeframe}:{date_str}"


@default_registry.handler(JobType.DATA_SYNC)
async def handle_data_sync(job: Job, ctx: dict[str, Any]) -> dict[str, Any]:
    """Handle a DATA_SYNC job.

    Expands core symbols into DataFetchJob tasks.

    Job Payload:
        exchange_id: str (optional) - If provided, sync only this exchange
        mode: str - "incremental" (since last data) or "full" (history window)

    Context:
        pool: Database connection pool
        events_repo: JobEventsRepository for logging
        job_repo: JobRepository for creating child jobs

    Returns:
        dict with:
            symbols_processed: int - Number of symbols processed
            jobs_enqueued: int - Number of DataFetchJobs created
            exchange_id: str - Exchange synced or "all"
    """
    pool = ctx["pool"]
    events_repo: JobEventsRepository = ctx["events_repo"]
    job_repo: JobRepository = ctx["job_repo"]
    settings = get_settings()

    # Parse payload
    payload = job.payload
    exchange_id = payload.get("exchange_id")  # Optional
    mode = payload.get("mode", "incremental")

    log = logger.bind(
        job_id=str(job.id),
        exchange_id=exchange_id or "all",
        mode=mode,
    )
    log.info("data_sync_started")

    await events_repo.info(
        job.id,
        f"Starting data sync (exchange={exchange_id or 'all'}, mode={mode})",
    )

    # List enabled core symbols
    symbols_repo = CoreSymbolsRepository(pool)
    symbols = await symbols_repo.list_symbols(
        exchange_id=exchange_id, enabled_only=True
    )

    if not symbols:
        log.info("data_sync_no_symbols")
        await events_repo.info(job.id, "No enabled symbols found, nothing to sync")
        return {
            "symbols_processed": 0,
            "jobs_enqueued": 0,
            "exchange_id": exchange_id or "all",
        }

    log.info("data_sync_symbols_found", count=len(symbols))

    # Initialize OHLCV repo for checking latest data
    ohlcv_repo = OHLCVRepository(pool)

    # Get current timestamp for date range calculation
    now = datetime.now(timezone.utc)
    today_date = now

    jobs_enqueued = 0

    for symbol in symbols:
        for timeframe in symbol.timeframes:
            # Calculate date range based on mode
            if mode == "incremental":
                # Check for existing data
                available_range = await ohlcv_repo.get_available_range(
                    exchange_id=symbol.exchange_id,
                    symbol=symbol.canonical_symbol,
                    timeframe=timeframe,
                )
                if available_range:
                    # Start from last available timestamp
                    start_ts = available_range[1]  # max_ts
                else:
                    # No existing data, use history window
                    history_days = settings.get_data_sync_history_days(timeframe)
                    start_ts = now - timedelta(days=history_days)
            else:  # full mode
                # Use full history window from settings
                history_days = settings.get_data_sync_history_days(timeframe)
                start_ts = now - timedelta(days=history_days)

            end_ts = now

            # Generate dedupe key to prevent duplicate fetches same day
            dedupe_key = generate_dedupe_key(
                exchange_id=symbol.exchange_id,
                symbol=symbol.canonical_symbol,
                timeframe=timeframe,
                date=today_date,
            )

            # Create DataFetchJob payload
            fetch_payload = {
                "exchange_id": symbol.exchange_id,
                "symbol": symbol.canonical_symbol,
                "timeframe": timeframe,
                "start_ts": start_ts.isoformat(),
                "end_ts": end_ts.isoformat(),
            }

            # Enqueue DataFetchJob
            await job_repo.create(
                job_type=JobType.DATA_FETCH,
                payload=fetch_payload,
                parent_job_id=job.id,
                dedupe_key=dedupe_key,
            )
            jobs_enqueued += 1

            log.debug(
                "data_fetch_enqueued",
                symbol=symbol.canonical_symbol,
                timeframe=timeframe,
                start_ts=start_ts.isoformat(),
            )

    log.info(
        "data_sync_completed",
        symbols_processed=len(symbols),
        jobs_enqueued=jobs_enqueued,
    )

    await events_repo.info(
        job.id,
        f"Sync completed: {len(symbols)} symbols, {jobs_enqueued} jobs enqueued",
        symbols_processed=len(symbols),
        jobs_enqueued=jobs_enqueued,
    )

    return {
        "symbols_processed": len(symbols),
        "jobs_enqueued": jobs_enqueued,
        "exchange_id": exchange_id or "all",
    }
