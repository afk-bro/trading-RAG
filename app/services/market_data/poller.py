"""Live price polling service.

Periodically fetches OHLCV candles for enabled symbols from exchanges,
with per-exchange rate limiting and exponential backoff on failures.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram

from app.config import Settings
from app.repositories.core_symbols import CoreSymbolsRepository
from app.repositories.ohlcv import Candle, OHLCVRepository
from app.repositories.price_poll_state import (
    PollKey as RepoPollKey,
    PricePollStateRepository,
)
from app.services.market_data.ccxt_provider import CcxtMarketDataProvider

logger = structlog.get_logger(__name__)


# Timeframe to seconds mapping
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}


# =============================================================================
# Types
# =============================================================================


class PollKey(NamedTuple):
    """Unique identifier for a poll target: (exchange, symbol, timeframe)."""

    exchange_id: str
    symbol: str
    timeframe: str


@dataclass
class PollState:
    """State for a single poll key (placeholder for LP3 DB-backed state)."""

    exchange_id: str
    symbol: str
    timeframe: str
    next_poll_at: Optional[datetime] = None
    failure_count: int = 0
    last_success_at: Optional[datetime] = None
    last_candle_ts: Optional[datetime] = None
    last_error: Optional[str] = None


# =============================================================================
# Prometheus Metrics
# =============================================================================

POLL_ENABLED = Gauge(
    "live_price_poll_enabled",
    "Whether live price polling is enabled (1=enabled, 0=disabled)",
)
POLL_LAST_RUN_TIMESTAMP = Gauge(
    "live_price_poll_last_run_timestamp",
    "Unix timestamp of last poll tick",
)
POLL_RUNS_TOTAL = Counter(
    "live_price_poll_runs_total",
    "Total poll ticks executed",
    ["status"],  # started, completed, failure
)
POLL_PAIRS_FETCHED_TOTAL = Counter(
    "live_price_poll_pairs_fetched_total",
    "Pairs fetched by poller",
    ["exchange_id", "status"],  # success, failure, provider_error
)
POLL_CANDLES_UPSERTED_TOTAL = Counter(
    "live_price_poll_candles_upserted_total",
    "Candles upserted by poller",
    ["exchange_id", "timeframe"],
)
POLL_INFLIGHT = Gauge(
    "live_price_poll_inflight",
    "Number of pairs currently being fetched",
    ["exchange_id"],
)
POLL_FETCH_DURATION = Histogram(
    "live_price_poll_fetch_duration_seconds",
    "Duration of individual pair fetches",
    ["exchange_id"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
LATEST_CANDLE_AGE = Gauge(
    "live_price_latest_candle_age_seconds",
    "Age of latest candle in seconds (staleness indicator)",
    ["exchange_id", "symbol", "timeframe"],
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PollTickResult:
    """Result of a single poll tick."""

    pairs_fetched: int = 0
    pairs_succeeded: int = 0
    pairs_failed: int = 0
    candles_upserted: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class PollerHealth:
    """Health status for the live price poller."""

    enabled: bool = False
    running: bool = False
    # Counts (computed on-demand in LP3)
    pairs_enabled: int = 0
    pairs_due: int = 0
    pairs_never_polled: int = 0
    # Staleness
    worst_staleness_seconds: Optional[float] = None
    # Last run
    last_run_at: Optional[datetime] = None
    last_run_pairs_fetched: int = 0
    last_run_candles_upserted: int = 0
    last_run_errors: int = 0
    last_error_message: Optional[str] = None
    # Config
    poll_interval_seconds: int = 60
    active_timeframes: list[str] = field(default_factory=list)


# =============================================================================
# Poller Service
# =============================================================================


class LivePricePoller:
    """
    Background service for polling live OHLCV prices.

    Features:
    - Runs every tick_seconds, selecting due pairs
    - Per-exchange semaphores for rate limiting
    - CCXT provider for exchange data fetching (LP2)
    - Smart lookback from last_candle_ts (LP2)
    - Staleness metrics per PollKey (LP2)
    - DB-backed scheduling with backoff (LP3)
    - Exposes Prometheus metrics
    - Can be started/stopped gracefully
    """

    def __init__(
        self,
        pool,
        settings: Settings,
    ):
        """
        Initialize poller.

        Args:
            pool: asyncpg connection pool
            settings: Application settings
        """
        self._pool = pool
        self._settings = settings
        self._symbols_repo = CoreSymbolsRepository(pool)
        self._ohlcv_repo = OHLCVRepository(pool)
        self._state_repo = PricePollStateRepository(
            pool,
            interval_seconds=settings.live_price_poll_interval_seconds,
            jitter_seconds=settings.live_price_poll_jitter_seconds,
            backoff_max_seconds=settings.live_price_poll_backoff_max_seconds,
        )

        # Per-exchange semaphores for rate limiting
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._inflight: dict[str, int] = {}

        # Provider cache (lazy-loaded per exchange)
        self._providers: dict[str, CcxtMarketDataProvider] = {}

        # Background task management
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._running = False

        # State tracking
        self._last_run_at: Optional[datetime] = None
        self._last_run_result: Optional[PollTickResult] = None

        # Active timeframes (from config)
        self._active_timeframes = set(settings.live_price_poll_timeframes)

        # Warned timeframes (log once per unsupported exchange/tf combo)
        self._warned_timeframes: set[tuple[str, str]] = set()

    @property
    def is_running(self) -> bool:
        """Check if poller is currently running."""
        return self._running

    def _get_semaphore(self, exchange_id: str) -> asyncio.Semaphore:
        """Get or create a semaphore for an exchange."""
        if exchange_id not in self._semaphores:
            self._semaphores[exchange_id] = asyncio.Semaphore(
                self._settings.live_price_poll_max_concurrency_per_exchange
            )
            self._inflight[exchange_id] = 0
        return self._semaphores[exchange_id]

    async def start(self) -> None:
        """Start the polling background task."""
        if self._running:
            logger.warning("Live price poller already running")
            return

        if not self._settings.live_price_poll_enabled:
            logger.info("Live price polling disabled (LIVE_PRICE_POLL_ENABLED=false)")
            POLL_ENABLED.set(0)
            return

        max_concurrency = self._settings.live_price_poll_max_concurrency_per_exchange
        logger.info(
            "Starting live price poller",
            tick_seconds=self._settings.live_price_poll_tick_seconds,
            interval_seconds=self._settings.live_price_poll_interval_seconds,
            max_concurrency_per_exchange=max_concurrency,
            max_pairs_per_tick=self._settings.live_price_poll_max_pairs_per_tick,
            timeframes=self._settings.live_price_poll_timeframes,
        )

        POLL_ENABLED.set(1)
        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the polling background task gracefully.

        Args:
            timeout: Max seconds to wait for current fetches to complete
        """
        if not self._running:
            return

        logger.info("Stopping live price poller")
        self._stop_event.set()

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Poller stop timeout, cancelling task")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        self._running = False
        POLL_ENABLED.set(0)
        logger.info("Live price poller stopped")

    async def run_once(self) -> PollTickResult:
        """
        Run a single poll cycle (for manual triggering).

        Returns:
            PollTickResult with fetch statistics
        """
        return await self._do_poll_tick()

    async def get_health(self) -> PollerHealth:
        """Get current poller health status."""
        # Compute pairs_enabled on-demand (symbols × timeframes)
        symbols = await self._symbols_repo.list_symbols(enabled_only=True)
        pairs_enabled = sum(
            len(sym.timeframes or self._settings.live_price_poll_timeframes)
            for sym in symbols
        )

        # Get counts from state repository (LP3)
        try:
            pairs_due = await self._state_repo.count_due_pairs()
            pairs_never_polled = await self._state_repo.count_never_polled()
            worst_staleness = await self._state_repo.get_worst_staleness()
        except Exception as e:
            logger.warning("Failed to get health counts from state repo", error=str(e))
            pairs_due = 0
            pairs_never_polled = 0
            worst_staleness = None

        return PollerHealth(
            enabled=self._settings.live_price_poll_enabled,
            running=self._running,
            pairs_enabled=pairs_enabled,
            pairs_due=pairs_due,
            pairs_never_polled=pairs_never_polled,
            worst_staleness_seconds=worst_staleness,
            last_run_at=self._last_run_at,
            last_run_pairs_fetched=(
                self._last_run_result.pairs_fetched if self._last_run_result else 0
            ),
            last_run_candles_upserted=(
                self._last_run_result.candles_upserted if self._last_run_result else 0
            ),
            last_run_errors=(
                len(self._last_run_result.errors) if self._last_run_result else 0
            ),
            last_error_message=(
                self._last_run_result.errors[0]
                if self._last_run_result and self._last_run_result.errors
                else None
            ),
            poll_interval_seconds=self._settings.live_price_poll_interval_seconds,
            active_timeframes=list(self._active_timeframes),
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _poll_loop(self) -> None:
        """Main polling loop - runs until stop_event is set."""
        tick_seconds = self._settings.live_price_poll_tick_seconds

        while not self._stop_event.is_set():
            try:
                result = await self._do_poll_tick()
                self._last_run_result = result
                self._last_run_at = datetime.now(timezone.utc)

                # Update metrics
                POLL_LAST_RUN_TIMESTAMP.set(self._last_run_at.timestamp())
                if result.pairs_fetched > 0 or result.errors:
                    status = "completed" if not result.errors else "partial"
                else:
                    status = "skipped"
                POLL_RUNS_TOTAL.labels(status=status).inc()

            except Exception as e:
                logger.exception("Poll tick failed", error=str(e))
                POLL_RUNS_TOTAL.labels(status="failure").inc()

            # Wait for next tick (interruptible)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=tick_seconds,
                )
                # If we get here, stop was requested
                break
            except asyncio.TimeoutError:
                # Normal timeout, continue to next tick
                pass

    async def _do_poll_tick(self) -> PollTickResult:
        """Execute a single poll tick."""
        start_time = time.time()
        result = PollTickResult()

        # Get due pairs (LP1: simple expansion, LP3: DB-first)
        due_pairs = await self._get_due_pairs(
            limit=self._settings.live_price_poll_max_pairs_per_tick
        )

        if not due_pairs:
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        logger.info(
            "poll_tick_start",
            pairs_selected=len(due_pairs),
        )

        # Fetch with per-exchange semaphores
        tasks = []
        for pair, state in due_pairs:
            sem = self._get_semaphore(pair.exchange_id)
            last_candle_ts = state.last_candle_ts if state else None
            tasks.append(self._fetch_with_semaphore(sem, pair, last_candle_ts))

        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and update scheduling state
        for (pair, state), fetch_result in zip(due_pairs, fetch_results):
            result.pairs_fetched += 1
            repo_key = RepoPollKey(pair.exchange_id, pair.symbol, pair.timeframe)

            if isinstance(fetch_result, Exception):
                result.pairs_failed += 1
                error_msg = str(fetch_result)
                result.errors.append(f"{pair}: {error_msg}")
                POLL_PAIRS_FETCHED_TOTAL.labels(
                    exchange_id=pair.exchange_id,
                    status="failure",
                ).inc()
                # Update state with failure (exponential backoff)
                try:
                    await self._state_repo.mark_failure(repo_key, error_msg)
                except Exception as e:
                    logger.warning(
                        "Failed to mark_failure in state repo",
                        pair=str(pair),
                        error=str(e),
                    )
            else:
                count, latest_ts = fetch_result
                result.pairs_succeeded += 1
                result.candles_upserted += count
                POLL_PAIRS_FETCHED_TOTAL.labels(
                    exchange_id=pair.exchange_id,
                    status="success",
                ).inc()
                if count > 0:
                    POLL_CANDLES_UPSERTED_TOTAL.labels(
                        exchange_id=pair.exchange_id,
                        timeframe=pair.timeframe,
                    ).inc(count)
                # Update state with success (schedule next poll)
                if latest_ts:
                    try:
                        await self._state_repo.mark_success(repo_key, latest_ts)
                    except Exception as e:
                        logger.warning(
                            "Failed to mark_success in state repo",
                            pair=str(pair),
                            error=str(e),
                        )

        result.duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "poll_tick_complete",
            pairs_fetched=result.pairs_fetched,
            pairs_succeeded=result.pairs_succeeded,
            pairs_failed=result.pairs_failed,
            candles_upserted=result.candles_upserted,
            duration_ms=result.duration_ms,
        )

        return result

    async def _get_due_pairs(
        self, limit: int
    ) -> list[tuple[PollKey, Optional[PollState]]]:
        """
        Get pairs due for polling (DB-first approach - LP3).

        1. Query due pairs from state table (ordered by next_poll_at NULLS FIRST)
        2. Validate against currently enabled symbols
        3. Seed new pairs for newly-enabled symbols missing state rows
        """
        # 1. Get due pairs from state table
        due_states = await self._state_repo.list_due_pairs(limit=limit * 2)

        # 2. Fetch enabled symbols to validate
        symbols = await self._symbols_repo.list_symbols(enabled_only=True)
        enabled_map = {(s.exchange_id, s.canonical_symbol): s for s in symbols}

        # 3. Filter to still-enabled symbols with valid timeframes
        valid_due: list[tuple[PollKey, Optional[PollState]]] = []
        for state in due_states:
            sym = enabled_map.get((state.exchange_id, state.symbol))
            if not sym:
                continue  # Symbol disabled since last poll
            symbol_tfs = sym.timeframes or self._settings.live_price_poll_timeframes
            if state.timeframe not in symbol_tfs:
                continue  # Timeframe no longer enabled for this symbol
            if state.timeframe not in self._active_timeframes:
                continue  # Timeframe not in global active set
            poll_state = PollState(
                exchange_id=state.exchange_id,
                symbol=state.symbol,
                timeframe=state.timeframe,
                next_poll_at=state.next_poll_at,
                failure_count=state.failure_count,
                last_success_at=state.last_success_at,
                last_candle_ts=state.last_candle_ts,
                last_error=state.last_error,
            )
            valid_due.append(
                (PollKey(state.exchange_id, state.symbol, state.timeframe), poll_state)
            )

        # 4. Top up with newly-enabled symbols missing state rows
        if len(valid_due) < limit:
            existing_keys = {
                (p.exchange_id, p.symbol, p.timeframe) for p, _ in valid_due
            }
            for sym in symbols:
                tfs = sym.timeframes or self._settings.live_price_poll_timeframes
                for tf in tfs:
                    if tf not in self._active_timeframes:
                        continue
                    key = (sym.exchange_id, sym.canonical_symbol, tf)
                    if key not in existing_keys:
                        # Seed state row (next_poll_at=NULL = due immediately)
                        repo_key = RepoPollKey(
                            sym.exchange_id, sym.canonical_symbol, tf
                        )
                        try:
                            await self._state_repo.upsert_state_if_missing(repo_key)
                        except Exception as e:
                            logger.warning(
                                "Failed to seed state row",
                                key=key,
                                error=str(e),
                            )
                        valid_due.append(
                            (PollKey(sym.exchange_id, sym.canonical_symbol, tf), None)
                        )
                        existing_keys.add(key)
                    if len(valid_due) >= limit:
                        break
                if len(valid_due) >= limit:
                    break

        return valid_due[:limit]

    async def _fetch_with_semaphore(
        self,
        sem: asyncio.Semaphore,
        pair: PollKey,
        last_candle_ts: Optional[datetime],
    ) -> tuple[int, Optional[datetime]]:
        """Fetch candles with semaphore-based rate limiting."""
        async with sem:
            self._inflight[pair.exchange_id] = (
                self._inflight.get(pair.exchange_id, 0) + 1
            )
            POLL_INFLIGHT.labels(exchange_id=pair.exchange_id).set(
                self._inflight[pair.exchange_id]
            )
            try:
                start_time = time.time()
                result = await self._fetch_pair(pair, last_candle_ts)
                duration = time.time() - start_time
                POLL_FETCH_DURATION.labels(exchange_id=pair.exchange_id).observe(
                    duration
                )
                return result
            finally:
                self._inflight[pair.exchange_id] -= 1
                POLL_INFLIGHT.labels(exchange_id=pair.exchange_id).set(
                    self._inflight[pair.exchange_id]
                )

    async def _fetch_pair(
        self,
        pair: PollKey,
        last_candle_ts: Optional[datetime],
    ) -> tuple[int, Optional[datetime]]:
        """
        Fetch candles for a single (exchange, symbol, timeframe) pair.

        Args:
            pair: The poll key (exchange_id, symbol, timeframe)
            last_candle_ts: Last known candle timestamp (for smart lookback)

        Returns:
            Tuple of (candles_upserted, latest_candle_timestamp)
        """
        now = datetime.now(timezone.utc)
        tf_seconds = TIMEFRAME_SECONDS.get(pair.timeframe)

        if not tf_seconds:
            logger.warning(
                "Unknown timeframe",
                exchange_id=pair.exchange_id,
                symbol=pair.symbol,
                timeframe=pair.timeframe,
            )
            return 0, None

        # Smart lookback: from last_candle_ts or fallback to config
        if last_candle_ts:
            # Fetch since last known candle with 1-candle overlap buffer
            start_ts = last_candle_ts - timedelta(seconds=tf_seconds)
        else:
            # Fallback: fetch lookback_candles worth
            lookback = self._settings.live_price_poll_lookback_candles
            start_ts = now - timedelta(seconds=lookback * tf_seconds)

        # Get or create provider for this exchange
        provider = self._get_provider(pair.exchange_id)

        # Fetch from exchange via CCXT provider
        market_candles = await provider.fetch_ohlcv(
            symbol=pair.symbol,
            timeframe=pair.timeframe,
            start_ts=start_ts,
            end_ts=now,
        )

        if not market_candles:
            return 0, None

        # Convert MarketDataCandle to repository Candle format
        candles = [
            Candle(
                exchange_id=pair.exchange_id,
                symbol=pair.symbol,
                timeframe=pair.timeframe,
                ts=mc.ts,
                open=mc.open,
                high=mc.high,
                low=mc.low,
                close=mc.close,
                volume=mc.volume,
            )
            for mc in market_candles
        ]

        # Upsert candles to repository
        count = await self._ohlcv_repo.upsert_candles(candles)

        # Compute latest candle timestamp and update staleness metric
        latest_ts = max(c.ts for c in candles)
        age_seconds = (now - latest_ts).total_seconds()
        LATEST_CANDLE_AGE.labels(
            exchange_id=pair.exchange_id,
            symbol=pair.symbol,
            timeframe=pair.timeframe,
        ).set(age_seconds)

        logger.debug(
            "Fetched candles",
            exchange_id=pair.exchange_id,
            symbol=pair.symbol,
            timeframe=pair.timeframe,
            count=count,
            latest_ts=latest_ts.isoformat(),
            age_seconds=round(age_seconds, 1),
        )

        return count, latest_ts

    def _get_provider(self, exchange_id: str) -> CcxtMarketDataProvider:
        """Get or create a CCXT provider for an exchange (cached)."""
        if exchange_id not in self._providers:
            self._providers[exchange_id] = CcxtMarketDataProvider(exchange_id)
            logger.info("Created CCXT provider", exchange_id=exchange_id)
        return self._providers[exchange_id]


# =============================================================================
# Module-level singleton (for lifespan management)
# =============================================================================

_poller: Optional[LivePricePoller] = None


def get_poller() -> Optional[LivePricePoller]:
    """Get the global poller instance."""
    return _poller


def set_poller(poller: Optional[LivePricePoller]) -> None:
    """Set the global poller instance."""
    global _poller
    _poller = poller
