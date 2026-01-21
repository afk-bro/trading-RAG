"""Live price polling service.

Periodically fetches OHLCV candles for enabled symbols from exchanges,
with per-exchange rate limiting and exponential backoff on failures.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram

from app.config import Settings
from app.repositories.core_symbols import CoreSymbolsRepository

logger = structlog.get_logger(__name__)


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

    Features (LP1):
    - Runs every tick_seconds, selecting due pairs
    - Per-exchange semaphores for rate limiting
    - Exposes Prometheus metrics
    - Can be started/stopped gracefully

    Future (LP2/LP3):
    - Actual CCXT fetch and OHLCV upsert
    - DB-backed scheduling with backoff
    - Staleness metrics
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

        # Per-exchange semaphores for rate limiting
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._inflight: dict[str, int] = {}

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
            logger.info(
                "Live price polling disabled (LIVE_PRICE_POLL_ENABLED=false)"
            )
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

        # LP1: Simple counts - LP3 will add DB-backed counting
        pairs_due = pairs_enabled  # All pairs considered "due" in LP1

        return PollerHealth(
            enabled=self._settings.live_price_poll_enabled,
            running=self._running,
            pairs_enabled=pairs_enabled,
            pairs_due=pairs_due,
            pairs_never_polled=0,  # LP3 will track this
            worst_staleness_seconds=None,  # LP3 will track this
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
        import time

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
        # LP1: Just log selection, LP2 will add actual fetching
        for pair, state in due_pairs:
            result.pairs_fetched += 1
            result.pairs_succeeded += 1  # LP1: No actual fetch, always "succeed"
            POLL_PAIRS_FETCHED_TOTAL.labels(
                exchange_id=pair.exchange_id,
                status="success",
            ).inc()

        result.duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "poll_tick_complete",
            pairs_fetched=result.pairs_fetched,
            pairs_succeeded=result.pairs_succeeded,
            pairs_failed=result.pairs_failed,
            duration_ms=result.duration_ms,
        )

        return result

    async def _get_due_pairs(
        self, limit: int
    ) -> list[tuple[PollKey, Optional[PollState]]]:
        """
        Get pairs due for polling (LP1: simple expansion).

        LP3 will replace this with DB-first approach.
        """
        # 1. Fetch enabled symbols
        symbols = await self._symbols_repo.list_symbols(enabled_only=True)

        # 2. Expand to (exchange, symbol, timeframe) pairs
        all_pairs: list[tuple[PollKey, Optional[PollState]]] = []
        for sym in symbols:
            # Use symbol's timeframes or fall back to config defaults
            symbol_tfs = sym.timeframes or self._settings.live_price_poll_timeframes
            for tf in symbol_tfs:
                if tf in self._active_timeframes:
                    pair = PollKey(sym.exchange_id, sym.canonical_symbol, tf)
                    all_pairs.append((pair, None))  # LP1: No state yet

        # LP1: All pairs are "due" - LP3 will filter by next_poll_at
        return all_pairs[:limit]


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
