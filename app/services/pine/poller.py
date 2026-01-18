"""Pine repository polling service.

Automatically scans enabled GitHub repositories on a schedule,
with exponential backoff on failures and concurrency limits.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import structlog
from prometheus_client import Counter, Gauge

from app.config import Settings
from app.services.pine.repo_registry import PineRepoRepository

logger = structlog.get_logger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

POLL_RUNS_TOTAL = Counter(
    "pine_repo_poll_runs_total",
    "Total poll runs executed",
    ["status"],  # success, failure, skipped
)
POLL_REPOS_SCANNED_TOTAL = Counter(
    "pine_repo_poll_repos_scanned_total",
    "Total repos scanned by poller",
    ["status"],  # success, failure
)
POLL_INFLIGHT = Gauge(
    "pine_repo_poll_inflight",
    "Number of repos currently being scanned",
)
POLL_DUE_COUNT = Gauge(
    "pine_repo_poll_due_count",
    "Number of repos currently due for scanning",
)
POLL_LAST_RUN_TIMESTAMP = Gauge(
    "pine_repo_poll_last_run_timestamp",
    "Timestamp of last poll run (unix seconds)",
)
POLL_ENABLED = Gauge(
    "pine_repo_poll_enabled",
    "Whether polling is enabled (1=enabled, 0=disabled)",
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PollRunResult:
    """Result of a single poll run (one tick)."""

    repos_scanned: int = 0
    repos_succeeded: int = 0
    repos_failed: int = 0
    repos_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class PollerHealth:
    """Health status for the poller."""

    enabled: bool = False
    running: bool = False
    last_run_at: Optional[datetime] = None
    last_run_repos_scanned: int = 0
    last_run_errors: int = 0
    repos_due_count: int = 0
    poll_interval_minutes: int = 15
    poll_max_concurrency: int = 2
    poll_tick_seconds: int = 60


# =============================================================================
# Poller Service
# =============================================================================


class PineRepoPoller:
    """
    Background service for polling GitHub repositories.

    Features:
    - Runs every tick_seconds, checking for repos due for scan
    - Scans up to max_repos_per_tick repos per tick
    - Limits concurrent scans with asyncio.Semaphore
    - Updates next_scan_at with jitter on success
    - Applies exponential backoff on failure
    - Exposes Prometheus metrics
    - Can be started/stopped gracefully
    """

    def __init__(
        self,
        pool,
        settings: Settings,
        qdrant_client=None,
    ):
        """
        Initialize poller.

        Args:
            pool: asyncpg connection pool
            settings: Application settings
            qdrant_client: Optional Qdrant client for KB ingestion
        """
        self._pool = pool
        self._settings = settings
        self._qdrant_client = qdrant_client
        self._repo_registry = PineRepoRepository(pool)

        # Concurrency control
        self._semaphore = asyncio.Semaphore(settings.pine_repo_poll_max_concurrency)
        self._inflight = 0

        # Background task management
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._running = False

        # State tracking
        self._last_run_at: Optional[datetime] = None
        self._last_run_result: Optional[PollRunResult] = None

    @property
    def is_running(self) -> bool:
        """Check if poller is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the polling background task."""
        if self._running:
            logger.warning("Poller already running")
            return

        if not self._settings.pine_repo_poll_enabled:
            logger.info("Pine repo polling disabled (PINE_REPO_POLL_ENABLED=false)")
            POLL_ENABLED.set(0)
            return

        logger.info(
            "Starting Pine repo poller",
            tick_seconds=self._settings.pine_repo_poll_tick_seconds,
            interval_minutes=self._settings.pine_repo_poll_interval_minutes,
            max_concurrency=self._settings.pine_repo_poll_max_concurrency,
            max_repos_per_tick=self._settings.pine_repo_poll_max_repos_per_tick,
        )

        POLL_ENABLED.set(1)
        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the polling background task gracefully.

        Args:
            timeout: Max seconds to wait for current scans to complete
        """
        if not self._running:
            return

        logger.info("Stopping Pine repo poller")
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
        logger.info("Pine repo poller stopped")

    async def run_once(self) -> PollRunResult:
        """
        Run a single poll cycle (for manual triggering).

        Returns:
            PollRunResult with scan statistics
        """
        return await self._do_poll_tick()

    async def get_health(self) -> PollerHealth:
        """Get current poller health status."""
        due_count = await self._repo_registry.count_due_for_poll()

        return PollerHealth(
            enabled=self._settings.pine_repo_poll_enabled,
            running=self._running,
            last_run_at=self._last_run_at,
            last_run_repos_scanned=(
                self._last_run_result.repos_scanned if self._last_run_result else 0
            ),
            last_run_errors=(
                self._last_run_result.repos_failed if self._last_run_result else 0
            ),
            repos_due_count=due_count,
            poll_interval_minutes=self._settings.pine_repo_poll_interval_minutes,
            poll_max_concurrency=self._settings.pine_repo_poll_max_concurrency,
            poll_tick_seconds=self._settings.pine_repo_poll_tick_seconds,
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _poll_loop(self) -> None:
        """Main polling loop - runs until stop_event is set."""
        tick_seconds = self._settings.pine_repo_poll_tick_seconds

        while not self._stop_event.is_set():
            try:
                result = await self._do_poll_tick()
                self._last_run_result = result
                self._last_run_at = datetime.now(timezone.utc)

                # Update metrics
                POLL_LAST_RUN_TIMESTAMP.set(self._last_run_at.timestamp())
                if result.repos_scanned > 0 or result.errors:
                    status = "success" if not result.errors else "partial"
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

    async def _do_poll_tick(self) -> PollRunResult:
        """Execute a single poll tick."""
        import time

        start_time = time.time()
        result = PollRunResult()

        # Get repos due for scanning
        due_repos = await self._repo_registry.list_due_for_poll(
            limit=self._settings.pine_repo_poll_max_repos_per_tick,
        )

        # Update due count metric
        total_due = await self._repo_registry.count_due_for_poll()
        POLL_DUE_COUNT.set(total_due)

        if not due_repos:
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        logger.info(
            "Poll tick starting",
            repos_due=len(due_repos),
            total_due=total_due,
        )

        # Scan repos concurrently with semaphore
        tasks = [self._scan_repo_with_semaphore(repo) for repo in due_repos]
        scan_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for repo, scan_result in zip(due_repos, scan_results):
            result.repos_scanned += 1

            if isinstance(scan_result, Exception):
                result.repos_failed += 1
                result.errors.append(f"{repo.repo_slug}: {scan_result}")
                POLL_REPOS_SCANNED_TOTAL.labels(status="failure").inc()
            elif scan_result is True:
                result.repos_succeeded += 1
                POLL_REPOS_SCANNED_TOTAL.labels(status="success").inc()
            else:
                # scan_result is False - scan was skipped or had issues
                result.repos_skipped += 1
                POLL_REPOS_SCANNED_TOTAL.labels(status="failure").inc()

        result.duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Poll tick complete",
            repos_scanned=result.repos_scanned,
            repos_succeeded=result.repos_succeeded,
            repos_failed=result.repos_failed,
            duration_ms=result.duration_ms,
        )

        return result

    async def _scan_repo_with_semaphore(self, repo) -> bool:
        """
        Scan a single repo with semaphore-controlled concurrency.

        Returns:
            True if scan succeeded, False otherwise
        """
        async with self._semaphore:
            self._inflight += 1
            POLL_INFLIGHT.set(self._inflight)
            try:
                return await self._scan_repo(repo)
            finally:
                self._inflight -= 1
                POLL_INFLIGHT.set(self._inflight)

    async def _scan_repo(self, repo) -> bool:
        """
        Scan a single repository.

        Updates poll state based on success/failure.

        Returns:
            True if scan succeeded, False otherwise
        """
        from app.services.pine.discovery import PineDiscoveryService

        log = logger.bind(
            repo_id=str(repo.id),
            repo_slug=repo.repo_slug,
        )

        try:
            log.debug("Starting poll scan")

            discovery = PineDiscoveryService(
                self._pool,
                self._settings,
                self._qdrant_client,
            )

            result = await discovery.discover_repo(
                workspace_id=repo.workspace_id,
                repo_id=repo.id,
                trigger="poll",
                force_full_scan=False,
            )

            if result.status in ("success", "partial"):
                # Update poll state: reset failure count, schedule next scan
                await self._repo_registry.update_poll_success(
                    repo_id=repo.id,
                    interval_minutes=self._settings.pine_repo_poll_interval_minutes,
                )
                log.info(
                    "Poll scan succeeded",
                    status=result.status,
                    scripts_new=result.scripts_new,
                    scripts_updated=result.scripts_updated,
                )
                return True
            else:
                # Scan returned error status
                await self._repo_registry.update_poll_failure(
                    repo_id=repo.id,
                    base_interval_minutes=self._settings.pine_repo_poll_interval_minutes,
                    max_backoff_multiplier=self._settings.pine_repo_poll_backoff_max_multiplier,
                )
                log.warning(
                    "Poll scan failed",
                    status=result.status,
                    errors=result.errors[:3],  # First 3 errors
                )
                return False

        except Exception as e:
            # Unexpected error during scan
            await self._repo_registry.update_poll_failure(
                repo_id=repo.id,
                base_interval_minutes=self._settings.pine_repo_poll_interval_minutes,
                max_backoff_multiplier=self._settings.pine_repo_poll_backoff_max_multiplier,
            )
            log.exception("Poll scan error", error=str(e))
            return False


# =============================================================================
# Module-level singleton (for lifespan management)
# =============================================================================

_poller: Optional[PineRepoPoller] = None


def get_poller() -> Optional[PineRepoPoller]:
    """Get the global poller instance."""
    return _poller


def set_poller(poller: Optional[PineRepoPoller]) -> None:
    """Set the global poller instance."""
    global _poller
    _poller = poller
