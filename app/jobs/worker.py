"""Job worker - claims and executes jobs from the queue."""
import asyncio
import os
import socket
import traceback
from typing import Optional

import structlog

from app import __version__
from app.config import get_settings
from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType
from app.repositories.jobs import JobRepository
from app.repositories.job_events import JobEventsRepository

logger = structlog.get_logger(__name__)


def generate_worker_id() -> str:
    """Generate a unique worker ID: hostname:pid."""
    return f"{socket.gethostname()}:{os.getpid()}"


class WorkerRunner:
    """Job worker that polls and executes jobs."""

    def __init__(
        self,
        pool,
        worker_id: Optional[str] = None,
        job_types: Optional[list[JobType]] = None,
    ):
        self._pool = pool
        self._worker_id = worker_id or generate_worker_id()
        self._job_types = job_types  # None = all types
        self._running = False
        self._job_repo: Optional[JobRepository] = None
        self._events_repo: Optional[JobEventsRepository] = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self):
        """Start the worker loop."""
        settings = get_settings()
        self._running = True
        self._job_repo = JobRepository(self._pool)
        self._events_repo = JobEventsRepository(self._pool)

        logger.info(
            "worker_started",
            worker_id=self._worker_id,
            version=__version__,
            job_types=[jt.value for jt in self._job_types] if self._job_types else "all",
        )

        # Register heartbeat
        await self._heartbeat()

        poll_interval = settings.job_poll_interval_s
        stale_timeout = settings.job_stale_timeout_minutes
        heartbeat_interval = 10  # seconds
        reap_interval = 60  # seconds

        last_heartbeat = asyncio.get_event_loop().time()
        last_reap = asyncio.get_event_loop().time()

        while self._running:
            try:
                # Try to claim a job
                job = await self._job_repo.claim(self._worker_id, self._job_types)

                if job:
                    await self._execute_job(job)
                else:
                    # No job available, sleep
                    await asyncio.sleep(poll_interval)

                # Periodic heartbeat
                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= heartbeat_interval:
                    await self._heartbeat()
                    last_heartbeat = now

                # Periodic stale job reaping
                if now - last_reap >= reap_interval:
                    await self._job_repo.reap_stale(stale_timeout)
                    last_reap = now

            except asyncio.CancelledError:
                logger.info("worker_cancelled", worker_id=self._worker_id)
                break
            except Exception as e:
                logger.error(
                    "worker_loop_error", error=str(e), traceback=traceback.format_exc()
                )
                await asyncio.sleep(poll_interval)

        logger.info("worker_stopped", worker_id=self._worker_id)

    async def stop(self):
        """Stop the worker loop gracefully."""
        self._running = False

    async def _execute_job(self, job: Job):
        """Execute a single job."""
        # These are guaranteed to be set by start() before _execute_job is called
        assert self._job_repo is not None
        assert self._events_repo is not None

        log = logger.bind(job_id=str(job.id), job_type=job.type.value)
        log.info("job_executing")

        try:
            # Get handler from registry
            handler = default_registry.get_handler(job.type)

            # Log job start
            await self._events_repo.info(job.id, "Job execution started")

            # Execute handler
            context = {
                "worker_id": self._worker_id,
                "pool": self._pool,
                "job_repo": self._job_repo,
                "events_repo": self._events_repo,
            }
            result = await handler(job, context)

            # Mark job complete
            await self._job_repo.complete(job.id, result)
            await self._events_repo.info(job.id, "Job completed successfully")
            log.info("job_succeeded")

        except KeyError:
            # No handler registered
            error = f"No handler registered for job type: {job.type.value}"
            log.error("job_no_handler", error=error)
            await self._events_repo.error(job.id, error)
            await self._job_repo.fail(job.id, error, should_retry=False)

        except Exception as e:
            # Handler failed
            error = str(e)
            tb = traceback.format_exc()
            log.error("job_handler_failed", error=error)
            await self._events_repo.error(job.id, error, traceback=tb)
            await self._job_repo.fail(job.id, error, should_retry=True)

    async def _heartbeat(self):
        """Update worker heartbeat."""
        query = """
            INSERT INTO worker_heartbeats (worker_id, version, last_seen)
            VALUES ($1, $2, now())
            ON CONFLICT (worker_id) DO UPDATE SET
                version = $2,
                last_seen = now()
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(query, self._worker_id, __version__)
        except Exception as e:
            logger.warning("heartbeat_failed", error=str(e))
