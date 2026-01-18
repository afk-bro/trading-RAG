"""Job handlers package.

This package contains handler implementations for different job types.
Handlers are registered with the default_registry and called by the worker.

Usage:
    # Import handlers to register them with the registry
    import app.jobs.handlers.data_fetch  # noqa: F401

Handler contract:
    async def handle_<job_type>(job: Job, ctx: dict) -> dict:
        - job: The Job model with payload and metadata
        - ctx: Context dict with pool, job_repo, events_repo, worker_id
        - Returns: Result dict stored in job.result on success
"""

# Import handlers to trigger registration
from app.jobs.handlers import data_fetch  # noqa: F401
from app.jobs.handlers import data_sync  # noqa: F401
from app.jobs.handlers import tune  # noqa: F401
from app.jobs.handlers import wfo  # noqa: F401

__all__ = ["data_fetch", "data_sync", "tune", "wfo"]
