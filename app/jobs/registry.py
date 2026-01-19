"""Job handler registry."""

from typing import Any, Callable, Coroutine

from app.jobs.types import JobType
from app.jobs.models import Job

# Handler signature: async def handler(job: Job, ctx: dict) -> dict
JobHandler = Callable[[Job, dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]]


class JobRegistry:
    """Registry mapping job types to their handlers."""

    def __init__(self):
        self._handlers: dict[JobType, JobHandler] = {}

    def register(self, job_type: JobType, handler: JobHandler) -> None:
        """Register a handler for a job type."""
        self._handlers[job_type] = handler

    def get_handler(self, job_type: JobType) -> JobHandler:
        """Get the handler for a job type. Raises KeyError if not found."""
        if job_type not in self._handlers:
            raise KeyError(f"No handler registered for job type: {job_type}")
        return self._handlers[job_type]

    def handler(self, job_type: JobType) -> Callable[[JobHandler], JobHandler]:
        """Decorator to register a handler."""

        def decorator(fn: JobHandler) -> JobHandler:
            self.register(job_type, fn)
            return fn

        return decorator


# Global registry instance
default_registry = JobRegistry()
