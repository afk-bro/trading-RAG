"""Job system data models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from app.jobs.types import JobType, JobStatus


@dataclass
class Job:
    """A job in the queue."""

    id: UUID
    type: JobType
    status: JobStatus
    payload: dict[str, Any]

    # Retry handling
    attempt: int = 0
    max_attempts: int = 3
    run_after: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Lock info
    locked_at: Optional[datetime] = None
    locked_by: Optional[str] = None

    # Relationships
    parent_job_id: Optional[UUID] = None
    workspace_id: Optional[UUID] = None
    dedupe_key: Optional[str] = None

    # Lifecycle timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Result (populated on completion)
    result: Optional[dict[str, Any]] = None
    priority: int = 100


@dataclass
class JobEvent:
    """An event logged during job execution."""

    job_id: UUID
    level: str  # "info", "warn", "error"
    message: str
    meta: Optional[dict[str, Any]] = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: Optional[int] = None
