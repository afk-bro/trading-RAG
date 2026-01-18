"""Job system type definitions."""

from enum import Enum


class JobType(str, Enum):
    """Job types for the automation system."""

    DATA_SYNC = "data_sync"
    DATA_FETCH = "data_fetch"
    TUNE = "tune"
    WFO = "wfo"


class JobStatus(str, Enum):
    """Job lifecycle statuses."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def is_terminal(self) -> bool:
        """Check if this status is terminal (job won't change)."""
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED)
