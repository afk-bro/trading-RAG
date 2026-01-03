"""Job status endpoint."""

from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status

from app.schemas import JobResponse, JobStatus

router = APIRouter()
logger = structlog.get_logger(__name__)

# In-memory job store (replace with Redis/DB in production)
_jobs: dict[str, dict] = {}


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    responses={
        200: {"description": "Job status retrieved"},
        404: {"description": "Job not found"},
    },
)
async def get_job_status(job_id: UUID) -> JobResponse:
    """
    Get the status of a background job.

    Job statuses:
    - started: Job has been created and is queued
    - running: Job is currently processing
    - completed: Job finished successfully
    - failed: Job encountered an error

    Progress is reported as a percentage (0-100).
    Error message is provided if status is 'failed'.
    """
    logger.info("Getting job status", job_id=str(job_id))

    job = _jobs.get(str(job_id))
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return JobResponse(
        job_id=job_id,
        status=JobStatus(job["status"]),
        progress=job.get("progress", 0.0),
        error=job.get("error"),
    )


# Helper functions for job management (used by other modules)
def create_job(job_id: UUID) -> None:
    """Create a new job record."""
    _jobs[str(job_id)] = {
        "status": "started",
        "progress": 0.0,
        "error": None,
    }


def update_job_progress(job_id: UUID, progress: float) -> None:
    """Update job progress."""
    if str(job_id) in _jobs:
        _jobs[str(job_id)]["status"] = "running"
        _jobs[str(job_id)]["progress"] = progress


def complete_job(job_id: UUID) -> None:
    """Mark job as completed."""
    if str(job_id) in _jobs:
        _jobs[str(job_id)]["status"] = "completed"
        _jobs[str(job_id)]["progress"] = 100.0


def fail_job(job_id: UUID, error: str) -> None:
    """Mark job as failed."""
    if str(job_id) in _jobs:
        _jobs[str(job_id)]["status"] = "failed"
        _jobs[str(job_id)]["error"] = error
