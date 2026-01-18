"""Tests for job models."""

from uuid import uuid4

from app.jobs.models import Job, JobEvent
from app.jobs.types import JobType, JobStatus


class TestJob:
    def test_create_job(self):
        job = Job(
            id=uuid4(),
            type=JobType.DATA_FETCH,
            status=JobStatus.PENDING,
            payload={"symbol": "BTC-USDT"},
        )
        assert job.type == JobType.DATA_FETCH
        assert job.status == JobStatus.PENDING
        assert job.attempt == 0
        assert job.max_attempts == 3

    def test_job_with_workspace(self):
        ws_id = uuid4()
        job = Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.PENDING,
            payload={},
            workspace_id=ws_id,
        )
        assert job.workspace_id == ws_id

    def test_job_with_parent(self):
        parent_id = uuid4()
        job = Job(
            id=uuid4(),
            type=JobType.TUNE,
            status=JobStatus.PENDING,
            payload={},
            parent_job_id=parent_id,
        )
        assert job.parent_job_id == parent_id


class TestJobEvent:
    def test_create_event(self):
        job_id = uuid4()
        event = JobEvent(
            job_id=job_id,
            level="info",
            message="Job started",
        )
        assert event.job_id == job_id
        assert event.level == "info"
        assert event.meta is None

    def test_event_with_meta(self):
        event = JobEvent(
            job_id=uuid4(),
            level="error",
            message="Fetch failed",
            meta={"error_code": "RATE_LIMIT"},
        )
        assert event.meta == {"error_code": "RATE_LIMIT"}
