"""Tests for job system types."""

from app.jobs.types import JobType, JobStatus


class TestJobType:
    def test_job_types_exist(self):
        assert JobType.DATA_SYNC == "data_sync"
        assert JobType.DATA_FETCH == "data_fetch"
        assert JobType.TUNE == "tune"
        assert JobType.WFO == "wfo"

    def test_job_type_has_required_members(self):
        """Required job types exist - won't break when new types added."""
        required = {"DATA_SYNC", "DATA_FETCH", "TUNE", "WFO"}
        actual = {jt.name for jt in JobType}
        missing = required - actual
        assert not missing, f"Missing JobType members: {missing}"


class TestJobStatus:
    def test_job_statuses_exist(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.SUCCEEDED == "succeeded"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELED == "canceled"

    def test_terminal_statuses(self):
        assert JobStatus.SUCCEEDED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELED.is_terminal
        assert not JobStatus.PENDING.is_terminal
        assert not JobStatus.RUNNING.is_terminal
