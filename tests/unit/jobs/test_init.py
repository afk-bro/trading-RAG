"""Test job package exports."""


def test_job_package_exports():
    from app.jobs import (
        JobType,
        JobStatus,
        Job,
        JobEvent,
        JobRegistry,
        default_registry,
    )

    assert JobType.TUNE == "tune"
    assert JobStatus.PENDING == "pending"
    assert Job is not None
    assert JobEvent is not None
    assert JobRegistry is not None
    assert default_registry is not None
