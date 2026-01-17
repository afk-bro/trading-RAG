"""Tests for retention admin endpoints with JobRunner."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


# Set required environment variables for tests before importing app
os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    return MagicMock()


@pytest.fixture
def workspace_id():
    """Create test workspace ID."""
    return uuid4()


@pytest.fixture
def mock_job_result_completed():
    """Create mock JobResult for completed job."""
    from app.services.jobs import JobResult

    return JobResult(
        run_id=uuid4(),
        lock_acquired=True,
        status="completed",
        duration_ms=150,
        metrics={"rows_affected": 42, "target_date": "2025-01-01", "dry_run": False},
        correlation_id="job-rollup_events-abc12345",
    )


@pytest.fixture
def mock_job_result_already_running():
    """Create mock JobResult for lock not acquired."""
    from app.services.jobs import JobResult

    return JobResult(
        run_id=None,
        lock_acquired=False,
        status="already_running",
        duration_ms=0,
        metrics={},
        correlation_id="job-rollup_events-def67890",
    )


class TestRollupEndpoint:
    """Tests for /admin/jobs/rollup-events endpoint."""

    def test_rollup_requires_admin_token(self, client, workspace_id):
        """Rollup endpoint requires admin auth."""
        response = client.post(f"/admin/jobs/rollup-events?workspace_id={workspace_id}")
        assert response.status_code in (401, 403)

    def test_rollup_requires_workspace_id(self, client):
        """Rollup endpoint requires workspace_id parameter."""
        response = client.post(
            "/admin/jobs/rollup-events",
            headers={"X-Admin-Token": "test-token"},
        )
        assert response.status_code == 422  # Validation error

    def test_rollup_success(
        self, client, mock_db_pool, workspace_id, mock_job_result_completed
    ):
        """Rollup endpoint returns 200 with JobResult on success."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_job_result_completed)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/rollup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["lock_acquired"] is True
        assert data["status"] == "completed"
        assert data["duration_ms"] == 150
        assert "correlation_id" in data

    def test_rollup_already_running_returns_409(
        self, client, mock_db_pool, workspace_id, mock_job_result_already_running
    ):
        """Rollup endpoint returns 409 when lock not acquired."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_job_result_already_running)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/rollup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 409
        data = response.json()
        assert data["lock_acquired"] is False
        assert data["status"] == "already_running"

    def test_rollup_with_custom_date(
        self, client, mock_db_pool, workspace_id, mock_job_result_completed
    ):
        """Rollup endpoint accepts custom target date."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_job_result_completed)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/rollup-events?workspace_id={workspace_id}&target_date=2025-01-01",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        # Verify JobRunner.run was called
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["job_name"] == "rollup_events"
        assert call_kwargs["workspace_id"] == workspace_id
        assert call_kwargs["triggered_by"] == "admin_token"

    def test_rollup_dry_run(self, client, mock_db_pool, workspace_id):
        """Rollup endpoint supports dry_run parameter."""
        from app.services.jobs import JobResult

        dry_run_result = JobResult(
            run_id=uuid4(),
            lock_acquired=True,
            status="completed",
            duration_ms=50,
            metrics={
                "dry_run": True,
                "target_date": "2025-01-01",
                "events_to_aggregate": 100,
                "rollup_rows_to_create": 5,
            },
            correlation_id="job-rollup_events-dryrun",
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=dry_run_result)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/rollup-events?workspace_id={workspace_id}&dry_run=true",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["dry_run"] is True
        # Verify dry_run was passed to JobRunner
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["dry_run"] is True

    def test_rollup_exception_returns_500(self, client, mock_db_pool, workspace_id):
        """Rollup endpoint returns 500 on unexpected exception."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(
            side_effect=RuntimeError("Database connection lost")
        )

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/rollup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "failed"
        assert "Database connection lost" in data["error"]


class TestCleanupEndpoint:
    """Tests for /admin/jobs/cleanup-events endpoint."""

    def test_cleanup_requires_admin_token(self, client, workspace_id):
        """Cleanup endpoint requires admin auth."""
        response = client.post(
            f"/admin/jobs/cleanup-events?workspace_id={workspace_id}"
        )
        assert response.status_code in (401, 403)

    def test_cleanup_requires_workspace_id(self, client):
        """Cleanup endpoint requires workspace_id parameter."""
        response = client.post(
            "/admin/jobs/cleanup-events",
            headers={"X-Admin-Token": "test-token"},
        )
        assert response.status_code == 422  # Validation error

    def test_cleanup_success(self, client, mock_db_pool, workspace_id):
        """Cleanup endpoint returns 200 with JobResult on success."""
        from app.services.jobs import JobResult

        cleanup_result = JobResult(
            run_id=uuid4(),
            lock_acquired=True,
            status="completed",
            duration_ms=200,
            metrics={
                "dry_run": False,
                "info_debug_deleted": 10,
                "warn_error_deleted": 5,
            },
            correlation_id="job-cleanup_events-abc12345",
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=cleanup_result)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/cleanup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["lock_acquired"] is True
        assert data["status"] == "completed"
        assert data["metrics"]["info_debug_deleted"] == 10
        assert data["metrics"]["warn_error_deleted"] == 5

    def test_cleanup_already_running_returns_409(
        self, client, mock_db_pool, workspace_id, mock_job_result_already_running
    ):
        """Cleanup endpoint returns 409 when lock not acquired."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_job_result_already_running)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/cleanup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 409
        data = response.json()
        assert data["lock_acquired"] is False
        assert data["status"] == "already_running"

    def test_cleanup_dry_run(self, client, mock_db_pool, workspace_id):
        """Cleanup endpoint supports dry_run parameter."""
        from app.services.jobs import JobResult

        dry_run_result = JobResult(
            run_id=uuid4(),
            lock_acquired=True,
            status="completed",
            duration_ms=30,
            metrics={
                "dry_run": True,
                "info_debug_would_delete": 50,
                "warn_error_would_delete": 10,
            },
            correlation_id="job-cleanup_events-dryrun",
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=dry_run_result)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/cleanup-events?workspace_id={workspace_id}&dry_run=true",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["dry_run"] is True
        assert data["metrics"]["info_debug_would_delete"] == 50
        # Verify dry_run was passed to JobRunner
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["dry_run"] is True

    def test_cleanup_exception_returns_500(self, client, mock_db_pool, workspace_id):
        """Cleanup endpoint returns 500 on unexpected exception."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(side_effect=RuntimeError("Disk full"))

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/cleanup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "failed"
        assert "Disk full" in data["error"]


class TestJobConcurrency:
    """Tests for job concurrency handling."""

    def test_concurrent_job_returns_409(self, client, mock_db_pool, workspace_id):
        """Second job attempt while first running returns 409."""
        from app.services.jobs import JobResult

        # Simulate lock not acquired (another job is running)
        lock_not_acquired_result = JobResult(
            run_id=None,
            lock_acquired=False,
            status="already_running",
            duration_ms=0,
            metrics={},
            correlation_id="job-rollup_events-concurrent",
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=lock_not_acquired_result)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/rollup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 409
        data = response.json()
        assert data["lock_acquired"] is False
        assert data["status"] == "already_running"
        assert data["run_id"] is None

    def test_cleanup_concurrent_returns_409(self, client, mock_db_pool, workspace_id):
        """Cleanup job also returns 409 when lock not acquired."""
        from app.services.jobs import JobResult

        lock_not_acquired_result = JobResult(
            run_id=None,
            lock_acquired=False,
            status="already_running",
            duration_ms=0,
            metrics={},
            correlation_id="job-cleanup_events-concurrent",
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=lock_not_acquired_result)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/cleanup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 409
        data = response.json()
        assert data["lock_acquired"] is False


class TestJobTriggeredBy:
    """Tests for triggered_by tracking."""

    def test_rollup_records_admin_token_trigger(
        self, client, mock_db_pool, workspace_id, mock_job_result_completed
    ):
        """Rollup job records admin_token as trigger."""
        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=mock_job_result_completed)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/rollup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        # Verify triggered_by was passed to JobRunner
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["triggered_by"] == "admin_token"

    def test_cleanup_records_admin_token_trigger(
        self, client, mock_db_pool, workspace_id
    ):
        """Cleanup job records admin_token as trigger."""
        from app.services.jobs import JobResult

        cleanup_result = JobResult(
            run_id=uuid4(),
            lock_acquired=True,
            status="completed",
            duration_ms=100,
            metrics={"dry_run": False, "info_debug_deleted": 5},
            correlation_id="job-cleanup-trigger-test",
        )

        mock_runner = MagicMock()
        mock_runner.run = AsyncMock(return_value=cleanup_result)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.services.jobs.JobRunner", return_value=mock_runner
        ):
            response = client.post(
                f"/admin/jobs/cleanup-events?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["triggered_by"] == "admin_token"
