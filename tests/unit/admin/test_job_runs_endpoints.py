"""Tests for job runs list/detail endpoints."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime

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


class TestListJobRunsEndpoint:
    """Tests for /admin/jobs/runs endpoint."""

    def test_list_runs_requires_admin_token(self, client):
        """List endpoint requires admin auth."""
        response = client.get("/admin/jobs/runs")
        assert response.status_code in (401, 403)

    def test_list_runs_success(self, client, mock_db_pool):
        """List endpoint returns runs."""
        run_id = uuid4()
        workspace_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_runs = AsyncMock(
            return_value=[
                {
                    "id": run_id,
                    "job_name": "rollup_events",
                    "workspace_id": workspace_id,
                    "status": "completed",
                    "started_at": datetime.now(),
                    "finished_at": datetime.now(),
                    "duration_ms": 150,
                    "dry_run": False,
                    "metrics_preview": '{"rows": 10}',
                    "display_status": "completed",
                }
            ]
        )
        mock_repo.count_runs = AsyncMock(return_value=1)

        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.repositories.job_runs.JobRunsRepository", return_value=mock_repo
        ):
            response = client.get(
                "/admin/jobs/runs",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert len(data["runs"]) == 1
        assert data["runs"][0]["job_name"] == "rollup_events"
        assert data["total"] == 1

    def test_list_runs_with_filters(self, client, mock_db_pool):
        """List endpoint accepts filters."""
        workspace_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_runs = AsyncMock(return_value=[])
        mock_repo.count_runs = AsyncMock(return_value=0)

        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.repositories.job_runs.JobRunsRepository", return_value=mock_repo
        ):
            url = (
                f"/admin/jobs/runs?job_name=cleanup_events&workspace_id={workspace_id}"
            )
            url += "&status=failed&limit=10&offset=5"
            response = client.get(
                url,
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        # Verify filters were passed to repo
        mock_repo.list_runs.assert_called_once()
        call_kwargs = mock_repo.list_runs.call_args[1]
        assert call_kwargs["job_name"] == "cleanup_events"
        assert call_kwargs["workspace_id"] == workspace_id
        assert call_kwargs["status"] == "failed"
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 5


class TestGetJobRunEndpoint:
    """Tests for /admin/jobs/runs/{run_id} endpoint."""

    def test_get_run_requires_admin_token(self, client):
        """Detail endpoint requires admin auth."""
        run_id = uuid4()
        response = client.get(f"/admin/jobs/runs/{run_id}")
        assert response.status_code in (401, 403)

    def test_get_run_success(self, client, mock_db_pool):
        """Detail endpoint returns run."""
        run_id = uuid4()
        workspace_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_run = AsyncMock(
            return_value={
                "id": run_id,
                "job_name": "rollup_events",
                "workspace_id": workspace_id,
                "status": "completed",
                "started_at": datetime.now(),
                "finished_at": datetime.now(),
                "duration_ms": 150,
                "dry_run": False,
                "metrics": {"rows": 10, "target_date": "2026-01-10"},
                "error": None,
                "display_status": "completed",
            }
        )

        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.repositories.job_runs.JobRunsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/jobs/runs/{run_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["job_name"] == "rollup_events"
        assert data["metrics"]["rows"] == 10

    def test_get_run_not_found(self, client, mock_db_pool):
        """Detail endpoint returns 404 when not found."""
        run_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_run = AsyncMock(return_value=None)

        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.repositories.job_runs.JobRunsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/jobs/runs/{run_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_run_invalid_uuid(self, client):
        """Detail endpoint validates UUID format."""
        response = client.get(
            "/admin/jobs/runs/not-a-uuid",
            headers={"X-Admin-Token": "test-token"},
        )
        assert response.status_code == 422  # Validation error
