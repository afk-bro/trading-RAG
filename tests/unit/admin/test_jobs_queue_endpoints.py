"""Tests for admin jobs queue endpoints (list/detail/cancel/trigger).

These test the new job queue management endpoints that work with the jobs table,
distinct from the existing job runs endpoints (which work with job_runs table).
"""

import os
from datetime import datetime, timezone
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


def make_job_dict(
    job_id=None,
    job_type="data_sync",
    status="pending",
    workspace_id=None,
    parent_job_id=None,
):
    """Create a mock job dictionary."""
    return {
        "id": job_id or uuid4(),
        "type": job_type,
        "status": status,
        "payload": {"exchange_id": "kucoin"},
        "attempt": 0,
        "max_attempts": 3,
        "run_after": datetime.now(timezone.utc),
        "locked_at": None,
        "locked_by": None,
        "parent_job_id": parent_job_id,
        "workspace_id": workspace_id,
        "dedupe_key": None,
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "priority": 100,
    }


def make_job_model(
    job_id=None,
    job_type="data_sync",
    status="pending",
    workspace_id=None,
    parent_job_id=None,
):
    """Create a mock Job model."""
    from app.jobs.models import Job
    from app.jobs.types import JobType, JobStatus

    return Job(
        id=job_id or uuid4(),
        type=JobType(job_type),
        status=JobStatus(status),
        payload={"exchange_id": "kucoin"},
        attempt=0,
        max_attempts=3,
        run_after=datetime.now(timezone.utc),
        locked_at=None,
        locked_by=None,
        parent_job_id=parent_job_id,
        workspace_id=workspace_id,
        dedupe_key=None,
        created_at=datetime.now(timezone.utc),
        started_at=None,
        completed_at=None,
        result=None,
        priority=100,
    )


def make_job_event_model(job_id, message="Test event", level="info"):
    """Create a mock JobEvent model."""
    from app.jobs.models import JobEvent

    return JobEvent(
        id=1,
        job_id=job_id,
        ts=datetime.now(timezone.utc),
        level=level,
        message=message,
        meta=None,
    )


class TestListJobsQueueEndpoint:
    """Tests for GET /admin/jobs/queue endpoint."""

    def test_list_jobs_requires_admin_token(self, client):
        """List endpoint requires admin auth."""
        response = client.get("/admin/jobs/queue")
        assert response.status_code in (401, 403)

    def test_list_jobs_success(self, client, mock_db_pool):
        """List endpoint returns jobs."""
        job_id = uuid4()
        workspace_id = uuid4()

        mock_job_repo = MagicMock()
        mock_job = make_job_model(
            job_id=job_id, workspace_id=workspace_id, status="running"
        )
        mock_job_repo.list_jobs = AsyncMock(return_value=([mock_job], 1))

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.get(
                "/admin/jobs/queue",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 1
        assert data["items"][0]["type"] == "data_sync"
        assert data["items"][0]["status"] == "running"
        assert data["total"] == 1
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_list_jobs_with_filters(self, client, mock_db_pool):
        """List endpoint accepts filters."""
        workspace_id = uuid4()

        mock_job_repo = MagicMock()
        mock_job_repo.list_jobs = AsyncMock(return_value=([], 0))

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.get(
                f"/admin/jobs/queue?status=failed&type=data_sync"
                f"&workspace_id={workspace_id}&limit=10&offset=5",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        # Verify filters were passed to repo
        mock_job_repo.list_jobs.assert_called_once()
        call_kwargs = mock_job_repo.list_jobs.call_args[1]
        assert call_kwargs["status"] == "failed"
        assert call_kwargs["job_type"] == "data_sync"
        assert call_kwargs["workspace_id"] == workspace_id
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 5

    def test_list_jobs_db_unavailable(self, client):
        """List endpoint returns 503 when DB not available."""
        with patch("app.admin.jobs._db_pool", None):
            response = client.get(
                "/admin/jobs/queue",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 503


class TestGetJobDetailEndpoint:
    """Tests for GET /admin/jobs/queue/{job_id} endpoint."""

    def test_get_job_requires_admin_token(self, client):
        """Detail endpoint requires admin auth."""
        job_id = uuid4()
        response = client.get(f"/admin/jobs/queue/{job_id}")
        assert response.status_code in (401, 403)

    def test_get_job_success(self, client, mock_db_pool):
        """Detail endpoint returns job with events and children."""
        job_id = uuid4()
        child_id = uuid4()

        mock_job = make_job_model(job_id=job_id, status="running")
        mock_child = make_job_model(
            job_id=child_id,
            parent_job_id=job_id,
            status="pending",
            job_type="data_fetch",
        )
        mock_event = make_job_event_model(job_id, message="Job started")

        mock_job_repo = MagicMock()
        mock_job_repo.get = AsyncMock(return_value=mock_job)
        mock_job_repo.list_by_parent = AsyncMock(return_value=[mock_child])

        mock_events_repo = MagicMock()
        mock_events_repo.list_for_job = AsyncMock(return_value=[mock_event])

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ), patch(
            "app.repositories.job_events.JobEventsRepository",
            return_value=mock_events_repo,
        ):
            response = client.get(
                f"/admin/jobs/queue/{job_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "job" in data
        assert data["job"]["status"] == "running"
        assert "events" in data
        assert len(data["events"]) == 1
        assert "children" in data
        assert len(data["children"]) == 1
        assert data["children"][0]["type"] == "data_fetch"

    def test_get_job_not_found(self, client, mock_db_pool):
        """Detail endpoint returns 404 when not found."""
        job_id = uuid4()

        mock_job_repo = MagicMock()
        mock_job_repo.get = AsyncMock(return_value=None)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.get(
                f"/admin/jobs/queue/{job_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestCancelJobEndpoint:
    """Tests for POST /admin/jobs/queue/{job_id}/cancel endpoint."""

    def test_cancel_job_requires_admin_token(self, client):
        """Cancel endpoint requires admin auth."""
        job_id = uuid4()
        response = client.post(f"/admin/jobs/queue/{job_id}/cancel")
        assert response.status_code in (401, 403)

    def test_cancel_job_success(self, client, mock_db_pool):
        """Cancel endpoint cancels job and children."""
        job_id = uuid4()

        mock_job = make_job_model(job_id=job_id, status="canceled")

        mock_job_repo = MagicMock()
        mock_job_repo.cancel_job_tree = AsyncMock(
            return_value=(mock_job, 1)  # job + 1 child canceled
        )

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.post(
                f"/admin/jobs/queue/{job_id}/cancel",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "job" in data
        assert data["job"]["status"] == "canceled"
        assert data["children_canceled"] == 1

    def test_cancel_job_not_found(self, client, mock_db_pool):
        """Cancel endpoint returns 404 when job not found."""
        job_id = uuid4()

        mock_job_repo = MagicMock()
        mock_job_repo.cancel_job_tree = AsyncMock(return_value=(None, 0))

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.post(
                f"/admin/jobs/queue/{job_id}/cancel",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404


class TestTriggerSyncEndpoint:
    """Tests for POST /admin/jobs/sync/trigger endpoint."""

    def test_trigger_sync_requires_admin_token(self, client):
        """Trigger endpoint requires admin auth."""
        response = client.post("/admin/jobs/sync/trigger")
        assert response.status_code in (401, 403)

    def test_trigger_sync_success(self, client, mock_db_pool):
        """Trigger endpoint creates a data sync job."""
        job_id = uuid4()
        mock_job = make_job_model(job_id=job_id, status="pending")

        mock_job_repo = MagicMock()
        mock_job_repo.create = AsyncMock(return_value=mock_job)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.post(
                "/admin/jobs/sync/trigger",
                headers={"X-Admin-Token": "test-token"},
                json={},  # No body required
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

        # Verify job was created with correct type
        mock_job_repo.create.assert_called_once()
        call_kwargs = mock_job_repo.create.call_args[1]
        assert call_kwargs["job_type"].value == "data_sync"

    def test_trigger_sync_with_exchange(self, client, mock_db_pool):
        """Trigger endpoint accepts exchange_id filter."""
        job_id = uuid4()
        mock_job = make_job_model(job_id=job_id, status="pending")

        mock_job_repo = MagicMock()
        mock_job_repo.create = AsyncMock(return_value=mock_job)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.post(
                "/admin/jobs/sync/trigger",
                headers={"X-Admin-Token": "test-token"},
                json={"exchange_id": "kucoin", "mode": "full"},
            )

        assert response.status_code == 200

        # Verify payload includes exchange_id and mode
        mock_job_repo.create.assert_called_once()
        call_kwargs = mock_job_repo.create.call_args[1]
        assert call_kwargs["payload"]["exchange_id"] == "kucoin"
        assert call_kwargs["payload"]["mode"] == "full"

    def test_trigger_sync_default_mode(self, client, mock_db_pool):
        """Trigger endpoint defaults to incremental mode."""
        job_id = uuid4()
        mock_job = make_job_model(job_id=job_id, status="pending")

        mock_job_repo = MagicMock()
        mock_job_repo.create = AsyncMock(return_value=mock_job)

        with patch("app.admin.jobs._db_pool", mock_db_pool), patch(
            "app.repositories.jobs.JobRepository", return_value=mock_job_repo
        ):
            response = client.post(
                "/admin/jobs/sync/trigger",
                headers={"X-Admin-Token": "test-token"},
                json={},
            )

        assert response.status_code == 200

        # Verify default mode is incremental
        call_kwargs = mock_job_repo.create.call_args[1]
        assert call_kwargs["payload"]["mode"] == "incremental"


class TestJobsQueueDbUnavailable:
    """Test DB unavailable scenarios for all endpoints."""

    def test_get_job_db_unavailable(self, client):
        """Detail endpoint returns 503 when DB not available."""
        with patch("app.admin.jobs._db_pool", None):
            response = client.get(
                f"/admin/jobs/queue/{uuid4()}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 503

    def test_cancel_job_db_unavailable(self, client):
        """Cancel endpoint returns 503 when DB not available."""
        with patch("app.admin.jobs._db_pool", None):
            response = client.post(
                f"/admin/jobs/queue/{uuid4()}/cancel",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 503

    def test_trigger_sync_db_unavailable(self, client):
        """Trigger endpoint returns 503 when DB not available."""
        with patch("app.admin.jobs._db_pool", None):
            response = client.post(
                "/admin/jobs/sync/trigger",
                headers={"X-Admin-Token": "test-token"},
                json={},
            )

        assert response.status_code == 503
