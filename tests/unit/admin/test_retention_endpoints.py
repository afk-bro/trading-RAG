"""Tests for retention admin endpoints."""

import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Set admin token for tests before importing app
os.environ["ADMIN_TOKEN"] = "test-token"


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
def mock_event_rollups_repo():
    """Create mock event rollups repository."""
    repo = MagicMock()
    repo.run_daily_rollup = AsyncMock(return_value=42)
    return repo


@pytest.fixture
def mock_retention_service():
    """Create mock retention service."""
    service = MagicMock()
    service.run_cleanup = AsyncMock(
        return_value={"info_debug_deleted": 10, "warn_error_deleted": 5}
    )
    return service


class TestRollupEndpoint:
    """Tests for /admin/jobs/rollup-events endpoint."""

    def test_rollup_requires_admin_token(self, client):
        """Rollup endpoint requires admin auth."""
        response = client.post("/admin/jobs/rollup-events")
        assert response.status_code in (401, 403)

    def test_rollup_with_valid_token(
        self, client, mock_db_pool, mock_event_rollups_repo
    ):
        """Rollup endpoint works with valid admin token."""
        # Patch the db pool and repository
        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.repositories.event_rollups.EventRollupsRepository",
            return_value=mock_event_rollups_repo,
        ):
            response = client.post(
                "/admin/jobs/rollup-events",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "target_date" in data
        assert data["rows_affected"] == 42

    def test_rollup_with_custom_date(
        self, client, mock_db_pool, mock_event_rollups_repo
    ):
        """Rollup endpoint accepts custom target date."""
        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.repositories.event_rollups.EventRollupsRepository",
            return_value=mock_event_rollups_repo,
        ):
            response = client.post(
                "/admin/jobs/rollup-events?target_date=2025-01-01",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["target_date"] == "2025-01-01"
        mock_event_rollups_repo.run_daily_rollup.assert_called_once_with(
            date(2025, 1, 1)
        )


class TestCleanupEndpoint:
    """Tests for /admin/jobs/cleanup-events endpoint."""

    def test_cleanup_requires_admin_token(self, client):
        """Cleanup endpoint requires admin auth."""
        response = client.post("/admin/jobs/cleanup-events")
        assert response.status_code in (401, 403)

    def test_cleanup_with_valid_token(
        self, client, mock_db_pool, mock_retention_service
    ):
        """Cleanup endpoint works with valid admin token."""
        with patch("app.admin.router._db_pool", mock_db_pool), patch(
            "app.services.retention.RetentionService",
            return_value=mock_retention_service,
        ):
            response = client.post(
                "/admin/jobs/cleanup-events",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["info_debug_deleted"] == 10
        assert data["warn_error_deleted"] == 5
