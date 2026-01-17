"""Unit tests for admin run plans endpoints."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

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
def mock_run_plans_repo():
    """Create mock run plans repository."""
    repo = MagicMock()
    repo.get_run_plan = AsyncMock()
    repo.list_runs_for_plan = AsyncMock()
    return repo


class TestGetRunPlan:
    """Tests for GET /admin/run-plans/{id}."""

    def test_get_run_plan_not_found_returns_404(self, client, mock_run_plans_repo):
        """GET /admin/run-plans/{id} returns 404 if plan not found."""
        plan_id = uuid4()
        mock_run_plans_repo.get_run_plan.return_value = None

        with patch(
            "app.admin.run_plans._get_run_plans_repo", return_value=mock_run_plans_repo
        ):
            response = client.get(
                f"/admin/run-plans/{plan_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
        assert response.json()["detail"] == "Run plan not found"

    def test_get_run_plan_returns_plan(self, client, mock_run_plans_repo):
        """GET /admin/run-plans/{id} returns plan data."""
        plan_id = uuid4()
        mock_run_plans_repo.get_run_plan.return_value = {
            "id": plan_id,
            "status": "completed",
            "n_variants": 10,
            "plan": {"inputs": {}, "resolved": {}, "provenance": {}},
        }

        with patch(
            "app.admin.run_plans._get_run_plans_repo", return_value=mock_run_plans_repo
        ):
            response = client.get(
                f"/admin/run-plans/{plan_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(plan_id)
        assert data["status"] == "completed"


class TestGetRunPlanRuns:
    """Tests for GET /admin/run-plans/{id}/runs."""

    def test_get_run_plan_runs_not_found_returns_404(self, client, mock_run_plans_repo):
        """GET /admin/run-plans/{id}/runs returns 404 if plan not found."""
        plan_id = uuid4()
        mock_run_plans_repo.get_run_plan.return_value = None

        with patch(
            "app.admin.run_plans._get_run_plans_repo", return_value=mock_run_plans_repo
        ):
            response = client.get(
                f"/admin/run-plans/{plan_id}/runs",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404

    def test_get_run_plan_runs_returns_runs(self, client, mock_run_plans_repo):
        """GET /admin/run-plans/{id}/runs returns runs list."""
        plan_id = uuid4()
        run_id = uuid4()

        mock_run_plans_repo.get_run_plan.return_value = {
            "id": plan_id,
            "status": "completed",
        }
        mock_run_plans_repo.list_runs_for_plan.return_value = (
            [
                {
                    "id": run_id,
                    "variant_index": 0,
                    "status": "completed",
                    "objective_score": 1.42,
                }
            ],
            1,
        )

        with patch(
            "app.admin.run_plans._get_run_plans_repo", return_value=mock_run_plans_repo
        ):
            response = client.get(
                f"/admin/run-plans/{plan_id}/runs",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["runs"]) == 1
        assert data["runs"][0]["id"] == str(run_id)


class TestRequiresAuth:
    """Tests that endpoints require authentication."""

    def test_get_run_plan_requires_auth(self, client):
        """GET /admin/run-plans/{id} requires auth."""
        plan_id = uuid4()

        response = client.get(f"/admin/run-plans/{plan_id}")

        # Should return 401 or 403 without valid token
        assert response.status_code in [401, 403]

    def test_get_run_plan_runs_requires_auth(self, client):
        """GET /admin/run-plans/{id}/runs requires auth."""
        plan_id = uuid4()

        response = client.get(f"/admin/run-plans/{plan_id}/runs")

        assert response.status_code in [401, 403]
