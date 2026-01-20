"""Regression tests for tune detail endpoint.

Protects against:
- Bad repo method names (get_tune_by_id vs get_tune)
- Template variable mismatches (total_runs vs total)
- UUID rendering assumptions (asyncpg UUID not subscriptable)

Run with: pytest tests/unit/admin/test_tune_detail_endpoints.py -v
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

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
def sample_tune():
    """Sample tune data with asyncpg-style UUID (not plain string)."""
    # Use actual UUID objects to catch slicing bugs
    return {
        "id": UUID("f51ca27c-e23a-4f6a-98d3-4551172064b0"),
        "workspace_id": UUID("00000000-0000-0000-0000-000000000001"),
        "strategy_entity_id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
        "strategy_name": "Test Strategy",
        "search_type": "grid",
        "n_trials": 10,
        "trials_completed": 5,
        "status": "running",
        "param_space": {"fast": [5, 10, 15], "slow": [20, 30, 40]},
        "objective_metric": "sharpe",
        "objective_type": "sharpe",
        "objective_params": None,
        "gates": {"max_drawdown_pct": 20},
        "best_run_id": None,
        "best_score": None,
        "best_params": None,
        "leaderboard": [],
        "created_at": datetime.now(timezone.utc),
        "started_at": datetime.now(timezone.utc),
        "completed_at": None,
        "error": None,
    }


class TestTuneDetailEndpoint:
    """Regression tests for GET /admin/backtests/tunes/{id}."""

    def test_tune_detail_renders_successfully(
        self, client, mock_db_pool, sample_tune
    ):
        """Tune detail page renders without 500 error.

        Regression test for:
        - get_tune_by_id → get_tune method name fix
        - UUID slicing in template (tune.id[:8])
        - total_runs → total variable name fix
        """
        tune_id = sample_tune["id"]

        mock_repo = MagicMock()
        mock_repo.get_tune = AsyncMock(return_value=sample_tune)
        mock_repo.list_tune_runs = AsyncMock(return_value=([], 0))
        mock_repo.get_tune_status_counts = AsyncMock(
            return_value={
                "queued": 0,
                "running": 2,
                "completed": 3,
                "failed": 0,
                "skipped": 0,
            }
        )

        with patch("app.admin.backtests._db_pool", mock_db_pool), patch(
            "app.admin.backtests._get_tune_repo", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/backtests/tunes/{tune_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Verify page contains expected content
        html = response.text
        assert "Tune Session" in html or "Tuning" in html
        assert "f51ca27c" in html  # First 8 chars of tune ID
        assert "Test Strategy" in html

    def test_tune_detail_404_when_not_found(self, client, mock_db_pool):
        """Returns 404 when tune doesn't exist."""
        tune_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_tune = AsyncMock(return_value=None)

        with patch("app.admin.backtests._db_pool", mock_db_pool), patch(
            "app.admin.backtests._get_tune_repo", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/backtests/tunes/{tune_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404

    def test_tune_detail_requires_admin_token(self, client):
        """Endpoint requires admin authentication."""
        response = client.get(f"/admin/backtests/tunes/{uuid4()}")
        assert response.status_code in (401, 403)
