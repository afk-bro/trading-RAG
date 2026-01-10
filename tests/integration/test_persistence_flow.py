"""Integration test for results persistence flow.

Verifies that run plans and variant results are correctly persisted
when executing through the testing router.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Set admin token for tests
os.environ["ADMIN_TOKEN"] = "test-token"


@pytest.fixture
def mock_db_pool():
    """Create mock database pool with connection context manager."""
    pool = MagicMock()
    conn = MagicMock()

    # Mock async context manager for pool.acquire()
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=conn)
    async_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = async_cm

    # Mock fetchval and execute
    conn.fetchval = AsyncMock(return_value=uuid4())
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)

    return pool


@pytest.fixture
def client_with_pool(mock_db_pool):
    """Create test client with mocked database pool."""
    from app.main import app
    from app.routers.testing import set_db_pool

    # Set the database pool
    set_db_pool(mock_db_pool)

    # Yield client
    client = TestClient(app)
    yield client, mock_db_pool

    # Cleanup
    set_db_pool(None)


@pytest.fixture
def valid_csv_content():
    """Valid OHLCV CSV content."""
    return """ts,open,high,low,close,volume
2024-01-01T00:00:00Z,100,105,99,104,1000
2024-01-02T00:00:00Z,104,108,103,107,1200
2024-01-03T00:00:00Z,107,110,106,109,1100
2024-01-04T00:00:00Z,109,112,108,111,1300
2024-01-05T00:00:00Z,111,115,110,114,1400"""


@pytest.fixture
def base_spec():
    """Valid base execution spec."""
    return {
        "workspace_id": str(uuid4()),
        "strategy_id": "breakout_52w_high",
        "name": "test_execution",
        "symbols": ["BTC-USD"],
        "timeframe": "daily",
        "risk": {
            "dollars_per_trade": 1000,
            "max_positions": 5,
        },
        "entry": {
            "type": "breakout_52w_high",
            "lookback_days": 252,
        },
        "exit": {
            "type": "eod",
        },
    }


@pytest.fixture
def constraints():
    """Simple constraints for testing."""
    return {
        "lookback_days_values": [200, 252],
        "dollars_per_trade_values": [],
        "max_positions_values": [],
        "include_ablations": False,
        "max_variants": 10,
        "objective": "sharpe",
    }


class TestPersistenceFlow:
    """Integration tests for the persistence flow."""

    def test_generate_and_execute_creates_run_plan(
        self, client_with_pool, valid_csv_content, base_spec, constraints
    ):
        """Execute creates a run_plan row in the database."""
        client, mock_pool = client_with_pool

        response = client.post(
            "/testing/run-plans/generate-and-execute",
            files={"file": ("test.csv", valid_csv_content.encode(), "text/csv")},
            data={
                "workspace_id": base_spec["workspace_id"],
                "base_spec_json": json.dumps(base_spec),
                "constraints_json": json.dumps(constraints),
                "objective": "sharpe",
            },
        )

        # Should succeed
        assert response.status_code == 200

        # Verify run plan was created
        # The create_run_plan INSERT should have been called
        conn_mock = mock_pool.acquire.return_value.__aenter__.return_value
        calls = conn_mock.fetchval.call_args_list

        # Check for INSERT INTO run_plans call
        run_plan_inserts = [
            c for c in calls if "run_plans" in str(c).lower() and "insert" in str(c).lower()
        ]
        assert len(run_plan_inserts) > 0, "Expected run_plan INSERT"

    def test_generate_and_execute_creates_variant_runs(
        self, client_with_pool, valid_csv_content, base_spec, constraints
    ):
        """Execute creates backtest_run rows for each variant."""
        client, mock_pool = client_with_pool

        response = client.post(
            "/testing/run-plans/generate-and-execute",
            files={"file": ("test.csv", valid_csv_content.encode(), "text/csv")},
            data={
                "workspace_id": base_spec["workspace_id"],
                "base_spec_json": json.dumps(base_spec),
                "constraints_json": json.dumps(constraints),
                "objective": "sharpe",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have at least 1 variant (baseline is always generated)
        n_variants = data["n_variants"]
        assert n_variants >= 1

        # Verify backtest_runs were created
        conn_mock = mock_pool.acquire.return_value.__aenter__.return_value
        calls = conn_mock.fetchval.call_args_list

        # Each variant creates a backtest_run
        backtest_run_inserts = [
            c for c in calls if "backtest_runs" in str(c).lower() and "insert" in str(c).lower()
        ]
        # Should have at least n_variants inserts (one per variant)
        assert len(backtest_run_inserts) >= n_variants

    def test_generate_and_execute_completes_run_plan(
        self, client_with_pool, valid_csv_content, base_spec, constraints
    ):
        """Execute updates run_plan with completion status."""
        client, mock_pool = client_with_pool

        response = client.post(
            "/testing/run-plans/generate-and-execute",
            files={"file": ("test.csv", valid_csv_content.encode(), "text/csv")},
            data={
                "workspace_id": base_spec["workspace_id"],
                "base_spec_json": json.dumps(base_spec),
                "constraints_json": json.dumps(constraints),
                "objective": "sharpe",
            },
        )

        assert response.status_code == 200

        # Verify run_plan was completed
        conn_mock = mock_pool.acquire.return_value.__aenter__.return_value
        calls = conn_mock.execute.call_args_list

        # Check for UPDATE run_plans ... status = 'completed'
        completion_updates = [
            c for c in calls
            if "run_plans" in str(c).lower()
            and "update" in str(c).lower()
            and "completed" in str(c).lower()
        ]
        assert len(completion_updates) > 0, "Expected run_plan completion UPDATE"

    def test_response_includes_best_variant(
        self, client_with_pool, valid_csv_content, base_spec, constraints
    ):
        """Response includes best_variant_id and best_score."""
        client, mock_pool = client_with_pool

        response = client.post(
            "/testing/run-plans/generate-and-execute",
            files={"file": ("test.csv", valid_csv_content.encode(), "text/csv")},
            data={
                "workspace_id": base_spec["workspace_id"],
                "base_spec_json": json.dumps(base_spec),
                "constraints_json": json.dumps(constraints),
                "objective": "sharpe",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Best variant should be identified
        assert "best_variant_id" in data
        assert "best_score" in data

    def test_results_have_correct_structure(
        self, client_with_pool, valid_csv_content, base_spec, constraints
    ):
        """Results array has expected fields."""
        client, mock_pool = client_with_pool

        response = client.post(
            "/testing/run-plans/generate-and-execute",
            files={"file": ("test.csv", valid_csv_content.encode(), "text/csv")},
            data={
                "workspace_id": base_spec["workspace_id"],
                "base_spec_json": json.dumps(base_spec),
                "constraints_json": json.dumps(constraints),
                "objective": "sharpe",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check results structure
        results = data["results"]
        assert len(results) > 0

        for result in results:
            assert "variant_id" in result
            assert "status" in result
            # Status should be success, failed, or skipped
            assert result["status"] in ["success", "failed", "skipped"]


class TestPersistenceWithoutPool:
    """Tests behavior when pool is not available."""

    def test_execute_without_pool_returns_503(self, valid_csv_content, base_spec, constraints):
        """Execute returns 503 when database pool is not set."""
        from app.main import app
        from app.routers.testing import set_db_pool

        # Ensure pool is None
        set_db_pool(None)

        client = TestClient(app)

        response = client.post(
            "/testing/run-plans/generate-and-execute",
            files={"file": ("test.csv", valid_csv_content.encode(), "text/csv")},
            data={
                "workspace_id": base_spec["workspace_id"],
                "base_spec_json": json.dumps(base_spec),
                "constraints_json": json.dumps(constraints),
                "objective": "sharpe",
            },
        )

        assert response.status_code == 503
