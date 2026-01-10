"""Unit tests for Testing API endpoints."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    return MagicMock()


@pytest.fixture
def client(mock_db_pool):
    """Create test client with mocked dependencies."""
    from app.routers import testing

    # Set up mock db pool
    testing.set_db_pool(mock_db_pool)

    # Create test app with just this router
    app = FastAPI()
    app.include_router(testing.router)

    return TestClient(app)


@pytest.fixture
def valid_base_spec():
    """Create a valid ExecutionSpec dict."""
    return {
        "strategy_id": "breakout_52w_high",
        "name": "Test Breakout Strategy",
        "workspace_id": str(uuid4()),
        "symbols": ["BTC"],
        "timeframe": "daily",
        "entry": {
            "type": "breakout_52w_high",
            "lookback_days": 252,
        },
        "exit": {
            "type": "eod",
        },
        "risk": {
            "dollars_per_trade": 1000.0,
            "max_positions": 5,
        },
    }


@pytest.fixture
def valid_constraints():
    """Create valid GeneratorConstraints dict."""
    return {
        "lookback_days_values": [200, 252],
        "dollars_per_trade_values": [500.0, 1000.0],
        "max_positions_values": [3, 5],
        "include_ablations": True,
        "max_variants": 25,
    }


@pytest.fixture
def minimal_csv():
    """Create minimal valid OHLCV CSV content."""
    return b"""ts,open,high,low,close,volume
2024-01-01T00:00:00Z,100.0,105.0,99.0,104.0,1000.0
2024-01-02T00:00:00Z,104.0,110.0,103.0,108.0,1200.0
2024-01-03T00:00:00Z,108.0,112.0,107.0,111.0,1100.0
"""


# =============================================================================
# Generate Endpoint Tests
# =============================================================================


class TestGenerateRunPlan:
    """Tests for POST /testing/run-plans/generate endpoint."""

    def test_generate_returns_correct_n_variants(
        self, client, valid_base_spec, valid_constraints
    ):
        """Should return correct number of variants."""
        workspace_id = uuid4()

        response = client.post(
            "/testing/run-plans/generate",
            json={
                "workspace_id": str(workspace_id),
                "base_spec": valid_base_spec,
                "dataset_ref": "test_dataset.csv",
                "constraints": valid_constraints,
                "objective": "sharpe_dd_penalty",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "run_plan_id" in data
        assert "n_variants" in data
        assert data["n_variants"] > 0
        assert "variants" in data
        assert len(data["variants"]) == data["n_variants"]
        assert data["objective"] == "sharpe_dd_penalty"

    def test_generate_includes_baseline_variant(
        self, client, valid_base_spec, valid_constraints
    ):
        """Should include baseline variant with 'baseline' tag."""
        response = client.post(
            "/testing/run-plans/generate",
            json={
                "workspace_id": str(uuid4()),
                "base_spec": valid_base_spec,
                "dataset_ref": "test_dataset.csv",
                "constraints": valid_constraints,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # First variant should be baseline
        variants = data["variants"]
        assert len(variants) > 0
        baseline = variants[0]
        assert baseline["label"] == "baseline"
        assert "baseline" in baseline["tags"]

    def test_generate_invalid_base_spec_returns_422(self, client, valid_constraints):
        """Should return 422 for invalid base_spec."""
        response = client.post(
            "/testing/run-plans/generate",
            json={
                "workspace_id": str(uuid4()),
                "base_spec": {"invalid": "spec"},  # Missing required fields
                "dataset_ref": "test_dataset.csv",
                "constraints": valid_constraints,
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "INVALID_BASE_SPEC" in str(data) or "error" in data.get("detail", {})

    def test_generate_invalid_constraints_returns_422(self, client, valid_base_spec):
        """Should return 422 for invalid constraints."""
        response = client.post(
            "/testing/run-plans/generate",
            json={
                "workspace_id": str(uuid4()),
                "base_spec": valid_base_spec,
                "dataset_ref": "test_dataset.csv",
                "constraints": {
                    "lookback_days_values": "not_a_list"
                },  # Invalid: should be list
            },
        )

        assert response.status_code == 422

    def test_generate_empty_sweep_values_returns_baseline_only(
        self, client, valid_base_spec
    ):
        """With empty sweep values, should return only baseline variant."""
        response = client.post(
            "/testing/run-plans/generate",
            json={
                "workspace_id": str(uuid4()),
                "base_spec": valid_base_spec,
                "dataset_ref": "test_dataset.csv",
                "constraints": {
                    "lookback_days_values": [],
                    "dollars_per_trade_values": [],
                    "max_positions_values": [],
                    "include_ablations": False,
                    "max_variants": 25,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Only baseline variant
        assert data["n_variants"] == 1
        assert data["variants"][0]["label"] == "baseline"


# =============================================================================
# Generate and Execute Endpoint Tests
# =============================================================================


class TestGenerateAndExecuteRunPlan:
    """Tests for POST /testing/run-plans/generate-and-execute endpoint."""

    def test_execute_returns_results_shape(
        self, client, valid_base_spec, valid_constraints, minimal_csv
    ):
        """Should return correct results shape."""
        with patch("app.routers.testing.RunOrchestrator") as MockOrchestrator:
            # Set up mock orchestrator
            from app.services.testing import RunResult, VariantMetrics

            mock_result = RunResult(
                run_plan_id=uuid4(),
                variant_id="abc123def456ghij",
                status="success",
                metrics=VariantMetrics(
                    sharpe=1.5,
                    return_pct=10.5,
                    max_drawdown_pct=5.0,
                    trade_count=10,
                    win_rate=0.6,
                    ending_equity=11000.0,
                    gross_profit=1200.0,
                    gross_loss=-200.0,
                ),
                objective_score=1.45,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_ms=100,
                events_recorded=5,
            )

            mock_instance = MockOrchestrator.return_value
            mock_instance.execute = AsyncMock(return_value=[mock_result])

            response = client.post(
                "/testing/run-plans/generate-and-execute",
                data={
                    "workspace_id": str(uuid4()),
                    "base_spec_json": json.dumps(valid_base_spec),
                    "constraints_json": json.dumps(valid_constraints),
                    "objective": "sharpe_dd_penalty",
                },
                files={"file": ("test.csv", minimal_csv, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()

        assert "run_plan_id" in data
        assert "n_variants" in data
        assert "results" in data
        assert isinstance(data["results"], list)

        # Check result structure
        if data["results"]:
            result = data["results"][0]
            assert "variant_id" in result
            assert "status" in result
            assert result["status"] in ["success", "failed"]

    def test_execute_with_invalid_csv_returns_422(
        self, client, valid_base_spec, valid_constraints
    ):
        """Should return 422 for invalid CSV."""
        with patch("app.routers.testing.RunOrchestrator") as MockOrchestrator:
            # Simulate CSV parsing error
            mock_instance = MockOrchestrator.return_value
            mock_instance.execute = AsyncMock(
                side_effect=ValueError("Missing required columns: {'ts'}")
            )

            invalid_csv = b"a,b,c\n1,2,3"

            response = client.post(
                "/testing/run-plans/generate-and-execute",
                data={
                    "workspace_id": str(uuid4()),
                    "base_spec_json": json.dumps(valid_base_spec),
                    "constraints_json": json.dumps(valid_constraints),
                },
                files={"file": ("test.csv", invalid_csv, "text/csv")},
            )

        assert response.status_code == 422
        data = response.json()
        assert "INVALID_CSV" in str(data) or "error" in data.get("detail", {})

    def test_execute_with_invalid_base_spec_json_returns_422(
        self, client, valid_constraints, minimal_csv
    ):
        """Should return 422 for invalid JSON in base_spec_json."""
        response = client.post(
            "/testing/run-plans/generate-and-execute",
            data={
                "workspace_id": str(uuid4()),
                "base_spec_json": "{not valid json",
                "constraints_json": json.dumps(valid_constraints),
            },
            files={"file": ("test.csv", minimal_csv, "text/csv")},
        )

        assert response.status_code == 422
        data = response.json()
        assert "INVALID_JSON" in str(data)

    def test_execute_with_invalid_constraints_json_returns_422(
        self, client, valid_base_spec, minimal_csv
    ):
        """Should return 422 for invalid JSON in constraints_json."""
        response = client.post(
            "/testing/run-plans/generate-and-execute",
            data={
                "workspace_id": str(uuid4()),
                "base_spec_json": json.dumps(valid_base_spec),
                "constraints_json": "not valid json",
            },
            files={"file": ("test.csv", minimal_csv, "text/csv")},
        )

        assert response.status_code == 422
        data = response.json()
        assert "INVALID_JSON" in str(data)

    def test_execute_returns_best_variant(
        self, client, valid_base_spec, valid_constraints, minimal_csv
    ):
        """Should return best_variant_id and best_score."""
        with patch("app.routers.testing.RunOrchestrator") as MockOrchestrator:
            from app.services.testing import RunResult, VariantMetrics

            # Two results with different scores
            mock_results = [
                RunResult(
                    run_plan_id=uuid4(),
                    variant_id="variant_1",
                    status="success",
                    metrics=VariantMetrics(
                        sharpe=1.0,
                        return_pct=5.0,
                        max_drawdown_pct=10.0,
                        trade_count=5,
                        win_rate=0.4,
                        ending_equity=10500.0,
                        gross_profit=600.0,
                        gross_loss=-100.0,
                    ),
                    objective_score=0.95,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=50,
                ),
                RunResult(
                    run_plan_id=uuid4(),
                    variant_id="variant_2",
                    status="success",
                    metrics=VariantMetrics(
                        sharpe=2.0,
                        return_pct=15.0,
                        max_drawdown_pct=8.0,
                        trade_count=8,
                        win_rate=0.7,
                        ending_equity=11500.0,
                        gross_profit=1800.0,
                        gross_loss=-300.0,
                    ),
                    objective_score=1.96,
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=60,
                ),
            ]

            mock_instance = MockOrchestrator.return_value
            mock_instance.execute = AsyncMock(return_value=mock_results)

            response = client.post(
                "/testing/run-plans/generate-and-execute",
                data={
                    "workspace_id": str(uuid4()),
                    "base_spec_json": json.dumps(valid_base_spec),
                    "constraints_json": json.dumps(valid_constraints),
                },
                files={"file": ("test.csv", minimal_csv, "text/csv")},
            )

        assert response.status_code == 200
        data = response.json()

        # Best variant should be variant_2 (higher score)
        assert data["best_variant_id"] == "variant_2"
        assert data["best_score"] == 1.96


# =============================================================================
# Database Unavailable Tests
# =============================================================================


class TestDatabaseUnavailable:
    """Tests for when database is unavailable."""

    def test_execute_without_db_returns_503(
        self, valid_base_spec, valid_constraints, minimal_csv
    ):
        """Should return 503 when database is unavailable."""
        from app.routers import testing
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # Set db_pool to None
        testing.set_db_pool(None)

        app = FastAPI()
        app.include_router(testing.router)
        client = TestClient(app)

        response = client.post(
            "/testing/run-plans/generate-and-execute",
            data={
                "workspace_id": str(uuid4()),
                "base_spec_json": json.dumps(valid_base_spec),
                "constraints_json": json.dumps(valid_constraints),
            },
            files={"file": ("test.csv", minimal_csv, "text/csv")},
        )

        assert response.status_code == 503
        data = response.json()
        assert "Database connection not available" in data["detail"]

    def test_generate_works_without_db(self, valid_base_spec, valid_constraints):
        """Generate endpoint should work without database (no DB needed for generation)."""
        from app.routers import testing
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # Even with db_pool as None, generate should work (doesn't use DB)
        testing.set_db_pool(None)

        app = FastAPI()
        app.include_router(testing.router)
        client = TestClient(app)

        response = client.post(
            "/testing/run-plans/generate",
            json={
                "workspace_id": str(uuid4()),
                "base_spec": valid_base_spec,
                "dataset_ref": "test_dataset.csv",
                "constraints": valid_constraints,
            },
        )

        # Generate doesn't need DB, should succeed
        assert response.status_code == 200
