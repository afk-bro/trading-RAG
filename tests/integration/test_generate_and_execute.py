"""Integration tests for /testing/run-plans/generate-and-execute endpoint.

This test file validates the end-to-end flow of the Test Generator and
Run Orchestrator, ensuring that:
1. CSV data is correctly parsed and used
2. Variants are generated and executed
3. Events are journaled (RUN_STARTED, RUN_COMPLETED)
4. Best variant selection is deterministic
"""

import io
import json
import csv
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.routers.testing import set_db_pool


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_csv() -> bytes:
    """Create a realistic 50-bar OHLCV CSV for testing.

    This simulates daily BTC data with some price movement.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ts", "open", "high", "low", "close", "volume"])

    # Start date and base price
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    price = 20000.0

    for i in range(50):
        ts = (start_date + timedelta(days=i)).isoformat()
        # Simulate some price movement with trend and volatility
        change_pct = ((i % 7) - 3) * 0.02 + (i / 50) * 0.01  # ~-6% to +8% with upward drift
        price = price * (1 + change_pct)

        open_price = price
        high_price = price * 1.02
        low_price = price * 0.98
        close_price = price * (1 + ((i % 3) - 1) * 0.005)  # Small close variation
        volume = 1000000 + (i % 10) * 100000

        writer.writerow([ts, f"{open_price:.2f}", f"{high_price:.2f}",
                        f"{low_price:.2f}", f"{close_price:.2f}", str(volume)])

    return output.getvalue().encode("utf-8")


@pytest.fixture
def minimal_execution_spec() -> dict:
    """Create a minimal valid ExecutionSpec for testing.

    Uses fixed UUIDs and timestamps for deterministic testing - variant IDs
    should be stable across runs when base_spec is identical.
    """
    # Fixed values for determinism in hash tests
    FIXED_INSTANCE_ID = "12345678-1234-5678-1234-567812345678"
    FIXED_WORKSPACE_ID = "87654321-4321-8765-4321-876543218765"
    FIXED_CREATED_AT = "2024-01-01T00:00:00"  # Fixed timestamp for determinism

    return {
        "strategy_id": "breakout_52w_high",
        "instance_id": FIXED_INSTANCE_ID,
        "name": "Test Breakout Strategy",
        "workspace_id": FIXED_WORKSPACE_ID,
        "symbols": ["BTC"],
        "timeframe": "daily",
        "entry": {
            "type": "breakout_52w_high",
            "lookback_days": 20,  # Using smaller lookback for 50-bar test data
        },
        "exit": {
            "type": "eod",
        },
        "risk": {
            "dollars_per_trade": 1000.0,
            "max_positions": 3,
        },
        "created_at": FIXED_CREATED_AT,  # Fixed for determinism
    }


@pytest.fixture
def minimal_constraints() -> dict:
    """Create minimal constraints for testing."""
    return {
        "lookback_days_values": [15, 20],  # Small values for test data
        "dollars_per_trade_values": [500, 1000],
        "max_positions_values": [2, 3],
        "include_ablations": False,  # Simpler test
        "max_variants": 10,
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_db_pool():
    """Create a mock database pool."""
    return AsyncMock()


@pytest.fixture
def mock_events_insert(mock_db_pool):
    """Mock the events repository insert method."""
    async def mock_insert(event):
        """Capture the event and return a fake ID."""
        # Store the event type for assertions
        if not hasattr(mock_insert, "events"):
            mock_insert.events = []
        mock_insert.events.append(event)
        return uuid4()

    return mock_insert


# =============================================================================
# Integration Tests
# =============================================================================


class TestGenerateAndExecuteIntegration:
    """Integration tests for the generate-and-execute endpoint."""

    @pytest.mark.asyncio
    async def test_full_pipeline_returns_results(
        self,
        sample_ohlcv_csv,
        minimal_execution_spec,
        minimal_constraints,
        mock_db_pool,
        mock_events_insert,
    ):
        """Full pipeline: upload CSV, generate variants, execute, get results."""
        # Set up the mock database pool
        set_db_pool(mock_db_pool)

        # Patch the TradeEventsRepository to use our mock
        with patch(
            "app.routers.testing.TradeEventsRepository"
        ) as MockEventsRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.insert = mock_events_insert
            MockEventsRepo.return_value = mock_repo_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Create multipart form data
                files = {"file": ("test_btc.csv", sample_ohlcv_csv, "text/csv")}
                data = {
                    "workspace_id": minimal_execution_spec["workspace_id"],
                    "base_spec_json": json.dumps(minimal_execution_spec),
                    "constraints_json": json.dumps(minimal_constraints),
                    "objective": "sharpe_dd_penalty",
                }

                response = await client.post(
                    "/testing/run-plans/generate-and-execute",
                    files=files,
                    data=data,
                )

        # Should succeed
        assert response.status_code == 200, f"Response: {response.json()}"
        result = response.json()

        # Should have results
        assert "run_plan_id" in result
        assert "results" in result
        assert len(result["results"]) > 0

        # Should have a best variant (if any succeeded)
        # Note: With placeholder implementation, all may have same score
        if any(r["status"] == "success" for r in result["results"]):
            assert result.get("best_variant_id") is not None

    @pytest.mark.asyncio
    async def test_events_journaled(
        self,
        sample_ohlcv_csv,
        minimal_execution_spec,
        minimal_constraints,
        mock_db_pool,
    ):
        """Verify that RUN_STARTED and RUN_COMPLETED events are journaled."""
        set_db_pool(mock_db_pool)

        # Track events
        recorded_events = []

        async def capture_event(event):
            recorded_events.append(event)
            return uuid4()

        with patch(
            "app.routers.testing.TradeEventsRepository"
        ) as MockEventsRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.insert = capture_event
            MockEventsRepo.return_value = mock_repo_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                files = {"file": ("test_btc.csv", sample_ohlcv_csv, "text/csv")}
                data = {
                    "workspace_id": minimal_execution_spec["workspace_id"],
                    "base_spec_json": json.dumps(minimal_execution_spec),
                    "constraints_json": json.dumps(minimal_constraints),
                    "objective": "sharpe",
                }

                response = await client.post(
                    "/testing/run-plans/generate-and-execute",
                    files=files,
                    data=data,
                )

        assert response.status_code == 200

        # Check events were recorded
        assert len(recorded_events) >= 2, "Should have at least RUN_STARTED and RUN_COMPLETED"

        # Find RUN_STARTED and RUN_COMPLETED events
        run_started_count = sum(
            1 for e in recorded_events
            if e.payload.get("run_event_type") == "RUN_STARTED"
        )
        run_completed_count = sum(
            1 for e in recorded_events
            if e.payload.get("run_event_type") == "RUN_COMPLETED"
        )

        assert run_started_count == 1, "Should have exactly one RUN_STARTED event"
        assert run_completed_count == 1, "Should have exactly one RUN_COMPLETED event"

    @pytest.mark.asyncio
    async def test_best_variant_deterministic(
        self,
        sample_ohlcv_csv,
        minimal_execution_spec,
        minimal_constraints,
        mock_db_pool,
    ):
        """Verify that best_variant_id is deterministic across identical runs."""
        set_db_pool(mock_db_pool)

        results_from_runs = []

        # Run twice with identical inputs
        for _ in range(2):
            with patch(
                "app.routers.testing.TradeEventsRepository"
            ) as MockEventsRepo:
                mock_repo_instance = AsyncMock()
                mock_repo_instance.insert = AsyncMock(return_value=uuid4())
                MockEventsRepo.return_value = mock_repo_instance

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    files = {"file": ("test_btc.csv", sample_ohlcv_csv, "text/csv")}
                    data = {
                        "workspace_id": minimal_execution_spec["workspace_id"],
                        "base_spec_json": json.dumps(minimal_execution_spec),
                        "constraints_json": json.dumps(minimal_constraints),
                        "objective": "sharpe_dd_penalty",
                    }

                    response = await client.post(
                        "/testing/run-plans/generate-and-execute",
                        files=files,
                        data=data,
                    )

            assert response.status_code == 200
            results_from_runs.append(response.json())

        # Best variant should be the same
        # Note: With placeholder implementation, this tests the selection logic
        assert (
            results_from_runs[0]["best_variant_id"] ==
            results_from_runs[1]["best_variant_id"]
        ), "Best variant ID should be deterministic"

        assert (
            results_from_runs[0]["best_score"] ==
            results_from_runs[1]["best_score"]
        ), "Best score should be deterministic"

    @pytest.mark.asyncio
    async def test_variant_ids_stable_across_runs(
        self,
        sample_ohlcv_csv,
        minimal_execution_spec,
        minimal_constraints,
        mock_db_pool,
    ):
        """Verify that variant IDs are stable/deterministic."""
        set_db_pool(mock_db_pool)

        variant_ids_from_runs = []

        for _ in range(2):
            with patch(
                "app.routers.testing.TradeEventsRepository"
            ) as MockEventsRepo:
                mock_repo_instance = AsyncMock()
                mock_repo_instance.insert = AsyncMock(return_value=uuid4())
                MockEventsRepo.return_value = mock_repo_instance

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    files = {"file": ("test_btc.csv", sample_ohlcv_csv, "text/csv")}
                    data = {
                        "workspace_id": minimal_execution_spec["workspace_id"],
                        "base_spec_json": json.dumps(minimal_execution_spec),
                        "constraints_json": json.dumps(minimal_constraints),
                        "objective": "sharpe",
                    }

                    response = await client.post(
                        "/testing/run-plans/generate-and-execute",
                        files=files,
                        data=data,
                    )

            assert response.status_code == 200
            result = response.json()
            variant_ids = [r["variant_id"] for r in result["results"]]
            variant_ids_from_runs.append(variant_ids)

        # Variant IDs should be identical across runs
        assert variant_ids_from_runs[0] == variant_ids_from_runs[1], (
            "Variant IDs should be stable/deterministic across runs"
        )

    @pytest.mark.asyncio
    async def test_invalid_csv_returns_422(
        self,
        minimal_execution_spec,
        minimal_constraints,
        mock_db_pool,
    ):
        """Invalid CSV should return 422 with clear error."""
        set_db_pool(mock_db_pool)

        # CSV with missing columns
        bad_csv = b"ts,open,high,low\n2023-01-01,100,105,99\n"

        with patch(
            "app.routers.testing.TradeEventsRepository"
        ) as MockEventsRepo:
            mock_repo_instance = AsyncMock()
            MockEventsRepo.return_value = mock_repo_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                files = {"file": ("bad.csv", bad_csv, "text/csv")}
                data = {
                    "workspace_id": minimal_execution_spec["workspace_id"],
                    "base_spec_json": json.dumps(minimal_execution_spec),
                    "constraints_json": json.dumps(minimal_constraints),
                    "objective": "sharpe",
                }

                response = await client.post(
                    "/testing/run-plans/generate-and-execute",
                    files=files,
                    data=data,
                )

        assert response.status_code == 422
        result = response.json()
        assert "error_code" in result["detail"]
        assert result["detail"]["error_code"] == "INVALID_CSV"

    @pytest.mark.asyncio
    async def test_non_monotonic_csv_returns_422(
        self,
        minimal_execution_spec,
        minimal_constraints,
        mock_db_pool,
    ):
        """Non-monotonic timestamps in CSV should return 422."""
        set_db_pool(mock_db_pool)

        # CSV with non-monotonic timestamps
        bad_csv = b"ts,open,high,low,close,volume\n"
        bad_csv += b"2023-01-02T00:00:00,104,108,103,107,1200\n"
        bad_csv += b"2023-01-01T00:00:00,100,105,99,104,1000\n"  # Earlier timestamp!

        with patch(
            "app.routers.testing.TradeEventsRepository"
        ) as MockEventsRepo:
            mock_repo_instance = AsyncMock()
            MockEventsRepo.return_value = mock_repo_instance

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                files = {"file": ("bad.csv", bad_csv, "text/csv")}
                data = {
                    "workspace_id": minimal_execution_spec["workspace_id"],
                    "base_spec_json": json.dumps(minimal_execution_spec),
                    "constraints_json": json.dumps(minimal_constraints),
                    "objective": "sharpe",
                }

                response = await client.post(
                    "/testing/run-plans/generate-and-execute",
                    files=files,
                    data=data,
                )

        assert response.status_code == 422
        result = response.json()
        assert "Non-monotonic" in result["detail"]["error"]
