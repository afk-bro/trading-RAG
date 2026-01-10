"""Tests for forward metrics endpoint."""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routers.forward_metrics import router
from app.repositories.recommendation_records import RecordStatus


@pytest.fixture
def app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestForwardMetricsRequest:
    """Tests for request validation."""

    def test_valid_request_structure(self, client):
        """Valid request structure is accepted."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = AsyncMock(return_value=uuid4())
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {
                        "return_pct": 0.023,
                        "sharpe_proxy": 1.2,
                        "hit_rate": 0.58,
                        "max_drawdown_pct": 0.08,
                        "expectancy": 0.15,
                    },
                },
            )

            assert response.status_code == 200

    def test_rejects_negative_bars_seen(self, client):
        """Rejects negative bars_seen value."""
        response = client.post(
            "/forward/metrics",
            json={
                "workspace_id": str(uuid4()),
                "record_id": str(uuid4()),
                "ts": "2026-01-09T12:00:00Z",
                "bars_seen": -1,
                "trades_seen": 42,
                "realized_metrics": {},
            },
        )

        assert response.status_code == 422

    def test_rejects_negative_trades_seen(self, client):
        """Rejects negative trades_seen value."""
        response = client.post(
            "/forward/metrics",
            json={
                "workspace_id": str(uuid4()),
                "record_id": str(uuid4()),
                "ts": "2026-01-09T12:00:00Z",
                "bars_seen": 500,
                "trades_seen": -5,
                "realized_metrics": {},
            },
        )

        assert response.status_code == 422

    def test_rejects_missing_required_fields(self, client):
        """Rejects request missing required fields."""
        response = client.post(
            "/forward/metrics",
            json={
                "workspace_id": str(uuid4()),
                # Missing record_id, ts, bars_seen, trades_seen, realized_metrics
            },
        )

        assert response.status_code == 422

    def test_rejects_extra_fields(self, client):
        """Rejects request with extra fields (strict validation)."""
        response = client.post(
            "/forward/metrics",
            json={
                "workspace_id": str(uuid4()),
                "record_id": str(uuid4()),
                "ts": "2026-01-09T12:00:00Z",
                "bars_seen": 500,
                "trades_seen": 42,
                "realized_metrics": {},
                "unexpected_field": "value",
            },
        )

        assert response.status_code == 422

    def test_accepts_zero_bars_and_trades(self, client):
        """Accepts zero for bars_seen and trades_seen."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = AsyncMock(return_value=uuid4())
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 0,
                    "trades_seen": 0,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 200


class TestForwardMetricsEndpoint:
    """Tests for POST /forward/metrics."""

    def test_accepts_valid_payload(self, client):
        """Accepts valid metrics payload and returns observation_id."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            obs_id = uuid4()
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = AsyncMock(return_value=obs_id)
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {
                        "return_pct": 0.023,
                        "sharpe_proxy": 1.2,
                    },
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "accepted"
            assert data["observation_id"] == str(obs_id)

    def test_returns_404_when_record_not_found(self, client):
        """Returns 404 when record does not exist."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_instance.get_record_by_id = AsyncMock(return_value=None)
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["code"] == "RECORD_NOT_FOUND"

    def test_returns_404_for_inactive_record(self, client):
        """Returns 404 for inactive record (closed status)."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.CLOSED
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["code"] == "RECORD_NOT_ACTIVE"

    def test_returns_404_for_superseded_record(self, client):
        """Returns 404 for superseded record."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.SUPERSEDED
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 404
            data = response.json()
            assert data["detail"]["code"] == "RECORD_NOT_ACTIVE"

    def test_returns_404_for_inactive_status(self, client):
        """Returns 404 for record with inactive status."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.INACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 404


class TestForwardMetricsDuplicateHandling:
    """Tests for duplicate observation handling (409 Conflict)."""

    def test_returns_409_for_duplicate_observation_asyncpg(self, client):
        """Returns 409 for duplicate (record_id, ts) with asyncpg error."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            from asyncpg.exceptions import UniqueViolationError

            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = AsyncMock(
                side_effect=UniqueViolationError("")
            )
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 409
            data = response.json()
            assert data["detail"]["code"] == "DUPLICATE_OBSERVATION"

    def test_returns_409_for_generic_unique_error(self, client):
        """Returns 409 for generic exception mentioning unique constraint."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = AsyncMock(
                side_effect=Exception("unique constraint violation")
            )
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 409

    def test_returns_409_for_duplicate_key_error(self, client):
        """Returns 409 for exception mentioning duplicate key."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = AsyncMock(
                side_effect=Exception("duplicate key value violates")
            )
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 409


class TestForwardMetricsServiceUnavailable:
    """Tests for service unavailable scenarios."""

    def test_returns_503_when_db_pool_not_set(self, client):
        """Returns 503 when database pool is not available."""
        # Reset the pool to None to simulate unavailable DB
        import app.routers.forward_metrics as fm_module

        original_pool = fm_module._db_pool
        fm_module._db_pool = None

        try:
            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            assert response.status_code == 503
        finally:
            fm_module._db_pool = original_pool


class TestForwardMetricsObservationCreation:
    """Tests for observation creation logic."""

    def test_passes_correct_observation_data(self, client):
        """Observation is created with correct data from request."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            captured_obs = None

            async def capture_observation(obs):
                nonlocal captured_obs
                captured_obs = obs
                return uuid4()

            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = capture_observation
            mock_repo.return_value = mock_instance

            record_id = uuid4()
            ts_str = "2026-01-09T12:00:00Z"
            realized_metrics = {
                "return_pct": 0.023,
                "sharpe_proxy": 1.2,
            }

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(record_id),
                    "ts": ts_str,
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": realized_metrics,
                },
            )

            assert response.status_code == 200
            assert captured_obs is not None
            assert captured_obs.record_id == record_id
            assert captured_obs.bars_seen == 500
            assert captured_obs.trades_seen == 42
            assert captured_obs.realized_metrics_json == realized_metrics


class TestForwardMetricsResponse:
    """Tests for response format."""

    def test_response_contains_status_and_observation_id(self, client):
        """Response contains status and observation_id fields."""
        with patch("app.routers.forward_metrics._get_repository") as mock_repo:
            obs_id = uuid4()
            mock_instance = MagicMock()
            mock_record = MagicMock()
            mock_record.status = RecordStatus.ACTIVE
            mock_instance.get_record_by_id = AsyncMock(return_value=mock_record)
            mock_instance.add_observation = AsyncMock(return_value=obs_id)
            mock_repo.return_value = mock_instance

            response = client.post(
                "/forward/metrics",
                json={
                    "workspace_id": str(uuid4()),
                    "record_id": str(uuid4()),
                    "ts": "2026-01-09T12:00:00Z",
                    "bars_seen": 500,
                    "trades_seen": 42,
                    "realized_metrics": {},
                },
            )

            data = response.json()
            assert "status" in data
            assert "observation_id" in data
            assert data["status"] == "accepted"
            # observation_id should be a valid UUID string
            from uuid import UUID

            UUID(data["observation_id"])  # Should not raise
