"""Integration tests for the execution API endpoints.

These tests verify the execution API flows work correctly.
Run with: pytest tests/integration/test_execution.py -v
"""

import os
import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# Test admin token - used in both headers and environment
TEST_ADMIN_TOKEN = "test-admin-token"


@pytest.fixture
def admin_headers():
    """Headers with admin token for protected endpoints."""
    return {"X-Admin-Token": TEST_ADMIN_TOKEN}


@pytest.fixture
def mock_settings():
    """Create mock settings for paper trading config."""
    mock = MagicMock()
    mock.config_profile = "development"
    mock.paper_starting_equity = 10000.0
    mock.paper_max_position_size_pct = 0.20
    return mock


@pytest.fixture
def client(mock_settings):
    """Create test client with mocked dependencies."""
    # Reset the broker singleton to ensure clean state
    from app.services.execution import factory
    factory.reset_paper_broker()

    # Set admin token in environment and patch settings
    with patch.dict(os.environ, {"ADMIN_TOKEN": TEST_ADMIN_TOKEN}), \
         patch("app.routers.execution.get_settings", return_value=mock_settings), \
         patch("app.main._db_pool", MagicMock()), \
         patch("app.main._qdrant_client", None):
        # Also need to patch the execution router's _db_pool
        import app.routers.execution as execution_module
        execution_module._db_pool = MagicMock()

        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client

        # Reset after test
        execution_module._db_pool = None
        factory.reset_paper_broker()


class TestPaperStateEndpoint:
    """Tests for GET /execute/paper/state/{workspace_id}."""

    def test_state_does_not_500(self, client, admin_headers):
        """Test that state endpoint returns 200, not 500, for any UUID."""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/execute/paper/state/{fake_uuid}", headers=admin_headers)

        # Should return 200 with fresh state, not crash
        assert response.status_code == 200
        data = response.json()
        assert "cash" in data
        assert "starting_equity" in data
        assert "positions" in data

    def test_state_returns_default_values(self, client, admin_headers):
        """Test that fresh state has correct defaults."""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/execute/paper/state/{fake_uuid}", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()

        # Check default paper state values
        assert data["starting_equity"] == 10000.0
        assert data["cash"] == 10000.0
        assert data["realized_pnl"] == 0.0
        assert data["positions"] == {}

    def test_state_requires_auth(self, client):
        """Test that state endpoint requires admin token."""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/execute/paper/state/{fake_uuid}")

        # Should return 401/403 without token
        assert response.status_code in (401, 403)

    def test_state_validates_uuid(self, client, admin_headers):
        """Test that state endpoint validates UUID format."""
        response = client.get("/execute/paper/state/not-a-uuid", headers=admin_headers)

        assert response.status_code == 422


class TestPaperPositionsEndpoint:
    """Tests for GET /execute/paper/positions/{workspace_id}."""

    def test_positions_does_not_500(self, client, admin_headers):
        """Test that positions endpoint returns 200, not 500."""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/execute/paper/positions/{fake_uuid}", headers=admin_headers)

        # Should return empty list, not crash
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_positions_requires_auth(self, client):
        """Test that positions endpoint requires admin token."""
        fake_uuid = str(uuid.uuid4())
        response = client.get(f"/execute/paper/positions/{fake_uuid}")

        assert response.status_code in (401, 403)


class TestPaperReconcileEndpoint:
    """Tests for POST /execute/paper/reconcile/{workspace_id}."""

    def test_reconcile_requires_auth(self, client):
        """Test that reconcile endpoint requires admin token."""
        fake_uuid = str(uuid.uuid4())
        response = client.post(f"/execute/paper/reconcile/{fake_uuid}")

        assert response.status_code in (401, 403)


class TestPaperResetEndpoint:
    """Tests for POST /execute/paper/reset/{workspace_id}."""

    def test_reset_works_in_non_production(self, client, admin_headers):
        """Test that reset works when not in production mode."""
        fake_uuid = str(uuid.uuid4())

        # Default config_profile is not "production"
        response = client.post(f"/execute/paper/reset/{fake_uuid}", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"
        assert data["workspace_id"] == fake_uuid

    def test_reset_forbidden_in_production(self, admin_headers):
        """Test that reset returns 403 in production mode."""
        from app.services.execution import factory
        factory.reset_paper_broker()

        # Mock settings for production mode
        prod_settings = MagicMock()
        prod_settings.config_profile = "production"

        with patch.dict(os.environ, {"ADMIN_TOKEN": TEST_ADMIN_TOKEN}), \
             patch("app.routers.execution.get_settings", return_value=prod_settings), \
             patch("app.main._db_pool", MagicMock()), \
             patch("app.main._qdrant_client", None):
            import app.routers.execution as execution_module
            execution_module._db_pool = MagicMock()

            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as test_client:
                fake_uuid = str(uuid.uuid4())
                response = test_client.post(
                    f"/execute/paper/reset/{fake_uuid}",
                    headers=admin_headers
                )

            execution_module._db_pool = None

        assert response.status_code == 403
        assert "production" in response.json()["detail"].lower()

    def test_reset_requires_auth(self, client):
        """Test that reset endpoint requires admin token."""
        fake_uuid = str(uuid.uuid4())
        response = client.post(f"/execute/paper/reset/{fake_uuid}")

        assert response.status_code in (401, 403)


class TestExecuteIntentEndpoint:
    """Tests for POST /execute/intents."""

    @pytest.fixture
    def valid_intent_request(self):
        """Create a valid execution request payload."""
        return {
            "intent": {
                "id": str(uuid.uuid4()),
                "workspace_id": str(uuid.uuid4()),
                "correlation_id": f"test-{uuid.uuid4().hex[:8]}",
                "action": "open_long",
                "strategy_entity_id": str(uuid.uuid4()),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "quantity": 1.0,
                "reason": "Test intent",
            },
            "fill_price": 50000.0,
            "mode": "paper",
        }

    def test_execute_requires_auth(self, client, valid_intent_request):
        """Test that execute endpoint requires admin token."""
        response = client.post("/execute/intents", json=valid_intent_request)

        assert response.status_code in (401, 403)

    def test_execute_validates_mode(self, client, admin_headers, valid_intent_request):
        """Test that only paper mode is supported in PR1."""
        valid_intent_request["mode"] = "live"

        with patch("app.routers.execution._get_events_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_get_repo.return_value = mock_repo

            response = client.post("/execute/intents", json=valid_intent_request, headers=admin_headers)

        assert response.status_code == 400
        assert "paper mode" in response.json()["detail"]["error"].lower()

    def test_execute_validates_action(self, client, admin_headers, valid_intent_request):
        """Test that unsupported actions return 400."""
        valid_intent_request["intent"]["action"] = "open_short"

        with patch("app.routers.execution._get_events_repo") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_order_filled_by_intent = AsyncMock(return_value=None)
            mock_get_repo.return_value = mock_repo

            response = client.post("/execute/intents", json=valid_intent_request, headers=admin_headers)

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error_code"] == "UNSUPPORTED_ACTION"

    def test_execute_validates_fill_price_schema(self, client, admin_headers, valid_intent_request):
        """Test that fill_price must be provided (FastAPI schema validation)."""
        del valid_intent_request["fill_price"]

        response = client.post("/execute/intents", json=valid_intent_request, headers=admin_headers)

        # Should fail schema validation (422)
        assert response.status_code == 422


class TestExecutionModeValidation:
    """Tests for execution mode handling."""

    def test_live_mode_rejected(self, client, admin_headers):
        """Test that live mode is rejected in PR1."""
        request = {
            "intent": {
                "id": str(uuid.uuid4()),
                "workspace_id": str(uuid.uuid4()),
                "correlation_id": f"test-{uuid.uuid4().hex[:8]}",
                "action": "open_long",
                "strategy_entity_id": str(uuid.uuid4()),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "quantity": 1.0,
                "reason": "Test",
            },
            "fill_price": 50000.0,
            "mode": "live",
        }

        response = client.post("/execute/intents", json=request, headers=admin_headers)

        assert response.status_code == 400
        assert response.json()["detail"]["error_code"] == "UNSUPPORTED_MODE"
