"""Shared fixtures for integration tests.

Provides proper mocking of DB and Qdrant connections without
coupling to internal module structure.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# Test admin token - shared across integration tests
TEST_ADMIN_TOKEN = "test-admin-token"


@pytest.fixture
def admin_headers():
    """Headers with admin token for protected endpoints."""
    return {"X-Admin-Token": TEST_ADMIN_TOKEN}


@pytest.fixture
def mock_db_pool():
    """Mock database pool."""
    return MagicMock()


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client - None simulates disconnected state."""
    return None


@pytest.fixture
def mock_settings():
    """Create mock settings for tests."""
    mock = MagicMock()
    mock.config_profile = "development"
    mock.paper_starting_equity = 10000.0
    mock.paper_max_position_size_pct = 0.20
    return mock


@pytest.fixture
def test_app(mock_db_pool, mock_qdrant_client):
    """Create FastAPI app with mocked dependencies.

    Patches the stable seam in app.core.lifespan, not internal module globals.
    """
    with patch("app.core.lifespan._db_pool", mock_db_pool), patch(
        "app.core.lifespan._qdrant_client", mock_qdrant_client
    ):
        from app.main import app

        yield app


@pytest.fixture
def client(test_app):
    """Create test client with mocked dependencies."""
    with TestClient(test_app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def client_with_admin(mock_db_pool, mock_qdrant_client, mock_settings):
    """Create test client with admin token and mocked dependencies."""
    # Reset the broker singleton to ensure clean state
    from app.services.execution import factory

    factory.reset_paper_broker()

    with patch.dict(os.environ, {"ADMIN_TOKEN": TEST_ADMIN_TOKEN}), patch(
        "app.core.lifespan._db_pool", mock_db_pool
    ), patch("app.core.lifespan._qdrant_client", mock_qdrant_client), patch(
        "app.routers.execution.get_settings", return_value=mock_settings
    ):
        from app.main import app

        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client


# Contract test - ensures the seam exists
def test_lifespan_exports_exist():
    """Verify the lifespan module exports the expected functions.

    This test fails fast if someone refactors the DB seam location,
    making it clear what needs updating instead of 36 cryptic patch failures.
    """
    from app.core.lifespan import get_db_pool, get_qdrant_client

    assert callable(get_db_pool)
    assert callable(get_qdrant_client)
