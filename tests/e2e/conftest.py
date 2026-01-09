"""
Playwright E2E test fixtures for Admin UI.

Run with:
    ADMIN_TOKEN=e2e-test-token pytest tests/e2e --base-url http://localhost:8000

Requires:
    1. Server running with ADMIN_TOKEN set:
       ADMIN_TOKEN=e2e-test-token uvicorn app.main:app --port 8000

    2. Same ADMIN_TOKEN in test environment (or use default)
"""

import os
import pytest
from playwright.sync_api import Page, BrowserContext


# Default test configuration
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_ADMIN_TOKEN = "e2e-test-token"


@pytest.fixture(scope="session")
def base_url() -> str:
    """Get base URL for the running server."""
    return os.environ.get("E2E_BASE_URL", DEFAULT_BASE_URL)


@pytest.fixture(scope="session")
def admin_token() -> str:
    """Get admin token for authentication."""
    return os.environ.get("ADMIN_TOKEN", DEFAULT_ADMIN_TOKEN)


@pytest.fixture(scope="function")
def admin_context(browser, admin_token: str) -> BrowserContext:
    """
    Create a browser context with admin authentication headers.

    All requests made in this context will include the X-Admin-Token header.
    """
    context = browser.new_context(
        extra_http_headers={
            "X-Admin-Token": admin_token,
        }
    )
    yield context
    context.close()


@pytest.fixture(scope="function")
def admin_page(admin_context: BrowserContext) -> Page:
    """
    Create a page with admin authentication.

    Use this fixture for all admin UI tests.
    """
    page = admin_context.new_page()
    yield page
    page.close()


@pytest.fixture(scope="function")
def unauthenticated_page(browser) -> Page:
    """
    Create a page without admin authentication.

    Use this fixture to test authentication enforcement.
    """
    context = browser.new_context()
    page = context.new_page()
    yield page
    page.close()
    context.close()


# Utility fixtures for common test data
@pytest.fixture
def admin_routes() -> dict:
    """Return common admin routes for testing."""
    return {
        "home": "/admin",
        "entities": "/admin/kb/entities",
        "claims": "/admin/kb/claims",
        "tunes": "/admin/backtests/tunes",
        "leaderboard": "/admin/backtests/leaderboard",
    }
