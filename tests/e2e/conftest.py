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
        "run_plans": "/admin/testing/run-plans",
        "trade_events": "/admin/trade/events",
        "coverage_cockpit": "/admin/coverage/cockpit",
    }


# =============================================================================
# E2E Test Helpers
# =============================================================================

# Fake UUID for testing pages that require an ID parameter
FAKE_UUID = "00000000-0000-0000-0000-000000000001"


def assert_no_500(page: Page) -> None:
    """
    Assert that the page does not show a 500 Internal Server Error.

    Use this as a baseline assertion for admin pages - they should
    gracefully handle missing data (404) but never crash (500).
    """
    from playwright.sync_api import expect

    expect(page.locator("body")).not_to_contain_text("Internal Server Error")


def visit_admin_page(page: Page, base_url: str, path: str) -> None:
    """
    Navigate to an admin page and verify it doesn't crash.

    Args:
        page: Playwright page with admin auth
        base_url: Base URL of the server
        path: Path to the admin page (e.g., "/admin/kb/entities")
    """
    page.goto(f"{base_url}{path}")
    assert_no_500(page)


def visit_detail_page(
    page: Page, base_url: str, path_template: str, uuid: str = FAKE_UUID
) -> None:
    """
    Navigate to a detail page with a UUID parameter.

    Args:
        page: Playwright page with admin auth
        base_url: Base URL of the server
        path_template: Path with {id} placeholder (e.g., "/admin/kb/entities/{id}")
        uuid: UUID to use (defaults to FAKE_UUID)
    """
    path = path_template.replace("{id}", uuid)
    page.goto(f"{base_url}{path}")
    assert_no_500(page)


@pytest.fixture
def e2e_helpers():
    """
    Provide E2E test helper functions.

    Usage:
        def test_my_page(admin_page, base_url, e2e_helpers):
            e2e_helpers.visit_detail_page(admin_page, base_url, "/admin/foo/{id}")
    """

    class E2EHelpers:
        FAKE_UUID = FAKE_UUID

        @staticmethod
        def assert_no_500(page: Page) -> None:
            assert_no_500(page)

        @staticmethod
        def visit_admin_page(page: Page, base_url: str, path: str) -> None:
            visit_admin_page(page, base_url, path)

        @staticmethod
        def visit_detail_page(
            page: Page, base_url: str, path_template: str, uuid: str = FAKE_UUID
        ) -> None:
            visit_detail_page(page, base_url, path_template, uuid)

    return E2EHelpers()


# =============================================================================
# API E2E Test Fixtures (for testing JSON API endpoints)
# =============================================================================


@pytest.fixture(scope="function")
def api_request(playwright, base_url: str, admin_token: str):
    """
    Create an API request context with admin authentication.

    Usage:
        def test_api_endpoint(api_request):
            response = api_request.get("/execute/paper/state/...")
            assert response.status == 200
    """
    context = playwright.request.new_context(
        base_url=base_url,
        extra_http_headers={
            "X-Admin-Token": admin_token,
            "Content-Type": "application/json",
        },
    )
    yield context
    context.dispose()


@pytest.fixture(scope="function")
def api_request_no_auth(playwright, base_url: str):
    """
    Create an API request context without authentication.

    Use for testing auth enforcement on API endpoints.
    """
    context = playwright.request.new_context(
        base_url=base_url,
        extra_http_headers={
            "Content-Type": "application/json",
        },
    )
    yield context
    context.dispose()
