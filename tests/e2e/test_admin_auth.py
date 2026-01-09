"""
E2E tests for admin authentication.

Tests that admin routes require proper authentication.
"""

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


class TestAdminAuthentication:
    """Tests for admin authentication enforcement."""

    def test_unauthenticated_request_rejected(
        self, unauthenticated_page: Page, base_url: str
    ):
        """Admin routes reject requests without token (401 if server has token configured)."""
        response = unauthenticated_page.goto(f"{base_url}/admin/backtests/tunes")

        # Without token: 401 (server has ADMIN_TOKEN) or 403 (server has no ADMIN_TOKEN)
        assert response.status in (401, 403)

    def test_authenticated_request_succeeds(
        self, admin_page: Page, base_url: str
    ):
        """Admin routes work with valid token."""
        response = admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Should get 200 OK (server must have matching ADMIN_TOKEN)
        assert response.status == 200, (
            f"Expected 200, got {response.status}. "
            "Ensure server is started with ADMIN_TOKEN=e2e-test-token"
        )

    def test_invalid_token_returns_403(
        self, browser, base_url: str
    ):
        """Admin routes return 403 with invalid token."""
        context = browser.new_context(
            extra_http_headers={
                "X-Admin-Token": "invalid-token-12345",
            }
        )
        page = context.new_page()

        response = page.goto(f"{base_url}/admin/backtests/tunes")

        # Wrong token gets 403 Forbidden
        assert response.status == 403

        page.close()
        context.close()


class TestAdminNavigation:
    """Tests for admin navigation structure."""

    def test_home_redirects_to_entities(
        self, admin_page: Page, base_url: str
    ):
        """Admin home redirects to entities list."""
        admin_page.goto(f"{base_url}/admin", wait_until="networkidle")

        # Should redirect to entities page
        expect(admin_page).to_have_url(f"{base_url}/admin/kb/entities")

    def test_navbar_links_present(
        self, admin_page: Page, base_url: str
    ):
        """Navigation bar contains all main links."""
        admin_page.goto(f"{base_url}/admin/kb/entities")

        # Check navbar links exist (nav element with anchor tags)
        expect(admin_page.locator("nav a[href='/admin/kb/entities']")).to_be_visible()
        expect(admin_page.locator("nav a[href='/admin/kb/claims']")).to_be_visible()
        expect(admin_page.locator("nav a[href='/admin/backtests/tunes']")).to_be_visible()

    def test_navbar_navigation_works(
        self, admin_page: Page, base_url: str
    ):
        """Clicking navbar links navigates correctly."""
        admin_page.goto(f"{base_url}/admin/kb/entities")

        # Click Tunes link
        admin_page.locator("nav a[href='/admin/backtests/tunes']").click()
        expect(admin_page).to_have_url(f"{base_url}/admin/backtests/tunes")

        # Click Claims link
        admin_page.locator("nav a[href='/admin/kb/claims']").click()
        expect(admin_page).to_have_url(f"{base_url}/admin/kb/claims")

        # Click Entities link
        admin_page.locator("nav a[href='/admin/kb/entities']").click()
        expect(admin_page).to_have_url(f"{base_url}/admin/kb/entities")
