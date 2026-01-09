"""
E2E tests for admin authentication.

Tests that admin routes require proper authentication.
"""

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


class TestAdminAuthentication:
    """Tests for admin authentication enforcement."""

    def test_unauthenticated_request_returns_401(
        self, unauthenticated_page: Page, base_url: str
    ):
        """Admin routes return 401 without token."""
        response = unauthenticated_page.goto(f"{base_url}/admin/backtests/tunes")

        # Should get 401 Unauthorized
        assert response.status == 401

    def test_authenticated_request_succeeds(
        self, admin_page: Page, base_url: str
    ):
        """Admin routes work with valid token."""
        response = admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Should get 200 OK
        assert response.status == 200

    def test_invalid_token_returns_401(
        self, browser, base_url: str
    ):
        """Admin routes return 401 with invalid token."""
        context = browser.new_context(
            extra_http_headers={
                "X-Admin-Token": "invalid-token-12345",
            }
        )
        page = context.new_page()

        response = page.goto(f"{base_url}/admin/backtests/tunes")

        # Should get 401 Unauthorized
        assert response.status == 401

        page.close()
        context.close()


class TestAdminNavigation:
    """Tests for admin navigation structure."""

    def test_home_redirects_to_entities(
        self, admin_page: Page, base_url: str
    ):
        """Admin home redirects to entities list."""
        admin_page.goto(f"{base_url}/admin")

        # Should redirect to entities page
        expect(admin_page).to_have_url(f"{base_url}/admin/kb/entities")

    def test_navbar_links_present(
        self, admin_page: Page, base_url: str
    ):
        """Navigation bar contains all main links."""
        admin_page.goto(f"{base_url}/admin/kb/entities")

        # Check navbar links exist
        expect(admin_page.locator("nav a:has-text('Entities')")).to_be_visible()
        expect(admin_page.locator("nav a:has-text('Claims')")).to_be_visible()
        expect(admin_page.locator("nav a:has-text('Tunes')")).to_be_visible()

    def test_navbar_navigation_works(
        self, admin_page: Page, base_url: str
    ):
        """Clicking navbar links navigates correctly."""
        admin_page.goto(f"{base_url}/admin/kb/entities")

        # Click Tunes link
        admin_page.locator("nav a:has-text('Tunes')").click()
        expect(admin_page).to_have_url(f"{base_url}/admin/backtests/tunes")

        # Click Claims link
        admin_page.locator("nav a:has-text('Claims')").click()
        expect(admin_page).to_have_url(f"{base_url}/admin/kb/claims")

        # Click Entities link
        admin_page.locator("nav a:has-text('Entities')").click()
        expect(admin_page).to_have_url(f"{base_url}/admin/kb/entities")
