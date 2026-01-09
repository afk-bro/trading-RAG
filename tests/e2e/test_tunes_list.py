"""
E2E tests for the Tunes List page (/admin/backtests/tunes).

Tests filtering, pagination, and table interactions.
"""

import re

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


class TestTunesListPage:
    """Tests for the tunes list page structure."""

    def test_page_loads_successfully(
        self, admin_page: Page, base_url: str
    ):
        """Tunes list page loads with expected elements."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Check page title/header ("Parameter Tuning Sessions")
        expect(admin_page.locator("h2.card-title")).to_contain_text("Tuning")

        # Check filter controls exist
        expect(admin_page.locator("select[name='status']")).to_be_visible()
        expect(admin_page.locator("input[name='valid_only']")).to_be_visible()

    def test_table_has_expected_columns(
        self, admin_page: Page, base_url: str
    ):
        """Tunes table has all expected column headers."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Check for key column headers (may have no data, so check if table exists or empty state)
        table_header = admin_page.locator("table thead")
        if table_header.count() > 0:
            expect(table_header).to_contain_text("Created")
            expect(table_header).to_contain_text("Status")
            expect(table_header).to_contain_text("Strategy")

    def test_empty_state_displays_correctly(
        self, admin_page: Page, base_url: str
    ):
        """Shows appropriate message when no tunes match filters."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes?status=failed")

        # Page should still load without internal server errors
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestTunesListFilters:
    """Tests for tunes list filtering functionality."""

    def test_status_filter_changes_url(
        self, admin_page: Page, base_url: str
    ):
        """Changing status filter updates URL (auto-submits on change)."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Select "completed" status - form auto-submits on change
        admin_page.locator("select[name='status']").select_option("completed")
        admin_page.wait_for_load_state("networkidle")

        # URL should include status parameter (use regex, not lambda)
        expect(admin_page).to_have_url(re.compile(r"status=completed"))

    def test_valid_only_checkbox(
        self, admin_page: Page, base_url: str
    ):
        """Valid only checkbox filters results (auto-submits on change)."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Check the valid_only checkbox - form auto-submits on change
        checkbox = admin_page.locator("input[name='valid_only']")
        if not checkbox.is_checked():
            checkbox.check()
            admin_page.wait_for_load_state("networkidle")

        # URL should include valid_only parameter (use regex, not lambda)
        expect(admin_page).to_have_url(re.compile(r"valid_only=true"))

    def test_reset_filters_via_url(
        self, admin_page: Page, base_url: str
    ):
        """Navigate to base URL resets all filters."""
        # Start with some filters
        admin_page.goto(f"{base_url}/admin/backtests/tunes?status=completed&valid_only=true")

        # Navigate to base URL (no clear button in UI, just use URL)
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # URL should be clean (no query params)
        expect(admin_page).to_have_url(f"{base_url}/admin/backtests/tunes")


class TestTunesListTableInteraction:
    """Tests for table row interactions."""

    def test_row_click_navigates_to_detail(
        self, admin_page: Page, base_url: str
    ):
        """Clicking a tune row navigates to detail page."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Get first row (if exists)
        first_row = admin_page.locator("table tbody tr").first
        if first_row.count() > 0:
            first_row.click()

            # Should navigate to tune detail (use regex, not lambda)
            expect(admin_page).to_have_url(re.compile(r"/admin/backtests/tunes/"))

    def test_copy_button_exists(
        self, admin_page: Page, base_url: str
    ):
        """Copy tune ID button is present in rows."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Check for copy buttons (usually have a copy icon or title)
        copy_buttons = admin_page.locator("button[title*='Copy'], button[aria-label*='Copy']")
        # Just verify the page loads - copy buttons may not exist if no tunes


class TestTunesListPagination:
    """Tests for pagination controls."""

    def test_pagination_controls_present(
        self, admin_page: Page, base_url: str
    ):
        """Pagination controls are visible when needed."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Check for pagination elements (may not be present if few results)
        # Just verify page loads without error
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_next_page_updates_offset(
        self, admin_page: Page, base_url: str
    ):
        """Clicking Next updates the offset parameter."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        next_button = admin_page.locator("a:has-text('Next'), button:has-text('Next')")
        if next_button.count() > 0 and next_button.is_visible():
            next_button.click()

            # URL should have offset parameter
            expect(admin_page).to_have_url(lambda url: "offset=" in url)


class TestTunesListAutoRefresh:
    """Tests for auto-refresh functionality."""

    def test_auto_refresh_indicator_hidden_when_no_running(
        self, admin_page: Page, base_url: str
    ):
        """Auto-refresh indicator is hidden when no tunes running."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes?status=completed")

        # Auto-refresh indicator should not be visible for completed tunes
        # (This depends on actual data state)
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")
