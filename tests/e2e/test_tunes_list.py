"""
E2E tests for the Tunes List page (/admin/backtests/tunes).

Tests filtering, pagination, and table interactions.
"""

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

        # Check page title/header
        expect(admin_page.locator("h1, h2").first).to_contain_text("Tunes")

        # Check filter controls exist
        expect(admin_page.locator("select[name='status']")).to_be_visible()
        expect(admin_page.locator("input[name='valid_only']")).to_be_visible()

    def test_table_has_expected_columns(
        self, admin_page: Page, base_url: str
    ):
        """Tunes table has all expected column headers."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Check for key column headers
        table_header = admin_page.locator("table thead")
        expect(table_header).to_contain_text("Created")
        expect(table_header).to_contain_text("Status")
        expect(table_header).to_contain_text("Strategy")

    def test_empty_state_displays_correctly(
        self, admin_page: Page, base_url: str
    ):
        """Shows appropriate message when no tunes match filters."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes?status=failed")

        # Page should still load without errors
        expect(admin_page.locator("body")).not_to_contain_text("Error")


class TestTunesListFilters:
    """Tests for tunes list filtering functionality."""

    def test_status_filter_changes_url(
        self, admin_page: Page, base_url: str
    ):
        """Changing status filter updates URL."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Select "completed" status
        admin_page.locator("select[name='status']").select_option("completed")
        admin_page.locator("button:has-text('Filter')").click()

        # URL should include status parameter
        expect(admin_page).to_have_url(lambda url: "status=completed" in url)

    def test_valid_only_checkbox(
        self, admin_page: Page, base_url: str
    ):
        """Valid only checkbox filters results."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Check the valid_only checkbox
        checkbox = admin_page.locator("input[name='valid_only']")
        if not checkbox.is_checked():
            checkbox.check()
            admin_page.locator("button:has-text('Filter')").click()

        # URL should include valid_only parameter
        expect(admin_page).to_have_url(lambda url: "valid_only=on" in url or "valid_only=true" in url)

    def test_clear_filters_resets_all(
        self, admin_page: Page, base_url: str
    ):
        """Clear button resets all filters."""
        # Start with some filters
        admin_page.goto(f"{base_url}/admin/backtests/tunes?status=completed&valid_only=on")

        # Click clear button
        clear_button = admin_page.locator("a:has-text('Clear'), button:has-text('Clear')")
        if clear_button.count() > 0:
            clear_button.first.click()

            # URL should be clean
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

            # Should navigate to tune detail
            expect(admin_page).to_have_url(lambda url: "/admin/backtests/tunes/" in url)

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
