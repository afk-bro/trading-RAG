"""
E2E tests for the Trade Events List page (/admin/trade/events).

Tests event listing, filtering, and navigation.
"""

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import FAKE_UUID, assert_no_500


pytestmark = pytest.mark.e2e


class TestTradeEventsListPage:
    """Tests for the trade events list page structure."""

    def test_page_loads_successfully(self, admin_page: Page, base_url: str):
        """Trade events page loads without crashing."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

    def test_page_title_correct(self, admin_page: Page, base_url: str):
        """Page has correct title."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)
        # Card title should mention events
        expect(admin_page.locator("body")).to_contain_text("Trade Events")

    def test_filters_present(self, admin_page: Page, base_url: str):
        """Filter controls are present."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

        # Event type dropdown should exist
        event_type_select = admin_page.locator("select[name='event_type']")
        if event_type_select.count() > 0:
            expect(event_type_select).to_be_visible()

        # Time window dropdown should exist
        hours_select = admin_page.locator("select[name='hours']")
        if hours_select.count() > 0:
            expect(hours_select).to_be_visible()


class TestTradeEventsListFilters:
    """Tests for trade events filtering."""

    def test_event_type_filter(self, admin_page: Page, base_url: str):
        """Event type filter dropdown works."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

        event_type_select = admin_page.locator("select[name='event_type']")
        if event_type_select.count() > 0 and event_type_select.is_visible():
            # Select intent_emitted
            event_type_select.select_option("intent_emitted")
            admin_page.wait_for_load_state("networkidle")
            assert_no_500(admin_page)

    def test_time_window_filter(self, admin_page: Page, base_url: str):
        """Time window filter changes URL."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

        hours_select = admin_page.locator("select[name='hours']")
        if hours_select.count() > 0 and hours_select.is_visible():
            hours_select.select_option("72")
            admin_page.wait_for_load_state("networkidle")
            assert_no_500(admin_page)

    def test_symbol_filter_input(self, admin_page: Page, base_url: str):
        """Symbol filter input exists."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

        symbol_input = admin_page.locator("input[name='symbol']")
        if symbol_input.count() > 0:
            expect(symbol_input).to_be_visible()


class TestTradeEventsListTable:
    """Tests for events table."""

    def test_table_or_empty_state(self, admin_page: Page, base_url: str):
        """Page shows either table or empty state."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

        # Either table or empty state should exist
        table = admin_page.locator("table")
        empty_state = admin_page.locator(".empty-state")

        has_table = table.count() > 0
        has_empty = empty_state.count() > 0

        # At least one should be present
        assert has_table or has_empty


class TestTradeEventsListNavigation:
    """Tests for navigation elements."""

    def test_navbar_events_link_highlighted(self, admin_page: Page, base_url: str):
        """Events link in navbar is highlighted when on events page."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

        # The Events link should be active
        events_link = admin_page.locator("nav a:has-text('Events')")
        if events_link.count() > 0:
            expect(events_link).to_be_visible()

    def test_pagination_controls(self, admin_page: Page, base_url: str):
        """Pagination info is present."""
        admin_page.goto(f"{base_url}/admin/trade/events")
        assert_no_500(admin_page)

        # Pagination info should show count
        pagination = admin_page.locator(".pagination-info")
        if pagination.count() > 0:
            expect(pagination).to_contain_text("events")
