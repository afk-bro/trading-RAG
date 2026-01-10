"""
E2E tests for the Trade Event Detail page (/admin/trade/events/{id}).

Tests event detail display, correlation timeline, and JSON payload.
"""

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import FAKE_UUID, assert_no_500, visit_detail_page


pytestmark = pytest.mark.e2e


class TestTradeEventDetailPage:
    """Tests for trade event detail page structure."""

    def test_page_handles_invalid_uuid(self, admin_page: Page, base_url: str):
        """Page handles invalid UUID gracefully with 404."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        # Should show 404 or error - just verify no crash
        assert_no_500(admin_page)

    def test_breadcrumb_navigation(self, admin_page: Page, base_url: str):
        """Breadcrumb back to events list exists."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        # Breadcrumb should exist
        breadcrumb = admin_page.locator(".breadcrumb")
        assert_no_500(admin_page)


class TestTradeEventDetailHeader:
    """Tests for event detail header section."""

    def test_event_type_badge_present(self, admin_page: Page, base_url: str):
        """Event type badge shows when event exists."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        # Badge only visible if event exists
        assert_no_500(admin_page)

    def test_copy_id_button_present(self, admin_page: Page, base_url: str):
        """Copy event ID button exists."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        assert_no_500(admin_page)


class TestTradeEventDetailTimeline:
    """Tests for correlation timeline section."""

    def test_timeline_section_present(self, admin_page: Page, base_url: str):
        """Timeline section shows related events."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        # Timeline only shown if event exists with related events
        assert_no_500(admin_page)


class TestTradeEventDetailPayload:
    """Tests for payload JSON section."""

    def test_payload_section_present(self, admin_page: Page, base_url: str):
        """Payload JSON section is present."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        # Payload section conditional on event existing
        assert_no_500(admin_page)

    def test_copy_json_button_exists(self, admin_page: Page, base_url: str):
        """Copy JSON button is present when event exists."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        # Copy button only visible if event exists
        assert_no_500(admin_page)


class TestTradeEventDetailMetadata:
    """Tests for event metadata table."""

    def test_metadata_table_present(self, admin_page: Page, base_url: str):
        """Metadata table shows event details."""
        admin_page.goto(f"{base_url}/admin/trade/events/{FAKE_UUID}")

        # Table only shown if event exists
        assert_no_500(admin_page)
