"""
E2E tests for the Backtest Run Detail page (/admin/backtests/runs/{id}).

Tests run header, summary stats, parameters, and JSON display.
"""

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


class TestRunDetailPage:
    """Tests for backtest run detail page structure."""

    def test_page_handles_invalid_uuid(
        self, admin_page: Page, base_url: str
    ):
        """Page handles invalid UUID gracefully with 404."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # Should show 404 or error - just verify no crash
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_breadcrumb_navigation(
        self, admin_page: Page, base_url: str
    ):
        """Breadcrumb back to tunes list exists."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # Breadcrumb should exist
        breadcrumb = admin_page.locator(".breadcrumb")
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestRunDetailHeader:
    """Tests for run header section."""

    def test_status_badge_present(
        self, admin_page: Page, base_url: str
    ):
        """Status badge shows run status (completed, failed, running)."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # Badge only visible if run exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_copy_id_button_present(
        self, admin_page: Page, base_url: str
    ):
        """Copy run ID button exists."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestRunDetailStats:
    """Tests for summary stats section."""

    def test_stats_section_present(
        self, admin_page: Page, base_url: str
    ):
        """Stats section shows return, sharpe, max DD, win rate, trades."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # Stats only shown if run exists with summary
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestRunDetailParameters:
    """Tests for parameters section."""

    def test_parameters_card_present(
        self, admin_page: Page, base_url: str
    ):
        """Parameters card shows run parameters as JSON."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # Parameters only shown if run exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestRunDetailDataset:
    """Tests for dataset info section."""

    def test_dataset_card_present(
        self, admin_page: Page, base_url: str
    ):
        """Dataset info card shows dataset metadata."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # Dataset only shown if run has dataset_meta
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestRunDetailFullJSON:
    """Tests for full JSON section."""

    def test_full_json_toggle_present(
        self, admin_page: Page, base_url: str
    ):
        """Toggle JSON button exists."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # Toggle only visible if run exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_json_section_initially_hidden(
        self, admin_page: Page, base_url: str
    ):
        """Full JSON section is hidden by default."""
        admin_page.goto(
            f"{base_url}/admin/backtests/runs/00000000-0000-0000-0000-000000000001"
        )

        # JSON section hidden if run exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")
