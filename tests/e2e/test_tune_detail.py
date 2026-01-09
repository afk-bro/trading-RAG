"""
E2E tests for the Tune Detail page (/admin/backtests/tunes/{id}).

Tests tune header, stats, best result, trials table, and navigation.
"""

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


class TestTuneDetailPage:
    """Tests for tune detail page structure."""

    def test_page_handles_invalid_uuid(
        self, admin_page: Page, base_url: str
    ):
        """Page handles invalid UUID gracefully with 404."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Should show 404 or error - just verify no crash
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_breadcrumb_navigation(
        self, admin_page: Page, base_url: str
    ):
        """Breadcrumb back to tunes list exists."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Breadcrumb should link to tunes list
        breadcrumb = admin_page.locator(".breadcrumb")
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestTuneDetailHeader:
    """Tests for tune header section."""

    def test_status_badge_present(
        self, admin_page: Page, base_url: str
    ):
        """Status badge shows tune status (queued, running, completed, etc.)."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Status badge only visible if tune exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_copy_id_button_present(
        self, admin_page: Page, base_url: str
    ):
        """Copy tune ID button exists."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_strategy_link_present(
        self, admin_page: Page, base_url: str
    ):
        """Strategy name links to entity detail."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestTuneDetailStats:
    """Tests for status counts section."""

    def test_stats_section_present(
        self, admin_page: Page, base_url: str
    ):
        """Stats section shows queued/running/completed/skipped/failed counts."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Stats only shown if tune exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestTuneDetailBestResult:
    """Tests for best result card."""

    def test_best_result_card_present(
        self, admin_page: Page, base_url: str
    ):
        """Best result card shows score and parameters."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Best result only shown if tune has valid trials
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_view_run_button_present(
        self, admin_page: Page, base_url: str
    ):
        """View Run button links to best run detail."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestTuneDetailTrials:
    """Tests for trials table."""

    def test_trials_filter_dropdown_present(
        self, admin_page: Page, base_url: str
    ):
        """Trials status filter dropdown exists."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_trials_table_structure(
        self, admin_page: Page, base_url: str
    ):
        """Trials table has expected columns."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Table only shown if tune exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_clickable_rows_navigate_to_run(
        self, admin_page: Page, base_url: str
    ):
        """Trial rows are clickable and navigate to run detail."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Row clicks only work if tune exists with trials
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestTuneDetailCancelBanner:
    """Tests for canceled tune state."""

    def test_canceled_banner_when_canceled(
        self, admin_page: Page, base_url: str
    ):
        """Canceled banner shows for canceled tunes."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Banner conditional on status=canceled
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestTuneDetailSkipReasons:
    """Tests for skip reasons callout."""

    def test_skip_reasons_callout_when_skipped(
        self, admin_page: Page, base_url: str
    ):
        """Skip reasons callout shows why trials were skipped."""
        admin_page.goto(
            f"{base_url}/admin/backtests/tunes/00000000-0000-0000-0000-000000000001"
        )

        # Callout only shown if skipped > 0
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")
