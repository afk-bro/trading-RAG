"""
E2E tests for the React Dashboard SPA (/dashboard/).

Tests that the dashboard loads, renders its shell layout, and handles
navigation without crashing. These tests are resilient to empty databases
— they verify rendering, not specific data content.

Run with:
    pytest tests/e2e/test_dashboard.py -m e2e --base-url http://localhost:8000
"""

import pytest
from playwright.sync_api import Page, expect

from tests.e2e.conftest import FAKE_UUID, assert_no_500


pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _goto_dashboard(page: Page, base_url: str, path: str = "/") -> None:
    """Navigate to a dashboard route and wait for the React SPA to hydrate."""
    url = f"{base_url}/dashboard{path}" if path != "/" else f"{base_url}/dashboard/"
    page.goto(url, wait_until="networkidle")
    assert_no_500(page)


# ---------------------------------------------------------------------------
# Dashboard shell & landing page
# ---------------------------------------------------------------------------


class TestDashboardLoads:
    """Verify the React SPA boots and the DashboardShell renders."""

    def test_dashboard_loads(self, admin_page: Page, base_url: str):
        """Navigate to /dashboard/, verify shell layout elements render."""
        _goto_dashboard(admin_page, base_url)

        # The DashboardShell <header> should be present
        header = admin_page.locator("header")
        expect(header).to_be_visible(timeout=15000)

        # Navigation links inside the header
        expect(header.locator("a", has_text="Dashboard")).to_be_visible()
        expect(header.locator("a", has_text="Backtests")).to_be_visible()

        # Workspace switcher button (has aria-haspopup="listbox")
        ws_button = admin_page.locator('button[aria-haspopup="listbox"]')
        expect(ws_button).to_be_visible()


# ---------------------------------------------------------------------------
# Backtests list page
# ---------------------------------------------------------------------------


class TestBacktestsPageLoads:
    """Verify the /dashboard/backtests route renders."""

    def test_backtests_page_loads(self, admin_page: Page, base_url: str):
        """Navigate to /dashboard/backtests, verify heading or empty state."""
        _goto_dashboard(admin_page, base_url, "/backtests")

        # The page should show either:
        #  - "Backtest Runs" heading (when a workspace is selected)
        #  - A workspace picker (when no workspace is in the URL)
        # Either way the page must not crash.
        body = admin_page.locator("body")
        expect(body).not_to_contain_text("Internal Server Error")

        # The header nav should still be visible (shell rendered)
        header = admin_page.locator("header")
        expect(header).to_be_visible(timeout=10000)


# ---------------------------------------------------------------------------
# Workspace switcher interaction
# ---------------------------------------------------------------------------


class TestWorkspaceSwitcher:
    """Verify the workspace switcher dropdown opens and shows content."""

    def test_workspace_switcher_opens(self, admin_page: Page, base_url: str):
        """Click the workspace switcher, verify the dropdown appears."""
        _goto_dashboard(admin_page, base_url)

        ws_button = admin_page.locator('button[aria-haspopup="listbox"]')
        expect(ws_button).to_be_visible(timeout=10000)

        # Button should start closed
        expect(ws_button).to_have_attribute("aria-expanded", "false")

        # Click to open
        ws_button.click()

        # After click, aria-expanded should flip to "true"
        expect(ws_button).to_have_attribute("aria-expanded", "true")

        # Dropdown content should appear — either workspace list or
        # "Loading..." or "No workspaces" text, plus "New workspace" button
        dropdown = admin_page.locator(
            'div[role="listbox"], div:has-text("No workspaces"), div:has-text("Loading")'
        )
        expect(dropdown.first).to_be_visible(timeout=5000)


# ---------------------------------------------------------------------------
# Run detail page
# ---------------------------------------------------------------------------


class TestBacktestRunDetail:
    """Verify the /dashboard/backtests/:runId route handles missing data gracefully."""

    def test_backtest_run_detail_loads(self, admin_page: Page, base_url: str):
        """Navigate to a run detail with a placeholder UUID.

        The page should render without crashing — showing either a
        "not found" state, an error alert, or a workspace picker.
        """
        _goto_dashboard(admin_page, base_url, f"/backtests/{FAKE_UUID}")

        # Wait for the SPA to finish rendering
        admin_page.wait_for_load_state("networkidle")

        # Must not crash with an unhandled exception
        body = admin_page.locator("body")
        expect(body).not_to_contain_text("Internal Server Error")

        # The shell header should still be intact
        header = admin_page.locator("header")
        expect(header).to_be_visible(timeout=10000)


# ---------------------------------------------------------------------------
# Tab navigation on run detail
# ---------------------------------------------------------------------------


class TestTabNavigation:
    """Verify Results and Replay tabs exist on the run detail page."""

    def test_tab_navigation(self, admin_page: Page, base_url: str):
        """On a run detail page, the Results and Replay tabs should be
        present and clickable (if run data loads), or the page should
        degrade gracefully.
        """
        _goto_dashboard(admin_page, base_url, f"/backtests/{FAKE_UUID}")
        admin_page.wait_for_load_state("networkidle")

        # Tabs only render when run data is loaded. With a fake UUID the
        # page may show a workspace picker, error alert, or "not found".
        # We check for tabs but do not fail if the page shows an expected
        # fallback state instead.
        tabs = admin_page.locator('button[role="tab"]')
        tab_count = tabs.count()

        if tab_count >= 2:
            # Tabs rendered — verify their labels
            results_tab = admin_page.locator('button[role="tab"]', has_text="Results")
            replay_tab = admin_page.locator('button[role="tab"]', has_text="Replay")

            expect(results_tab).to_be_visible()
            expect(replay_tab).to_be_visible()

            # Results tab should be selected by default
            expect(results_tab).to_have_attribute("aria-selected", "true")
            expect(replay_tab).to_have_attribute("aria-selected", "false")

            # Click Replay tab
            replay_tab.click()
            expect(replay_tab).to_have_attribute("aria-selected", "true")
            expect(results_tab).to_have_attribute("aria-selected", "false")

            # Click back to Results
            results_tab.click()
            expect(results_tab).to_have_attribute("aria-selected", "true")
        else:
            # Tabs not rendered — page is in a fallback state.
            # Verify the page is still alive (no crash).
            body = admin_page.locator("body")
            expect(body).not_to_contain_text("Internal Server Error")
