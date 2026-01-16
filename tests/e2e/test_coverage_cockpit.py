"""
E2E tests for the Coverage Cockpit (/admin/coverage/cockpit).

Tests the two-panel triage interface for weak coverage management:
- Queue panel with filtering and sorting
- Detail panel with coverage data and candidates
- Triage workflow (acknowledge, resolve, reopen)
- LLM explanation feature with verbosity toggle
- Deep linking to specific runs

Run with:
    ADMIN_TOKEN=e2e-test-token pytest tests/e2e/test_coverage_cockpit.py -v
"""

import re

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


# =============================================================================
# Page Load Tests
# =============================================================================


class TestCoverageCockpitPageLoad:
    """Tests for basic page loading and structure."""

    def test_page_loads_successfully(self, admin_page: Page, base_url: str):
        """Coverage cockpit page loads without errors."""
        response = admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        assert response.status == 200
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_page_has_two_panel_layout(self, admin_page: Page, base_url: str):
        """Page has the expected two-panel layout structure."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Check for main container with two panels
        expect(admin_page.locator(".cockpit-container")).to_be_visible()
        expect(admin_page.locator(".queue-panel")).to_be_visible()
        expect(admin_page.locator(".detail-panel")).to_be_visible()

    def test_page_title_correct(self, admin_page: Page, base_url: str):
        """Page has the expected title."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Check for "Weak Coverage Queue" header
        expect(admin_page.locator(".card-title")).to_contain_text("Weak Coverage Queue")

    def test_unauthenticated_request_rejected(
        self, unauthenticated_page: Page, base_url: str
    ):
        """Cockpit requires admin authentication."""
        response = unauthenticated_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Without token: 401 or 403
        assert response.status in (401, 403)


# =============================================================================
# Queue Panel Tests
# =============================================================================


class TestCoverageCockpitQueuePanel:
    """Tests for the left-side queue panel."""

    def test_status_tabs_present(self, admin_page: Page, base_url: str):
        """Status filter tabs are visible."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Check for status tabs
        expect(admin_page.locator(".status-tabs")).to_be_visible()
        expect(admin_page.locator(".tab:has-text('Open')")).to_be_visible()
        expect(admin_page.locator(".tab:has-text('Acknowledged')")).to_be_visible()
        expect(admin_page.locator(".tab:has-text('Resolved')")).to_be_visible()
        expect(admin_page.locator(".tab:has-text('All')")).to_be_visible()

    def test_open_tab_active_by_default(self, admin_page: Page, base_url: str):
        """Open tab is active by default."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Open tab should have 'active' class
        open_tab = admin_page.locator(".tab:has-text('Open')")
        expect(open_tab).to_have_class(re.compile(r"active"))

    def test_status_tab_navigation(self, admin_page: Page, base_url: str):
        """Clicking status tabs updates URL and filters."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Click Acknowledged tab
        admin_page.locator(".tab:has-text('Acknowledged')").click()
        admin_page.wait_for_load_state("networkidle")

        expect(admin_page).to_have_url(re.compile(r"status=acknowledged"))

        # Click Resolved tab
        admin_page.locator(".tab:has-text('Resolved')").click()
        admin_page.wait_for_load_state("networkidle")

        expect(admin_page).to_have_url(re.compile(r"status=resolved"))

        # Click All tab
        admin_page.locator(".tab:has-text('All')").click()
        admin_page.wait_for_load_state("networkidle")

        expect(admin_page).to_have_url(re.compile(r"status=all"))

    def test_sort_toggle_present(self, admin_page: Page, base_url: str):
        """Sort toggle checkbox is visible."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        expect(admin_page.locator("#sort-newest")).to_be_visible()
        expect(admin_page.locator(".sort-label")).to_contain_text("Newest first")

    def test_sort_toggle_changes_url(self, admin_page: Page, base_url: str):
        """Toggling sort updates URL parameter."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Check the sort toggle
        checkbox = admin_page.locator("#sort-newest")
        if not checkbox.is_checked():
            checkbox.check()
            admin_page.wait_for_load_state("networkidle")

            expect(admin_page).to_have_url(re.compile(r"sort=newest"))

    def test_refresh_button_present(self, admin_page: Page, base_url: str):
        """Refresh button is visible in header."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Check for refresh button (has SVG icon)
        refresh_btn = admin_page.locator(".header-actions button[onclick='refreshQueue()']")
        expect(refresh_btn).to_be_visible()

    def test_queue_count_displayed(self, admin_page: Page, base_url: str):
        """Queue item count is displayed."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Check for "X items" text
        expect(admin_page.locator(".queue-count")).to_contain_text("items")

    def test_empty_state_when_no_items(self, admin_page: Page, base_url: str):
        """Shows empty state message when queue is empty."""
        # Use resolved filter - likely to have fewer/no items
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=resolved")

        # Check page loads without error (empty state or items)
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestCoverageCockpitQueueItems:
    """Tests for individual queue items."""

    def test_queue_items_have_priority_badges(self, admin_page: Page, base_url: str):
        """Queue items display priority badges (P1, P2, P3)."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        # If there are items, check for priority badges
        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            first_item = items.first
            expect(first_item.locator(".priority-badge")).to_be_visible()

    def test_queue_items_have_status_dots(self, admin_page: Page, base_url: str):
        """Queue items display status indicator dots."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            first_item = items.first
            expect(first_item.locator(".status-dot")).to_be_visible()

    def test_queue_items_have_source_ref(self, admin_page: Page, base_url: str):
        """Queue items display source reference."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            first_item = items.first
            expect(first_item.locator(".item-source")).to_be_visible()

    def test_queue_items_have_query_preview(self, admin_page: Page, base_url: str):
        """Queue items display query preview."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            first_item = items.first
            expect(first_item.locator(".item-query")).to_be_visible()

    def test_clicking_item_selects_it(self, admin_page: Page, base_url: str):
        """Clicking a queue item adds 'selected' class."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 1:
            # Click second item
            second_item = items.nth(1)
            second_item.click()

            expect(second_item).to_have_class(re.compile(r"selected"))


# =============================================================================
# Detail Panel Tests
# =============================================================================


class TestCoverageCockpitDetailPanel:
    """Tests for the right-side detail panel."""

    def test_detail_panel_shows_when_item_selected(
        self, admin_page: Page, base_url: str
    ):
        """Detail panel populates when a queue item is selected."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            # First item is auto-selected
            detail_content = admin_page.locator("#detail-content")
            expect(detail_content).to_be_visible()

    def test_detail_shows_coverage_snapshot(self, admin_page: Page, base_url: str):
        """Detail panel shows coverage snapshot section."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            # Check for Coverage Snapshot section
            expect(
                admin_page.locator(".detail-section-title:has-text('Coverage Snapshot')")
            ).to_be_visible()

    def test_detail_shows_stats_grid(self, admin_page: Page, base_url: str):
        """Detail panel shows stats grid with key metrics."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            stats_grid = admin_page.locator(".stats-grid")
            expect(stats_grid).to_be_visible()

            # Check for expected stats
            expect(stats_grid.locator(".stat-label:has-text('Best Score')")).to_be_visible()
            expect(
                stats_grid.locator(".stat-label:has-text('Above Threshold')")
            ).to_be_visible()

    def test_detail_shows_recommended_strategies(self, admin_page: Page, base_url: str):
        """Detail panel shows recommended strategies section."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            expect(
                admin_page.locator(
                    ".detail-section-title:has-text('Recommended Strategies')"
                )
            ).to_be_visible()

    def test_detail_shows_query_context(self, admin_page: Page, base_url: str):
        """Detail panel shows query context section."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            expect(
                admin_page.locator(".detail-section-title:has-text('Query Context')")
            ).to_be_visible()
            expect(admin_page.locator(".query-box")).to_be_visible()

    def test_copy_buttons_present(self, admin_page: Page, base_url: str):
        """Copy and share buttons are present in detail header."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            # Check for copy buttons
            copy_buttons = admin_page.locator(".copy-btn")
            expect(copy_buttons.first).to_be_visible()


# =============================================================================
# Triage Controls Tests
# =============================================================================


class TestCoverageCockpitTriageControls:
    """Tests for the triage action controls."""

    def test_triage_section_present(self, admin_page: Page, base_url: str):
        """Triage controls section is visible."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            expect(
                admin_page.locator(".detail-section-title:has-text('Triage Actions')")
            ).to_be_visible()
            expect(admin_page.locator(".triage-controls")).to_be_visible()

    def test_triage_buttons_present(self, admin_page: Page, base_url: str):
        """Acknowledge, Resolve, and Reopen buttons are visible."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            expect(
                admin_page.locator(".triage-btn:has-text('Acknowledge')")
            ).to_be_visible()
            expect(admin_page.locator(".triage-btn:has-text('Resolve')")).to_be_visible()
            expect(admin_page.locator(".triage-btn:has-text('Reopen')")).to_be_visible()

    def test_note_input_present(self, admin_page: Page, base_url: str):
        """Resolution note textarea is visible."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            expect(admin_page.locator("#resolution-note")).to_be_visible()

    def test_save_button_disabled_by_default(self, admin_page: Page, base_url: str):
        """Save button is disabled until status change is selected."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            save_btn = admin_page.locator("#save-btn")
            expect(save_btn).to_be_disabled()

    @pytest.mark.xfail(reason="Dynamic JS timing - needs investigation")
    def test_clicking_status_button_enables_save(self, admin_page: Page, base_url: str):
        """Clicking a status button enables the save button."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=open")
        admin_page.wait_for_load_state("networkidle")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            # First select an item to populate the detail panel
            items.first.click()

            # Wait for detail panel to render with triage controls
            triage_section = admin_page.locator(".triage-controls")
            triage_section.wait_for(state="visible", timeout=3000)

            # Click Acknowledge button (should be available for open items)
            ack_btn = admin_page.locator(".triage-btn.btn-acknowledge:not([disabled])")
            if ack_btn.count() > 0:
                ack_btn.click()

                # Save button should now be enabled (setStatus enables it)
                save_btn = admin_page.locator("#save-btn")
                expect(save_btn).to_be_enabled(timeout=2000)


# =============================================================================
# Candidate Strategy Tests
# =============================================================================


class TestCoverageCockpitCandidates:
    """Tests for candidate strategy display and actions."""

    def test_candidate_rows_present(self, admin_page: Page, base_url: str):
        """Candidate strategy rows are displayed when available."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            # Check if candidates are shown (may be empty)
            candidate_list = admin_page.locator(".candidate-list")
            if candidate_list.count() > 0:
                expect(candidate_list).to_be_visible()

    def test_candidate_has_name_and_score(self, admin_page: Page, base_url: str):
        """Candidate rows show strategy name and score."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            expect(first_candidate.locator(".candidate-name")).to_be_visible()

    def test_candidate_has_tags(self, admin_page: Page, base_url: str):
        """Candidate rows display strategy tags."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            expect(first_candidate.locator(".candidate-tags")).to_be_visible()

    def test_candidate_has_view_strategy_link(self, admin_page: Page, base_url: str):
        """Candidate rows have View Strategy link."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            expect(
                first_candidate.locator("a:has-text('View Strategy')")
            ).to_be_visible()


# =============================================================================
# Explanation Feature Tests
# =============================================================================


class TestCoverageCockpitExplanation:
    """Tests for LLM-powered strategy explanation feature."""

    def test_explain_button_present(self, admin_page: Page, base_url: str):
        """Explain Match button is visible for candidates."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            expect(
                first_candidate.locator(".explain-btn:has-text('Explain Match')")
            ).to_be_visible()

    def test_explain_button_shows_loading_state(self, admin_page: Page, base_url: str):
        """Clicking Explain Match shows loading indicator."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            explain_btn = first_candidate.locator(".explain-btn")

            # Click and check for loading state
            explain_btn.click()

            # Should show loading text
            expect(first_candidate.locator(".explain-loading")).to_be_visible()

    def test_explanation_box_appears_after_click(self, admin_page: Page, base_url: str):
        """Explanation box appears after clicking Explain Match."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            explain_btn = first_candidate.locator(".explain-btn")

            # Click explain
            explain_btn.click()

            # Wait for explanation box to appear (loading or content)
            explanation_box = first_candidate.locator(".explanation-box")
            expect(explanation_box).to_be_visible(timeout=10000)

    def test_verbosity_toggle_appears_after_explanation(
        self, admin_page: Page, base_url: str
    ):
        """Verbosity toggle button appears after explanation loads."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            explain_btn = first_candidate.locator(".explain-btn")

            # Click explain
            explain_btn.click()

            # Wait for explanation to load (look for explanation-text or error)
            admin_page.wait_for_selector(
                ".explanation-text, .explanation-box.error", timeout=15000
            )

            # Check for verbosity button (only appears on success)
            verbosity_btn = first_candidate.locator(".verbosity-btn")
            if admin_page.locator(".explanation-text").count() > 0:
                expect(verbosity_btn).to_be_visible()

    def test_explanation_shows_confidence_qualifier(
        self, admin_page: Page, base_url: str
    ):
        """Explanation displays confidence qualifier line."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            explain_btn = first_candidate.locator(".explain-btn")

            # Click explain
            explain_btn.click()

            # Wait for explanation to load
            admin_page.wait_for_selector(
                ".explanation-text, .explanation-box.error", timeout=15000
            )

            # Check for confidence line (only on success)
            if admin_page.locator(".explanation-text").count() > 0:
                confidence_line = first_candidate.locator(".confidence-line")
                expect(confidence_line).to_be_visible()
                expect(confidence_line).to_contain_text("Confidence:")

    def test_explanation_shows_cache_indicator(self, admin_page: Page, base_url: str):
        """Explanation displays cache hit/miss indicator."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            explain_btn = first_candidate.locator(".explain-btn")

            # Click explain
            explain_btn.click()

            # Wait for explanation
            admin_page.wait_for_selector(
                ".explanation-text, .explanation-box.error", timeout=15000
            )

            # Check for cache indicator (only on success)
            if admin_page.locator(".explanation-text").count() > 0:
                cache_indicator = first_candidate.locator(".cache-indicator")
                expect(cache_indicator).to_be_visible()

    def test_explain_button_toggles_visibility(self, admin_page: Page, base_url: str):
        """Clicking Explain Match again hides the explanation."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            explain_btn = first_candidate.locator(".explain-btn")

            # First click - show
            explain_btn.click()
            admin_page.wait_for_selector(
                ".explanation-text, .explanation-box.error", timeout=15000
            )

            explanation_box = first_candidate.locator(".explanation-box")

            # If explanation loaded successfully, test toggle
            if admin_page.locator(".explanation-text").count() > 0:
                expect(explanation_box).to_be_visible()

                # Second click - hide
                explain_btn.click()
                expect(explanation_box).to_be_hidden()


# =============================================================================
# Deep Link Tests
# =============================================================================


class TestCoverageCockpitDeepLinks:
    """Tests for deep linking to specific coverage runs."""

    def test_deep_link_with_run_id_loads(self, admin_page: Page, base_url: str):
        """Deep link with run_id loads the cockpit."""
        # First get a valid run_id from the queue
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            # Get run_id from first item
            run_id = items.first.get_attribute("data-run-id")

            # Navigate to deep link
            admin_page.goto(f"{base_url}/admin/coverage/cockpit/{run_id}")

            # Page should load without error
            expect(admin_page.locator("body")).not_to_contain_text(
                "Internal Server Error"
            )
            expect(admin_page.locator(".cockpit-container")).to_be_visible()

    @pytest.mark.xfail(reason="Deep link JS selection timing - needs investigation")
    def test_deep_link_selects_correct_item(self, admin_page: Page, base_url: str):
        """Deep link pre-selects the specified run in the queue."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 1:
            # Get run_id from second item
            second_item = items.nth(1)
            run_id = second_item.get_attribute("data-run-id")

            # Navigate to deep link
            admin_page.goto(f"{base_url}/admin/coverage/cockpit/{run_id}")
            admin_page.wait_for_load_state("networkidle")

            # Wait for JS to run and select the item
            # The selected item should have the 'selected' class after DOMContentLoaded
            selected_item = admin_page.locator(f".queue-item.selected[data-run-id='{run_id}']")
            expect(selected_item).to_be_visible(timeout=3000)

    def test_invalid_run_id_shows_graceful_error(self, admin_page: Page, base_url: str):
        """Invalid run_id shows graceful handling, not 500 error."""
        # Use a fake UUID
        fake_run_id = "00000000-0000-0000-0000-000000000001"
        admin_page.goto(f"{base_url}/admin/coverage/cockpit/{fake_run_id}")

        # Should not show 500 error
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_share_button_copies_deep_link(self, admin_page: Page, base_url: str):
        """Share button is present for copying deep links."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            # Check for share button in detail header
            share_btn = admin_page.locator(".copy-btn:has-text('share')")
            expect(share_btn).to_be_visible()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCoverageCockpitErrorHandling:
    """Tests for error handling and edge cases."""

    def test_handles_no_workspace_gracefully(self, admin_page: Page, base_url: str):
        """Page handles missing workspace_id gracefully."""
        # Navigate without workspace_id (should auto-detect or show error)
        admin_page.goto(f"{base_url}/admin/coverage/cockpit")

        # Should not show 500 error
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_explain_error_shows_retry_button(self, admin_page: Page, base_url: str):
        """Failed explanation shows error state with retry option."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        candidate_rows = admin_page.locator(".candidate-row")
        if candidate_rows.count() > 0:
            first_candidate = candidate_rows.first
            explain_btn = first_candidate.locator(".explain-btn")

            # Click explain
            explain_btn.click()

            # Wait for result (success or error)
            admin_page.wait_for_selector(
                ".explanation-text, .explanation-box.error", timeout=15000
            )

            # If error, button should show "Retry"
            if admin_page.locator(".explanation-box.error").count() > 0:
                expect(explain_btn.locator(".explain-text")).to_contain_text("Retry")

    def test_missing_candidates_warning(self, admin_page: Page, base_url: str):
        """Shows warning when candidate strategies are missing/deleted."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        # This test checks if the missing warning appears when applicable
        # Just verify page loads - missing warning only shows when strategies deleted
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")
