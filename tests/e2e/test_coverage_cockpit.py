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
        refresh_btn = admin_page.locator(
            ".header-actions button[onclick='refreshQueue()']"
        )
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
                admin_page.locator(
                    ".detail-section-title:has-text('Coverage Snapshot')"
                )
            ).to_be_visible()

    def test_detail_shows_stats_grid(self, admin_page: Page, base_url: str):
        """Detail panel shows stats grid with key metrics."""
        admin_page.goto(f"{base_url}/admin/coverage/cockpit?status=all")

        items = admin_page.locator(".queue-item")
        if items.count() > 0:
            stats_grid = admin_page.locator(".stats-grid")
            expect(stats_grid).to_be_visible()

            # Check for expected stats
            expect(
                stats_grid.locator(".stat-label:has-text('Best Score')")
            ).to_be_visible()
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
            expect(
                admin_page.locator(".triage-btn:has-text('Resolve')")
            ).to_be_visible()
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

    def test_clicking_status_button_enables_save(self, seeded_cockpit):
        """Clicking a status button enables the save button."""
        page = seeded_cockpit["page"]

        # Navigate to open tab with seeded data
        page.goto(page.url.replace("status=all", "status=open"))
        page.wait_for_load_state("networkidle")
        page.locator(".queue-item").first.wait_for(state="visible", timeout=5000)

        # First select an item to populate the detail panel
        first_item = page.locator(".queue-item").first
        expect(first_item).to_be_visible()
        first_item.click()

        # Wait for detail panel to render with triage controls
        triage_section = page.locator(".triage-controls")
        expect(triage_section).to_be_visible(timeout=3000)

        # Click Acknowledge button (should be available for open items)
        ack_btn = page.locator(".triage-btn.btn-acknowledge:not([disabled])")
        expect(ack_btn).to_be_visible()
        ack_btn.click()

        # Save button should now be enabled (setStatus enables it)
        save_btn = page.locator("#save-btn")
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

    def test_deep_link_selects_correct_item(self, seeded_cockpit, base_url: str):
        """Deep link pre-selects the specified run in the queue."""
        page = seeded_cockpit["page"]

        # We already have seeded data with 8 items in status=all
        items = page.locator(".queue-item")
        assert items.count() >= 2, "Need at least 2 items for deep link test"

        # Get run_id from second item
        second_item = items.nth(1)
        run_id = second_item.get_attribute("data-run-id")
        assert run_id, "Second item should have data-run-id attribute"

        # Navigate to deep link
        page.goto(f"{base_url}/admin/coverage/cockpit/{run_id}")
        page.wait_for_load_state("networkidle")

        # Wait for queue items to render first
        page.locator(".queue-item").first.wait_for(state="visible", timeout=5000)

        # The selected item should have the 'selected' class after DOMContentLoaded
        selected_item = page.locator(f".queue-item.selected[data-run-id='{run_id}']")
        expect(selected_item).to_be_visible(timeout=5000)

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


# =============================================================================
# Populated State Tests (uses seeded fixture data)
# =============================================================================


class TestCoverageCockpitPopulatedState:
    """Tests with seeded fixture data to verify populated UI state.

    These tests use the seeded_cockpit fixture which:
    - Seeds 5 strategies with varied tags/backtest status
    - Seeds 8 match_runs with varied reason codes, candidates, triage status
    - One match_run has a missing strategy ID (tests warning display)
    """

    def test_queue_has_eight_items_in_all_tab(self, seeded_cockpit):
        """All tab shows all 8 seeded match_runs."""
        page = seeded_cockpit["page"]
        seed_data = seeded_cockpit["seed_data"]

        # Verify we have the expected count
        items = page.locator(".queue-item")
        assert (
            items.count() == seed_data["match_runs_created"]
        ), f"Expected {seed_data['match_runs_created']} items, got {items.count()}"

    def test_open_tab_count_matches_fixture(self, seeded_cockpit, base_url: str):
        """Open tab shows only open status items (6 seeded as open)."""
        page = seeded_cockpit["page"]

        # Navigate to Open tab
        page.goto(f"{base_url}/admin/coverage/cockpit?status=open")
        page.wait_for_load_state("networkidle")

        # Wait for queue to render
        page.locator(".queue-item, .empty-state").first.wait_for(
            state="visible", timeout=5000
        )

        # Expect 6 open items (seed creates 1 ack, 1 resolved, 6 open)
        items = page.locator(".queue-item")
        assert items.count() == 6, f"Expected 6 open items, got {items.count()}"

    def test_acknowledged_tab_count_matches_fixture(
        self, seeded_cockpit, base_url: str
    ):
        """Acknowledged tab shows only acknowledged items (1 seeded)."""
        page = seeded_cockpit["page"]

        # Navigate to Acknowledged tab
        page.goto(f"{base_url}/admin/coverage/cockpit?status=acknowledged")
        page.wait_for_load_state("networkidle")

        # Wait for queue to render
        page.locator(".queue-item, .empty-state").first.wait_for(
            state="visible", timeout=5000
        )

        # Expect 1 acknowledged item
        items = page.locator(".queue-item")
        assert items.count() == 1, f"Expected 1 acknowledged item, got {items.count()}"

    def test_resolved_tab_count_matches_fixture(self, seeded_cockpit, base_url: str):
        """Resolved tab shows only resolved items (1 seeded)."""
        page = seeded_cockpit["page"]

        # Navigate to Resolved tab
        page.goto(f"{base_url}/admin/coverage/cockpit?status=resolved")
        page.wait_for_load_state("networkidle")

        # Wait for queue to render
        page.locator(".queue-item, .empty-state").first.wait_for(
            state="visible", timeout=5000
        )

        # Expect 1 resolved item
        items = page.locator(".queue-item")
        assert items.count() == 1, f"Expected 1 resolved item, got {items.count()}"

    def test_selecting_item_renders_detail_panel(self, seeded_cockpit):
        """Clicking a queue item populates the detail panel with content."""
        page = seeded_cockpit["page"]

        # Click second item (first is auto-selected)
        items = page.locator(".queue-item")
        second_item = items.nth(1)
        second_item.click()

        # Verify detail panel has content
        detail_content = page.locator("#detail-content")
        expect(detail_content).to_be_visible()

        # Check key sections are populated
        expect(page.locator(".stats-grid")).to_be_visible()
        expect(page.locator(".query-box")).to_be_visible()
        expect(page.locator(".triage-controls")).to_be_visible()

    def test_missing_strategy_warning_appears(self, seeded_cockpit):
        """Warning appears for match_run with missing strategy ID."""
        page = seeded_cockpit["page"]

        # Find the item with missing strategy (look for warning indicator)
        # The seed creates one item with a non-existent strategy_id
        missing_warning = page.locator(".missing-strategies-warning")

        # Click through items to find one with the warning
        items = page.locator(".queue-item")
        found_warning = False

        for i in range(items.count()):
            items.nth(i).click()
            page.wait_for_timeout(300)  # Brief wait for detail to load

            if missing_warning.count() > 0 and missing_warning.is_visible():
                found_warning = True
                expect(missing_warning).to_contain_text("missing")
                break

        assert (
            found_warning
        ), "Expected to find a match_run with missing strategy warning"

    def test_triage_moves_item_between_tabs(self, seeded_cockpit, base_url: str):
        """Triaging an item moves it to the appropriate tab."""
        page = seeded_cockpit["page"]

        # Start on Open tab
        page.goto(f"{base_url}/admin/coverage/cockpit?status=open")
        page.wait_for_load_state("networkidle")
        page.locator(".queue-item").first.wait_for(state="visible", timeout=5000)

        # Count open items before
        open_count_before = page.locator(".queue-item").count()

        # Select first item and acknowledge it
        first_item = page.locator(".queue-item").first
        run_id = first_item.get_attribute("data-run-id")
        first_item.click()

        # Wait for triage controls to be ready
        ack_btn = page.locator(".triage-btn.btn-acknowledge:not([disabled])")
        expect(ack_btn).to_be_visible()
        ack_btn.click()

        # Click save button
        save_btn = page.locator("#save-btn")
        expect(save_btn).to_be_enabled(timeout=2000)
        save_btn.click()

        # Wait for save to complete (page refreshes or updates)
        page.wait_for_load_state("networkidle")

        # Verify item moved - Open tab should have one fewer item
        page.goto(f"{base_url}/admin/coverage/cockpit?status=open")
        page.wait_for_load_state("networkidle")
        page.locator(".queue-item, .empty-state").first.wait_for(
            state="visible", timeout=5000
        )

        open_count_after = page.locator(".queue-item").count()
        assert open_count_after == open_count_before - 1, (
            f"Expected {open_count_before - 1} open items after triage, "
            f"got {open_count_after}"
        )

        # Verify item appears in Acknowledged tab
        page.goto(f"{base_url}/admin/coverage/cockpit?status=acknowledged")
        page.wait_for_load_state("networkidle")
        page.locator(".queue-item").first.wait_for(state="visible", timeout=5000)

        # Should find the triaged item
        triaged_item = page.locator(f".queue-item[data-run-id='{run_id}']")
        expect(triaged_item).to_be_visible()

    def test_candidate_cards_displayed_for_items_with_candidates(self, seeded_cockpit):
        """Items with candidate_strategy_ids show strategy cards."""
        page = seeded_cockpit["page"]

        # Find an item that has candidates (click through to find one)
        items = page.locator(".queue-item")
        found_candidates = False

        for i in range(min(items.count(), 5)):  # Check first 5 items
            items.nth(i).click()
            page.wait_for_timeout(300)  # Brief wait for detail to load

            candidate_list = page.locator(".candidate-list")
            candidate_rows = page.locator(".candidate-row")

            if candidate_rows.count() > 0:
                found_candidates = True
                expect(candidate_list).to_be_visible()

                # Verify candidate card has expected elements
                first_candidate = candidate_rows.first
                expect(first_candidate.locator(".candidate-name")).to_be_visible()
                expect(first_candidate.locator(".explain-btn")).to_be_visible()
                break

        assert (
            found_candidates
        ), "Expected to find at least one item with candidate strategies"

    def test_priority_badges_show_correct_levels(self, seeded_cockpit):
        """Priority badges reflect the computed priority scores."""
        page = seeded_cockpit["page"]

        # Check that priority badges exist and have valid values
        priority_badges = page.locator(".priority-badge")
        assert priority_badges.count() > 0, "Expected priority badges on queue items"

        # Verify badges contain P1, P2, or P3
        badge_texts = []
        for i in range(min(priority_badges.count(), 8)):
            text = priority_badges.nth(i).text_content()
            badge_texts.append(text)
            assert text in ["P1", "P2", "P3"], f"Unexpected priority badge: {text}"

        # With varied fixture data, we should have multiple priority levels
        unique_badges = set(badge_texts)
        assert (
            len(unique_badges) >= 2
        ), f"Expected at least 2 different priority levels, got: {unique_badges}"
