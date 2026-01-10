"""
E2E tests for Run Plans Admin UI.

Tests that the run plans list and detail pages work correctly.
"""

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


# Fake UUID for testing detail pages
FAKE_RUN_PLAN_ID = "00000000-0000-0000-0000-000000000001"


class TestRunPlansListPage:
    """Tests for /admin/testing/run-plans list page."""

    def test_run_plans_list_returns_200(
        self, admin_page: Page, base_url: str
    ):
        """Run plans list page returns 200 with valid auth."""
        response = admin_page.goto(f"{base_url}/admin/testing/run-plans")
        assert response.status == 200

    def test_run_plans_list_shows_empty_state(
        self, admin_page: Page, base_url: str
    ):
        """Run plans list shows empty state when no run plans exist."""
        admin_page.goto(f"{base_url}/admin/testing/run-plans")

        # Should either show run plans table or empty state
        # Both are valid - page should not crash
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_run_plans_list_no_500(
        self, admin_page: Page, base_url: str
    ):
        """Run plans list never returns 500 error."""
        admin_page.goto(f"{base_url}/admin/testing/run-plans")

        # Page should not contain 500 error text
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")
        expect(admin_page.locator("body")).not_to_contain_text("500")

    def test_run_plans_list_has_nav_link(
        self, admin_page: Page, base_url: str
    ):
        """Run Plans link exists in navigation."""
        admin_page.goto(f"{base_url}/admin/testing/run-plans")

        # Nav link should be visible
        expect(admin_page.locator("nav a[href='/admin/testing/run-plans']")).to_be_visible()


class TestRunPlanDetailPage:
    """Tests for /admin/testing/run-plans/{run_plan_id} detail page."""

    def test_run_plan_detail_fake_uuid_graceful(
        self, admin_page: Page, base_url: str
    ):
        """Run plan detail page handles non-existent UUID gracefully (not 500)."""
        response = admin_page.goto(
            f"{base_url}/admin/testing/run-plans/{FAKE_RUN_PLAN_ID}"
        )

        # Should return 200 with graceful error message, not 500
        assert response.status == 200

        # Should show "not found" message, not crash
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_run_plan_detail_random_uuid_graceful(
        self, admin_page: Page, base_url: str
    ):
        """Run plan detail page handles random UUID gracefully."""
        import uuid
        random_uuid = str(uuid.uuid4())

        response = admin_page.goto(
            f"{base_url}/admin/testing/run-plans/{random_uuid}"
        )

        # Should return 200 with graceful error, not 500
        assert response.status == 200
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_run_plan_detail_invalid_id_graceful(
        self, admin_page: Page, base_url: str
    ):
        """Run plan detail page handles invalid ID formats gracefully."""
        # Non-UUID string
        response = admin_page.goto(
            f"{base_url}/admin/testing/run-plans/not-a-valid-uuid"
        )

        # Should handle gracefully - either 200 with error message or 422
        # Should NOT be 500
        assert response.status in (200, 404, 422)
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestRunPlansNavigation:
    """Tests for Run Plans navigation."""

    def test_can_navigate_to_run_plans_from_tunes(
        self, admin_page: Page, base_url: str
    ):
        """Can navigate from Tunes page to Run Plans via navbar."""
        admin_page.goto(f"{base_url}/admin/backtests/tunes")

        # Click Run Plans link
        admin_page.locator("nav a[href='/admin/testing/run-plans']").click()

        # Should navigate to run plans page
        expect(admin_page).to_have_url(f"{base_url}/admin/testing/run-plans")

    def test_run_plans_active_in_nav(
        self, admin_page: Page, base_url: str
    ):
        """Run Plans nav link is active when on run plans page."""
        admin_page.goto(f"{base_url}/admin/testing/run-plans")

        # Nav link should have active class
        run_plans_link = admin_page.locator("nav a[href='/admin/testing/run-plans']")
        expect(run_plans_link).to_have_attribute("class", "active")
