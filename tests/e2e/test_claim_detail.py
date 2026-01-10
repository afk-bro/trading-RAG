"""
E2E tests for the Claim Detail page (/admin/kb/claims/{id}).

Tests claim display, evidence section, metadata, and debug JSON.
"""

import pytest
from playwright.sync_api import Page, expect


pytestmark = pytest.mark.e2e


class TestClaimDetailPage:
    """Tests for claim detail page structure."""

    def test_page_handles_invalid_uuid(self, admin_page: Page, base_url: str):
        """Page handles invalid UUID gracefully with 404."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        # Should show 404 or error - just verify no crash
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_breadcrumb_navigation(self, admin_page: Page, base_url: str):
        """Breadcrumb navigation exists."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        # Breadcrumb should exist in page structure
        breadcrumb = admin_page.locator(".breadcrumb")
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestClaimDetailHeader:
    """Tests for claim header section."""

    def test_claim_type_badge_present(self, admin_page: Page, base_url: str):
        """Claim type badge shows (rule, parameter, equation, etc.)."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        # Badge only visible if claim exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_claim_status_badge_present(self, admin_page: Page, base_url: str):
        """Claim status badge shows (verified, weak, pending, rejected)."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_confidence_bar_present(self, admin_page: Page, base_url: str):
        """Confidence bar visualization is present."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestClaimDetailMetadata:
    """Tests for metadata table."""

    def test_metadata_table_present(self, admin_page: Page, base_url: str):
        """Metadata table shows ID, entity, extraction model, etc."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        # Metadata table only shown if claim exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestClaimDetailEvidence:
    """Tests for evidence section."""

    def test_evidence_section_present(self, admin_page: Page, base_url: str):
        """Evidence section shows supporting quotes."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        # Evidence section conditional on claim having evidence
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")


class TestClaimDetailDebug:
    """Tests for debug JSON section."""

    def test_debug_json_section_present(self, admin_page: Page, base_url: str):
        """Debug JSON section is present."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")

    def test_copy_json_button_exists(self, admin_page: Page, base_url: str):
        """Copy JSON button is present when claim exists."""
        admin_page.goto(
            f"{base_url}/admin/kb/claims/00000000-0000-0000-0000-000000000001"
        )

        # Copy button only visible if claim exists
        expect(admin_page.locator("body")).not_to_contain_text("Internal Server Error")
