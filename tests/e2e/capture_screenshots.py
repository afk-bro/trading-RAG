#!/usr/bin/env python3
"""Capture portfolio-grade screenshots of the Coverage Cockpit UI.

Seeds fixture data first to ensure populated states.

Usage:
    ADMIN_TOKEN=dev-admin-token python tests/e2e/capture_screenshots.py
"""

import os
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "dev-admin-token")
OUTPUT_DIR = Path(__file__).parent / "screenshots"


def seed_fixture_data() -> dict:
    """Seed deterministic fixture data via API."""
    print("Seeding fixture data...")
    response = requests.post(
        f"{BASE_URL}/admin/coverage/seed?clear_existing=true",
        headers={"X-Admin-Token": ADMIN_TOKEN},
    )
    if response.status_code != 200:
        print(f"Warning: Seed failed with status {response.status_code}")
        print(response.text)
        return {}

    data = response.json()
    print(f"  - Created {data['strategies_created']} strategies")
    print(f"  - Created {data['match_runs_created']} match_runs")
    return data


def capture_cockpit_screenshots():
    """Capture screenshots of all cockpit UI states."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Seed data first
    seed_data = seed_fixture_data()
    if not seed_data:
        print("Proceeding without seeded data...")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            extra_http_headers={"X-Admin-Token": ADMIN_TOKEN},
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        # =========================================================================
        # Empty/Default States (01-03)
        # =========================================================================

        # 01. Main cockpit page (Open tab, may be empty initially)
        print("Capturing: 01_cockpit_main.png (Open tab default)")
        page.goto(f"{BASE_URL}/admin/coverage/cockpit")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=OUTPUT_DIR / "01_cockpit_main.png", full_page=True)

        # 02. Status tabs overview
        print("Capturing: 02_tab_*.png (status tabs)")
        for tab in ["acknowledged", "resolved"]:
            page.goto(f"{BASE_URL}/admin/coverage/cockpit?status={tab}")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=OUTPUT_DIR / f"02_tab_{tab}.png", full_page=True)

        # 03. Empty state (use resolved which might be empty)
        print("Capturing: 03_empty_or_resolved.png")
        page.goto(f"{BASE_URL}/admin/coverage/cockpit?status=resolved")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=OUTPUT_DIR / "03_empty_or_resolved.png", full_page=True)

        # =========================================================================
        # Populated States (04-10)
        # =========================================================================

        # 04. Queue populated with all items
        print("Capturing: 04_queue_populated.png (All tab with 8 items)")
        page.goto(f"{BASE_URL}/admin/coverage/cockpit?status=all")
        page.wait_for_load_state("networkidle")
        page.locator(".queue-item").first.wait_for(state="visible", timeout=5000)
        page.screenshot(path=OUTPUT_DIR / "04_queue_populated.png", full_page=True)

        queue_items = page.locator(".queue-item")
        item_count = queue_items.count()
        print(f"  Found {item_count} queue items")

        if item_count > 0:
            # 05. Detail panel with item selected
            print("Capturing: 05_detail_panel.png (first item selected)")
            queue_items.first.click()
            page.wait_for_timeout(500)
            page.screenshot(path=OUTPUT_DIR / "05_detail_panel.png", full_page=True)

            # 06. Triage controls section focused
            print("Capturing: 06_triage_controls.png")
            triage_section = page.locator(".triage-controls")
            if triage_section.count() > 0:
                triage_section.first.scroll_into_view_if_needed()
                page.wait_for_timeout(200)
                page.screenshot(path=OUTPUT_DIR / "06_triage_controls.png", full_page=True)

            # 07. Candidate strategies section
            print("Capturing: 07_candidates.png")
            # Find an item with candidates
            for i in range(min(item_count, 5)):
                queue_items.nth(i).click()
                page.wait_for_timeout(300)
                candidates = page.locator(".candidate-row")
                if candidates.count() > 0:
                    candidates.first.scroll_into_view_if_needed()
                    page.screenshot(
                        path=OUTPUT_DIR / "07_candidates.png", full_page=True
                    )
                    print(f"  Found {candidates.count()} candidates on item {i}")
                    break

            # 08. Missing strategy warning
            print("Capturing: 08_missing_warning.png")
            for i in range(item_count):
                queue_items.nth(i).click()
                page.wait_for_timeout(300)
                missing_warning = page.locator(".missing-strategies-warning")
                if missing_warning.count() > 0 and missing_warning.is_visible():
                    missing_warning.scroll_into_view_if_needed()
                    page.screenshot(
                        path=OUTPUT_DIR / "08_missing_warning.png", full_page=True
                    )
                    print(f"  Found missing warning on item {i}")
                    break

            # 09. Explanation loading state
            print("Capturing: 09_explain_loading.png")
            # Find an item with candidates and click explain
            for i in range(min(item_count, 5)):
                queue_items.nth(i).click()
                page.wait_for_timeout(300)
                explain_btn = page.locator(".explain-btn").first
                if explain_btn.count() > 0 and explain_btn.is_visible():
                    explain_btn.click()
                    # Capture loading state quickly
                    page.wait_for_timeout(100)
                    page.screenshot(
                        path=OUTPUT_DIR / "09_explain_loading.png", full_page=True
                    )
                    break

            # 10. Explanation result (success or error)
            print("Capturing: 10_explain_result.png")
            # Wait for explanation to load (success or error)
            try:
                page.wait_for_selector(
                    ".explanation-text, .explanation-box.error", timeout=15000
                )
                page.screenshot(
                    path=OUTPUT_DIR / "10_explain_result.png", full_page=True
                )
                print("  Captured explanation result")
            except Exception:
                print("  Explanation timed out - no screenshot")

        # =========================================================================
        # Responsive/Mobile (11)
        # =========================================================================

        print("Capturing: 11_mobile.png (mobile viewport)")
        context.close()

        mobile_context = browser.new_context(
            extra_http_headers={"X-Admin-Token": ADMIN_TOKEN},
            viewport={"width": 375, "height": 812},
        )
        mobile_page = mobile_context.new_page()
        mobile_page.goto(f"{BASE_URL}/admin/coverage/cockpit?status=all")
        mobile_page.wait_for_load_state("networkidle")
        mobile_page.locator(".queue-item, .empty-state").first.wait_for(
            state="visible", timeout=5000
        )
        mobile_page.screenshot(path=OUTPUT_DIR / "11_mobile.png", full_page=True)

        mobile_context.close()
        browser.close()

    print(f"\nScreenshots saved to: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    capture_cockpit_screenshots()
