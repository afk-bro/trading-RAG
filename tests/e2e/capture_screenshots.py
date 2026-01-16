#!/usr/bin/env python3
"""Capture screenshots of the Coverage Cockpit UI for review."""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "dev-admin-token")
OUTPUT_DIR = Path(__file__).parent / "screenshots"


def capture_cockpit_screenshots():
    """Capture screenshots of all cockpit UI states."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            extra_http_headers={"X-Admin-Token": ADMIN_TOKEN},
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        # 1. Main cockpit page with queue
        print("Capturing: Main cockpit page...")
        page.goto(f"{BASE_URL}/admin/coverage/cockpit")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=OUTPUT_DIR / "01_cockpit_main.png", full_page=True)

        # 2. Try different status tabs
        print("Capturing: Status tabs...")
        for tab in ["acknowledged", "resolved", "all"]:
            page.goto(f"{BASE_URL}/admin/coverage/cockpit?status={tab}")
            page.wait_for_load_state("networkidle")
            page.screenshot(path=OUTPUT_DIR / f"02_tab_{tab}.png", full_page=True)

        # 3. Queue panel with items (if any exist)
        print("Capturing: Queue panel...")
        page.goto(f"{BASE_URL}/admin/coverage/cockpit?status=all")
        page.wait_for_load_state("networkidle")

        # Check if there are any queue items
        queue_items = page.locator(".queue-item")
        if queue_items.count() > 0:
            print(f"Found {queue_items.count()} queue items")

            # 4. Click first item to show detail panel
            print("Capturing: Detail panel...")
            queue_items.first.click()
            page.wait_for_timeout(500)  # Wait for detail to load
            page.screenshot(path=OUTPUT_DIR / "03_detail_panel.png", full_page=True)

            # 5. Capture triage controls section
            print("Capturing: Triage controls...")
            triage_section = page.locator(".triage-section, .triage-controls")
            if triage_section.count() > 0:
                triage_section.first.scroll_into_view_if_needed()
                page.screenshot(path=OUTPUT_DIR / "04_triage_controls.png", full_page=True)

            # 6. Capture candidates section
            print("Capturing: Candidates section...")
            candidates = page.locator(".candidate-row, .strategy-candidate")
            if candidates.count() > 0:
                candidates.first.scroll_into_view_if_needed()
                page.screenshot(path=OUTPUT_DIR / "05_candidates.png", full_page=True)

                # 7. Try explain button
                print("Capturing: Explanation feature...")
                explain_btn = page.locator("button:has-text('Explain')")
                if explain_btn.count() > 0:
                    explain_btn.first.click()
                    page.wait_for_timeout(2000)  # Wait for LLM response (may timeout)
                    page.screenshot(path=OUTPUT_DIR / "06_explanation.png", full_page=True)
        else:
            print("No queue items found - capturing empty state")
            page.screenshot(path=OUTPUT_DIR / "03_empty_state.png", full_page=True)

        # 8. Capture at different viewport sizes
        print("Capturing: Mobile viewport...")
        context.close()

        mobile_context = browser.new_context(
            extra_http_headers={"X-Admin-Token": ADMIN_TOKEN},
            viewport={"width": 375, "height": 812},
        )
        mobile_page = mobile_context.new_page()
        mobile_page.goto(f"{BASE_URL}/admin/coverage/cockpit")
        mobile_page.wait_for_load_state("networkidle")
        mobile_page.screenshot(path=OUTPUT_DIR / "07_mobile.png", full_page=True)

        mobile_context.close()
        browser.close()

    print(f"\nScreenshots saved to: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    capture_cockpit_screenshots()
