"""
Playwright configuration for E2E tests.

This file configures pytest-playwright behavior.
"""

# Browser settings
BROWSER = "chromium"
HEADLESS = True
SLOW_MO = 0  # Milliseconds to slow down operations (useful for debugging)

# Timeouts
DEFAULT_TIMEOUT = 30000  # 30 seconds
NAVIGATION_TIMEOUT = 60000  # 60 seconds

# Screenshot settings
SCREENSHOT_ON_FAILURE = True
SCREENSHOT_DIR = "tests/e2e/screenshots"

# Video settings (disabled by default for speed)
VIDEO_ON_FAILURE = False
VIDEO_DIR = "tests/e2e/videos"
