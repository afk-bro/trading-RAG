"""Root conftest for test suite.

Auto-skips e2e and smoke tests that require a running server.
Run explicitly with: pytest tests/e2e -m e2e
                  or: pytest tests/smoke -m smoke
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip e2e and smoke tests unless explicitly requested.

    These tests require a running server and should not run in CI
    unless explicitly invoked.
    """
    # Check if user explicitly requested e2e, smoke, or slow tests
    # via -m marker or by specifying the test path directly
    markexpr = config.getoption("-m", default="")
    explicit_e2e = "e2e" in markexpr
    explicit_smoke = "smoke" in markexpr
    explicit_slow = "slow" in markexpr

    # Check if running specific test paths
    args = config.args
    running_e2e_path = any("tests/e2e" in str(arg) for arg in args)
    running_smoke_path = any("tests/smoke" in str(arg) for arg in args)

    skip_e2e = pytest.mark.skip(
        reason="e2e tests require running server. Run with: pytest tests/e2e -m e2e"
    )
    skip_smoke = pytest.mark.skip(
        reason="smoke tests require running server. Run with: pytest tests/smoke -m smoke"
    )
    skip_slow = pytest.mark.skip(
        reason="slow tests skipped by default. Run with: pytest -m slow"
    )

    for item in items:
        # Skip e2e tests unless explicitly requested
        if "e2e" in item.keywords and not explicit_e2e and not running_e2e_path:
            item.add_marker(skip_e2e)

        # Skip smoke tests unless explicitly requested
        if "smoke" in item.keywords and not explicit_smoke and not running_smoke_path:
            item.add_marker(skip_smoke)

        # Skip slow tests unless explicitly requested
        if "slow" in item.keywords and not explicit_slow:
            item.add_marker(skip_slow)
