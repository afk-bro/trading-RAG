"""Pytest configuration for slow tests.

Slow tests require real model inference and are skipped by default.
To run slow tests, set RUN_SLOW_TESTS=1 environment variable:

    RUN_SLOW_TESTS=1 pytest tests/slow/ -v -m slow

This ensures slow tests don't accidentally run in PR CI pipelines.
"""

import os
import pytest


def pytest_configure(config):
    """Register slow marker if not already registered."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless RUN_SLOW_TESTS=1 is set.

    This provides a safety gate to prevent slow tests from accidentally
    running in CI pipelines that don't explicitly opt in.
    """
    run_slow = os.environ.get("RUN_SLOW_TESTS", "0") == "1"

    if run_slow:
        # User explicitly requested slow tests - run them
        return

    skip_slow = pytest.mark.skip(
        reason="Slow tests require RUN_SLOW_TESTS=1 environment variable"
    )

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
