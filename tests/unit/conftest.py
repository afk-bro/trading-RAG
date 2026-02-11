"""Shared fixtures for unit tests."""

from uuid import uuid4

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def _bypass_auth_for_router_tests(request):
    """Auto-patch auth dependencies so router unit tests don't hit 401.

    The require_auth() dependency calls get_current_user_v2() and
    get_workspace_ctx() internally. By patching these to return
    admin-level contexts, router tests can focus on business logic
    without needing to set up JWT tokens.

    Tests in test_security_auth.py explicitly test the auth functions
    and provide their own mocks, so they opt out via the marker check.
    """
    # Skip for tests that explicitly test auth (they mock internally)
    if "test_security_auth" in request.node.nodeid:
        yield
        return

    from app.deps.security import RequestContext, WorkspaceContext

    admin_ctx = RequestContext(is_admin=True, is_admin_token=True)
    dummy_ws = WorkspaceContext(workspace_id=uuid4())

    with (
        patch(
            "app.deps.security.get_current_user_v2",
            new_callable=AsyncMock,
            return_value=admin_ctx,
        ),
        patch(
            "app.deps.security.get_workspace_ctx",
            return_value=dummy_ws,
        ),
        patch(
            "app.core.lifespan.get_db_pool",
            return_value=MagicMock(),
        ),
    ):
        yield
