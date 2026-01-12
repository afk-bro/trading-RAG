"""Tests for auth dependencies."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import HTTPException


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_default_values(self):
        from app.deps.security import RequestContext

        ctx = RequestContext()
        assert ctx.user_id is None
        assert ctx.workspace_id is None
        assert ctx.role is None
        assert ctx.is_admin is False

    def test_admin_context(self):
        from app.deps.security import RequestContext

        ctx = RequestContext(is_admin=True, workspace_id=uuid4())
        assert ctx.is_admin is True
        assert ctx.user_id is None


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_admin_token_bypass(self):
        """Admin token returns admin context."""
        from app.deps.security import get_current_user_v2

        with patch("app.deps.security.verify_admin_token", return_value=True):
            ctx = await get_current_user_v2(
                authorization=None,
                x_admin_token="valid-token",
            )

        assert ctx.is_admin is True
        assert ctx.user_id is None

    @pytest.mark.asyncio
    async def test_missing_auth_raises_401(self):
        """Missing authorization raises 401."""
        from app.deps.security import get_current_user_v2

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_v2(
                authorization=None,
                x_admin_token=None,
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_bearer_format_raises_401(self):
        """Non-Bearer auth raises 401."""
        from app.deps.security import get_current_user_v2

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_v2(
                authorization="Basic abc123",
                x_admin_token=None,
            )

        assert exc_info.value.status_code == 401


class TestRequireWorkspaceAccess:
    """Tests for require_workspace_access dependency."""

    @pytest.mark.asyncio
    async def test_admin_bypass_with_workspace(self):
        """Admin can access any workspace."""
        from app.deps.security import RequestContext, require_workspace_access_v2

        ctx = RequestContext(is_admin=True)
        workspace_id = uuid4()

        mock_pool = MagicMock()

        result = await require_workspace_access_v2(
            ctx=ctx,
            workspace_id=workspace_id,
            min_role="viewer",
            pool=mock_pool,
        )

        assert result.is_admin is True
        assert result.workspace_id == workspace_id

    @pytest.mark.asyncio
    async def test_non_member_raises_403(self):
        """Non-member raises 403."""
        from app.deps.security import RequestContext, require_workspace_access_v2

        ctx = RequestContext(user_id=uuid4())
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None  # No membership
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(HTTPException) as exc_info:
            await require_workspace_access_v2(
                ctx=ctx,
                workspace_id=workspace_id,
                min_role="viewer",
                pool=mock_pool,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_insufficient_role_raises_403(self):
        """Insufficient role raises 403."""
        from app.deps.security import RequestContext, require_workspace_access_v2

        ctx = RequestContext(user_id=uuid4())
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"role": "viewer"}  # Has viewer
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(HTTPException) as exc_info:
            await require_workspace_access_v2(
                ctx=ctx,
                workspace_id=workspace_id,
                min_role="admin",  # Requires admin
                pool=mock_pool,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_sufficient_role_returns_context(self):
        """Member with sufficient role returns context."""
        from app.deps.security import RequestContext, require_workspace_access_v2

        user_id = uuid4()
        ctx = RequestContext(user_id=user_id)
        workspace_id = uuid4()

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"role": "admin"}
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await require_workspace_access_v2(
            ctx=ctx,
            workspace_id=workspace_id,
            min_role="member",
            pool=mock_pool,
        )

        assert result.user_id == user_id
        assert result.workspace_id == workspace_id
        assert result.role == "admin"
