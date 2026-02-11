"""Tests for auth dependencies (JWT, require_auth, workspace access)."""

import time

import jwt as pyjwt
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JWT_SECRET = "test-jwt-secret-for-unit-tests"
SUPABASE_URL = "https://test-project.supabase.co"
ISSUER = SUPABASE_URL + "/auth/v1"
AUDIENCE = "authenticated"


def _make_settings(**overrides):
    """Create a mock Settings object with JWT fields."""
    s = MagicMock()
    s.supabase_jwt_secret = overrides.get("supabase_jwt_secret", JWT_SECRET)
    s.supabase_jwt_audience = overrides.get("supabase_jwt_audience", AUDIENCE)
    s.supabase_jwt_issuer = overrides.get("supabase_jwt_issuer", None)
    s.supabase_url = overrides.get("supabase_url", SUPABASE_URL)
    return s


def _make_token(
    sub=None,
    exp_delta=3600,
    iss=ISSUER,
    aud=AUDIENCE,
    secret=JWT_SECRET,
    extra=None,
):
    """Create a signed JWT for testing."""
    payload = {}
    if sub is not None:
        payload["sub"] = sub
    if exp_delta is not None:
        payload["exp"] = int(time.time()) + exp_delta
    if iss is not None:
        payload["iss"] = iss
    if aud is not None:
        payload["aud"] = aud
    if extra:
        payload.update(extra)
    return pyjwt.encode(payload, secret, algorithm="HS256")


class _DictLike(dict):
    """Dict subclass that allows .get() without MagicMock interference."""

    pass


def _make_request(
    headers=None,
    cookies=None,
    path_params=None,
    query_params=None,
):
    """Create a mock Request object."""
    req = MagicMock()
    req.headers = _DictLike(headers or {})
    req.cookies = _DictLike(cookies or {})
    req.path_params = _DictLike(path_params or {})
    req.query_params = _DictLike(query_params or {})
    req.url = MagicMock()
    req.url.path = "/test"
    req.client = MagicMock()
    req.client.host = "127.0.0.1"
    return req


# ---------------------------------------------------------------------------
# TestRequestContext
# ---------------------------------------------------------------------------


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_default_values(self):
        from app.deps.security import RequestContext

        ctx = RequestContext()
        assert ctx.user_id is None
        assert ctx.workspace_id is None
        assert ctx.role is None
        assert ctx.is_admin is False
        assert ctx.is_admin_token is False

    def test_admin_context(self):
        from app.deps.security import RequestContext

        ctx = RequestContext(is_admin=True, workspace_id=uuid4())
        assert ctx.is_admin is True
        assert ctx.user_id is None

    def test_admin_token_flag(self):
        from app.deps.security import RequestContext

        ctx = RequestContext(is_admin=True, is_admin_token=True)
        assert ctx.is_admin_token is True


# ---------------------------------------------------------------------------
# TestGetCurrentUserJWT
# ---------------------------------------------------------------------------


class TestGetCurrentUserJWT:
    """Tests for get_current_user_v2 with local JWT decode."""

    @pytest.mark.asyncio
    async def test_admin_token_bypass(self):
        """Admin token via X-Admin-Token header returns admin context."""
        from app.deps.security import get_current_user_v2

        req = _make_request(headers={"X-Admin-Token": "valid-token"})
        with patch("app.deps.security.verify_admin_token", return_value=True):
            ctx = await get_current_user_v2(req)

        assert ctx.is_admin is True
        assert ctx.is_admin_token is True
        assert ctx.user_id is None

    @pytest.mark.asyncio
    async def test_invalid_admin_token_raises_401(self):
        """Invalid X-Admin-Token raises 401."""
        from app.deps.security import get_current_user_v2

        req = _make_request(headers={"X-Admin-Token": "bad-token"})
        with patch("app.deps.security.verify_admin_token", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_auth_raises_401(self):
        """No auth headers at all raises 401."""
        from app.deps.security import get_current_user_v2

        req = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_v2(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_bearer_format_raises_401(self):
        """Non-Bearer auth raises 401."""
        from app.deps.security import get_current_user_v2

        req = _make_request(headers={"Authorization": "Basic abc123"})
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_v2(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_token_as_bearer(self):
        """Admin token sent via Bearer header returns admin context."""
        from app.deps.security import get_current_user_v2

        req = _make_request(headers={"Authorization": "Bearer my-admin-token"})
        with patch("app.deps.security.verify_admin_token", return_value=True):
            ctx = await get_current_user_v2(req)

        assert ctx.is_admin is True
        assert ctx.is_admin_token is True

    @pytest.mark.asyncio
    async def test_admin_token_cookie_fallback(self):
        """Admin token via cookie works when no Authorization header."""
        from app.deps.security import get_current_user_v2

        req = _make_request(cookies={"admin_token": "cookie-token"})
        with patch("app.deps.security.verify_admin_token", return_value=True):
            ctx = await get_current_user_v2(req)

        assert ctx.is_admin is True
        assert ctx.is_admin_token is True

    @pytest.mark.asyncio
    async def test_valid_jwt_returns_context(self):
        """Valid JWT with correct claims returns user context."""
        from app.deps.security import get_current_user_v2

        user_id = uuid4()
        token = _make_token(sub=str(user_id))
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings()
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            ctx = await get_current_user_v2(req)

        assert ctx.user_id == user_id
        assert ctx.is_admin is False
        assert ctx.is_admin_token is False

    @pytest.mark.asyncio
    async def test_expired_jwt_raises_401(self):
        """Expired JWT raises 401."""
        from app.deps.security import get_current_user_v2

        token = _make_token(sub=str(uuid4()), exp_delta=-3600)
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings()
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_bad_signature_raises_401(self):
        """JWT signed with wrong secret raises 401."""
        from app.deps.security import get_current_user_v2

        token = _make_token(sub=str(uuid4()), secret="wrong-secret")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings()
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_issuer_raises_401(self):
        """JWT with wrong issuer raises 401."""
        from app.deps.security import get_current_user_v2

        token = _make_token(sub=str(uuid4()), iss="https://evil.example.com")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings()
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_audience_raises_401(self):
        """JWT with wrong audience raises 401."""
        from app.deps.security import get_current_user_v2

        token = _make_token(sub=str(uuid4()), aud="wrong-audience")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings()
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_sub_raises_401(self):
        """JWT without sub claim raises 401."""
        from app.deps.security import get_current_user_v2

        token = _make_token(sub=None)  # No sub
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings()
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_bad_uuid_in_sub_raises_401(self):
        """JWT with non-UUID sub raises 401."""
        from app.deps.security import get_current_user_v2

        token = _make_token(sub="not-a-uuid")
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings()
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 401
        assert "Invalid user ID" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_no_jwt_secret_raises_500(self):
        """Missing SUPABASE_JWT_SECRET raises 500."""
        from app.deps.security import get_current_user_v2

        token = _make_token(sub=str(uuid4()))
        req = _make_request(headers={"Authorization": f"Bearer {token}"})

        settings = _make_settings(supabase_jwt_secret=None)
        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_v2(req)
        assert exc_info.value.status_code == 500
        assert "JWT_SECRET" in exc_info.value.detail


# ---------------------------------------------------------------------------
# TestRequireAuth
# ---------------------------------------------------------------------------


class TestRequireAuth:
    """Tests for the require_auth() composable dependency."""

    @pytest.mark.asyncio
    async def test_valid_jwt_plus_membership(self):
        """Valid JWT + workspace membership returns context with role."""
        from app.deps.security import require_auth

        user_id = uuid4()
        ws_id = uuid4()
        token = _make_token(sub=str(user_id))

        req = _make_request(
            headers={
                "Authorization": f"Bearer {token}",
                "X-Workspace-Id": str(ws_id),
            },
        )

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"role": "member"}
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        settings = _make_settings()
        dep = require_auth("member")

        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
            patch("app.core.lifespan.get_db_pool", return_value=mock_pool),
        ):
            ctx = await dep(req)

        assert ctx.user_id == user_id
        assert ctx.workspace_id == ws_id
        assert ctx.role == "member"

    @pytest.mark.asyncio
    async def test_admin_bypass(self):
        """Admin token bypasses workspace RBAC."""
        from app.deps.security import require_auth

        ws_id = uuid4()
        req = _make_request(
            headers={
                "X-Admin-Token": "admin-token",
                "X-Workspace-Id": str(ws_id),
            },
        )

        dep = require_auth("admin")

        with patch("app.deps.security.verify_admin_token", return_value=True):
            ctx = await dep(req)

        assert ctx.is_admin is True
        assert ctx.workspace_id == ws_id

    @pytest.mark.asyncio
    async def test_no_auth_raises_401(self):
        """No auth headers raises 401."""
        from app.deps.security import require_auth

        req = _make_request()
        dep = require_auth("member")

        with pytest.raises(HTTPException) as exc_info:
            await dep(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_workspace_raises_400(self):
        """Valid JWT but no workspace context raises 400."""
        from app.deps.security import require_auth

        user_id = uuid4()
        token = _make_token(sub=str(user_id))
        req = _make_request(
            headers={"Authorization": f"Bearer {token}"},
        )

        settings = _make_settings()
        dep = require_auth("member")

        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await dep(req)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_insufficient_role_raises_403(self):
        """Valid JWT + membership but insufficient role raises 403."""
        from app.deps.security import require_auth

        user_id = uuid4()
        ws_id = uuid4()
        token = _make_token(sub=str(user_id))

        req = _make_request(
            headers={
                "Authorization": f"Bearer {token}",
                "X-Workspace-Id": str(ws_id),
            },
        )

        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {"role": "member"}
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        settings = _make_settings()
        dep = require_auth("admin")  # requires admin

        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
            patch("app.core.lifespan.get_db_pool", return_value=mock_pool),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await dep(req)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_database_unavailable_raises_503(self):
        """No database pool raises 503."""
        from app.deps.security import require_auth

        user_id = uuid4()
        ws_id = uuid4()
        token = _make_token(sub=str(user_id))

        req = _make_request(
            headers={
                "Authorization": f"Bearer {token}",
                "X-Workspace-Id": str(ws_id),
            },
        )

        settings = _make_settings()
        dep = require_auth("member")

        with (
            patch("app.deps.security.verify_admin_token", return_value=False),
            patch("app.config.get_settings", return_value=settings),
            patch("app.core.lifespan.get_db_pool", return_value=None),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await dep(req)
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# TestRequireWorkspaceAccess
# ---------------------------------------------------------------------------


class TestRequireWorkspaceAccess:
    """Tests for require_workspace_access_v2 dependency."""

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


# ---------------------------------------------------------------------------
# TestRouterAuthCoverage (smoke test)
# ---------------------------------------------------------------------------

# Public paths that should NOT have auth dependencies
PUBLIC_PATHS = {
    "/health",
    "/ready",
    "/metrics",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}

# Prefixes for admin/debug endpoints that use their own auth pattern
ADMIN_PREFIXES = ("/admin/", "/debug/")

# Prefixes for static / SPA routes
STATIC_PREFIXES = ("/dashboard", "/static", "/")


class TestRouterAuthCoverage:
    """Verify all non-public routes have an auth dependency."""

    def test_all_workspace_routes_have_auth(self):
        """Every non-public, non-admin route should have require_auth dependency."""
        # Import the router (not the full app to avoid startup side effects)
        from app.api.router import api_router

        unprotected = []
        for route in api_router.routes:
            path = getattr(route, "path", "")

            # Skip public paths
            if path in PUBLIC_PATHS:
                continue

            # Skip admin paths (they use require_admin_token)
            if any(path.startswith(p) for p in ADMIN_PREFIXES):
                continue

            # Check route dependencies for require_auth
            deps = getattr(route, "dependencies", []) or []
            # Also check parent router dependencies (already merged by FastAPI)
            dependables = [getattr(d, "dependency", None) for d in deps]

            # require_auth returns a closure, so we check if any dependency
            # was created by require_auth (its __qualname__ contains
            # "require_auth.<locals>.dependency")
            has_auth = any(
                getattr(fn, "__qualname__", "").startswith("require_auth")
                for fn in dependables
                if fn is not None
            )

            if not has_auth:
                unprotected.append(path)

        # If there are unprotected routes, report them for debugging
        assert unprotected == [], (
            f"Unprotected workspace routes found: {unprotected}. "
            "Add dependencies=[Depends(require_auth(...))] to the router."
        )
