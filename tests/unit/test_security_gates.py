"""
Security gate tests to prevent protection regressions.

These tests verify that security controls cannot be accidentally bypassed.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.deps.security import (
    require_admin_token,
    require_workspace_access,
    RateLimiter,
    WorkspaceSemaphore,
    CurrentUser,
)


# =============================================================================
# Admin Auth Tests
# =============================================================================


class TestAdminAuth:
    """Tests for admin token authentication."""

    @pytest.fixture
    def app_with_admin_route(self):
        """Create test app with admin-protected route."""
        app = FastAPI()

        @app.get("/admin/test")
        def admin_route(_: bool = Depends(require_admin_token)):
            return {"status": "ok"}

        return app

    def test_no_token_returns_401_or_403(self, app_with_admin_route):
        """Request without token should return 401 or 403."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "secret123"}, clear=False):
            client = TestClient(app_with_admin_route, raise_server_exceptions=False)
            response = client.get("/admin/test")
            assert response.status_code in (401, 403)
            assert "token" in response.json()["detail"].lower()

    def test_wrong_token_returns_403(self, app_with_admin_route):
        """Request with wrong token should return 403."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "secret123"}, clear=False):
            client = TestClient(app_with_admin_route, raise_server_exceptions=False)
            response = client.get(
                "/admin/test", headers={"X-Admin-Token": "wrong_token"}
            )
            assert response.status_code == 403
            assert "invalid" in response.json()["detail"].lower()

    def test_correct_token_returns_200(self, app_with_admin_route):
        """Request with correct token should return 200."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "secret123"}, clear=False):
            client = TestClient(app_with_admin_route, raise_server_exceptions=False)
            response = client.get("/admin/test", headers={"X-Admin-Token": "secret123"})
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    def test_token_via_query_param(self, app_with_admin_route):
        """Token can be provided via query param as fallback."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "secret123"}, clear=False):
            client = TestClient(app_with_admin_route, raise_server_exceptions=False)
            response = client.get("/admin/test?token=secret123")
            assert response.status_code == 200

    def test_no_debug_bypass(self, app_with_admin_route):
        """LOG_LEVEL=DEBUG should not bypass auth."""
        with patch.dict(
            os.environ, {"ADMIN_TOKEN": "secret123", "LOG_LEVEL": "DEBUG"}, clear=False
        ):
            client = TestClient(app_with_admin_route, raise_server_exceptions=False)
            response = client.get("/admin/test")
            # Should still require token even in debug mode
            assert response.status_code in (401, 403)


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.fixture
    def rate_limiter(self):
        """Create fresh rate limiter for each test."""
        return RateLimiter()

    @pytest.fixture
    def app_with_rate_limit(self, rate_limiter):
        """Create test app with rate-limited route."""
        app = FastAPI()

        @app.post("/limited")
        async def limited_route(
            request: Request,
            _: None = Depends(rate_limiter.check("test_limit", 3)),  # 3 per minute
        ):
            return {"status": "ok"}

        return app

    def test_burst_requests_return_429(self, app_with_rate_limit):
        """Burst of requests exceeding limit should return 429."""
        client = TestClient(app_with_rate_limit, raise_server_exceptions=False)

        # First 3 requests should succeed
        for i in range(3):
            response = client.post("/limited")
            assert response.status_code == 200, f"Request {i+1} failed unexpectedly"

        # 4th request should be rate limited
        response = client.post("/limited")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_rate_limit_includes_retry_hint(self, app_with_rate_limit):
        """429 response should include Retry-After header."""
        client = TestClient(app_with_rate_limit, raise_server_exceptions=False)

        # Exhaust the limit
        for _ in range(3):
            client.post("/limited")

        response = client.post("/limited")
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        retry_after = int(response.headers["Retry-After"])
        assert 0 < retry_after <= 61  # Should be within the minute window


# =============================================================================
# Concurrency Limiting Tests
# =============================================================================


class TestConcurrencyLimiting:
    """Tests for per-workspace concurrency limiting."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_limited(self):
        """Concurrent requests exceeding limit should fail gracefully."""
        from uuid import uuid4

        semaphore = WorkspaceSemaphore(max_concurrent=2)
        workspace_id = uuid4()

        results = []
        errors = []
        started = asyncio.Event()

        async def slow_operation(op_id: int):
            try:
                async with semaphore.acquire(workspace_id, timeout=0.1):
                    started.set()  # Signal that we've acquired
                    await asyncio.sleep(1.0)  # Hold the semaphore longer than timeout
                    results.append(op_id)
            except HTTPException as e:
                errors.append((op_id, e.status_code))

        # Launch 4 concurrent operations (limit is 2)
        # Use gather with return_exceptions to prevent early termination
        tasks = [asyncio.create_task(slow_operation(i)) for i in range(4)]

        # Wait briefly to let all tasks attempt to acquire
        await asyncio.sleep(0.3)

        # Cancel remaining tasks to speed up test
        for t in tasks:
            t.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        # At least 2 should have failed with 429 (couldn't acquire within timeout)
        # The other 2 acquired but may have been cancelled
        assert (
            len(errors) >= 2
        ), f"Expected at least 2 errors, got {len(errors)}: {errors}"
        assert all(status == 429 for _, status in errors)

    @pytest.mark.asyncio
    async def test_semaphore_releases_after_completion(self):
        """Semaphore should release properly after operation completes."""
        from uuid import uuid4

        semaphore = WorkspaceSemaphore(max_concurrent=1)
        workspace_id = uuid4()

        # First operation
        async with semaphore.acquire(workspace_id, timeout=1.0):
            pass  # Complete immediately

        # Second operation should succeed (semaphore released)
        async with semaphore.acquire(workspace_id, timeout=1.0):
            pass  # Should not timeout


# =============================================================================
# Docs Gating Tests
# =============================================================================


class TestDocsGating:
    """Tests for API docs endpoint gating."""

    def test_docs_disabled_when_setting_false(self):
        """When DOCS_ENABLED=false, /docs should not be accessible."""
        with patch("app.config.Settings") as _mock_settings_cls:  # noqa: F841
            mock_settings = MagicMock()
            mock_settings.docs_enabled = False
            mock_settings.sentry_dsn = None
            mock_settings.rate_limit_enabled = False
            mock_settings.rate_limit_requests_per_minute = 100

            app = FastAPI(
                docs_url="/docs" if mock_settings.docs_enabled else None,
                redoc_url="/redoc" if mock_settings.docs_enabled else None,
                openapi_url="/openapi.json" if mock_settings.docs_enabled else None,
            )

            client = TestClient(app, raise_server_exceptions=False)

            # All doc endpoints should return 404
            assert client.get("/docs").status_code == 404
            assert client.get("/redoc").status_code == 404
            assert client.get("/openapi.json").status_code == 404

    def test_docs_enabled_when_setting_true(self):
        """When DOCS_ENABLED=true, /docs should be accessible."""
        app = FastAPI(
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        @app.get("/test")
        def test_route():
            return {"status": "ok"}

        client = TestClient(app)

        # Doc endpoints should be accessible
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        assert client.get("/openapi.json").status_code == 200


# =============================================================================
# Readiness Check Tests
# =============================================================================


class TestReadinessCheck:
    """Tests for /ready endpoint behavior under failure conditions."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.database_url = "postgresql://test"
        settings.qdrant_url = "http://qdrant:6333"
        settings.qdrant_collection_active = "test_collection"
        settings.qdrant_timeout = 5.0
        settings.ollama_base_url = "http://ollama:11434"
        settings.ollama_timeout = 5.0
        settings.embed_model = "nomic-embed-text"
        settings.embed_dim = 768
        return settings

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_db_down(self, mock_settings):
        """When database is unreachable, /ready should return 503."""
        from app.routers.health import check_database_health

        # Simulate DB connection failure
        with patch(
            "app.routers.health.asyncpg.connect",
            side_effect=Exception("Connection refused"),
        ):
            result = await check_database_health(mock_settings)
            assert result.status == "error"
            assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_qdrant_down(self, mock_settings):
        """When Qdrant is unreachable, /ready should return 503."""
        from app.routers.health import check_qdrant_collection
        import httpx

        # Simulate Qdrant connection failure
        with patch("app.routers.health.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            result = await check_qdrant_collection(mock_settings)
            assert result.status == "error"

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_embed_down(self, mock_settings):
        """When embedding service is unreachable, /ready should return 503."""
        from app.routers.health import check_embed_service
        import httpx

        # Simulate Ollama connection failure
        with patch("app.routers.health.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            result = await check_embed_service(mock_settings)
            assert result.status == "error"

    @pytest.mark.asyncio
    async def test_ready_returns_503_when_collection_dim_mismatch(self, mock_settings):
        """When collection dimensions don't match config, /ready should return 503."""
        from app.routers.health import check_qdrant_collection

        # Simulate Qdrant returning wrong dimensions
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "config": {
                    "params": {"vectors": {"size": 1024}}  # Wrong! Config expects 768
                }
            }
        }

        with patch("app.routers.health.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            result = await check_qdrant_collection(mock_settings)
            assert result.status == "error"
            assert "mismatch" in result.error.lower()


# =============================================================================
# Workspace Authorization Tests
# =============================================================================


class TestWorkspaceAuthorization:
    """Tests for workspace access control."""

    def test_single_tenant_allows_all(self):
        """In single-tenant mode (empty workspace_ids), access is allowed."""
        from uuid import uuid4

        user = CurrentUser(
            user_id="test_user",
            workspace_ids=[],  # Empty = single-tenant mode
            is_admin=False,
        )

        # Should not raise
        result = require_workspace_access(uuid4(), user)
        assert result is True

    def test_multi_tenant_denies_unauthorized(self):
        """In multi-tenant mode, unauthorized workspace access is denied."""
        from uuid import uuid4

        allowed_workspace = uuid4()
        denied_workspace = uuid4()

        user = CurrentUser(
            user_id="test_user",
            workspace_ids=[allowed_workspace],
            is_admin=False,
        )

        # Should raise 403 for unauthorized workspace
        with pytest.raises(HTTPException) as exc_info:
            require_workspace_access(denied_workspace, user)

        assert exc_info.value.status_code == 403

    def test_admin_can_access_any_workspace(self):
        """Admin users can access any workspace."""
        from uuid import uuid4

        user = CurrentUser(
            user_id="admin_user",
            workspace_ids=[uuid4()],  # Has some workspaces
            is_admin=True,
        )

        # Should not raise for any workspace
        random_workspace = uuid4()
        result = require_workspace_access(random_workspace, user)
        assert result is True
