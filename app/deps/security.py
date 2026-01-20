"""Security dependencies for FastAPI routes.

Provides:
- Admin token authentication (constant-time compare)
- Workspace authorization (multi-tenant stub)
- Rate limiting (in-process with Redis upgrade path)
- Concurrency semaphores (per-workspace CPU protection)
"""

import asyncio
import hmac
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import structlog
from fastapi import Depends, HTTPException, Request, status

logger = structlog.get_logger(__name__)


# =============================================================================
# Admin Token Authentication
# =============================================================================


def require_admin_token(request: Request) -> bool:
    """
    Require valid admin token for protected routes.

    Security guarantees:
    - Uses hmac.compare_digest() for constant-time comparison
    - NO debug bypass (LOG_LEVEL has no effect)
    - NO localhost bypass in production
    - Returns 401 for missing token, 403 for invalid token

    Usage:
        @router.post("/admin/kb/ingest")
        async def ingest(..., _: bool = Depends(require_admin_token)):
            ...
    """
    admin_token = os.environ.get("ADMIN_TOKEN")

    if not admin_token:
        # In development without token, allow localhost only
        # In production, ADMIN_TOKEN should always be set
        allow_localhost = (
            os.environ.get("ALLOW_LOCALHOST_ADMIN", "false").lower() == "true"
        )
        if allow_localhost:
            host = request.headers.get("host", "")
            if "localhost" in host or "127.0.0.1" in host:
                logger.warning(
                    "Admin access via localhost (no token)",
                    path=request.url.path,
                    client=request.client.host if request.client else "unknown",
                )
                return True

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="ADMIN_TOKEN not configured. Contact system administrator.",
        )

    # Get token from header (preferred) or query param (fallback)
    provided_token = request.headers.get("X-Admin-Token")
    if not provided_token:
        provided_token = request.query_params.get("token")

    if not provided_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin token required. Provide X-Admin-Token header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(provided_token.encode(), admin_token.encode()):
        logger.warning(
            "Invalid admin token attempt",
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token",
        )

    return True


# =============================================================================
# Workspace Authorization (Multi-tenant stub)
# =============================================================================


@dataclass
class CurrentUser:
    """Current authenticated user (stub for future auth integration)."""

    user_id: Optional[str] = None
    workspace_ids: list[UUID] = field(default_factory=list)
    is_admin: bool = False


def get_current_user(request: Request) -> CurrentUser:
    """
    Get current user from request (stub).

    In production, this would:
    - Validate JWT/session token
    - Return user with authorized workspace_ids
    - Check admin status

    For now: returns allow-all user for single-tenant mode.

    Auth Integration Path:
    1. Add Authorization header parsing (Bearer token)
    2. Validate JWT via Supabase Auth API (use get_current_user_v2 as reference)
    3. Query workspace_members table for user's workspace access
    4. Return CurrentUser with populated workspace_ids list
    5. Update require_workspace_access to enforce multi-tenant checks
    6. Add integration tests for auth flow

    Note: get_current_user_v2() (lines 205-251) provides the v2 implementation
    with JWT validation. Migrate callers from v1 (this function) to v2 when
    ready to enable multi-tenant authentication.
    """
    # Single-tenant mode: allow-all user
    return CurrentUser(
        user_id="single-tenant-user",
        workspace_ids=[],  # Empty = allow all (single-tenant mode)
        is_admin=False,
    )


def require_workspace_access(
    workspace_id: UUID,
    user: CurrentUser = Depends(get_current_user),
) -> bool:
    """
    Verify user has access to the requested workspace.

    In single-tenant mode: always allows access.
    In multi-tenant mode: checks workspace_ids list.

    Usage:
        @router.post("/kb/trials/recommend")
        async def recommend(
            req: RecommendRequest,
            _: bool = Depends(lambda: require_workspace_access(req.workspace_id)),
        ):
            ...
    """
    # Single-tenant mode: empty list = allow all
    if not user.workspace_ids:
        return True

    # Multi-tenant mode: check membership
    if user.is_admin:
        return True

    if workspace_id not in user.workspace_ids:
        logger.warning(
            "Workspace access denied",
            user_id=user.user_id,
            requested_workspace=str(workspace_id),
            allowed_workspaces=[str(w) for w in user.workspace_ids],
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to workspace {workspace_id}",
        )

    return True


# =============================================================================
# JWT-based Authentication (v2)
# =============================================================================


@dataclass
class RequestContext:
    """Auth context resolved from request.

    This is the new v2 auth context that supports JWT-based authentication
    with Supabase Auth. Unlike CurrentUser (v1 stub), this resolves actual
    user identity from tokens.
    """

    user_id: Optional[UUID] = None
    workspace_id: Optional[UUID] = None
    role: Optional[str] = None
    is_admin: bool = False


ROLE_RANK = {"viewer": 1, "member": 2, "admin": 3, "owner": 4}


def verify_admin_token(token: str) -> bool:
    """
    Verify admin token using constant-time comparison.

    Returns True if token is valid, False otherwise.
    """
    admin_token = os.environ.get("ADMIN_TOKEN")
    if not admin_token:
        return False
    return hmac.compare_digest(token.encode(), admin_token.encode())


async def get_current_user_v2(
    authorization: Optional[str] = None,
    x_admin_token: Optional[str] = None,
) -> RequestContext:
    """
    Resolve user identity from JWT or admin token.

    Does NOT resolve workspace - that's separate.

    Args:
        authorization: Bearer token from Authorization header
        x_admin_token: Admin token from X-Admin-Token header

    Returns:
        RequestContext with resolved user identity

    Raises:
        HTTPException 401: Missing or invalid authentication
    """
    # (1) Admin token bypass
    if x_admin_token:
        if verify_admin_token(x_admin_token):
            return RequestContext(is_admin=True)
        raise HTTPException(status_code=401, detail="Invalid admin token")

    # (2) Require Authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = authorization.split(" ", 1)[1]

    # (3) Validate via Supabase Auth API
    try:
        from app.deps.supabase import get_supabase_client

        supabase = get_supabase_client()
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return RequestContext(user_id=UUID(user_response.user.id))
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Auth validation failed", error=str(e))
        raise HTTPException(status_code=401, detail="Authentication failed")


async def require_workspace_access_v2(
    ctx: RequestContext,
    workspace_id: UUID,
    min_role: str = "viewer",
    pool=None,
) -> RequestContext:
    """
    Verify user has access to workspace with minimum role.

    Admin bypass still requires explicit workspace_id.

    Args:
        ctx: RequestContext from get_current_user_v2
        workspace_id: Workspace to check access for
        min_role: Minimum role required (viewer, member, admin, owner)
        pool: Database connection pool

    Returns:
        RequestContext with workspace_id and role populated

    Raises:
        HTTPException 401: User ID required but missing
        HTTPException 403: Not a member or insufficient role
    """
    # Admin bypass - still needs workspace_id for scoping
    if ctx.is_admin:
        return RequestContext(is_admin=True, workspace_id=workspace_id)

    if not ctx.user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    # Look up membership
    query = """
        SELECT role FROM workspace_members
        WHERE workspace_id = $1 AND user_id = $2
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, workspace_id, ctx.user_id)

    if not row:
        raise HTTPException(status_code=403, detail="Not a member of this workspace")

    role = row["role"]
    if ROLE_RANK.get(role, 0) < ROLE_RANK.get(min_role, 0):
        raise HTTPException(
            status_code=403, detail=f"Requires {min_role} role, you have {role}"
        )

    return RequestContext(
        user_id=ctx.user_id,
        workspace_id=workspace_id,
        role=role,
    )


# =============================================================================
# Rate Limiting (In-Process with Redis upgrade path)
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int
    burst_allowance: int = 0  # Extra requests allowed in burst


class RateLimiter:
    """
    In-process rate limiter using sliding window.

    Note: This is per-process only. For multi-replica deployments,
    replace with Redis-backed implementation (Upstash recommended).

    Usage:
        limiter = RateLimiter()

        @router.post("/upload")
        async def upload(
            request: Request,
            _: None = Depends(limiter.check("upload", 5)),  # 5/min
        ):
            ...
    """

    def __init__(self):
        # key -> list of request timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def _cleanup_old(self, key: str, window_seconds: float = 60.0):
        """Remove requests older than window."""
        cutoff = time.time() - window_seconds
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

    def check(
        self,
        limit_name: str,
        requests_per_minute: int,
        key_func: Optional[callable] = None,
    ):
        """
        Create a dependency that checks rate limit.

        Args:
            limit_name: Name for this limit (for logging)
            requests_per_minute: Max requests allowed per minute
            key_func: Function(request) -> str for rate limit key.
                      Default: client IP address.
        """

        async def rate_limit_dependency(request: Request):
            # Build rate limit key
            if key_func:
                key = f"{limit_name}:{key_func(request)}"
            else:
                # Default: by client IP
                client_ip = request.client.host if request.client else "unknown"
                key = f"{limit_name}:ip:{client_ip}"

            async with self._lock:
                await self._cleanup_old(key)

                if len(self._requests[key]) >= requests_per_minute:
                    # Calculate retry-after
                    oldest = (
                        min(self._requests[key]) if self._requests[key] else time.time()
                    )
                    retry_after = int(60 - (time.time() - oldest)) + 1

                    logger.warning(
                        "Rate limit exceeded",
                        limit_name=limit_name,
                        key=key,
                        requests=len(self._requests[key]),
                        limit=requests_per_minute,
                    )

                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Rate limit exceeded: {requests_per_minute}/min for {limit_name}",
                        headers={"Retry-After": str(retry_after)},
                    )

                self._requests[key].append(time.time())

        return rate_limit_dependency


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# =============================================================================
# Concurrency Semaphores (Per-workspace CPU protection)
# =============================================================================


class WorkspaceSemaphore:
    """
    Per-workspace concurrency limiter.

    Prevents a single workspace from starving others during
    CPU-intensive operations (embedding, recommendation).

    Usage:
        semaphore = WorkspaceSemaphore(max_concurrent=2)

        @router.post("/recommend")
        async def recommend(req: RecommendRequest):
            async with semaphore.acquire(req.workspace_id):
                # CPU-intensive work
                ...
    """

    def __init__(self, max_concurrent: int = 2):
        self.max_concurrent = max_concurrent
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def _get_semaphore(self, workspace_id: UUID) -> asyncio.Semaphore:
        """Get or create semaphore for workspace."""
        key = str(workspace_id)
        async with self._lock:
            if key not in self._semaphores:
                self._semaphores[key] = asyncio.Semaphore(self.max_concurrent)
            return self._semaphores[key]

    class _SemaphoreContext:
        """Context manager for semaphore acquisition with timeout."""

        def __init__(
            self, semaphore: asyncio.Semaphore, workspace_id: UUID, timeout: float
        ):
            self.semaphore = semaphore
            self.workspace_id = workspace_id
            self.timeout = timeout

        async def __aenter__(self):
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Workspace concurrency limit reached",
                    workspace_id=str(self.workspace_id),
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many concurrent requests for this workspace. Please wait.",
                    headers={"Retry-After": "5"},
                )
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.semaphore.release()

    def acquire(self, workspace_id: UUID, timeout: float = 30.0):
        """
        Acquire semaphore for workspace.

        Args:
            workspace_id: Workspace to limit
            timeout: Max seconds to wait for semaphore

        Returns:
            Context manager for the semaphore
        """

        async def get_context():
            semaphore = await self._get_semaphore(workspace_id)
            return self._SemaphoreContext(semaphore, workspace_id, timeout)

        # Return a coroutine that creates the context manager
        return _AsyncContextManagerWrapper(get_context)


class _AsyncContextManagerWrapper:
    """Wrapper to make async context manager creation awaitable."""

    def __init__(self, factory):
        self._factory = factory
        self._context = None

    async def __aenter__(self):
        self._context = await self._factory()
        return await self._context.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            return await self._context.__aexit__(exc_type, exc_val, exc_tb)


# Global workspace semaphore instance
_workspace_semaphore: Optional[WorkspaceSemaphore] = None


def get_workspace_semaphore(max_concurrent: int = 2) -> WorkspaceSemaphore:
    """Get global workspace semaphore instance."""
    global _workspace_semaphore
    if _workspace_semaphore is None:
        _workspace_semaphore = WorkspaceSemaphore(max_concurrent=max_concurrent)
    return _workspace_semaphore
