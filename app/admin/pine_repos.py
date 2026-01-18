"""Admin endpoints for GitHub repository management.

Provides API access to register, list, and scan GitHub repositories
for Pine script discovery.
"""

import shutil
import time
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.deps.security import require_admin_token
from app.services.pine.adapters.git import (
    InvalidRepoUrlError,
    extract_slug_from_url,
)

router = APIRouter(prefix="/pine/repos", tags=["admin-pine-repos"])
logger = structlog.get_logger(__name__)

# Global connection pool and clients (set during app startup)
_db_pool = None
_qdrant_client = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def set_qdrant_client(client):
    """Set the Qdrant client for this router."""
    global _qdrant_client
    _qdrant_client = client


def _get_pool():
    """Get the database pool."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return _db_pool


# ===========================================
# Request/Response Models
# ===========================================


class RegisterRepoRequest(BaseModel):
    """Request payload for registering a GitHub repository."""

    workspace_id: UUID = Field(
        ..., description="Workspace to associate the repository with"
    )
    repo_url: str = Field(
        ...,
        description="Full GitHub URL (https://github.com/owner/repo)",
        examples=["https://github.com/owner/pine-scripts"],
    )
    branch: str = Field(
        default="main",
        description="Git branch to track",
    )
    scan_globs: list[str] = Field(
        default=["**/*.pine"],
        description="Glob patterns for matching Pine files",
    )
    run_now: bool = Field(
        default=True,
        description="Clone and scan immediately after registration",
    )

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v):
        """Validate GitHub URL format."""
        try:
            extract_slug_from_url(v)
        except InvalidRepoUrlError as e:
            raise ValueError(str(e))
        return v


class RepoResponse(BaseModel):
    """Response model for a registered repository."""

    id: UUID
    workspace_id: UUID
    repo_slug: str
    repo_url: str
    branch: str
    clone_path: Optional[str] = None
    enabled: bool
    scripts_count: int
    last_scan_at: Optional[str] = None
    last_scan_ok: Optional[bool] = None
    last_scan_error: Optional[str] = None
    last_seen_commit: Optional[str] = None
    scan_globs: list[str]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RepoListResponse(BaseModel):
    """Response model for listing repositories."""

    items: list[RepoResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class DeleteRepoRequest(BaseModel):
    """Request payload for deleting a repository."""

    delete_clone: bool = Field(
        default=False,
        description="Also delete the local clone directory",
    )


class ScanRepoRequest(BaseModel):
    """Request payload for triggering a repository scan."""

    force_full_scan: bool = Field(
        default=False,
        description="Force a full scan even if we have a previous commit",
    )


class ScanRepoResponse(BaseModel):
    """Response from a repository scan."""

    status: Literal["success", "partial", "error"]
    scan_run_id: str
    scripts_scanned: int
    scripts_new: int
    scripts_updated: int
    scripts_deleted: int
    scripts_unchanged: int
    specs_generated: int
    scripts_ingested: int
    scripts_ingest_failed: int
    chunks_created: int
    commit_before: Optional[str] = None
    commit_after: Optional[str] = None
    is_full_scan: bool
    errors: list[str]
    duration_ms: int


# ===========================================
# Helper Functions
# ===========================================


def _repo_to_response(repo) -> RepoResponse:
    """Convert PineRepo model to response."""
    return RepoResponse(
        id=repo.id,
        workspace_id=repo.workspace_id,
        repo_slug=repo.repo_slug,
        repo_url=repo.repo_url,
        branch=repo.branch,
        clone_path=repo.clone_path,
        enabled=repo.enabled,
        scripts_count=repo.scripts_count,
        last_scan_at=repo.last_scan_at.isoformat() if repo.last_scan_at else None,
        last_scan_ok=repo.last_scan_ok,
        last_scan_error=repo.last_scan_error,
        last_seen_commit=repo.last_seen_commit,
        scan_globs=repo.scan_globs,
        created_at=repo.created_at.isoformat() if repo.created_at else None,
        updated_at=repo.updated_at.isoformat() if repo.updated_at else None,
    )


# ===========================================
# Endpoints
# ===========================================


@router.post(
    "/register",
    response_model=RepoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a GitHub repository",
    description="""
Register a new GitHub repository for Pine script discovery.

Validates the repository URL, extracts the owner/repo slug, and creates
a database entry. If run_now=true (default), also clones the repository
and performs an initial scan.

The repository must be publicly accessible. Private repository support
is planned for a future release.
""",
)
async def register_repo(
    request: RegisterRepoRequest,
    settings: Settings = Depends(get_settings),
    _: str = Depends(require_admin_token),
):
    """Register a new GitHub repository for discovery."""
    from app.services.pine.repo_registry import PineRepoRepository
    from app.services.pine.discovery import PineDiscoveryService

    pool = _get_pool()
    log = logger.bind(
        workspace_id=str(request.workspace_id),
        repo_url=request.repo_url,
    )

    try:
        # Extract and validate slug
        repo_slug = extract_slug_from_url(request.repo_url)
        log = log.bind(repo_slug=repo_slug)
        log.info("registering_repo")

        # Create repo registration
        repo_registry = PineRepoRepository(pool)

        try:
            repo = await repo_registry.create(
                workspace_id=request.workspace_id,
                repo_url=request.repo_url,
                repo_slug=repo_slug,
                branch=request.branch,
                scan_globs=request.scan_globs,
            )
        except Exception as e:
            error_str = str(e)
            if "unique" in error_str.lower() or "duplicate" in error_str.lower():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Repository already registered: {repo_slug}",
                )
            if "check" in error_str.lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid repository slug format: {repo_slug}",
                )
            raise

        log.info("repo_registered", repo_id=str(repo.id))

        # Run initial scan if requested
        if request.run_now:
            try:
                discovery = PineDiscoveryService(pool, settings, _qdrant_client)
                result = await discovery.discover_repo(
                    workspace_id=request.workspace_id,
                    repo_id=repo.id,
                    trigger="manual",
                    force_full_scan=True,
                )
                log.info(
                    "initial_scan_complete",
                    status=result.status,
                    scripts_new=result.scripts_new,
                )
                # Refresh repo to get updated stats
                refreshed = await repo_registry.get(repo.id)
                if refreshed:
                    repo = refreshed
            except Exception as e:
                log.warning("initial_scan_failed", error=str(e))
                # Don't fail registration if scan fails
                refreshed = await repo_registry.get(repo.id)
                if refreshed:
                    repo = refreshed

        return _repo_to_response(repo)

    except HTTPException:
        raise
    except InvalidRepoUrlError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        log.exception("register_repo_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register repository: {e}",
        )


@router.get(
    "",
    response_model=RepoListResponse,
    summary="List registered repositories",
    description="List all repositories registered for a workspace.",
)
async def list_repos(
    workspace_id: UUID = Query(..., description="Workspace ID to list repos for"),
    enabled_only: bool = Query(False, description="Only return enabled repos"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: str = Depends(require_admin_token),
):
    """List registered repositories."""
    from app.services.pine.repo_registry import PineRepoRepository

    pool = _get_pool()
    repo_registry = PineRepoRepository(pool)

    repos, total = await repo_registry.list_by_workspace(
        workspace_id=workspace_id,
        enabled_only=enabled_only,
        limit=limit,
        offset=offset,
    )

    return RepoListResponse(
        items=[_repo_to_response(r) for r in repos],
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(repos)) < total,
    )


@router.get(
    "/{repo_id}",
    response_model=RepoResponse,
    summary="Get repository details",
    description="Get details of a registered repository.",
)
async def get_repo(
    repo_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Get repository details."""
    from app.services.pine.repo_registry import PineRepoRepository

    pool = _get_pool()
    repo_registry = PineRepoRepository(pool)

    repo = await repo_registry.get(repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    return _repo_to_response(repo)


@router.delete(
    "/{repo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unregister a repository",
    description="""
Remove a repository registration. Optionally deletes the local clone.

Note: This does NOT delete the discovered scripts from strategy_scripts.
Scripts remain but become orphaned (repo_id becomes NULL on CASCADE).
""",
)
async def unregister_repo(
    repo_id: UUID,
    delete_clone: bool = Query(False, description="Also delete local clone"),
    settings: Settings = Depends(get_settings),
    _: str = Depends(require_admin_token),
):
    """Unregister a repository."""
    from app.services.pine.repo_registry import PineRepoRepository

    pool = _get_pool()
    log = logger.bind(repo_id=str(repo_id))

    repo_registry = PineRepoRepository(pool)
    repo = await repo_registry.get(repo_id)

    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    log = log.bind(repo_slug=repo.repo_slug)

    # Delete from DB
    deleted = await repo_registry.delete(repo_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    log.info("repo_unregistered")

    # Optionally delete clone
    if delete_clone and repo.clone_path:
        try:
            clone_path = Path(repo.clone_path)
            if clone_path.exists():
                shutil.rmtree(clone_path)
                log.info("clone_deleted", clone_path=repo.clone_path)
        except Exception as e:
            log.warning("clone_delete_failed", error=str(e))
            # Don't fail the request

    return None


@router.post(
    "/{repo_id}/scan",
    response_model=ScanRepoResponse,
    summary="Trigger repository scan",
    description="""
Trigger a discovery scan for a registered repository.

Fetches the latest commits from the remote, computes changes since the
last scan, and processes any new/modified/deleted Pine scripts.

If force_full_scan=true, ignores the last seen commit and rescans all
matching files.
""",
)
async def scan_repo(
    repo_id: UUID,
    request: ScanRepoRequest = ScanRepoRequest(),
    settings: Settings = Depends(get_settings),
    _: str = Depends(require_admin_token),
):
    """Trigger a repository scan."""
    from uuid import uuid4
    from app.services.pine.repo_registry import PineRepoRepository
    from app.services.pine.discovery import PineDiscoveryService

    pool = _get_pool()
    run_id = f"scan-{uuid4().hex[:8]}"
    log = logger.bind(repo_id=str(repo_id), scan_run_id=run_id)

    # Get repo to validate and get workspace_id
    repo_registry = PineRepoRepository(pool)
    repo = await repo_registry.get(repo_id)

    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    if not repo.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Repository is disabled",
        )

    log = log.bind(repo_slug=repo.repo_slug)
    log.info("scan_started", force_full_scan=request.force_full_scan)

    start_time = time.time()

    try:
        discovery = PineDiscoveryService(pool, settings, _qdrant_client)
        result = await discovery.discover_repo(
            workspace_id=repo.workspace_id,
            repo_id=repo_id,
            trigger="manual",
            force_full_scan=request.force_full_scan,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        log.info(
            "scan_complete",
            status=result.status,
            scripts_new=result.scripts_new,
            scripts_updated=result.scripts_updated,
            scripts_deleted=result.scripts_deleted,
            duration_ms=duration_ms,
        )

        return ScanRepoResponse(
            status=result.status,
            scan_run_id=run_id,
            scripts_scanned=result.scripts_scanned,
            scripts_new=result.scripts_new,
            scripts_updated=result.scripts_updated,
            scripts_deleted=result.scripts_deleted,
            scripts_unchanged=result.scripts_unchanged,
            specs_generated=result.specs_generated,
            scripts_ingested=result.scripts_ingested,
            scripts_ingest_failed=result.scripts_ingest_failed,
            chunks_created=result.chunks_created,
            commit_before=result.commit_before,
            commit_after=result.commit_after,
            is_full_scan=result.is_full_scan,
            errors=result.errors,
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        log.exception("scan_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {e}",
        )


@router.patch(
    "/{repo_id}/enable",
    response_model=RepoResponse,
    summary="Enable a repository",
    description="Enable a previously disabled repository for scanning.",
)
async def enable_repo(
    repo_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Enable a repository."""
    from app.services.pine.repo_registry import PineRepoRepository

    pool = _get_pool()
    repo_registry = PineRepoRepository(pool)

    repo = await repo_registry.get(repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    await repo_registry.set_enabled(repo_id, True)
    refreshed = await repo_registry.get(repo_id)
    # Repo should exist since we just checked it
    if refreshed:
        repo = refreshed

    logger.info("repo_enabled", repo_id=str(repo_id), repo_slug=repo.repo_slug)

    return _repo_to_response(repo)


@router.patch(
    "/{repo_id}/disable",
    response_model=RepoResponse,
    summary="Disable a repository",
    description="Disable a repository from being included in polling scans.",
)
async def disable_repo(
    repo_id: UUID,
    _: str = Depends(require_admin_token),
):
    """Disable a repository."""
    from app.services.pine.repo_registry import PineRepoRepository

    pool = _get_pool()
    repo_registry = PineRepoRepository(pool)

    repo = await repo_registry.get(repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    await repo_registry.set_enabled(repo_id, False)
    refreshed = await repo_registry.get(repo_id)
    # Repo should exist since we just checked it
    if refreshed:
        repo = refreshed

    logger.info("repo_disabled", repo_id=str(repo_id), repo_slug=repo.repo_slug)

    return _repo_to_response(repo)


# ===========================================
# Poll Management Endpoints
# ===========================================


class PollRunResponse(BaseModel):
    """Response from a manual poll run."""

    status: Literal["success", "partial", "error", "disabled"]
    repos_scanned: int
    repos_succeeded: int
    repos_failed: int
    repos_skipped: int
    errors: list[str]
    duration_ms: int


class PollerHealthResponse(BaseModel):
    """Poller health status."""

    enabled: bool
    running: bool
    last_run_at: Optional[str] = None
    last_run_repos_scanned: int = 0
    last_run_errors: int = 0
    repos_due_count: int = 0
    poll_interval_minutes: int
    poll_max_concurrency: int
    poll_tick_seconds: int


@router.post(
    "/poll-run",
    response_model=PollRunResponse,
    summary="Trigger manual poll run",
    description="""
Manually trigger a single poll cycle.

This scans repos that are due for polling (next_scan_at <= now or NULL)
up to the configured max_repos_per_tick limit.

Useful for:
- Testing polling setup without waiting for tick
- Forcing immediate scan after configuration changes
- Recovery from stuck state

Note: This does not affect the background poller schedule.
""",
)
async def trigger_poll_run(
    settings: Settings = Depends(get_settings),
    _: str = Depends(require_admin_token),
):
    """Trigger a manual poll run."""
    from app.services.pine.poller import get_poller, PineRepoPoller

    pool = _get_pool()
    log = logger.bind(endpoint="poll-run")

    # Check if poller exists and can run
    poller = get_poller()

    if poller is None:
        # Create a temporary poller for manual run
        log.info("Creating temporary poller for manual run")
        poller = PineRepoPoller(pool, settings, _qdrant_client)

    try:
        result = await poller.run_once()

        # Determine status
        status_val: Literal["success", "partial", "error", "disabled"]
        if result.repos_scanned == 0:
            status_val = "success"  # No repos due is still success
        elif result.repos_failed == 0:
            status_val = "success"
        elif result.repos_succeeded > 0:
            status_val = "partial"
        else:
            status_val = "error"

        log.info(
            "poll_run_complete",
            status=status_val,
            repos_scanned=result.repos_scanned,
            repos_succeeded=result.repos_succeeded,
            repos_failed=result.repos_failed,
            duration_ms=result.duration_ms,
        )

        return PollRunResponse(
            status=status_val,
            repos_scanned=result.repos_scanned,
            repos_succeeded=result.repos_succeeded,
            repos_failed=result.repos_failed,
            repos_skipped=result.repos_skipped,
            errors=result.errors[:10],  # Limit to first 10 errors
            duration_ms=result.duration_ms,
        )

    except Exception as e:
        log.exception("poll_run_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Poll run failed: {e}",
        )


@router.get(
    "/poll-status",
    response_model=PollerHealthResponse,
    summary="Get poller status",
    description="Get the current status of the background poller.",
)
async def get_poller_status(
    settings: Settings = Depends(get_settings),
    _: str = Depends(require_admin_token),
):
    """Get poller status."""
    from app.services.pine.poller import get_poller

    poller = get_poller()

    if poller is None:
        # Return disabled status if poller not running
        return PollerHealthResponse(
            enabled=settings.pine_repo_poll_enabled,
            running=False,
            repos_due_count=0,
            poll_interval_minutes=settings.pine_repo_poll_interval_minutes,
            poll_max_concurrency=settings.pine_repo_poll_max_concurrency,
            poll_tick_seconds=settings.pine_repo_poll_tick_seconds,
        )

    health = await poller.get_health()

    return PollerHealthResponse(
        enabled=health.enabled,
        running=health.running,
        last_run_at=health.last_run_at.isoformat() if health.last_run_at else None,
        last_run_repos_scanned=health.last_run_repos_scanned,
        last_run_errors=health.last_run_errors,
        repos_due_count=health.repos_due_count,
        poll_interval_minutes=health.poll_interval_minutes,
        poll_max_concurrency=health.poll_max_concurrency,
        poll_tick_seconds=health.poll_tick_seconds,
    )
