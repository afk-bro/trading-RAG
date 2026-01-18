"""Repository for pine_repos table operations."""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

ERROR_TRUNCATE_LEN = 1000


# =============================================================================
# Models
# =============================================================================


@dataclass
class PineRepo:
    """Model for pine_repos table row."""

    id: UUID
    workspace_id: UUID
    repo_url: str
    repo_slug: str
    clone_path: Optional[str] = None
    branch: str = "main"
    last_seen_commit: Optional[str] = None
    last_scan_at: Optional[datetime] = None
    last_scan_ok: Optional[bool] = None
    last_scan_error: Optional[str] = None
    scripts_count: int = 0
    last_pull_at: Optional[datetime] = None
    last_pull_ok: Optional[bool] = None
    pull_error: Optional[str] = None
    enabled: bool = True
    scan_globs: list[str] = field(default_factory=lambda: ["**/*.pine"])
    # Polling fields (added in migration 061)
    next_scan_at: Optional[datetime] = None
    failure_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class RepoHealthStats:
    """Health statistics for repos in a workspace."""

    repos_total: int = 0
    repos_enabled: int = 0
    repos_pull_failed: int = 0
    repos_stale: int = 0  # Not scanned in 7+ days
    oldest_scan_age_hours: Optional[float] = None


# =============================================================================
# Repository
# =============================================================================


class PineRepoRepository:
    """Repository for pine_repos table."""

    def __init__(self, pool):
        """Initialize with connection pool."""
        self._pool = pool

    async def create(
        self,
        workspace_id: UUID,
        repo_url: str,
        repo_slug: str,
        branch: str = "main",
        scan_globs: Optional[list[str]] = None,
    ) -> PineRepo:
        """
        Create a new repo registration.

        Args:
            workspace_id: Workspace to associate repo with
            repo_url: Full GitHub URL
            repo_slug: Normalized owner/repo identifier
            branch: Branch to track (default: main)
            scan_globs: Glob patterns for matching Pine files

        Returns:
            Created PineRepo

        Raises:
            asyncpg.UniqueViolationError: If repo already registered
            asyncpg.CheckViolationError: If slug format is invalid
        """
        if scan_globs is None:
            scan_globs = ["**/*.pine"]

        query = """
            INSERT INTO pine_repos (
                workspace_id, repo_url, repo_slug, branch, scan_globs
            ) VALUES (
                $1, $2, $3, $4, $5
            )
            RETURNING id, workspace_id, repo_url, repo_slug, clone_path,
                      branch, last_seen_commit, last_scan_at, last_scan_ok,
                      last_scan_error, scripts_count, last_pull_at, last_pull_ok,
                      pull_error, enabled, scan_globs, next_scan_at, failure_count,
                      created_at, updated_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                repo_url,
                repo_slug,
                branch,
                scan_globs,
            )
            return self._row_to_model(row)

    async def get(self, repo_id: UUID) -> Optional[PineRepo]:
        """Get repo by ID."""
        query = """
            SELECT id, workspace_id, repo_url, repo_slug, clone_path,
                   branch, last_seen_commit, last_scan_at, last_scan_ok,
                   last_scan_error, scripts_count, last_pull_at, last_pull_ok,
                   pull_error, enabled, scan_globs, next_scan_at, failure_count,
                   created_at, updated_at
            FROM pine_repos
            WHERE id = $1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, repo_id)
            if row:
                return self._row_to_model(row)
        return None

    async def get_by_slug(
        self,
        workspace_id: UUID,
        repo_slug: str,
    ) -> Optional[PineRepo]:
        """Get repo by workspace and slug."""
        query = """
            SELECT id, workspace_id, repo_url, repo_slug, clone_path,
                   branch, last_seen_commit, last_scan_at, last_scan_ok,
                   last_scan_error, scripts_count, last_pull_at, last_pull_ok,
                   pull_error, enabled, scan_globs, next_scan_at, failure_count,
                   created_at, updated_at
            FROM pine_repos
            WHERE workspace_id = $1 AND repo_slug = $2
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id, repo_slug)
            if row:
                return self._row_to_model(row)
        return None

    async def list_by_workspace(
        self,
        workspace_id: UUID,
        enabled_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PineRepo], int]:
        """
        List repos for a workspace with pagination.

        Args:
            workspace_id: Workspace to list repos for
            enabled_only: Only return enabled repos
            limit: Max results
            offset: Pagination offset

        Returns:
            Tuple of (repos, total_count)
        """
        where_clause = "workspace_id = $1"
        params: list = [workspace_id]

        if enabled_only:
            where_clause += " AND enabled = TRUE"

        count_query = f"SELECT COUNT(*) FROM pine_repos WHERE {where_clause}"
        data_query = f"""
            SELECT id, workspace_id, repo_url, repo_slug, clone_path,
                   branch, last_seen_commit, last_scan_at, last_scan_ok,
                   last_scan_error, scripts_count, last_pull_at, last_pull_ok,
                   pull_error, enabled, scan_globs, next_scan_at, failure_count,
                   created_at, updated_at
            FROM pine_repos
            WHERE {where_clause}
            ORDER BY repo_slug
            LIMIT $2 OFFSET $3
        """

        async with self._pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params)
            rows = await conn.fetch(data_query, workspace_id, limit, offset)
            repos = [self._row_to_model(row) for row in rows]
            return repos, total or 0

    async def list_enabled(self, workspace_id: UUID) -> list[PineRepo]:
        """List all enabled repos for a workspace."""
        query = """
            SELECT id, workspace_id, repo_url, repo_slug, clone_path,
                   branch, last_seen_commit, last_scan_at, last_scan_ok,
                   last_scan_error, scripts_count, last_pull_at, last_pull_ok,
                   pull_error, enabled, scan_globs, next_scan_at, failure_count,
                   created_at, updated_at
            FROM pine_repos
            WHERE workspace_id = $1 AND enabled = TRUE
            ORDER BY repo_slug
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id)
            return [self._row_to_model(row) for row in rows]

    async def update_clone_path(self, repo_id: UUID, clone_path: str) -> None:
        """Update clone path after successful clone."""
        query = """
            UPDATE pine_repos
            SET clone_path = $1
            WHERE id = $2
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, clone_path, repo_id)

    async def update_scan_result(
        self,
        repo_id: UUID,
        commit: str,
        ok: bool,
        error: Optional[str],
        scripts_count: int,
    ) -> None:
        """
        Update scan result after discovery run.

        Args:
            repo_id: Repo to update
            commit: New commit SHA (only updated if ok=True)
            ok: Whether scan succeeded
            error: Error message if failed (truncated)
            scripts_count: Number of scripts found (only updated if ok=True)
        """
        # Truncate error
        if error and len(error) > ERROR_TRUNCATE_LEN:
            error = error[: ERROR_TRUNCATE_LEN - 3] + "..."

        if ok:
            query = """
                UPDATE pine_repos SET
                    last_scan_at = $1,
                    last_scan_ok = TRUE,
                    last_scan_error = NULL,
                    last_seen_commit = $2,
                    scripts_count = $3
                WHERE id = $4
            """
            async with self._pool.acquire() as conn:
                await conn.execute(
                    query,
                    datetime.now(timezone.utc),
                    commit,
                    scripts_count,
                    repo_id,
                )
        else:
            query = """
                UPDATE pine_repos SET
                    last_scan_at = $1,
                    last_scan_ok = FALSE,
                    last_scan_error = $2
                WHERE id = $3
            """
            async with self._pool.acquire() as conn:
                await conn.execute(
                    query,
                    datetime.now(timezone.utc),
                    error,
                    repo_id,
                )

    async def update_pull_result(
        self,
        repo_id: UUID,
        ok: bool,
        error: Optional[str],
    ) -> None:
        """
        Update pull/fetch result.

        Args:
            repo_id: Repo to update
            ok: Whether pull succeeded
            error: Error message if failed (truncated)
        """
        # Truncate error
        if error and len(error) > ERROR_TRUNCATE_LEN:
            error = error[: ERROR_TRUNCATE_LEN - 3] + "..."

        query = """
            UPDATE pine_repos SET
                last_pull_at = $1,
                last_pull_ok = $2,
                pull_error = $3
            WHERE id = $4
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                datetime.now(timezone.utc),
                ok,
                error if not ok else None,
                repo_id,
            )

    async def set_enabled(self, repo_id: UUID, enabled: bool) -> None:
        """Enable or disable a repo."""
        query = """
            UPDATE pine_repos
            SET enabled = $1
            WHERE id = $2
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, enabled, repo_id)

    # =========================================================================
    # Polling Methods
    # =========================================================================

    async def list_due_for_poll(
        self,
        workspace_id: Optional[UUID] = None,
        limit: int = 10,
    ) -> list[PineRepo]:
        """
        List repos due for polling scan.

        Selection criteria:
        - enabled = true
        - next_scan_at IS NULL OR next_scan_at <= now()

        Order: next_scan_at NULLS FIRST (never scanned), then by last_scan_at ASC

        Args:
            workspace_id: Optional filter by workspace (None = all workspaces)
            limit: Max repos to return (default 10, to cap per tick)

        Returns:
            List of repos due for scanning
        """
        where_clause = (
            "enabled = true AND (next_scan_at IS NULL OR next_scan_at <= NOW())"
        )
        params: list = []

        if workspace_id:
            where_clause += " AND workspace_id = $1"
            params = [workspace_id]

        query = f"""
            SELECT id, workspace_id, repo_url, repo_slug, clone_path,
                   branch, last_seen_commit, last_scan_at, last_scan_ok,
                   last_scan_error, scripts_count, last_pull_at, last_pull_ok,
                   pull_error, enabled, scan_globs, next_scan_at, failure_count,
                   created_at, updated_at
            FROM pine_repos
            WHERE {where_clause}
            ORDER BY next_scan_at NULLS FIRST, last_scan_at ASC
            LIMIT ${len(params) + 1}
        """
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_model(row) for row in rows]

    async def update_poll_success(
        self,
        repo_id: UUID,
        interval_minutes: int = 15,
        jitter_pct: float = 0.1,
    ) -> None:
        """
        Update repo after successful poll scan.

        Sets:
        - failure_count = 0
        - next_scan_at = now + interval (with jitter)

        Args:
            repo_id: Repo to update
            interval_minutes: Base interval for next scan
            jitter_pct: Random jitter percentage (±10% default)
        """
        # Apply jitter: ± jitter_pct
        jitter = random.uniform(-jitter_pct, jitter_pct)
        interval_with_jitter = interval_minutes * (1 + jitter)
        next_scan = datetime.now(timezone.utc) + timedelta(minutes=interval_with_jitter)

        query = """
            UPDATE pine_repos SET
                failure_count = 0,
                next_scan_at = $1
            WHERE id = $2
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, next_scan, repo_id)

    async def update_poll_failure(
        self,
        repo_id: UUID,
        base_interval_minutes: int = 15,
        max_backoff_multiplier: int = 16,
    ) -> None:
        """
        Update repo after failed poll scan.

        Sets:
        - failure_count += 1
        - next_scan_at = now + (min(2^failure_count, max_multiplier) * base_interval)

        Args:
            repo_id: Repo to update
            base_interval_minutes: Base interval for backoff
            max_backoff_multiplier: Cap on backoff multiplier (default 16x = 4h at 15min base)
        """
        # First increment failure count and get new value
        query = """
            UPDATE pine_repos SET
                failure_count = failure_count + 1,
                next_scan_at = NOW() + (
                    LEAST(POWER(2, failure_count + 1), $1) * $2 * INTERVAL '1 minute'
                )
            WHERE id = $3
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                max_backoff_multiplier,
                base_interval_minutes,
                repo_id,
            )

    async def count_due_for_poll(self, workspace_id: Optional[UUID] = None) -> int:
        """
        Count repos currently due for polling.

        Args:
            workspace_id: Optional filter by workspace

        Returns:
            Number of repos due for scan
        """
        where_clause = (
            "enabled = true AND (next_scan_at IS NULL OR next_scan_at <= NOW())"
        )
        params: list = []

        if workspace_id:
            where_clause += " AND workspace_id = $1"
            params = [workspace_id]

        query = f"SELECT COUNT(*) FROM pine_repos WHERE {where_clause}"

        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *params) or 0

    async def delete(self, repo_id: UUID) -> bool:
        """
        Delete a repo registration.

        Returns True if deleted, False if not found.
        """
        query = "DELETE FROM pine_repos WHERE id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, repo_id)
            count = int(result.split()[-1]) if result else 0
            return count > 0

    async def get_health_stats(
        self, workspace_id: Optional[UUID] = None
    ) -> RepoHealthStats:
        """
        Get health statistics for repos.

        Args:
            workspace_id: Filter by workspace (None = all workspaces)

        Returns:
            RepoHealthStats with counts and metrics
        """
        where_clause = ""
        params: list = []

        if workspace_id:
            where_clause = "WHERE workspace_id = $1"
            params = [workspace_id]

        query = f"""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE enabled = TRUE) as enabled,
                COUNT(*) FILTER (WHERE last_pull_ok = FALSE) as pull_failed,
                COUNT(*) FILTER (
                    WHERE last_scan_at IS NOT NULL
                    AND last_scan_at < NOW() - INTERVAL '7 days'
                ) as stale,
                EXTRACT(EPOCH FROM (NOW() - MIN(last_scan_at))) / 3600.0 as oldest_scan_hours
            FROM pine_repos
            {where_clause}
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return RepoHealthStats(
                repos_total=row["total"] or 0,
                repos_enabled=row["enabled"] or 0,
                repos_pull_failed=row["pull_failed"] or 0,
                repos_stale=row["stale"] or 0,
                oldest_scan_age_hours=row["oldest_scan_hours"],
            )

    def _row_to_model(self, row) -> PineRepo:
        """Convert DB row to model."""
        return PineRepo(
            id=row["id"],
            workspace_id=row["workspace_id"],
            repo_url=row["repo_url"],
            repo_slug=row["repo_slug"],
            clone_path=row["clone_path"],
            branch=row["branch"],
            last_seen_commit=row["last_seen_commit"],
            last_scan_at=row["last_scan_at"],
            last_scan_ok=row["last_scan_ok"],
            last_scan_error=row["last_scan_error"],
            scripts_count=row["scripts_count"],
            last_pull_at=row["last_pull_at"],
            last_pull_ok=row["last_pull_ok"],
            pull_error=row["pull_error"],
            enabled=row["enabled"],
            scan_globs=row["scan_globs"] or ["**/*.pine"],
            next_scan_at=row["next_scan_at"],
            failure_count=row["failure_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
