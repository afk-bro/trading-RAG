"""
Git adapter for Pine Script repository scanning.

Provides async git operations for cloning, fetching, diffing, and reading files
from the git object database (not working tree).

Key design decisions:
- Commit-addressed reads via `git show <commit>:<path>` (no working tree reads)
- Explicit `origin/<branch>` resolution (not HEAD)
- Dual locking: asyncio.Lock (in-process) + file lock (cross-process)
- Glob filtering via PurePosixPath.match() for predictable ** behavior
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import re
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Literal, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class GitAdapterError(Exception):
    """Base exception for git adapter errors."""

    pass


class GitCommandError(GitAdapterError):
    """Git command failed."""

    def __init__(self, command: list[str], returncode: int, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        cmd_str = " ".join(command[:3]) + ("..." if len(command) > 3 else "")
        super().__init__(
            f"Git command failed ({returncode}): {cmd_str}: {stderr[:200]}"
        )


class InvalidRepoSlugError(GitAdapterError):
    """Invalid repository slug format."""

    def __init__(self, slug: str, reason: str = "invalid format"):
        self.slug = slug
        self.reason = reason
        super().__init__(f"Invalid repo slug '{slug}': {reason}")


class InvalidRepoUrlError(GitAdapterError):
    """Invalid GitHub repository URL."""

    def __init__(self, url: str, reason: str = "invalid format"):
        self.url = url
        self.reason = reason
        super().__init__(f"Invalid repo URL '{url}': {reason}")


class BranchNotFoundError(GitAdapterError):
    """Branch not found on remote."""

    def __init__(self, branch: str, repo_slug: str):
        self.branch = branch
        self.repo_slug = repo_slug
        super().__init__(
            f"Branch '{branch}' not found on remote for {repo_slug}. "
            "Check repository settings."
        )


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GitRepo:
    """Repository configuration for git operations."""

    repo_id: UUID
    repo_slug: str
    clone_path: Path
    branch: str
    last_seen_commit: Optional[str]
    scan_globs: list[str] = field(default_factory=lambda: ["**/*.pine"])


@dataclass
class FileChange:
    """A single file change from git diff."""

    path: str  # Relative path from repo root
    status: Literal["A", "M", "D"]  # Added, Modified, Deleted


@dataclass
class GitScanResult:
    """Result of a fetch and diff operation."""

    current_commit: str  # New HEAD after fetch (origin/<branch>)
    changes: list[FileChange]  # Changed files matching globs
    is_full_scan: bool  # True if no previous commit (first scan)


# =============================================================================
# Slug and URL Validation
# =============================================================================

# Valid slug format: owner/repo with safe characters only
SLUG_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# GitHub URL pattern: https://github.com/owner/repo(.git)?
GITHUB_URL_PATTERN = re.compile(
    r"^https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)


def validate_repo_slug(slug: str) -> bool:
    """
    Validate repo_slug format (owner/repo, safe chars only).

    Returns True if valid, False otherwise.
    """
    if not slug:
        return False
    if ".." in slug:  # Path traversal attempt
        return False
    return bool(SLUG_PATTERN.match(slug))


def extract_slug_from_url(url: str) -> str:
    """
    Extract owner/repo from GitHub URL.

    Args:
        url: GitHub repository URL (https://github.com/owner/repo)

    Returns:
        Normalized slug (owner/repo)

    Raises:
        InvalidRepoUrlError: If URL format is invalid
    """
    if not url:
        raise InvalidRepoUrlError(url, "empty URL")

    match = GITHUB_URL_PATTERN.match(url)
    if not match:
        raise InvalidRepoUrlError(
            url,
            "must be https://github.com/owner/repo format",
        )

    owner, repo = match.group(1), match.group(2)
    slug = f"{owner}/{repo}"

    # Validate extracted slug
    if not validate_repo_slug(slug):
        raise InvalidRepoUrlError(url, f"extracted slug '{slug}' is invalid")

    return slug


def slug_to_clone_dirname(slug: str) -> str:
    """
    Convert repo slug to clone directory name.

    owner/repo -> owner__repo
    """
    return slug.replace("/", "__")


# =============================================================================
# Git Adapter
# =============================================================================


class GitAdapter:
    """
    Async git operations for repository scanning.

    Provides:
    - Cloning with locking
    - Fetching with explicit branch resolution
    - Diffing with rename/copy handling
    - Commit-addressed file reads

    Thread/process safety:
    - asyncio.Lock per repo (in-process)
    - File lock per repo (cross-process)
    """

    def __init__(self, data_dir: Path):
        """
        Initialize git adapter.

        Args:
            data_dir: Base data directory (repos stored in data_dir/repos/)
        """
        self._repos_dir = Path(data_dir) / "repos"
        self._locks_dir = self._repos_dir / ".locks"
        self._repos_dir.mkdir(parents=True, exist_ok=True)
        self._locks_dir.mkdir(parents=True, exist_ok=True)
        self._async_locks: dict[str, asyncio.Lock] = {}

    def get_clone_path(self, repo_slug: str) -> Path:
        """
        Derive clone path from slug.

        owner/repo -> data_dir/repos/owner__repo
        """
        if not validate_repo_slug(repo_slug):
            raise InvalidRepoSlugError(repo_slug)
        return self._repos_dir / slug_to_clone_dirname(repo_slug)

    def _get_file_lock_path(self, repo_slug: str) -> Path:
        """File lock path for cross-process safety."""
        return self._locks_dir / f"{slug_to_clone_dirname(repo_slug)}.lock"

    @asynccontextmanager
    async def _repo_lock(self, repo_slug: str) -> AsyncIterator[None]:
        """
        Acquire both async lock (in-process) and file lock (cross-process).

        This prevents concurrent git operations on the same clone.
        """
        # Ensure async lock exists for this repo
        if repo_slug not in self._async_locks:
            self._async_locks[repo_slug] = asyncio.Lock()

        async with self._async_locks[repo_slug]:
            lock_path = self._get_file_lock_path(repo_slug)
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            # File-based lock for cross-process safety
            with open(lock_path, "w") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    yield
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def _run_git(
        self,
        args: list[str],
        cwd: Optional[Path] = None,
        check: bool = True,
    ) -> str:
        """
        Run a git command asynchronously.

        Args:
            args: Git command arguments (without 'git')
            cwd: Working directory (optional)
            check: Raise exception on non-zero return code

        Returns:
            stdout as string

        Raises:
            GitCommandError: If command fails and check=True
        """
        cmd = ["git"] + args
        logger.debug("Running git command: %s (cwd=%s)", " ".join(cmd), cwd)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if check and process.returncode != 0:
            raise GitCommandError(cmd, process.returncode or 1, stderr)

        return stdout

    async def ensure_clone(self, repo_url: str, repo_slug: str) -> Path:
        """
        Clone repository if not already cloned.

        Args:
            repo_url: Full GitHub URL
            repo_slug: Normalized owner/repo

        Returns:
            Path to cloned repository

        Raises:
            InvalidRepoSlugError: If slug is invalid
            GitCommandError: If clone fails
        """
        if not validate_repo_slug(repo_slug):
            raise InvalidRepoSlugError(repo_slug)

        clone_path = self.get_clone_path(repo_slug)

        async with self._repo_lock(repo_slug):
            if (clone_path / ".git").exists():
                logger.debug("Clone already exists: %s", clone_path)
                return clone_path

            logger.info("Cloning %s to %s", repo_url, clone_path)
            await self._run_git(["clone", repo_url, str(clone_path)])

        return clone_path

    async def fetch_and_diff(self, repo: GitRepo) -> GitScanResult:
        """
        Fetch latest commits and compute diff.

        1. git fetch origin <branch>
        2. Resolve new commit via rev-parse origin/<branch>
        3. If last_seen_commit: diff old..new with --name-status
        4. Else: list all files via ls-tree (full scan)
        5. Filter results by scan_globs

        Args:
            repo: Repository configuration

        Returns:
            GitScanResult with current commit and changes

        Raises:
            BranchNotFoundError: If branch doesn't exist
            GitCommandError: If git operations fail
        """
        async with self._repo_lock(repo.repo_slug):
            clone_path = repo.clone_path

            # Fetch target branch explicitly
            try:
                await self._run_git(
                    ["fetch", "origin", repo.branch],
                    cwd=clone_path,
                )
            except GitCommandError as e:
                if "couldn't find remote ref" in e.stderr.lower():
                    raise BranchNotFoundError(repo.branch, repo.repo_slug) from e
                raise

            # Resolve new commit from origin/<branch>
            try:
                new_commit = await self._run_git(
                    ["rev-parse", f"origin/{repo.branch}"],
                    cwd=clone_path,
                )
                new_commit = new_commit.strip()
            except GitCommandError as e:
                raise BranchNotFoundError(repo.branch, repo.repo_slug) from e

            if repo.last_seen_commit and not repo.last_seen_commit == new_commit:
                # Incremental: diff old..new with --name-status
                diff_output = await self._run_git(
                    [
                        "diff",
                        "--name-status",
                        f"{repo.last_seen_commit}..{new_commit}",
                    ],
                    cwd=clone_path,
                )
                changes = self._parse_diff_name_status(diff_output, repo.scan_globs)
                is_full_scan = False
            else:
                # Full scan: list all files at commit
                tree_output = await self._run_git(
                    ["ls-tree", "-r", "--name-only", new_commit],
                    cwd=clone_path,
                )
                all_files = [f for f in tree_output.strip().split("\n") if f]
                matching = self._filter_by_globs(all_files, repo.scan_globs)
                changes = [FileChange(path=p, status="A") for p in matching]
                is_full_scan = True

            return GitScanResult(
                current_commit=new_commit,
                changes=changes,
                is_full_scan=is_full_scan,
            )

    async def read_file_at(
        self,
        clone_path: Path,
        commit: str,
        rel_path: str,
    ) -> str:
        """
        Read file content from git object database (not working tree).

        Uses `git show <commit>:<path>` to read file at specific commit.

        Args:
            clone_path: Path to cloned repository
            commit: Commit SHA
            rel_path: Relative path from repo root

        Returns:
            File content as string

        Raises:
            GitCommandError: If file doesn't exist at commit
        """
        return await self._run_git(
            ["show", f"{commit}:{rel_path}"],
            cwd=clone_path,
        )

    def delete_clone(self, repo_slug: str) -> bool:
        """
        Delete a cloned repository.

        Args:
            repo_slug: Repository slug (owner/repo)

        Returns:
            True if deleted, False if didn't exist
        """
        if not validate_repo_slug(repo_slug):
            raise InvalidRepoSlugError(repo_slug)

        clone_path = self.get_clone_path(repo_slug)
        if clone_path.exists():
            shutil.rmtree(clone_path)
            logger.info("Deleted clone: %s", clone_path)
            return True
        return False

    def _parse_diff_name_status(
        self,
        output: str,
        globs: list[str],
    ) -> list[FileChange]:
        """
        Parse 'git diff --name-status' output, filter by globs.

        Handles:
        - A (added), M (modified), D (deleted) - single path
        - R (renamed) - treat as D old_path + A new_path
        - C (copied) - treat as A new_path

        Format: STATUS\tpath or STATUS\told_path\tnew_path (for R/C)
        """
        changes: list[FileChange] = []

        for line in output.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            status = parts[0]

            if status.startswith("R"):  # Rename: R100\told\tnew
                if len(parts) >= 3:
                    old_path, new_path = parts[1], parts[2]
                    # D old + A new
                    if self._matches_globs(old_path, globs):
                        changes.append(FileChange(path=old_path, status="D"))
                    if self._matches_globs(new_path, globs):
                        changes.append(FileChange(path=new_path, status="A"))
            elif status.startswith("C"):  # Copy: C100\tsrc\tdst
                if len(parts) >= 3:
                    new_path = parts[2]
                    if self._matches_globs(new_path, globs):
                        changes.append(FileChange(path=new_path, status="A"))
            elif status in ("A", "M", "D"):
                path = parts[1]
                if self._matches_globs(path, globs):
                    # Cast is safe because of the check above
                    changes.append(
                        FileChange(path=path, status=status)  # type: ignore[arg-type]
                    )

        return changes

    def _matches_globs(self, path: str, globs: list[str]) -> bool:
        """
        Check if path matches any of the glob patterns.

        Uses PurePosixPath.match() for predictable ** recursive behavior.
        """
        p = PurePosixPath(path)
        return any(p.match(g) for g in globs)

    def _filter_by_globs(self, paths: list[str], globs: list[str]) -> list[str]:
        """Filter paths by glob patterns."""
        return [p for p in paths if self._matches_globs(p, globs)]


# =============================================================================
# Utility Functions
# =============================================================================


def build_github_blob_url(repo_slug: str, commit: str, rel_path: str) -> str:
    """
    Build commit-specific GitHub blob URL.

    Returns:
        URL like https://github.com/owner/repo/blob/abc123/path/to/file.pine
    """
    return f"https://github.com/{repo_slug}/blob/{commit}/{rel_path}"
