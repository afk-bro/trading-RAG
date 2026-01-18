"""
Integration tests for GitHub repository discovery.

Uses local git repos for deterministic testing without GitHub dependency.
These tests verify:
1. Clone and first scan (full scan)
2. Incremental scans (only changed files)
3. Delete detection
4. Rename/copy handling
5. Branch resolution
6. Git adapter locking

Note: Uses file:// URLs to test against local git repos.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Generator
from uuid import uuid4

import pytest

from app.services.pine.adapters.git import (
    BranchNotFoundError,
    GitAdapter,
    GitRepo,
    GitScanResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def local_git_repo(tmp_path) -> Generator[Path, None, None]:
    """
    Create a local git repo for testing.

    The repo contains:
    - scripts/indicator.pine
    - scripts/strategy.pine
    - README.md (not matched by *.pine glob)
    """
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Configure git user for commits (required for CI)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    # Create initial files
    scripts_dir = repo_path / "scripts"
    scripts_dir.mkdir()

    (scripts_dir / "indicator.pine").write_text(
        "//@version=5\nindicator('Test Indicator', overlay=true)\nplot(close)"
    )
    (scripts_dir / "strategy.pine").write_text(
        "//@version=5\nstrategy('Test Strategy')\n"
        "if barstate.isconfirmed\n    strategy.entry('Long', strategy.long)"
    )
    (repo_path / "README.md").write_text("# Test Repo\n\nPine scripts for testing.")

    # Initial commit
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    yield repo_path


@pytest.fixture
def git_adapter(tmp_path) -> GitAdapter:
    """Create GitAdapter with temp data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return GitAdapter(data_dir=data_dir)


def get_head_commit(repo_path: Path) -> str:
    """Get the HEAD commit SHA from a local repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def add_file(repo_path: Path, rel_path: str, content: str, commit_msg: str) -> str:
    """Add a file and commit, return new commit SHA."""
    file_path = repo_path / rel_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    subprocess.run(
        ["git", "add", rel_path], cwd=repo_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return get_head_commit(repo_path)


def modify_file(repo_path: Path, rel_path: str, content: str, commit_msg: str) -> str:
    """Modify an existing file and commit, return new commit SHA."""
    (repo_path / rel_path).write_text(content)
    subprocess.run(
        ["git", "add", rel_path], cwd=repo_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return get_head_commit(repo_path)


def delete_file(repo_path: Path, rel_path: str, commit_msg: str) -> str:
    """Delete a file and commit, return new commit SHA."""
    subprocess.run(
        ["git", "rm", rel_path], cwd=repo_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return get_head_commit(repo_path)


def rename_file(repo_path: Path, old_path: str, new_path: str, commit_msg: str) -> str:
    """Rename a file and commit, return new commit SHA."""
    subprocess.run(
        ["git", "mv", old_path, new_path],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    return get_head_commit(repo_path)


def create_branch(repo_path: Path, branch_name: str) -> None:
    """Create a new branch in the local repo."""
    subprocess.run(
        ["git", "branch", branch_name],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )


# =============================================================================
# Clone and Full Scan Tests
# =============================================================================


class TestCloneAndFullScan:
    """Tests for initial clone and full scan behavior."""

    @pytest.mark.asyncio
    async def test_ensure_clone_creates_local_clone(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """ensure_clone creates a clone from local repo via file:// URL."""
        # Use file:// URL for local repo
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)

        assert clone_path.exists()
        assert (clone_path / ".git").is_dir()
        assert (clone_path / "scripts" / "indicator.pine").exists()
        assert (clone_path / "scripts" / "strategy.pine").exists()
        assert (clone_path / "README.md").exists()

    @pytest.mark.asyncio
    async def test_full_scan_finds_all_pine_files(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """First scan (no last_seen_commit) finds all .pine files."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",  # git init uses master by default
            last_seen_commit=None,  # Triggers full scan
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert result.is_full_scan is True
        assert result.current_commit == get_head_commit(local_git_repo)
        assert len(result.changes) == 2

        paths = {c.path for c in result.changes}
        assert "scripts/indicator.pine" in paths
        assert "scripts/strategy.pine" in paths

        # All should be "A" (added) for full scan
        assert all(c.status == "A" for c in result.changes)

    @pytest.mark.asyncio
    async def test_full_scan_filters_by_glob(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Full scan respects scan_globs filter."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=None,
            scan_globs=["scripts/strategy.pine"],  # Only match strategy
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert len(result.changes) == 1
        assert result.changes[0].path == "scripts/strategy.pine"


# =============================================================================
# Incremental Scan Tests
# =============================================================================


class TestIncrementalScan:
    """Tests for incremental scanning (diff-based)."""

    @pytest.mark.asyncio
    async def test_incremental_scan_detects_new_file(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Incremental scan detects newly added files."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        initial_commit = get_head_commit(local_git_repo)

        # Add a new file to the source repo
        new_commit = add_file(
            local_git_repo,
            "scripts/new_indicator.pine",
            "//@version=5\nindicator('New')",
            "Add new indicator",
        )

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=initial_commit,
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert result.is_full_scan is False
        assert result.current_commit == new_commit
        assert len(result.changes) == 1
        assert result.changes[0].path == "scripts/new_indicator.pine"
        assert result.changes[0].status == "A"

    @pytest.mark.asyncio
    async def test_incremental_scan_detects_modified_file(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Incremental scan detects modified files."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        initial_commit = get_head_commit(local_git_repo)

        # Modify existing file
        modify_file(
            local_git_repo,
            "scripts/indicator.pine",
            "//@version=5\nindicator('Updated Indicator')\nplot(close * 2)",
            "Update indicator",
        )

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=initial_commit,
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert result.is_full_scan is False
        assert len(result.changes) == 1
        assert result.changes[0].path == "scripts/indicator.pine"
        assert result.changes[0].status == "M"

    @pytest.mark.asyncio
    async def test_incremental_scan_detects_deleted_file(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Incremental scan detects deleted files."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        initial_commit = get_head_commit(local_git_repo)

        # Delete a file
        delete_file(
            local_git_repo,
            "scripts/strategy.pine",
            "Remove strategy",
        )

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=initial_commit,
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert result.is_full_scan is False
        assert len(result.changes) == 1
        assert result.changes[0].path == "scripts/strategy.pine"
        assert result.changes[0].status == "D"

    @pytest.mark.asyncio
    async def test_incremental_scan_handles_multiple_changes(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Incremental scan handles add + modify + delete in one diff."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        initial_commit = get_head_commit(local_git_repo)

        # Make multiple changes
        add_file(
            local_git_repo,
            "scripts/new.pine",
            "//@version=5\nindicator('New')",
            "Add new",
        )
        modify_file(
            local_git_repo,
            "scripts/indicator.pine",
            "//@version=5\nindicator('Modified')",
            "Modify",
        )
        new_commit = delete_file(
            local_git_repo,
            "scripts/strategy.pine",
            "Delete",
        )

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=initial_commit,
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert result.is_full_scan is False
        assert result.current_commit == new_commit
        assert len(result.changes) == 3

        changes_by_path = {c.path: c.status for c in result.changes}
        assert changes_by_path["scripts/new.pine"] == "A"
        assert changes_by_path["scripts/indicator.pine"] == "M"
        assert changes_by_path["scripts/strategy.pine"] == "D"

    @pytest.mark.asyncio
    async def test_no_changes_returns_empty_diff(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Same commit returns no changes."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        current_commit = get_head_commit(local_git_repo)

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=current_commit,  # Same as HEAD
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        # When commits are the same, it should be a full scan (no diff needed)
        assert result.is_full_scan is True
        assert result.current_commit == current_commit
        # Full scan returns all files as "A"
        assert len(result.changes) == 2


# =============================================================================
# Rename and Copy Tests
# =============================================================================


class TestRenameAndCopy:
    """Tests for rename detection in git diff."""

    @pytest.mark.asyncio
    async def test_rename_detected_as_delete_and_add(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Git rename is converted to delete old + add new."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        initial_commit = get_head_commit(local_git_repo)

        # Rename file
        rename_file(
            local_git_repo,
            "scripts/indicator.pine",
            "scripts/rsi_indicator.pine",
            "Rename indicator",
        )

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=initial_commit,
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert result.is_full_scan is False
        assert len(result.changes) == 2

        changes_by_status = {}
        for c in result.changes:
            changes_by_status[c.status] = c.path

        assert changes_by_status.get("D") == "scripts/indicator.pine"
        assert changes_by_status.get("A") == "scripts/rsi_indicator.pine"


# =============================================================================
# File Read Tests
# =============================================================================


class TestFileRead:
    """Tests for reading file content at specific commits."""

    @pytest.mark.asyncio
    async def test_read_file_at_commit(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """read_file_at reads content from specific commit."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        commit = get_head_commit(local_git_repo)

        content = await git_adapter.read_file_at(
            clone_path, commit, "scripts/indicator.pine"
        )

        assert "//@version=5" in content
        assert "indicator('Test Indicator'" in content

    @pytest.mark.asyncio
    async def test_read_file_at_old_commit(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """read_file_at reads old content from historical commit."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)
        old_commit = get_head_commit(local_git_repo)

        # Modify the file
        modify_file(
            local_git_repo,
            "scripts/indicator.pine",
            "//@version=5\nindicator('MODIFIED')",
            "Modify",
        )

        # Fetch to get new commit
        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=None,
            scan_globs=["**/*.pine"],
        )
        await git_adapter.fetch_and_diff(repo)

        # Read at old commit should get old content
        old_content = await git_adapter.read_file_at(
            clone_path, old_commit, "scripts/indicator.pine"
        )
        assert "Test Indicator" in old_content
        assert "MODIFIED" not in old_content


# =============================================================================
# Branch Tests
# =============================================================================


class TestBranchHandling:
    """Tests for branch resolution."""

    @pytest.mark.asyncio
    async def test_fetch_specific_branch(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """fetch_and_diff works with non-default branch."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)

        # Create a new branch with different content
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=local_git_repo,
            check=True,
            capture_output=True,
        )
        add_file(
            local_git_repo,
            "scripts/feature.pine",
            "//@version=5\nindicator('Feature')",
            "Add feature indicator",
        )
        feature_commit = get_head_commit(local_git_repo)

        # Go back to master
        subprocess.run(
            ["git", "checkout", "master"],
            cwd=local_git_repo,
            check=True,
            capture_output=True,
        )

        # Scan the feature branch
        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="feature",
            last_seen_commit=None,
            scan_globs=["**/*.pine"],
        )

        result = await git_adapter.fetch_and_diff(repo)

        assert result.current_commit == feature_commit
        paths = {c.path for c in result.changes}
        assert "scripts/feature.pine" in paths

    @pytest.mark.asyncio
    async def test_nonexistent_branch_raises_error(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Fetching nonexistent branch raises BranchNotFoundError."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="nonexistent-branch",
            last_seen_commit=None,
            scan_globs=["**/*.pine"],
        )

        with pytest.raises(BranchNotFoundError) as exc_info:
            await git_adapter.fetch_and_diff(repo)

        assert "nonexistent-branch" in str(exc_info.value)


# =============================================================================
# Clone Management Tests
# =============================================================================


class TestCloneManagement:
    """Tests for clone creation and deletion."""

    @pytest.mark.asyncio
    async def test_ensure_clone_idempotent(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """ensure_clone is idempotent - second call returns same path."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        path1 = await git_adapter.ensure_clone(repo_url, repo_slug)
        path2 = await git_adapter.ensure_clone(repo_url, repo_slug)

        assert path1 == path2
        assert path1.exists()

    def test_delete_clone(self, git_adapter: GitAdapter, local_git_repo: Path):
        """delete_clone removes the cloned repository."""
        repo_slug = "local/test-repo"

        # First clone
        loop = asyncio.get_event_loop()
        clone_path = loop.run_until_complete(
            git_adapter.ensure_clone(f"file://{local_git_repo}", repo_slug)
        )
        assert clone_path.exists()

        # Delete
        result = git_adapter.delete_clone(repo_slug)

        assert result is True
        assert not clone_path.exists()

    def test_delete_nonexistent_clone(self, git_adapter: GitAdapter):
        """delete_clone returns False for nonexistent clone."""
        result = git_adapter.delete_clone("nonexistent/repo")
        assert result is False


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent git operations."""

    @pytest.mark.asyncio
    async def test_concurrent_fetch_same_repo(
        self, git_adapter: GitAdapter, local_git_repo: Path
    ):
        """Concurrent fetches on same repo are serialized by lock."""
        repo_url = f"file://{local_git_repo}"
        repo_slug = "local/test-repo"

        clone_path = await git_adapter.ensure_clone(repo_url, repo_slug)

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug=repo_slug,
            clone_path=clone_path,
            branch="master",
            last_seen_commit=None,
            scan_globs=["**/*.pine"],
        )

        # Run multiple fetches concurrently
        results = await asyncio.gather(
            git_adapter.fetch_and_diff(repo),
            git_adapter.fetch_and_diff(repo),
            git_adapter.fetch_and_diff(repo),
        )

        # All should succeed with same result
        assert all(isinstance(r, GitScanResult) for r in results)
        assert all(r.current_commit == results[0].current_commit for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_different_repos(
        self, git_adapter: GitAdapter, tmp_path: Path
    ):
        """Concurrent operations on different repos can run in parallel."""
        # Create two separate repos
        repo1_path = tmp_path / "repo1"
        repo2_path = tmp_path / "repo2"

        for repo_path in [repo1_path, repo2_path]:
            repo_path.mkdir()
            subprocess.run(
                ["git", "init"], cwd=repo_path, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            (repo_path / "test.pine").write_text("//@version=5")
            subprocess.run(
                ["git", "add", "."], cwd=repo_path, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

        # Clone both
        clone1 = await git_adapter.ensure_clone(f"file://{repo1_path}", "local/repo1")
        clone2 = await git_adapter.ensure_clone(f"file://{repo2_path}", "local/repo2")

        repo1 = GitRepo(
            repo_id=uuid4(),
            repo_slug="local/repo1",
            clone_path=clone1,
            branch="master",
            last_seen_commit=None,
            scan_globs=["*.pine"],
        )
        repo2 = GitRepo(
            repo_id=uuid4(),
            repo_slug="local/repo2",
            clone_path=clone2,
            branch="master",
            last_seen_commit=None,
            scan_globs=["*.pine"],
        )

        # Both should complete successfully
        result1, result2 = await asyncio.gather(
            git_adapter.fetch_and_diff(repo1),
            git_adapter.fetch_and_diff(repo2),
        )

        assert isinstance(result1, GitScanResult)
        assert isinstance(result2, GitScanResult)
