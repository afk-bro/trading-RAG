"""
Unit tests for Pine Script Git adapter.

Tests:
- Repository slug validation
- URL parsing to extract slug
- Clone path derivation
- Git diff parsing (A/M/D/R/C)
- Glob pattern matching
- Error handling for git commands
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.services.pine.adapters.git import (
    BranchNotFoundError,
    FileChange,
    GitAdapter,
    GitCommandError,
    GitRepo,
    GitScanResult,
    InvalidRepoUrlError,
    build_github_blob_url,
    extract_slug_from_url,
    validate_repo_slug,
)


class TestValidateRepoSlug:
    """Tests for validate_repo_slug()."""

    def test_valid_simple_slug(self):
        """Simple owner/repo slugs are valid."""
        assert validate_repo_slug("owner/repo") is True
        assert validate_repo_slug("acme/trading-scripts") is True
        assert validate_repo_slug("user123/pine_indicators") is True

    def test_valid_slug_with_dots(self):
        """Slugs with dots are valid."""
        assert validate_repo_slug("owner.name/repo.name") is True
        assert validate_repo_slug("my.org/my.scripts") is True

    def test_valid_slug_with_underscores(self):
        """Slugs with underscores are valid."""
        assert validate_repo_slug("owner_name/repo_name") is True
        assert validate_repo_slug("trading_bot/pine_v5") is True

    def test_valid_slug_with_dashes(self):
        """Slugs with dashes are valid."""
        assert validate_repo_slug("owner-name/repo-name") is True
        assert validate_repo_slug("my-org/my-scripts") is True

    def test_invalid_slug_no_slash(self):
        """Slugs without slash are invalid."""
        assert validate_repo_slug("owner-repo") is False

    def test_invalid_slug_multiple_slashes(self):
        """Slugs with multiple slashes are invalid."""
        assert validate_repo_slug("owner/repo/extra") is False

    def test_invalid_slug_empty_parts(self):
        """Slugs with empty parts are invalid."""
        assert validate_repo_slug("/repo") is False
        assert validate_repo_slug("owner/") is False
        assert validate_repo_slug("/") is False

    def test_invalid_slug_special_chars(self):
        """Slugs with special characters are invalid."""
        assert validate_repo_slug("owner/repo@latest") is False
        assert validate_repo_slug("owner/repo#main") is False
        assert validate_repo_slug("owner/../repo") is False

    def test_invalid_slug_path_traversal(self):
        """Path traversal attempts are blocked."""
        assert validate_repo_slug("../etc/passwd") is False
        assert validate_repo_slug("owner/..%2F..%2F") is False


class TestExtractSlugFromUrl:
    """Tests for extract_slug_from_url()."""

    def test_https_url(self):
        """Extract slug from HTTPS URL."""
        assert extract_slug_from_url("https://github.com/owner/repo") == "owner/repo"

    def test_https_url_with_git_suffix(self):
        """Extract slug from URL with .git suffix."""
        assert (
            extract_slug_from_url("https://github.com/owner/repo.git") == "owner/repo"
        )

    def test_https_url_with_trailing_slash(self):
        """Extract slug from URL with trailing slash."""
        assert extract_slug_from_url("https://github.com/owner/repo/") == "owner/repo"

    def test_http_url(self):
        """Extract slug from HTTP URL."""
        assert extract_slug_from_url("http://github.com/owner/repo") == "owner/repo"

    def test_complex_slug(self):
        """Extract slug with dots, dashes, underscores."""
        assert (
            extract_slug_from_url("https://github.com/my-org/trading_scripts.v2")
            == "my-org/trading_scripts.v2"
        )

    def test_invalid_url_wrong_domain(self):
        """Non-GitHub URLs raise error."""
        with pytest.raises(InvalidRepoUrlError):
            extract_slug_from_url("https://gitlab.com/owner/repo")

    def test_invalid_url_no_repo(self):
        """URLs without repo path raise error."""
        with pytest.raises(InvalidRepoUrlError):
            extract_slug_from_url("https://github.com/owner")

    def test_invalid_url_wrong_format(self):
        """Malformed URLs raise error."""
        with pytest.raises(InvalidRepoUrlError):
            extract_slug_from_url("not-a-url")
        with pytest.raises(InvalidRepoUrlError):
            extract_slug_from_url("github.com/owner/repo")  # Missing protocol


class TestBuildGithubBlobUrl:
    """Tests for build_github_blob_url()."""

    def test_basic_blob_url(self):
        """Build basic blob URL."""
        url = build_github_blob_url("owner/repo", "abc123", "path/file.pine")
        assert url == "https://github.com/owner/repo/blob/abc123/path/file.pine"

    def test_blob_url_special_chars(self):
        """Build blob URL with special characters in path."""
        url = build_github_blob_url(
            "my-org/scripts", "def456", "strategies/52w-high.pine"
        )
        assert (
            url
            == "https://github.com/my-org/scripts/blob/def456/strategies/52w-high.pine"
        )


class TestGitRepoDataclass:
    """Tests for GitRepo dataclass."""

    def test_git_repo_creation(self):
        """GitRepo stores all fields correctly."""
        repo_id = uuid4()
        repo = GitRepo(
            repo_id=repo_id,
            repo_slug="owner/repo",
            clone_path=Path("/data/repos/owner__repo"),
            branch="main",
            last_seen_commit="abc123",
            scan_globs=["**/*.pine"],
        )
        assert repo.repo_id == repo_id
        assert repo.repo_slug == "owner/repo"
        assert repo.branch == "main"
        assert repo.last_seen_commit == "abc123"

    def test_git_repo_none_last_commit(self):
        """GitRepo allows None last_seen_commit."""
        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug="owner/repo",
            clone_path=Path("/data/repos/owner__repo"),
            branch="main",
            last_seen_commit=None,
            scan_globs=["**/*.pine"],
        )
        assert repo.last_seen_commit is None


class TestFileChangeDataclass:
    """Tests for FileChange dataclass."""

    def test_file_change_added(self):
        """FileChange records added files."""
        change = FileChange(path="scripts/new.pine", status="A")
        assert change.path == "scripts/new.pine"
        assert change.status == "A"

    def test_file_change_modified(self):
        """FileChange records modified files."""
        change = FileChange(path="scripts/updated.pine", status="M")
        assert change.status == "M"

    def test_file_change_deleted(self):
        """FileChange records deleted files."""
        change = FileChange(path="scripts/old.pine", status="D")
        assert change.status == "D"


class TestGitScanResultDataclass:
    """Tests for GitScanResult dataclass."""

    def test_git_scan_result(self):
        """GitScanResult stores scan results."""
        changes = [
            FileChange(path="a.pine", status="A"),
            FileChange(path="b.pine", status="M"),
        ]
        result = GitScanResult(
            current_commit="abc123",
            changes=changes,
            is_full_scan=False,
        )
        assert result.current_commit == "abc123"
        assert len(result.changes) == 2
        assert result.is_full_scan is False

    def test_git_scan_result_full_scan(self):
        """GitScanResult indicates full scan."""
        result = GitScanResult(
            current_commit="def456",
            changes=[],
            is_full_scan=True,
        )
        assert result.is_full_scan is True


class TestGitAdapter:
    """Tests for GitAdapter class."""

    @pytest.fixture
    def adapter(self, tmp_path):
        """Create GitAdapter with temp directory."""
        return GitAdapter(data_dir=tmp_path)

    def test_get_clone_path(self, adapter):
        """get_clone_path converts slug to path."""
        path = adapter.get_clone_path("owner/repo")
        assert path.name == "owner__repo"
        assert path.parent == adapter._repos_dir

    def test_get_clone_path_special_chars(self, adapter):
        """get_clone_path handles special characters."""
        path = adapter.get_clone_path("my-org/trading_scripts.v2")
        assert path.name == "my-org__trading_scripts.v2"

    def test_matches_globs_simple(self, adapter):
        """_matches_globs matches simple patterns."""
        assert adapter._matches_globs("scripts/a.pine", ["**/*.pine"]) is True
        assert adapter._matches_globs("scripts/a.txt", ["**/*.pine"]) is False

    def test_matches_globs_multiple(self, adapter):
        """_matches_globs with multiple patterns."""
        # Note: PurePosixPath.match("**/*.pine") requires a directory in path
        # For root-level files, need explicit "*.pine" pattern
        globs = ["*.pine", "*.pinescript", "**/*.pine", "**/*.pinescript"]
        assert adapter._matches_globs("a.pine", globs) is True
        assert adapter._matches_globs("a.pinescript", globs) is True
        assert adapter._matches_globs("a.txt", globs) is False
        # Nested paths work with **
        assert adapter._matches_globs("dir/a.pine", globs) is True

    def test_matches_globs_directory(self, adapter):
        """_matches_globs works with directory patterns."""
        # strategies/**/*.pine requires a subdirectory under strategies/
        globs = ["strategies/*.pine", "strategies/**/*.pine"]
        assert adapter._matches_globs("strategies/breakout.pine", globs) is True
        assert adapter._matches_globs("strategies/momentum/rsi.pine", globs) is True
        assert adapter._matches_globs("indicators/ma.pine", globs) is False

    def test_filter_by_globs(self, adapter):
        """_filter_by_globs filters path list."""
        paths = [
            "strategies/a.pine",
            "strategies/b.pine",
            "readme.md",
            "src/lib.js",
        ]
        filtered = adapter._filter_by_globs(paths, ["**/*.pine"])
        assert filtered == ["strategies/a.pine", "strategies/b.pine"]


class TestGitDiffParsing:
    """Tests for git diff --name-status parsing."""

    @pytest.fixture
    def adapter(self, tmp_path):
        """Create GitAdapter with temp directory."""
        return GitAdapter(data_dir=tmp_path)

    def test_parse_simple_add(self, adapter):
        """Parse simple add (A) status."""
        output = "A\tscripts/new.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 1
        assert changes[0].path == "scripts/new.pine"
        assert changes[0].status == "A"

    def test_parse_simple_modify(self, adapter):
        """Parse simple modify (M) status."""
        output = "M\tscripts/existing.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 1
        assert changes[0].status == "M"

    def test_parse_simple_delete(self, adapter):
        """Parse simple delete (D) status."""
        output = "D\tscripts/old.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 1
        assert changes[0].status == "D"

    def test_parse_rename(self, adapter):
        """Parse rename (R100) as delete old + add new."""
        output = "R100\tscripts/old.pine\tscripts/new.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 2
        # Old path is deleted
        assert any(c.path == "scripts/old.pine" and c.status == "D" for c in changes)
        # New path is added
        assert any(c.path == "scripts/new.pine" and c.status == "A" for c in changes)

    def test_parse_copy(self, adapter):
        """Parse copy (C100) as add new."""
        output = "C100\tscripts/src.pine\tscripts/dst.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 1
        assert changes[0].path == "scripts/dst.pine"
        assert changes[0].status == "A"

    def test_parse_filters_by_glob(self, adapter):
        """Parse filters results by glob pattern."""
        output = "A\tscripts/a.pine\nA\treadme.md\nM\tscripts/b.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 2
        paths = [c.path for c in changes]
        assert "scripts/a.pine" in paths
        assert "scripts/b.pine" in paths
        assert "readme.md" not in paths

    def test_parse_rename_partial_match(self, adapter):
        """Parse rename where only one path matches glob."""
        # Rename from non-pine to pine
        output = "R100\tscripts/old.txt\tscripts/new.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 1
        assert changes[0].path == "scripts/new.pine"
        assert changes[0].status == "A"

        # Rename from pine to non-pine
        output = "R100\tscripts/old.pine\tscripts/new.txt"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 1
        assert changes[0].path == "scripts/old.pine"
        assert changes[0].status == "D"

    def test_parse_empty_output(self, adapter):
        """Parse empty output returns empty list."""
        changes = adapter._parse_diff_name_status("", ["**/*.pine"])
        assert changes == []

    def test_parse_multiple_lines(self, adapter):
        """Parse multiple lines correctly."""
        output = "A\tscripts/a.pine\nM\tscripts/b.pine\nD\tscripts/c.pine"
        changes = adapter._parse_diff_name_status(output, ["**/*.pine"])
        assert len(changes) == 3


class TestGitAdapterAsync:
    """Async tests for GitAdapter."""

    @pytest.fixture
    def adapter(self, tmp_path):
        """Create GitAdapter with temp directory."""
        return GitAdapter(data_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_ensure_clone_creates_clone(self, adapter):
        """ensure_clone clones repo if not exists."""
        with patch.object(adapter, "_run_git", new_callable=AsyncMock) as mock_git:
            mock_git.return_value = ""

            clone_path = await adapter.ensure_clone(
                "https://github.com/owner/repo",
                "owner/repo",
            )

            assert clone_path.name == "owner__repo"
            # Clone should be called
            mock_git.assert_called_once()
            call_args = mock_git.call_args[0][0]
            assert "clone" in call_args

    @pytest.mark.asyncio
    async def test_ensure_clone_skips_if_exists(self, adapter, tmp_path):
        """ensure_clone skips clone if .git exists."""
        # Create fake clone directory with .git
        clone_dir = adapter._repos_dir / "owner__repo"
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()

        with patch.object(adapter, "_run_git", new_callable=AsyncMock) as mock_git:
            clone_path = await adapter.ensure_clone(
                "https://github.com/owner/repo",
                "owner/repo",
            )

            assert clone_path == clone_dir
            # Clone should NOT be called
            mock_git.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_and_diff_incremental(self, adapter, tmp_path):
        """fetch_and_diff does incremental diff when last_seen_commit exists."""
        # Create fake clone directory
        clone_dir = adapter._repos_dir / "owner__repo"
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug="owner/repo",
            clone_path=clone_dir,
            branch="main",
            last_seen_commit="old123",
            scan_globs=["**/*.pine"],
        )

        with patch.object(adapter, "_run_git", new_callable=AsyncMock) as mock_git:
            # Mock fetch, rev-parse, diff
            mock_git.side_effect = [
                "",  # fetch
                "new456\n",  # rev-parse
                "A\tscripts/new.pine\n",  # diff
            ]

            result = await adapter.fetch_and_diff(repo)

            assert result.current_commit == "new456"
            assert result.is_full_scan is False
            assert len(result.changes) == 1
            assert result.changes[0].path == "scripts/new.pine"

    @pytest.mark.asyncio
    async def test_fetch_and_diff_full_scan(self, adapter, tmp_path):
        """fetch_and_diff does full scan when no last_seen_commit."""
        clone_dir = adapter._repos_dir / "owner__repo"
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug="owner/repo",
            clone_path=clone_dir,
            branch="main",
            last_seen_commit=None,  # No previous commit
            scan_globs=["**/*.pine"],
        )

        with patch.object(adapter, "_run_git", new_callable=AsyncMock) as mock_git:
            # Mock fetch, rev-parse, ls-tree
            mock_git.side_effect = [
                "",  # fetch
                "abc123\n",  # rev-parse
                "scripts/a.pine\nscripts/b.pine\nreadme.md\n",  # ls-tree
            ]

            result = await adapter.fetch_and_diff(repo)

            assert result.current_commit == "abc123"
            assert result.is_full_scan is True
            # Only .pine files
            assert len(result.changes) == 2
            assert all(c.status == "A" for c in result.changes)

    @pytest.mark.asyncio
    async def test_read_file_at(self, adapter, tmp_path):
        """read_file_at uses git show to read file content."""
        clone_dir = tmp_path / "repo"
        clone_dir.mkdir()

        with patch.object(adapter, "_run_git", new_callable=AsyncMock) as mock_git:
            mock_git.return_value = "//@version=5\nstrategy('Test')\n"

            content = await adapter.read_file_at(
                clone_dir, "abc123", "scripts/test.pine"
            )

            assert content == "//@version=5\nstrategy('Test')\n"
            mock_git.assert_called_once()
            call_args = mock_git.call_args[0][0]
            assert "show" in call_args
            assert "abc123:scripts/test.pine" in call_args

    @pytest.mark.asyncio
    async def test_branch_not_found_error(self, adapter, tmp_path):
        """fetch_and_diff raises BranchNotFoundError for missing branch."""
        clone_dir = adapter._repos_dir / "owner__repo"
        clone_dir.mkdir(parents=True)
        (clone_dir / ".git").mkdir()

        repo = GitRepo(
            repo_id=uuid4(),
            repo_slug="owner/repo",
            clone_path=clone_dir,
            branch="nonexistent",
            last_seen_commit=None,
            scan_globs=["**/*.pine"],
        )

        with patch.object(adapter, "_run_git", new_callable=AsyncMock) as mock_git:
            # Fetch succeeds, rev-parse fails
            mock_git.side_effect = [
                "",  # fetch
                GitCommandError(
                    ["git", "rev-parse", "origin/nonexistent"],
                    128,
                    "fatal: ambiguous argument 'origin/nonexistent'",
                ),
            ]

            with pytest.raises(BranchNotFoundError) as exc_info:
                await adapter.fetch_and_diff(repo)

            assert "nonexistent" in str(exc_info.value)


class TestGitAdapterLocking:
    """Tests for GitAdapter locking mechanism."""

    @pytest.fixture
    def adapter(self, tmp_path):
        """Create GitAdapter with temp directory."""
        return GitAdapter(data_dir=tmp_path)

    @pytest.mark.asyncio
    async def test_repo_lock_creates_lock_file(self, adapter):
        """_repo_lock creates lock file for cross-process safety."""
        async with adapter._repo_lock("owner/repo"):
            lock_path = adapter._get_file_lock_path("owner/repo")
            assert lock_path.exists()

    @pytest.mark.asyncio
    async def test_in_process_lock_serializes(self, adapter):
        """In-process async lock serializes concurrent operations."""
        execution_order = []

        async def worker(n):
            async with adapter._repo_lock("owner/repo"):
                execution_order.append(f"start-{n}")
                await asyncio.sleep(0.01)
                execution_order.append(f"end-{n}")

        # Run two workers concurrently
        await asyncio.gather(worker(1), worker(2))

        # Should be serialized: start-1, end-1, start-2, end-2
        # (or start-2, end-2, start-1, end-1)
        assert execution_order[0].startswith("start")
        assert execution_order[1].startswith("end")
        assert execution_order[2].startswith("start")
        assert execution_order[3].startswith("end")
        # Same worker should complete before other starts
        assert execution_order[0][-1] == execution_order[1][-1]


class TestPineReposMetrics:
    """Tests for Pine repos Prometheus metrics."""

    def test_pine_repos_metrics_defined(self):
        """Pine repos metrics are properly defined."""
        from app.routers.metrics import (
            PINE_REPOS_TOTAL,
            PINE_REPOS_ENABLED,
            PINE_REPOS_PULL_FAILED,
            PINE_REPOS_STALE,
            PINE_REPOS_OLDEST_SCAN_AGE_HOURS,
            PINE_REPO_SCAN_RUNS,
            PINE_REPO_SCAN_DURATION,
        )

        assert PINE_REPOS_TOTAL is not None
        assert PINE_REPOS_ENABLED is not None
        assert PINE_REPOS_PULL_FAILED is not None
        assert PINE_REPOS_STALE is not None
        assert PINE_REPOS_OLDEST_SCAN_AGE_HOURS is not None
        assert PINE_REPO_SCAN_RUNS is not None
        assert PINE_REPO_SCAN_DURATION is not None

    def test_set_pine_repos_metrics(self):
        """set_pine_repos_metrics updates gauge values."""
        from app.routers.metrics import (
            PINE_REPOS_TOTAL,
            PINE_REPOS_ENABLED,
            set_pine_repos_metrics,
        )

        set_pine_repos_metrics(
            total=10,
            enabled=8,
            pull_failed=2,
            stale=1,
            oldest_scan_age_hours=24.5,
        )

        assert PINE_REPOS_TOTAL._value._value == 10
        assert PINE_REPOS_ENABLED._value._value == 8

    def test_record_pine_repo_scan(self):
        """record_pine_repo_scan records scan metrics."""
        from app.routers.metrics import record_pine_repo_scan

        # Should not raise
        record_pine_repo_scan(
            status="success",
            duration=5.5,
            scripts_new=3,
            scripts_updated=2,
            scripts_deleted=1,
        )

        assert True
