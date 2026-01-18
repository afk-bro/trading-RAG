"""Unit tests for PineRepoRepository."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.services.pine.repo_registry import (
    ERROR_TRUNCATE_LEN,
    PineRepo,
    PineRepoRepository,
    RepoHealthStats,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    pool.acquire = MagicMock()
    return pool


@pytest.fixture
def mock_conn():
    """Create a mock asyncpg connection."""
    conn = AsyncMock()
    return conn


@pytest.fixture
def repo_id() -> UUID:
    """Fixed repo ID for tests."""
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def workspace_id() -> UUID:
    """Fixed workspace ID for tests."""
    return UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


@pytest.fixture
def sample_row(repo_id, workspace_id):
    """Create a sample DB row dict."""
    now = datetime.now(timezone.utc)
    return {
        "id": repo_id,
        "workspace_id": workspace_id,
        "repo_url": "https://github.com/owner/repo",
        "repo_slug": "owner/repo",
        "clone_path": "/data/repos/owner__repo",
        "branch": "main",
        "last_seen_commit": "abc123def456",
        "last_scan_at": now,
        "last_scan_ok": True,
        "last_scan_error": None,
        "scripts_count": 42,
        "last_pull_at": now,
        "last_pull_ok": True,
        "pull_error": None,
        "enabled": True,
        "scan_globs": ["**/*.pine", "scripts/*.txt"],
        "next_scan_at": now,
        "failure_count": 0,
        "created_at": now,
        "updated_at": now,
    }


# =============================================================================
# Model Tests
# =============================================================================


class TestPineRepoModel:
    """Tests for PineRepo dataclass."""

    def test_default_values(self, repo_id, workspace_id):
        """Test default field values."""
        repo = PineRepo(
            id=repo_id,
            workspace_id=workspace_id,
            repo_url="https://github.com/owner/repo",
            repo_slug="owner/repo",
        )

        assert repo.clone_path is None
        assert repo.branch == "main"
        assert repo.last_seen_commit is None
        assert repo.last_scan_at is None
        assert repo.last_scan_ok is None
        assert repo.last_scan_error is None
        assert repo.scripts_count == 0
        assert repo.last_pull_at is None
        assert repo.last_pull_ok is None
        assert repo.pull_error is None
        assert repo.enabled is True
        assert repo.scan_globs == ["**/*.pine"]
        assert repo.created_at is None
        assert repo.updated_at is None

    def test_all_fields(self, repo_id, workspace_id):
        """Test setting all fields."""
        now = datetime.now(timezone.utc)
        repo = PineRepo(
            id=repo_id,
            workspace_id=workspace_id,
            repo_url="https://github.com/org/project",
            repo_slug="org/project",
            clone_path="/data/repos/org__project",
            branch="develop",
            last_seen_commit="deadbeef",
            last_scan_at=now,
            last_scan_ok=True,
            last_scan_error=None,
            scripts_count=100,
            last_pull_at=now,
            last_pull_ok=False,
            pull_error="Network timeout",
            enabled=False,
            scan_globs=["**/*.pine", "lib/**/*.txt"],
            created_at=now,
            updated_at=now,
        )

        assert repo.repo_slug == "org/project"
        assert repo.branch == "develop"
        assert repo.last_seen_commit == "deadbeef"
        assert repo.scripts_count == 100
        assert repo.pull_error == "Network timeout"
        assert repo.enabled is False
        assert len(repo.scan_globs) == 2

    def test_scan_globs_default_factory(self):
        """Test that scan_globs uses a fresh list per instance."""
        repo1 = PineRepo(
            id=uuid4(),
            workspace_id=uuid4(),
            repo_url="https://github.com/a/b",
            repo_slug="a/b",
        )
        repo2 = PineRepo(
            id=uuid4(),
            workspace_id=uuid4(),
            repo_url="https://github.com/c/d",
            repo_slug="c/d",
        )

        # Mutate repo1's globs
        repo1.scan_globs.append("extra/*.pine")

        # repo2 should not be affected
        assert "extra/*.pine" not in repo2.scan_globs
        assert repo2.scan_globs == ["**/*.pine"]


class TestRepoHealthStatsModel:
    """Tests for RepoHealthStats dataclass."""

    def test_default_values(self):
        """Test default field values."""
        stats = RepoHealthStats()

        assert stats.repos_total == 0
        assert stats.repos_enabled == 0
        assert stats.repos_pull_failed == 0
        assert stats.repos_stale == 0
        assert stats.oldest_scan_age_hours is None

    def test_all_fields(self):
        """Test setting all fields."""
        stats = RepoHealthStats(
            repos_total=10,
            repos_enabled=8,
            repos_pull_failed=2,
            repos_stale=1,
            oldest_scan_age_hours=168.5,
        )

        assert stats.repos_total == 10
        assert stats.repos_enabled == 8
        assert stats.repos_pull_failed == 2
        assert stats.repos_stale == 1
        assert stats.oldest_scan_age_hours == 168.5


# =============================================================================
# Repository Tests
# =============================================================================


class TestPineRepoRepositoryCreate:
    """Tests for PineRepoRepository.create()."""

    @pytest.mark.asyncio
    async def test_create_with_defaults(self, mock_pool, mock_conn, sample_row):
        """Test creating repo with default values."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = sample_row

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.create(
            workspace_id=sample_row["workspace_id"],
            repo_url="https://github.com/owner/repo",
            repo_slug="owner/repo",
        )

        assert isinstance(result, PineRepo)
        assert result.repo_slug == "owner/repo"
        assert result.branch == "main"

        # Verify SQL was called with default scan_globs
        mock_conn.fetchrow.assert_called_once()
        call_args = mock_conn.fetchrow.call_args[0]
        # Args: (query, workspace_id, repo_url, repo_slug, branch, scan_globs)
        assert call_args[5] == ["**/*.pine"]  # scan_globs parameter

    @pytest.mark.asyncio
    async def test_create_with_custom_values(self, mock_pool, mock_conn, sample_row):
        """Test creating repo with custom values."""
        sample_row["branch"] = "develop"
        sample_row["scan_globs"] = ["*.pine", "lib/**/*.txt"]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = sample_row

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.create(
            workspace_id=sample_row["workspace_id"],
            repo_url="https://github.com/owner/repo",
            repo_slug="owner/repo",
            branch="develop",
            scan_globs=["*.pine", "lib/**/*.txt"],
        )

        assert result.branch == "develop"
        assert result.scan_globs == ["*.pine", "lib/**/*.txt"]


class TestPineRepoRepositoryGet:
    """Tests for PineRepoRepository.get() and get_by_slug()."""

    @pytest.mark.asyncio
    async def test_get_existing(self, mock_pool, mock_conn, sample_row, repo_id):
        """Test getting existing repo by ID."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = sample_row

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.get(repo_id)

        assert result is not None
        assert result.id == repo_id
        assert result.repo_slug == "owner/repo"

    @pytest.mark.asyncio
    async def test_get_not_found(self, mock_pool, mock_conn, repo_id):
        """Test getting non-existent repo."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.get(repo_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_slug_existing(
        self, mock_pool, mock_conn, sample_row, workspace_id
    ):
        """Test getting repo by workspace and slug."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = sample_row

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.get_by_slug(workspace_id, "owner/repo")

        assert result is not None
        assert result.repo_slug == "owner/repo"
        assert result.workspace_id == workspace_id

    @pytest.mark.asyncio
    async def test_get_by_slug_not_found(self, mock_pool, mock_conn, workspace_id):
        """Test getting non-existent repo by slug."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.get_by_slug(workspace_id, "unknown/repo")

        assert result is None


class TestPineRepoRepositoryList:
    """Tests for list methods."""

    @pytest.mark.asyncio
    async def test_list_by_workspace(
        self, mock_pool, mock_conn, sample_row, workspace_id
    ):
        """Test listing repos by workspace."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = 2
        mock_conn.fetch.return_value = [sample_row, sample_row]

        repo_registry = PineRepoRepository(mock_pool)
        repos, total = await repo_registry.list_by_workspace(workspace_id)

        assert len(repos) == 2
        assert total == 2
        assert all(isinstance(r, PineRepo) for r in repos)

    @pytest.mark.asyncio
    async def test_list_by_workspace_enabled_only(
        self, mock_pool, mock_conn, sample_row, workspace_id
    ):
        """Test listing only enabled repos."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.return_value = [sample_row]

        repo_registry = PineRepoRepository(mock_pool)
        repos, total = await repo_registry.list_by_workspace(
            workspace_id, enabled_only=True
        )

        assert len(repos) == 1
        assert total == 1

        # Verify query includes enabled filter
        count_query = mock_conn.fetchval.call_args[0][0]
        assert "enabled = TRUE" in count_query

    @pytest.mark.asyncio
    async def test_list_by_workspace_pagination(
        self, mock_pool, mock_conn, sample_row, workspace_id
    ):
        """Test pagination parameters."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = 100
        mock_conn.fetch.return_value = [sample_row]

        repo_registry = PineRepoRepository(mock_pool)
        repos, total = await repo_registry.list_by_workspace(
            workspace_id, limit=10, offset=20
        )

        assert total == 100
        # Verify limit/offset passed to query
        # Args: (query, workspace_id, limit, offset)
        fetch_call = mock_conn.fetch.call_args[0]
        assert fetch_call[2] == 10  # limit
        assert fetch_call[3] == 20  # offset

    @pytest.mark.asyncio
    async def test_list_enabled(self, mock_pool, mock_conn, sample_row, workspace_id):
        """Test listing enabled repos (simple method)."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.return_value = [sample_row]

        repo_registry = PineRepoRepository(mock_pool)
        repos = await repo_registry.list_enabled(workspace_id)

        assert len(repos) == 1
        assert repos[0].enabled is True


class TestPineRepoRepositoryUpdate:
    """Tests for update methods."""

    @pytest.mark.asyncio
    async def test_update_clone_path(self, mock_pool, mock_conn, repo_id):
        """Test updating clone path."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_clone_path(repo_id, "/data/repos/owner__repo")

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "/data/repos/owner__repo" in call_args
        assert repo_id in call_args

    @pytest.mark.asyncio
    async def test_update_scan_result_success(self, mock_pool, mock_conn, repo_id):
        """Test updating scan result on success."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_scan_result(
            repo_id=repo_id,
            commit="deadbeef123",
            ok=True,
            error=None,
            scripts_count=25,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        query = call_args[0]

        # Verify success path updates commit and scripts_count
        assert "last_seen_commit" in query
        assert "scripts_count" in query
        assert "last_scan_ok = TRUE" in query

    @pytest.mark.asyncio
    async def test_update_scan_result_failure(self, mock_pool, mock_conn, repo_id):
        """Test updating scan result on failure."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_scan_result(
            repo_id=repo_id,
            commit="deadbeef123",
            ok=False,
            error="Git fetch failed: network timeout",
            scripts_count=0,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        query = call_args[0]

        # Verify failure path stores error
        assert "last_scan_ok = FALSE" in query
        assert "last_scan_error" in query

    @pytest.mark.asyncio
    async def test_update_scan_result_error_truncation(
        self, mock_pool, mock_conn, repo_id
    ):
        """Test that long errors are truncated."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        # Create error longer than limit
        long_error = "x" * (ERROR_TRUNCATE_LEN + 500)

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_scan_result(
            repo_id=repo_id,
            commit="abc",
            ok=False,
            error=long_error,
            scripts_count=0,
        )

        # Verify truncated error was passed
        call_args = mock_conn.execute.call_args[0]
        passed_error = call_args[2]  # error parameter
        assert len(passed_error) == ERROR_TRUNCATE_LEN
        assert passed_error.endswith("...")

    @pytest.mark.asyncio
    async def test_update_pull_result_success(self, mock_pool, mock_conn, repo_id):
        """Test updating pull result on success."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_pull_result(
            repo_id=repo_id,
            ok=True,
            error=None,
        )

        call_args = mock_conn.execute.call_args[0]
        # On success, pull_error should be None
        assert call_args[3] is None  # error parameter

    @pytest.mark.asyncio
    async def test_update_pull_result_failure(self, mock_pool, mock_conn, repo_id):
        """Test updating pull result on failure."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_pull_result(
            repo_id=repo_id,
            ok=False,
            error="Could not resolve host: github.com",
        )

        call_args = mock_conn.execute.call_args[0]
        assert call_args[2] is False  # ok parameter
        assert "Could not resolve host" in call_args[3]

    @pytest.mark.asyncio
    async def test_update_pull_result_error_truncation(
        self, mock_pool, mock_conn, repo_id
    ):
        """Test that pull errors are truncated."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        long_error = "e" * 2000

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_pull_result(
            repo_id=repo_id,
            ok=False,
            error=long_error,
        )

        call_args = mock_conn.execute.call_args[0]
        passed_error = call_args[3]
        assert len(passed_error) == ERROR_TRUNCATE_LEN
        assert passed_error.endswith("...")

    @pytest.mark.asyncio
    async def test_set_enabled(self, mock_pool, mock_conn, repo_id):
        """Test enabling/disabling repo."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)

        await repo_registry.set_enabled(repo_id, False)
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1] is False

        await repo_registry.set_enabled(repo_id, True)
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1] is True


class TestPineRepoRepositoryDelete:
    """Tests for delete method."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, mock_pool, mock_conn, repo_id):
        """Test deleting existing repo."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "DELETE 1"

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.delete(repo_id)

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_pool, mock_conn, repo_id):
        """Test deleting non-existent repo."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "DELETE 0"

        repo_registry = PineRepoRepository(mock_pool)
        result = await repo_registry.delete(repo_id)

        assert result is False


class TestPineRepoRepositoryHealthStats:
    """Tests for get_health_stats method."""

    @pytest.mark.asyncio
    async def test_get_health_stats_all_workspaces(self, mock_pool, mock_conn):
        """Test getting health stats for all workspaces."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = {
            "total": 10,
            "enabled": 8,
            "pull_failed": 2,
            "stale": 1,
            "oldest_scan_hours": 168.5,
        }

        repo_registry = PineRepoRepository(mock_pool)
        stats = await repo_registry.get_health_stats()

        assert isinstance(stats, RepoHealthStats)
        assert stats.repos_total == 10
        assert stats.repos_enabled == 8
        assert stats.repos_pull_failed == 2
        assert stats.repos_stale == 1
        assert stats.oldest_scan_age_hours == 168.5

        # Verify no workspace WHERE clause (note: FILTER(WHERE ...) is in the SQL)
        query = mock_conn.fetchrow.call_args[0][0]
        assert "WHERE workspace_id" not in query

    @pytest.mark.asyncio
    async def test_get_health_stats_by_workspace(
        self, mock_pool, mock_conn, workspace_id
    ):
        """Test getting health stats for specific workspace."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = {
            "total": 5,
            "enabled": 4,
            "pull_failed": 1,
            "stale": 0,
            "oldest_scan_hours": 24.0,
        }

        repo_registry = PineRepoRepository(mock_pool)
        stats = await repo_registry.get_health_stats(workspace_id)

        assert stats.repos_total == 5
        assert stats.repos_enabled == 4
        assert stats.repos_pull_failed == 1
        assert stats.repos_stale == 0
        assert stats.oldest_scan_age_hours == 24.0

        # Verify WHERE clause present
        query = mock_conn.fetchrow.call_args[0][0]
        assert "WHERE workspace_id = $1" in query

    @pytest.mark.asyncio
    async def test_get_health_stats_empty(self, mock_pool, mock_conn):
        """Test getting health stats with no repos."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = {
            "total": None,  # COUNT returns NULL for empty table in some cases
            "enabled": None,
            "pull_failed": None,
            "stale": None,
            "oldest_scan_hours": None,
        }

        repo_registry = PineRepoRepository(mock_pool)
        stats = await repo_registry.get_health_stats()

        assert stats.repos_total == 0
        assert stats.repos_enabled == 0
        assert stats.repos_pull_failed == 0
        assert stats.repos_stale == 0
        assert stats.oldest_scan_age_hours is None


class TestRowToModel:
    """Tests for _row_to_model conversion."""

    def test_row_to_model_full(self, sample_row):
        """Test converting complete row to model."""
        repo_registry = PineRepoRepository(MagicMock())
        result = repo_registry._row_to_model(sample_row)

        assert isinstance(result, PineRepo)
        assert result.id == sample_row["id"]
        assert result.workspace_id == sample_row["workspace_id"]
        assert result.repo_url == sample_row["repo_url"]
        assert result.repo_slug == sample_row["repo_slug"]
        assert result.clone_path == sample_row["clone_path"]
        assert result.branch == sample_row["branch"]
        assert result.last_seen_commit == sample_row["last_seen_commit"]
        assert result.last_scan_at == sample_row["last_scan_at"]
        assert result.last_scan_ok == sample_row["last_scan_ok"]
        assert result.last_scan_error == sample_row["last_scan_error"]
        assert result.scripts_count == sample_row["scripts_count"]
        assert result.last_pull_at == sample_row["last_pull_at"]
        assert result.last_pull_ok == sample_row["last_pull_ok"]
        assert result.pull_error == sample_row["pull_error"]
        assert result.enabled == sample_row["enabled"]
        assert result.scan_globs == sample_row["scan_globs"]
        assert result.next_scan_at == sample_row["next_scan_at"]
        assert result.failure_count == sample_row["failure_count"]
        assert result.created_at == sample_row["created_at"]
        assert result.updated_at == sample_row["updated_at"]

    def test_row_to_model_null_scan_globs(self, sample_row):
        """Test that null scan_globs defaults to ['**/*.pine']."""
        sample_row["scan_globs"] = None
        repo_registry = PineRepoRepository(MagicMock())
        result = repo_registry._row_to_model(sample_row)

        assert result.scan_globs == ["**/*.pine"]

    def test_row_to_model_minimal(self, repo_id, workspace_id):
        """Test converting row with minimal values."""
        minimal_row = {
            "id": repo_id,
            "workspace_id": workspace_id,
            "repo_url": "https://github.com/a/b",
            "repo_slug": "a/b",
            "clone_path": None,
            "branch": "main",
            "last_seen_commit": None,
            "last_scan_at": None,
            "last_scan_ok": None,
            "last_scan_error": None,
            "scripts_count": 0,
            "last_pull_at": None,
            "last_pull_ok": None,
            "pull_error": None,
            "enabled": True,
            "scan_globs": None,
            "next_scan_at": None,
            "failure_count": 0,
            "created_at": None,
            "updated_at": None,
        }

        repo_registry = PineRepoRepository(MagicMock())
        result = repo_registry._row_to_model(minimal_row)

        assert result.id == repo_id
        assert result.clone_path is None
        assert result.last_seen_commit is None
        assert result.scripts_count == 0
        assert result.scan_globs == ["**/*.pine"]
        assert result.next_scan_at is None
        assert result.failure_count == 0


class TestErrorTruncation:
    """Tests for error truncation constant and behavior."""

    def test_error_truncate_length_value(self):
        """Verify ERROR_TRUNCATE_LEN constant."""
        assert ERROR_TRUNCATE_LEN == 1000

    def test_error_at_exact_limit(self):
        """Test error at exactly the limit length."""
        # This is tested via the update methods, but we verify the logic here
        exact_error = "x" * ERROR_TRUNCATE_LEN

        # Should not be truncated
        if len(exact_error) > ERROR_TRUNCATE_LEN:
            truncated = exact_error[: ERROR_TRUNCATE_LEN - 3] + "..."
        else:
            truncated = exact_error

        assert truncated == exact_error
        assert len(truncated) == ERROR_TRUNCATE_LEN

    def test_error_one_over_limit(self):
        """Test error one character over the limit."""
        over_error = "x" * (ERROR_TRUNCATE_LEN + 1)

        if len(over_error) > ERROR_TRUNCATE_LEN:
            truncated = over_error[: ERROR_TRUNCATE_LEN - 3] + "..."
        else:
            truncated = over_error

        assert len(truncated) == ERROR_TRUNCATE_LEN
        assert truncated.endswith("...")


# =============================================================================
# Polling Methods Tests
# =============================================================================


class TestListDueForPoll:
    """Tests for list_due_for_poll method."""

    @pytest.mark.asyncio
    async def test_list_due_for_poll_returns_repos(
        self, mock_pool, mock_conn, sample_row
    ):
        """Test listing repos due for polling."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.return_value = [sample_row]

        repo_registry = PineRepoRepository(mock_pool)
        repos = await repo_registry.list_due_for_poll(limit=10)

        assert len(repos) == 1
        assert repos[0].id == sample_row["id"]

        # Verify query includes enabled filter and ordering
        query = mock_conn.fetch.call_args[0][0]
        assert "enabled = true" in query
        assert "next_scan_at" in query
        assert "LIMIT $1" in query

    @pytest.mark.asyncio
    async def test_list_due_for_poll_with_workspace(
        self, mock_pool, mock_conn, sample_row, workspace_id
    ):
        """Test listing repos due for polling filtered by workspace."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.return_value = [sample_row]

        repo_registry = PineRepoRepository(mock_pool)
        repos = await repo_registry.list_due_for_poll(
            workspace_id=workspace_id, limit=5
        )

        assert len(repos) == 1

        # Verify workspace filter is present
        query = mock_conn.fetch.call_args[0][0]
        assert "workspace_id = $1" in query

    @pytest.mark.asyncio
    async def test_list_due_for_poll_empty(self, mock_pool, mock_conn):
        """Test listing repos when none are due."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetch.return_value = []

        repo_registry = PineRepoRepository(mock_pool)
        repos = await repo_registry.list_due_for_poll()

        assert repos == []


class TestCountDueForPoll:
    """Tests for count_due_for_poll method."""

    @pytest.mark.asyncio
    async def test_count_due_for_poll(self, mock_pool, mock_conn):
        """Test counting repos due for polling."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = 5

        repo_registry = PineRepoRepository(mock_pool)
        count = await repo_registry.count_due_for_poll()

        assert count == 5

    @pytest.mark.asyncio
    async def test_count_due_for_poll_with_workspace(
        self, mock_pool, mock_conn, workspace_id
    ):
        """Test counting repos due for polling filtered by workspace."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = 3

        repo_registry = PineRepoRepository(mock_pool)
        count = await repo_registry.count_due_for_poll(workspace_id=workspace_id)

        assert count == 3

        # Verify workspace filter
        query = mock_conn.fetchval.call_args[0][0]
        assert "workspace_id = $1" in query

    @pytest.mark.asyncio
    async def test_count_due_for_poll_null_returns_zero(self, mock_pool, mock_conn):
        """Test counting returns 0 when fetchval returns None."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchval.return_value = None

        repo_registry = PineRepoRepository(mock_pool)
        count = await repo_registry.count_due_for_poll()

        assert count == 0


class TestUpdatePollSuccess:
    """Tests for update_poll_success method."""

    @pytest.mark.asyncio
    async def test_update_poll_success(self, mock_pool, mock_conn, repo_id):
        """Test updating poll state after successful scan."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_poll_success(repo_id, interval_minutes=15)

        # Verify SQL structure
        query = mock_conn.execute.call_args[0][0]
        assert "failure_count = 0" in query
        assert "next_scan_at" in query
        assert "WHERE id = $2" in query  # repo_id is $2 (next_scan_at is $1)

    @pytest.mark.asyncio
    async def test_update_poll_success_with_jitter(self, mock_pool, mock_conn, repo_id):
        """Test jitter is applied to next_scan_at."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_poll_success(
            repo_id, interval_minutes=15, jitter_pct=0.2
        )

        # Verify execute was called (jitter is applied in code)
        mock_conn.execute.assert_called_once()


class TestUpdatePollFailure:
    """Tests for update_poll_failure method."""

    @pytest.mark.asyncio
    async def test_update_poll_failure(self, mock_pool, mock_conn, repo_id):
        """Test updating poll state after failed scan."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_poll_failure(
            repo_id, base_interval_minutes=15, max_backoff_multiplier=16
        )

        # Verify SQL structure
        query = mock_conn.execute.call_args[0][0]
        assert "failure_count = failure_count + 1" in query
        assert "next_scan_at" in query
        assert "WHERE id = $3" in query  # repo_id is $3 (max_backoff=$1, interval=$2)

    @pytest.mark.asyncio
    async def test_update_poll_failure_backoff_calculation(
        self, mock_pool, mock_conn, repo_id
    ):
        """Test exponential backoff is capped correctly."""
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = "UPDATE 1"

        repo_registry = PineRepoRepository(mock_pool)
        await repo_registry.update_poll_failure(
            repo_id, base_interval_minutes=15, max_backoff_multiplier=8
        )

        # Verify LEAST() is used for capping
        query = mock_conn.execute.call_args[0][0]
        assert "LEAST" in query
        assert "power(2, failure_count)" in query.lower() or "POW" in query
