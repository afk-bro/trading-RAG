"""Unit tests for BackfillRunRepository and resume semantics."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.kb_backfill import BackfillRunRepository, BackfillRun


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


@pytest.fixture
def repo(mock_pool):
    """Create a BackfillRunRepository with mock pool."""
    return BackfillRunRepository(mock_pool)


class TestBackfillRunCreate:
    """Test creating backfill runs."""

    @pytest.mark.asyncio
    async def test_create_returns_run_with_id(self, repo, mock_pool):
        """Create should return a BackfillRun with generated ID."""
        workspace_id = uuid4()
        run_id = uuid4()
        now = datetime.now()

        # Mock the connection context manager and fetchrow
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": run_id, "started_at": now})
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run = await repo.create(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config={"limit": 1000},
            dry_run=False,
        )

        assert run.id == run_id
        assert run.workspace_id == workspace_id
        assert run.backfill_type == "candidacy"
        assert run.status == "running"
        assert run.config == {"limit": 1000}
        assert run.dry_run is False

    @pytest.mark.asyncio
    async def test_create_dry_run_sets_flag(self, repo, mock_pool):
        """Dry-run should set the dry_run flag to True."""
        workspace_id = uuid4()
        run_id = uuid4()
        now = datetime.now()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": run_id, "started_at": now})
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run = await repo.create(
            workspace_id=workspace_id,
            backfill_type="regime",
            config={},
            dry_run=True,
        )

        assert run.dry_run is True

    @pytest.mark.asyncio
    async def test_create_config_serialized_as_json(self, repo, mock_pool):
        """Config dict should be serialized as JSON for storage."""
        workspace_id = uuid4()
        run_id = uuid4()
        now = datetime.now()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": run_id, "started_at": now})
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        config = {"since": "2025-01-01", "limit": 5000, "experiment_type": "sweep"}

        await repo.create(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config=config,
            dry_run=False,
        )

        # Verify JSON serialization in the call
        call_args = mock_conn.fetchrow.call_args
        # Config is serialized as JSON - check that one of the args is the serialized config
        # Args: query, workspace_id, backfill_type, json_config, dry_run
        json_config_arg = call_args[0][3]
        assert json.loads(json_config_arg) == config


class TestBackfillRunUpdateProgress:
    """Test updating progress during backfill."""

    @pytest.mark.asyncio
    async def test_update_progress_sets_counts_and_cursor(self, repo, mock_pool):
        """Update progress should set all counters and cursor."""
        run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_progress(
            run_id=run_id,
            processed_count=100,
            skipped_count=50,
            error_count=5,
            last_processed_cursor="abc123",
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1] == run_id
        assert call_args[2] == 100  # processed_count
        assert call_args[3] == 50  # skipped_count
        assert call_args[4] == 5  # error_count
        assert call_args[5] == "abc123"  # cursor


class TestBackfillRunComplete:
    """Test marking runs as completed."""

    @pytest.mark.asyncio
    async def test_complete_sets_status_and_counts(self, repo, mock_pool):
        """Complete should set status to completed with final counts."""
        run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.complete(
            run_id=run_id,
            processed_count=500,
            skipped_count=100,
            error_count=10,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        # Query should contain 'completed' status
        assert "status = 'completed'" in call_args[0]


class TestBackfillRunFail:
    """Test marking runs as failed."""

    @pytest.mark.asyncio
    async def test_fail_sets_status_and_error(self, repo, mock_pool):
        """Fail should set status to failed with error message."""
        run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.fail(
            run_id=run_id,
            error="Connection timeout",
            processed_count=50,
            skipped_count=10,
            error_count=1,
            last_processed_cursor="cursor123",
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        # Query should contain 'failed' status
        assert "status = 'failed'" in call_args[0]
        assert call_args[2] == "Connection timeout"


class TestBackfillRunFindResumable:
    """Test finding resumable runs."""

    @pytest.mark.asyncio
    async def test_find_resumable_returns_matching_run(self, repo, mock_pool):
        """Find resumable should return run with matching config."""
        workspace_id = uuid4()
        run_id = uuid4()
        now = datetime.now()
        config = {"limit": 1000}

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": run_id,
                "workspace_id": workspace_id,
                "backfill_type": "candidacy",
                "status": "failed",
                "started_at": now,
                "completed_at": now,
                "processed_count": 100,
                "skipped_count": 50,
                "error_count": 1,
                "last_processed_cursor": "abc123",
                "config": json.dumps(config),
                "error": "Connection reset",
                "dry_run": False,
            }
        )
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run = await repo.find_resumable(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config=config,
        )

        assert run is not None
        assert run.id == run_id
        assert run.status == "failed"
        assert run.last_processed_cursor == "abc123"
        assert run.processed_count == 100

    @pytest.mark.asyncio
    async def test_find_resumable_returns_none_when_no_match(self, repo, mock_pool):
        """Find resumable should return None when no matching run exists."""
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run = await repo.find_resumable(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config={"limit": 1000},
        )

        assert run is None

    @pytest.mark.asyncio
    async def test_find_resumable_requires_cursor(self, repo, mock_pool):
        """Find resumable query should require last_processed_cursor IS NOT NULL."""
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.find_resumable(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config={},
        )

        # Check that the query includes the cursor requirement
        call_args = mock_conn.fetchrow.call_args[0]
        assert "last_processed_cursor IS NOT NULL" in call_args[0]


class TestBackfillRunGetLatest:
    """Test getting the latest run."""

    @pytest.mark.asyncio
    async def test_get_latest_returns_most_recent(self, repo, mock_pool):
        """Get latest should return the most recent run."""
        workspace_id = uuid4()
        run_id = uuid4()
        now = datetime.now()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": run_id,
                "workspace_id": workspace_id,
                "backfill_type": "regime",
                "status": "completed",
                "started_at": now,
                "completed_at": now,
                "processed_count": 200,
                "skipped_count": 0,
                "error_count": 0,
                "last_processed_cursor": None,
                "config": "{}",
                "error": None,
                "dry_run": False,
            }
        )
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run = await repo.get_latest(
            workspace_id=workspace_id,
            backfill_type="regime",
        )

        assert run is not None
        assert run.id == run_id
        assert run.status == "completed"


class TestBackfillRunListRecent:
    """Test listing recent runs."""

    @pytest.mark.asyncio
    async def test_list_recent_returns_runs(self, repo, mock_pool):
        """List recent should return runs ordered by started_at."""
        workspace_id = uuid4()
        now = datetime.now()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": uuid4(),
                    "workspace_id": workspace_id,
                    "backfill_type": "candidacy",
                    "status": "completed",
                    "started_at": now,
                    "completed_at": now,
                    "processed_count": 100,
                    "skipped_count": 50,
                    "error_count": 0,
                    "last_processed_cursor": None,
                    "config": "{}",
                    "error": None,
                    "dry_run": False,
                },
                {
                    "id": uuid4(),
                    "workspace_id": workspace_id,
                    "backfill_type": "regime",
                    "status": "running",
                    "started_at": now,
                    "completed_at": None,
                    "processed_count": 10,
                    "skipped_count": 0,
                    "error_count": 0,
                    "last_processed_cursor": "cursor",
                    "config": "{}",
                    "error": None,
                    "dry_run": True,
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        runs = await repo.list_recent(workspace_id=workspace_id, limit=10)

        assert len(runs) == 2
        assert runs[0].backfill_type == "candidacy"
        assert runs[1].backfill_type == "regime"
        assert runs[1].dry_run is True


class TestBackfillRunDataclass:
    """Test BackfillRun dataclass."""

    def test_backfill_run_defaults(self):
        """BackfillRun should have sensible defaults."""
        run = BackfillRun(
            id=uuid4(),
            workspace_id=uuid4(),
            backfill_type="candidacy",
            status="running",
            started_at=datetime.now(),
        )

        assert run.completed_at is None
        assert run.processed_count == 0
        assert run.skipped_count == 0
        assert run.error_count == 0
        assert run.last_processed_cursor is None
        assert run.config == {}
        assert run.error is None
        assert run.dry_run is False


class TestRowToRunConversion:
    """Test converting database rows to BackfillRun."""

    def test_row_to_run_parses_json_string_config(self, repo):
        """Config as JSON string should be parsed to dict."""
        row = {
            "id": uuid4(),
            "workspace_id": uuid4(),
            "backfill_type": "candidacy",
            "status": "completed",
            "started_at": datetime.now(),
            "completed_at": datetime.now(),
            "processed_count": 100,
            "skipped_count": 50,
            "error_count": 0,
            "last_processed_cursor": None,
            "config": '{"limit": 1000, "since": "2025-01-01"}',
            "error": None,
            "dry_run": False,
        }

        run = repo._row_to_run(row)

        assert run.config == {"limit": 1000, "since": "2025-01-01"}

    def test_row_to_run_handles_dict_config(self, repo):
        """Config as dict (from asyncpg JSONB) should be passed through."""
        config = {"limit": 500}
        row = {
            "id": uuid4(),
            "workspace_id": uuid4(),
            "backfill_type": "regime",
            "status": "running",
            "started_at": datetime.now(),
            "completed_at": None,
            "processed_count": 0,
            "skipped_count": 0,
            "error_count": 0,
            "last_processed_cursor": None,
            "config": config,  # Already a dict
            "error": None,
            "dry_run": True,
        }

        run = repo._row_to_run(row)

        assert run.config == config
