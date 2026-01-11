"""Unit tests for BackfillRunRepository and resume semantics."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.kb_backfill import (
    BackfillRunRepository,
    BackfillRun,
    _compute_config_hash,
)


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


class TestConfigHash:
    """Test config hash computation for stable matching."""

    def test_hash_is_deterministic(self):
        """Same config should always produce same hash."""
        config = {"limit": 1000, "since": "2025-01-01"}
        hash1 = _compute_config_hash(config)
        hash2 = _compute_config_hash(config)
        assert hash1 == hash2

    def test_hash_ignores_key_ordering(self):
        """Different key ordering should produce same hash."""
        config1 = {"a": 1, "b": 2, "c": 3}
        config2 = {"c": 3, "a": 1, "b": 2}
        config3 = {"b": 2, "c": 3, "a": 1}

        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)
        hash3 = _compute_config_hash(config3)

        assert hash1 == hash2 == hash3

    def test_hash_is_sha256_hex(self):
        """Hash should be 64-character hex string (SHA256)."""
        config = {"test": "value"}
        config_hash = _compute_config_hash(config)

        assert len(config_hash) == 64
        assert all(c in "0123456789abcdef" for c in config_hash)

    def test_different_configs_produce_different_hashes(self):
        """Different configs should produce different hashes."""
        config1 = {"limit": 1000}
        config2 = {"limit": 2000}

        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)

        assert hash1 != hash2

    def test_empty_config_has_stable_hash(self):
        """Empty config should have a stable, predictable hash."""
        config = {}
        config_hash = _compute_config_hash(config)
        # sha256 of "{}" (empty JSON object with canonical formatting)
        expected = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        assert config_hash == expected


class TestNewestWinsResumeSemantics:
    """Test that the newest matching run wins when resuming."""

    @pytest.mark.asyncio
    async def test_find_resumable_returns_newest_when_multiple_match(
        self, repo, mock_pool
    ):
        """When multiple failed runs match config, newest (by started_at) wins."""
        workspace_id = uuid4()
        config = {"limit": 1000}

        # Simulate finding the newest run
        # The query should ORDER BY started_at DESC LIMIT 1
        newer_run_id = uuid4()
        newer_time = datetime.now()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": newer_run_id,
                "workspace_id": workspace_id,
                "backfill_type": "candidacy",
                "status": "failed",
                "started_at": newer_time,
                "completed_at": newer_time,
                "processed_count": 500,  # More progress than older run
                "skipped_count": 100,
                "error_count": 1,
                "last_processed_cursor": "newer_cursor",
                "config": json.dumps(config),
                "error": "Timeout",
                "dry_run": False,
            }
        )
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        run = await repo.find_resumable(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config=config,
        )

        # Verify the query uses ORDER BY started_at DESC
        call_args = mock_conn.fetchrow.call_args[0]
        assert "ORDER BY started_at DESC" in call_args[0]
        assert "LIMIT 1" in call_args[0]

        # Verify we got the newer run
        assert run is not None
        assert run.id == newer_run_id
        assert run.last_processed_cursor == "newer_cursor"
        assert run.processed_count == 500

    @pytest.mark.asyncio
    async def test_find_resumable_uses_config_hash(self, repo, mock_pool):
        """Find resumable should match on config_hash for indexed lookup."""
        workspace_id = uuid4()
        config = {"limit": 1000, "since": "2025-01-01"}

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.find_resumable(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config=config,
        )

        # Verify the query uses config_hash with fallback
        call_args = mock_conn.fetchrow.call_args[0]
        query = call_args[0]
        assert "config_hash = $3" in query
        assert "config_hash IS NULL AND config = $4" in query

    @pytest.mark.asyncio
    async def test_create_stores_config_hash(self, repo, mock_pool):
        """Create should store config_hash for fast matching."""
        workspace_id = uuid4()
        run_id = uuid4()
        now = datetime.now()
        config = {"limit": 1000}

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": run_id, "started_at": now})
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.create(
            workspace_id=workspace_id,
            backfill_type="candidacy",
            config=config,
            dry_run=False,
        )

        # Verify the INSERT includes config_hash
        call_args = mock_conn.fetchrow.call_args[0]
        query = call_args[0]
        assert "config_hash" in query

        # Verify config_hash is passed as argument (should be arg $4)
        expected_hash = _compute_config_hash(config)
        assert call_args[4] == expected_hash
