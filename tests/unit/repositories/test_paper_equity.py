"""Unit tests for paper equity snapshots repository."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.paper_equity import (
    EquitySnapshot,
    PaperEquityRepository,
    compute_inputs_hash,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_snapshot_row():
    """Sample database row for an equity snapshot."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "strategy_version_id": uuid4(),
        "snapshot_ts": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        "computed_at": datetime(2024, 1, 15, 10, 30, 5, tzinfo=timezone.utc),
        "equity": 105000.0,
        "cash": 50000.0,
        "positions_value": 55000.0,
        "realized_pnl": 5000.0,
        "inputs_hash": "a" * 64,
    }


@pytest.fixture
def sample_snapshot_row_no_version():
    """Sample row with null strategy_version_id (workspace-level)."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "strategy_version_id": None,
        "snapshot_ts": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        "computed_at": datetime(2024, 1, 15, 10, 30, 5, tzinfo=timezone.utc),
        "equity": 100000.0,
        "cash": 100000.0,
        "positions_value": 0.0,
        "realized_pnl": 0.0,
        "inputs_hash": "b" * 64,
    }


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = MagicMock()
    conn = AsyncMock()

    # Setup acquire as async context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm

    return pool, conn


# =============================================================================
# EquitySnapshot Dataclass Tests
# =============================================================================


class TestEquitySnapshotFromRow:
    """Tests for EquitySnapshot.from_row()."""

    def test_from_row_basic(self, sample_snapshot_row):
        """Test basic row conversion."""
        snapshot = EquitySnapshot.from_row(sample_snapshot_row)

        assert snapshot.id == sample_snapshot_row["id"]
        assert snapshot.workspace_id == sample_snapshot_row["workspace_id"]
        assert (
            snapshot.strategy_version_id == sample_snapshot_row["strategy_version_id"]
        )
        assert snapshot.snapshot_ts == sample_snapshot_row["snapshot_ts"]
        assert snapshot.computed_at == sample_snapshot_row["computed_at"]
        assert snapshot.equity == 105000.0
        assert snapshot.cash == 50000.0
        assert snapshot.positions_value == 55000.0
        assert snapshot.realized_pnl == 5000.0
        assert snapshot.inputs_hash == "a" * 64

    def test_from_row_null_version(self, sample_snapshot_row_no_version):
        """Test row with null strategy_version_id."""
        snapshot = EquitySnapshot.from_row(sample_snapshot_row_no_version)

        assert snapshot.strategy_version_id is None
        assert snapshot.equity == 100000.0

    def test_from_row_null_inputs_hash(self, sample_snapshot_row):
        """Test row with null inputs_hash."""
        sample_snapshot_row["inputs_hash"] = None

        snapshot = EquitySnapshot.from_row(sample_snapshot_row)

        assert snapshot.inputs_hash is None


# =============================================================================
# compute_inputs_hash Tests
# =============================================================================


class TestComputeInputsHash:
    """Tests for compute_inputs_hash()."""

    def test_deterministic(self):
        """Test that same inputs produce same hash."""
        ws_id = uuid4()
        hash1 = compute_inputs_hash(ws_id, 100000.0, 50000.0, 5000.0)
        hash2 = compute_inputs_hash(ws_id, 100000.0, 50000.0, 5000.0)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_different_workspace_different_hash(self):
        """Test that different workspaces produce different hashes."""
        hash1 = compute_inputs_hash(uuid4(), 100000.0, 50000.0, 5000.0)
        hash2 = compute_inputs_hash(uuid4(), 100000.0, 50000.0, 5000.0)

        assert hash1 != hash2

    def test_different_values_different_hash(self):
        """Test that different values produce different hashes."""
        ws_id = uuid4()
        hash1 = compute_inputs_hash(ws_id, 100000.0, 50000.0, 5000.0)
        hash2 = compute_inputs_hash(ws_id, 100001.0, 50000.0, 5000.0)

        assert hash1 != hash2

    def test_float_precision_tolerance(self):
        """Test that sub-threshold float differences produce same hash (intentional)."""
        ws_id = uuid4()
        # Differences smaller than 6 decimal places are intentionally ignored
        # to avoid false dedupe misses from floating point noise
        hash1 = compute_inputs_hash(ws_id, 100000.0000001, 50000.0, 5000.0)
        hash2 = compute_inputs_hash(ws_id, 100000.0000002, 50000.0, 5000.0)

        # Same hash (rounding absorbs noise)
        assert hash1 == hash2

        # But larger differences (> 0.000001) are captured
        hash3 = compute_inputs_hash(ws_id, 100000.000001, 50000.0, 5000.0)
        hash4 = compute_inputs_hash(ws_id, 100000.000002, 50000.0, 5000.0)
        assert hash3 != hash4


# =============================================================================
# Repository Tests
# =============================================================================


class TestInsertSnapshot:
    """Tests for insert_snapshot()."""

    @pytest.mark.asyncio
    async def test_insert_basic(self, mock_pool, sample_snapshot_row):
        """Test basic snapshot insertion."""
        pool, conn = mock_pool
        conn.fetchval = AsyncMock(return_value=None)  # No existing hash
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = PaperEquityRepository(pool)
        snapshot = await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            snapshot_ts=sample_snapshot_row["snapshot_ts"],
            equity=105000.0,
            cash=50000.0,
            positions_value=55000.0,
            realized_pnl=5000.0,
        )

        assert snapshot is not None
        assert snapshot.equity == 105000.0
        conn.fetchrow.assert_called_once()

        # Verify SQL
        call_args = conn.fetchrow.call_args
        sql = call_args[0][0]
        assert "INSERT INTO paper_equity_snapshots" in sql
        assert "RETURNING *" in sql

    @pytest.mark.asyncio
    async def test_insert_with_version(self, mock_pool, sample_snapshot_row):
        """Test insertion with strategy_version_id."""
        pool, conn = mock_pool
        conn.fetchval = AsyncMock(return_value=None)
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = PaperEquityRepository(pool)
        version_id = uuid4()

        await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            snapshot_ts=sample_snapshot_row["snapshot_ts"],
            equity=105000.0,
            cash=50000.0,
            positions_value=55000.0,
            realized_pnl=5000.0,
            strategy_version_id=version_id,
        )

        # Verify version_id is passed
        call_args = conn.fetchrow.call_args[0]
        assert call_args[2] == version_id  # Second positional arg

    @pytest.mark.asyncio
    async def test_insert_dedupe_skip(self, mock_pool, sample_snapshot_row):
        """Test that duplicate inputs are skipped."""
        pool, conn = mock_pool
        # Return existing hash that matches
        existing_hash = compute_inputs_hash(
            sample_snapshot_row["workspace_id"],
            50000.0,
            55000.0,
            5000.0,
        )
        conn.fetchval = AsyncMock(return_value=existing_hash)

        repo = PaperEquityRepository(pool)
        snapshot = await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            snapshot_ts=sample_snapshot_row["snapshot_ts"],
            equity=105000.0,
            cash=50000.0,
            positions_value=55000.0,
            realized_pnl=5000.0,
        )

        assert snapshot is None
        conn.fetchrow.assert_not_called()  # Insert should be skipped

    @pytest.mark.asyncio
    async def test_insert_skip_dedupe(self, mock_pool, sample_snapshot_row):
        """Test skip_dedupe flag bypasses dedupe check."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = PaperEquityRepository(pool)
        snapshot = await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            snapshot_ts=sample_snapshot_row["snapshot_ts"],
            equity=105000.0,
            cash=50000.0,
            positions_value=55000.0,
            realized_pnl=5000.0,
            skip_dedupe=True,
        )

        assert snapshot is not None
        conn.fetchval.assert_not_called()  # No dedupe check


class TestGetLatestSnapshot:
    """Tests for get_latest_snapshot()."""

    @pytest.mark.asyncio
    async def test_get_latest_found(self, mock_pool, sample_snapshot_row):
        """Test getting latest snapshot when one exists."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = PaperEquityRepository(pool)
        workspace_id = sample_snapshot_row["workspace_id"]

        snapshot = await repo.get_latest_snapshot(workspace_id)

        assert snapshot is not None
        assert snapshot.workspace_id == workspace_id

        # Verify query orders by snapshot_ts DESC
        call_args = conn.fetchrow.call_args
        sql = call_args[0][0]
        assert "ORDER BY snapshot_ts DESC" in sql
        assert "LIMIT 1" in sql

    @pytest.mark.asyncio
    async def test_get_latest_not_found(self, mock_pool):
        """Test getting latest snapshot when none exist."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        repo = PaperEquityRepository(pool)

        snapshot = await repo.get_latest_snapshot(uuid4())

        assert snapshot is None


class TestListWindow:
    """Tests for list_window()."""

    @pytest.mark.asyncio
    async def test_list_window_basic(self, mock_pool, sample_snapshot_row):
        """Test basic window listing."""
        pool, conn = mock_pool
        # Return multiple snapshots
        conn.fetch = AsyncMock(return_value=[sample_snapshot_row, sample_snapshot_row])

        repo = PaperEquityRepository(pool)
        workspace_id = sample_snapshot_row["workspace_id"]

        snapshots = await repo.list_window(workspace_id, window_days=30)

        assert len(snapshots) == 2

        # Verify query
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "snapshot_ts >= $2" in sql
        assert "ORDER BY snapshot_ts ASC" in sql  # Chronological order
        assert "LIMIT $3" in sql

    @pytest.mark.asyncio
    async def test_list_window_custom_days(self, mock_pool, sample_snapshot_row):
        """Test window with custom days."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = PaperEquityRepository(pool)

        await repo.list_window(uuid4(), window_days=7)

        # Verify cutoff is calculated correctly
        call_args = conn.fetch.call_args[0]
        cutoff = call_args[2]
        expected_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        # Allow 1 second tolerance
        assert abs((cutoff - expected_cutoff).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_list_window_empty(self, mock_pool):
        """Test window when no snapshots exist."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = PaperEquityRepository(pool)

        snapshots = await repo.list_window(uuid4())

        assert snapshots == []


class TestComputeDrawdown:
    """Tests for compute_drawdown()."""

    @pytest.mark.asyncio
    async def test_compute_drawdown_basic(self, mock_pool, sample_snapshot_row):
        """Test basic drawdown computation."""
        pool, conn = mock_pool
        workspace_id = sample_snapshot_row["workspace_id"]

        # Create snapshots: peak at 110k, current at 100k (9.09% DD)
        snapshots = [
            {
                **sample_snapshot_row,
                "snapshot_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
                "equity": 100000.0,
            },
            {
                **sample_snapshot_row,
                "snapshot_ts": datetime(2024, 1, 12, tzinfo=timezone.utc),
                "equity": 110000.0,  # Peak
            },
            {
                **sample_snapshot_row,
                "snapshot_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
                "equity": 100000.0,  # Current
            },
        ]
        conn.fetch = AsyncMock(return_value=snapshots)

        repo = PaperEquityRepository(pool)

        result = await repo.compute_drawdown(workspace_id, window_days=30)

        assert result is not None
        assert result.workspace_id == workspace_id
        assert result.peak_equity == 110000.0
        assert result.current_equity == 100000.0
        # DD = (110000 - 100000) / 110000 = 0.0909...
        assert abs(result.drawdown_pct - 0.0909) < 0.001
        assert result.snapshot_count == 3

    @pytest.mark.asyncio
    async def test_compute_drawdown_no_loss(self, mock_pool, sample_snapshot_row):
        """Test drawdown when at peak (no drawdown)."""
        pool, conn = mock_pool
        workspace_id = sample_snapshot_row["workspace_id"]

        # Equity only going up
        snapshots = [
            {
                **sample_snapshot_row,
                "snapshot_ts": datetime(2024, 1, 10, tzinfo=timezone.utc),
                "equity": 100000.0,
            },
            {
                **sample_snapshot_row,
                "snapshot_ts": datetime(2024, 1, 15, tzinfo=timezone.utc),
                "equity": 110000.0,  # At peak
            },
        ]
        conn.fetch = AsyncMock(return_value=snapshots)

        repo = PaperEquityRepository(pool)

        result = await repo.compute_drawdown(workspace_id)

        assert result is not None
        assert result.drawdown_pct == 0.0

    @pytest.mark.asyncio
    async def test_compute_drawdown_no_data(self, mock_pool):
        """Test drawdown with no snapshots."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = PaperEquityRepository(pool)

        result = await repo.compute_drawdown(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_compute_drawdown_single_snapshot(
        self, mock_pool, sample_snapshot_row
    ):
        """Test drawdown with single snapshot (peak = current)."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[sample_snapshot_row])

        repo = PaperEquityRepository(pool)

        result = await repo.compute_drawdown(sample_snapshot_row["workspace_id"])

        assert result is not None
        assert result.drawdown_pct == 0.0
        assert result.snapshot_count == 1


class TestListWorkspacesWithSnapshots:
    """Tests for list_workspaces_with_snapshots()."""

    @pytest.mark.asyncio
    async def test_list_workspaces_basic(self, mock_pool):
        """Test listing workspaces with recent snapshots."""
        pool, conn = mock_pool
        ws1, ws2 = uuid4(), uuid4()
        conn.fetch = AsyncMock(
            return_value=[{"workspace_id": ws1}, {"workspace_id": ws2}]
        )

        repo = PaperEquityRepository(pool)

        workspaces = await repo.list_workspaces_with_snapshots(since_days=7)

        assert len(workspaces) == 2
        assert ws1 in workspaces
        assert ws2 in workspaces

    @pytest.mark.asyncio
    async def test_list_workspaces_empty(self, mock_pool):
        """Test when no workspaces have snapshots."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = PaperEquityRepository(pool)

        workspaces = await repo.list_workspaces_with_snapshots()

        assert workspaces == []


class TestDeleteOldSnapshots:
    """Tests for delete_old_snapshots()."""

    @pytest.mark.asyncio
    async def test_delete_old_basic(self, mock_pool):
        """Test deleting old snapshots."""
        pool, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 15")

        repo = PaperEquityRepository(pool)

        count = await repo.delete_old_snapshots(retention_days=90)

        assert count == 15

        # Verify cutoff
        call_args = conn.execute.call_args[0]
        sql = call_args[0]
        assert "DELETE FROM paper_equity_snapshots" in sql
        assert "snapshot_ts < $1" in sql

    @pytest.mark.asyncio
    async def test_delete_old_none(self, mock_pool):
        """Test when no old snapshots exist."""
        pool, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 0")

        repo = PaperEquityRepository(pool)

        count = await repo.delete_old_snapshots()

        assert count == 0
