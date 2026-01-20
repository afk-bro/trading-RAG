"""Unit tests for strategy intelligence repository."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.repositories.strategy_intel import IntelSnapshot, StrategyIntelRepository


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_snapshot_row():
    """Sample database row for an intel snapshot."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "strategy_version_id": uuid4(),
        "as_of_ts": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        "computed_at": datetime(2024, 1, 15, 10, 30, 5, tzinfo=timezone.utc),
        "regime": "trending_bullish",
        "confidence_score": 0.85,
        "confidence_components": {"regime_fit": 0.9, "backtest_oos": 0.8},
        "features": {"rsi": 65.5, "macd_histogram": 0.12},
        "explain": {"summary": "Strong uptrend with momentum confirmation"},
        "engine_version": "1.0.0",
        "inputs_hash": "a" * 64,
        "run_id": uuid4(),
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
# IntelSnapshot Dataclass Tests
# =============================================================================


class TestIntelSnapshotFromRow:
    """Tests for IntelSnapshot.from_row()."""

    def test_from_row_basic(self, sample_snapshot_row):
        """Test basic row conversion."""
        snapshot = IntelSnapshot.from_row(sample_snapshot_row)

        assert snapshot.id == sample_snapshot_row["id"]
        assert snapshot.workspace_id == sample_snapshot_row["workspace_id"]
        assert snapshot.strategy_version_id == sample_snapshot_row["strategy_version_id"]
        assert snapshot.as_of_ts == sample_snapshot_row["as_of_ts"]
        assert snapshot.computed_at == sample_snapshot_row["computed_at"]
        assert snapshot.regime == "trending_bullish"
        assert snapshot.confidence_score == 0.85
        assert snapshot.confidence_components == {"regime_fit": 0.9, "backtest_oos": 0.8}
        assert snapshot.features == {"rsi": 65.5, "macd_histogram": 0.12}
        assert snapshot.explain == {"summary": "Strong uptrend with momentum confirmation"}
        assert snapshot.engine_version == "1.0.0"
        assert snapshot.inputs_hash == "a" * 64
        assert snapshot.run_id == sample_snapshot_row["run_id"]

    def test_from_row_jsonb_as_string(self, sample_snapshot_row):
        """Test parsing when JSONB fields come as strings."""
        sample_snapshot_row["confidence_components"] = '{"regime_fit": 0.9}'
        sample_snapshot_row["features"] = '{"rsi": 65.5}'
        sample_snapshot_row["explain"] = '{"summary": "Test"}'

        snapshot = IntelSnapshot.from_row(sample_snapshot_row)

        assert snapshot.confidence_components == {"regime_fit": 0.9}
        assert snapshot.features == {"rsi": 65.5}
        assert snapshot.explain == {"summary": "Test"}

    def test_from_row_null_optionals(self, sample_snapshot_row):
        """Test row with null optional fields."""
        sample_snapshot_row["engine_version"] = None
        sample_snapshot_row["inputs_hash"] = None
        sample_snapshot_row["run_id"] = None
        sample_snapshot_row["confidence_components"] = None
        sample_snapshot_row["features"] = None
        sample_snapshot_row["explain"] = None

        snapshot = IntelSnapshot.from_row(sample_snapshot_row)

        assert snapshot.engine_version is None
        assert snapshot.inputs_hash is None
        assert snapshot.run_id is None
        assert snapshot.confidence_components == {}
        assert snapshot.features == {}
        assert snapshot.explain == {}


# =============================================================================
# Repository Tests
# =============================================================================


class TestInsertSnapshot:
    """Tests for insert_snapshot()."""

    @pytest.mark.asyncio
    async def test_insert_basic(self, mock_pool, sample_snapshot_row):
        """Test basic snapshot insertion."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = StrategyIntelRepository(pool)
        snapshot = await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            strategy_version_id=sample_snapshot_row["strategy_version_id"],
            as_of_ts=sample_snapshot_row["as_of_ts"],
            regime="trending_bullish",
            confidence_score=0.85,
            confidence_components={"regime_fit": 0.9, "backtest_oos": 0.8},
            features={"rsi": 65.5},
            explain={"summary": "Test"},
            engine_version="1.0.0",
            inputs_hash="a" * 64,
            run_id=sample_snapshot_row["run_id"],
        )

        assert snapshot.regime == "trending_bullish"
        assert snapshot.confidence_score == 0.85
        conn.fetchrow.assert_called_once()

        # Verify the SQL and arguments
        call_args = conn.fetchrow.call_args
        sql = call_args[0][0]
        assert "INSERT INTO strategy_intel_snapshots" in sql
        assert "RETURNING *" in sql

    @pytest.mark.asyncio
    async def test_insert_minimal(self, mock_pool, sample_snapshot_row):
        """Test insertion with minimal required fields."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = StrategyIntelRepository(pool)
        snapshot = await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            strategy_version_id=sample_snapshot_row["strategy_version_id"],
            as_of_ts=sample_snapshot_row["as_of_ts"],
            regime="ranging",
            confidence_score=0.5,
        )

        assert snapshot is not None

        # Verify defaults are passed as empty dicts
        call_args = conn.fetchrow.call_args[0]
        # Arguments 6, 7, 8 are confidence_components, features, explain
        assert call_args[6] == "{}"  # JSON-serialized empty dict
        assert call_args[7] == "{}"
        assert call_args[8] == "{}"

    @pytest.mark.asyncio
    async def test_insert_rejects_invalid_confidence(self, mock_pool):
        """Test that confidence_score outside [0, 1] raises ValueError."""
        pool, conn = mock_pool
        repo = StrategyIntelRepository(pool)

        with pytest.raises(ValueError, match="confidence_score must be in"):
            await repo.insert_snapshot(
                workspace_id=uuid4(),
                strategy_version_id=uuid4(),
                as_of_ts=datetime.now(timezone.utc),
                regime="trending",
                confidence_score=1.5,  # Invalid
            )

        with pytest.raises(ValueError, match="confidence_score must be in"):
            await repo.insert_snapshot(
                workspace_id=uuid4(),
                strategy_version_id=uuid4(),
                as_of_ts=datetime.now(timezone.utc),
                regime="trending",
                confidence_score=-0.1,  # Invalid
            )

    @pytest.mark.asyncio
    async def test_insert_boundary_confidence(self, mock_pool, sample_snapshot_row):
        """Test boundary values for confidence_score."""
        pool, conn = mock_pool
        sample_snapshot_row["confidence_score"] = 0.0
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = StrategyIntelRepository(pool)

        # 0.0 should be accepted
        snapshot = await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            strategy_version_id=sample_snapshot_row["strategy_version_id"],
            as_of_ts=sample_snapshot_row["as_of_ts"],
            regime="low_confidence",
            confidence_score=0.0,
        )
        assert snapshot.confidence_score == 0.0

        # 1.0 should be accepted
        sample_snapshot_row["confidence_score"] = 1.0
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)
        snapshot = await repo.insert_snapshot(
            workspace_id=sample_snapshot_row["workspace_id"],
            strategy_version_id=sample_snapshot_row["strategy_version_id"],
            as_of_ts=sample_snapshot_row["as_of_ts"],
            regime="high_confidence",
            confidence_score=1.0,
        )
        assert snapshot.confidence_score == 1.0


class TestGetLatestSnapshot:
    """Tests for get_latest_snapshot()."""

    @pytest.mark.asyncio
    async def test_get_latest_found(self, mock_pool, sample_snapshot_row):
        """Test getting latest snapshot when one exists."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=sample_snapshot_row)

        repo = StrategyIntelRepository(pool)
        version_id = sample_snapshot_row["strategy_version_id"]

        snapshot = await repo.get_latest_snapshot(version_id)

        assert snapshot is not None
        assert snapshot.strategy_version_id == version_id

        # Verify query orders by as_of_ts DESC
        call_args = conn.fetchrow.call_args
        sql = call_args[0][0]
        assert "ORDER BY as_of_ts DESC" in sql
        assert "LIMIT 1" in sql

    @pytest.mark.asyncio
    async def test_get_latest_not_found(self, mock_pool):
        """Test getting latest snapshot when none exist."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        repo = StrategyIntelRepository(pool)

        snapshot = await repo.get_latest_snapshot(uuid4())

        assert snapshot is None


class TestListSnapshots:
    """Tests for list_snapshots()."""

    @pytest.mark.asyncio
    async def test_list_basic(self, mock_pool, sample_snapshot_row):
        """Test basic listing without cursor."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[sample_snapshot_row, sample_snapshot_row])

        repo = StrategyIntelRepository(pool)
        version_id = sample_snapshot_row["strategy_version_id"]

        snapshots = await repo.list_snapshots(version_id, limit=50)

        assert len(snapshots) == 2

        # Verify query
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "ORDER BY as_of_ts DESC" in sql
        assert "LIMIT $2" in sql

    @pytest.mark.asyncio
    async def test_list_with_cursor(self, mock_pool, sample_snapshot_row):
        """Test listing with cursor pagination."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[sample_snapshot_row])

        repo = StrategyIntelRepository(pool)
        version_id = sample_snapshot_row["strategy_version_id"]
        cursor = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        snapshots = await repo.list_snapshots(version_id, limit=50, cursor=cursor)

        assert len(snapshots) == 1

        # Verify cursor is used in WHERE clause
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "as_of_ts < $2" in sql
        assert "LIMIT $3" in sql

    @pytest.mark.asyncio
    async def test_list_caps_limit(self, mock_pool, sample_snapshot_row):
        """Test that limit is capped at 200."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = StrategyIntelRepository(pool)

        await repo.list_snapshots(uuid4(), limit=500)  # Request 500

        # Verify limit is capped
        call_args = conn.fetch.call_args
        args = call_args[0]
        assert args[2] == 200  # Should be capped at 200

    @pytest.mark.asyncio
    async def test_list_empty(self, mock_pool):
        """Test listing when no snapshots exist."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = StrategyIntelRepository(pool)

        snapshots = await repo.list_snapshots(uuid4())

        assert snapshots == []


class TestListWorkspaceSnapshots:
    """Tests for list_workspace_snapshots()."""

    @pytest.mark.asyncio
    async def test_list_workspace_basic(self, mock_pool, sample_snapshot_row):
        """Test workspace-level listing."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[sample_snapshot_row])

        repo = StrategyIntelRepository(pool)
        workspace_id = sample_snapshot_row["workspace_id"]

        snapshots = await repo.list_workspace_snapshots(workspace_id, limit=50)

        assert len(snapshots) == 1

        # Verify workspace_id filter
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "workspace_id = $1" in sql
        assert "ORDER BY as_of_ts DESC" in sql

    @pytest.mark.asyncio
    async def test_list_workspace_with_cursor(self, mock_pool, sample_snapshot_row):
        """Test workspace listing with cursor."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = StrategyIntelRepository(pool)
        workspace_id = sample_snapshot_row["workspace_id"]
        cursor = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        await repo.list_workspace_snapshots(workspace_id, cursor=cursor)

        # Verify cursor is used
        call_args = conn.fetch.call_args
        sql = call_args[0][0]
        assert "as_of_ts < $2" in sql

    @pytest.mark.asyncio
    async def test_list_workspace_caps_limit(self, mock_pool):
        """Test workspace listing caps limit at 200."""
        pool, conn = mock_pool
        conn.fetch = AsyncMock(return_value=[])

        repo = StrategyIntelRepository(pool)

        await repo.list_workspace_snapshots(uuid4(), limit=1000)

        # Verify limit is capped
        call_args = conn.fetch.call_args
        args = call_args[0]
        assert args[2] == 200
