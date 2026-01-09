"""Tests for recommendation records repository."""

import pytest
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from app.repositories.recommendation_records import (
    RecommendationRecordsRepository,
    RecommendationRecord,
    RecommendationObservation,
    EvaluationSlice,
    RecordStatus,
)


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


class TestRecordStatus:
    """Tests for RecordStatus enum."""

    def test_status_values(self):
        """RecordStatus has all required values."""
        assert RecordStatus.ACTIVE.value == "active"
        assert RecordStatus.SUPERSEDED.value == "superseded"
        assert RecordStatus.INACTIVE.value == "inactive"
        assert RecordStatus.CLOSED.value == "closed"

    def test_status_is_string_enum(self):
        """RecordStatus can be compared to strings."""
        assert RecordStatus.ACTIVE == "active"
        assert RecordStatus.CLOSED == "closed"


class TestRecommendationRecord:
    """Tests for RecommendationRecord dataclass."""

    def test_record_creation_with_required_fields(self):
        """RecommendationRecord can be created with required fields."""
        workspace_id = uuid4()
        strategy_id = uuid4()

        record = RecommendationRecord(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_id,
            symbol="BTC/USDT",
            timeframe="5m",
            params_json={"period": 14, "threshold": 0.5},
            params_hash="abc123def456",
            regime_key_start="trend=uptrend|vol=high_vol",
            regime_dims_start={"trend": "uptrend", "vol": "high_vol"},
            regime_features_start={"atr_pct": 0.02, "rsi": 65.0},
            confidence_json={"regime_fit_confidence": 0.85, "distance_z": 0.3},
            expected_baselines_json={"sharpe_mean": 1.2, "return_pct_mean": 0.05},
        )

        assert record.workspace_id == workspace_id
        assert record.strategy_entity_id == strategy_id
        assert record.symbol == "BTC/USDT"
        assert record.timeframe == "5m"
        assert record.status == RecordStatus.ACTIVE  # default
        assert record.schema_version == 1  # default
        assert record.id is None  # default
        assert record.created_at is None  # default
        assert record.updated_at is None  # default

    def test_record_creation_with_optional_fields(self):
        """RecommendationRecord accepts optional fields."""
        record_id = uuid4()
        now = datetime.now(timezone.utc)

        record = RecommendationRecord(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
            symbol="ETH/USDT",
            timeframe="1h",
            params_json={},
            params_hash="xyz789",
            regime_key_start="trend=flat|vol=low_vol",
            regime_dims_start={"trend": "flat", "vol": "low_vol"},
            regime_features_start={},
            confidence_json={},
            expected_baselines_json={},
            id=record_id,
            status=RecordStatus.CLOSED,
            schema_version=2,
            created_at=now,
            updated_at=now,
        )

        assert record.id == record_id
        assert record.status == RecordStatus.CLOSED
        assert record.schema_version == 2
        assert record.created_at == now
        assert record.updated_at == now


class TestRecommendationObservation:
    """Tests for RecommendationObservation dataclass."""

    def test_observation_creation(self):
        """RecommendationObservation can be created with required fields."""
        record_id = uuid4()
        ts = datetime.now(timezone.utc)

        obs = RecommendationObservation(
            record_id=record_id,
            ts=ts,
            bars_seen=500,
            trades_seen=42,
            realized_metrics_json={
                "return_pct": 0.023,
                "sharpe_proxy": 1.1,
                "max_drawdown_pct": 0.05,
            },
        )

        assert obs.record_id == record_id
        assert obs.ts == ts
        assert obs.bars_seen == 500
        assert obs.trades_seen == 42
        assert obs.realized_metrics_json["return_pct"] == 0.023
        assert obs.id is None  # default
        assert obs.created_at is None  # default


class TestEvaluationSlice:
    """Tests for EvaluationSlice dataclass."""

    def test_slice_creation(self):
        """EvaluationSlice can be created with required fields."""
        record_id = uuid4()
        start_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_ts = datetime(2026, 1, 9, tzinfo=timezone.utc)

        slice_ = EvaluationSlice(
            record_id=record_id,
            slice_start_ts=start_ts,
            slice_end_ts=end_ts,
            trigger_type="regime_change",
            regime_key_during="trend=uptrend|vol=high_vol",
            realized_summary_json={"return_pct": 0.05, "sharpe": 1.2},
            expected_summary_json={"return_pct_mean": 0.04, "sharpe_mean": 1.0},
        )

        assert slice_.record_id == record_id
        assert slice_.slice_start_ts == start_ts
        assert slice_.slice_end_ts == end_ts
        assert slice_.trigger_type == "regime_change"
        assert slice_.regime_key_during == "trend=uptrend|vol=high_vol"
        assert slice_.performance_surprise_z is None  # default
        assert slice_.drift_flags_json is None  # default
        assert slice_.id is None  # default

    def test_slice_with_optional_fields(self):
        """EvaluationSlice accepts optional fields."""
        slice_ = EvaluationSlice(
            record_id=uuid4(),
            slice_start_ts=datetime.now(timezone.utc),
            slice_end_ts=datetime.now(timezone.utc),
            trigger_type="milestone",
            regime_key_during="trend=flat|vol=mid_vol",
            realized_summary_json={},
            expected_summary_json={},
            performance_surprise_z=1.5,
            drift_flags_json={"volatility_drift": True},
        )

        assert slice_.performance_surprise_z == 1.5
        assert slice_.drift_flags_json["volatility_drift"] is True

    def test_slice_trigger_types(self):
        """EvaluationSlice supports different trigger types."""
        trigger_types = ["regime_change", "milestone", "manual"]

        for trigger in trigger_types:
            slice_ = EvaluationSlice(
                record_id=uuid4(),
                slice_start_ts=datetime.now(timezone.utc),
                slice_end_ts=datetime.now(timezone.utc),
                trigger_type=trigger,
                regime_key_during="trend=uptrend|vol=low_vol",
                realized_summary_json={},
                expected_summary_json={},
            )
            assert slice_.trigger_type == trigger


class TestRecommendationRecordsRepository:
    """Tests for RecommendationRecordsRepository."""

    @pytest.mark.asyncio
    async def test_create_record_supersedes_existing(self, mock_pool):
        """Creating a new record supersedes the existing active one."""
        repo = RecommendationRecordsRepository(mock_pool)

        executed_queries = []

        async def mock_execute(query, *args):
            executed_queries.append(query)

        new_record_id = uuid4()
        conn = AsyncMock()
        conn.execute = mock_execute
        conn.fetchrow = AsyncMock(return_value={"id": new_record_id})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        record = RecommendationRecord(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            params_json={"period": 14},
            params_hash="abc123",
            regime_key_start="trend=uptrend|vol=high_vol",
            regime_dims_start={"trend": "uptrend", "vol": "high_vol"},
            regime_features_start={"atr_pct": 0.02},
            confidence_json={"regime_fit_confidence": 0.8},
            expected_baselines_json={"sharpe_mean": 1.2},
        )

        result_id = await repo.create_record(record)

        # Should have UPDATE to supersede existing + INSERT new
        assert any("UPDATE" in q and "superseded" in q for q in executed_queries)
        assert result_id == new_record_id

    @pytest.mark.asyncio
    async def test_create_record_returns_uuid(self, mock_pool):
        """create_record returns the new record ID."""
        repo = RecommendationRecordsRepository(mock_pool)

        new_id = uuid4()
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": new_id})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        record = RecommendationRecord(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
            params_json={},
            params_hash="hash",
            regime_key_start="trend=flat|vol=mid_vol",
            regime_dims_start={},
            regime_features_start={},
            confidence_json={},
            expected_baselines_json={},
        )

        result = await repo.create_record(record)

        assert result == new_id

    @pytest.mark.asyncio
    async def test_get_active_record_returns_record_when_found(self, mock_pool):
        """Gets the active record for symbol+strategy."""
        repo = RecommendationRecordsRepository(mock_pool)

        record_id = uuid4()
        workspace_id = uuid4()
        strategy_id = uuid4()
        now = datetime.now(timezone.utc)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "id": record_id,
                "workspace_id": workspace_id,
                "strategy_entity_id": strategy_id,
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "params_json": {"period": 14},
                "params_hash": "abc123",
                "regime_key_start": "trend=uptrend|vol=high_vol",
                "regime_dims_start": {"trend": "uptrend", "vol": "high_vol"},
                "regime_features_start": {"atr_pct": 0.02},
                "confidence_json": {"regime_fit_confidence": 0.8},
                "expected_baselines_json": {"sharpe_mean": 1.2},
                "status": "active",
                "schema_version": 1,
                "created_at": now,
                "updated_at": now,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_active_record(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_id,
            symbol="BTC/USDT",
            timeframe="5m",
        )

        assert result is not None
        assert result.id == record_id
        assert result.status == RecordStatus.ACTIVE
        assert result.symbol == "BTC/USDT"
        assert result.params_json == {"period": 14}

    @pytest.mark.asyncio
    async def test_get_active_record_returns_none_when_missing(self, mock_pool):
        """Returns None when no active record exists."""
        repo = RecommendationRecordsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_active_record(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
            symbol="BTC/USDT",
            timeframe="5m",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_record_by_id_returns_record(self, mock_pool):
        """Gets record by ID."""
        repo = RecommendationRecordsRepository(mock_pool)

        record_id = uuid4()
        now = datetime.now(timezone.utc)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "id": record_id,
                "workspace_id": uuid4(),
                "strategy_entity_id": uuid4(),
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "params_json": {},
                "params_hash": "xyz",
                "regime_key_start": "trend=downtrend|vol=low_vol",
                "regime_dims_start": {},
                "regime_features_start": {},
                "confidence_json": {},
                "expected_baselines_json": {},
                "status": "active",
                "schema_version": 1,
                "created_at": now,
                "updated_at": now,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_record_by_id(record_id)

        assert result is not None
        assert result.id == record_id
        assert result.symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_get_record_by_id_returns_none_when_missing(self, mock_pool):
        """Returns None when record ID not found."""
        repo = RecommendationRecordsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_record_by_id(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_close_record_updates_status(self, mock_pool):
        """Closing record updates status."""
        repo = RecommendationRecordsRepository(mock_pool)

        executed_queries = []
        executed_args = []

        async def mock_execute(query, *args):
            executed_queries.append(query)
            executed_args.append(args)

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        record_id = uuid4()
        await repo.close_record(
            record_id=record_id,
            new_status=RecordStatus.CLOSED,
        )

        assert len(executed_queries) == 1
        assert "UPDATE" in executed_queries[0]
        assert record_id in executed_args[0]
        assert "closed" in executed_args[0]

    @pytest.mark.asyncio
    async def test_close_record_with_inactive_status(self, mock_pool):
        """Can close record with inactive status."""
        repo = RecommendationRecordsRepository(mock_pool)

        executed_args = []

        async def mock_execute(query, *args):
            executed_args.append(args)

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        await repo.close_record(
            record_id=uuid4(),
            new_status=RecordStatus.INACTIVE,
        )

        assert "inactive" in executed_args[0]

    @pytest.mark.asyncio
    async def test_add_observation_returns_uuid(self, mock_pool):
        """add_observation returns observation ID."""
        repo = RecommendationRecordsRepository(mock_pool)

        obs_id = uuid4()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": obs_id})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        obs = RecommendationObservation(
            record_id=uuid4(),
            ts=datetime.now(timezone.utc),
            bars_seen=500,
            trades_seen=42,
            realized_metrics_json={"return_pct": 0.02},
        )

        result = await repo.add_observation(obs)

        assert result == obs_id

    @pytest.mark.asyncio
    async def test_add_observation_inserts_correct_data(self, mock_pool):
        """add_observation inserts observation with correct fields."""
        repo = RecommendationRecordsRepository(mock_pool)

        inserted_args = []

        async def mock_fetchrow(query, *args):
            inserted_args.extend(args)
            return {"id": uuid4()}

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        record_id = uuid4()
        ts = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)

        obs = RecommendationObservation(
            record_id=record_id,
            ts=ts,
            bars_seen=1000,
            trades_seen=85,
            realized_metrics_json={"sharpe_proxy": 1.5},
        )

        await repo.add_observation(obs)

        assert record_id in inserted_args
        assert ts in inserted_args
        assert 1000 in inserted_args
        assert 85 in inserted_args

    @pytest.mark.asyncio
    async def test_create_slice_returns_uuid(self, mock_pool):
        """create_slice returns slice ID."""
        repo = RecommendationRecordsRepository(mock_pool)

        slice_id = uuid4()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": slice_id})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        slice_ = EvaluationSlice(
            record_id=uuid4(),
            slice_start_ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
            slice_end_ts=datetime(2026, 1, 9, tzinfo=timezone.utc),
            trigger_type="regime_change",
            regime_key_during="trend=uptrend|vol=high_vol",
            realized_summary_json={"return_pct": 0.05},
            expected_summary_json={"return_pct_mean": 0.04},
        )

        result = await repo.create_slice(slice_)

        assert result == slice_id

    @pytest.mark.asyncio
    async def test_create_slice_inserts_correct_data(self, mock_pool):
        """create_slice inserts slice with correct fields."""
        repo = RecommendationRecordsRepository(mock_pool)

        inserted_args = []

        async def mock_fetchrow(query, *args):
            inserted_args.extend(args)
            return {"id": uuid4()}

        conn = AsyncMock()
        conn.fetchrow = mock_fetchrow
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        record_id = uuid4()
        start_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end_ts = datetime(2026, 1, 9, tzinfo=timezone.utc)

        slice_ = EvaluationSlice(
            record_id=record_id,
            slice_start_ts=start_ts,
            slice_end_ts=end_ts,
            trigger_type="milestone",
            regime_key_during="trend=flat|vol=mid_vol",
            realized_summary_json={},
            expected_summary_json={},
            performance_surprise_z=2.5,
            drift_flags_json={"volume_drift": True},
        )

        await repo.create_slice(slice_)

        assert record_id in inserted_args
        assert start_ts in inserted_args
        assert end_ts in inserted_args
        assert "milestone" in inserted_args
        assert 2.5 in inserted_args

    @pytest.mark.asyncio
    async def test_create_slice_with_none_optional_fields(self, mock_pool):
        """create_slice handles None optional fields."""
        repo = RecommendationRecordsRepository(mock_pool)

        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": uuid4()})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        slice_ = EvaluationSlice(
            record_id=uuid4(),
            slice_start_ts=datetime.now(timezone.utc),
            slice_end_ts=datetime.now(timezone.utc),
            trigger_type="manual",
            regime_key_during="trend=downtrend|vol=high_vol",
            realized_summary_json={},
            expected_summary_json={},
            performance_surprise_z=None,
            drift_flags_json=None,
        )

        # Should not raise
        result = await repo.create_slice(slice_)
        assert result is not None


class TestRowConversion:
    """Tests for row-to-dataclass conversion."""

    @pytest.mark.asyncio
    async def test_row_to_record_handles_json_strings(self, mock_pool):
        """Row conversion handles JSON as both dicts and strings."""
        repo = RecommendationRecordsRepository(mock_pool)

        record_id = uuid4()
        conn = AsyncMock()
        # Simulate JSON being returned as strings (some DB drivers do this)
        conn.fetchrow = AsyncMock(
            return_value={
                "id": record_id,
                "workspace_id": uuid4(),
                "strategy_entity_id": uuid4(),
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "params_json": '{"period": 14}',  # JSON string
                "params_hash": "abc",
                "regime_key_start": "trend=uptrend|vol=high_vol",
                "regime_dims_start": '{"trend": "uptrend"}',  # JSON string
                "regime_features_start": '{"atr_pct": 0.02}',  # JSON string
                "confidence_json": '{"conf": 0.8}',  # JSON string
                "expected_baselines_json": '{"sharpe": 1.2}',  # JSON string
                "status": "active",
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_record_by_id(record_id)

        assert result is not None
        assert result.params_json == {"period": 14}
        assert result.regime_dims_start == {"trend": "uptrend"}
        assert result.regime_features_start == {"atr_pct": 0.02}
        assert result.confidence_json == {"conf": 0.8}
        assert result.expected_baselines_json == {"sharpe": 1.2}

    @pytest.mark.asyncio
    async def test_row_to_record_handles_json_dicts(self, mock_pool):
        """Row conversion handles JSON as dicts (asyncpg default)."""
        repo = RecommendationRecordsRepository(mock_pool)

        record_id = uuid4()
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={
                "id": record_id,
                "workspace_id": uuid4(),
                "strategy_entity_id": uuid4(),
                "symbol": "BTC/USDT",
                "timeframe": "5m",
                "params_json": {"period": 14},  # Dict
                "params_hash": "abc",
                "regime_key_start": "trend=uptrend|vol=high_vol",
                "regime_dims_start": {"trend": "uptrend"},  # Dict
                "regime_features_start": {"atr_pct": 0.02},  # Dict
                "confidence_json": {"conf": 0.8},  # Dict
                "expected_baselines_json": {"sharpe": 1.2},  # Dict
                "status": "superseded",
                "schema_version": 2,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await repo.get_record_by_id(record_id)

        assert result is not None
        assert result.params_json == {"period": 14}
        assert result.status == RecordStatus.SUPERSEDED
        assert result.schema_version == 2
