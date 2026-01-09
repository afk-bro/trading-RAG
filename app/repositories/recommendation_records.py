"""
Repository for recommendation records and related entities.

Handles CRUD for recommendation_records, recommendation_observations,
and recommendation_evaluation_slices tables.

Used for v1.5 expected-vs-realized tracking.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class RecordStatus(str, Enum):
    """Recommendation record status."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INACTIVE = "inactive"
    CLOSED = "closed"


@dataclass
class RecommendationRecord:
    """
    Recommendation expectation contract.

    Stores the recommended parameters, starting regime conditions,
    confidence metrics, and expected baseline metrics.
    """

    workspace_id: UUID
    strategy_entity_id: UUID
    symbol: str
    timeframe: str
    params_json: dict
    params_hash: str
    regime_key_start: str
    regime_dims_start: dict
    regime_features_start: dict
    confidence_json: dict
    expected_baselines_json: dict
    id: Optional[UUID] = None
    status: RecordStatus = RecordStatus.ACTIVE
    schema_version: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class RecommendationObservation:
    """
    Streaming realized metrics observation.

    Append-only record of realized metrics from forward runs.
    Keyed by (record_id, ts) for idempotency.
    """

    record_id: UUID
    ts: datetime
    bars_seen: int
    trades_seen: int
    realized_metrics_json: dict
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None


@dataclass
class EvaluationSlice:
    """
    Accountability checkpoint.

    Immutable snapshot of realized vs expected metrics for a time slice.
    Created on regime change, milestone, or manual trigger.
    """

    record_id: UUID
    slice_start_ts: datetime
    slice_end_ts: datetime
    trigger_type: str  # regime_change | milestone | manual
    regime_key_during: str
    realized_summary_json: dict
    expected_summary_json: dict
    performance_surprise_z: Optional[float] = None
    drift_flags_json: Optional[dict] = None
    id: Optional[UUID] = None
    created_at: Optional[datetime] = None


class RecommendationRecordsRepository:
    """
    Repository for recommendation records and related tables.

    Provides:
    - create_record: Creates new record, supersedes existing active one
    - get_active_record: Gets active record for (workspace, strategy, symbol, timeframe)
    - get_record_by_id: Gets record by ID
    - close_record: Closes record with specified status
    - add_observation: Adds streaming observation
    - create_slice: Creates evaluation slice
    """

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def create_record(self, record: RecommendationRecord) -> UUID:
        """
        Create new recommendation record, superseding any existing active one.

        Only one active record per (workspace_id, strategy_entity_id, symbol, timeframe).

        Args:
            record: RecommendationRecord to create

        Returns:
            ID of created record
        """
        async with self.pool.acquire() as conn:
            # Supersede existing active record
            await conn.execute(
                """
                UPDATE recommendation_records
                SET status = 'superseded', updated_at = now()
                WHERE workspace_id = $1
                  AND strategy_entity_id = $2
                  AND symbol = $3
                  AND timeframe = $4
                  AND status = 'active'
                """,
                record.workspace_id,
                record.strategy_entity_id,
                record.symbol,
                record.timeframe,
            )

            # Insert new record
            row = await conn.fetchrow(
                """
                INSERT INTO recommendation_records (
                    workspace_id, strategy_entity_id, symbol, timeframe,
                    params_json, params_hash,
                    regime_key_start, regime_dims_start, regime_features_start,
                    schema_version, confidence_json, expected_baselines_json,
                    status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, 'active')
                RETURNING id
                """,
                record.workspace_id,
                record.strategy_entity_id,
                record.symbol,
                record.timeframe,
                json.dumps(record.params_json),
                record.params_hash,
                record.regime_key_start,
                json.dumps(record.regime_dims_start),
                json.dumps(record.regime_features_start),
                record.schema_version,
                json.dumps(record.confidence_json),
                json.dumps(record.expected_baselines_json),
            )

            logger.info(
                "recommendation_record_created",
                record_id=str(row["id"]),
                workspace_id=str(record.workspace_id),
                strategy_entity_id=str(record.strategy_entity_id),
                symbol=record.symbol,
                timeframe=record.timeframe,
                regime_key=record.regime_key_start,
            )

            return row["id"]

    async def get_active_record(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        symbol: str,
        timeframe: str,
    ) -> Optional[RecommendationRecord]:
        """
        Get active record for symbol+strategy combination.

        Args:
            workspace_id: Workspace ID
            strategy_entity_id: Strategy entity ID
            symbol: Trading symbol
            timeframe: Timeframe string

        Returns:
            RecommendationRecord or None if not found
        """
        query = """
            SELECT * FROM recommendation_records
            WHERE workspace_id = $1
              AND strategy_entity_id = $2
              AND symbol = $3
              AND timeframe = $4
              AND status = 'active'
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query, workspace_id, strategy_entity_id, symbol, timeframe
            )

        if row is None:
            return None

        return self._row_to_record(row)

    async def get_record_by_id(self, record_id: UUID) -> Optional[RecommendationRecord]:
        """
        Get record by ID.

        Args:
            record_id: Record ID

        Returns:
            RecommendationRecord or None if not found
        """
        query = """
            SELECT * FROM recommendation_records
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, record_id)

        if row is None:
            return None

        return self._row_to_record(row)

    async def close_record(
        self,
        record_id: UUID,
        new_status: RecordStatus,
    ) -> None:
        """
        Close a record with specified status.

        Args:
            record_id: Record ID to close
            new_status: New status (closed, inactive)
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE recommendation_records
                SET status = $2, updated_at = now()
                WHERE id = $1
                """,
                record_id,
                new_status.value,
            )

        logger.info(
            "recommendation_record_closed",
            record_id=str(record_id),
            new_status=new_status.value,
        )

    async def add_observation(self, obs: RecommendationObservation) -> UUID:
        """
        Add observation to record.

        Args:
            obs: RecommendationObservation to add

        Returns:
            Observation ID

        Raises:
            UniqueViolationError: If (record_id, ts) already exists
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO recommendation_observations (
                    record_id, ts, bars_seen, trades_seen, realized_metrics_json
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
                """,
                obs.record_id,
                obs.ts,
                obs.bars_seen,
                obs.trades_seen,
                json.dumps(obs.realized_metrics_json),
            )

            logger.debug(
                "observation_added",
                observation_id=str(row["id"]),
                record_id=str(obs.record_id),
                ts=obs.ts.isoformat(),
                bars_seen=obs.bars_seen,
                trades_seen=obs.trades_seen,
            )

            return row["id"]

    async def create_slice(self, slice_: EvaluationSlice) -> UUID:
        """
        Create evaluation slice.

        Args:
            slice_: EvaluationSlice to create

        Returns:
            Slice ID
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO recommendation_evaluation_slices (
                    record_id, slice_start_ts, slice_end_ts,
                    trigger_type, regime_key_during,
                    realized_summary_json, expected_summary_json,
                    performance_surprise_z, drift_flags_json
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                slice_.record_id,
                slice_.slice_start_ts,
                slice_.slice_end_ts,
                slice_.trigger_type,
                slice_.regime_key_during,
                json.dumps(slice_.realized_summary_json),
                json.dumps(slice_.expected_summary_json),
                slice_.performance_surprise_z,
                json.dumps(slice_.drift_flags_json) if slice_.drift_flags_json else None,
            )

            logger.info(
                "evaluation_slice_created",
                slice_id=str(row["id"]),
                record_id=str(slice_.record_id),
                trigger_type=slice_.trigger_type,
                regime_key=slice_.regime_key_during,
                performance_surprise_z=slice_.performance_surprise_z,
            )

            return row["id"]

    def _row_to_record(self, row: dict) -> RecommendationRecord:
        """
        Convert database row to RecommendationRecord.

        Handles JSON deserialization for dict fields.

        Args:
            row: Database row dict

        Returns:
            RecommendationRecord instance
        """
        return RecommendationRecord(
            id=row["id"],
            workspace_id=row["workspace_id"],
            strategy_entity_id=row["strategy_entity_id"],
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            params_json=(
                row["params_json"]
                if isinstance(row["params_json"], dict)
                else json.loads(row["params_json"])
            ),
            params_hash=row["params_hash"],
            regime_key_start=row["regime_key_start"],
            regime_dims_start=(
                row["regime_dims_start"]
                if isinstance(row["regime_dims_start"], dict)
                else json.loads(row["regime_dims_start"])
            ),
            regime_features_start=(
                row["regime_features_start"]
                if isinstance(row["regime_features_start"], dict)
                else json.loads(row["regime_features_start"])
            ),
            confidence_json=(
                row["confidence_json"]
                if isinstance(row["confidence_json"], dict)
                else json.loads(row["confidence_json"])
            ),
            expected_baselines_json=(
                row["expected_baselines_json"]
                if isinstance(row["expected_baselines_json"], dict)
                else json.loads(row["expected_baselines_json"])
            ),
            status=RecordStatus(row["status"]),
            schema_version=row.get("schema_version", 1),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
