"""Repository for strategy intelligence snapshots.

Append-only time series of regime + confidence intel per strategy version.
No business logic - just CRUD operations. Computation owned by v1.5 Step 2.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class IntelSnapshot:
    """Strategy intelligence snapshot model."""

    id: UUID
    workspace_id: UUID
    strategy_version_id: UUID
    as_of_ts: datetime
    computed_at: datetime
    regime: str
    confidence_score: float
    confidence_components: dict
    features: dict
    explain: dict
    engine_version: Optional[str]
    inputs_hash: Optional[str]
    run_id: Optional[UUID]

    @classmethod
    def from_row(cls, row: dict) -> "IntelSnapshot":
        """Create from database row."""
        # Handle JSONB fields
        components = row.get("confidence_components", {})
        if isinstance(components, str):
            components = json.loads(components)

        features = row.get("features", {})
        if isinstance(features, str):
            features = json.loads(features)

        explain = row.get("explain", {})
        if isinstance(explain, str):
            explain = json.loads(explain)

        return cls(
            id=row["id"],
            workspace_id=row["workspace_id"],
            strategy_version_id=row["strategy_version_id"],
            as_of_ts=row["as_of_ts"],
            computed_at=row["computed_at"],
            regime=row["regime"],
            confidence_score=float(row["confidence_score"]),
            confidence_components=components or {},
            features=features or {},
            explain=explain or {},
            engine_version=row.get("engine_version"),
            inputs_hash=row.get("inputs_hash"),
            run_id=row.get("run_id"),
        )


class StrategyIntelRepository:
    """Repository for strategy intelligence snapshot operations."""

    def __init__(self, pool):
        """Initialize with database connection pool."""
        self._pool = pool

    async def insert_snapshot(
        self,
        workspace_id: UUID,
        strategy_version_id: UUID,
        as_of_ts: datetime,
        regime: str,
        confidence_score: float,
        confidence_components: Optional[dict] = None,
        features: Optional[dict] = None,
        explain: Optional[dict] = None,
        engine_version: Optional[str] = None,
        inputs_hash: Optional[str] = None,
        run_id: Optional[UUID] = None,
    ) -> IntelSnapshot:
        """Insert a new intelligence snapshot.

        Args:
            workspace_id: Workspace scope
            strategy_version_id: Strategy version this intel is for
            as_of_ts: Market time the intel refers to (e.g., bar close time)
            regime: Current regime classification
            confidence_score: Aggregated confidence [0, 1]
            confidence_components: Breakdown of confidence factors
            features: Raw feature values used for computation
            explain: Human-readable explanation
            engine_version: Version of computation engine
            inputs_hash: SHA256 of inputs for deduplication
            run_id: Link to job/workflow run

        Returns:
            Created IntelSnapshot

        Raises:
            ValueError: If confidence_score outside [0, 1]
        """
        if not 0.0 <= confidence_score <= 1.0:
            raise ValueError(f"confidence_score must be in [0, 1], got {confidence_score}")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO strategy_intel_snapshots (
                    workspace_id, strategy_version_id, as_of_ts, regime,
                    confidence_score, confidence_components, features,
                    explain, engine_version, inputs_hash, run_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING *
                """,
                workspace_id,
                strategy_version_id,
                as_of_ts,
                regime,
                confidence_score,
                json.dumps(confidence_components or {}),
                json.dumps(features or {}),
                json.dumps(explain or {}),
                engine_version,
                inputs_hash,
                run_id,
            )

            snapshot = IntelSnapshot.from_row(dict(row))
            logger.info(
                "intel_snapshot_inserted",
                snapshot_id=str(snapshot.id),
                version_id=str(strategy_version_id),
                regime=regime,
                confidence=confidence_score,
            )
            return snapshot

    async def get_latest_snapshot(
        self, strategy_version_id: UUID
    ) -> Optional[IntelSnapshot]:
        """Get the most recent snapshot for a strategy version.

        Uses as_of_ts for ordering (market time, not compute time).

        Args:
            strategy_version_id: Strategy version to query

        Returns:
            Latest IntelSnapshot or None if no snapshots exist
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM strategy_intel_snapshots
                WHERE strategy_version_id = $1
                ORDER BY as_of_ts DESC
                LIMIT 1
                """,
                strategy_version_id,
            )

            if row is None:
                return None

            return IntelSnapshot.from_row(dict(row))

    async def list_snapshots(
        self,
        strategy_version_id: UUID,
        limit: int = 50,
        cursor: Optional[datetime] = None,
    ) -> list[IntelSnapshot]:
        """List snapshots for a strategy version with cursor-based pagination.

        Returns snapshots in reverse chronological order (newest first).

        Args:
            strategy_version_id: Strategy version to query
            limit: Maximum number of snapshots to return (default 50, max 200)
            cursor: as_of_ts to start after (exclusive, for pagination)

        Returns:
            List of IntelSnapshots
        """
        limit = min(limit, 200)  # Cap at 200

        async with self._pool.acquire() as conn:
            if cursor is None:
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM strategy_intel_snapshots
                    WHERE strategy_version_id = $1
                    ORDER BY as_of_ts DESC
                    LIMIT $2
                    """,
                    strategy_version_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM strategy_intel_snapshots
                    WHERE strategy_version_id = $1
                      AND as_of_ts < $2
                    ORDER BY as_of_ts DESC
                    LIMIT $3
                    """,
                    strategy_version_id,
                    cursor,
                    limit,
                )

            return [IntelSnapshot.from_row(dict(row)) for row in rows]

    async def list_workspace_snapshots(
        self,
        workspace_id: UUID,
        limit: int = 50,
        cursor: Optional[datetime] = None,
    ) -> list[IntelSnapshot]:
        """List snapshots across all versions in a workspace.

        Useful for dashboard views showing all recent intel.

        Args:
            workspace_id: Workspace to query
            limit: Maximum number of snapshots to return (default 50, max 200)
            cursor: as_of_ts to start after (exclusive, for pagination)

        Returns:
            List of IntelSnapshots
        """
        limit = min(limit, 200)

        async with self._pool.acquire() as conn:
            if cursor is None:
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM strategy_intel_snapshots
                    WHERE workspace_id = $1
                    ORDER BY as_of_ts DESC
                    LIMIT $2
                    """,
                    workspace_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT *
                    FROM strategy_intel_snapshots
                    WHERE workspace_id = $1
                      AND as_of_ts < $2
                    ORDER BY as_of_ts DESC
                    LIMIT $3
                    """,
                    workspace_id,
                    cursor,
                    limit,
                )

            return [IntelSnapshot.from_row(dict(row)) for row in rows]
