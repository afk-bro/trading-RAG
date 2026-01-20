"""Repository for paper equity snapshots.

Append-only time series of paper broker equity state for drawdown computation.
Workspace-level equity tracking with optional strategy_version_id for future per-version isolation.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EquitySnapshot:
    """Paper equity snapshot model."""

    id: UUID
    workspace_id: UUID
    strategy_version_id: Optional[UUID]
    snapshot_ts: datetime
    computed_at: datetime
    equity: float
    cash: float
    positions_value: float
    realized_pnl: float
    inputs_hash: Optional[str]

    @classmethod
    def from_row(cls, row: dict) -> "EquitySnapshot":
        """Create from database row."""
        return cls(
            id=row["id"],
            workspace_id=row["workspace_id"],
            strategy_version_id=row.get("strategy_version_id"),
            snapshot_ts=row["snapshot_ts"],
            computed_at=row["computed_at"],
            equity=float(row["equity"]),
            cash=float(row["cash"]),
            positions_value=float(row["positions_value"]),
            realized_pnl=float(row["realized_pnl"]),
            inputs_hash=row.get("inputs_hash"),
        )


def compute_inputs_hash(
    workspace_id: UUID,
    cash: float,
    positions_value: float,
    realized_pnl: float,
) -> str:
    """Compute SHA256 hash of snapshot inputs for deduplication.

    Args:
        workspace_id: Workspace scope
        cash: Cash balance
        positions_value: Position values
        realized_pnl: Realized P&L

    Returns:
        64-character hex SHA256 hash
    """
    # Round floats to avoid floating point noise
    data = f"{workspace_id}:{cash:.6f}:{positions_value:.6f}:{realized_pnl:.6f}"
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class DrawdownResult:
    """Result of drawdown computation."""

    workspace_id: UUID
    current_equity: float
    peak_equity: float
    peak_ts: datetime
    current_ts: datetime
    drawdown_pct: float
    window_days: int
    snapshot_count: int


class PaperEquityRepository:
    """Repository for paper equity snapshot operations."""

    def __init__(self, pool):
        """Initialize with database connection pool."""
        self._pool = pool

    async def insert_snapshot(
        self,
        workspace_id: UUID,
        snapshot_ts: datetime,
        equity: float,
        cash: float,
        positions_value: float,
        realized_pnl: float,
        strategy_version_id: Optional[UUID] = None,
        skip_dedupe: bool = False,
    ) -> Optional[EquitySnapshot]:
        """Insert a new equity snapshot.

        Deduplication: If the latest snapshot for this workspace has the same
        inputs_hash, the insert is skipped to avoid spam.

        Args:
            workspace_id: Workspace scope
            snapshot_ts: Market time for the snapshot (bar close, trade time, etc.)
            equity: Total equity (cash + positions_value)
            cash: Available cash
            positions_value: Market value of positions
            realized_pnl: Cumulative realized P&L
            strategy_version_id: Optional version association
            skip_dedupe: If True, skip deduplication check

        Returns:
            Created EquitySnapshot, or None if dedupe skipped the insert
        """
        inputs_hash = compute_inputs_hash(workspace_id, cash, positions_value, realized_pnl)

        async with self._pool.acquire() as conn:
            # Dedupe check: skip if latest snapshot has same hash
            if not skip_dedupe:
                existing_hash = await conn.fetchval(
                    """
                    SELECT inputs_hash
                    FROM paper_equity_snapshots
                    WHERE workspace_id = $1
                      AND inputs_hash IS NOT NULL
                    ORDER BY snapshot_ts DESC
                    LIMIT 1
                    """,
                    workspace_id,
                )
                if existing_hash == inputs_hash:
                    logger.debug(
                        "equity_snapshot_dedupe_skip",
                        workspace_id=str(workspace_id),
                        inputs_hash=inputs_hash[:16],
                    )
                    return None

            row = await conn.fetchrow(
                """
                INSERT INTO paper_equity_snapshots (
                    workspace_id, strategy_version_id, snapshot_ts,
                    equity, cash, positions_value, realized_pnl, inputs_hash
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
                """,
                workspace_id,
                strategy_version_id,
                snapshot_ts,
                equity,
                cash,
                positions_value,
                realized_pnl,
                inputs_hash,
            )

            snapshot = EquitySnapshot.from_row(dict(row))
            logger.info(
                "equity_snapshot_inserted",
                snapshot_id=str(snapshot.id),
                workspace_id=str(workspace_id),
                equity=equity,
                cash=cash,
                positions_value=positions_value,
            )
            return snapshot

    async def get_latest_snapshot(
        self, workspace_id: UUID
    ) -> Optional[EquitySnapshot]:
        """Get the most recent snapshot for a workspace.

        Args:
            workspace_id: Workspace to query

        Returns:
            Latest EquitySnapshot or None if no snapshots exist
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM paper_equity_snapshots
                WHERE workspace_id = $1
                ORDER BY snapshot_ts DESC
                LIMIT 1
                """,
                workspace_id,
            )

            if row is None:
                return None

            return EquitySnapshot.from_row(dict(row))

    async def list_window(
        self,
        workspace_id: UUID,
        window_days: int = 30,
        limit: int = 1000,
    ) -> list[EquitySnapshot]:
        """List snapshots within a time window for drawdown computation.

        Returns snapshots in chronological order (oldest first) for peak computation.

        Args:
            workspace_id: Workspace to query
            window_days: Number of days to look back (default 30)
            limit: Maximum number of snapshots to return (default 1000)

        Returns:
            List of EquitySnapshots in chronological order
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM paper_equity_snapshots
                WHERE workspace_id = $1
                  AND snapshot_ts >= $2
                ORDER BY snapshot_ts ASC
                LIMIT $3
                """,
                workspace_id,
                cutoff,
                limit,
            )

            return [EquitySnapshot.from_row(dict(row)) for row in rows]

    async def compute_drawdown(
        self,
        workspace_id: UUID,
        window_days: int = 30,
    ) -> Optional[DrawdownResult]:
        """Compute current drawdown from peak within window.

        Args:
            workspace_id: Workspace to compute drawdown for
            window_days: Rolling window in days (default 30)

        Returns:
            DrawdownResult with peak, current, and drawdown_pct, or None if no data
        """
        snapshots = await self.list_window(workspace_id, window_days)

        if not snapshots:
            return None

        # Find peak equity and its timestamp
        peak_equity = 0.0
        peak_ts = snapshots[0].snapshot_ts
        for snap in snapshots:
            if snap.equity > peak_equity:
                peak_equity = snap.equity
                peak_ts = snap.snapshot_ts

        # Current is the latest snapshot
        current = snapshots[-1]

        # Compute drawdown percentage
        if peak_equity <= 0:
            drawdown_pct = 0.0
        else:
            drawdown_pct = (peak_equity - current.equity) / peak_equity

        return DrawdownResult(
            workspace_id=workspace_id,
            current_equity=current.equity,
            peak_equity=peak_equity,
            peak_ts=peak_ts,
            current_ts=current.snapshot_ts,
            drawdown_pct=drawdown_pct,
            window_days=window_days,
            snapshot_count=len(snapshots),
        )

    async def list_workspaces_with_snapshots(
        self,
        since_days: int = 7,
    ) -> list[UUID]:
        """List workspaces that have equity snapshots in the given period.

        Used by alert evaluator to find active paper trading workspaces.

        Args:
            since_days: Look back period (default 7 days)

        Returns:
            List of workspace UUIDs
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT workspace_id
                FROM paper_equity_snapshots
                WHERE snapshot_ts >= $1
                ORDER BY workspace_id
                """,
                cutoff,
            )

            return [row["workspace_id"] for row in rows]

    async def delete_old_snapshots(
        self,
        retention_days: int = 90,
    ) -> int:
        """Delete snapshots older than retention period.

        Retention cleanup for storage management.

        Args:
            retention_days: Delete snapshots older than this (default 90)

        Returns:
            Number of rows deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM paper_equity_snapshots
                WHERE snapshot_ts < $1
                """,
                cutoff,
            )

            # Parse "DELETE N" result
            count = int(result.split()[-1]) if result else 0
            if count > 0:
                logger.info(
                    "equity_snapshots_deleted",
                    count=count,
                    retention_days=retention_days,
                )
            return count
