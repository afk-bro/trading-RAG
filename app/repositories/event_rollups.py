"""Repository for trade event rollups."""

from datetime import date
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class EventRollupsRepository:
    """Repository for event rollup operations."""

    async def preview_daily_rollup(
        self, conn: Any, workspace_id: UUID, target_date: date
    ) -> dict[str, int]:
        """
        Preview what would be aggregated without making changes.

        Args:
            conn: Database connection (must be passed in for JobRunner pattern)
            workspace_id: Workspace to scope the rollup
            target_date: Date to aggregate

        Returns:
            Dict with preview counts
        """
        # Count events that would be aggregated
        event_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM trade_events
            WHERE workspace_id = $1
              AND created_at >= $2::date
              AND created_at < ($2::date + INTERVAL '1 day')
            """,
            workspace_id,
            target_date,
        )

        # Count distinct groupings (how many rollup rows would be created)
        group_count = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT (strategy_entity_id, event_type))
            FROM trade_events
            WHERE workspace_id = $1
              AND created_at >= $2::date
              AND created_at < ($2::date + INTERVAL '1 day')
            """,
            workspace_id,
            target_date,
        )

        return {
            "events_to_aggregate": event_count or 0,
            "rollup_rows_to_create": group_count or 0,
        }

    async def run_daily_rollup(
        self, conn: Any, workspace_id: UUID, target_date: date
    ) -> int:
        """
        Aggregate events from target_date into rollups for a workspace.

        Idempotent via ON CONFLICT - safe to run multiple times.

        Args:
            conn: Database connection (must be passed in for JobRunner pattern)
            workspace_id: Workspace to scope the rollup
            target_date: Date to aggregate

        Returns:
            Number of rollup rows upserted
        """
        query = """
            INSERT INTO trade_event_rollups (
                workspace_id, strategy_entity_id, event_type, rollup_date,
                event_count, error_count, sample_correlation_ids
            )
            SELECT
                workspace_id,
                strategy_entity_id,
                event_type,
                $2::date as rollup_date,
                COUNT(*) as event_count,
                COUNT(*) FILTER (WHERE severity = 'error') as error_count,
                (ARRAY_AGG(DISTINCT correlation_id)
                    FILTER (WHERE correlation_id IS NOT NULL))[1:5]
            FROM trade_events
            WHERE workspace_id = $1
              AND created_at >= $2::date
              AND created_at < ($2::date + INTERVAL '1 day')
            GROUP BY workspace_id, strategy_entity_id, event_type
            ON CONFLICT (workspace_id, strategy_entity_id, event_type, rollup_date)
            DO UPDATE SET
                event_count = EXCLUDED.event_count,
                error_count = EXCLUDED.error_count,
                sample_correlation_ids = EXCLUDED.sample_correlation_ids
        """

        result = await conn.execute(query, workspace_id, target_date)

        # Parse "INSERT 0 N" or "UPDATE N"
        count = int(result.split()[-1])
        logger.info(
            "Daily rollup complete",
            workspace_id=str(workspace_id),
            date=str(target_date),
            rows=count,
        )
        return count

    async def get_rollups(
        self,
        workspace_id: UUID,
        start_date: date,
        end_date: date,
        strategy_entity_id: Optional[UUID] = None,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Get rollups for a workspace in date range.

        Args:
            workspace_id: Workspace ID
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            strategy_entity_id: Optional filter by strategy
            event_type: Optional filter by event type

        Returns:
            List of rollup records
        """
        conditions = ["workspace_id = $1", "rollup_date >= $2", "rollup_date <= $3"]
        params: list[Any] = [workspace_id, start_date, end_date]
        param_idx = 4

        if strategy_entity_id:
            conditions.append(f"strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if event_type:
            conditions.append(f"event_type = ${param_idx}")
            params.append(event_type)
            param_idx += 1

        where = " AND ".join(conditions)
        query = f"""
            SELECT
                id, workspace_id, strategy_entity_id, event_type,
                rollup_date, event_count, error_count, sample_correlation_ids,
                created_at
            FROM trade_event_rollups
            WHERE {where}
            ORDER BY rollup_date DESC, event_type
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]
