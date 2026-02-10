"""Repository for backtest run events (replay data)."""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class RunEventsRepository:
    """CRUD for backtest_run_events table."""

    def __init__(self, pool):
        self.pool = pool

    async def get_events(
        self,
        run_id: UUID,
        workspace_id: UUID,
    ) -> Optional[list[dict[str, Any]]]:
        """Get events for a run, workspace-scoped.

        Returns None if run not found, or list of event dicts.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT events
                FROM backtest_run_events
                WHERE run_id = $1 AND workspace_id = $2
                """,
                run_id,
                workspace_id,
            )

        if not row:
            return None

        raw = row["events"]
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    async def save_events(
        self,
        run_id: UUID,
        workspace_id: UUID,
        events: list[dict[str, Any]],
    ) -> None:
        """Upsert events for a run.

        Idempotent: re-calling with the same run_id overwrites events and
        bumps updated_at. The workspace_id on update is verified to match
        (WHERE clause ensures no cross-workspace overwrites).
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO backtest_run_events
                    (run_id, workspace_id, events, updated_at)
                VALUES ($1, $2, $3::jsonb, now())
                ON CONFLICT (run_id) DO UPDATE
                    SET events = EXCLUDED.events,
                        updated_at = now()
                WHERE backtest_run_events.workspace_id = EXCLUDED.workspace_id
                """,
                run_id,
                workspace_id,
                json.dumps(events),
            )

    async def get_event_count(
        self,
        run_id: UUID,
        workspace_id: UUID,
    ) -> int:
        """Get event count for a run without loading full events."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT event_count
                FROM backtest_run_events
                WHERE run_id = $1 AND workspace_id = $2
                """,
                run_id,
                workspace_id,
            )
        return count or 0
