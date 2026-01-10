"""Repository adapters for KB trials admin endpoints.

These implement the Protocol interfaces from kb/status_service.py
for use with the admin router's database pool.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import structlog

from app.services.kb.status_service import CurrentStatus

logger = structlog.get_logger(__name__)


class AdminKBStatusRepository:
    """Repository for KB status operations in admin context."""

    def __init__(self, pool):
        self._pool = pool

    async def get_current_status(
        self,
        source_type: str,
        source_id: UUID,
    ) -> Optional[CurrentStatus]:
        """Get current status for a trial.

        Returns CurrentStatus or None if not found.
        """
        if source_type == "tune_run":
            query = """
                SELECT
                    t.workspace_id,
                    tr.kb_status,
                    tr.kb_promoted_at
                FROM backtest_tune_runs tr
                JOIN backtest_tunes t ON tr.tune_id = t.id
                WHERE tr.run_id = $1
            """
        else:  # test_variant
            query = """
                SELECT
                    workspace_id,
                    kb_status,
                    kb_promoted_at
                FROM backtest_runs
                WHERE id = $1 AND run_kind = 'test_variant'
            """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, source_id)
            if row:
                return CurrentStatus(
                    workspace_id=row["workspace_id"],
                    kb_status=row["kb_status"] or "excluded",
                    kb_promoted_at=row["kb_promoted_at"],
                )
            return None

    async def update_status(
        self,
        source_type: str,
        source_id: UUID,
        to_status: str,
        changed_by: Optional[str],
        reason: Optional[str],
    ) -> None:
        """Update kb_status for a trial."""
        now = datetime.now(timezone.utc)

        if source_type == "tune_run":
            query = """
                UPDATE backtest_tune_runs
                SET
                    kb_status = $2,
                    kb_status_changed_at = $3,
                    kb_status_changed_by = $4,
                    kb_status_reason = $5
                WHERE run_id = $1
            """
        else:  # test_variant
            query = """
                UPDATE backtest_runs
                SET
                    kb_status = $2,
                    kb_status_changed_at = $3,
                    kb_status_changed_by = $4,
                    kb_status_reason = $5
                WHERE id = $1 AND run_kind = 'test_variant'
            """

        async with self._pool.acquire() as conn:
            await conn.execute(query, source_id, to_status, now, changed_by, reason)

    async def set_promoted_at(
        self,
        source_type: str,
        source_id: UUID,
        promoted_by: Optional[str],
    ) -> None:
        """Set kb_promoted_at timestamp."""
        now = datetime.now(timezone.utc)

        if source_type == "tune_run":
            query = """
                UPDATE backtest_tune_runs
                SET kb_promoted_at = $2, kb_promoted_by = $3
                WHERE run_id = $1
            """
        else:  # test_variant
            query = """
                UPDATE backtest_runs
                SET kb_promoted_at = $2, kb_promoted_by = $3
                WHERE id = $1 AND run_kind = 'test_variant'
            """

        async with self._pool.acquire() as conn:
            await conn.execute(query, source_id, now, promoted_by)

    async def insert_history(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
        from_status: str,
        to_status: str,
        actor_type: str,
        actor_id: Optional[str],
        reason: Optional[str],
    ) -> None:
        """Insert audit log entry."""
        query = """
            INSERT INTO kb_status_history (
                workspace_id,
                source_type,
                source_id,
                from_status,
                to_status,
                actor_type,
                actor_id,
                reason
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                workspace_id,
                source_type,
                source_id,
                from_status,
                to_status,
                actor_type,
                actor_id,
                reason,
            )


class AdminKBIndexRepository:
    """Repository for KB index operations in admin context."""

    def __init__(self, pool):
        self._pool = pool

    async def archive_trial(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
        reason: str,
        actor: Optional[str],
    ) -> bool:
        """Archive a trial in kb_trial_index.

        Returns True if archived, False if not found.
        """
        now = datetime.now(timezone.utc)

        query = """
            UPDATE kb_trial_index
            SET
                archived_at = $4,
                archived_reason = $5,
                archived_by = $6
            WHERE workspace_id = $1
              AND source_type = $2
              AND source_id = $3
              AND archived_at IS NULL
            RETURNING id
        """

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(
                query, workspace_id, source_type, source_id, now, reason, actor
            )
            return result is not None

    async def unarchive_trial(
        self,
        source_type: str,
        source_id: UUID,
    ) -> bool:
        """Unarchive a trial in kb_trial_index.

        Clears archived_at, reason, and by fields.
        Returns True if unarchived, False if not found or not archived.
        """
        query = """
            UPDATE kb_trial_index
            SET
                archived_at = NULL,
                archived_reason = NULL,
                archived_by = NULL
            WHERE source_type = $1
              AND source_id = $2
              AND archived_at IS NOT NULL
            RETURNING id
        """

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(query, source_type, source_id)
            return result is not None

    async def get_index_entry(
        self,
        workspace_id: UUID,
        source_type: str,
        source_id: UUID,
    ) -> Optional[dict]:
        """Get index entry for a trial."""
        query = """
            SELECT
                id,
                workspace_id,
                source_type,
                source_id,
                qdrant_point_id,
                content_hash,
                content_hash_algo,
                embed_model,
                collection_name,
                ingested_at,
                archived_at,
                archived_reason,
                archived_by
            FROM kb_trial_index
            WHERE workspace_id = $1
              AND source_type = $2
              AND source_id = $3
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id, source_type, source_id)
            if row:
                return dict(row)
            return None
