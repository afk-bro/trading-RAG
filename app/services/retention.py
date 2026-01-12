"""Event retention and cleanup service."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class RetentionService:
    """Service for managing event retention."""

    # Retention periods
    INFO_DEBUG_DAYS = 30
    WARN_ERROR_DAYS = 90

    async def preview_cleanup(self, conn: Any, workspace_id: UUID) -> dict[str, int]:
        """
        Preview what would be deleted without making changes.

        Args:
            conn: Database connection (must be passed in for JobRunner pattern)
            workspace_id: Workspace to scope the cleanup

        Returns:
            Dict with counts of events that would be deleted
        """
        now = datetime.now(timezone.utc)

        info_cutoff = now - timedelta(days=self.INFO_DEBUG_DAYS)
        info_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM trade_events
            WHERE workspace_id = $1
              AND created_at < $2
              AND severity IN ('debug', 'info')
              AND pinned = FALSE
            """,
            workspace_id,
            info_cutoff,
        )

        error_cutoff = now - timedelta(days=self.WARN_ERROR_DAYS)
        error_count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM trade_events
            WHERE workspace_id = $1
              AND created_at < $2
              AND severity IN ('warn', 'error')
              AND pinned = FALSE
            """,
            workspace_id,
            error_cutoff,
        )

        return {
            "info_debug_would_delete": info_count or 0,
            "warn_error_would_delete": error_count or 0,
        }

    async def run_cleanup(self, conn: Any, workspace_id: UUID) -> dict[str, int]:
        """
        Delete expired events based on severity tier.

        Retention policy:
        - INFO/DEBUG: 30 days
        - WARN/ERROR: 90 days
        - Pinned: Never deleted

        Args:
            conn: Database connection (must be passed in for JobRunner pattern)
            workspace_id: Workspace to scope the cleanup

        Returns:
            Dict with counts of deleted events per tier
        """
        now = datetime.now(timezone.utc)

        # Delete INFO/DEBUG older than 30 days (not pinned)
        info_cutoff = now - timedelta(days=self.INFO_DEBUG_DAYS)
        info_result = await conn.execute(
            """
            DELETE FROM trade_events
            WHERE workspace_id = $1
              AND created_at < $2
              AND severity IN ('debug', 'info')
              AND pinned = FALSE
            """,
            workspace_id,
            info_cutoff,
        )

        # Delete WARN/ERROR older than 90 days (not pinned)
        error_cutoff = now - timedelta(days=self.WARN_ERROR_DAYS)
        error_result = await conn.execute(
            """
            DELETE FROM trade_events
            WHERE workspace_id = $1
              AND created_at < $2
              AND severity IN ('warn', 'error')
              AND pinned = FALSE
            """,
            workspace_id,
            error_cutoff,
        )

        info_deleted = int(info_result.split()[-1])
        error_deleted = int(error_result.split()[-1])

        logger.info(
            "Retention cleanup complete",
            workspace_id=str(workspace_id),
            info_debug_deleted=info_deleted,
            warn_error_deleted=error_deleted,
        )

        return {
            "info_debug_deleted": info_deleted,
            "warn_error_deleted": error_deleted,
        }
