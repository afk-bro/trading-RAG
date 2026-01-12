"""Event retention and cleanup service."""

from datetime import datetime, timedelta, timezone

import structlog

logger = structlog.get_logger(__name__)


class RetentionService:
    """Service for managing event retention."""

    # Retention periods
    INFO_DEBUG_DAYS = 30
    WARN_ERROR_DAYS = 90

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def run_cleanup(self) -> dict[str, int]:
        """
        Delete expired events based on severity tier.

        Retention policy:
        - INFO/DEBUG: 30 days
        - WARN/ERROR: 90 days
        - Pinned: Never deleted

        Returns:
            Dict with counts of deleted events per tier
        """
        now = datetime.now(timezone.utc)

        async with self.pool.acquire() as conn:
            # Delete INFO/DEBUG older than 30 days (not pinned)
            info_cutoff = now - timedelta(days=self.INFO_DEBUG_DAYS)
            info_result = await conn.execute(
                """
                DELETE FROM trade_events
                WHERE created_at < $1
                  AND severity IN ('debug', 'info')
                  AND pinned = FALSE
                """,
                info_cutoff,
            )

            # Delete WARN/ERROR older than 90 days (not pinned)
            error_cutoff = now - timedelta(days=self.WARN_ERROR_DAYS)
            error_result = await conn.execute(
                """
                DELETE FROM trade_events
                WHERE created_at < $1
                  AND severity IN ('warn', 'error')
                  AND pinned = FALSE
                """,
                error_cutoff,
            )

        info_deleted = int(info_result.split()[-1])
        error_deleted = int(error_result.split()[-1])

        logger.info(
            "Retention cleanup complete",
            info_debug_deleted=info_deleted,
            warn_error_deleted=error_deleted,
        )

        return {
            "info_debug_deleted": info_deleted,
            "warn_error_deleted": error_deleted,
        }
