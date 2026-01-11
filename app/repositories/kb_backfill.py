"""
Repository for KB backfill run tracking.

Provides CRUD operations for kb_backfill_runs table:
- Start/complete/fail tracking
- Resume from last cursor
- Progress updates

Progress Update Semantics:
    Cursor is updated AFTER successful processing of each batch.
    This means on resume, items up to and including the cursor have been
    processed. The query uses `id > cursor::uuid` to skip already-processed items.

    If a failure occurs mid-batch:
    - Items before the last cursor update are complete
    - Items after may need reprocessing (idempotent writes assumed)
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


def _compute_config_hash(config: dict) -> str:
    """
    Compute SHA256 hash of canonical JSON config.

    Canonical form: sorted keys, no whitespace, consistent serialization.
    This ensures stable matching regardless of dict key ordering or
    formatting differences.

    Args:
        config: Configuration dict to hash

    Returns:
        Hex-encoded SHA256 hash (64 chars)
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


BackfillType = Literal["candidacy", "regime"]
BackfillStatus = Literal["running", "completed", "failed"]


@dataclass
class BackfillRun:
    """Represents a KB backfill run for tracking and resume."""

    id: UUID
    workspace_id: UUID
    backfill_type: BackfillType
    status: BackfillStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    last_processed_cursor: Optional[str] = None
    config: dict = field(default_factory=dict)
    error: Optional[str] = None
    dry_run: bool = False


class BackfillRunRepository:
    """
    Repository for kb_backfill_runs table.

    Provides:
    - create: Start a new backfill run
    - update_progress: Update counts and cursor during processing
    - complete: Mark run as completed
    - fail: Mark run as failed with error
    - find_resumable: Find most recent resumable run
    - get_latest: Get most recent run for workspace/type
    """

    def __init__(self, pool):
        """Initialize repository with asyncpg pool."""
        self.pool = pool

    async def create(
        self,
        workspace_id: UUID,
        backfill_type: BackfillType,
        config: dict,
        dry_run: bool = False,
    ) -> BackfillRun:
        """
        Create a new backfill run.

        Args:
            workspace_id: Target workspace
            backfill_type: Type of backfill ('candidacy' or 'regime')
            config: Configuration snapshot (since, limit, etc.)
            dry_run: Whether this is a dry-run

        Returns:
            Created BackfillRun with generated ID

        Raises:
            asyncpg.UniqueViolationError: If a run with same config is already running
                (concurrency guard via partial unique index)
        """
        config_hash = _compute_config_hash(config)

        query = """
            INSERT INTO kb_backfill_runs (
                workspace_id, backfill_type, status, config, config_hash, dry_run
            ) VALUES ($1, $2, 'running', $3, $4, $5)
            RETURNING id, started_at
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                backfill_type,
                json.dumps(config),
                config_hash,
                dry_run,
            )

        run = BackfillRun(
            id=row["id"],
            workspace_id=workspace_id,
            backfill_type=backfill_type,
            status="running",
            started_at=row["started_at"],
            config=config,
            dry_run=dry_run,
        )

        logger.info(
            "backfill_run_created",
            run_id=str(run.id),
            workspace_id=str(workspace_id),
            backfill_type=backfill_type,
            dry_run=dry_run,
        )

        return run

    async def update_progress(
        self,
        run_id: UUID,
        processed_count: int,
        skipped_count: int,
        error_count: int,
        last_processed_cursor: Optional[str] = None,
    ) -> None:
        """
        Update progress counters and cursor.

        Call periodically during processing for resume support.

        Args:
            run_id: Backfill run ID
            processed_count: Total processed so far
            skipped_count: Total skipped so far
            error_count: Total errors so far
            last_processed_cursor: Cursor for resume (entity ID, timestamp, etc.)
        """
        query = """
            UPDATE kb_backfill_runs SET
                processed_count = $2,
                skipped_count = $3,
                error_count = $4,
                last_processed_cursor = $5
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run_id,
                processed_count,
                skipped_count,
                error_count,
                last_processed_cursor,
            )

    async def complete(
        self,
        run_id: UUID,
        processed_count: int,
        skipped_count: int,
        error_count: int,
    ) -> None:
        """
        Mark run as completed.

        Args:
            run_id: Backfill run ID
            processed_count: Final processed count
            skipped_count: Final skipped count
            error_count: Final error count
        """
        query = """
            UPDATE kb_backfill_runs SET
                status = 'completed',
                completed_at = NOW(),
                processed_count = $2,
                skipped_count = $3,
                error_count = $4
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query, run_id, processed_count, skipped_count, error_count
            )

        logger.info(
            "backfill_run_completed",
            run_id=str(run_id),
            processed_count=processed_count,
            skipped_count=skipped_count,
            error_count=error_count,
        )

    async def fail(
        self,
        run_id: UUID,
        error: str,
        processed_count: int,
        skipped_count: int,
        error_count: int,
        last_processed_cursor: Optional[str] = None,
    ) -> None:
        """
        Mark run as failed.

        Preserves progress for potential resume.

        Args:
            run_id: Backfill run ID
            error: Error message
            processed_count: Processed count at failure
            skipped_count: Skipped count at failure
            error_count: Error count at failure
            last_processed_cursor: Last successfully processed cursor
        """
        query = """
            UPDATE kb_backfill_runs SET
                status = 'failed',
                completed_at = NOW(),
                error = $2,
                processed_count = $3,
                skipped_count = $4,
                error_count = $5,
                last_processed_cursor = $6
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run_id,
                error,
                processed_count,
                skipped_count,
                error_count,
                last_processed_cursor,
            )

        logger.warning(
            "backfill_run_failed",
            run_id=str(run_id),
            error=error,
            processed_count=processed_count,
        )

    async def find_resumable(
        self,
        workspace_id: UUID,
        backfill_type: BackfillType,
        config: dict,
    ) -> Optional[BackfillRun]:
        """
        Find the most recent resumable run.

        A run is resumable if:
        - Same workspace_id, backfill_type, and config_hash (or config for legacy)
        - Status is 'running' or 'failed'
        - Has a last_processed_cursor

        Matching priority:
        1. Try config_hash match first (fast, indexed)
        2. Fall back to config JSONB equality (for legacy rows without hash)

        The most recent (by started_at DESC) matching run wins.

        Args:
            workspace_id: Target workspace
            backfill_type: Type of backfill
            config: Configuration to match

        Returns:
            BackfillRun if resumable run found, None otherwise
        """
        config_hash = _compute_config_hash(config)

        # Query matches on config_hash OR config equality (fallback for legacy rows)
        query = """
            SELECT
                id, workspace_id, backfill_type, status,
                started_at, completed_at,
                processed_count, skipped_count, error_count,
                last_processed_cursor, config, error, dry_run
            FROM kb_backfill_runs
            WHERE workspace_id = $1
              AND backfill_type = $2
              AND (config_hash = $3 OR (config_hash IS NULL AND config = $4))
              AND status IN ('running', 'failed')
              AND last_processed_cursor IS NOT NULL
            ORDER BY started_at DESC
            LIMIT 1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                backfill_type,
                config_hash,
                json.dumps(config),
            )

        if row is None:
            return None

        return self._row_to_run(row)

    async def get_latest(
        self,
        workspace_id: UUID,
        backfill_type: BackfillType,
    ) -> Optional[BackfillRun]:
        """
        Get the most recent run for workspace/type.

        Args:
            workspace_id: Target workspace
            backfill_type: Type of backfill

        Returns:
            Most recent BackfillRun or None
        """
        query = """
            SELECT
                id, workspace_id, backfill_type, status,
                started_at, completed_at,
                processed_count, skipped_count, error_count,
                last_processed_cursor, config, error, dry_run
            FROM kb_backfill_runs
            WHERE workspace_id = $1
              AND backfill_type = $2
            ORDER BY started_at DESC
            LIMIT 1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id, backfill_type)

        if row is None:
            return None

        return self._row_to_run(row)

    async def list_recent(
        self,
        workspace_id: Optional[UUID] = None,
        limit: int = 20,
    ) -> list[BackfillRun]:
        """
        List recent backfill runs.

        Args:
            workspace_id: Optional filter by workspace
            limit: Maximum number of runs to return

        Returns:
            List of BackfillRun, most recent first
        """
        if workspace_id:
            query = """
                SELECT
                    id, workspace_id, backfill_type, status,
                    started_at, completed_at,
                    processed_count, skipped_count, error_count,
                    last_processed_cursor, config, error, dry_run
                FROM kb_backfill_runs
                WHERE workspace_id = $1
                ORDER BY started_at DESC
                LIMIT $2
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, workspace_id, limit)
        else:
            query = """
                SELECT
                    id, workspace_id, backfill_type, status,
                    started_at, completed_at,
                    processed_count, skipped_count, error_count,
                    last_processed_cursor, config, error, dry_run
                FROM kb_backfill_runs
                ORDER BY started_at DESC
                LIMIT $1
            """
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, limit)

        return [self._row_to_run(row) for row in rows]

    def _row_to_run(self, row: dict) -> BackfillRun:
        """Convert database row to BackfillRun."""
        config = row["config"]
        if isinstance(config, str):
            config = json.loads(config)

        return BackfillRun(
            id=row["id"],
            workspace_id=row["workspace_id"],
            backfill_type=row["backfill_type"],
            status=row["status"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            processed_count=row["processed_count"],
            skipped_count=row["skipped_count"],
            error_count=row["error_count"],
            last_processed_cursor=row["last_processed_cursor"],
            config=config,
            error=row["error"],
            dry_run=row["dry_run"],
        )
