"""Repository for artifact tracking.

Artifacts are generated outputs from tune/WFO jobs (e.g., tune_json, trials_csv,
equity_csv). This repository provides CRUD operations for the artifact_index table
and supports retention management via pin/unpin operations.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Artifact:
    """An artifact record tracking a generated output file."""

    id: UUID
    workspace_id: UUID
    run_id: UUID
    job_type: str  # 'tune' or 'wfo'
    artifact_kind: str  # 'tune_json', 'trials_csv', etc.
    artifact_path: str
    job_id: Optional[UUID] = None
    file_size_bytes: Optional[int] = None
    data_revision: Optional[dict] = None
    is_pinned: bool = False
    pinned_at: Optional[datetime] = None
    pinned_by: Optional[str] = None
    created_at: Optional[datetime] = None


class ArtifactRepository:
    """Repository for artifact CRUD operations."""

    def __init__(self, pool):
        self._pool = pool

    async def create(
        self,
        workspace_id: UUID,
        run_id: UUID,
        job_type: str,
        artifact_kind: str,
        artifact_path: str,
        job_id: Optional[UUID] = None,
        file_size_bytes: Optional[int] = None,
        data_revision: Optional[dict] = None,
    ) -> Artifact:
        """Create a new artifact record.

        Args:
            workspace_id: Workspace UUID
            run_id: Reference to tune_id or wfo_id
            job_type: Type of job ('tune' or 'wfo')
            artifact_kind: Type of artifact ('tune_json', 'trials_csv', etc.)
            artifact_path: Relative path under data/artifacts/
            job_id: Optional reference to jobs table
            file_size_bytes: Optional file size in bytes
            data_revision: Optional data revision metadata (checksum, row_count, etc.)

        Returns:
            The created Artifact record
        """
        query = """
            INSERT INTO artifact_index
                (workspace_id, run_id, job_type, artifact_kind, artifact_path,
                 job_id, file_size_bytes, data_revision)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *
        """
        # Convert dict to JSON string for JSONB column
        data_revision_json = json.dumps(data_revision) if data_revision else None

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                run_id,
                job_type,
                artifact_kind,
                artifact_path,
                job_id,
                file_size_bytes,
                data_revision_json,
            )

        logger.debug(
            "artifact_created",
            artifact_id=str(row["id"]),
            workspace_id=str(workspace_id),
            run_id=str(run_id),
            job_type=job_type,
            artifact_kind=artifact_kind,
        )

        return self._row_to_artifact(row)

    async def get_by_run(self, run_id: UUID) -> list[Artifact]:
        """Get all artifacts for a run (tune or wfo).

        Args:
            run_id: The tune_id or wfo_id

        Returns:
            List of Artifact records for the run
        """
        query = """
            SELECT * FROM artifact_index
            WHERE run_id = $1
            ORDER BY created_at ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, run_id)

        return [self._row_to_artifact(row) for row in rows]

    async def get_by_kind(self, run_id: UUID, artifact_kind: str) -> Optional[Artifact]:
        """Get specific artifact by kind.

        Args:
            run_id: The tune_id or wfo_id
            artifact_kind: The artifact type to retrieve

        Returns:
            Artifact if found, None otherwise
        """
        query = """
            SELECT * FROM artifact_index
            WHERE run_id = $1 AND artifact_kind = $2
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, run_id, artifact_kind)

        return self._row_to_artifact(row) if row else None

    async def pin(self, artifact_id: UUID, pinned_by: str) -> bool:
        """Pin an artifact for retention.

        Pinned artifacts are retained indefinitely and excluded from cleanup.

        Args:
            artifact_id: The artifact UUID to pin
            pinned_by: Identifier of who pinned (email, username, etc.)

        Returns:
            True if artifact was pinned, False if not found
        """
        query = """
            UPDATE artifact_index
            SET is_pinned = true,
                pinned_at = now(),
                pinned_by = $2
            WHERE id = $1
            RETURNING true
        """
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(query, artifact_id, pinned_by)

        if result:
            logger.info(
                "artifact_pinned",
                artifact_id=str(artifact_id),
                pinned_by=pinned_by,
            )

        return result is True

    async def unpin(self, artifact_id: UUID) -> bool:
        """Unpin an artifact.

        Args:
            artifact_id: The artifact UUID to unpin

        Returns:
            True if artifact was unpinned, False if not found
        """
        query = """
            UPDATE artifact_index
            SET is_pinned = false,
                pinned_at = NULL,
                pinned_by = NULL
            WHERE id = $1
            RETURNING true
        """
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(query, artifact_id)

        if result:
            logger.info(
                "artifact_unpinned",
                artifact_id=str(artifact_id),
            )

        return result is True

    async def list_unpinned_older_than(self, days: int) -> list[Artifact]:
        """List unpinned artifacts older than N days.

        Used for retention cleanup to identify artifacts eligible for deletion.

        Args:
            days: Minimum age in days for artifacts to be listed

        Returns:
            List of unpinned Artifact records older than the threshold
        """
        query = """
            SELECT * FROM artifact_index
            WHERE is_pinned = false
              AND created_at < now() - ($1 || ' days')::interval
            ORDER BY created_at ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, str(days))

        return [self._row_to_artifact(row) for row in rows]

    async def delete(self, artifact_id: UUID) -> bool:
        """Delete an artifact record.

        Note: This only deletes the database record. Actual file deletion
        should be handled separately by the caller.

        Args:
            artifact_id: The artifact UUID to delete

        Returns:
            True if artifact was deleted, False if not found
        """
        query = """
            DELETE FROM artifact_index
            WHERE id = $1
            RETURNING true
        """
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(query, artifact_id)

        if result:
            logger.info(
                "artifact_deleted",
                artifact_id=str(artifact_id),
            )

        return result is True

    def _row_to_artifact(self, row) -> Artifact:
        """Convert a database row to an Artifact model."""
        return Artifact(
            id=row["id"],
            workspace_id=row["workspace_id"],
            run_id=row["run_id"],
            job_type=row["job_type"],
            artifact_kind=row["artifact_kind"],
            artifact_path=row["artifact_path"],
            job_id=row["job_id"],
            file_size_bytes=row["file_size_bytes"],
            data_revision=row["data_revision"],
            is_pinned=row["is_pinned"],
            pinned_at=row["pinned_at"],
            pinned_by=row["pinned_by"],
            created_at=row["created_at"],
        )
