"""Repository for strategy_scripts table operations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StrategyScript:
    """Model for strategy_scripts table row."""

    id: UUID
    workspace_id: UUID
    rel_path: str
    source_type: str
    sha256: str
    pine_version: Optional[str] = None
    script_type: Optional[str] = None
    title: Optional[str] = None
    status: str = "discovered"
    spec_json: Optional[dict] = None
    spec_generated_at: Optional[datetime] = None
    lint_json: Optional[dict] = None
    strategy_id: Optional[UUID] = None
    published_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    # Ingest tracking fields
    doc_id: Optional[UUID] = None
    last_ingested_at: Optional[datetime] = None
    last_ingested_sha: Optional[str] = None
    ingest_status: Optional[str] = None  # 'ok', 'error', or None
    ingest_error: Optional[str] = None

    def canonical_url(self) -> str:
        """Derive canonical URL from source type and path."""
        normalized = self.rel_path.replace("\\", "/").lstrip("/")
        normalized = "/".join(p for p in normalized.split("/") if p not in (".", ".."))
        return f"pine://{self.source_type}/{normalized}"

    def needs_ingest(self) -> bool:
        """Check if script needs (re-)ingestion based on content hash."""
        if self.ingest_status is None:
            return True  # Never ingested
        if self.last_ingested_sha != self.sha256:
            return True  # Content changed
        return False


@dataclass
class UpsertResult:
    """Result of an upsert operation."""

    script: StrategyScript
    is_new: bool
    changed_fields: list[str] = field(default_factory=list)


@dataclass
class ArchivedScript:
    """Info about an archived script (for event emission)."""

    id: UUID
    workspace_id: UUID
    rel_path: str
    last_seen_at: Optional[datetime]


@dataclass
class ArchiveResult:
    """Result of an archive operation."""

    archived_count: int
    archived_scripts: list[ArchivedScript] = field(default_factory=list)


# Fields to compare for change detection
CHANGE_DETECT_FIELDS = [
    "sha256",
    "pine_version",
    "script_type",
    "title",
]


def _compute_changed_fields(
    existing: StrategyScript, incoming: StrategyScript
) -> list[str]:
    """Compute which fields changed between existing and incoming."""
    changes = []
    for field_name in CHANGE_DETECT_FIELDS:
        old_val = getattr(existing, field_name)
        new_val = getattr(incoming, field_name)
        if old_val != new_val:
            changes.append(field_name)
    return changes


class StrategyScriptRepository:
    """Repository for strategy_scripts table."""

    def __init__(self, pool):
        """Initialize with connection pool."""
        self._pool = pool

    async def get_by_path(
        self,
        workspace_id: UUID,
        source_type: str,
        rel_path: str,
    ) -> Optional[StrategyScript]:
        """Get script by unique path key."""
        query = """
            SELECT id, workspace_id, rel_path, source_type, sha256,
                   pine_version, script_type, title, status,
                   spec_json, spec_generated_at, lint_json,
                   strategy_id, published_at, last_seen_at,
                   created_at, updated_at
            FROM strategy_scripts
            WHERE workspace_id = $1
              AND source_type = $2
              AND rel_path = $3
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id, source_type, rel_path)
            if row:
                return self._row_to_model(row)
        return None

    async def get_by_id(self, script_id: UUID) -> Optional[StrategyScript]:
        """Get script by ID."""
        query = """
            SELECT id, workspace_id, rel_path, source_type, sha256,
                   pine_version, script_type, title, status,
                   spec_json, spec_generated_at, lint_json,
                   strategy_id, published_at, last_seen_at,
                   created_at, updated_at
            FROM strategy_scripts
            WHERE id = $1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, script_id)
            if row:
                return self._row_to_model(row)
        return None

    async def upsert(self, script: StrategyScript) -> UpsertResult:
        """
        Upsert a script using Pattern A (fetch-diff-upsert).

        1. Fetch existing by (workspace_id, source_type, rel_path)
        2. Compute changed_fields in Python
        3. If new: INSERT
        4. If changed: UPDATE only changed fields + updated_at
        5. If unchanged: UPDATE last_seen_at only (no updated_at churn)

        Returns (script, is_new, changed_fields)
        """
        existing = await self.get_by_path(
            script.workspace_id, script.source_type, script.rel_path
        )

        if existing is None:
            # New script - INSERT
            return await self._insert(script)
        else:
            # Existing script - compare and update
            changed_fields = _compute_changed_fields(existing, script)
            if changed_fields:
                # Fields changed - UPDATE all fields + updated_at
                return await self._update(existing.id, script, changed_fields)
            else:
                # No changes - just update last_seen_at
                await self._touch_last_seen(existing.id)
                existing.last_seen_at = datetime.now(timezone.utc)
                return UpsertResult(script=existing, is_new=False, changed_fields=[])

    async def _insert(self, script: StrategyScript) -> UpsertResult:
        """Insert a new script."""
        import json

        query = """
            INSERT INTO strategy_scripts (
                id, workspace_id, rel_path, source_type, sha256,
                pine_version, script_type, title, status,
                spec_json, spec_generated_at, lint_json,
                strategy_id, published_at, last_seen_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            )
            RETURNING id, created_at, updated_at, last_seen_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                script.id,
                script.workspace_id,
                script.rel_path,
                script.source_type,
                script.sha256,
                script.pine_version,
                script.script_type,
                script.title,
                script.status,
                json.dumps(script.spec_json) if script.spec_json else None,
                script.spec_generated_at,
                json.dumps(script.lint_json) if script.lint_json else None,
                script.strategy_id,
                script.published_at,
                datetime.now(timezone.utc),
            )
            script.id = row["id"]
            script.created_at = row["created_at"]
            script.updated_at = row["updated_at"]
            script.last_seen_at = row["last_seen_at"]

        return UpsertResult(script=script, is_new=True, changed_fields=[])

    async def _update(
        self, script_id: UUID, script: StrategyScript, changed_fields: list[str]
    ) -> UpsertResult:
        """Update an existing script with changed fields."""
        query = """
            UPDATE strategy_scripts SET
                sha256 = $1,
                pine_version = $2,
                script_type = $3,
                title = $4,
                last_seen_at = $5
            WHERE id = $6
            RETURNING id, created_at, updated_at, last_seen_at
        """
        now = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                script.sha256,
                script.pine_version,
                script.script_type,
                script.title,
                now,
                script_id,
            )
            script.id = row["id"]
            script.created_at = row["created_at"]
            script.updated_at = row["updated_at"]
            script.last_seen_at = row["last_seen_at"]

        return UpsertResult(script=script, is_new=False, changed_fields=changed_fields)

    async def _touch_last_seen(self, script_id: UUID) -> None:
        """Update only last_seen_at (no updated_at churn)."""
        # Use raw SQL to avoid trigger updating updated_at
        query = """
            UPDATE strategy_scripts
            SET last_seen_at = $1
            WHERE id = $2
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, datetime.now(timezone.utc), script_id)

    async def update_spec(
        self,
        script_id: UUID,
        spec_json: dict,
        lint_json: Optional[dict] = None,
    ) -> Optional[StrategyScript]:
        """
        Update spec after generation.

        Sets:
        - status = 'spec_generated'
        - spec_json = <spec>
        - spec_generated_at = NOW()
        - lint_json = <optional>
        """
        import json

        query = """
            UPDATE strategy_scripts SET
                status = 'spec_generated',
                spec_json = $1,
                spec_generated_at = $2,
                lint_json = $3
            WHERE id = $4
            RETURNING id, workspace_id, rel_path, source_type, sha256,
                      pine_version, script_type, title, status,
                      spec_json, spec_generated_at, lint_json,
                      strategy_id, published_at, last_seen_at,
                      created_at, updated_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                json.dumps(spec_json),
                datetime.now(timezone.utc),
                json.dumps(lint_json) if lint_json else None,
                script_id,
            )
            if row:
                return self._row_to_model(row)
        return None

    async def update_last_seen_batch(self, script_ids: list[UUID]) -> int:
        """Update last_seen_at for multiple scripts."""
        if not script_ids:
            return 0

        query = """
            UPDATE strategy_scripts
            SET last_seen_at = $1
            WHERE id = ANY($2)
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, datetime.now(timezone.utc), script_ids)
            # Parse "UPDATE N" to get count
            count = int(result.split()[-1]) if result else 0
            return count

    async def list_by_status(
        self,
        workspace_id: UUID,
        status: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[StrategyScript]:
        """List scripts by status."""
        query = """
            SELECT id, workspace_id, rel_path, source_type, sha256,
                   pine_version, script_type, title, status,
                   spec_json, spec_generated_at, lint_json,
                   strategy_id, published_at, last_seen_at,
                   created_at, updated_at
            FROM strategy_scripts
            WHERE workspace_id = $1
              AND status = $2
            ORDER BY last_seen_at DESC
            LIMIT $3 OFFSET $4
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, status, limit, offset)
            return [self._row_to_model(row) for row in rows]

    async def list_active(
        self,
        workspace_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[StrategyScript]:
        """List non-archived scripts."""
        query = """
            SELECT id, workspace_id, rel_path, source_type, sha256,
                   pine_version, script_type, title, status,
                   spec_json, spec_generated_at, lint_json,
                   strategy_id, published_at, last_seen_at,
                   created_at, updated_at
            FROM strategy_scripts
            WHERE workspace_id = $1
              AND status != 'archived'
            ORDER BY last_seen_at DESC
            LIMIT $2 OFFSET $3
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, limit, offset)
            return [self._row_to_model(row) for row in rows]

    async def count_by_status(self, workspace_id: UUID) -> dict[str, int]:
        """Get counts per status."""
        query = """
            SELECT status, COUNT(*) as count
            FROM strategy_scripts
            WHERE workspace_id = $1
            GROUP BY status
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id)
            return {row["status"]: row["count"] for row in rows}

    async def count_pending_ingest(self, workspace_id: UUID) -> int:
        """Count scripts that need (re-)ingest.

        Uses the same logic as needs_ingest():
        - ingest_status IS NULL (never ingested), OR
        - last_ingested_sha != sha256 (content changed)

        This matches the idx_strategy_scripts_ingest_pending index.
        """
        query = """
            SELECT COUNT(*) FROM strategy_scripts
            WHERE workspace_id = $1
              AND status != 'archived'
              AND (ingest_status IS NULL OR last_ingested_sha IS DISTINCT FROM sha256)
        """
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, workspace_id) or 0

    async def mark_archived(
        self,
        workspace_id: UUID,
        older_than_days: int = 7,
    ) -> ArchiveResult:
        """
        Mark stale scripts as archived.

        Archives scripts that:
        - Belong to the workspace
        - Have status != 'archived' (idempotent)
        - Have last_seen_at older than N days

        Args:
            workspace_id: Workspace to archive scripts in
            older_than_days: Scripts not seen in this many days will be archived

        Returns:
            ArchiveResult with count and list of archived scripts
        """
        # First, get the scripts that will be archived (for event emission)
        select_query = """
            SELECT id, workspace_id, rel_path, last_seen_at
            FROM strategy_scripts
            WHERE workspace_id = $1
              AND status != 'archived'
              AND last_seen_at < NOW() - ($2 || ' days')::INTERVAL
        """

        # Then update them
        update_query = """
            UPDATE strategy_scripts
            SET status = 'archived'
            WHERE workspace_id = $1
              AND status != 'archived'
              AND last_seen_at < NOW() - ($2 || ' days')::INTERVAL
        """

        async with self._pool.acquire() as conn:
            # Get scripts to archive
            rows = await conn.fetch(select_query, workspace_id, str(older_than_days))
            archived_scripts = [
                ArchivedScript(
                    id=row["id"],
                    workspace_id=row["workspace_id"],
                    rel_path=row["rel_path"],
                    last_seen_at=row["last_seen_at"],
                )
                for row in rows
            ]

            # Archive them
            result = await conn.execute(update_query, workspace_id, str(older_than_days))
            count = int(result.split()[-1]) if result else 0

            return ArchiveResult(
                archived_count=count,
                archived_scripts=archived_scripts,
            )

    async def list_scripts(
        self,
        workspace_id: UUID,
        status: Optional[str] = None,
        script_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[StrategyScript], int]:
        """
        List scripts with filtering and pagination.

        Args:
            workspace_id: Workspace to list scripts from
            status: Filter by status (None = all non-archived, 'all' = include archived)
            script_type: Filter by script type (strategy, indicator, library)
            limit: Max results to return
            offset: Pagination offset

        Returns:
            Tuple of (scripts, total_count)
        """
        # Build WHERE clauses
        conditions = ["workspace_id = $1"]
        params: list = [workspace_id]
        param_idx = 2

        if status == "all":
            pass  # No status filter
        elif status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1
        else:
            # Default: exclude archived
            conditions.append("status != 'archived'")

        if script_type:
            conditions.append(f"script_type = ${param_idx}")
            params.append(script_type)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"""
            SELECT COUNT(*) FROM strategy_scripts
            WHERE {where_clause}
        """

        # Data query
        data_query = f"""
            SELECT id, workspace_id, rel_path, source_type, sha256,
                   pine_version, script_type, title, status,
                   spec_json, spec_generated_at, lint_json,
                   strategy_id, published_at, last_seen_at,
                   created_at, updated_at
            FROM strategy_scripts
            WHERE {where_clause}
            ORDER BY last_seen_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self._pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(data_query, *params)
            scripts = [self._row_to_model(row) for row in rows]

            return scripts, total or 0

    async def update_ingest_status(
        self,
        script_id: UUID,
        doc_id: Optional[UUID],
        status: str,
        error: Optional[str] = None,
    ) -> Optional[StrategyScript]:
        """
        Update ingest tracking fields after ingest attempt.

        Args:
            script_id: Script ID to update
            doc_id: Document ID from ingest (None if failed)
            status: 'ok' or 'error'
            error: Error message if status='error' (truncated to 500 chars)

        Returns:
            Updated StrategyScript or None if not found
        """
        # Truncate error message
        if error and len(error) > 500:
            error = error[:497] + "..."

        query = """
            UPDATE strategy_scripts SET
                doc_id = $1,
                last_ingested_at = $2,
                last_ingested_sha = sha256,
                ingest_status = $3,
                ingest_error = $4
            WHERE id = $5
            RETURNING id, workspace_id, rel_path, source_type, sha256,
                      pine_version, script_type, title, status,
                      spec_json, spec_generated_at, lint_json,
                      strategy_id, published_at, last_seen_at,
                      created_at, updated_at,
                      doc_id, last_ingested_at, last_ingested_sha,
                      ingest_status, ingest_error
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                doc_id,
                datetime.now(timezone.utc),
                status,
                error,
                script_id,
            )
            if row:
                return self._row_to_model(row)
        return None

    def _row_to_model(self, row) -> StrategyScript:
        """Convert DB row to model."""
        import json

        spec_json = row["spec_json"]
        if isinstance(spec_json, str):
            spec_json = json.loads(spec_json)

        lint_json = row["lint_json"]
        if isinstance(lint_json, str):
            lint_json = json.loads(lint_json)

        # Handle optional ingest fields (may not exist in older schemas)
        doc_id = row.get("doc_id")
        last_ingested_at = row.get("last_ingested_at")
        last_ingested_sha = row.get("last_ingested_sha")
        ingest_status = row.get("ingest_status")
        ingest_error = row.get("ingest_error")

        return StrategyScript(
            id=row["id"],
            workspace_id=row["workspace_id"],
            rel_path=row["rel_path"],
            source_type=row["source_type"],
            sha256=row["sha256"],
            pine_version=row["pine_version"],
            script_type=row["script_type"],
            title=row["title"],
            status=row["status"],
            spec_json=spec_json,
            spec_generated_at=row["spec_generated_at"],
            lint_json=lint_json,
            strategy_id=row["strategy_id"],
            published_at=row["published_at"],
            last_seen_at=row["last_seen_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            doc_id=doc_id,
            last_ingested_at=last_ingested_at,
            last_ingested_sha=last_ingested_sha,
            ingest_status=ingest_status,
            ingest_error=ingest_error,
        )
