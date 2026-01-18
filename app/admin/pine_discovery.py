"""Admin endpoints for Pine script discovery.

Provides API access to the Pine script auto-discovery service for
scanning filesystem paths, parsing scripts, and generating specs.
"""

import time
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.config import Settings, get_settings
from app.deps.security import require_admin_token
from app.routers.metrics import (
    record_pine_discovery_run,
    record_pine_discovery_timestamp,
    set_pine_pending_ingest,
)

router = APIRouter(prefix="/pine", tags=["admin-pine"])
logger = structlog.get_logger(__name__)

# Global connection pool and clients (set during app startup)
_db_pool = None
_qdrant_client = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def set_qdrant_client(client):
    """Set the Qdrant client for this router."""
    global _qdrant_client
    _qdrant_client = client


def _get_pool():
    """Get the database pool."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return _db_pool


# ===========================================
# Request/Response Models
# ===========================================


class DiscoverRequest(BaseModel):
    """Request payload for Pine script discovery."""

    workspace_id: UUID = Field(..., description="Workspace to associate scripts with")
    scan_paths: Optional[list[str]] = Field(
        default=None,
        description="Paths to scan. Defaults to [DATA_DIR/pine] if not specified.",
        max_length=10,
    )
    generate_specs: bool = Field(
        default=True,
        description="Whether to generate StrategySpec for strategy scripts",
    )
    auto_ingest: bool = Field(
        default=True,
        description="Whether to auto-ingest new/changed scripts to KB",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, preview changes without persisting to DB or emitting events",
    )

    @field_validator("scan_paths")
    @classmethod
    def validate_scan_paths_length(cls, v):
        """Validate max 10 scan paths."""
        if v is not None and len(v) > 10:
            raise ValueError("Maximum 10 scan paths allowed")
        return v


class DiscoverResponse(BaseModel):
    """Response from Pine script discovery."""

    status: Literal["success", "partial", "failed", "dry_run"] = Field(
        ..., description="Overall operation status"
    )
    discovery_run_id: str = Field(
        ..., description="Unique ID for log correlation and debugging"
    )
    scripts_scanned: int = Field(..., description="Total Pine scripts found and parsed")
    scripts_new: int = Field(..., description="New scripts discovered")
    scripts_updated: int = Field(
        ..., description="Existing scripts with content changes"
    )
    scripts_unchanged: int = Field(..., description="Scripts with no content changes")
    specs_generated: int = Field(
        ..., description="Strategy specs generated (strategies only)"
    )
    scripts_ingested: int = Field(default=0, description="Scripts ingested to KB")
    scripts_ingest_failed: int = Field(
        default=0, description="Scripts that failed ingest"
    )
    chunks_created: int = Field(
        default=0, description="Total chunks created during ingest"
    )
    errors: list[str] = Field(
        default_factory=list, description="Non-fatal errors during discovery"
    )


class ArchiveRequest(BaseModel):
    """Request payload for archiving stale scripts."""

    workspace_id: UUID = Field(..., description="Workspace to archive scripts in")
    older_than_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Archive scripts not seen in this many days",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, preview which scripts would be archived",
    )


class ArchiveResponse(BaseModel):
    """Response from archiving stale scripts."""

    status: Literal["success", "dry_run"] = Field(..., description="Operation status")
    archived_count: int = Field(..., description="Number of scripts archived")
    archived_scripts: list[dict] = Field(
        default_factory=list,
        description="Details of archived scripts (id, rel_path, last_seen_at)",
    )


class ScriptListItem(BaseModel):
    """Summary of a discovered script for list view."""

    id: UUID
    rel_path: str
    source_type: str
    title: Optional[str]
    script_type: Optional[str]
    pine_version: Optional[str]
    status: str
    sha256: str
    has_spec: bool
    last_seen_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


class ScriptListResponse(BaseModel):
    """Paginated list of scripts."""

    items: list[ScriptListItem]
    total: int
    limit: int
    offset: int
    has_more: bool


# ===========================================
# Path Validation
# ===========================================


def _validate_path_within_data_dir(path: str, data_dir: str) -> Path:
    """
    Validate that a path is within DATA_DIR.

    Prevents path traversal attacks and restricts file access.

    Args:
        path: Path to validate (may be relative or absolute)
        data_dir: Allowed root directory

    Returns:
        Resolved Path object

    Raises:
        HTTPException 422: If path escapes DATA_DIR
    """
    data_dir_resolved = Path(data_dir).resolve()
    path_resolved = (data_dir_resolved / path).resolve()

    # Check path is within DATA_DIR
    try:
        path_resolved.relative_to(data_dir_resolved)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Path must be within DATA_DIR: {path}",
        )

    return path_resolved


# ===========================================
# Endpoints
# ===========================================


@router.post(
    "/discover",
    response_model=DiscoverResponse,
    dependencies=[Depends(require_admin_token)],
    summary="Discover Pine scripts in specified paths",
    description="""
    Scan filesystem paths for .pine files, parse them, and track discovery state.

    **Workflow:**
    1. Scan paths for .pine files
    2. Parse each file (version, type, inputs, imports)
    3. Compare SHA256 against stored fingerprints
    4. Upsert new/changed scripts to DB
    5. Generate StrategySpec for strategy-type scripts
    6. Emit SSE events for discovered/updated scripts

    **Guardrails:**
    - Maximum 10 scan paths per request
    - All paths must be within DATA_DIR
    - Defaults to [DATA_DIR/pine] if no paths specified

    **Dry Run:**
    Use dry_run=true to preview discovery without DB writes or events.
    """,
)
async def discover_pine_scripts(
    request: DiscoverRequest,
    settings: Settings = Depends(get_settings),
):
    """
    Discover and catalog Pine scripts from filesystem.

    Scans the specified paths (or default DATA_DIR/pine), parses each .pine file,
    computes fingerprints, and tracks discovery state in strategy_scripts table.
    """
    pool = _get_pool()
    discovery_run_id = f"disc-{uuid4().hex[:8]}"

    log = logger.bind(
        discovery_run_id=discovery_run_id,
        workspace_id=str(request.workspace_id),
        dry_run=request.dry_run,
    )

    # Validate and resolve scan paths
    scan_paths: list[str] = []
    if request.scan_paths:
        for path in request.scan_paths:
            validated = _validate_path_within_data_dir(path, settings.data_dir)
            scan_paths.append(str(validated))
    else:
        # Default to DATA_DIR/pine
        default_path = Path(settings.data_dir) / "pine"
        scan_paths = [str(default_path)]

    log.info(
        "discovery_request_received",
        scan_paths=scan_paths,
        generate_specs=request.generate_specs,
        auto_ingest=request.auto_ingest,
    )

    start_time = time.monotonic()

    try:
        # Import discovery service
        from app.services.pine.discovery import PineDiscoveryService

        service = PineDiscoveryService(pool, settings, qdrant_client=_qdrant_client)

        # Run discovery
        result = await service.discover(
            workspace_id=request.workspace_id,
            scan_paths=scan_paths,
            generate_specs=request.generate_specs,
            auto_ingest=request.auto_ingest,
            dry_run=request.dry_run,
            discovery_run_id=discovery_run_id,
        )

        # Calculate duration
        duration = time.monotonic() - start_time

        # Determine status
        if request.dry_run:
            op_status = "dry_run"
        elif result.errors or result.scripts_ingest_failed > 0:
            op_status = "partial"
        else:
            op_status = "success"

        # Record Prometheus metrics (skip dry_run - no side effects)
        if not request.dry_run:
            record_pine_discovery_run(
                status=op_status,
                duration=duration,
                scripts_scanned=result.scripts_scanned,
                scripts_new=result.scripts_new,
                specs_generated=result.specs_generated,
                scripts_ingested=result.scripts_ingested,
                scripts_ingest_failed=result.scripts_ingest_failed,
                chunks_created=result.chunks_created,
                errors_count=len(result.errors),
            )

            # Record timestamp gauges
            record_pine_discovery_timestamp(success=op_status == "success")

            # Update pending ingest gauge
            try:
                pending_count = await service._repo.count_pending_ingest(
                    request.workspace_id
                )
                set_pine_pending_ingest(pending_count)
            except Exception:
                log.warning("failed_to_update_pending_ingest_gauge")

        log.info(
            "discovery_request_complete",
            status=op_status,
            scripts_scanned=result.scripts_scanned,
            scripts_new=result.scripts_new,
            scripts_updated=result.scripts_updated,
            specs_generated=result.specs_generated,
            scripts_ingested=result.scripts_ingested,
            scripts_ingest_failed=result.scripts_ingest_failed,
            chunks_created=result.chunks_created,
            errors_count=len(result.errors),
            duration_seconds=round(duration, 3),
        )

        return DiscoverResponse(
            status=op_status,
            discovery_run_id=discovery_run_id,
            scripts_scanned=result.scripts_scanned,
            scripts_new=result.scripts_new,
            scripts_updated=result.scripts_updated,
            scripts_unchanged=result.scripts_unchanged,
            specs_generated=result.specs_generated,
            scripts_ingested=result.scripts_ingested,
            scripts_ingest_failed=result.scripts_ingest_failed,
            chunks_created=result.chunks_created,
            errors=result.errors,
        )

    except Exception as e:
        # Record failed run metrics
        duration = time.monotonic() - start_time
        record_pine_discovery_run(
            status="failed",
            duration=duration,
            errors_count=1,
        )
        record_pine_discovery_timestamp(success=False)

        log.exception("discovery_request_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Discovery failed: {e}",
        )


@router.get(
    "/scripts/stats",
    dependencies=[Depends(require_admin_token)],
    summary="Get discovery stats by status",
)
async def get_discovery_stats(workspace_id: UUID):
    """
    Get counts of discovered scripts by status.

    Returns a breakdown of scripts in each discovery status:
    - discovered: Found and cataloged
    - spec_generated: StrategySpec created
    - published: Linked to strategies table
    - archived: No longer tracked
    """
    pool = _get_pool()

    from app.services.pine.discovery_repository import StrategyScriptRepository

    repo = StrategyScriptRepository(pool)
    counts = await repo.count_by_status(workspace_id)

    return {
        "workspace_id": str(workspace_id),
        "counts_by_status": counts,
        "total": sum(counts.values()),
    }


@router.post(
    "/archive",
    response_model=ArchiveResponse,
    dependencies=[Depends(require_admin_token)],
    summary="Archive stale scripts",
    description="""
    Archive scripts not seen in the specified number of days.

    **Behavior:**
    - Sets status = 'archived' for scripts where last_seen_at < NOW() - N days
    - Idempotent: already-archived scripts are not affected
    - Emits pine.script.archived SSE events for each archived script

    **Use Cases:**
    - Manual cleanup of scripts from deleted files
    - Scheduled maintenance (via cron or pg_cron)
    """,
)
async def archive_stale_scripts(request: ArchiveRequest):
    """Archive scripts not seen in the specified number of days."""
    pool = _get_pool()

    from app.routers.metrics import record_pine_archive_run
    from app.services.pine.discovery_repository import StrategyScriptRepository

    log = logger.bind(
        workspace_id=str(request.workspace_id),
        older_than_days=request.older_than_days,
        dry_run=request.dry_run,
    )

    repo = StrategyScriptRepository(pool)

    start_time = time.monotonic()

    try:
        if request.dry_run:
            # Dry run: just count without archiving
            # We need to query the same way mark_archived does
            async with pool.acquire() as conn:
                query = """
                    SELECT id, workspace_id, rel_path, last_seen_at
                    FROM strategy_scripts
                    WHERE workspace_id = $1
                      AND status != 'archived'
                      AND last_seen_at < NOW() - ($2 || ' days')::INTERVAL
                """
                rows = await conn.fetch(
                    query, request.workspace_id, str(request.older_than_days)
                )

            archived_scripts = [
                {
                    "id": str(row["id"]),
                    "rel_path": row["rel_path"],
                    "last_seen_at": (
                        row["last_seen_at"].isoformat() if row["last_seen_at"] else None
                    ),
                }
                for row in rows
            ]

            log.info(
                "archive_dry_run_complete",
                would_archive=len(archived_scripts),
            )

            return ArchiveResponse(
                status="dry_run",
                archived_count=len(archived_scripts),
                archived_scripts=archived_scripts,
            )

        # Actual archive operation
        result = await repo.mark_archived(request.workspace_id, request.older_than_days)

        # Emit events for archived scripts
        if result.archived_count > 0:
            from app.services.events import get_event_bus
            from app.services.events.schemas import pine_script_archived

            bus = get_event_bus()
            for script in result.archived_scripts:
                try:
                    last_seen_str = None
                    if script.last_seen_at:
                        last_seen_str = script.last_seen_at.isoformat()

                    event = pine_script_archived(
                        event_id="",
                        workspace_id=request.workspace_id,
                        script_id=script.id,
                        rel_path=script.rel_path,
                        last_seen_at=last_seen_str,
                    )
                    await bus.publish(event)
                except Exception as e:
                    log.warning(
                        "archived_event_emit_error",
                        script_id=str(script.id),
                        error=str(e),
                    )

        duration = time.monotonic() - start_time

        # Record metrics
        record_pine_archive_run(
            status="success",
            duration=duration,
            archived_count=result.archived_count,
        )

        archived_scripts = [
            {
                "id": str(s.id),
                "rel_path": s.rel_path,
                "last_seen_at": s.last_seen_at.isoformat() if s.last_seen_at else None,
            }
            for s in result.archived_scripts
        ]

        log.info(
            "archive_complete",
            archived_count=result.archived_count,
            duration_seconds=round(duration, 3),
        )

        return ArchiveResponse(
            status="success",
            archived_count=result.archived_count,
            archived_scripts=archived_scripts,
        )

    except Exception as e:
        duration = time.monotonic() - start_time
        record_pine_archive_run(
            status="failed",
            duration=duration,
            archived_count=0,
        )

        log.exception("archive_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Archive operation failed: {e}",
        )


@router.get(
    "/scripts",
    response_model=ScriptListResponse,
    dependencies=[Depends(require_admin_token)],
    summary="List discovered scripts",
    description="""
    List scripts with optional filtering and pagination.

    **Filters:**
    - status: Filter by discovery status (discovered, spec_generated, published, archived, all)
    - script_type: Filter by Pine script type (strategy, indicator, library)

    **Pagination:**
    - limit: Max results per page (default 50, max 100)
    - offset: Number of results to skip

    **Sorting:**
    - Results ordered by last_seen_at descending (most recent first)
    """,
)
async def list_scripts(
    workspace_id: UUID,
    status: Optional[str] = None,
    script_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List discovered scripts with filtering and pagination."""
    pool = _get_pool()

    from app.services.pine.discovery_repository import StrategyScriptRepository

    # Validate limit
    if limit < 1:
        limit = 1
    elif limit > 100:
        limit = 100

    if offset < 0:
        offset = 0

    repo = StrategyScriptRepository(pool)
    scripts, total = await repo.list_scripts(
        workspace_id=workspace_id,
        status=status,
        script_type=script_type,
        limit=limit,
        offset=offset,
    )

    items = [
        ScriptListItem(
            id=s.id,
            rel_path=s.rel_path,
            source_type=s.source_type,
            title=s.title,
            script_type=s.script_type,
            pine_version=s.pine_version,
            status=s.status,
            sha256=s.sha256,
            has_spec=s.spec_json is not None,
            last_seen_at=s.last_seen_at.isoformat() if s.last_seen_at else None,
            created_at=s.created_at.isoformat() if s.created_at else None,
            updated_at=s.updated_at.isoformat() if s.updated_at else None,
        )
        for s in scripts
    ]

    return ScriptListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(items)) < total,
    )
