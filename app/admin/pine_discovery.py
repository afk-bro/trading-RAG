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
from app.routers.metrics import record_pine_discovery_run

router = APIRouter(prefix="/pine", tags=["admin-pine"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup via set_db_pool)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


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
    errors: list[str] = Field(
        default_factory=list, description="Non-fatal errors during discovery"
    )


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
    )

    start_time = time.monotonic()

    try:
        # Import discovery service
        from app.services.pine.discovery import PineDiscoveryService

        service = PineDiscoveryService(pool, settings)

        # Run discovery
        result = await service.discover(
            workspace_id=request.workspace_id,
            scan_paths=scan_paths,
            generate_specs=request.generate_specs,
            dry_run=request.dry_run,
            discovery_run_id=discovery_run_id,
        )

        # Calculate duration
        duration = time.monotonic() - start_time

        # Determine status
        if request.dry_run:
            op_status = "dry_run"
        elif result.errors:
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
                errors_count=len(result.errors),
            )

        log.info(
            "discovery_request_complete",
            status=op_status,
            scripts_scanned=result.scripts_scanned,
            scripts_new=result.scripts_new,
            scripts_updated=result.scripts_updated,
            specs_generated=result.specs_generated,
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
