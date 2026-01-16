"""Pine Script registry ingestion endpoint."""

from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.deps.security import require_admin_token
from app.schemas import PineIngestRequest, PineIngestResponse, PineIngestStatus
from app.services.pine import PineIngestService, load_registry

router = APIRouter()
logger = structlog.get_logger(__name__)

# Maximum registry file size (20 MB)
MAX_REGISTRY_BYTES = 20 * 1024 * 1024


def validate_path(
    path: str,
    settings: Settings,
    *,
    must_be_file: bool = True,
    allowed_extensions: set[str] | None = None,
    max_size_bytes: int | None = None,
) -> Path:
    """
    Validate path is safe and within allowed directory.

    Args:
        path: User-provided path string
        settings: App settings with data_dir
        must_be_file: If True, path must be a file; if False, must be directory
        allowed_extensions: Set of allowed file extensions (e.g., {".json"})
        max_size_bytes: Maximum file size in bytes

    Returns:
        Resolved Path object

    Raises:
        HTTPException: 400 for invalid extension/size, 403 for path traversal, 404 for missing
    """
    resolved = Path(path).resolve()

    # Ensure DATA_DIR exists (strict=True follows symlinks safely)
    try:
        allowed_root = Path(settings.data_dir).resolve(strict=True)
    except FileNotFoundError:
        logger.error("data_dir does not exist", data_dir=settings.data_dir)
        raise HTTPException(500, "Server configuration error: data_dir not found")

    # Must be within DATA_DIR (compatible with Python 3.8+)
    try:
        resolved.relative_to(allowed_root)
    except ValueError:
        logger.warning(
            "path_traversal_attempt",
            path=path,
            resolved=str(resolved),
            allowed_root=str(allowed_root),
        )
        raise HTTPException(403, f"Path must be within {allowed_root}")

    # Extension check
    if allowed_extensions and resolved.suffix.lower() not in allowed_extensions:
        raise HTTPException(400, f"File must have extension: {allowed_extensions}")

    # Existence and type check
    if must_be_file:
        if not resolved.is_file():
            raise HTTPException(404, f"File not found: {path}")
    else:
        if not resolved.is_dir():
            raise HTTPException(404, f"Directory not found: {path}")

    # Size check (files only)
    if must_be_file and max_size_bytes:
        file_size = resolved.stat().st_size
        if file_size > max_size_bytes:
            raise HTTPException(
                400, f"File too large: {file_size} bytes (max {max_size_bytes})"
            )

    return resolved


def derive_lint_path(registry_path: Path) -> Path | None:
    """
    Derive lint report path from registry path.

    Tries:
    1. Replace 'pine_registry.json' with 'pine_lint_report.json'
    2. Look for 'pine_lint_report.json' in same directory

    Returns None if no lint report found.
    """
    # Try replacing filename if it contains "pine_registry"
    if "pine_registry" in registry_path.name:
        lint_path = registry_path.parent / registry_path.name.replace(
            "pine_registry", "pine_lint_report"
        )
        if lint_path.exists():
            return lint_path

    # Try standard name in same directory
    lint_path = registry_path.parent / "pine_lint_report.json"
    if lint_path.exists():
        return lint_path

    return None


@router.post(
    "/ingest",
    response_model=PineIngestResponse,
    responses={
        200: {"description": "Ingest completed (check status field)"},
        400: {"description": "Invalid path (not .json, too large)"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token or path outside allowlist"},
        404: {"description": "Registry file not found"},
    },
    summary="Ingest Pine Script registry (admin)",
    description="Ingest Pine Script files from a registry into the RAG system. "
    "Requires admin token. All paths must be within DATA_DIR.",
)
async def ingest_pine_registry(
    request: PineIngestRequest,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_token),
) -> PineIngestResponse:
    """
    Ingest Pine Script files from a registry file.

    The registry must be a JSON file within DATA_DIR. If include_source=True,
    source_root must also be within DATA_DIR and contain the .pine files
    referenced by the registry.
    """
    ingest_run_id = f"pine-ingest-{uuid4().hex[:8]}"
    log = logger.bind(
        ingest_run_id=ingest_run_id,
        workspace_id=str(request.workspace_id),
        dry_run=request.dry_run,
    )
    log.info("pine_ingest_started", registry_path=request.registry_path)

    # Validate registry path
    registry_path = validate_path(
        request.registry_path,
        settings,
        must_be_file=True,
        allowed_extensions={".json"},
        max_size_bytes=MAX_REGISTRY_BYTES,
    )

    # Validate lint path if provided
    lint_path: Path | None = None
    if request.lint_path:
        lint_path = validate_path(
            request.lint_path,
            settings,
            must_be_file=True,
            allowed_extensions={".json"},
            max_size_bytes=MAX_REGISTRY_BYTES,
        )
    else:
        # Auto-derive lint path
        lint_path = derive_lint_path(registry_path)
        if lint_path:
            log.info("lint_path_auto_derived", lint_path=str(lint_path))

    # Validate source_root if include_source is True
    source_root: Path | None = None
    if request.include_source:
        if request.source_root:
            source_root = validate_path(
                request.source_root,
                settings,
                must_be_file=False,
            )
        else:
            # Default to registry's parent directory
            source_root = registry_path.parent
            log.info("source_root_defaulted", source_root=str(source_root))

    # Load registry to get script count for dry_run
    try:
        registry = load_registry(registry_path)
    except Exception as e:
        log.error("registry_load_failed", error=str(e))
        raise HTTPException(400, f"Failed to load registry: {e}")

    scripts_total = len(registry.scripts)
    log.info("registry_loaded", scripts_total=scripts_total)

    # Dry run - return what would happen
    if request.dry_run:
        # Count scripts that would be skipped due to lint errors
        scripts_skipped = 0
        if request.skip_lint_errors:
            for entry in registry.scripts.values():
                if entry.lint and entry.lint.has_errors:
                    scripts_skipped += 1
                elif lint_path is None and request.skip_lint_errors:
                    # Would need inline lint - count as processed
                    pass

        return PineIngestResponse(
            status=PineIngestStatus.DRY_RUN,
            scripts_processed=scripts_total,
            scripts_indexed=0,
            scripts_already_indexed=0,
            scripts_skipped=scripts_skipped,
            scripts_failed=0,
            chunks_added=0,
            errors=[],
            ingest_run_id=ingest_run_id,
        )

    # Import here to avoid circular dependencies
    from app.routers.ingest import _db_pool, _qdrant_client

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")
    if _qdrant_client is None:
        raise HTTPException(503, "Qdrant connection not available")

    # Create service and run ingestion
    service = PineIngestService(_db_pool, _qdrant_client, settings)

    try:
        result = await service.ingest_from_registry(
            workspace_id=request.workspace_id,
            registry_path=registry_path,
            source_root=source_root,
            include_source=request.include_source,
            max_source_lines=request.max_source_lines,
            skip_lint_errors=request.skip_lint_errors,
            update_existing=request.update_existing,
        )
    except Exception as e:
        log.error("ingest_failed", error=str(e))
        raise HTTPException(500, f"Ingestion failed: {e}")

    # Determine status
    if (
        result.scripts_failed == result.scripts_processed
        and result.scripts_processed > 0
    ):
        status = PineIngestStatus.FAILED
    elif result.scripts_failed > 0:
        status = PineIngestStatus.PARTIAL
    else:
        status = PineIngestStatus.SUCCESS

    # Collect errors from failed scripts
    errors = [
        f"{r.rel_path}: {r.error}" for r in result.results if not r.success and r.error
    ]

    # Count already indexed (exists status)
    scripts_already_indexed = sum(1 for r in result.results if r.status == "exists")

    log.info(
        "pine_ingest_completed",
        status=status.value,
        scripts_processed=result.scripts_processed,
        scripts_indexed=result.scripts_indexed,
        scripts_already_indexed=scripts_already_indexed,
        scripts_skipped=result.scripts_skipped,
        scripts_failed=result.scripts_failed,
        chunks_added=result.total_chunks,
    )

    return PineIngestResponse(
        status=status,
        scripts_processed=result.scripts_processed,
        scripts_indexed=result.scripts_indexed,
        scripts_already_indexed=scripts_already_indexed,
        scripts_skipped=result.scripts_skipped,
        scripts_failed=result.scripts_failed,
        chunks_added=result.total_chunks,
        errors=errors,
        ingest_run_id=ingest_run_id,
    )
