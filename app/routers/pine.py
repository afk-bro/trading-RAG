"""Pine Script registry and read endpoints."""

from pathlib import Path
from typing import Literal, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import Settings, get_settings
from app.deps.security import require_admin_token
from app.schemas import (
    PineBuildStats,
    PineChunkItem,
    PineImportSchema,
    PineIngestRequest,
    PineIngestResponse,
    PineIngestStatus,
    PineInputSchema,
    PineLintFinding,
    PineLintSummary,
    PineRebuildAndIngestRequest,
    PineRebuildAndIngestResponse,
    PineScriptDetailResponse,
    PineScriptListItem,
    PineScriptListResponse,
    PineScriptLookupResponse,
)
from app.services.pine import PineIngestService, build_and_write_registry, load_registry

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
            lint_report_path=lint_path,
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


@router.post(
    "/rebuild-and-ingest",
    response_model=PineRebuildAndIngestResponse,
    responses={
        200: {"description": "Rebuild and ingest completed (check status field)"},
        400: {"description": "Invalid path or build error"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token or path outside allowlist"},
        404: {"description": "Scripts directory not found"},
    },
    summary="Rebuild registry and ingest (admin)",
    description="Scan directory for Pine files, build registry, and ingest into RAG. "
    "Single admin action for updating Pine knowledge base.",
)
async def rebuild_and_ingest_pine(
    request: PineRebuildAndIngestRequest,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_token),
) -> PineRebuildAndIngestResponse:
    """
    Rebuild Pine registry from source files and ingest into RAG.

    This combines:
    1. Scanning scripts_root for .pine files
    2. Building registry and lint report
    3. Ingesting scripts into the workspace

    All paths must be within DATA_DIR for security.
    """
    import time

    ingest_run_id = f"pine-rebuild-{uuid4().hex[:8]}"
    log = logger.bind(
        ingest_run_id=ingest_run_id,
        workspace_id=str(request.workspace_id),
        dry_run=request.dry_run,
    )
    start_time = time.monotonic()

    # Validate scripts_root (must be directory within DATA_DIR)
    scripts_root = validate_path(
        request.scripts_root,
        settings,
        must_be_file=False,  # Directory
    )

    # Validate output_dir if provided
    if request.output_dir:
        output_dir = validate_path(
            request.output_dir,
            settings,
            must_be_file=False,
        )
    else:
        output_dir = scripts_root  # Default to same directory

    log.info(
        "pine_rebuild_started",
        scripts_root=str(scripts_root),
        output_dir=str(output_dir),
    )

    # Phase 1: Build registry
    errors: list[str] = []
    try:
        build_result = build_and_write_registry(
            root=scripts_root,
            output_dir=output_dir,
        )
    except Exception as e:
        log.error("registry_build_failed", error=str(e))
        return PineRebuildAndIngestResponse(
            status=PineIngestStatus.FAILED,
            build=PineBuildStats(parse_errors=1),
            errors=[f"Build failed: {e}"],
            ingest_run_id=ingest_run_id,
        )

    build_stats = PineBuildStats(
        files_scanned=build_result.files_scanned,
        files_parsed=build_result.files_parsed,
        parse_errors=build_result.parse_errors,
        lint_errors=build_result.total_errors,
        lint_warnings=build_result.total_warnings,
        registry_path=str(build_result.registry_path) if build_result.registry_path else None,
        lint_report_path=str(build_result.lint_report_path) if build_result.lint_report_path else None,
    )

    log.info(
        "pine_build_completed",
        files_scanned=build_stats.files_scanned,
        files_parsed=build_stats.files_parsed,
        parse_errors=build_stats.parse_errors,
        lint_errors=build_stats.lint_errors,
    )

    # Check if build produced any scripts
    if build_result.files_parsed == 0:
        return PineRebuildAndIngestResponse(
            status=PineIngestStatus.SUCCESS if build_result.files_scanned == 0 else PineIngestStatus.FAILED,
            build=build_stats,
            scripts_processed=0,
            errors=["No .pine files found"] if build_result.files_scanned == 0 else ["All files failed to parse"],
            ingest_run_id=ingest_run_id,
        )

    # Dry run - return build stats only
    if request.dry_run:
        return PineRebuildAndIngestResponse(
            status=PineIngestStatus.DRY_RUN,
            build=build_stats,
            scripts_processed=len(build_result.registry.scripts),
            ingest_run_id=ingest_run_id,
        )

    # Phase 2: Ingest from built registry
    from app.routers.ingest import _db_pool, _qdrant_client

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")
    if _qdrant_client is None:
        raise HTTPException(503, "Qdrant connection not available")

    service = PineIngestService(_db_pool, _qdrant_client, settings)

    try:
        ingest_result = await service.ingest_from_registry(
            workspace_id=request.workspace_id,
            registry_path=build_result.registry_path,
            source_root=scripts_root if request.include_source else None,
            lint_report_path=build_result.lint_report_path,
            include_source=request.include_source,
            max_source_lines=request.max_source_lines,
            skip_lint_errors=request.skip_lint_errors,
            update_existing=request.update_existing,
        )
    except Exception as e:
        log.error("ingest_failed", error=str(e))
        return PineRebuildAndIngestResponse(
            status=PineIngestStatus.FAILED,
            build=build_stats,
            scripts_processed=len(build_result.registry.scripts),
            errors=[f"Ingest failed: {e}"],
            ingest_run_id=ingest_run_id,
        )

    # Determine final status
    if (
        ingest_result.scripts_failed == ingest_result.scripts_processed
        and ingest_result.scripts_processed > 0
    ):
        status = PineIngestStatus.FAILED
    elif ingest_result.scripts_failed > 0:
        status = PineIngestStatus.PARTIAL
    else:
        status = PineIngestStatus.SUCCESS

    # Collect errors
    errors = [
        f"{r.rel_path}: {r.error}" for r in ingest_result.results if not r.success and r.error
    ]

    scripts_already_indexed = sum(1 for r in ingest_result.results if r.status == "exists")

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    log.info(
        "pine_rebuild_and_ingest_completed",
        status=status.value,
        scripts_processed=ingest_result.scripts_processed,
        scripts_indexed=ingest_result.scripts_indexed,
        scripts_already_indexed=scripts_already_indexed,
        elapsed_ms=elapsed_ms,
    )

    return PineRebuildAndIngestResponse(
        status=status,
        build=build_stats,
        scripts_processed=ingest_result.scripts_processed,
        scripts_indexed=ingest_result.scripts_indexed,
        scripts_already_indexed=scripts_already_indexed,
        scripts_skipped=ingest_result.scripts_skipped,
        scripts_failed=ingest_result.scripts_failed,
        chunks_added=ingest_result.total_chunks,
        errors=errors,
        ingest_run_id=ingest_run_id,
    )


# =============================================================================
# Read Endpoints
# =============================================================================


def _extract_rel_path(canonical_url: str) -> str:
    """Extract rel_path from canonical URL (pine://source/rel/path.pine)."""
    if canonical_url.startswith("pine://"):
        # Remove pine:// prefix and source_id
        parts = canonical_url[7:].split("/", 1)
        if len(parts) > 1:
            return parts[1]
    return canonical_url


def _build_lint_summary(metadata: Optional[dict]) -> PineLintSummary:
    """Build PineLintSummary from pine_metadata."""
    if not metadata or "lint_summary" not in metadata:
        return PineLintSummary()
    lint = metadata["lint_summary"]
    return PineLintSummary(
        errors=lint.get("errors", 0),
        warnings=lint.get("warnings", 0),
        info=lint.get("info", 0),
    )


def _build_list_item(
    doc: dict, symbols: list[str], chunk_count: int
) -> PineScriptListItem:
    """Build PineScriptListItem from document row."""
    import json

    raw_metadata = doc.get("pine_metadata")
    # Handle JSONB returned as string (asyncpg codec issue)
    if isinstance(raw_metadata, str):
        metadata = json.loads(raw_metadata)
    else:
        metadata = raw_metadata or {}

    # Extract rel_path from canonical_url or metadata
    rel_path = metadata.get("rel_path") or _extract_rel_path(doc["canonical_url"])

    # Title fallback to basename
    title = doc.get("title") or Path(rel_path).name

    return PineScriptListItem(
        id=doc["id"],
        canonical_url=doc["canonical_url"],
        rel_path=rel_path,
        title=title,
        script_type=metadata.get("script_type"),
        pine_version=metadata.get("pine_version"),
        symbols=symbols,
        lint_summary=_build_lint_summary(metadata),
        lint_available=metadata.get("lint_available", False),
        sha256=doc["content_hash"],
        chunk_count=chunk_count,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        status=doc["status"],
    )


@router.get(
    "/scripts",
    response_model=PineScriptListResponse,
    responses={
        200: {"description": "List of Pine scripts"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
    },
    summary="List indexed Pine scripts (admin)",
    description="List Pine scripts indexed in the RAG system with filtering and pagination.",
)
async def list_pine_scripts(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    symbol: Optional[str] = Query(None, description="Filter by ticker symbol"),
    status: Literal["active", "superseded", "deleted", "all"] = Query(
        "active", description="Filter by document status"
    ),
    q: Optional[str] = Query(None, description="Free-text search in title/path"),
    order_by: Literal["updated_at", "created_at", "title"] = Query(
        "updated_at", description="Sort field"
    ),
    order_dir: Literal["asc", "desc"] = Query("desc", description="Sort direction"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
) -> PineScriptListResponse:
    """List Pine scripts with filtering and pagination."""
    from app.routers.ingest import _db_pool

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")

    # Build query with filters
    params: list = [workspace_id]
    param_idx = 1

    # Base query - get documents with chunk counts and aggregated symbols
    query = """
        WITH script_data AS (
            SELECT
                d.id, d.canonical_url, d.title, d.content_hash,
                d.pine_metadata, d.status, d.created_at, d.updated_at,
                COUNT(DISTINCT c.id) as chunk_count,
                array_agg(DISTINCT s) FILTER (WHERE s IS NOT NULL) as symbols
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.id
            LEFT JOIN LATERAL unnest(c.symbols) AS s ON true
            WHERE d.workspace_id = $1
              AND d.source_type = 'pine_script'
    """

    # Status filter
    if status != "all":
        param_idx += 1
        query += f" AND d.status = ${param_idx}"
        params.append(status)

    # Symbol filter (check if symbol is in any chunk's symbols array)
    if symbol:
        param_idx += 1
        query += f"""
            AND EXISTS (
                SELECT 1 FROM chunks c2
                WHERE c2.doc_id = d.id
                AND ${param_idx} = ANY(c2.symbols)
            )
        """
        params.append(symbol.upper())

    # Free-text search
    if q:
        param_idx += 1
        query += f"""
            AND (
                d.title ILIKE '%' || ${param_idx} || '%'
                OR d.canonical_url ILIKE '%' || ${param_idx} || '%'
            )
        """
        params.append(q)

    query += """
            GROUP BY d.id
        )
        SELECT *, (SELECT COUNT(*) FROM script_data) as total_count
        FROM script_data
    """

    # Ordering
    order_col = {
        "updated_at": "updated_at",
        "created_at": "created_at",
        "title": "title",
    }[order_by]
    order_dir_sql = "DESC" if order_dir == "desc" else "ASC"
    query += f" ORDER BY {order_col} {order_dir_sql} NULLS LAST"

    # Pagination
    param_idx += 1
    query += f" LIMIT ${param_idx}"
    params.append(limit)

    param_idx += 1
    query += f" OFFSET ${param_idx}"
    params.append(offset)

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    # Build response
    items = []
    total = 0
    for row in rows:
        total = row.get("total_count", 0)
        symbols_list = row.get("symbols") or []
        chunk_count = row.get("chunk_count", 0)
        items.append(_build_list_item(dict(row), symbols_list, chunk_count))

    has_more = offset + len(items) < total
    next_offset = offset + limit if has_more else None

    return PineScriptListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=next_offset,
    )


@router.get(
    "/scripts/lookup",
    response_model=PineScriptLookupResponse,
    responses={
        200: {"description": "Script found"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
        404: {"description": "Script not found"},
    },
    summary="Lookup Pine script by rel_path (admin)",
    description="Find a Pine script by its relative file path. Useful for linking from filesystem.",
)
async def lookup_pine_script(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    rel_path: str = Query(..., description="Relative file path (e.g., 'macd_mean_reversion.pine')"),
    _: bool = Depends(require_admin_token),
) -> PineScriptLookupResponse:
    """Lookup a Pine script by its relative path."""
    import json

    from app.routers.ingest import _db_pool

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")

    async with _db_pool.acquire() as conn:
        # Try matching pine_metadata->>'rel_path' or canonical_url ending
        doc = await conn.fetchrow(
            """
            SELECT id, canonical_url, title, status, pine_metadata
            FROM documents
            WHERE workspace_id = $1
              AND source_type = 'pine_script'
              AND (
                  pine_metadata->>'rel_path' = $2
                  OR canonical_url LIKE '%/' || $2
                  OR canonical_url = 'pine://local/' || $2
              )
            LIMIT 1
            """,
            workspace_id,
            rel_path,
        )

    if not doc:
        raise HTTPException(404, f"Pine script not found: {rel_path}")

    doc_dict = dict(doc)
    raw_metadata = doc_dict.get("pine_metadata")
    if isinstance(raw_metadata, str):
        metadata = json.loads(raw_metadata)
    else:
        metadata = raw_metadata or {}

    actual_rel_path = metadata.get("rel_path") or _extract_rel_path(doc_dict["canonical_url"])
    title = doc_dict.get("title") or Path(actual_rel_path).name

    return PineScriptLookupResponse(
        id=doc_dict["id"],
        canonical_url=doc_dict["canonical_url"],
        rel_path=actual_rel_path,
        title=title,
        status=doc_dict["status"],
        script_type=metadata.get("script_type"),
        pine_version=metadata.get("pine_version"),
    )


@router.get(
    "/scripts/{doc_id}",
    response_model=PineScriptDetailResponse,
    responses={
        200: {"description": "Pine script details"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token or workspace mismatch"},
        404: {"description": "Document not found"},
    },
    summary="Get Pine script details (admin)",
    description="Get detailed information about a Pine script including metadata and chunks.",
)
async def get_pine_script(
    doc_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID (must match document)"),
    include_chunks: bool = Query(False, description="Include chunk content"),
    chunk_limit: int = Query(50, ge=1, le=200, description="Max chunks to return"),
    chunk_offset: int = Query(0, ge=0, description="Chunk pagination offset"),
    include_lint_findings: bool = Query(False, description="Include lint findings"),
    _: bool = Depends(require_admin_token),
) -> PineScriptDetailResponse:
    """Get Pine script details by document ID."""
    from app.routers.ingest import _db_pool

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")

    async with _db_pool.acquire() as conn:
        # Get document
        doc = await conn.fetchrow(
            """
            SELECT * FROM documents
            WHERE id = $1 AND source_type = 'pine_script'
            """,
            doc_id,
        )

        if not doc:
            raise HTTPException(404, f"Pine script not found: {doc_id}")

        # Check workspace matches
        if doc["workspace_id"] != workspace_id:
            raise HTTPException(403, "Workspace mismatch")

        # Get aggregated symbols
        symbols_row = await conn.fetchrow(
            """
            SELECT array_agg(DISTINCT s) FILTER (WHERE s IS NOT NULL) as symbols
            FROM chunks c
            CROSS JOIN LATERAL unnest(c.symbols) AS s
            WHERE c.doc_id = $1
            """,
            doc_id,
        )
        symbols = symbols_row["symbols"] or [] if symbols_row else []

        # Get chunk total
        chunk_total_row = await conn.fetchrow(
            "SELECT COUNT(*) as cnt FROM chunks WHERE doc_id = $1",
            doc_id,
        )
        chunk_total = chunk_total_row["cnt"] if chunk_total_row else 0

        # Get chunks if requested
        chunks = None
        chunk_has_more = None
        chunk_next_offset = None
        if include_chunks:
            chunk_rows = await conn.fetch(
                """
                SELECT id, chunk_index, content, token_count, symbols
                FROM chunks
                WHERE doc_id = $1
                ORDER BY chunk_index
                LIMIT $2 OFFSET $3
                """,
                doc_id,
                chunk_limit,
                chunk_offset,
            )
            chunks = [
                PineChunkItem(
                    id=row["id"],
                    index=row["chunk_index"],
                    content=row["content"],
                    token_count=row["token_count"],
                    symbols=row["symbols"] or [],
                )
                for row in chunk_rows
            ]
            chunk_has_more = chunk_offset + len(chunks) < chunk_total
            chunk_next_offset = chunk_offset + chunk_limit if chunk_has_more else None

    # Extract metadata
    import json

    doc_dict = dict(doc)
    raw_metadata = doc_dict.get("pine_metadata")
    # Handle JSONB returned as string (asyncpg codec issue)
    if isinstance(raw_metadata, str):
        metadata = json.loads(raw_metadata)
    else:
        metadata = raw_metadata or {}
    rel_path = metadata.get("rel_path") or _extract_rel_path(doc_dict["canonical_url"])
    title = doc_dict.get("title") or Path(rel_path).name

    # Build lint findings if requested
    lint_findings = None
    if include_lint_findings and metadata.get("lint_findings"):
        lint_findings = [
            PineLintFinding(
                code=f["code"],
                severity=f["severity"],
                message=f["message"],
                line=f.get("line"),
                column=f.get("column"),
            )
            for f in metadata["lint_findings"]
        ]

    # Build inputs/imports from metadata
    inputs = None
    if metadata.get("inputs"):
        inputs = [
            PineInputSchema(
                name=inp["name"],
                type=inp.get("type", "unknown"),
                default=str(inp["default"]) if inp.get("default") is not None else None,
                tooltip=inp.get("tooltip"),
            )
            for inp in metadata["inputs"]
        ]

    imports = None
    if metadata.get("imports"):
        imports = [
            PineImportSchema(path=imp["path"], alias=imp.get("alias"))
            for imp in metadata["imports"]
        ]

    return PineScriptDetailResponse(
        id=doc_dict["id"],
        canonical_url=doc_dict["canonical_url"],
        rel_path=rel_path,
        title=title,
        script_type=metadata.get("script_type"),
        pine_version=metadata.get("pine_version"),
        symbols=symbols,
        lint_summary=_build_lint_summary(metadata),
        lint_available=metadata.get("lint_available", False),
        lint_findings=lint_findings,
        sha256=doc_dict["content_hash"],
        created_at=doc_dict["created_at"],
        updated_at=doc_dict["updated_at"],
        status=doc_dict["status"],
        inputs=inputs,
        imports=imports,
        features=metadata.get("features"),
        chunk_total=chunk_total,
        chunks=chunks,
        chunk_has_more=chunk_has_more,
        chunk_next_offset=chunk_next_offset,
    )
