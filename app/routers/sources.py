"""Generic sources listing and detail endpoints."""

import json
from typing import Literal, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps.security import require_admin_token
from app.schemas import (
    SourceDetailResponse,
    SourceListItem,
    SourceListResponse,
    SourceType,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


def _build_source_list_item(doc: dict, chunk_count: int) -> SourceListItem:
    """Build SourceListItem from document row."""
    return SourceListItem(
        id=doc["id"],
        source_type=doc["source_type"],
        canonical_url=doc["canonical_url"],
        title=doc.get("title"),
        author=doc.get("author"),
        channel=doc.get("channel"),
        video_id=doc.get("video_id"),
        status=doc["status"],
        chunk_count=chunk_count,
        version=doc.get("version", 1),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        last_indexed_at=doc.get("last_indexed_at"),
    )


@router.get(
    "",
    response_model=SourceListResponse,
    responses={
        200: {"description": "List of sources"},
        400: {"description": "Invalid parameters"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
    },
    summary="List indexed sources (admin)",
    description="List all sources (documents) indexed in the knowledge base with filtering.",
)
async def list_sources(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    source_type: Optional[SourceType] = Query(
        None, description="Filter by source type (youtube, pdf, pine_script, etc.)"
    ),
    status: Literal["active", "superseded", "deleted", "all"] = Query(
        "active", description="Filter by document status"
    ),
    video_id: Optional[str] = Query(None, description="Filter by YouTube video ID"),
    q: Optional[str] = Query(None, description="Free-text search in title"),
    order_by: Literal["updated_at", "created_at", "title", "published_at"] = Query(
        "updated_at", description="Sort field"
    ),
    order_dir: Literal["asc", "desc"] = Query("desc", description="Sort direction"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    _: bool = Depends(require_admin_token),
) -> SourceListResponse:
    """List sources with filtering and pagination."""
    from app.routers.ingest import _db_pool

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")

    # Build query with filters
    params: list = [workspace_id]
    param_idx = 1

    # Base query with chunk counts
    query = """
        WITH source_data AS (
            SELECT
                d.id, d.source_type, d.canonical_url, d.title,
                d.author, d.channel, d.video_id, d.status,
                d.version, d.created_at, d.updated_at, d.last_indexed_at,
                d.pine_metadata, d.published_at,
                COUNT(DISTINCT c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.id
            WHERE d.workspace_id = $1
    """

    # Status filter
    if status != "all":
        param_idx += 1
        query += f" AND d.status = ${param_idx}"
        params.append(status)

    # Source type filter
    if source_type:
        param_idx += 1
        query += f" AND d.source_type = ${param_idx}"
        params.append(source_type.value)

    # Video ID filter
    if video_id:
        param_idx += 1
        query += f" AND d.video_id = ${param_idx}"
        params.append(video_id)

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
        SELECT *, (SELECT COUNT(*) FROM source_data) as total_count
        FROM source_data
    """

    # Ordering
    order_col = {
        "updated_at": "updated_at",
        "created_at": "created_at",
        "title": "title",
        "published_at": "published_at",
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
        chunk_count = row.get("chunk_count", 0)
        items.append(_build_source_list_item(dict(row), chunk_count))

    has_more = offset + len(items) < total
    next_offset = offset + limit if has_more else None

    return SourceListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
        next_offset=next_offset,
    )


@router.get(
    "/{source_id}",
    response_model=SourceDetailResponse,
    responses={
        200: {"description": "Source details"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
        404: {"description": "Source not found"},
    },
    summary="Get source details (admin)",
    description="Get detailed information about a single source (document).",
)
async def get_source(
    source_id: UUID,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    include_chunks: bool = Query(False, description="Include chunk content"),
    include_health: bool = Query(False, description="Include health status"),
    chunk_limit: int = Query(50, ge=1, le=200, description="Chunks per page"),
    chunk_offset: int = Query(0, ge=0, description="Chunk pagination offset"),
    _: bool = Depends(require_admin_token),
) -> SourceDetailResponse:
    """Get source details with optional chunks."""
    from app.routers.ingest import _db_pool

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")

    async with _db_pool.acquire() as conn:
        # Fetch document
        doc = await conn.fetchrow(
            """
            SELECT *
            FROM documents
            WHERE id = $1 AND workspace_id = $2
            """,
            source_id,
            workspace_id,
        )

        if not doc:
            raise HTTPException(404, "Source not found")

        # Get chunk count
        chunk_count_row = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM chunks WHERE doc_id = $1",
            source_id,
        )
        chunk_count = chunk_count_row["count"] if chunk_count_row else 0

        # Optionally get chunks
        chunks = None
        chunks_total = chunk_count
        if include_chunks:
            chunk_rows = await conn.fetch(
                """
                SELECT id, content, chunk_index, token_count,
                       time_start_secs, time_end_secs, page_start, page_end,
                       symbols, entities, topics
                FROM chunks
                WHERE doc_id = $1
                ORDER BY chunk_index
                LIMIT $2 OFFSET $3
                """,
                source_id,
                chunk_limit,
                chunk_offset,
            )
            chunks = [
                {
                    "id": str(r["id"]),
                    "content": r["content"],
                    "chunk_index": r["chunk_index"],
                    "token_count": r["token_count"],
                    "time_start_secs": r["time_start_secs"],
                    "time_end_secs": r["time_end_secs"],
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "symbols": r["symbols"] or [],
                    "entities": r["entities"] or [],
                    "topics": r["topics"] or [],
                }
                for r in chunk_rows
            ]

    # Handle JSONB fields
    raw_metadata = doc.get("pine_metadata")
    if isinstance(raw_metadata, str):
        pine_metadata = json.loads(raw_metadata)
    else:
        pine_metadata = raw_metadata

    # Get health status if requested
    health = None
    if include_health:
        from app.schemas import SourceHealth, SourceHealthCheck
        from app.services.ingest import get_source_health_summary

        health_data = await get_source_health_summary(_db_pool, workspace_id, source_id)
        health = SourceHealth(
            status=health_data["status"],
            chunk_count_ok=health_data["chunk_count_ok"],
            embeddings_ok=health_data["embeddings_ok"],
            checks=[
                SourceHealthCheck(
                    name=c["name"],
                    passed=c["passed"],
                    message=c["message"],
                )
                for c in health_data["checks"]
            ],
        )

    return SourceDetailResponse(
        id=doc["id"],
        source_type=doc["source_type"],
        canonical_url=doc["canonical_url"],
        source_url=doc.get("source_url"),
        title=doc.get("title"),
        author=doc.get("author"),
        channel=doc.get("channel"),
        video_id=doc.get("video_id"),
        playlist_id=doc.get("playlist_id"),
        published_at=doc.get("published_at"),
        language=doc.get("language"),
        duration_secs=doc.get("duration_secs"),
        content_hash=doc["content_hash"],
        status=doc["status"],
        version=doc.get("version", 1),
        chunk_count=chunk_count,
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        last_indexed_at=doc.get("last_indexed_at"),
        health=health,
        pine_metadata=pine_metadata,
        chunks=chunks,
        chunks_total=chunks_total,
        chunks_has_more=(
            (chunk_offset + len(chunks) < chunks_total) if chunks else False
        ),
    )


@router.get(
    "/by-video/{video_id}",
    response_model=SourceDetailResponse,
    responses={
        200: {"description": "Source details"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
        404: {"description": "Source not found"},
    },
    summary="Get YouTube source by video ID (admin)",
    description="Look up a YouTube source by its video ID.",
)
async def get_source_by_video_id(
    video_id: str,
    workspace_id: UUID = Query(..., description="Workspace ID"),
    include_chunks: bool = Query(False, description="Include chunk content"),
    _: bool = Depends(require_admin_token),
) -> SourceDetailResponse:
    """Get YouTube source by video ID."""
    from app.routers.ingest import _db_pool

    if _db_pool is None:
        raise HTTPException(503, "Database connection not available")

    async with _db_pool.acquire() as conn:
        # Fetch document by video_id
        doc = await conn.fetchrow(
            """
            SELECT id
            FROM documents
            WHERE workspace_id = $1
              AND source_type = 'youtube'
              AND video_id = $2
              AND status = 'active'
            """,
            workspace_id,
            video_id,
        )

        if not doc:
            raise HTTPException(
                404, f"No YouTube source found for video_id: {video_id}"
            )

    # Delegate to main get_source endpoint
    return await get_source(
        source_id=doc["id"],
        workspace_id=workspace_id,
        include_chunks=include_chunks,
    )
