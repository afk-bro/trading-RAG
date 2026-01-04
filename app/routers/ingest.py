"""Document ingestion endpoint."""

import hashlib
import time
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.schemas import IngestRequest, IngestResponse, SourceType
from app.services.chunker import Chunk, Chunker
from app.services.embedder import get_embedder
from app.services.extractor import get_extractor

router = APIRouter()
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


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def ingest_pipeline(
    workspace_id: UUID,
    content: str,
    source_type: SourceType,
    source_url: Optional[str],
    canonical_url: str,
    idempotency_key: Optional[str],
    content_hash: Optional[str],
    title: Optional[str] = None,
    author: Optional[str] = None,
    published_at=None,
    language: str = "en",
    duration_secs: Optional[int] = None,
    video_id: Optional[str] = None,
    playlist_id: Optional[str] = None,
    pre_chunks: Optional[list] = None,
    settings: Settings = None,
) -> IngestResponse:
    """
    Core ingestion pipeline.

    This function handles:
    1. Content deduplication
    2. Chunking (if not pre-chunked)
    3. Metadata extraction
    4. Embedding generation
    5. Database storage
    6. Vector storage
    """
    from app.repositories.chunks import ChunkRepository
    from app.repositories.documents import DocumentRepository
    from app.repositories.vectors import ChunkVectorRepository, VectorRepository

    settings = settings or get_settings()

    # Compute content hash if not provided
    if not content_hash:
        content_hash = compute_content_hash(content)

    # Initialize repositories
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    doc_repo = DocumentRepository(_db_pool)
    chunk_repo = ChunkRepository(_db_pool)
    chunk_vector_repo = ChunkVectorRepository(_db_pool)
    vector_repo = VectorRepository(client=_qdrant_client)

    # Check for existing document (deduplication)
    existing = await doc_repo.get_by_canonical_url(
        workspace_id=workspace_id,
        source_type=source_type,
        canonical_url=canonical_url,
    )

    if existing:
        logger.info(
            "Document already exists",
            doc_id=str(existing["id"]),
            canonical_url=canonical_url,
        )
        return IngestResponse(
            doc_id=existing["id"],
            chunks_created=0,
            vectors_created=0,
            status="exists",
        )

    # Also check by content hash
    existing_by_hash = await doc_repo.get_by_content_hash(
        workspace_id=workspace_id,
        content_hash=content_hash,
    )

    if existing_by_hash:
        logger.info(
            "Document with same content exists",
            doc_id=str(existing_by_hash["id"]),
            content_hash=content_hash[:16] + "...",
        )
        return IngestResponse(
            doc_id=existing_by_hash["id"],
            chunks_created=0,
            vectors_created=0,
            status="exists",
        )

    # Create document
    doc_id = await doc_repo.create(
        workspace_id=workspace_id,
        source_url=source_url,
        canonical_url=canonical_url,
        source_type=source_type,
        content_hash=content_hash,
        title=title,
        author=author,
        channel=author,  # Use author as channel for non-YouTube
        published_at=published_at,
        language=language,
        duration_secs=duration_secs,
        video_id=video_id,
        playlist_id=playlist_id,
    )

    logger.info(
        "Created document",
        doc_id=str(doc_id),
        source_type=source_type.value,
    )

    # Chunk content
    chunk_start = time.perf_counter()
    chunker = Chunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        encoding_name=settings.chunk_tokenizer_encoding,
    )
    extractor = get_extractor()

    if pre_chunks:
        # Use pre-chunked content
        chunks = [
            Chunk(
                content=pc.content,
                chunk_index=idx,
                token_count=chunker.count_tokens(pc.content),
                time_start_secs=pc.time_start_secs,
                time_end_secs=pc.time_end_secs,
                page_start=pc.page_start,
                page_end=pc.page_end,
                section=pc.section,
            )
            for idx, pc in enumerate(pre_chunks)
        ]
    else:
        # Chunk the content
        chunks = chunker.chunk_text(content)

    chunk_duration_ms = (time.perf_counter() - chunk_start) * 1000
    logger.info(
        "Chunking completed",
        doc_id=str(doc_id),
        chunk_count=len(chunks) if chunks else 0,
        chunking_duration_ms=round(chunk_duration_ms, 2),
    )

    if not chunks:
        logger.warning("No chunks created", doc_id=str(doc_id))
        return IngestResponse(
            doc_id=doc_id,
            chunks_created=0,
            vectors_created=0,
            status="created",
        )

    # Extract metadata and prepare chunk records
    chunk_records = []
    for chunk in chunks:
        metadata = extractor.extract(chunk.content)

        # Compute locator label if not set
        locator_label = chunk.locator_label
        if not locator_label:
            if chunk.time_start_secs is not None:
                locator_label = chunker._format_timestamp(chunk.time_start_secs)
            elif chunk.page_start is not None:
                locator_label = f"p. {chunk.page_start}"

        chunk_records.append({
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "content_hash": compute_content_hash(chunk.content),
            "time_start_secs": chunk.time_start_secs,
            "time_end_secs": chunk.time_end_secs,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section": chunk.section,
            "locator_label": locator_label,
            "symbols": metadata.symbols,
            "entities": metadata.entities,
            "topics": metadata.topics,
            "quality_score": metadata.quality_score,
            "speaker": metadata.speaker,
        })

    # Store chunks in database
    db_write_start = time.perf_counter()
    chunk_ids = await chunk_repo.create_batch(
        doc_id=doc_id,
        workspace_id=workspace_id,
        chunks=chunk_records,
    )
    db_write_duration_ms = (time.perf_counter() - db_write_start) * 1000

    logger.info(
        "Database write completed",
        doc_id=str(doc_id),
        chunk_count=len(chunk_ids),
        database_write_duration_ms=round(db_write_duration_ms, 2),
    )

    # Generate embeddings
    embed_start = time.perf_counter()
    embedder = get_embedder()

    try:
        texts = [c["content"] for c in chunk_records]
        embeddings = await embedder.embed_batch(texts)
        embed_duration_ms = (time.perf_counter() - embed_start) * 1000

        logger.info(
            "Embedding completed",
            doc_id=str(doc_id),
            count=len(embeddings),
            dimension=len(embeddings[0]) if embeddings else 0,
            embedding_duration_ms=round(embed_duration_ms, 2),
        )
    except Exception as e:
        logger.error(
            "Embedding failed",
            doc_id=str(doc_id),
            error=str(e),
        )
        # Continue without vectors - they can be added later
        return IngestResponse(
            doc_id=doc_id,
            chunks_created=len(chunk_ids),
            vectors_created=0,
            status="partial",
        )

    # Prepare Qdrant points
    qdrant_points = []
    for i, (chunk_id, chunk_record, embedding) in enumerate(zip(chunk_ids, chunk_records, embeddings)):
        qdrant_points.append({
            "id": chunk_id,
            "vector": embedding,
            "payload": {
                "workspace_id": str(workspace_id),
                "doc_id": str(doc_id),
                "source_type": source_type.value,
                "author": author,
                "channel": author,  # Same as author for consistency in queries
                "symbols": chunk_record["symbols"],
                "topics": chunk_record["topics"],
                "entities": chunk_record["entities"],
                "time_start_secs": chunk_record.get("time_start_secs"),
                "published_at": int(published_at.timestamp()) if published_at else None,
            },
        })

    # Store in Qdrant
    try:
        await vector_repo.upsert_batch(qdrant_points)

        logger.info(
            "Stored vectors in Qdrant",
            doc_id=str(doc_id),
            count=len(qdrant_points),
        )
    except Exception as e:
        logger.error(
            "Qdrant upsert failed",
            doc_id=str(doc_id),
            error=str(e),
        )
        return IngestResponse(
            doc_id=doc_id,
            chunks_created=len(chunk_ids),
            vectors_created=0,
            status="partial",
        )

    # Record vectors in chunk_vectors table
    vector_records = [
        {
            "chunk_id": chunk_id,
            "workspace_id": workspace_id,
            "embed_provider": "ollama",
            "embed_model": settings.embed_model,
            "collection": settings.qdrant_collection_active,
            "vector_dim": len(embeddings[0]) if embeddings else 768,
        }
        for chunk_id in chunk_ids
    ]

    try:
        await chunk_vector_repo.create_batch(vector_records)
    except Exception as e:
        logger.warning(
            "Failed to record chunk_vectors",
            doc_id=str(doc_id),
            error=str(e),
        )

    # Update document last_indexed_at
    await doc_repo.update_last_indexed(doc_id)

    return IngestResponse(
        doc_id=doc_id,
        chunks_created=len(chunk_ids),
        vectors_created=len(qdrant_points),
        status="indexed",
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Document already exists (idempotent)"},
        201: {"description": "Document created successfully"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
async def ingest_document(
    request: IngestRequest,
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """
    Ingest a document into the RAG pipeline.

    This endpoint handles:
    - Content deduplication via content_hash or idempotency_key
    - Automatic chunking (if chunks not provided)
    - Metadata extraction (symbols, entities, topics)
    - Embedding generation via Ollama
    - Vector storage in Qdrant
    - Document/chunk storage in Supabase

    The operation is idempotent when idempotency_key is provided.
    """
    logger.info(
        "Ingesting document",
        workspace_id=str(request.workspace_id),
        source_type=request.source.type.value,
        idempotency_key=request.idempotency_key,
        content_length=len(request.content),
        pre_chunked=request.chunks is not None,
    )

    # Determine canonical URL
    canonical_url = str(request.source.url) if request.source.url else request.idempotency_key
    if not canonical_url:
        canonical_url = compute_content_hash(request.content)[:32]

    try:
        response = await ingest_pipeline(
            workspace_id=request.workspace_id,
            content=request.content,
            source_type=request.source.type,
            source_url=str(request.source.url) if request.source.url else None,
            canonical_url=canonical_url,
            idempotency_key=request.idempotency_key,
            content_hash=request.content_hash,
            title=request.metadata.title if request.metadata else None,
            author=request.metadata.author if request.metadata else None,
            published_at=request.metadata.published_at if request.metadata else None,
            language=request.metadata.language if request.metadata else "en",
            duration_secs=request.metadata.duration_secs if request.metadata else None,
            pre_chunks=request.chunks,
            settings=settings,
        )

        logger.info(
            "Ingestion complete",
            doc_id=str(response.doc_id),
            status=response.status,
            chunks_created=response.chunks_created,
            vectors_created=response.vectors_created,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
