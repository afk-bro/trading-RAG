"""Document ingestion endpoint."""

import structlog
from fastapi import APIRouter, HTTPException, status

from app.schemas import IngestRequest, IngestResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Document already exists (idempotent)"},
        201: {"description": "Document created successfully"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def ingest_document(request: IngestRequest) -> IngestResponse:
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

    # TODO: Implement ingestion pipeline
    # 1. Check for existing document (idempotency_key or content_hash)
    # 2. If exists, return existing doc_id with status='exists'
    # 3. Chunk content (if not pre-chunked)
    # 4. Extract metadata (symbols, entities, topics)
    # 5. Generate embeddings via Ollama
    # 6. Store in Supabase (documents, chunks, chunk_vectors)
    # 7. Upsert to Qdrant
    # 8. Return response

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Ingestion pipeline not yet implemented",
    )
