"""Re-embed endpoint for model migration."""

import asyncio
import uuid
from typing import Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.routers.jobs import complete_job, create_job, fail_job, update_job_progress
from app.schemas import JobStatus, ReembedRequest, ReembedResponse
from app.services.embedder import OllamaEmbedder

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


async def reembed_chunks_task(
    job_id: uuid.UUID,
    workspace_id: uuid.UUID,
    target_collection: str,
    embed_provider: str,
    embed_model: str,
    doc_ids: Optional[list[uuid.UUID]],
    settings: Settings,
):
    """
    Background task to re-embed chunks with a new model.

    This task:
    1. Creates the target collection in Qdrant
    2. Fetches all chunks for the workspace
    3. Embeds chunks in batches
    4. Upserts to Qdrant
    5. Updates chunk_vectors table
    """
    from app.repositories.chunks import ChunkRepository
    from app.repositories.vectors import ChunkVectorRepository, VectorRepository

    logger.info(
        "Starting reembed task",
        job_id=str(job_id),
        workspace_id=str(workspace_id),
        target_collection=target_collection,
    )

    try:
        # Initialize repositories
        chunk_repo = ChunkRepository(_db_pool)
        chunk_vector_repo = ChunkVectorRepository(_db_pool)
        vector_repo = VectorRepository(client=_qdrant_client, collection=target_collection)

        # Initialize embedder for new model
        embedder = OllamaEmbedder(model=embed_model)

        # Get embedding dimension
        dimension = await embedder.get_dimension()

        # Create target collection
        await vector_repo.ensure_collection(dimension=dimension)
        logger.info(
            "Created target collection",
            collection=target_collection,
            dimension=dimension,
        )

        # Fetch chunks
        chunks = await chunk_repo.get_by_workspace(
            workspace_id=workspace_id,
            doc_ids=doc_ids,
            limit=100000,  # Large limit for re-embedding
        )

        total_chunks = len(chunks)
        logger.info(
            "Fetched chunks for reembedding",
            count=total_chunks,
        )

        if total_chunks == 0:
            complete_job(job_id)
            return

        # Process in batches
        batch_size = settings.embed_batch_size
        processed = 0

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]

            # Extract texts
            texts = [c["content"] for c in batch]

            # Embed
            embeddings = await embedder.embed_batch(texts)

            # Prepare Qdrant points
            points = []
            for chunk, embedding in zip(batch, embeddings):
                # Get published_at as unix timestamp if available
                published_at = chunk.get("published_at")
                published_at_ts = int(published_at.timestamp()) if published_at else None

                # Get author/channel - prefer author, fallback to channel
                author_value = chunk.get("author") or chunk.get("channel")

                points.append({
                    "id": chunk["id"],
                    "vector": embedding,
                    "payload": {
                        "workspace_id": str(workspace_id),
                        "doc_id": str(chunk["doc_id"]),
                        "source_type": chunk.get("source_type", ""),
                        "author": author_value,
                        "channel": author_value,  # Same as author for consistency
                        "symbols": chunk.get("symbols", []),
                        "topics": chunk.get("topics", []),
                        "entities": chunk.get("entities", []),
                        "time_start_secs": chunk.get("time_start_secs"),
                        "published_at": published_at_ts,
                    },
                })

            # Upsert to Qdrant
            await vector_repo.upsert_batch(points)

            # Record in chunk_vectors table
            vector_records = [
                {
                    "chunk_id": chunk["id"],
                    "workspace_id": workspace_id,
                    "embed_provider": embed_provider,
                    "embed_model": embed_model,
                    "collection": target_collection,
                    "vector_dim": dimension,
                }
                for chunk in batch
            ]
            await chunk_vector_repo.create_batch(vector_records)

            # Update progress
            processed += len(batch)
            progress = (processed / total_chunks) * 100
            update_job_progress(job_id, progress)

            logger.debug(
                "Processed batch",
                batch_num=i // batch_size + 1,
                processed=processed,
                total=total_chunks,
                progress=f"{progress:.1f}%",
            )

        # Mark job as complete
        complete_job(job_id)
        logger.info(
            "Reembed job completed",
            job_id=str(job_id),
            total_processed=processed,
        )

    except Exception as e:
        logger.exception(
            "Reembed job failed",
            job_id=str(job_id),
            error=str(e),
        )
        fail_job(job_id, str(e))


@router.post(
    "/reembed",
    response_model=ReembedResponse,
    responses={
        200: {"description": "Re-embedding job started"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def reembed(
    request: ReembedRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> ReembedResponse:
    """
    Start a re-embedding job for model migration.

    This endpoint:
    - Creates a new Qdrant collection (target_collection)
    - Queues all chunks for re-embedding with new model
    - Runs as background job (check status via GET /jobs/{job_id})
    - Updates chunk_vectors table when complete

    Use cases:
    - Migrating to a new embedding model
    - Rebuilding vectors after model update
    - Creating parallel collections for A/B testing

    Collection naming convention: kb_{model}_{version}
    Example: kb_nomic_embed_text_v1 → kb_new_model_v2

    After job completes:
    - Update QDRANT_COLLECTION_ACTIVE env var
    - Restart service to switch to new collection
    """
    from app.repositories.chunks import ChunkRepository

    logger.info(
        "Starting re-embed job",
        workspace_id=str(request.workspace_id),
        target_collection=request.target_collection,
        embed_provider=request.embed_provider,
        embed_model=request.embed_model,
        doc_ids=len(request.doc_ids) if request.doc_ids else "all",
    )

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    # Count chunks to be processed
    chunk_repo = ChunkRepository(_db_pool)
    chunks = await chunk_repo.get_by_workspace(
        workspace_id=request.workspace_id,
        doc_ids=request.doc_ids,
        limit=100000,
    )
    chunks_queued = len(chunks)

    if chunks_queued == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No chunks found for workspace",
        )

    # Create job
    job_id = uuid.uuid4()
    create_job(job_id)

    # Start background task
    background_tasks.add_task(
        reembed_chunks_task,
        job_id=job_id,
        workspace_id=request.workspace_id,
        target_collection=request.target_collection,
        embed_provider=request.embed_provider,
        embed_model=request.embed_model,
        doc_ids=request.doc_ids,
        settings=settings,
    )

    logger.info(
        "Re-embed job created",
        job_id=str(job_id),
        chunks_queued=chunks_queued,
    )

    return ReembedResponse(
        job_id=job_id,
        chunks_queued=chunks_queued,
        status=JobStatus.STARTED,
    )
