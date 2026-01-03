"""Re-embed endpoint for model migration."""

import structlog
from fastapi import APIRouter, HTTPException, status

from app.schemas import ReembedRequest, ReembedResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/reembed",
    response_model=ReembedResponse,
    responses={
        200: {"description": "Re-embedding job started"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def reembed(request: ReembedRequest) -> ReembedResponse:
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
    logger.info(
        "Starting re-embed job",
        workspace_id=str(request.workspace_id),
        target_collection=request.target_collection,
        embed_provider=request.embed_provider,
        embed_model=request.embed_model,
        doc_ids=len(request.doc_ids) if request.doc_ids else "all",
    )

    # TODO: Implement re-embed job
    # 1. Create target collection in Qdrant (if not exists)
    # 2. Query chunks for workspace (optionally filtered by doc_ids)
    # 3. Create job record
    # 4. Start background task to:
    #    - Embed chunks in batches
    #    - Upsert to target collection
    #    - Update chunk_vectors table
    #    - Update job progress
    # 5. Return job_id

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Re-embed not yet implemented",
    )
