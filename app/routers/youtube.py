"""YouTube ingestion endpoint."""

import structlog
from fastapi import APIRouter, HTTPException, status

from app.schemas import YouTubeIngestRequest, YouTubeIngestResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post(
    "/ingest",
    response_model=YouTubeIngestResponse,
    responses={
        200: {"description": "Video ingested or playlist expanded"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def ingest_youtube(request: YouTubeIngestRequest) -> YouTubeIngestResponse:
    """
    Ingest a YouTube video or playlist.

    For single videos:
    - Fetches transcript with retry + exponential backoff
    - Extracts metadata (title, channel, published_at, duration)
    - Normalizes transcript (removes [Music], repeated phrases)
    - Chunks with timestamp preservation
    - Runs through standard ingestion pipeline

    For playlists:
    - Returns is_playlist=True with video_urls array
    - Caller (n8n) should fan out to individual videos

    Error handling:
    - Missing transcript: terminal error (retryable=false)
    - Rate limiting: retryable error (retryable=true)
    - Network issues: retryable error (retryable=true)
    """
    logger.info(
        "Ingesting YouTube content",
        workspace_id=str(request.workspace_id),
        url=request.url,
        idempotency_key=request.idempotency_key,
    )

    # TODO: Implement YouTube ingestion
    # 1. Parse URL to detect video vs playlist
    # 2. If playlist:
    #    - Fetch video list
    #    - Return is_playlist=True with video_urls
    # 3. If video:
    #    - Fetch metadata
    #    - Fetch transcript with retry
    #    - Normalize transcript
    #    - Chunk with timestamps
    #    - Run through ingest pipeline
    #    - Return doc_id, video_id, status

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="YouTube ingestion not yet implemented",
    )
