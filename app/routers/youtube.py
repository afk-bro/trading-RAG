"""YouTube ingestion endpoint."""

import asyncio
import re
from datetime import datetime
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.routers.ingest import compute_content_hash, ingest_pipeline
from app.schemas import SourceType, YouTubeIngestRequest, YouTubeIngestResponse
from app.services.chunker import Chunker, normalize_transcript

router = APIRouter()
logger = structlog.get_logger(__name__)


def parse_youtube_url(url: str) -> dict:
    """
    Parse YouTube URL to extract video/playlist IDs.

    Returns:
        dict with keys:
        - video_id: str or None
        - playlist_id: str or None
        - is_playlist: bool
    """
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    video_id = None
    playlist_id = None

    # Handle different URL formats
    if "youtube.com" in parsed.netloc or "www.youtube.com" in parsed.netloc:
        # Standard YouTube URLs
        if "/watch" in parsed.path:
            video_id = query.get("v", [None])[0]
        elif "/playlist" in parsed.path:
            playlist_id = query.get("list", [None])[0]
        elif "/embed/" in parsed.path:
            # /embed/VIDEO_ID
            parts = parsed.path.split("/embed/")
            if len(parts) > 1:
                video_id = parts[1].split("/")[0].split("?")[0]
        elif "/v/" in parsed.path:
            # /v/VIDEO_ID
            parts = parsed.path.split("/v/")
            if len(parts) > 1:
                video_id = parts[1].split("/")[0].split("?")[0]

        # Check for playlist in query params
        if not playlist_id:
            playlist_id = query.get("list", [None])[0]

    elif "youtu.be" in parsed.netloc:
        # Short YouTube URLs: youtu.be/VIDEO_ID
        video_id = parsed.path.lstrip("/").split("?")[0]
        playlist_id = query.get("list", [None])[0]

    # Determine if this is primarily a playlist request
    is_playlist = playlist_id is not None and video_id is None

    return {
        "video_id": video_id,
        "playlist_id": playlist_id,
        "is_playlist": is_playlist,
    }


async def fetch_video_metadata(video_id: str, api_key: Optional[str] = None) -> dict:
    """
    Fetch video metadata from YouTube.

    If no API key, uses basic oembed endpoint.
    """
    if api_key:
        # Use YouTube Data API
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "id": video_id,
            "part": "snippet,contentDetails",
            "key": api_key,
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data.get("items"):
                raise ValueError(f"Video not found: {video_id}")

            item = data["items"][0]
            snippet = item["snippet"]
            content_details = item.get("contentDetails", {})

            # Parse duration (ISO 8601)
            duration_str = content_details.get("duration", "PT0S")
            duration_secs = parse_iso8601_duration(duration_str)

            return {
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "duration_secs": duration_secs,
                "description": snippet.get("description", "")[:500],
            }
    else:
        # Use oembed (no API key required, but limited info)
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            return {
                "title": data.get("title"),
                "channel": data.get("author_name"),
                "published_at": None,  # Not available via oembed
                "duration_secs": None,
                "description": "",
            }


def parse_iso8601_duration(duration: str) -> int:
    """Parse ISO 8601 duration (e.g., PT1H30M45S) to seconds."""
    pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
    match = re.match(pattern, duration)
    if not match:
        return 0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return hours * 3600 + minutes * 60 + seconds


async def fetch_transcript(
    video_id: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> list[dict]:
    """
    Fetch transcript for a YouTube video using youtube-transcript-api.

    Returns list of segments with 'text', 'start', 'duration' keys.
    """
    from youtube_transcript_api import (
        NoTranscriptFound,
        TranscriptsDisabled,
        YouTubeTranscriptApi,
    )

    delay = initial_delay

    for attempt in range(max_retries):
        try:
            # Try to get transcript (prefers manual > auto-generated)
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Prefer manually created transcripts
            try:
                transcript = transcript_list.find_manually_created_transcript(["en"])
            except Exception:
                # Fall back to auto-generated
                try:
                    transcript = transcript_list.find_generated_transcript(["en"])
                except Exception:
                    # Try any available transcript
                    transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])

            # Fetch the transcript data
            transcript_data = transcript.fetch()

            return [
                {
                    "text": segment.get("text", ""),
                    "start": segment.get("start", 0),
                    "end": segment.get("start", 0) + segment.get("duration", 0),
                }
                for segment in transcript_data
            ]

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            # Terminal error - no transcript available
            logger.warning(
                "No transcript available",
                video_id=video_id,
                error=str(e),
            )
            raise ValueError(f"No transcript available for video {video_id}")

        except Exception as e:
            logger.warning(
                "Transcript fetch failed, retrying",
                video_id=video_id,
                attempt=attempt + 1,
                error=str(e),
            )

            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise

    raise RuntimeError(f"Failed to fetch transcript after {max_retries} attempts")


async def fetch_playlist_videos(
    playlist_id: str,
    api_key: str,
    max_results: int = 50,
) -> list[str]:
    """
    Fetch video URLs from a YouTube playlist.

    Requires YouTube Data API key.
    """
    url = "https://www.googleapis.com/youtube/v3/playlistItems"
    video_urls = []
    next_page_token = None

    async with httpx.AsyncClient() as client:
        while True:
            params = {
                "playlistId": playlist_id,
                "part": "contentDetails",
                "maxResults": min(max_results - len(video_urls), 50),
                "key": api_key,
            }

            if next_page_token:
                params["pageToken"] = next_page_token

            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            for item in data.get("items", []):
                video_id = item.get("contentDetails", {}).get("videoId")
                if video_id:
                    video_urls.append(f"https://www.youtube.com/watch?v={video_id}")

            next_page_token = data.get("nextPageToken")

            if not next_page_token or len(video_urls) >= max_results:
                break

    return video_urls


@router.post(
    "/ingest",
    response_model=YouTubeIngestResponse,
    responses={
        200: {"description": "Video ingested or playlist expanded"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def ingest_youtube(
    request: YouTubeIngestRequest,
    settings: Settings = Depends(get_settings),
) -> YouTubeIngestResponse:
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

    # Parse URL
    parsed = parse_youtube_url(request.url)
    video_id = parsed["video_id"]
    playlist_id = parsed["playlist_id"]
    is_playlist = parsed["is_playlist"]

    logger.info(
        "Parsed YouTube URL",
        video_id=video_id,
        playlist_id=playlist_id,
        is_playlist=is_playlist,
    )

    # Handle playlist
    if is_playlist:
        if not settings.youtube_api_key:
            return YouTubeIngestResponse(
                status="error",
                retryable=False,
                is_playlist=True,
                playlist_id=playlist_id,
                error_reason="YouTube API key required for playlist expansion",
            )

        try:
            video_urls = await fetch_playlist_videos(
                playlist_id=playlist_id,
                api_key=settings.youtube_api_key,
            )

            logger.info(
                "Expanded playlist",
                playlist_id=playlist_id,
                video_count=len(video_urls),
            )

            return YouTubeIngestResponse(
                status="playlist_expanded",
                is_playlist=True,
                playlist_id=playlist_id,
                video_urls=video_urls,
            )

        except Exception as e:
            logger.error(
                "Playlist expansion failed",
                playlist_id=playlist_id,
                error=str(e),
            )
            return YouTubeIngestResponse(
                status="error",
                retryable=True,
                is_playlist=True,
                playlist_id=playlist_id,
                error_reason=f"Failed to expand playlist: {str(e)}",
            )

    # Handle single video
    if not video_id:
        return YouTubeIngestResponse(
            status="error",
            retryable=False,
            error_reason="Could not extract video ID from URL",
        )

    # Fetch metadata
    try:
        metadata = await fetch_video_metadata(
            video_id=video_id,
            api_key=settings.youtube_api_key,
        )
        logger.info(
            "Fetched video metadata",
            video_id=video_id,
            title=metadata.get("title"),
        )
    except Exception as e:
        logger.error(
            "Metadata fetch failed",
            video_id=video_id,
            error=str(e),
        )
        return YouTubeIngestResponse(
            video_id=video_id,
            status="error",
            retryable=True,
            error_reason=f"Failed to fetch metadata: {str(e)}",
        )

    # Fetch transcript
    try:
        segments = await fetch_transcript(video_id)
        logger.info(
            "Fetched transcript",
            video_id=video_id,
            segments=len(segments),
        )
    except ValueError as e:
        # Terminal error - no transcript
        logger.warning(
            "No transcript available",
            video_id=video_id,
            error=str(e),
        )
        return YouTubeIngestResponse(
            video_id=video_id,
            status="error",
            retryable=False,
            error_reason="no_transcript",
        )
    except Exception as e:
        # Retryable error
        logger.error(
            "Transcript fetch failed",
            video_id=video_id,
            error=str(e),
        )
        return YouTubeIngestResponse(
            video_id=video_id,
            status="error",
            retryable=True,
            error_reason=f"Failed to fetch transcript: {str(e)}",
        )

    # Combine and normalize transcript
    full_text = " ".join(seg["text"] for seg in segments)
    normalized_text = normalize_transcript(full_text)

    # Chunk with timestamps
    chunker = Chunker()
    chunks = chunker.chunk_timestamped_content(segments)

    logger.info(
        "Chunked transcript",
        video_id=video_id,
        chunks=len(chunks),
    )

    # Parse published_at if available
    published_at = None
    if metadata.get("published_at"):
        try:
            published_at = datetime.fromisoformat(
                metadata["published_at"].replace("Z", "+00:00")
            )
        except Exception:
            pass

    # Build canonical URL
    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    # Run through ingestion pipeline
    try:
        response = await ingest_pipeline(
            workspace_id=request.workspace_id,
            content=normalized_text,
            source_type=SourceType.YOUTUBE,
            source_url=request.url,
            canonical_url=canonical_url,
            idempotency_key=request.idempotency_key or f"youtube:{video_id}",
            content_hash=compute_content_hash(normalized_text),
            title=metadata.get("title"),
            author=metadata.get("channel"),
            published_at=published_at,
            language="en",
            duration_secs=metadata.get("duration_secs"),
            pre_chunks=None,  # We'll handle chunking ourselves with timestamps
            settings=settings,
        )

        logger.info(
            "YouTube ingestion complete",
            video_id=video_id,
            doc_id=str(response.doc_id),
            chunks_created=response.chunks_created,
        )

        return YouTubeIngestResponse(
            doc_id=response.doc_id,
            video_id=video_id,
            playlist_id=playlist_id,
            status="ingested" if response.status == "indexed" else response.status,
            chunks_created=response.chunks_created,
        )

    except HTTPException as e:
        logger.error(
            "Ingestion pipeline failed",
            video_id=video_id,
            error=e.detail,
        )
        return YouTubeIngestResponse(
            video_id=video_id,
            status="error",
            retryable=True,
            error_reason=str(e.detail),
        )

    except Exception as e:
        logger.exception(
            "Unexpected error during ingestion",
            video_id=video_id,
            error=str(e),
        )
        return YouTubeIngestResponse(
            video_id=video_id,
            status="error",
            retryable=True,
            error_reason=f"Ingestion failed: {str(e)}",
        )
