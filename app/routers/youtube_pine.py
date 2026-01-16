# app/routers/youtube_pine.py
"""YouTube to Pine Script matching endpoint."""

from typing import Literal, Optional, cast
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.deps.security import require_admin_token
from app.routers.pine import match_pine_scripts
from app.routers.youtube import (
    fetch_transcript,
    fetch_video_metadata,
    parse_youtube_url,
)
from app.schemas import (
    IngestRequestHint,
    PineMatchRankedResult,
    YouTubeMatchPineRequest,
    YouTubeMatchPineResponse,
)
from app.services.chunker import normalize_transcript
from app.services.intent import (
    build_filters,
    build_query_string,
    get_intent_extractor,
    rerank,
)

router = APIRouter()
logger = structlog.get_logger(__name__)

# Max characters to use for extraction
MAX_EXTRACTION_CHARS = 50_000


async def check_kb_for_video(
    workspace_id: UUID,
    video_id: str,
) -> tuple[bool, Optional[UUID], Optional[str], Optional[list[str]]]:
    """
    Check if video is already in knowledge base.

    Returns:
        (in_kb, doc_id, title, chunk_texts)
    """
    from app.routers.ingest import _db_pool

    if _db_pool is None:
        return False, None, None, None

    async with _db_pool.acquire() as conn:
        # Check for document
        doc = await conn.fetchrow(
            """
            SELECT id, title
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
            return False, None, None, None

        doc_id = doc["id"]

        # Load top chunks
        rows = await conn.fetch(
            """
            SELECT content
            FROM chunks
            WHERE doc_id = $1
            ORDER BY chunk_index
            LIMIT 20
            """,
            doc_id,
        )

        chunk_texts = [r["content"] for r in rows]
        return True, doc_id, doc["title"], chunk_texts


@router.post(
    "/match-pine",
    response_model=YouTubeMatchPineResponse,
    responses={
        200: {"description": "Match results"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
        404: {"description": "Video not found or no transcript"},
        422: {"description": "Invalid request parameters"},
        502: {"description": "Upstream service failure"},
    },
    summary="Match YouTube video to Pine scripts",
    description="Extract trading intent from YouTube transcript and find matching Pine scripts.",
)
async def youtube_match_pine(
    request: YouTubeMatchPineRequest,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_token),
) -> YouTubeMatchPineResponse:
    """
    Match YouTube video content to Pine scripts.

    Hybrid flow:
    1. Check if video is already in KB
    2. If in KB (and not force_transient): use stored chunks
    3. Otherwise: fetch transcript transiently
    4. Extract trading intent
    5. Build query and filters
    6. Match against Pine scripts
    7. Rerank results
    """
    log = logger.bind(
        workspace_id=str(request.workspace_id),
        url=request.url,
        force_transient=request.force_transient,
    )
    log.info("youtube_pine_match_started")

    # Parse URL
    parsed = parse_youtube_url(request.url)
    video_id = parsed.get("video_id")

    if not video_id:
        raise HTTPException(422, "Invalid YouTube URL: could not extract video ID")

    # Check KB
    in_kb, kb_doc_id, kb_title, kb_chunks = await check_kb_for_video(
        request.workspace_id, video_id
    )

    # Determine transcript source
    transcript_text: str = ""
    title: Optional[str] = None
    channel: Optional[str] = None
    source_id: Optional[UUID] = kb_doc_id if in_kb else None
    transcript_source: Literal["kb", "transient"] = "transient"

    if in_kb and not request.force_transient and kb_chunks:
        # Use KB chunks
        transcript_source = "kb"
        transcript_text = " ".join(kb_chunks)[:MAX_EXTRACTION_CHARS]
        title = kb_title
        log.info("using_kb_chunks", chunk_count=len(kb_chunks))
    else:
        # Fetch transcript transiently
        try:
            # Fetch metadata
            metadata = await fetch_video_metadata(
                video_id, api_key=settings.youtube_api_key
            )
            title = metadata.get("title")
            channel = metadata.get("channel")

            # Fetch transcript
            transcript_segments = await fetch_transcript(video_id)
            if not transcript_segments:
                raise HTTPException(404, "No transcript available for this video")

            # Normalize and join
            raw_text = " ".join(seg.get("text", "") for seg in transcript_segments)
            transcript_text = normalize_transcript(raw_text)[:MAX_EXTRACTION_CHARS]

            log.info(
                "fetched_transcript",
                chars=len(transcript_text),
                segments=len(transcript_segments),
            )

        except HTTPException:
            raise
        except Exception as e:
            log.error("transcript_fetch_failed", error=str(e))
            raise HTTPException(502, f"Failed to fetch transcript: {e}")

    # Extract intent
    extractor = get_intent_extractor()
    intent = extractor.extract(transcript_text)

    log.info(
        "intent_extracted",
        archetypes=intent.strategy_archetypes,
        indicators=intent.indicators,
        confidence=intent.overall_confidence,
    )

    # Build query and filters
    query_string = build_query_string(intent, request)
    filters = build_filters(intent, request)

    log.info(
        "query_built",
        query=query_string,
        symbols=filters.symbols,
        script_type=filters.script_type,
    )

    # Call Pine match (internal)
    try:
        # Cast to match the function signature expectation
        script_type_val = cast(
            Optional[Literal["indicator", "strategy", "library"]],
            filters.script_type,
        )
        match_response = await match_pine_scripts(
            workspace_id=request.workspace_id,
            q=query_string,
            symbol=filters.symbols[0] if filters.symbols else None,
            script_type=script_type_val,
            lint_ok=filters.lint_ok if filters.lint_ok else None,
            top_k=request.top_k * 2,  # Get more for reranking
        )
    except Exception as e:
        log.error("pine_match_failed", error=str(e))
        raise HTTPException(500, f"Pine match failed: {e}")

    # Rerank results
    ranked = rerank(match_response.results, intent)

    # Convert to response format
    ranked_results = [
        PineMatchRankedResult(
            id=r.result.id,
            rel_path=r.result.rel_path,
            title=r.result.title,
            script_type=r.result.script_type,
            pine_version=r.result.pine_version,
            score=r.result.score,
            match_reasons=r.result.match_reasons,
            snippet=r.result.snippet,
            inputs_preview=r.result.inputs_preview,
            lint_ok=r.result.lint_ok,
            base_score=r.base_score,
            boost=r.boost,
            final_score=r.final_score,
        )
        for r in ranked[: request.top_k]
    ]

    # Build response
    return YouTubeMatchPineResponse(
        video_id=video_id,
        title=title,
        channel=channel,
        in_knowledge_base=in_kb,
        source_id=source_id,
        transcript_source=transcript_source,
        transcript_chars_used=len(transcript_text),
        match_intent=intent.model_dump(),
        extraction_method="rule_based",
        results=ranked_results,
        total_searched=match_response.total_searched,
        query_used=query_string,
        filters_applied=filters.model_dump(),
        ingest_available=not in_kb,
        ingest_request_hint=(
            IngestRequestHint(workspace_id=request.workspace_id, url=request.url)
            if not in_kb
            else None
        ),
    )
