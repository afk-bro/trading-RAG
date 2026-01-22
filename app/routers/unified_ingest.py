"""Unified ingestion endpoint with auto-detection."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.deps.security import require_admin_token
from app.schemas import IngestResponse, SourceType
from app.services.ingest import DetectedSource, detect_source_type

router = APIRouter()
logger = structlog.get_logger(__name__)


class UnifiedIngestResponse(IngestResponse):
    """Extended response with detection debug info."""

    detected_type: Optional[str] = None
    normalized_url: Optional[str] = None


@router.post(
    "/ingest/unified",
    response_model=UnifiedIngestResponse,
    responses={
        200: {"description": "Content ingested successfully"},
        400: {"description": "Invalid input or unsupported file type"},
        403: {"description": "Admin token required"},
        422: {"description": "Missing required input"},
        500: {"description": "Internal server error"},
    },
    tags=["Ingestion"],
)
async def unified_ingest(
    workspace_id: UUID = Form(..., description="Workspace identifier"),
    url: Optional[str] = Form(
        None, description="URL to ingest (YouTube, article, or PDF)"
    ),
    content: Optional[str] = Form(None, description="Raw text/markdown content"),
    title: Optional[str] = Form(None, description="Override auto-detected title"),
    source_type: Optional[str] = Form(None, description="Override auto-detection"),
    file: Optional[UploadFile] = File(
        None, description="File upload (.pdf, .txt, .md)"
    ),
    idempotency_key: Optional[str] = Form(None, description="Idempotency key"),
    _: bool = Depends(require_admin_token),
    settings: Settings = Depends(get_settings),
) -> UnifiedIngestResponse:
    """
    Unified ingestion endpoint with auto-detection.

    Accepts URLs, file uploads, or raw content and automatically routes
    to the appropriate handler based on content type detection.

    **Detection logic:**
    - YouTube URLs (youtube.com, youtu.be) → YouTube transcript ingestion
    - PDF URLs (*.pdf) → Fetch and process as PDF
    - Other URLs → Article extraction (trafilatura)
    - .pdf files → PDF extraction
    - .txt/.md files → Text/markdown ingestion
    - .pine files → Pine script ingestion
    - Raw content → Generic text ingestion

    **Input requirements:**
    - Exactly one of `url`, `file`, or `content` must be provided
    - `workspace_id` is always required

    **Override:**
    - Use `source_type` to override auto-detection (must match input type)
    """
    logger.info(
        "Unified ingest request",
        workspace_id=str(workspace_id),
        has_url=bool(url),
        has_file=bool(file),
        has_content=bool(content),
        source_type_override=source_type,
    )

    # Validate exactly one input
    input_count = sum([bool(url), bool(file), bool(content)])
    if input_count == 0:
        raise HTTPException(
            status_code=422,
            detail="Must provide exactly one of: url, file, or content",
        )
    if input_count > 1:
        raise HTTPException(
            status_code=422,
            detail="Must provide exactly one of: url, file, or content (got multiple)",
        )

    # Detect source type
    filename = file.filename if file else None
    detected = detect_source_type(
        url=url,
        filename=filename,
        content=content,
        source_type_override=source_type,
    )

    logger.info("Detected source type", detected=detected.value)

    # Dispatch to appropriate handler
    try:
        if detected == DetectedSource.YOUTUBE:
            response = await _handle_youtube(
                workspace_id=workspace_id,
                url=url,
                title_override=title,
                idempotency_key=idempotency_key,
                settings=settings,
            )
        elif detected == DetectedSource.ARTICLE_URL:
            response = await _handle_article(
                workspace_id=workspace_id,
                url=url,
                title_override=title,
                idempotency_key=idempotency_key,
                settings=settings,
            )
        elif detected == DetectedSource.PDF_URL:
            response = await _handle_pdf_url(
                workspace_id=workspace_id,
                url=url,
                title_override=title,
                idempotency_key=idempotency_key,
                settings=settings,
            )
        elif detected == DetectedSource.PDF_FILE:
            response = await _handle_pdf_file(
                workspace_id=workspace_id,
                file=file,
                title_override=title,
                idempotency_key=idempotency_key,
                settings=settings,
            )
        elif detected == DetectedSource.TEXT_FILE:
            response = await _handle_text_file(
                workspace_id=workspace_id,
                file=file,
                title_override=title,
                idempotency_key=idempotency_key,
                settings=settings,
            )
        elif detected == DetectedSource.PINE_FILE:
            response = await _handle_pine_file(
                workspace_id=workspace_id,
                file=file,
                title_override=title,
                idempotency_key=idempotency_key,
                settings=settings,
            )
        elif detected == DetectedSource.TEXT_CONTENT:
            response = await _handle_text_content(
                workspace_id=workspace_id,
                content=content,
                title_override=title,
                idempotency_key=idempotency_key,
                settings=settings,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported source type: {detected.value}",
            )

        # Return extended response with debug info
        return UnifiedIngestResponse(
            doc_id=response.doc_id,
            chunks_created=response.chunks_created,
            vectors_created=response.vectors_created,
            status=response.status,
            version=response.version,
            superseded_doc_id=response.superseded_doc_id,
            detected_type=detected.value,
            normalized_url=url,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unified ingest failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


# Handler implementations


async def _handle_youtube(
    workspace_id: UUID,
    url: str,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Handle YouTube URL ingestion."""
    from app.routers.youtube import (
        fetch_transcript,
        fetch_video_metadata,
        normalize_transcript,
        parse_youtube_url,
    )
    from app.routers.ingest import compute_content_hash, ingest_pipeline
    from app.services.chunker import Chunker
    from app.schemas import ChunkInput
    from datetime import datetime

    # Parse URL
    parsed = parse_youtube_url(url)
    video_id = parsed["video_id"]

    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="Could not extract video ID from YouTube URL",
        )

    # Playlists not supported in unified endpoint
    if parsed["is_playlist"]:
        raise HTTPException(
            status_code=400,
            detail="Playlists not supported in unified endpoint. Use /sources/youtube/ingest",
        )

    # Fetch metadata
    metadata = await fetch_video_metadata(
        video_id=video_id,
        api_key=settings.youtube_api_key,
    )

    # Fetch transcript
    transcript_result = await fetch_transcript(video_id)
    segments = transcript_result["segments"]
    transcript_language = transcript_result["language"]
    is_auto_generated = transcript_result["is_auto_generated"]

    # Normalize and chunk
    full_text = " ".join(seg["text"] for seg in segments)
    normalized_text = normalize_transcript(full_text)

    chunker = Chunker()
    chunks = chunker.chunk_timestamped_content(segments)

    # Parse published_at
    published_at = None
    if metadata.get("published_at"):
        try:
            published_at = datetime.fromisoformat(
                metadata["published_at"].replace("Z", "+00:00")
            )
        except Exception:
            pass

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    pre_chunks = [
        ChunkInput(
            content=chunk.content,
            time_start_secs=chunk.time_start_secs,
            time_end_secs=chunk.time_end_secs,
        )
        for chunk in chunks
    ]

    return await ingest_pipeline(
        workspace_id=workspace_id,
        content=normalized_text,
        source_type=SourceType.YOUTUBE,
        source_url=url,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key or f"youtube:{video_id}",
        content_hash=compute_content_hash(normalized_text),
        title=title_override or metadata.get("title"),
        author=metadata.get("channel"),
        published_at=published_at,
        language=transcript_language or "en",
        duration_secs=metadata.get("duration_secs"),
        video_id=video_id,
        pre_chunks=pre_chunks,
        settings=settings,
        is_auto_generated=is_auto_generated,
    )


async def _handle_article(
    workspace_id: UUID,
    url: str,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Handle article URL ingestion."""
    from app.services.article_extractor import extract_article
    from app.routers.ingest import compute_content_hash, ingest_pipeline

    # Extract article
    article = await extract_article(url)

    canonical_url = article.url  # Use final URL after redirects
    content_hash = compute_content_hash(article.text)

    return await ingest_pipeline(
        workspace_id=workspace_id,
        content=article.text,
        source_type=SourceType.ARTICLE,
        source_url=url,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key or f"article:{content_hash[:16]}",
        content_hash=content_hash,
        title=title_override or article.title,
        author=article.author,
        published_at=article.published_at,
        language="en",
        settings=settings,
    )


async def _handle_pdf_url(
    workspace_id: UUID,
    url: str,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Handle PDF URL ingestion (fetch and process)."""
    import httpx

    # Fetch PDF
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        pdf_bytes = response.content

    # Extract filename from URL
    filename = url.split("/")[-1].split("?")[0]
    if not filename.lower().endswith(".pdf"):
        filename = "document.pdf"

    return await _process_pdf_bytes(
        workspace_id=workspace_id,
        pdf_bytes=pdf_bytes,
        filename=filename,
        title_override=title_override,
        idempotency_key=idempotency_key,
        source_url=url,
        settings=settings,
    )


async def _handle_pdf_file(
    workspace_id: UUID,
    file: UploadFile,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Handle PDF file upload."""
    pdf_bytes = await file.read()

    return await _process_pdf_bytes(
        workspace_id=workspace_id,
        pdf_bytes=pdf_bytes,
        filename=file.filename or "upload.pdf",
        title_override=title_override,
        idempotency_key=idempotency_key,
        source_url=None,
        settings=settings,
    )


async def _process_pdf_bytes(
    workspace_id: UUID,
    pdf_bytes: bytes,
    filename: str,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    source_url: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Process PDF bytes and ingest into pipeline."""
    import hashlib
    from app.routers.ingest import ingest_pipeline
    from app.services.chunker import Chunk, Chunker
    from app.services.pdf_extractor import (
        PDFBackend,
        PDFConfig as ExtractorPDFConfig,
        extract_pdf,
        get_page_markers,
    )
    from app.schemas import ChunkInput

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty PDF file")

    # Extract PDF content
    pdf_config = ExtractorPDFConfig(
        backend=PDFBackend.PYMUPDF,
        max_pages=None,
        min_chars_per_page=10,
    )

    try:
        extraction_result = extract_pdf(pdf_bytes, pdf_config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

    if not extraction_result.text.strip():
        raise HTTPException(status_code=400, detail="No extractable text in PDF")

    # Compute content hash
    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Determine title and canonical URL
    doc_title = title_override or extraction_result.metadata.get("title") or filename
    canonical_url = idempotency_key or f"pdf://{content_hash[:32]}"

    # Create page-aware chunks
    chunker = Chunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        encoding_name=settings.chunk_tokenizer_encoding,
    )

    page_markers = get_page_markers(extraction_result)
    raw_chunks = chunker.chunk_text(extraction_result.text)

    # Assign page numbers to chunks
    chunks_with_pages = []
    for chunk in raw_chunks:
        chunk_start_char = 0
        for i, prev_chunk in enumerate(raw_chunks):
            if prev_chunk is chunk:
                break
            chunk_start_char += len(prev_chunk.content) + len(
                pdf_config.join_pages_with
            )

        page_num = 1
        for char_start, pn in page_markers:
            if char_start <= chunk_start_char:
                page_num = pn
            else:
                break

        chunk_end_char = chunk_start_char + len(chunk.content)
        end_page = page_num
        for char_start, pn in page_markers:
            if char_start <= chunk_end_char:
                end_page = pn

        chunks_with_pages.append(
            Chunk(
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                page_start=page_num,
                page_end=end_page if end_page != page_num else None,
                locator_label=(
                    f"p. {page_num}"
                    if page_num == end_page
                    else f"pp. {page_num}-{end_page}"
                ),
            )
        )

    pre_chunks = [
        ChunkInput(
            content=c.content,
            page_start=c.page_start,
            page_end=c.page_end,
        )
        for c in chunks_with_pages
    ]

    return await ingest_pipeline(
        workspace_id=workspace_id,
        content=extraction_result.text,
        source_type=SourceType.PDF,
        source_url=source_url,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key,
        content_hash=content_hash,
        title=doc_title,
        author=extraction_result.metadata.get("author"),
        pre_chunks=pre_chunks,
        settings=settings,
    )


async def _handle_text_file(
    workspace_id: UUID,
    file: UploadFile,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Handle text/markdown file upload."""
    from app.services.ingest.text import extract_text_content
    from app.routers.ingest import ingest_pipeline

    # Extract content
    extracted = await extract_text_content(
        file=file,
        title_override=title_override,
    )

    canonical_url = f"text://{extracted.content_hash[:32]}"

    return await ingest_pipeline(
        workspace_id=workspace_id,
        content=extracted.text,
        source_type=(
            SourceType.NOTE if not extracted.is_markdown else SourceType.ARTICLE
        ),
        source_url=None,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key or canonical_url,
        content_hash=extracted.content_hash,
        title=extracted.title,
        author=None,
        published_at=None,
        language="en",
        settings=settings,
    )


async def _handle_pine_file(
    workspace_id: UUID,
    file: UploadFile,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Handle Pine script file upload."""
    from app.routers.ingest import compute_content_hash, ingest_pipeline

    # Read content
    raw_bytes = await file.read()
    content = raw_bytes.decode("utf-8")

    # Extract title from first line comment or filename
    title = title_override
    if not title:
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("//"):
                title = line[2:].strip()
                break
        if not title:
            title = file.filename or "pine_script"

    content_hash = compute_content_hash(content)
    canonical_url = f"pine://{content_hash[:32]}"

    return await ingest_pipeline(
        workspace_id=workspace_id,
        content=content,
        source_type=SourceType.PINE_SCRIPT,
        source_url=None,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key or canonical_url,
        content_hash=content_hash,
        title=title,
        author=None,
        published_at=None,
        language="en",
        settings=settings,
    )


async def _handle_text_content(
    workspace_id: UUID,
    content: str,
    title_override: Optional[str],
    idempotency_key: Optional[str],
    settings: Settings,
) -> IngestResponse:
    """Handle raw text/markdown content."""
    from app.services.ingest.text import extract_text_content
    from app.routers.ingest import ingest_pipeline

    # Extract content
    extracted = await extract_text_content(
        content=content,
        title_override=title_override,
    )

    canonical_url = f"text://{extracted.content_hash[:32]}"

    return await ingest_pipeline(
        workspace_id=workspace_id,
        content=extracted.text,
        source_type=SourceType.NOTE,
        source_url=None,
        canonical_url=canonical_url,
        idempotency_key=idempotency_key or canonical_url,
        content_hash=extracted.content_hash,
        title=extracted.title,
        author=None,
        published_at=None,
        language="en",
        settings=settings,
    )
