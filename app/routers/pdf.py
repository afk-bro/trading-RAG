"""PDF ingestion endpoint."""

import hashlib
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.schemas import PDFIngestResponse, SourceType
from app.services.chunker import Chunk, Chunker
from app.services.pdf_extractor import (
    PDFBackend,
    PDFConfig as ExtractorPDFConfig,
    extract_pdf,
    get_page_markers,
)

router = APIRouter()
logger = structlog.get_logger(__name__)

# Import the ingest pipeline from main ingest router
from app.routers.ingest import ingest_pipeline  # noqa: E402


def compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


@router.post(
    "/sources/pdf/ingest",
    response_model=PDFIngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "Document already exists (idempotent)"},
        201: {"description": "Document created successfully"},
        400: {"description": "Invalid PDF file"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    workspace_id: UUID = Form(..., description="Workspace identifier"),
    idempotency_key: Optional[str] = Form(None, description="Idempotency key"),
    title: Optional[str] = Form(None, description="Document title"),
    author: Optional[str] = Form(None, description="Author name"),
    backend: str = Form("pymupdf", description="PDF backend: pymupdf or pdfplumber"),
    max_pages: Optional[int] = Form(None, description="Max pages to extract"),
    min_chars_per_page: int = Form(10, description="Min chars per page"),
    settings: Settings = Depends(get_settings),
) -> PDFIngestResponse:
    """
    Ingest a PDF document into the RAG pipeline.

    This endpoint:
    1. Accepts PDF file upload via multipart form
    2. Extracts text using configurable backend (PyMuPDF or pdfplumber)
    3. Chunks text with page number preservation
    4. Generates embeddings and stores in Qdrant
    5. Stores document/chunks in Supabase

    The operation is idempotent when idempotency_key is provided.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a PDF",
        )

    # Read file content
    try:
        file_bytes = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    logger.info(
        "Ingesting PDF",
        workspace_id=str(workspace_id),
        filename=file.filename,
        file_size=len(file_bytes),
        backend=backend,
        max_pages=max_pages,
    )

    # Build PDF config
    try:
        pdf_backend = PDFBackend(backend)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid backend: {backend}. Use 'pymupdf' or 'pdfplumber'",
        )

    pdf_config = ExtractorPDFConfig(
        backend=pdf_backend,
        max_pages=max_pages,
        min_chars_per_page=min_chars_per_page,
    )

    # Extract PDF content
    try:
        extraction_result = extract_pdf(file_bytes, pdf_config)
    except ImportError as e:
        logger.error("PDF library not installed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.error("PDF extraction failed", error=str(e))
        return PDFIngestResponse(
            doc_id=None,
            status="failed",
            error_reason=f"PDF extraction failed: {str(e)}",
            warnings=extraction_result.warnings if "extraction_result" in dir() else [],
        )

    if not extraction_result.text.strip():
        logger.warning("PDF has no extractable text", filename=file.filename)
        return PDFIngestResponse(
            doc_id=None,
            status="failed",
            total_pages=extraction_result.page_count,
            pages_extracted=0,
            error_reason="No extractable text in PDF",
            warnings=extraction_result.warnings,
        )

    # Use title from metadata if not provided
    doc_title = title or extraction_result.metadata.get("title") or file.filename

    # Compute content hash
    content_hash = compute_content_hash(file_bytes)

    # Determine canonical URL
    canonical_url = idempotency_key or f"pdf://{content_hash[:32]}"

    # Create page-aware chunks
    chunker = Chunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        encoding_name=settings.chunk_tokenizer_encoding,
    )

    # Get page markers for chunk assignment
    page_markers = get_page_markers(extraction_result)

    # Chunk the text
    raw_chunks = chunker.chunk_text(extraction_result.text)

    # Assign page numbers to chunks based on character positions
    chunks_with_pages = []
    for chunk in raw_chunks:
        # Find which page this chunk starts on
        # Simple approach: find the page whose char_start is <= chunk start position
        chunk_start_char = 0
        for i, prev_chunk in enumerate(raw_chunks):
            if prev_chunk is chunk:
                break
            chunk_start_char += len(prev_chunk.content) + len(
                pdf_config.join_pages_with
            )

        # Find page for this position
        page_num = 1
        for char_start, pn in page_markers:
            if char_start <= chunk_start_char:
                page_num = pn
            else:
                break

        # Find end page (for chunks that span pages)
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

    # Convert to pre_chunks format for ingest_pipeline
    from app.schemas import ChunkInput

    pre_chunks = [
        ChunkInput(
            content=c.content,
            page_start=c.page_start,
            page_end=c.page_end,
        )
        for c in chunks_with_pages
    ]

    # Call the main ingest pipeline
    try:
        ingest_result = await ingest_pipeline(
            workspace_id=workspace_id,
            content=extraction_result.text,
            source_type=SourceType.PDF,
            source_url=None,
            canonical_url=canonical_url,
            idempotency_key=idempotency_key,
            content_hash=content_hash,
            title=doc_title,
            author=author or extraction_result.metadata.get("author"),
            pre_chunks=pre_chunks,
            settings=settings,
        )

        logger.info(
            "PDF ingestion complete",
            doc_id=str(ingest_result.doc_id),
            status=ingest_result.status,
            chunks_created=ingest_result.chunks_created,
            pages_extracted=extraction_result.extracted_page_count,
        )

        return PDFIngestResponse(
            doc_id=ingest_result.doc_id,
            status=ingest_result.status,
            chunks_created=ingest_result.chunks_created,
            vectors_created=ingest_result.vectors_created,
            pages_extracted=extraction_result.extracted_page_count,
            total_pages=extraction_result.page_count,
            warnings=extraction_result.warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF ingestion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )
