"""Ingest-related schemas: documents, chunks, YouTube, PDF, reembed."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl

from app.schemas.common import JobStatus, SourceType


# ===========================================
# Source & Document Metadata
# ===========================================


class SourceInfo(BaseModel):
    """Source information for document ingestion."""

    url: Optional[HttpUrl] = Field(None, description="Source URL")
    type: SourceType = Field(..., description="Source type")


class DocumentMetadata(BaseModel):
    """Optional metadata for document ingestion."""

    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Author or channel name")
    published_at: Optional[datetime] = Field(None, description="Publication date")
    language: Optional[str] = Field(default="en", description="Content language")
    duration_secs: Optional[int] = Field(None, description="Duration in seconds")


class ChunkInput(BaseModel):
    """Pre-chunked content input."""

    content: str = Field(..., description="Chunk content")
    time_start_secs: Optional[int] = Field(None, description="Start time in seconds")
    time_end_secs: Optional[int] = Field(None, description="End time in seconds")
    page_start: Optional[int] = Field(None, description="Start page number")
    page_end: Optional[int] = Field(None, description="End page number")
    section: Optional[str] = Field(None, description="Section name")
    speaker: Optional[str] = Field(None, description="Speaker name")


# ===========================================
# Generic Ingest
# ===========================================


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")
    content_hash: Optional[str] = Field(None, description="Pre-computed content hash")
    source: SourceInfo = Field(..., description="Source information")
    content: str = Field(..., min_length=1, description="Document content")
    metadata: Optional[DocumentMetadata] = Field(
        default=None, description="Document metadata"
    )
    chunks: Optional[list[ChunkInput]] = Field(
        default=None, description="Pre-chunked content"
    )
    video_id: Optional[str] = Field(
        default=None, description="YouTube video ID (for youtube source type)"
    )
    update_existing: bool = Field(
        default=False,
        description="If true, supersede existing document at same canonical_url and create new version",  # noqa: E501
    )


class IngestResponse(BaseModel):
    """Response for document ingestion."""

    doc_id: UUID = Field(..., description="Created document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    vectors_created: int = Field(..., description="Number of vectors created")
    status: str = Field(..., description="Ingestion status")
    version: int = Field(default=1, description="Document version number")
    superseded_doc_id: Optional[UUID] = Field(
        default=None, description="ID of superseded document if this was an update"
    )


# ===========================================
# YouTube Ingest
# ===========================================


class YouTubeIngestRequest(BaseModel):
    """Request body for YouTube ingestion."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    url: str = Field(..., description="YouTube URL (video or playlist)")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")


class YouTubeIngestResponse(BaseModel):
    """Response for YouTube ingestion."""

    doc_id: Optional[UUID] = Field(None, description="Created document ID")
    video_id: Optional[str] = Field(None, description="YouTube video ID")
    playlist_id: Optional[str] = Field(None, description="YouTube playlist ID")
    status: str = Field(..., description="Ingestion status")
    retryable: bool = Field(default=False, description="Whether error is retryable")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    is_playlist: bool = Field(default=False, description="Whether URL is a playlist")
    video_urls: Optional[list[str]] = Field(
        default=None, description="Video URLs if playlist"
    )
    error_reason: Optional[str] = Field(None, description="Error reason if failed")


# ===========================================
# PDF Ingest
# ===========================================


class PDFBackendType(str, Enum):
    """Available PDF extraction backends."""

    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"


class PDFConfig(BaseModel):
    """Configuration for PDF extraction.

    Matches workspace.config.pdf schema.
    """

    backend: PDFBackendType = Field(
        default=PDFBackendType.PYMUPDF, description="PDF extraction backend"
    )
    max_pages: Optional[int] = Field(
        None, description="Max pages to extract (None=all)"
    )
    min_chars_per_page: int = Field(10, description="Skip pages with fewer chars")
    join_pages_with: str = Field("\n\n", description="Separator between pages")
    enable_ocr: bool = Field(False, description="Enable OCR (reserved for future)")


class PDFIngestRequest(BaseModel):
    """Request body for PDF ingestion (JSON metadata, file via multipart)."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")
    title: Optional[str] = Field(
        None, description="Document title (auto-detected if not set)"
    )
    author: Optional[str] = Field(None, description="Author name")
    pdf_config: Optional[PDFConfig] = Field(None, description="PDF extraction config")


class PDFIngestResponse(BaseModel):
    """Response for PDF ingestion."""

    doc_id: Optional[UUID] = Field(None, description="Created document ID")
    status: str = Field(..., description="Ingestion status")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    vectors_created: int = Field(default=0, description="Number of vectors created")
    pages_extracted: int = Field(default=0, description="Number of pages extracted")
    total_pages: int = Field(default=0, description="Total pages in PDF")
    warnings: list[str] = Field(default_factory=list, description="Extraction warnings")
    error_reason: Optional[str] = Field(None, description="Error reason if failed")


# ===========================================
# Reembed & Job Status
# ===========================================


class ReembedRequest(BaseModel):
    """Request body for re-embedding."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    target_collection: str = Field(..., description="Target collection name")
    embed_provider: str = Field(default="ollama", description="Embedding provider")
    embed_model: str = Field(..., description="Embedding model name")
    doc_ids: Optional[list[UUID]] = Field(
        default=None, description="Specific documents to re-embed"
    )


class ReembedResponse(BaseModel):
    """Response for re-embed endpoint."""

    job_id: UUID = Field(..., description="Job identifier")
    chunks_queued: int = Field(..., description="Number of chunks queued")
    status: JobStatus = Field(..., description="Job status")


class JobResponse(BaseModel):
    """Response for job status endpoint."""

    job_id: UUID = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(..., description="Progress percentage (0-100)")
    error: Optional[str] = Field(None, description="Error message if failed")
