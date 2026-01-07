"""Pydantic models for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


# Enums
class SourceType(str, Enum):
    """Supported source types for documents."""

    YOUTUBE = "youtube"
    PDF = "pdf"
    ARTICLE = "article"
    NOTE = "note"
    TRANSCRIPT = "transcript"


class DocumentStatus(str, Enum):
    """Document lifecycle status."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class VectorStatus(str, Enum):
    """Vector indexing status."""

    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class QueryMode(str, Enum):
    """Query response mode."""

    RETRIEVE = "retrieve"
    ANSWER = "answer"


class SymbolsMode(str, Enum):
    """Symbol filter matching mode."""

    ANY = "any"
    ALL = "all"


class JobStatus(str, Enum):
    """Background job status."""

    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Request Models
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


class YouTubeIngestRequest(BaseModel):
    """Request body for YouTube ingestion."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    url: str = Field(..., description="YouTube URL (video or playlist)")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")


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
    max_pages: Optional[int] = Field(None, description="Max pages to extract (None=all)")
    min_chars_per_page: int = Field(10, description="Skip pages with fewer chars")
    join_pages_with: str = Field("\n\n", description="Separator between pages")
    enable_ocr: bool = Field(False, description="Enable OCR (reserved for future)")


class PDFIngestRequest(BaseModel):
    """Request body for PDF ingestion (JSON metadata, file via multipart)."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")
    title: Optional[str] = Field(None, description="Document title (auto-detected if not set)")
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


class QueryFilters(BaseModel):
    """Filters for query endpoint."""

    source_types: Optional[list[SourceType]] = Field(
        default=None, description="Filter by source types"
    )
    symbols: Optional[list[str]] = Field(default=None, description="Filter by symbols")
    symbols_mode: SymbolsMode = Field(
        default=SymbolsMode.ANY, description="Symbol matching mode"
    )
    topics: Optional[list[str]] = Field(default=None, description="Filter by topics")
    entities: Optional[list[str]] = Field(
        default=None, description="Filter by entities"
    )
    authors: Optional[list[str]] = Field(
        default=None, description="Filter by authors/channels"
    )
    published_from: Optional[datetime] = Field(
        default=None, description="Published after this date"
    )
    published_to: Optional[datetime] = Field(
        default=None, description="Published before this date"
    )


class QueryRequest(BaseModel):
    """Request body for query endpoint."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    question: str = Field(..., min_length=1, description="Query question")
    mode: QueryMode = Field(default=QueryMode.RETRIEVE, description="Query mode")
    filters: Optional[QueryFilters] = Field(default=None, description="Query filters")
    retrieve_k: int = Field(default=20, ge=1, le=100, description="Candidates to retrieve")
    top_k: int = Field(default=5, ge=1, le=50, description="Results to return")
    rerank: bool = Field(default=False, description="Enable reranking")
    answer_model: Optional[str] = Field(None, description="Override answer model")
    max_context_tokens: Optional[int] = Field(
        None, description="Override max context tokens"
    )


class ReembedRequest(BaseModel):
    """Request body for re-embedding."""

    workspace_id: UUID = Field(..., description="Workspace identifier")
    target_collection: str = Field(..., description="Target collection name")
    embed_provider: str = Field(default="ollama", description="Embedding provider")
    embed_model: str = Field(..., description="Embedding model name")
    doc_ids: Optional[list[UUID]] = Field(
        default=None, description="Specific documents to re-embed"
    )


# Response Models
class IngestResponse(BaseModel):
    """Response for document ingestion."""

    doc_id: UUID = Field(..., description="Created document ID")
    chunks_created: int = Field(..., description="Number of chunks created")
    vectors_created: int = Field(..., description="Number of vectors created")
    status: str = Field(..., description="Ingestion status")


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


class ChunkResult(BaseModel):
    """A single chunk result from query."""

    chunk_id: UUID = Field(..., description="Chunk identifier")
    doc_id: UUID = Field(..., description="Document identifier")
    content: str = Field(..., description="Chunk content")
    score: float = Field(..., description="Relevance score")
    source_url: Optional[str] = Field(None, description="Source URL")
    citation_url: Optional[str] = Field(None, description="Citation URL with locator")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Author/channel")
    published_at: Optional[datetime] = Field(None, description="Publication date")
    locator_label: Optional[str] = Field(None, description="Locator label")
    symbols: list[str] = Field(default_factory=list, description="Detected symbols")
    topics: list[str] = Field(default_factory=list, description="Detected topics")


class QueryResponse(BaseModel):
    """Response for query endpoint."""

    results: list[ChunkResult] = Field(..., description="Matching chunks")
    answer: Optional[str] = Field(None, description="Generated answer if mode=answer")


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


class DependencyHealth(BaseModel):
    """Health status for a dependency."""

    status: str = Field(..., description="Dependency status (ok/error)")
    latency_ms: Optional[float] = Field(None, description="Response latency in ms")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    """Response for health endpoint."""

    status: str = Field(..., description="Overall service status")
    qdrant: DependencyHealth = Field(..., description="Qdrant health")
    supabase: DependencyHealth = Field(..., description="Supabase health")
    ollama: DependencyHealth = Field(..., description="Ollama health")
    active_collection: str = Field(..., description="Active Qdrant collection")
    embed_model: str = Field(..., description="Active embedding model")
    latency_ms: dict[str, float] = Field(..., description="Latency per dependency")
    version: str = Field(..., description="Service version")


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    retryable: bool = Field(default=False, description="Whether error is retryable")
