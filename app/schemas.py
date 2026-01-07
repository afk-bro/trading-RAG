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
    LEARN = "learn"  # Extract → verify → persist → synthesize
    KB_ANSWER = "kb_answer"  # Answer from verified claims (truth store)


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
    update_existing: bool = Field(
        default=False,
        description="If true, supersede existing document at same canonical_url and create new version"
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
    version: int = Field(default=1, description="Document version number")
    superseded_doc_id: Optional[UUID] = Field(
        default=None, description="ID of superseded document if this was an update"
    )


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


class KnowledgeExtractionStats(BaseModel):
    """Statistics from knowledge extraction pipeline (mode=learn)."""

    entities_extracted: int = Field(default=0, description="Entities extracted")
    claims_extracted: int = Field(default=0, description="Claims extracted")
    relations_extracted: int = Field(default=0, description="Relations extracted")
    claims_verified: int = Field(default=0, description="Claims verified (high confidence)")
    claims_weak: int = Field(default=0, description="Claims with weak support")
    claims_rejected: int = Field(default=0, description="Claims rejected")
    entities_persisted: int = Field(default=0, description="Entities persisted to truth store")
    claims_persisted: int = Field(default=0, description="Claims persisted to truth store")
    claims_skipped_duplicate: int = Field(default=0, description="Claims skipped (already in truth store)")
    claims_skipped_invalid: int = Field(default=0, description="Claims skipped (invalid evidence)")


class QueryResponse(BaseModel):
    """Response for query endpoint."""

    results: list[ChunkResult] = Field(..., description="Matching chunks")
    answer: Optional[str] = Field(None, description="Generated answer if mode=answer/learn/kb_answer")
    knowledge_stats: Optional[KnowledgeExtractionStats] = Field(
        None, description="Knowledge extraction stats if mode=learn"
    )

    # KB Answer specific fields (only populated when mode=kb_answer)
    kb_answer: Optional["KBAnswerResponse"] = Field(
        None, description="Structured KB answer if mode=kb_answer"
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


# ===========================================
# Knowledge Base (KB) Endpoint Models
# ===========================================

class KBEntityType(str, Enum):
    """Entity types for KB filtering."""

    CONCEPT = "concept"
    INDICATOR = "indicator"
    STRATEGY = "strategy"
    EQUATION = "equation"
    TEST = "test"
    METRIC = "metric"
    ASSET = "asset"
    PATTERN = "pattern"
    PARAMETER = "parameter"
    OTHER = "other"


class KBClaimType(str, Enum):
    """Claim types for KB filtering."""

    DEFINITION = "definition"
    RULE = "rule"
    ASSUMPTION = "assumption"
    WARNING = "warning"
    PARAMETER = "parameter"
    EQUATION = "equation"
    OBSERVATION = "observation"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


class KBClaimStatus(str, Enum):
    """Claim verification status for KB filtering."""

    PENDING = "pending"
    VERIFIED = "verified"
    WEAK = "weak"
    REJECTED = "rejected"


class KBEntityItem(BaseModel):
    """Single entity in KB list response."""

    id: UUID = Field(..., description="Entity ID")
    type: KBEntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(None, description="Entity description")
    verified_claim_count: Optional[int] = Field(None, description="Number of verified claims")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class KBEntityListResponse(BaseModel):
    """Response for GET /kb/entities."""

    items: list[KBEntityItem] = Field(..., description="Entity list")
    total: int = Field(..., description="Total matching entities")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class KBEntityStats(BaseModel):
    """Statistics for a single entity."""

    verified_claims: int = Field(default=0, description="Verified claim count")
    weak_claims: int = Field(default=0, description="Weak claim count")
    total_claims: int = Field(default=0, description="Total claim count")
    relations_count: int = Field(default=0, description="Number of relations")


class KBEntityDetailResponse(BaseModel):
    """Response for GET /kb/entities/{entity_id}."""

    id: UUID = Field(..., description="Entity ID")
    type: KBEntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(None, description="Entity description")
    stats: KBEntityStats = Field(..., description="Entity statistics")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class KBEvidenceItem(BaseModel):
    """Evidence for a claim."""

    id: UUID = Field(..., description="Evidence ID")
    doc_id: UUID = Field(..., description="Source document ID")
    chunk_id: UUID = Field(..., description="Source chunk ID")
    quote: str = Field(..., description="Evidence quote")
    relevance_score: float = Field(..., description="Relevance score")
    doc_title: Optional[str] = Field(None, description="Document title")


class KBClaimItem(BaseModel):
    """Single claim in KB list response."""

    id: UUID = Field(..., description="Claim ID")
    claim_type: KBClaimType = Field(..., description="Claim type")
    text: str = Field(..., description="Claim text")
    status: KBClaimStatus = Field(..., description="Verification status")
    confidence: float = Field(..., description="Confidence score")
    entity_id: Optional[UUID] = Field(None, description="Linked entity ID")
    entity_name: Optional[str] = Field(None, description="Linked entity name")
    entity_type: Optional[KBEntityType] = Field(None, description="Linked entity type")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class KBClaimListResponse(BaseModel):
    """Response for GET /kb/claims."""

    items: list[KBClaimItem] = Field(..., description="Claim list")
    total: int = Field(..., description="Total matching claims")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class KBClaimDetailResponse(BaseModel):
    """Response for GET /kb/claims/{claim_id}."""

    id: UUID = Field(..., description="Claim ID")
    claim_type: KBClaimType = Field(..., description="Claim type")
    text: str = Field(..., description="Claim text")
    status: KBClaimStatus = Field(..., description="Verification status")
    confidence: float = Field(..., description="Confidence score")
    entity_id: Optional[UUID] = Field(None, description="Linked entity ID")
    entity_name: Optional[str] = Field(None, description="Linked entity name")
    entity_type: Optional[KBEntityType] = Field(None, description="Linked entity type")
    evidence: list[KBEvidenceItem] = Field(default_factory=list, description="Supporting evidence")
    extraction_model: Optional[str] = Field(None, description="Extraction model")
    verification_model: Optional[str] = Field(None, description="Verification model")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class KBAnswerClaimRef(BaseModel):
    """Claim reference in kb_answer response."""

    id: str = Field(..., description="Short reference ID (e.g., C12)")
    claim_id: UUID = Field(..., description="Full claim UUID")
    confidence: float = Field(..., description="Claim confidence")


class KBAnswerResponse(BaseModel):
    """Response for mode=kb_answer queries."""

    mode: str = Field(default="kb_answer", description="Query mode")
    llm_enabled: bool = Field(..., description="Whether LLM was used")
    answer: Optional[str] = Field(None, description="Synthesized answer")
    supported: list[str] = Field(default_factory=list, description="Supported statements with claim refs")
    not_specified: list[str] = Field(default_factory=list, description="Unanswered aspects")
    claims_used: list[KBAnswerClaimRef] = Field(default_factory=list, description="Claims used in answer")
    fallback_used: bool = Field(default=False, description="Whether chunk RAG fallback was used")
    fallback_reason: Optional[str] = Field(None, description="Reason for fallback if used")
