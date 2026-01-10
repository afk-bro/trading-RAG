"""Pydantic models for request/response validation."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

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


class RerankState(str, Enum):
    """Rerank execution state for observability.

    State machine:
    - DISABLED: rerank.enabled=false, no cross-encoder/LLM cost
    - OK: rerank completed successfully within timeout
    - TIMEOUT_FALLBACK: rerank timed out, fell back to vector order
    - ERROR_FALLBACK: rerank failed (exception), fell back to vector order
    """

    DISABLED = "disabled"
    OK = "ok"
    TIMEOUT_FALLBACK = "timeout_fallback"
    ERROR_FALLBACK = "error_fallback"


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
        description="If true, supersede existing document at same canonical_url and create new version",  # noqa: E501
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
    retrieve_k: Optional[int] = Field(
        default=None, ge=1, le=200, description="Override candidates to retrieve"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=50, description="Override results to return"
    )
    rerank: Optional[bool] = Field(default=None, description="Override rerank enabled")
    rerank_method: Optional[str] = Field(
        default=None, description="Override rerank method (cross_encoder/llm)"
    )
    debug: bool = Field(default=False, description="Include debug info in response")
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


class ChunkResultDebug(BaseModel):
    """Debug information for a chunk result (populated when rerank is enabled)."""

    vector_score: float = Field(..., description="Original vector similarity score")
    rerank_score: Optional[float] = Field(
        None, description="Rerank score (None if no rerank)"
    )
    rerank_rank: Optional[int] = Field(
        None, description="Rank after reranking (0-based)"
    )
    is_neighbor: bool = Field(
        default=False, description="Whether this is a neighbor chunk"
    )
    neighbor_of: Optional[str] = Field(
        None, description="Chunk ID this is a neighbor of"
    )


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
    debug: Optional[ChunkResultDebug] = Field(
        None, description="Debug info (when rerank enabled)"
    )


class KnowledgeExtractionStats(BaseModel):
    """Statistics from knowledge extraction pipeline (mode=learn)."""

    entities_extracted: int = Field(default=0, description="Entities extracted")
    claims_extracted: int = Field(default=0, description="Claims extracted")
    relations_extracted: int = Field(default=0, description="Relations extracted")
    claims_verified: int = Field(
        default=0, description="Claims verified (high confidence)"
    )
    claims_weak: int = Field(default=0, description="Claims with weak support")
    claims_rejected: int = Field(default=0, description="Claims rejected")
    entities_persisted: int = Field(
        default=0, description="Entities persisted to truth store"
    )
    claims_persisted: int = Field(
        default=0, description="Claims persisted to truth store"
    )
    claims_skipped_duplicate: int = Field(
        default=0, description="Claims skipped (already in truth store)"
    )
    claims_skipped_invalid: int = Field(
        default=0, description="Claims skipped (invalid evidence)"
    )


class QueryMeta(BaseModel):
    """Metadata about query execution for observability."""

    # Timing (milliseconds)
    embed_ms: int = Field(..., description="Query embedding time")
    search_ms: int = Field(..., description="Vector search time")
    rerank_ms: Optional[int] = Field(None, description="Reranking time (if enabled)")
    expand_ms: Optional[int] = Field(
        None, description="Neighbor expansion time (if enabled)"
    )
    answer_ms: Optional[int] = Field(
        None, description="Answer generation time (if mode=answer)"
    )
    total_ms: int = Field(..., description="Total query time")

    # Counts
    candidates_searched: int = Field(..., description="Candidates from vector search")
    seeds_count: int = Field(..., description="Seeds after rerank (or vector fallback)")
    chunks_after_expand: int = Field(..., description="Chunks after neighbor expansion")
    neighbors_added: int = Field(default=0, description="Neighbor chunks added")

    # Rerank state (primary health metric for dashboards)
    rerank_state: RerankState = Field(
        default=RerankState.DISABLED,
        description="Rerank execution state: disabled, ok, timeout_fallback, error_fallback",
    )
    rerank_enabled: bool = Field(
        default=False, description="Whether reranking was enabled"
    )
    rerank_method: Optional[str] = Field(
        None, description="Rerank method (cross_encoder/llm)"
    )
    rerank_model: Optional[str] = Field(None, description="Rerank model used")
    rerank_timeout: bool = Field(default=False, description="True if rerank timed out")
    rerank_fallback: bool = Field(
        default=False, description="True if fell back to vector order"
    )

    # Neighbor info
    neighbor_enabled: bool = Field(
        default=True, description="Whether neighbor expansion was enabled"
    )


class QueryResponse(BaseModel):
    """Response for query endpoint."""

    results: list[ChunkResult] = Field(..., description="Matching chunks")
    answer: Optional[str] = Field(
        None, description="Generated answer if mode=answer/learn/kb_answer"
    )
    knowledge_stats: Optional[KnowledgeExtractionStats] = Field(
        None, description="Knowledge extraction stats if mode=learn"
    )
    meta: Optional[QueryMeta] = Field(None, description="Query execution metadata")

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


class ReadinessResponse(BaseModel):
    """Response for readiness endpoint (Kubernetes /ready probe)."""

    ready: bool = Field(..., description="True if service is ready to accept traffic")
    checks: dict[str, DependencyHealth] = Field(
        ..., description="Individual check results"
    )
    version: str = Field(..., description="Service version")
    git_sha: Optional[str] = Field(None, description="Git commit SHA")
    build_time: Optional[str] = Field(None, description="Build timestamp ISO8601")
    config_profile: str = Field(
        ..., description="Configuration profile (dev/staging/prod)"
    )


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
    verified_claim_count: Optional[int] = Field(
        None, description="Number of verified claims"
    )
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
    evidence: list[KBEvidenceItem] = Field(
        default_factory=list, description="Supporting evidence"
    )
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
    supported: list[str] = Field(
        default_factory=list, description="Supported statements with claim refs"
    )
    not_specified: list[str] = Field(
        default_factory=list, description="Unanswered aspects"
    )
    claims_used: list[KBAnswerClaimRef] = Field(
        default_factory=list, description="Claims used in answer"
    )
    fallback_used: bool = Field(
        default=False, description="Whether chunk RAG fallback was used"
    )
    fallback_reason: Optional[str] = Field(
        None, description="Reason for fallback if used"
    )


# ============================================================================
# Strategy Spec Schemas
# ============================================================================


class StrategySpecStatus(str, Enum):
    """Status of a strategy specification."""

    DRAFT = "draft"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


class StrategySpecResponse(BaseModel):
    """Response for GET /kb/strategies/{entity_id}/spec."""

    id: UUID = Field(..., description="Spec ID")
    strategy_entity_id: UUID = Field(..., description="Strategy entity ID")
    strategy_name: str = Field(..., description="Strategy name")
    spec_json: dict = Field(..., description="The compiled specification")
    status: StrategySpecStatus = Field(..., description="Approval status")
    version: int = Field(..., description="Spec version number")
    derived_from_claim_ids: list[str] = Field(
        default_factory=list, description="Source claim IDs"
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")
    approved_by: Optional[str] = Field(None, description="Approver identifier")


class StrategySpecRefreshRequest(BaseModel):
    """Request for POST /kb/strategies/{entity_id}/spec/refresh."""

    pass  # No body needed, entity_id is in path


class StrategyCompileResponse(BaseModel):
    """Response for POST /kb/strategies/{entity_id}/compile."""

    spec_id: str = Field(..., description="Source spec ID")
    spec_version: int = Field(..., description="Spec version used")
    spec_status: StrategySpecStatus = Field(..., description="Spec approval status")
    param_schema: dict = Field(..., description="JSON Schema for parameter UI form")
    backtest_config: dict = Field(
        ..., description="Engine-agnostic backtest configuration"
    )
    pseudocode: str = Field(..., description="Human-readable strategy description")
    citations: list[str] = Field(..., description="Claim IDs used to derive the spec")


class StrategySpecStatusUpdate(BaseModel):
    """Request for PATCH /kb/strategies/{entity_id}/spec."""

    status: StrategySpecStatus = Field(..., description="New status")
    approved_by: Optional[str] = Field(
        None, description="Approver identifier (for approved status)"
    )


# ===========================================
# Workspace Configuration Schemas
# ===========================================


class CrossEncoderConfig(BaseModel):
    """Configuration for cross-encoder reranking."""

    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    max_text_chars: int = Field(default=2000, ge=100, le=10000)
    batch_size: int = Field(default=16, ge=1, le=64)
    max_concurrent: int = Field(default=2, ge=1, le=4)


class LLMRerankConfig(BaseModel):
    """Configuration for LLM-based reranking."""

    model: Optional[str] = None


class RerankConfig(BaseModel):
    """Configuration for reranking pipeline stage."""

    enabled: bool = False
    method: str = "cross_encoder"  # "cross_encoder" | "llm"
    candidates_k: int = Field(default=50, ge=10, le=200)
    final_k: int = Field(default=10, ge=1, le=50)
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    llm: LLMRerankConfig = Field(default_factory=LLMRerankConfig)


class NeighborConfig(BaseModel):
    """Configuration for neighbor expansion."""

    enabled: bool = True
    window: int = Field(default=1, ge=0, le=3)
    pdf_window: int = Field(default=2, ge=0, le=5)
    min_chars: int = Field(default=200, ge=0)
    max_total: int = Field(default=20, ge=1, le=50)


class RetrievalConfig(BaseModel):
    """Configuration for base retrieval parameters."""

    top_k: int = Field(default=8, ge=1, le=100)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    size: int = Field(default=512, ge=64, le=2048)
    overlap: int = Field(default=50, ge=0, le=256)


class WorkspaceConfig(BaseModel):
    """
    Complete workspace configuration schema.

    Stored as JSON in workspace.config column.
    Uses extra="allow" to support additional custom fields.
    """

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    neighbor: NeighborConfig = Field(default_factory=NeighborConfig)

    model_config = {"extra": "allow"}

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "WorkspaceConfig":
        """
        Create WorkspaceConfig from a dict, with defaults for missing keys.

        Args:
            data: Raw config dict from database (may be None or partial)

        Returns:
            Fully populated WorkspaceConfig
        """
        if data is None:
            return cls()
        return cls.model_validate(data)


# ===========================================
# Query Compare Endpoint Models
# ===========================================


class QueryCompareRequest(BaseModel):
    """Request for A/B comparison between vector-only and reranked retrieval."""

    workspace_id: UUID = Field(..., description="Workspace to search within")
    question: str = Field(
        ..., min_length=1, max_length=2000, description="Query question"
    )

    # Filters (same as QueryRequest)
    filters: Optional[QueryFilters] = Field(None, description="Optional search filters")

    # Retrieval parameters
    retrieve_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=200,
        description="Candidates to retrieve (shared by both runs)",
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=50, description="Results to return from each run"
    )

    # Compare options
    debug: bool = Field(default=True, description="Include debug info in results")
    skip_neighbors: bool = Field(
        default=True,
        description="Skip neighbor expansion for faster comparison (recommended)",
    )


class CompareMetrics(BaseModel):
    """Metrics comparing vector-only vs reranked results."""

    # Set overlap
    jaccard: float = Field(
        ..., ge=0.0, le=1.0, description="Jaccard similarity: |A ∩ B| / |A ∪ B|"
    )
    overlap_count: int = Field(..., ge=0, description="Number of shared chunk IDs")
    union_count: int = Field(..., ge=0, description="Total unique chunk IDs")

    # Rank correlation (only when overlap >= 2)
    spearman: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Spearman rank correlation over intersection (None if overlap < 2)",
    )

    # Rank delta stats (for intersection)
    rank_delta_mean: Optional[float] = Field(
        None, description="Mean absolute rank delta for overlapping IDs"
    )
    rank_delta_max: Optional[int] = Field(
        None, description="Max absolute rank delta for overlapping IDs"
    )

    # Raw lists for UI rendering
    vector_only_ids: list[str] = Field(
        ..., description="Chunk IDs from vector-only run"
    )
    reranked_ids: list[str] = Field(..., description="Chunk IDs from reranked run")
    intersection_ids: list[str] = Field(..., description="Chunk IDs in both runs")


class QueryCompareResponse(BaseModel):
    """Response for A/B comparison endpoint."""

    vector_only: QueryResponse = Field(..., description="Results without reranking")
    reranked: QueryResponse = Field(..., description="Results with reranking")
    metrics: CompareMetrics = Field(..., description="Comparison metrics")


# ===========================================
# Trade Intent & Policy Engine Schemas
# ===========================================


class IntentAction(str, Enum):
    """What the brain wants to do."""

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    CANCEL_ORDER = "cancel_order"


class TradeIntent(BaseModel):
    """
    A declaration of what the trading brain wants to do.

    Provider-agnostic: this is what the strategy wants, not how
    to execute it. The Policy Engine decides if it's allowed.
    """

    # Identity
    id: UUID = Field(default_factory=uuid4, description="Intent UUID")
    correlation_id: str = Field(..., description="Correlation ID for tracing")
    workspace_id: UUID = Field(..., description="Workspace this intent belongs to")

    # What
    action: IntentAction = Field(..., description="Requested action")
    strategy_entity_id: UUID = Field(..., description="Strategy making the request")
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 4h)")

    # Parameters (optional, depends on action)
    quantity: Optional[float] = Field(None, ge=0, description="Position size")
    price: Optional[float] = Field(None, description="Limit price (None for market)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")

    # Context
    signal_strength: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Signal confidence [0,1]"
    )
    regime_snapshot: Optional[dict] = Field(None, description="Current regime state")
    reason: Optional[str] = Field(None, description="Human-readable reason for intent")

    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When intent was created"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class PolicyReason(str, Enum):
    """Why a policy decision was made."""

    # Rejections
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    REGIME_DRIFT = "regime_drift"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    COOLDOWN_ACTIVE = "cooldown_active"
    INVALID_SYMBOL = "invalid_symbol"
    INVALID_TIMEFRAME = "invalid_timeframe"
    STRATEGY_DISABLED = "strategy_disabled"
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"

    # Approvals
    ALL_RULES_PASSED = "all_rules_passed"
    MANUAL_OVERRIDE = "manual_override"


class PolicyDecision(BaseModel):
    """
    The Policy Engine's verdict on a TradeIntent.

    This is the gatekeeper's output: approved, rejected, or held.
    """

    # Decision
    approved: bool = Field(..., description="Whether intent is approved for execution")
    reason: PolicyReason = Field(..., description="Primary reason for decision")
    reason_details: Optional[str] = Field(None, description="Additional details")

    # Audit trail
    rules_evaluated: list[str] = Field(
        default_factory=list, description="Rules that were evaluated"
    )
    rules_passed: list[str] = Field(
        default_factory=list, description="Rules that passed"
    )
    rules_failed: list[str] = Field(
        default_factory=list, description="Rules that failed"
    )

    # Modifications (for partial approvals)
    modified_quantity: Optional[float] = Field(
        None, description="Adjusted quantity if capped"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-blocking warnings"
    )

    # Context
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    evaluation_ms: Optional[int] = Field(None, description="Evaluation time in ms")


class PositionState(BaseModel):
    """Current position for a symbol."""

    symbol: str = Field(..., description="Trading symbol")
    side: Optional[str] = Field(None, description="'long', 'short', or None if flat")
    quantity: float = Field(default=0.0, description="Position size")
    entry_price: Optional[float] = Field(None, description="Average entry price")
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized P&L")
    realized_pnl_today: Optional[float] = Field(None, description="Realized P&L today")


class CurrentState(BaseModel):
    """
    Minimal current state snapshot for policy evaluation.

    This is what the Policy Engine needs to make decisions.
    Kept minimal to avoid stale state issues.
    """

    # System state
    kill_switch_active: bool = Field(default=False, description="Global kill switch")
    trading_enabled: bool = Field(default=True, description="Trading allowed")

    # Positions (optional - may not be available)
    positions: list[PositionState] = Field(
        default_factory=list, description="Current positions"
    )

    # Account metrics (optional)
    account_equity: Optional[float] = Field(None, description="Current account equity")
    daily_pnl: Optional[float] = Field(None, description="Today's realized P&L")
    max_drawdown_today: Optional[float] = Field(
        None, description="Today's max drawdown %"
    )

    # Regime (from v1.5)
    current_regime: Optional[dict] = Field(None, description="Current regime snapshot")
    regime_distance_z: Optional[float] = Field(
        None, description="Z-score from training regime"
    )

    # Timestamps
    snapshot_at: datetime = Field(default_factory=datetime.utcnow)


# ===========================================
# Trade Event Journal Schemas
# ===========================================


class TradeEventType(str, Enum):
    """Types of events recorded in the trade journal."""

    # Intent lifecycle
    INTENT_EMITTED = "intent_emitted"
    INTENT_VALIDATED = "intent_validated"
    INTENT_INVALID = "intent_invalid"

    # Policy evaluation
    POLICY_EVALUATED = "policy_evaluated"
    INTENT_APPROVED = "intent_approved"
    INTENT_REJECTED = "intent_rejected"

    # Execution
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIAL_FILL = "order_partial_fill"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # Position changes
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_SCALED = "position_scaled"

    # System events
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    KILL_SWITCH_DEACTIVATED = "kill_switch_deactivated"
    REGIME_DRIFT_DETECTED = "regime_drift_detected"

    # Run plan events (Test Generator / Orchestrator)
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


class TradeEvent(BaseModel):
    """
    Immutable event record for the trade journal.

    Append-only audit trail of all trading decisions.
    """

    id: UUID = Field(default_factory=uuid4)
    correlation_id: str = Field(..., description="Links related events together")
    workspace_id: UUID = Field(..., description="Workspace this event belongs to")

    # Event type and timing
    event_type: TradeEventType = Field(..., description="Type of event")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Context
    strategy_entity_id: Optional[UUID] = Field(
        None, description="Strategy that triggered event"
    )
    symbol: Optional[str] = Field(None, description="Trading symbol if applicable")
    timeframe: Optional[str] = Field(None, description="Timeframe if applicable")

    # References
    intent_id: Optional[UUID] = Field(None, description="Related intent ID")
    order_id: Optional[str] = Field(None, description="External order ID")
    position_id: Optional[str] = Field(None, description="External position ID")

    # Payload (event-specific data)
    payload: dict = Field(default_factory=dict, description="Event-specific data")

    # Metadata
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class TradeEventListResponse(BaseModel):
    """Response for GET /admin/trade/events."""

    items: list[TradeEvent] = Field(..., description="Event list")
    total: int = Field(..., description="Total matching events")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class IntentEvaluateRequest(BaseModel):
    """Request for POST /intents/evaluate."""

    intent: TradeIntent = Field(..., description="Intent to evaluate")
    state: Optional[CurrentState] = Field(
        None, description="Current state (uses defaults if not provided)"
    )
    dry_run: bool = Field(default=False, description="If true, don't journal the event")


class IntentEvaluateResponse(BaseModel):
    """Response for POST /intents/evaluate."""

    intent_id: UUID = Field(..., description="Intent that was evaluated")
    decision: PolicyDecision = Field(..., description="Policy engine decision")
    events_recorded: int = Field(default=0, description="Number of events journaled")
    correlation_id: str = Field(..., description="Correlation ID for tracing")


# ===========================================
# Paper Execution Schemas
# ===========================================


class OrderSide(str, Enum):
    """Order side for execution."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExecutionMode(str, Enum):
    """Execution mode."""

    PAPER = "paper"
    LIVE = "live"  # Future


class PaperOrder(BaseModel):
    """Simulated order for paper trading."""

    id: UUID = Field(default_factory=uuid4)
    intent_id: UUID = Field(..., description="Intent that triggered this order")
    correlation_id: str = Field(..., description="Correlation ID for tracing")
    workspace_id: UUID = Field(..., description="Workspace this order belongs to")

    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    quantity: float = Field(..., gt=0, description="Order quantity")
    fill_price: float = Field(..., gt=0, description="Execution price")

    status: OrderStatus = Field(default=OrderStatus.FILLED, description="Order status")
    fees: float = Field(default=0.0, ge=0, description="Execution fees")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = Field(
        default=None, description="When order was filled"
    )

    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class PaperPosition(BaseModel):
    """Paper trading position state."""

    workspace_id: UUID = Field(..., description="Workspace this position belongs to")
    symbol: str = Field(..., description="Trading symbol")
    side: Optional[str] = Field(
        None, description="Position side ('long' or None if flat)"
    )
    quantity: float = Field(default=0.0, ge=0, description="Position size")
    avg_price: float = Field(default=0.0, ge=0, description="Average entry price")

    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    realized_pnl: float = Field(
        default=0.0, description="Realized P&L from this position"
    )

    opened_at: Optional[datetime] = Field(None, description="When position was opened")
    last_updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Tracking
    order_ids: list[str] = Field(
        default_factory=list, description="Orders that built this position"
    )
    intent_ids: list[str] = Field(
        default_factory=list, description="Intents that triggered orders"
    )


class PaperState(BaseModel):
    """Complete paper trading state for a workspace."""

    workspace_id: UUID = Field(..., description="Workspace this state belongs to")

    # Cash ledger
    starting_equity: float = Field(default=10000.0, description="Starting equity")
    cash: float = Field(default=10000.0, description="Current cash balance")
    realized_pnl: float = Field(default=0.0, description="Total realized P&L")

    # Positions by symbol
    positions: dict[str, PaperPosition] = Field(
        default_factory=dict, description="Positions keyed by symbol"
    )

    # Tracking
    orders_count: int = Field(default=0, description="Total orders executed")
    trades_count: int = Field(default=0, description="Total trades (round trips)")

    # Reconciliation
    last_event_id: Optional[UUID] = Field(None, description="Last processed event ID")
    last_event_at: Optional[datetime] = Field(None, description="Last event timestamp")
    reconciled_at: Optional[datetime] = Field(
        None, description="When state was reconciled"
    )


class ExecutionRequest(BaseModel):
    """Request for POST /execute/intents."""

    intent: TradeIntent = Field(..., description="Intent to execute")
    fill_price: float = Field(..., gt=0, description="Fill price (required)")
    mode: ExecutionMode = Field(
        default=ExecutionMode.PAPER, description="Execution mode"
    )


class ExecutionResult(BaseModel):
    """Result of intent execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    intent_id: UUID = Field(..., description="Intent that was executed")

    order_id: Optional[UUID] = Field(None, description="Order ID if created")
    fill_price: Optional[float] = Field(None, description="Actual fill price")
    quantity_filled: float = Field(default=0.0, description="Quantity filled")
    fees: float = Field(default=0.0, description="Fees charged")

    position_action: Optional[str] = Field(
        None, description="Position action: opened, closed, scaled"
    )
    position: Optional[PaperPosition] = Field(None, description="Updated position")

    events_recorded: int = Field(default=0, description="Events journaled")
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for tracing"
    )

    # Error info
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Error code if failed")


class ReconciliationResult(BaseModel):
    """Result of journal reconciliation."""

    success: bool = Field(..., description="Whether reconciliation succeeded")
    workspace_id: UUID = Field(..., description="Workspace reconciled")

    events_replayed: int = Field(default=0, description="Events processed")
    orders_rebuilt: int = Field(default=0, description="Orders reconstructed")
    positions_rebuilt: int = Field(default=0, description="Positions with qty > 0")

    cash_after: float = Field(default=0.0, description="Cash after reconciliation")
    realized_pnl_after: float = Field(default=0.0, description="Realized P&L after")

    last_event_at: Optional[datetime] = Field(None, description="Last event timestamp")
    errors: list[str] = Field(default_factory=list, description="Errors encountered")
