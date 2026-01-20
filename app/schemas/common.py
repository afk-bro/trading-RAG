"""Common schemas: enums, error responses, health checks."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ===========================================
# Enums
# ===========================================


class SourceType(str, Enum):
    """Supported source types for documents."""

    YOUTUBE = "youtube"
    PDF = "pdf"
    ARTICLE = "article"
    NOTE = "note"
    TRANSCRIPT = "transcript"
    PINE_SCRIPT = "pine_script"


class DocumentStatus(str, Enum):
    """Document lifecycle status."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"
    FAILED = "failed"  # Health validation failed (no chunks or missing embeddings)


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


class PineIngestStatus(str, Enum):
    """Status of pine script ingest operation."""

    SUCCESS = "success"  # All processed without failures
    PARTIAL = "partial"  # Some succeeded, some failed
    FAILED = "failed"  # All failed or critical error
    DRY_RUN = "dry_run"  # Validation only, no changes


# ===========================================
# Health & Error Responses
# ===========================================


class DependencyHealth(BaseModel):
    """Health status for a dependency."""

    status: str = Field(..., description="Dependency status (ok/error)")
    latency_ms: Optional[float] = Field(None, description="Response latency in ms")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class CircuitBreakerStatus(BaseModel):
    """Status of a circuit breaker."""

    failures: int = Field(..., description="Consecutive failure count")
    is_open: bool = Field(
        ..., description="True if circuit is open (blocking requests)"
    )
    last_failure: Optional[str] = Field(
        None, description="ISO timestamp of last failure"
    )


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
    circuit_breakers: Optional[dict[str, CircuitBreakerStatus]] = Field(
        None, description="Circuit breaker status for recovery tracking"
    )


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
