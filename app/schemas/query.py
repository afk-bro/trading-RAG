"""Query-related schemas: requests, responses, compare."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import QueryMode, RerankState, SourceType, SymbolsMode
from app.schemas.kb import KBAnswerResponse


# ===========================================
# Query Filters
# ===========================================


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


# ===========================================
# Query Request
# ===========================================


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


# ===========================================
# Query Response
# ===========================================


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
