"""Core query pipeline logic, extracted for reuse by /query and /query/compare."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import structlog

from app.config import Settings
from app.schemas import (
    ChunkResult,
    ChunkResultDebug,
    QueryMeta,
    QueryRequest,
    RerankState,
)
from app.services.neighbor_expansion import ExpandedChunk, expand_neighbors
from app.services.reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    get_reranker,
)

logger = structlog.get_logger(__name__)

# Safety caps (prevent accidental latency bombs)
MAX_CANDIDATES_K = 200
MAX_FINAL_K = 50
MAX_NEIGHBOR_TOTAL = 50


@dataclass
class ResolvedConfig:
    """Resolved configuration after applying precedence and caps."""

    rerank_enabled: bool
    rerank_method: str
    candidates_k: int
    final_k: int
    rerank_config: dict
    neighbor_config: dict


@dataclass
class PipelineContext:
    """Shared context for pipeline execution."""

    workspace_id: UUID
    question: str
    debug: bool = False

    # Precomputed data (for sharing between compare runs)
    query_embedding: Optional[list[float]] = None
    search_results: Optional[list[dict]] = None
    chunks_map: Optional[dict] = None
    candidates: Optional[list[RerankCandidate]] = None

    # Timing from shared steps
    embed_ms: int = 0
    search_ms: int = 0


@dataclass
class PipelineResult:
    """Result from a single pipeline run."""

    results: list[ChunkResult]
    meta: QueryMeta
    chunk_ids: list[str] = field(default_factory=list)  # For metrics computation


def resolve_config(
    request: QueryRequest,
    *,
    force_rerank: Optional[bool] = None,
) -> ResolvedConfig:
    """Resolve effective config from request, workspace defaults, and caps.

    Args:
        request: The query request
        force_rerank: Override rerank_enabled (for compare endpoint)

    Returns:
        ResolvedConfig with all values resolved and capped
    """
    # Default configs (would come from workspace.config JSONB)
    workspace_rerank = {
        "enabled": False,
        "method": "cross_encoder",
        "candidates_k": 50,
        "final_k": 10,
        "cross_encoder": {"device": "cuda", "model": "BAAI/bge-reranker-v2-m3"},
    }
    workspace_neighbor = {
        "enabled": True,
        "window": 1,
        "pdf_window": 2,
        "min_chars": 200,
        "max_total": 20,
    }
    workspace_retrieval = {
        "top_k": 8,
    }

    # Apply request overrides (force_rerank takes highest precedence)
    if force_rerank is not None:
        rerank_enabled = force_rerank
    else:
        rerank_enabled = (
            request.rerank if request.rerank is not None else workspace_rerank["enabled"]
        )

    rerank_method = (
        request.rerank_method if request.rerank_method else workspace_rerank["method"]
    )
    candidates_k = (
        request.retrieve_k
        if request.retrieve_k is not None
        else workspace_rerank["candidates_k"]
    )
    final_k = (
        request.top_k if request.top_k is not None else workspace_rerank["final_k"]
    )

    # Enforce hard safety caps
    if candidates_k > MAX_CANDIDATES_K:
        logger.warning(
            "Clamping candidates_k to safety cap",
            requested=candidates_k,
            capped_to=MAX_CANDIDATES_K,
        )
        candidates_k = MAX_CANDIDATES_K

    if final_k > MAX_FINAL_K:
        logger.warning(
            "Clamping final_k to safety cap",
            requested=final_k,
            capped_to=MAX_FINAL_K,
        )
        final_k = MAX_FINAL_K

    # Cap neighbor max_total
    neighbor_config = workspace_neighbor.copy()
    if neighbor_config["max_total"] > MAX_NEIGHBOR_TOTAL:
        neighbor_config["max_total"] = MAX_NEIGHBOR_TOTAL

    # Build merged rerank config
    rerank_config = {
        **workspace_rerank,
        "enabled": rerank_enabled,
        "method": rerank_method,
        "candidates_k": candidates_k,
        "final_k": final_k,
    }

    # Enforce constraints
    final_k = min(final_k, candidates_k)

    # When rerank disabled, search only final_k (no over-fetch needed)
    if not rerank_enabled:
        candidates_k = (
            request.top_k
            if request.top_k is not None
            else workspace_retrieval["top_k"]
        )
        candidates_k = min(candidates_k, MAX_FINAL_K)
        final_k = candidates_k

    return ResolvedConfig(
        rerank_enabled=rerank_enabled,
        rerank_method=rerank_method,
        candidates_k=candidates_k,
        final_k=final_k,
        rerank_config=rerank_config,
        neighbor_config=neighbor_config,
    )


def _vector_fallback(
    candidates: list[RerankCandidate],
    final_k: int,
) -> list[RerankResult]:
    """Convert candidates to RerankResults in vector score order (no reranking)."""
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.vector_score, c.chunk_id),
        reverse=True,
    )
    return [
        RerankResult(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            rerank_score=0.0,
            rerank_rank=i,
            vector_score=c.vector_score,
            source_type=c.source_type,
        )
        for i, c in enumerate(sorted_candidates[:final_k])
    ]


def _build_citation_url(
    source_url: Optional[str],
    source_type: str,
    video_id: Optional[str],
    time_start_secs: Optional[int],
    page_start: Optional[int],
) -> Optional[str]:
    """Build a citation URL with timestamp or page locator."""
    if not source_url:
        return None

    if source_type == "youtube" and video_id and time_start_secs is not None:
        base_url = f"https://www.youtube.com/watch?v={video_id}"
        return f"{base_url}&t={time_start_secs}"

    if source_type == "pdf" and page_start is not None:
        return f"{source_url}#page={page_start}"

    return source_url


async def run_retrieval_pipeline(
    ctx: PipelineContext,
    config: ResolvedConfig,
    chunk_repo,
    settings: Settings,
    *,
    skip_neighbors: bool = False,
) -> PipelineResult:
    """Run the retrieval pipeline (rerank + neighbor expansion).

    This is the core pipeline used by both /query and /query/compare.
    It assumes embedding and vector search have already been done (in ctx).

    Args:
        ctx: Pipeline context with precomputed embedding/search results
        config: Resolved configuration
        chunk_repo: Chunk repository for neighbor expansion
        settings: Application settings
        skip_neighbors: If True, skip neighbor expansion (for faster compare)

    Returns:
        PipelineResult with results, meta, and chunk_ids
    """
    total_start = time.perf_counter()

    # Use precomputed data from context
    candidates = ctx.candidates or []
    chunks_map = ctx.chunks_map or {}

    if not candidates:
        total_ms = int((time.perf_counter() - total_start) * 1000)
        return PipelineResult(
            results=[],
            meta=QueryMeta(
                embed_ms=ctx.embed_ms,
                search_ms=ctx.search_ms,
                rerank_ms=None,
                expand_ms=None,
                answer_ms=None,
                total_ms=total_ms,
                candidates_searched=len(ctx.search_results or []),
                seeds_count=0,
                chunks_after_expand=0,
                neighbors_added=0,
                rerank_state=RerankState.DISABLED,
                rerank_enabled=config.rerank_enabled,
                rerank_method=None,
                rerank_model=None,
                rerank_timeout=False,
                rerank_fallback=False,
                neighbor_enabled=config.neighbor_config.get("enabled", True),
            ),
            chunk_ids=[],
        )

    # Rerank OR vector fallback
    rerank_ms: Optional[int] = None
    rerank_method_used: Optional[str] = None
    rerank_model_used: Optional[str] = None
    rerank_state = RerankState.DISABLED
    rerank_timeout = False
    rerank_fallback = False

    reranker: Optional[BaseReranker] = (
        get_reranker(config.rerank_config) if config.rerank_enabled else None
    )

    if config.rerank_enabled and reranker:
        rerank_start = time.perf_counter()
        try:
            seeds = await asyncio.wait_for(
                reranker.rerank(ctx.question, candidates, config.final_k),
                timeout=settings.rerank_timeout_s,
            )
            rerank_method_used = reranker.method
            rerank_model_used = reranker.model_id
            rerank_ms = int((time.perf_counter() - rerank_start) * 1000)

            if not seeds:
                logger.warning("Reranker returned empty, falling back to vector order")
                seeds = _vector_fallback(candidates, config.final_k)
                rerank_state = RerankState.ERROR_FALLBACK
                rerank_fallback = True
            else:
                rerank_state = RerankState.OK
        except asyncio.TimeoutError:
            rerank_ms = int((time.perf_counter() - rerank_start) * 1000)
            logger.warning(
                "Rerank timeout, failing open to vector fallback",
                timeout_s=settings.rerank_timeout_s,
                latency_ms=rerank_ms,
            )
            seeds = _vector_fallback(candidates, config.final_k)
            rerank_state = RerankState.TIMEOUT_FALLBACK
            rerank_timeout = True
            rerank_fallback = True
        except Exception as e:
            rerank_ms = int((time.perf_counter() - rerank_start) * 1000)
            logger.warning("Reranking failed, using vector fallback", error=str(e))
            seeds = _vector_fallback(candidates, config.final_k)
            rerank_state = RerankState.ERROR_FALLBACK
            rerank_fallback = True
    else:
        seeds = _vector_fallback(candidates, config.final_k)

    # Neighbor expansion (optional)
    expand_ms: Optional[int] = None
    neighbors_added = 0

    if skip_neighbors or not config.neighbor_config.get("enabled", True):
        # Convert seeds to ExpandedChunk without neighbor expansion
        expanded = [
            ExpandedChunk(
                chunk_id=s.chunk_id,
                document_id=s.document_id,
                chunk_index=s.chunk_index,
                rerank_score=s.rerank_score,
                rerank_rank=s.rerank_rank,
                vector_score=s.vector_score,
                source_type=s.source_type,
                is_neighbor=False,
                neighbor_of=None,
            )
            for s in seeds
        ]
    else:
        expand_start = time.perf_counter()
        try:
            expanded, new_chunk_ids = await expand_neighbors(
                seeds,
                chunk_repo,
                config.neighbor_config,
                already_have_ids=set(chunks_map.keys()),
            )
            expand_ms = int((time.perf_counter() - expand_start) * 1000)

            if new_chunk_ids:
                new_chunks = await chunk_repo.get_by_ids_map(new_chunk_ids)
                chunks_map.update(new_chunks)

            neighbors_added = len([e for e in expanded if e.is_neighbor])
        except Exception as e:
            logger.warning("Neighbor expansion failed", error=str(e))
            expanded = [
                ExpandedChunk(
                    chunk_id=s.chunk_id,
                    document_id=s.document_id,
                    chunk_index=s.chunk_index,
                    rerank_score=s.rerank_score,
                    rerank_rank=s.rerank_rank,
                    vector_score=s.vector_score,
                    source_type=s.source_type,
                    is_neighbor=False,
                    neighbor_of=None,
                )
                for s in seeds
            ]
            expand_ms = int((time.perf_counter() - expand_start) * 1000)

    # Build ChunkResult objects
    results = []
    chunk_ids = []

    for exp in expanded:
        row = chunks_map.get(exp.chunk_id)
        if not row:
            continue

        score = exp.rerank_score if exp.rerank_score > 0.0 else exp.vector_score

        citation_url = _build_citation_url(
            source_url=row.get("source_url") or row.get("canonical_url"),
            source_type=row.get("source_type", ""),
            video_id=row.get("video_id"),
            time_start_secs=row.get("time_start_secs"),
            page_start=row.get("page_start"),
        )

        debug = None
        if ctx.debug:
            debug = ChunkResultDebug(
                vector_score=exp.vector_score,
                rerank_score=exp.rerank_score if exp.rerank_score > 0.0 else None,
                rerank_rank=exp.rerank_rank if exp.rerank_rank >= 0 else None,
                is_neighbor=exp.is_neighbor,
                neighbor_of=exp.neighbor_of,
            )

        results.append(
            ChunkResult(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                content=row["content"],
                score=score,
                source_url=row.get("source_url") or row.get("canonical_url"),
                citation_url=citation_url,
                title=row.get("title"),
                author=row.get("author") or row.get("channel"),
                published_at=row.get("published_at"),
                locator_label=row.get("locator_label"),
                symbols=row.get("symbols", []),
                topics=row.get("topics", []),
                debug=debug,
            )
        )
        # Track non-neighbor chunk IDs for metrics
        if not exp.is_neighbor:
            chunk_ids.append(str(row["id"]))

    total_ms = int((time.perf_counter() - total_start) * 1000)

    meta = QueryMeta(
        embed_ms=ctx.embed_ms,
        search_ms=ctx.search_ms,
        rerank_ms=rerank_ms,
        expand_ms=expand_ms,
        answer_ms=None,
        total_ms=total_ms,
        candidates_searched=len(ctx.search_results or []),
        seeds_count=len(seeds),
        chunks_after_expand=len(expanded),
        neighbors_added=neighbors_added,
        rerank_state=rerank_state,
        rerank_enabled=config.rerank_enabled,
        rerank_method=rerank_method_used,
        rerank_model=rerank_model_used,
        rerank_timeout=rerank_timeout,
        rerank_fallback=rerank_fallback,
        neighbor_enabled=config.neighbor_config.get("enabled", True),
    )

    return PipelineResult(results=results, meta=meta, chunk_ids=chunk_ids)
