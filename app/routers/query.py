"""Query endpoint for semantic search and answer generation."""

import asyncio
import time
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.schemas import (
    ChunkResult,
    ChunkResultDebug,
    CompareMetrics,
    KnowledgeExtractionStats,
    KBAnswerResponse,
    KBAnswerClaimRef,
    QueryCompareRequest,
    QueryCompareResponse,
    QueryMeta,
    QueryMode,
    QueryRequest,
    QueryResponse,
    RerankState,
)
from app.services.query_pipeline import (
    PipelineContext,
    resolve_config,
    run_retrieval_pipeline,
)
from app.services.embedder import get_embedder
from app.services.llm_factory import get_llm, get_llm_status
from app.services.neighbor_expansion import ExpandedChunk, expand_neighbors
from app.services.reranker import (
    BaseReranker,
    RerankCandidate,
    RerankResult,
    get_reranker,
)

router = APIRouter()
logger = structlog.get_logger(__name__)

# Global connection pool and clients (set during app startup)
_db_pool = None
_qdrant_client = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def set_qdrant_client(client):
    """Set the Qdrant client for this router."""
    global _qdrant_client
    _qdrant_client = client


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


def build_citation_url(
    source_url: Optional[str],
    source_type: str,
    video_id: Optional[str],
    time_start_secs: Optional[int],
    page_start: Optional[int],
) -> Optional[str]:
    """
    Build a citation URL with timestamp or page locator.

    For YouTube videos, adds ?t=123 parameter.
    For PDFs, could add #page=17 fragment.
    """
    if not source_url:
        return None

    # Handle YouTube videos
    if source_type == "youtube" and video_id and time_start_secs is not None:
        # Build YouTube URL with timestamp
        base_url = f"https://www.youtube.com/watch?v={video_id}"
        return f"{base_url}&t={time_start_secs}"

    # Handle PDFs
    if source_type == "pdf" and page_start is not None:
        return f"{source_url}#page={page_start}"

    return source_url


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        200: {"description": "Query executed successfully"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
async def query(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """
    Execute a semantic search query with optional answer generation.

    Query modes:
    - retrieve: Returns only matching chunks (no LLM call)
    - answer: Returns chunks + LLM-generated answer with citations
    - learn: Extract → verify → persist → synthesize (builds truth store)
    - kb_answer: Answer from verified claims (truth store), falls back to chunk RAG if insufficient claims  # noqa: E501

    Filter capabilities:
    - source_types: Filter by document source (youtube, pdf, article, etc.)
    - symbols: Filter by stock symbols (with any/all matching)
    - topics: Filter by detected topics
    - entities: Filter by detected entities
    - authors: Filter by author/channel
    - published_from/to: Date range filtering

    Pipeline:
    1. Embed query via Ollama
    2. Search Qdrant with filters (candidates_k for rerank or top_k otherwise)
    3. Fetch chunk metadata from Postgres
    4. Optional cross-encoder/LLM reranking
    5. Neighbor expansion (context continuity)
    6. Build citation URLs and results
    7. If mode=answer: generate LLM response with citations
    """
    from app.repositories.chunks import ChunkRepository
    from app.repositories.vectors import VectorRepository

    total_start = time.perf_counter()

    logger.info(
        "Executing query",
        workspace_id=str(request.workspace_id),
        question=request.question[:100],
        mode=request.mode.value,
        retrieve_k=request.retrieve_k,
        top_k=request.top_k,
        rerank=request.rerank,
        has_filters=request.filters is not None,
    )

    # Initialize services and repositories
    embedder = get_embedder()
    vector_repo = VectorRepository(client=_qdrant_client)

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    chunk_repo = ChunkRepository(_db_pool)

    # --- Config extraction ---
    # Precedence: request override > workspace config > defaults
    # TODO: Fetch workspace config from DB when workspace table is extended
    # For now, use hardcoded defaults with request overrides

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

    # --- Safety caps (prevent accidental latency bombs) ---
    MAX_CANDIDATES_K = 200  # CrossEncoder MAX_CANDIDATES
    MAX_FINAL_K = 50
    MAX_NEIGHBOR_TOTAL = 50
    # Rerank timeout configured in settings for testability

    # Apply request overrides
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

    # Enforce hard safety caps early (log when clamped)
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
    if workspace_neighbor["max_total"] > MAX_NEIGHBOR_TOTAL:
        workspace_neighbor["max_total"] = MAX_NEIGHBOR_TOTAL

    # Build merged rerank config
    rerank_config = {
        **workspace_rerank,
        "enabled": rerank_enabled,
        "method": rerank_method,
        "candidates_k": candidates_k,
        "final_k": final_k,
    }
    neighbor_config = workspace_neighbor.copy()

    # Enforce constraints
    final_k = min(final_k, candidates_k)

    # When rerank disabled, search only final_k (no over-fetch needed)
    if not rerank_enabled:
        candidates_k = (
            request.top_k if request.top_k is not None else workspace_retrieval["top_k"]
        )
        candidates_k = min(candidates_k, MAX_FINAL_K)  # Apply cap
        final_k = candidates_k

    # Get reranker singleton (only if enabled)
    reranker: BaseReranker | None = (
        get_reranker(rerank_config) if rerank_enabled else None
    )

    # Timing tracking
    embed_ms = 0
    search_ms = 0
    rerank_ms: int | None = None
    expand_ms: int | None = None

    # Step 1: Embed query
    embed_start = time.perf_counter()
    try:
        query_embedding = await embedder.embed(request.question)
        embed_ms = int((time.perf_counter() - embed_start) * 1000)
        logger.debug(
            "Embedded query",
            dimension=len(query_embedding),
            latency_ms=embed_ms,
        )
    except Exception as e:
        logger.error("Failed to embed query", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}",
        )

    # Step 2: Vector search with filters
    search_start = time.perf_counter()
    try:
        search_results = await vector_repo.search(
            vector=query_embedding,
            workspace_id=request.workspace_id,
            filters=request.filters,
            limit=candidates_k,
        )
        search_ms = int((time.perf_counter() - search_start) * 1000)
        logger.info(
            "Vector search complete",
            candidates=len(search_results),
            latency_ms=search_ms,
        )
    except Exception as e:
        logger.error("Vector search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector search failed: {str(e)}",
        )

    if not search_results:
        total_ms = int((time.perf_counter() - total_start) * 1000)
        return QueryResponse(
            results=[],
            answer=None,
            meta=QueryMeta(
                embed_ms=embed_ms,
                search_ms=search_ms,
                rerank_ms=None,
                expand_ms=None,
                answer_ms=None,
                total_ms=total_ms,
                candidates_searched=0,
                seeds_count=0,
                chunks_after_expand=0,
                neighbors_added=0,
                rerank_state=RerankState.DISABLED,
                rerank_enabled=rerank_enabled,
                rerank_method=None,
                rerank_model=None,
                rerank_timeout=False,
                rerank_fallback=False,
                neighbor_enabled=neighbor_config.get("enabled", True),
            ),
        )

    # Step 3: Fetch chunk metadata (needed for rerank OR neighbor expansion)
    chunk_ids = [r["id"] for r in search_results]
    try:
        chunks_map = await chunk_repo.get_by_ids_map([str(cid) for cid in chunk_ids])
        logger.debug("Fetched chunk metadata", count=len(chunks_map))
    except Exception as e:
        logger.error("Failed to fetch chunks", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunk content: {str(e)}",
        )

    # Build candidates
    candidates: list[RerankCandidate] = []
    for r in search_results:
        chunk_id = str(r["id"])
        ch = chunks_map.get(chunk_id)
        if not ch:
            continue
        candidates.append(
            RerankCandidate(
                chunk_id=chunk_id,
                document_id=str(ch["doc_id"]),
                chunk_index=ch.get("chunk_index", 0),
                text=ch["content"],
                vector_score=r["score"],
                workspace_id=str(request.workspace_id),
                source_type=ch.get("source_type"),
            )
        )

    if not candidates:
        logger.warning("No candidates after Postgres mapping")
        total_ms = int((time.perf_counter() - total_start) * 1000)
        return QueryResponse(
            results=[],
            answer=None,
            meta=QueryMeta(
                embed_ms=embed_ms,
                search_ms=search_ms,
                rerank_ms=None,
                expand_ms=None,
                answer_ms=None,
                total_ms=total_ms,
                candidates_searched=len(search_results),
                seeds_count=0,
                chunks_after_expand=0,
                neighbors_added=0,
                rerank_state=RerankState.DISABLED,
                rerank_enabled=rerank_enabled,
                rerank_method=None,
                rerank_model=None,
                rerank_timeout=False,
                rerank_fallback=False,
                neighbor_enabled=neighbor_config.get("enabled", True),
            ),
        )

    # Step 4: Rerank OR vector fallback (with timeout for fail-open safety)
    rerank_method_used: str | None = None
    rerank_model_used: str | None = None
    rerank_state = RerankState.DISABLED
    rerank_timeout = False
    rerank_fallback = False

    if rerank_enabled and reranker:
        rerank_start = time.perf_counter()
        try:
            # Wrap with timeout to fail open to vector fallback on GPU contention
            # Note: asyncio.wait_for timeout cancels the coroutine but the underlying
            # ThreadPoolExecutor thread may continue to completion. This is acceptable
            # because concurrency is limited by semaphore, and we log timeouts for monitoring.
            seeds = await asyncio.wait_for(
                reranker.rerank(request.question, candidates, final_k),
                timeout=settings.rerank_timeout_s,
            )
            rerank_method_used = reranker.method
            rerank_model_used = reranker.model_id
            rerank_ms = int((time.perf_counter() - rerank_start) * 1000)

            if not seeds:
                logger.warning("Reranker returned empty, falling back to vector order")
                seeds = _vector_fallback(candidates, final_k)
                rerank_state = RerankState.ERROR_FALLBACK
                rerank_fallback = True
            else:
                rerank_state = RerankState.OK
                logger.info(
                    "Reranked results",
                    method=rerank_method_used,
                    model=rerank_model_used,
                    original_count=len(candidates),
                    reranked_count=len(seeds),
                    latency_ms=rerank_ms,
                )
        except asyncio.TimeoutError:
            rerank_ms = int((time.perf_counter() - rerank_start) * 1000)
            logger.warning(
                "Rerank timeout, failing open to vector fallback",
                timeout_s=settings.rerank_timeout_s,
                latency_ms=rerank_ms,
            )
            seeds = _vector_fallback(candidates, final_k)
            rerank_state = RerankState.TIMEOUT_FALLBACK
            rerank_timeout = True
            rerank_fallback = True
        except Exception as e:
            rerank_ms = int((time.perf_counter() - rerank_start) * 1000)
            logger.warning("Reranking failed, using vector fallback", error=str(e))
            seeds = _vector_fallback(candidates, final_k)
            rerank_state = RerankState.ERROR_FALLBACK
            rerank_fallback = True
    else:
        seeds = _vector_fallback(candidates, final_k)
        # rerank_state remains DISABLED

    # Step 5: Neighbor expansion
    expand_start = time.perf_counter()
    try:
        expanded, new_chunk_ids = await expand_neighbors(
            seeds,
            chunk_repo,
            neighbor_config,
            already_have_ids=set(chunks_map.keys()),
        )
        expand_ms = int((time.perf_counter() - expand_start) * 1000)

        # Fetch any new chunks introduced by expansion
        if new_chunk_ids:
            new_chunks = await chunk_repo.get_by_ids_map(new_chunk_ids)
            chunks_map.update(new_chunks)

        neighbors_added = len([e for e in expanded if e.is_neighbor])
        logger.info(
            "Neighbor expansion complete",
            seeds=len(seeds),
            expanded=len(expanded),
            neighbors_added=neighbors_added,
            latency_ms=expand_ms,
        )
    except Exception as e:
        logger.warning("Neighbor expansion failed", error=str(e))
        # Convert seeds to ExpandedChunk format
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
        neighbors_added = 0
        expand_ms = int((time.perf_counter() - expand_start) * 1000)

    # Step 6: Build ChunkResult objects with citation URLs and debug info
    results = []
    for exp in expanded:
        row = chunks_map.get(exp.chunk_id)
        if not row:
            continue

        # Use rerank_score as primary score if available, else vector_score
        score = exp.rerank_score if exp.rerank_score > 0.0 else exp.vector_score

        # Build citation URL
        citation_url = build_citation_url(
            source_url=row.get("source_url") or row.get("canonical_url"),
            source_type=row.get("source_type", ""),
            video_id=row.get("video_id"),
            time_start_secs=row.get("time_start_secs"),
            page_start=row.get("page_start"),
        )

        # Build debug info (when request.debug=True, regardless of rerank status)
        debug = None
        if request.debug:
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

    # Structured observability log for query pipeline
    logger.info(
        "Query pipeline complete",
        workspace_id=str(request.workspace_id),
        mode=request.mode.value,
        rerank_enabled=rerank_enabled,
        rerank_state=rerank_state.value,
        rerank_method=rerank_method_used,
        rerank_model=rerank_model_used,
        rerank_timeout=rerank_timeout,
        rerank_fallback=rerank_fallback,
        candidates_k=candidates_k,
        final_k=final_k,
        seeds_count=len(seeds),
        expanded_count=len(expanded),
        neighbors_added=neighbors_added,
        results_count=len(results),
        embed_ms=embed_ms,
        search_ms=search_ms,
        rerank_ms=rerank_ms,
        expand_ms=expand_ms,
    )

    # Step 7: If mode=answer, generate LLM response
    answer = None
    answer_ms: int | None = None
    llm = get_llm()
    llm_status = get_llm_status()

    if request.mode == QueryMode.ANSWER and results:
        answer_start = time.perf_counter()
        if not llm or not llm_status.enabled:
            logger.info("LLM not configured, returning retrieval-only results")
            answer = (
                "[LLM generation is disabled] "
                "Retrieval is working, but no LLM provider is configured. "
                "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env to enable answer mode. "
                f"Retrieved {len(results)} relevant chunks below."
            )
        else:
            try:
                # Prepare context for LLM
                context_chunks = [
                    {
                        "content": r.content,
                        "title": r.title,
                        "source_url": r.source_url,
                        "author": r.author,
                        "locator_label": r.locator_label,
                    }
                    for r in results
                ]

                response = await llm.generate_answer(
                    question=request.question,
                    chunks=context_chunks,
                    max_context_tokens=request.max_context_tokens
                    or settings.max_context_tokens,
                    model=request.answer_model,
                )
                answer = response.text

                logger.info(
                    "Generated answer",
                    answer_length=len(answer),
                    provider=response.provider,
                    model=response.model,
                    latency_ms=response.latency_ms,
                )

            except Exception as e:
                logger.error("Answer generation failed", error=str(e))
                # Don't fail the whole request, just return without answer
                answer = f"[Error generating answer: {str(e)}]"
        answer_ms = int((time.perf_counter() - answer_start) * 1000)

    # Step 8: If mode=learn, run KB pipeline (extract → verify → persist → synthesize)
    knowledge_stats = None

    if request.mode == QueryMode.LEARN and results:
        if not llm or not llm_status.enabled:
            logger.info("LLM not configured, cannot run learn mode")
            answer = (
                "[LLM generation is disabled] "
                "Learn mode requires an LLM provider. "
                "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env to enable learn mode."
            )
        else:
            try:
                from app.services.kb_pipeline import KBPipeline
                from app.repositories.kb import KnowledgeBaseRepository

                # Prepare context chunks for pipeline
                context_chunks = [
                    {
                        "content": r.content,
                        "title": r.title,
                        "source_url": r.source_url,
                        "doc_id": str(r.doc_id),
                        "chunk_id": str(r.chunk_id),
                    }
                    for r in results
                ]

                # Run KB pipeline
                pipeline = KBPipeline(llm=llm)
                pipeline_result = await pipeline.run(
                    chunks=context_chunks,
                    question=request.question,
                    synthesize=True,
                )

                # Persist to truth store
                kb_repo = KnowledgeBaseRepository(_db_pool)
                chunk_ids = [r.chunk_id for r in results]
                doc_id = results[0].doc_id if results else None

                if doc_id and pipeline_result.extraction.claims:
                    persistence_stats = await kb_repo.persist_extraction(
                        workspace_id=request.workspace_id,
                        entities=pipeline_result.extraction.entities,
                        claims=pipeline_result.extraction.claims,
                        relations=pipeline_result.extraction.relations,
                        verdicts=pipeline_result.verification.verdicts,
                        chunk_ids=chunk_ids,
                        doc_id=doc_id,
                        extraction_model=pipeline.extraction_model,
                        verification_model=pipeline.verification_model,
                    )
                else:
                    from app.services.kb_types import PersistenceStats

                    persistence_stats = PersistenceStats()

                # Use synthesized answer from pipeline
                answer = pipeline_result.synthesized_answer

                # Build stats
                knowledge_stats = KnowledgeExtractionStats(
                    entities_extracted=len(pipeline_result.extraction.entities),
                    claims_extracted=len(pipeline_result.extraction.claims),
                    relations_extracted=len(pipeline_result.extraction.relations),
                    claims_verified=pipeline_result.verified_claims_count,
                    claims_weak=pipeline_result.weak_claims_count,
                    claims_rejected=pipeline_result.rejected_claims_count,
                    entities_persisted=persistence_stats.entities_created
                    + persistence_stats.entities_updated,
                    claims_persisted=persistence_stats.claims_created,
                    claims_skipped_duplicate=persistence_stats.claims_skipped_duplicate,
                    claims_skipped_invalid=persistence_stats.claims_skipped_invalid,
                )

                logger.info(
                    "Learn mode complete",
                    entities=knowledge_stats.entities_extracted,
                    claims=knowledge_stats.claims_extracted,
                    verified=knowledge_stats.claims_verified,
                    persisted=knowledge_stats.claims_persisted,
                )

            except Exception as e:
                logger.error("Learn mode failed", error=str(e))
                answer = f"[Error in learn mode: {str(e)}]"

    # Step 9: If mode=kb_answer, answer from truth store
    kb_answer_response = None
    MIN_CLAIMS_FOR_KB_ANSWER = 3

    if request.mode == QueryMode.KB_ANSWER:
        if not llm or not llm_status.enabled:
            logger.info("LLM not configured, cannot run kb_answer mode")
            answer = (
                "[LLM generation is disabled] "
                "KB answer mode requires an LLM provider. "
                "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env to enable kb_answer mode."
            )
        else:
            try:
                from app.repositories.kb import KnowledgeBaseRepository
                from app.services.kb_prompts import (
                    KB_ANSWER_SYSTEM_PROMPT,
                    build_kb_answer_prompt,
                    extract_json_from_response,
                )

                kb_repo = KnowledgeBaseRepository(_db_pool)

                # Search verified claims using text search
                verified_claims = await kb_repo.search_claims_for_answer(
                    workspace_id=request.workspace_id,
                    query_text=request.question,
                    limit=20,
                    min_confidence=0.5,
                )

                logger.info(
                    "KB answer: retrieved claims",
                    claim_count=len(verified_claims),
                    min_required=MIN_CLAIMS_FOR_KB_ANSWER,
                )

                # Check if we have enough claims
                if len(verified_claims) >= MIN_CLAIMS_FOR_KB_ANSWER:
                    # Synthesize answer from claims
                    prompt = build_kb_answer_prompt(
                        question=request.question,
                        claims=verified_claims,
                    )

                    response = await llm.generate(
                        messages=[
                            {"role": "system", "content": KB_ANSWER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=2000,
                    )

                    # Parse JSON response
                    try:
                        data = extract_json_from_response(response.text)
                        answer = data.get("answer", "")
                        supported = data.get("supported", [])
                        not_specified = data.get("not_specified", [])
                    except ValueError as e:
                        logger.warning("Failed to parse kb_answer JSON", error=str(e))
                        answer = response.text
                        supported = []
                        not_specified = []

                    # Build claim references
                    claims_used = [
                        KBAnswerClaimRef(
                            id=f"C{i+1}",
                            claim_id=claim["id"],
                            confidence=claim.get("confidence", 0.5),
                        )
                        for i, claim in enumerate(verified_claims)
                    ]

                    kb_answer_response = KBAnswerResponse(
                        mode="kb_answer",
                        llm_enabled=True,
                        answer=answer,
                        supported=supported,
                        not_specified=not_specified,
                        claims_used=claims_used,
                        fallback_used=False,
                    )

                    logger.info(
                        "KB answer synthesized",
                        claims_used=len(claims_used),
                        answer_length=len(answer),
                    )

                else:
                    # Not enough claims - fallback to chunk RAG or indicate insufficient knowledge
                    logger.info(
                        "KB answer: insufficient claims, falling back to chunk RAG",
                        claim_count=len(verified_claims),
                    )

                    # Fallback: use chunk RAG if we have results
                    if results:
                        context_chunks = [
                            {
                                "content": r.content,
                                "title": r.title,
                                "source_url": r.source_url,
                                "author": r.author,
                                "locator_label": r.locator_label,
                            }
                            for r in results
                        ]

                        response = await llm.generate_answer(
                            question=request.question,
                            chunks=context_chunks,
                            max_context_tokens=request.max_context_tokens
                            or settings.max_context_tokens,
                            model=request.answer_model,
                        )
                        answer = response.text

                        kb_answer_response = KBAnswerResponse(
                            mode="kb_answer",
                            llm_enabled=True,
                            answer=answer,
                            supported=[],
                            not_specified=[
                                "Insufficient verified claims in knowledge base, used chunk RAG fallback"  # noqa: E501
                            ],
                            claims_used=[],
                            fallback_used=True,
                            fallback_reason=f"Only {len(verified_claims)} verified claims found (minimum {MIN_CLAIMS_FOR_KB_ANSWER} required)",  # noqa: E501
                        )
                    else:
                        answer = (
                            f"Insufficient knowledge in truth store. "
                            f"Found {len(verified_claims)} verified claims (minimum {MIN_CLAIMS_FOR_KB_ANSWER} required). "  # noqa: E501
                            f"Consider using mode=learn to build the knowledge base first."
                        )
                        kb_answer_response = KBAnswerResponse(
                            mode="kb_answer",
                            llm_enabled=True,
                            answer=None,
                            supported=[],
                            not_specified=[answer],
                            claims_used=[],
                            fallback_used=False,
                            fallback_reason=f"Only {len(verified_claims)} verified claims found, no chunks available for fallback",  # noqa: E501
                        )

            except Exception as e:
                logger.error("KB answer mode failed", error=str(e))
                answer = f"[Error in kb_answer mode: {str(e)}]"

    # Build final QueryMeta
    total_ms = int((time.perf_counter() - total_start) * 1000)
    meta = QueryMeta(
        embed_ms=embed_ms,
        search_ms=search_ms,
        rerank_ms=rerank_ms,
        expand_ms=expand_ms,
        answer_ms=answer_ms,
        total_ms=total_ms,
        candidates_searched=len(search_results),
        seeds_count=len(seeds),
        chunks_after_expand=len(expanded),
        neighbors_added=neighbors_added,
        rerank_state=rerank_state,
        rerank_enabled=rerank_enabled,
        rerank_method=rerank_method_used,
        rerank_model=rerank_model_used,
        rerank_timeout=rerank_timeout,
        rerank_fallback=rerank_fallback,
        neighbor_enabled=neighbor_config.get("enabled", True),
    )

    return QueryResponse(
        results=results,
        answer=answer,
        knowledge_stats=knowledge_stats,
        meta=meta,
        kb_answer=kb_answer_response,
    )


def _compute_spearman(ranks_a: list[int], ranks_b: list[int]) -> float:
    """Compute Spearman rank correlation coefficient.

    Uses the formula: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    where d is the difference in ranks.
    """
    n = len(ranks_a)
    if n < 2:
        raise ValueError("Need at least 2 items for Spearman correlation")

    d_squared_sum = sum((a - b) ** 2 for a, b in zip(ranks_a, ranks_b))
    return 1 - (6 * d_squared_sum) / (n * (n**2 - 1))


def compute_compare_metrics(
    vector_ids: list[str],
    reranked_ids: list[str],
) -> CompareMetrics:
    """Compute comparison metrics between vector-only and reranked results.

    Args:
        vector_ids: Chunk IDs from vector-only run (in rank order)
        reranked_ids: Chunk IDs from reranked run (in rank order)

    Returns:
        CompareMetrics with Jaccard, Spearman, and rank delta stats
    """
    set_a = set(vector_ids)
    set_b = set(reranked_ids)

    intersection = set_a & set_b
    union = set_a | set_b

    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 1.0

    # Build rank maps
    rank_a = {id_: i for i, id_ in enumerate(vector_ids)}
    rank_b = {id_: i for i, id_ in enumerate(reranked_ids)}

    # Spearman correlation (only over intersection)
    spearman = None
    if len(intersection) >= 2:
        intersection_list = list(intersection)
        ranks_in_a = [rank_a[id_] for id_ in intersection_list]
        ranks_in_b = [rank_b[id_] for id_ in intersection_list]
        try:
            spearman = _compute_spearman(ranks_in_a, ranks_in_b)
        except (ValueError, ZeroDivisionError):
            spearman = None

    # Rank delta stats
    rank_delta_mean = None
    rank_delta_max = None
    if intersection:
        deltas = [abs(rank_a[id_] - rank_b[id_]) for id_ in intersection]
        rank_delta_mean = sum(deltas) / len(deltas)
        rank_delta_max = max(deltas)

    return CompareMetrics(
        jaccard=jaccard,
        overlap_count=len(intersection),
        union_count=len(union),
        spearman=spearman,
        rank_delta_mean=rank_delta_mean,
        rank_delta_max=rank_delta_max,
        vector_only_ids=vector_ids,
        reranked_ids=reranked_ids,
        intersection_ids=list(intersection),
    )


@router.post(
    "/query/compare",
    response_model=QueryCompareResponse,
    responses={
        200: {"description": "Comparison executed successfully"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def query_compare(
    request: QueryCompareRequest,
    settings: Settings = Depends(get_settings),
) -> QueryCompareResponse:
    """
    Compare vector-only vs reranked retrieval for tuning.

    Runs both pipelines on the same candidate set for fair comparison:
    1. Embed query once
    2. Vector search once (using rerank-enabled candidates_k)
    3. Run vector-only pipeline (no rerank)
    4. Run reranked pipeline (with rerank)
    5. Compute comparison metrics (Jaccard, Spearman, rank deltas)

    Use this endpoint to:
    - Validate whether reranking improves results for your data
    - Tune retrieve_k and top_k parameters
    - Compare result overlap and rank correlation

    Note: Always uses mode=retrieve (no LLM answer generation).
    """
    from app.repositories.chunks import ChunkRepository
    from app.repositories.vectors import VectorRepository

    logger.info(
        "Executing query compare",
        workspace_id=str(request.workspace_id),
        question=request.question[:100],
        retrieve_k=request.retrieve_k,
        top_k=request.top_k,
    )

    # Initialize services
    embedder = get_embedder()
    vector_repo = VectorRepository(client=_qdrant_client)

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    chunk_repo = ChunkRepository(_db_pool)

    # Build a synthetic QueryRequest for config resolution
    # (QueryCompareRequest doesn't have mode, rerank, etc.)
    synthetic_request = QueryRequest(
        workspace_id=request.workspace_id,
        question=request.question,
        mode=QueryMode.RETRIEVE,
        filters=request.filters,
        retrieve_k=request.retrieve_k,
        top_k=request.top_k,
        rerank=True,  # Use rerank-enabled config for candidates_k
        debug=request.debug,
    )

    # Resolve config with rerank=True to get the superset candidates_k
    rerank_config = resolve_config(synthetic_request, force_rerank=True)

    # Step 1: Embed query once
    embed_start = time.perf_counter()
    try:
        query_embedding = await embedder.embed(request.question)
        embed_ms = int((time.perf_counter() - embed_start) * 1000)
    except Exception as e:
        logger.error("Failed to embed query", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}",
        )

    # Step 2: Vector search once (using rerank candidates_k as superset)
    search_start = time.perf_counter()
    try:
        search_results = await vector_repo.search(
            vector=query_embedding,
            workspace_id=request.workspace_id,
            filters=request.filters,
            limit=rerank_config.candidates_k,
        )
        search_ms = int((time.perf_counter() - search_start) * 1000)
    except Exception as e:
        logger.error("Vector search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector search failed: {str(e)}",
        )

    # Step 3: Fetch chunk metadata once
    if not search_results:
        # Empty results - return empty comparison
        empty_meta = QueryMeta(
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=None,
            expand_ms=None,
            answer_ms=None,
            total_ms=embed_ms + search_ms,
            candidates_searched=0,
            seeds_count=0,
            chunks_after_expand=0,
            neighbors_added=0,
            rerank_state=RerankState.DISABLED,
            rerank_enabled=False,
            rerank_method=None,
            rerank_model=None,
            rerank_timeout=False,
            rerank_fallback=False,
            neighbor_enabled=True,
        )
        empty_response = QueryResponse(results=[], answer=None, meta=empty_meta)
        empty_metrics = CompareMetrics(
            jaccard=1.0,
            overlap_count=0,
            union_count=0,
            spearman=None,
            rank_delta_mean=None,
            rank_delta_max=None,
            vector_only_ids=[],
            reranked_ids=[],
            intersection_ids=[],
        )
        return QueryCompareResponse(
            vector_only=empty_response,
            reranked=empty_response,
            metrics=empty_metrics,
        )

    chunk_ids = [r["id"] for r in search_results]
    try:
        chunks_map = await chunk_repo.get_by_ids_map([str(cid) for cid in chunk_ids])
    except Exception as e:
        logger.error("Failed to fetch chunks", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunk content: {str(e)}",
        )

    # Build candidates once
    candidates: list[RerankCandidate] = []
    for r in search_results:
        chunk_id = str(r["id"])
        ch = chunks_map.get(chunk_id)
        if not ch:
            continue
        candidates.append(
            RerankCandidate(
                chunk_id=chunk_id,
                document_id=str(ch["doc_id"]),
                chunk_index=ch.get("chunk_index", 0),
                text=ch["content"],
                vector_score=r["score"],
                workspace_id=str(request.workspace_id),
                source_type=ch.get("source_type"),
            )
        )

    # Build shared context
    ctx = PipelineContext(
        workspace_id=request.workspace_id,
        question=request.question,
        debug=request.debug,
        query_embedding=query_embedding,
        search_results=search_results,
        chunks_map=chunks_map,
        candidates=candidates,
        embed_ms=embed_ms,
        search_ms=search_ms,
    )

    # Step 4: Run vector-only pipeline
    vector_config = resolve_config(synthetic_request, force_rerank=False)
    vector_result = await run_retrieval_pipeline(
        ctx,
        vector_config,
        chunk_repo,
        settings,
        skip_neighbors=request.skip_neighbors,
    )

    # Step 5: Run reranked pipeline
    reranked_result = await run_retrieval_pipeline(
        ctx,
        rerank_config,
        chunk_repo,
        settings,
        skip_neighbors=request.skip_neighbors,
    )

    # Step 6: Compute metrics
    metrics = compute_compare_metrics(
        vector_ids=vector_result.chunk_ids,
        reranked_ids=reranked_result.chunk_ids,
    )

    # Structured eval logging for bulk analysis
    # This is the primary analytics event for compare endpoint tuning
    logger.info(
        "query_compare",
        # Identifiers
        workspace_id=str(request.workspace_id),
        # Config
        candidates_k=rerank_config.candidates_k,
        top_k=rerank_config.final_k,
        share_candidates=True,
        skip_neighbors=request.skip_neighbors,
        rerank_method=reranked_result.meta.rerank_method,
        rerank_model=reranked_result.meta.rerank_model,
        # Metrics
        jaccard=metrics.jaccard,
        spearman=metrics.spearman,
        rank_delta_mean=metrics.rank_delta_mean,
        rank_delta_max=metrics.rank_delta_max,
        overlap_count=metrics.overlap_count,
        union_count=metrics.union_count,
        # Latency (ms)
        embed_ms=embed_ms,
        search_ms=search_ms,
        vector_total_ms=vector_result.meta.total_ms,
        rerank_ms=reranked_result.meta.rerank_ms,
        rerank_total_ms=reranked_result.meta.total_ms,
        # State
        rerank_state=reranked_result.meta.rerank_state.value,
        rerank_timeout=reranked_result.meta.rerank_timeout,
        rerank_fallback=reranked_result.meta.rerank_fallback,
        # Top-5 IDs for spot-checking (lightweight, no content)
        vector_top5_ids=vector_result.chunk_ids[:5],
        reranked_top5_ids=reranked_result.chunk_ids[:5],
    )

    # Persist evaluation (fail-safe, never breaks the endpoint)
    if settings.eval_persist_enabled and _db_pool is not None:
        try:
            from app.repositories.evals import EvalRepository

            eval_repo = EvalRepository(_db_pool)
            await eval_repo.insert(
                workspace_id=request.workspace_id,
                question=request.question,
                candidates_k=rerank_config.candidates_k,
                top_k=rerank_config.final_k,
                share_candidates=True,
                skip_neighbors=request.skip_neighbors,
                rerank_method=reranked_result.meta.rerank_method,
                rerank_model=reranked_result.meta.rerank_model,
                jaccard=metrics.jaccard,
                spearman=metrics.spearman,
                rank_delta_mean=metrics.rank_delta_mean,
                rank_delta_max=metrics.rank_delta_max,
                overlap_count=metrics.overlap_count,
                union_count=metrics.union_count,
                embed_ms=embed_ms,
                search_ms=search_ms,
                vector_total_ms=vector_result.meta.total_ms,
                rerank_ms=reranked_result.meta.rerank_ms,
                rerank_total_ms=reranked_result.meta.total_ms,
                rerank_state=reranked_result.meta.rerank_state.value,
                rerank_timeout=reranked_result.meta.rerank_timeout,
                rerank_fallback=reranked_result.meta.rerank_fallback,
                vector_top5_ids=vector_result.chunk_ids[:5],
                reranked_top5_ids=reranked_result.chunk_ids[:5],
                store_question_preview=settings.eval_store_question_preview,
            )
        except Exception as e:
            # Log but don't fail the request
            logger.warning("Failed to persist eval", error=str(e))

    return QueryCompareResponse(
        vector_only=QueryResponse(
            results=vector_result.results,
            answer=None,
            meta=vector_result.meta,
        ),
        reranked=QueryResponse(
            results=reranked_result.results,
            answer=None,
            meta=reranked_result.meta,
        ),
        metrics=metrics,
    )
