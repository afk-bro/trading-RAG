"""Query endpoint for semantic search and answer generation."""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.schemas import (
    ChunkResult,
    KnowledgeExtractionStats,
    KBAnswerResponse,
    KBAnswerClaimRef,
    QueryMode,
    QueryRequest,
    QueryResponse,
)
from app.services.embedder import get_embedder
from app.services.llm_factory import get_llm, get_llm_status

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
    - kb_answer: Answer from verified claims (truth store), falls back to chunk RAG if insufficient claims

    Filter capabilities:
    - source_types: Filter by document source (youtube, pdf, article, etc.)
    - symbols: Filter by stock symbols (with any/all matching)
    - topics: Filter by detected topics
    - entities: Filter by detected entities
    - authors: Filter by author/channel
    - published_from/to: Date range filtering

    Pipeline:
    1. Embed query via Ollama
    2. Search Qdrant with filters (retrieve_k candidates)
    3. Optional reranking
    4. Truncate to top_k
    5. Hydrate from Postgres (preserving rank order)
    6. Build citation URLs
    7. If mode=answer: generate LLM response with citations
    """
    from app.repositories.chunks import ChunkRepository
    from app.repositories.vectors import VectorRepository

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

    # Step 1: Embed query
    try:
        query_embedding = await embedder.embed(request.question)
        logger.debug(
            "Embedded query",
            dimension=len(query_embedding),
        )
    except Exception as e:
        logger.error("Failed to embed query", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {str(e)}",
        )

    # Step 2: Search Qdrant with filters
    try:
        search_results = await vector_repo.search(
            vector=query_embedding,
            workspace_id=request.workspace_id,
            filters=request.filters,
            limit=request.retrieve_k,
        )
        logger.info(
            "Vector search complete",
            candidates=len(search_results),
        )
    except Exception as e:
        logger.error("Vector search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector search failed: {str(e)}",
        )

    if not search_results:
        return QueryResponse(results=[], answer=None)

    # Step 3: Optional reranking
    llm = get_llm()
    if request.rerank and llm and len(search_results) > request.top_k:
        try:
            # We need content for reranking - fetch it first
            chunk_ids = [r["id"] for r in search_results]
            chunks_data = await chunk_repo.get_by_ids(chunk_ids, preserve_order=True)

            # Merge content into results
            content_map = {row["id"]: row["content"] for row in chunks_data}
            for result in search_results:
                result["content"] = content_map.get(result["id"], "")

            rerank_chunks = [
                {"content": r.get("content", ""), "score": r["score"], "id": r["id"]}
                for r in search_results
            ]

            reranked = await llm.rerank(
                query=request.question,
                chunks=rerank_chunks,
                top_k=request.top_k,
            )
            # Convert RankedChunk back to search_results format
            search_results = [
                {"id": rc.chunk["id"], "score": rc.score, "content": rc.chunk.get("content", "")}
                for rc in reranked
            ]
            logger.info(
                "Reranked results",
                original_count=len(rerank_chunks),
                reranked_count=len(search_results),
            )
        except Exception as e:
            logger.warning("Reranking failed, using original order", error=str(e))
            # Continue without reranking
            search_results = search_results[:request.top_k]
    else:
        # Step 4: Truncate to top_k
        search_results = search_results[:request.top_k]

    # Step 5: Hydrate from Postgres
    chunk_ids = [r["id"] for r in search_results]

    try:
        chunks_data = await chunk_repo.get_by_ids(chunk_ids, preserve_order=True)
        logger.debug(
            "Hydrated chunks",
            count=len(chunks_data),
        )
    except Exception as e:
        logger.error("Failed to hydrate chunks", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunk content: {str(e)}",
        )

    # Build score map from search results
    score_map = {r["id"]: r["score"] for r in search_results}

    # Step 6: Build ChunkResult objects with citation URLs
    results = []
    for row in chunks_data:
        chunk_id = row["id"]
        score = score_map.get(chunk_id, 0.0)

        # Build citation URL
        citation_url = build_citation_url(
            source_url=row.get("source_url") or row.get("canonical_url"),
            source_type=row.get("source_type", ""),
            video_id=row.get("video_id"),
            time_start_secs=row.get("time_start_secs"),
            page_start=row.get("page_start"),
        )

        results.append(
            ChunkResult(
                chunk_id=chunk_id,
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
            )
        )

    logger.info(
        "Built results",
        count=len(results),
    )

    # Step 7: If mode=answer, generate LLM response
    answer = None
    llm_status = get_llm_status()

    if request.mode == QueryMode.ANSWER and results:
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
                    max_context_tokens=request.max_context_tokens or settings.max_context_tokens,
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
                    entities_persisted=persistence_stats.entities_created + persistence_stats.entities_updated,
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
                            max_context_tokens=request.max_context_tokens or settings.max_context_tokens,
                        )
                        answer = response.text

                        kb_answer_response = KBAnswerResponse(
                            mode="kb_answer",
                            llm_enabled=True,
                            answer=answer,
                            supported=[],
                            not_specified=["Insufficient verified claims in knowledge base, used chunk RAG fallback"],
                            claims_used=[],
                            fallback_used=True,
                            fallback_reason=f"Only {len(verified_claims)} verified claims found (minimum {MIN_CLAIMS_FOR_KB_ANSWER} required)",
                        )
                    else:
                        answer = (
                            f"Insufficient knowledge in truth store. "
                            f"Found {len(verified_claims)} verified claims (minimum {MIN_CLAIMS_FOR_KB_ANSWER} required). "
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
                            fallback_reason=f"Only {len(verified_claims)} verified claims found, no chunks available for fallback",
                        )

            except Exception as e:
                logger.error("KB answer mode failed", error=str(e))
                answer = f"[Error in kb_answer mode: {str(e)}]"

    return QueryResponse(
        results=results,
        answer=answer,
        knowledge_stats=knowledge_stats,
        kb_answer=kb_answer_response,
    )
