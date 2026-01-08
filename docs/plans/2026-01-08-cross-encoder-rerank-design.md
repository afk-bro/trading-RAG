# Cross-Encoder Reranker Design

**Status:** Ready for implementation
**Created:** 2026-01-08
**Milestone:** Cross-encoder Rerank v1 (production-shape)

## Overview

Add a cross-encoder reranker to the retrieval pipeline for significant relevance improvements. The reranker scores (query, passage) pairs directly using a dedicated ML model, providing better ranking than vector similarity alone.

**Pipeline after this change:**
```
vector_search (top 50) → rerank (top 10) → neighbor_expand → answer
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Reranker strategy | Parallel (cross-encoder default, LLM optional) | Cross-encoder as default; LLM rerank kept as power tool for A/B testing |
| Model | BAAI/bge-reranker-v2-m3 | Apache 2.0 license, strong RAG performance, multilingual |
| GPU | Required | ~100ms for 50 pairs vs ~500ms on CPU |
| Neighbor expansion | After rerank | Neighbors are for continuity, not relevance scoring |

## Architecture

### Service Layer

**File:** `app/services/reranker.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class RerankCandidate:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    vector_score: float
    workspace_id: str
    source_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class RerankResult:
    chunk_id: str
    document_id: str
    chunk_index: int
    rerank_score: float      # 0.0 if no rerank ran
    rerank_rank: int         # 0-based
    vector_score: float
    source_type: str | None = None

class BaseReranker(ABC):
    @property
    @abstractmethod
    def method(self) -> str:
        """Reranker method identifier."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str | None:
        """Model identifier for observability."""
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int
    ) -> list[RerankResult]:
        pass

    def close(self, wait: bool = True):
        """Cleanup resources."""
        pass
```

**Singleton factory:**

```python
_cross_encoder_reranker: CrossEncoderReranker | None = None
_llm_reranker: LLMReranker | None = None

def get_reranker(config: dict) -> BaseReranker | None:
    global _cross_encoder_reranker, _llm_reranker

    if not config.get("enabled", False):
        return None

    method = config.get("method", "cross_encoder")

    if method == "cross_encoder":
        if _cross_encoder_reranker is None:
            _cross_encoder_reranker = CrossEncoderReranker(
                config.get("cross_encoder", {})
            )
        return _cross_encoder_reranker

    elif method == "llm":
        if _llm_reranker is None:
            _llm_reranker = LLMReranker(config.get("llm", {}))
        return _llm_reranker

    else:
        logger.warning(f"Unknown rerank method '{method}', falling back to cross_encoder")
        if _cross_encoder_reranker is None:
            _cross_encoder_reranker = CrossEncoderReranker(
                config.get("cross_encoder", {})
            )
        return _cross_encoder_reranker
```

### CrossEncoderReranker Implementation

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker(BaseReranker):
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    MAX_CANDIDATES = 200

    def __init__(self, config: dict):
        self._model_id = config.get("model", self.DEFAULT_MODEL)
        self._device = config.get("device", "cuda")
        self._max_text_chars = config.get("max_text_chars", 2000)
        self._batch_size = config.get("batch_size", 16)
        self._max_concurrent = config.get("max_concurrent", 2)

        self._model: CrossEncoder | None = None
        self._init_lock: asyncio.Lock | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)  # single thread for CUDA

    @property
    def method(self) -> str:
        return "cross_encoder"

    @property
    def model_id(self) -> str:
        return self._model_id

    def _get_init_lock(self) -> asyncio.Lock:
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def _ensure_model(self) -> CrossEncoder:
        if self._model is not None:
            return self._model

        async with self._get_init_lock():
            if self._model is None:
                logger.info(f"Loading cross-encoder model: {self._model_id}")
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    self._executor,
                    lambda: CrossEncoder(self._model_id, device=self._device)
                )
                logger.info(f"Cross-encoder model loaded on {self._device}")
        return self._model

    def _score_sync(
        self,
        model: CrossEncoder,
        query: str,
        texts: list[str]
    ) -> list[float]:
        pairs = [[query, t] for t in texts]
        scores = model.predict(pairs, batch_size=self._batch_size)
        if hasattr(scores, "tolist"):
            return scores.tolist()
        return list(scores)

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int
    ) -> list[RerankResult]:
        if not candidates:
            return []

        # Cap candidates, preserving best vector hits
        if len(candidates) > self.MAX_CANDIDATES:
            logger.warning(
                f"Candidate count {len(candidates)} exceeds max {self.MAX_CANDIDATES}, "
                "truncating to top vector scores"
            )
            candidates = sorted(
                candidates,
                key=lambda c: c.vector_score,
                reverse=True
            )[:self.MAX_CANDIDATES]

        model = await self._ensure_model()
        semaphore = self._get_semaphore()

        texts = [c.text[:self._max_text_chars] for c in candidates]

        async with semaphore:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                self._executor,
                self._score_sync,
                model,
                query,
                texts
            )

        # Sort with fully deterministic tie-breaker
        scored = list(zip(candidates, scores))
        scored.sort(
            key=lambda x: (x[1], x[0].vector_score, x[0].chunk_id),
            reverse=True
        )

        results = []
        for rank, (candidate, score) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                chunk_index=candidate.chunk_index,
                rerank_score=float(score),
                rerank_rank=rank,
                vector_score=candidate.vector_score,
                source_type=candidate.source_type,
            ))

        return results

    def close(self, wait: bool = True):
        self._executor.shutdown(wait=wait)
        self._model = None
```

### LLMReranker Implementation

Wraps existing `BaseLLMClient.rerank()` to fit the new interface:

```python
from app.services.llm_factory import get_llm

class LLMReranker(BaseReranker):
    def __init__(self, config: dict):
        self._config = config

    @property
    def method(self) -> str:
        return "llm"

    @property
    def model_id(self) -> str | None:
        llm = get_llm()
        return llm.rerank_model if llm else None

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int
    ) -> list[RerankResult]:
        llm = get_llm()
        if not llm:
            logger.warning("No LLM available for reranking, returning vector order")
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
                for i, c in enumerate(candidates[:top_k])
            ]

        chunks = [{"chunk_id": c.chunk_id, "text": c.text} for c in candidates]
        ranked = await llm.rerank(query, chunks, top_k)

        chunk_map = {c.chunk_id: c for c in candidates}
        results = []

        for rank, r in enumerate(ranked):
            # Handle both object and dict returns
            if hasattr(r, "chunk_id"):
                chunk_id, score = r.chunk_id, r.score
            else:
                chunk_id, score = r.get("chunk_id"), r.get("score", 0.0)

            candidate = chunk_map.get(chunk_id)
            if candidate:
                results.append(RerankResult(
                    chunk_id=chunk_id,
                    document_id=candidate.document_id,
                    chunk_index=candidate.chunk_index,
                    rerank_score=float(score),
                    rerank_rank=rank,
                    vector_score=candidate.vector_score,
                    source_type=candidate.source_type,
                ))
            else:
                logger.debug(f"LLM returned unmapped chunk_id: {chunk_id}")

        return results

    def close(self):
        pass
```

## Pipeline Integration

**File:** `app/routers/query.py`

```python
async def search(
    request: QueryRequest,
    workspace: Workspace,
    db_pool: asyncpg.Pool,
    qdrant: AsyncQdrantClient,
) -> QueryResponse:

    # --- Config extraction ---
    rerank_config = (workspace.config or {}).get("rerank", {})
    rerank_enabled = bool(rerank_config.get("enabled", False))
    reranker = get_reranker(rerank_config) if rerank_enabled else None

    # Determine k values
    if rerank_enabled:
        candidates_k = int(rerank_config.get("candidates_k", 50))
        final_k = int(rerank_config.get("final_k", 10))
    else:
        candidates_k = request.top_k
        final_k = request.top_k
    final_k = min(final_k, candidates_k)

    # --- 1. Embed query ---
    embedder = OllamaEmbedder()
    query_vector = await embedder.embed(request.query)

    # --- 2. Vector search ---
    vector_repo = VectorRepository(qdrant)
    search_results = await vector_repo.search(
        collection=workspace.default_collection,
        vector=query_vector,
        limit=candidates_k,
        workspace_id=str(workspace.id),
        filters=build_filters(request),
    )

    if not search_results:
        return QueryResponse(results=[], meta=QueryMeta(...))

    # --- 3. Fetch chunk metadata (needed for rerank OR neighbor expansion) ---
    chunk_ids = [r.chunk_id for r in search_results]
    chunk_repo = ChunkRepository(db_pool)
    chunks_map = await chunk_repo.get_by_ids_map(chunk_ids)

    # Build candidates
    candidates: list[RerankCandidate] = []
    for r in search_results:
        ch = chunks_map.get(r.chunk_id)
        if not ch:
            continue
        candidates.append(RerankCandidate(
            chunk_id=r.chunk_id,
            document_id=ch.document_id,
            chunk_index=ch.chunk_index,
            text=ch.text,
            vector_score=r.score,
            workspace_id=str(workspace.id),
            source_type=ch.source_type,
        ))

    if not candidates:
        logger.warning("No candidates after Postgres mapping")
        return QueryResponse(results=[], meta=QueryMeta(...))

    # --- 4. Rerank OR vector fallback ---
    rerank_method: str | None = None
    rerank_model: str | None = None

    if rerank_enabled and reranker:
        seeds = await reranker.rerank(request.query, candidates, final_k)
        rerank_method = reranker.method
        rerank_model = reranker.model_id

        if not seeds:
            logger.warning("Reranker returned empty, falling back to vector order")
            seeds = _vector_fallback(candidates, final_k)
    else:
        seeds = _vector_fallback(candidates, final_k)

    # --- 5. Neighbor expansion ---
    neighbor_config = (workspace.config or {}).get("neighbor", {})
    expanded, new_chunk_ids = await expand_neighbors(
        seeds,
        chunk_repo,
        neighbor_config,
        already_have_ids=set(chunks_map.keys())
    )

    # Fetch any new chunks introduced by expansion
    if new_chunk_ids:
        new_chunks = await chunk_repo.get_by_ids_map(new_chunk_ids)
        chunks_map.update(new_chunks)

    # --- 6. Build results ---
    results = build_chunk_results(expanded, chunks_map, rerank_enabled)

    # --- 7. Answer (if mode=answer) ---
    answer = None
    if request.mode == QueryMode.ANSWER:
        answer = await generate_answer(request.query, results, ...)

    return QueryResponse(
        results=results,
        answer=answer,
        meta=QueryMeta(
            rerank_enabled=rerank_enabled,
            rerank_method=rerank_method,
            rerank_model=rerank_model,
            seeds_count=len(seeds),
            chunks_after_expand=len(expanded),
            neighbors_added=len([e for e in expanded if e.is_neighbor]),
        ),
    )


def _vector_fallback(
    candidates: list[RerankCandidate],
    final_k: int
) -> list[RerankResult]:
    """Convert candidates to RerankResults in vector score order."""
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.vector_score, c.chunk_id),
        reverse=True
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
```

## Neighbor Expansion

**File:** `app/services/neighbor_expansion.py`

```python
@dataclass
class ExpandedChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    rerank_score: float
    rerank_rank: int
    vector_score: float
    source_type: str | None
    is_neighbor: bool
    neighbor_of: str | None

async def expand_neighbors(
    seeds: list[RerankResult],
    chunk_repo: ChunkRepository,
    config: dict,
    already_have_ids: set[str] | None = None,
) -> tuple[list[ExpandedChunk], list[str]]:
    """
    Expand seeds with neighboring chunks.

    Returns:
        - expanded list ordered by (best_doc_rank, doc_id, chunk_index)
        - list of new chunk_ids that need fetching
    """
    already_have_ids = already_have_ids or set()

    if not seeds:
        return [], []

    enabled = config.get("enabled", True)

    # Convert seeds to ExpandedChunk
    seed_expanded = [
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

    if not enabled:
        return seed_expanded, []

    window = config.get("window", 1)
    pdf_window = config.get("pdf_window", 2)
    min_chars = config.get("min_chars", 200)
    max_total = config.get("max_total", 20)

    # Build deduped neighbor requests, best seed (lowest rank) wins
    seed_set = {s.chunk_id for s in seeds}
    req_map: dict[tuple[str, int], tuple[int, str]] = {}

    for seed in seeds:
        st = (seed.source_type or "").lower()
        w = pdf_window if st == "pdf" else window

        for offset in range(-w, w + 1):
            if offset == 0:
                continue
            neighbor_idx = seed.chunk_index + offset
            if neighbor_idx < 0:
                continue

            key = (seed.document_id, neighbor_idx)
            cur = req_map.get(key)
            if cur is None or seed.rerank_rank < cur[0]:
                req_map[key] = (seed.rerank_rank, seed.chunk_id)

    neighbor_requests = [
        (doc_id, idx, seed_id)
        for (doc_id, idx), (_, seed_id) in req_map.items()
    ]

    if not neighbor_requests:
        return seed_expanded, []

    # Fetch neighbors
    neighbors = await chunk_repo.get_neighbors_by_doc_indices(neighbor_requests)

    # Filter and build neighbor ExpandedChunks
    valid_neighbors: list[ExpandedChunk] = []
    seen_ids = set(seed_set) | already_have_ids

    for n in neighbors:
        if n.chunk_id in seen_ids:
            continue
        if len(n.text) < min_chars:
            continue
        seen_ids.add(n.chunk_id)

        valid_neighbors.append(ExpandedChunk(
            chunk_id=n.chunk_id,
            document_id=n.document_id,
            chunk_index=n.chunk_index,
            rerank_score=0.0,
            rerank_rank=-1,
            vector_score=0.0,
            source_type=n.source_type,
            is_neighbor=True,
            neighbor_of=n.seed_chunk_id,
        ))

    # Compute best seed rank per document
    best_doc_rank: dict[str, int] = {}
    for s in seeds:
        doc_id = s.document_id
        best_doc_rank[doc_id] = min(best_doc_rank.get(doc_id, 10**9), s.rerank_rank)

    all_expanded = seed_expanded + valid_neighbors

    # Sort: best doc first, then by chunk_index within doc
    all_expanded.sort(key=lambda x: (
        best_doc_rank.get(x.document_id, 10**9),
        x.document_id,
        x.chunk_index,
    ))

    # Soft cap: preserves all seeds, trims neighbors only
    if len(all_expanded) > max_total:
        kept_seeds = [e for e in all_expanded if not e.is_neighbor]

        if len(kept_seeds) >= max_total:
            all_expanded = kept_seeds
        else:
            kept_neighbors = [e for e in all_expanded if e.is_neighbor]
            remaining = max_total - len(kept_seeds)
            all_expanded = kept_seeds + kept_neighbors[:remaining]

        all_expanded.sort(key=lambda x: (
            best_doc_rank.get(x.document_id, 10**9),
            x.document_id,
            x.chunk_index,
        ))

    new_ids = [
        e.chunk_id
        for e in all_expanded
        if e.is_neighbor and e.chunk_id not in already_have_ids
    ]

    return all_expanded, new_ids
```

## Repository Method

**File:** `app/repositories/chunks.py`

```python
import uuid
from dataclasses import dataclass

@dataclass
class NeighborChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_type: str | None
    seed_chunk_id: str

class ChunkRepository:

    async def get_neighbors_by_doc_indices(
        self,
        requests: list[tuple[str, int, str]],
    ) -> list[NeighborChunk]:
        """
        Fetch chunks by (document_id, chunk_index) pairs with seed attribution.
        """
        if not requests:
            return []

        doc_ids = [uuid.UUID(r[0]) if isinstance(r[0], str) else r[0] for r in requests]
        indices = [r[1] for r in requests]
        seed_ids = [uuid.UUID(r[2]) if isinstance(r[2], str) else r[2] for r in requests]

        query = """
            WITH req(document_id, chunk_index, seed_chunk_id) AS (
                SELECT * FROM unnest($1::uuid[], $2::int[], $3::uuid[])
            )
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.chunk_index,
                c.text,
                d.source_type,
                req.seed_chunk_id
            FROM req
            JOIN chunks c ON c.document_id = req.document_id
                         AND c.chunk_index = req.chunk_index
            JOIN documents d ON d.id = c.document_id
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, doc_ids, indices, seed_ids)

        return [
            NeighborChunk(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                chunk_index=row["chunk_index"],
                text=row["text"],
                source_type=row["source_type"],
                seed_chunk_id=str(row["seed_chunk_id"]),
            )
            for row in rows
        ]
```

## Workspace Config Schema

**File:** `app/schemas/workspace_config.py`

```python
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Literal

class CrossEncoderConfig(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    max_text_chars: int = Field(default=2000, ge=100, le=10000)
    batch_size: int = Field(default=16, ge=1, le=64)
    max_concurrent: int = Field(default=2, ge=1, le=4)

class LLMRerankConfig(BaseModel):
    model: str | None = None

class RerankConfig(BaseModel):
    enabled: bool = False
    method: Literal["cross_encoder", "llm"] = "cross_encoder"
    candidates_k: int = Field(default=50, ge=10, le=200)
    final_k: int = Field(default=10, ge=1, le=50)
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    llm: LLMRerankConfig = Field(default_factory=LLMRerankConfig)

    @model_validator(mode="after")
    def validate_k_values(self):
        if self.final_k > self.candidates_k:
            self.final_k = self.candidates_k
        return self

class NeighborConfig(BaseModel):
    enabled: bool = True
    window: int = Field(default=1, ge=0, le=3)
    pdf_window: int = Field(default=2, ge=0, le=5)
    min_chars: int = Field(default=200, ge=0)
    max_total: int = Field(default=20, ge=1, le=50)

class RetrievalConfig(BaseModel):
    top_k: int = Field(default=8, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)

class ChunkingConfig(BaseModel):
    size: int = Field(default=512, ge=64, le=2048)
    overlap: int = Field(default=50, ge=0, le=256)

class WorkspaceConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    neighbor: NeighborConfig = Field(default_factory=NeighborConfig)

    model_config = ConfigDict(extra="allow")
```

**Example workspace config:**

```json
{
  "rerank": {
    "enabled": true,
    "method": "cross_encoder",
    "candidates_k": 50,
    "final_k": 10
  },
  "neighbor": {
    "enabled": true,
    "window": 1,
    "pdf_window": 2
  }
}
```

## Response Schema

**File:** `app/schemas/query.py`

```python
class ChunkResultDebug(BaseModel):
    vector_score: float
    rerank_score: float | None    # None if no rerank ran
    rerank_rank: int | None       # 0-based, None if no rerank
    is_neighbor: bool
    neighbor_of: str | None

class ChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    # ... existing fields
    debug: ChunkResultDebug | None = None

class QueryMeta(BaseModel):
    # Timing
    embed_ms: int
    search_ms: int
    rerank_ms: int | None
    expand_ms: int | None
    answer_ms: int | None
    total_ms: int

    # Counts
    candidates_searched: int
    seeds_count: int              # after rerank or fallback
    chunks_after_expand: int
    neighbors_added: int

    # Rerank info
    rerank_enabled: bool
    rerank_method: Literal["cross_encoder", "llm"] | None
    rerank_model: str | None

    # Neighbor info
    neighbor_enabled: bool

class QueryResponse(BaseModel):
    results: list[ChunkResult]
    answer: str | None = None
    meta: QueryMeta
```

**Debug field mapping:**
- Internal `rerank_score=0.0` maps to API `rerank_score=None` when `rerank_enabled=False`
- This avoids "0.0 looks like a real model score" confusion

## Lifespan Integration

**File:** `app/main.py`

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_pool, _qdrant_client

    # --- Startup ---
    _qdrant_client = AsyncQdrantClient(...)
    _db_pool = await asyncpg.create_pool(...)

    # Optional: warm up reranker for fast first query
    await warmup_reranker()

    logger.info("Application startup complete")

    yield

    # --- Shutdown ---
    from app.services import reranker as reranker_module
    if reranker_module._cross_encoder_reranker is not None:
        reranker_module._cross_encoder_reranker.close(wait=True)
        reranker_module._cross_encoder_reranker = None
        logger.info("CrossEncoderReranker closed")

    if reranker_module._llm_reranker is not None:
        reranker_module._llm_reranker.close()
        reranker_module._llm_reranker = None

    await _qdrant_client.close()
    await _db_pool.close()

    logger.info("Application shutdown complete")


async def warmup_reranker():
    """Pre-load cross-encoder model using the singleton."""
    from app.services.reranker import get_reranker, RerankCandidate

    config = {"enabled": True, "cross_encoder": {"device": "cuda"}}
    reranker = get_reranker(config)

    if reranker and reranker.method == "cross_encoder":
        dummy = [RerankCandidate(
            chunk_id="warmup",
            document_id="warmup",
            chunk_index=0,
            text="warmup text for model loading",
            vector_score=1.0,
            workspace_id="warmup",
        )]
        await reranker.rerank("warmup query", dummy, top_k=1)
        logger.info(f"Reranker warmed up: {reranker.model_id}")
```

## Testing Strategy

### Unit Tests (mock model, fast)

**File:** `tests/unit/test_reranker.py`

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

class TestCrossEncoderReranker:

    @pytest.fixture
    def mock_cross_encoder(self):
        with patch("app.services.reranker.CrossEncoder") as mock:
            instance = Mock()
            instance.predict.return_value = [0.9, 0.7, 0.8, 0.6, 0.5]
            mock.return_value = instance
            yield mock

    async def test_rerank_returns_top_k(self, mock_cross_encoder, candidates):
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        results = await reranker.rerank("test query", candidates, top_k=3)

        assert len(results) == 3
        reranker.close()

    async def test_truncates_to_max_candidates(self, mock_cross_encoder):
        candidates = [make_candidate(i, vector_score=i/300) for i in range(300)]
        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        await reranker.rerank("test", candidates, top_k=10)

        # Should only score MAX_CANDIDATES (200)
        call_args = mock_cross_encoder.return_value.predict.call_args
        assert len(call_args[0][0]) <= 200
        reranker.close()

    async def test_deterministic_ordering(self, mock_cross_encoder):
        # Same scores for all -> should order by vector_score then chunk_id
        mock_cross_encoder.return_value.predict.return_value = [0.5, 0.5, 0.5]
        candidates = [
            make_candidate("c", vector_score=0.8),
            make_candidate("a", vector_score=0.8),
            make_candidate("b", vector_score=0.9),
        ]

        config = {"device": "cpu"}
        reranker = CrossEncoderReranker(config)

        results = await reranker.rerank("test", candidates, top_k=3)

        # b first (higher vector), then a before c (alphabetical)
        assert [r.chunk_id for r in results] == ["b", "a", "c"]
        reranker.close()


class TestGetReranker:

    def test_disabled_returns_none(self):
        assert get_reranker({"enabled": False}) is None

    def test_singleton_same_instance(self):
        config = {"enabled": True, "method": "cross_encoder"}
        r1 = get_reranker(config)
        r2 = get_reranker(config)
        assert r1 is r2
```

**File:** `tests/unit/test_neighbor_expansion.py`

```python
class TestExpandNeighbors:

    async def test_disabled_returns_seeds_only(self, mock_repo):
        seeds = [make_seed("s1", chunk_index=5)]
        expanded, new_ids = await expand_neighbors(
            seeds, mock_repo, {"enabled": False}
        )

        assert len(expanded) == 1
        assert not expanded[0].is_neighbor
        assert new_ids == []

    async def test_best_seed_wins_attribution(self, mock_repo):
        seeds = [
            make_seed("s1", doc="d1", idx=5, rank=1),
            make_seed("s2", doc="d1", idx=7, rank=0),
        ]

        expanded, _ = await expand_neighbors(
            seeds, mock_repo, {"enabled": True, "window": 1}
        )

        # Chunk 6 is between both; s2 (rank 0) should win
        neighbor_6 = next(e for e in expanded if e.chunk_index == 6)
        assert neighbor_6.neighbor_of == "s2"

    async def test_soft_cap_preserves_all_seeds(self, mock_repo):
        seeds = [make_seed(f"s{i}", idx=i*10) for i in range(5)]

        expanded, _ = await expand_neighbors(
            seeds, mock_repo, {"enabled": True, "max_total": 3}
        )

        seed_count = sum(1 for e in expanded if not e.is_neighbor)
        assert seed_count == 5  # All preserved despite max_total=3
```

### Integration Tests (real DB, marked slow)

**File:** `tests/integration/test_rerank_pipeline.py`

```python
@pytest.mark.requires_db
class TestRerankPipeline:

    async def test_full_pipeline_rerank_enabled(self, db_pool, qdrant):
        # Ingest test docs -> query with rerank -> verify ordering
        pass

    async def test_fallback_when_disabled(self, db_pool, qdrant):
        # Query with rerank disabled -> verify rerank_score=None in response
        pass
```

### Slow Test (real model)

**File:** `tests/slow/test_real_reranker.py`

```python
@pytest.mark.slow
async def test_real_cross_encoder_inference():
    """Actually load BGE model and score pairs. Run with pytest -m slow."""
    config = {"model": "BAAI/bge-reranker-v2-m3", "device": "cpu"}
    reranker = CrossEncoderReranker(config)

    candidates = [
        make_candidate("a", text="Python is a programming language"),
        make_candidate("b", text="The weather is sunny today"),
    ]

    results = await reranker.rerank("What is Python?", candidates, top_k=2)

    # "Python is a programming language" should rank higher
    assert results[0].chunk_id == "a"
    reranker.close()
```

## Implementation Checklist

1. **Create files:**
   - [ ] `app/services/reranker.py` - BaseReranker, CrossEncoderReranker, LLMReranker, get_reranker
   - [ ] `app/services/neighbor_expansion.py` - ExpandedChunk, expand_neighbors
   - [ ] `app/schemas/workspace_config.py` - Pydantic config models
   - [ ] `app/repositories/chunks.py` - Add get_neighbors_by_doc_indices method

2. **Modify files:**
   - [ ] `app/routers/query.py` - Integrate rerank + neighbor expansion
   - [ ] `app/schemas/query.py` - Add debug fields, update QueryMeta
   - [ ] `app/main.py` - Add reranker lifecycle (warmup + shutdown)
   - [ ] `requirements.txt` - Add sentence-transformers

3. **Tests:**
   - [ ] `tests/unit/test_reranker.py`
   - [ ] `tests/unit/test_neighbor_expansion.py`
   - [ ] `tests/integration/test_rerank_pipeline.py`
   - [ ] `tests/slow/test_real_reranker.py`

4. **Config:**
   - [ ] Update workspace config in DB for testing workspace
   - [ ] Document config options in admin UI

## Defaults Summary

| Setting | Default | Notes |
|---------|---------|-------|
| rerank.enabled | false | Opt-in per workspace |
| rerank.method | cross_encoder | LLM as fallback |
| rerank.candidates_k | 50 | Vector search limit |
| rerank.final_k | 10 | After rerank |
| cross_encoder.model | BAAI/bge-reranker-v2-m3 | Apache 2.0 |
| cross_encoder.device | cuda | GPU required |
| neighbor.enabled | true | Context expansion |
| neighbor.window | 1 | ±1 for non-PDF |
| neighbor.pdf_window | 2 | ±2 for PDF |
| neighbor.max_total | 20 | Soft cap (seeds sacred) |
