"""
Cross-encoder and LLM reranking service.

Provides two reranking strategies:
- CrossEncoderReranker: Uses sentence-transformers CrossEncoder for fast, accurate reranking
- LLMReranker: Uses existing LLM client for reranking (fallback/A-B testing)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """Input candidate for reranking."""

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
    """Output from reranking."""

    chunk_id: str
    document_id: str
    chunk_index: int
    rerank_score: float  # 0.0 if no rerank ran
    rerank_rank: int  # 0-based
    vector_score: float
    source_type: str | None = None


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

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
        top_k: int,
    ) -> list[RerankResult]:
        """
        Rerank candidates by relevance to query.

        Args:
            query: User query
            candidates: Candidates from vector search
            top_k: Number of results to return

        Returns:
            Top K results sorted by rerank score
        """
        pass

    def close(self, wait: bool = True) -> None:
        """Cleanup resources. Override in subclasses as needed."""
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers."""

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    MAX_CANDIDATES = 200

    def __init__(self, config: dict):
        self._model_id = config.get("model", self.DEFAULT_MODEL)
        self._device = config.get("device", "cuda")
        self._max_text_chars = config.get("max_text_chars", 2000)
        self._batch_size = config.get("batch_size", 16)
        self._max_concurrent = config.get("max_concurrent", 2)

        self._model: Any | None = None  # CrossEncoder
        self._init_lock: asyncio.Lock | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)  # Single thread for CUDA

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

    async def _ensure_model(self) -> Any:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return self._model

        async with self._get_init_lock():
            if self._model is None:
                logger.info(f"Loading cross-encoder model: {self._model_id}")
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(
                    self._executor,
                    self._load_model,
                )
                logger.info(f"Cross-encoder model loaded on {self._device}")
        return self._model

    def _load_model(self) -> Any:
        """Load model synchronously (called from executor)."""
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self._model_id, device=self._device)

    def _score_sync(
        self,
        model: Any,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """Score query-text pairs synchronously."""
        pairs = [[query, t] for t in texts]
        scores = model.predict(pairs, batch_size=self._batch_size)
        if hasattr(scores, "tolist"):
            return scores.tolist()
        return list(scores)

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int,
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
                reverse=True,
            )[: self.MAX_CANDIDATES]

        model = await self._ensure_model()
        semaphore = self._get_semaphore()

        texts = [c.text[: self._max_text_chars] for c in candidates]

        async with semaphore:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                self._executor,
                self._score_sync,
                model,
                query,
                texts,
            )

        # Sort with fully deterministic tie-breaker
        scored = list(zip(candidates, scores))
        scored.sort(
            key=lambda x: (x[1], x[0].vector_score, x[0].chunk_id),
            reverse=True,
        )

        results = []
        for rank, (candidate, score) in enumerate(scored[:top_k]):
            results.append(
                RerankResult(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    chunk_index=candidate.chunk_index,
                    rerank_score=float(score),
                    rerank_rank=rank,
                    vector_score=candidate.vector_score,
                    source_type=candidate.source_type,
                )
            )

        return results

    def close(self, wait: bool = True) -> None:
        """Shutdown executor and release model."""
        self._executor.shutdown(wait=wait)
        self._model = None


class LLMReranker(BaseReranker):
    """LLM-based reranker wrapping existing BaseLLMClient.rerank()."""

    def __init__(self, config: dict):
        self._config = config

    @property
    def method(self) -> str:
        return "llm"

    @property
    def model_id(self) -> str | None:
        from app.services.llm_factory import get_llm

        llm = get_llm()
        return llm.effective_rerank_model if llm else None

    async def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        top_k: int,
    ) -> list[RerankResult]:
        from app.services.llm_factory import get_llm

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

        # Convert to format expected by BaseLLMClient.rerank()
        chunks = [{"chunk_id": c.chunk_id, "content": c.text} for c in candidates]
        ranked = await llm.rerank(query, chunks, top_k)

        # Build lookup map
        chunk_map = {c.chunk_id: c for c in candidates}
        results = []

        for rank, r in enumerate(ranked):
            # Handle both RankedChunk object and dict returns
            if hasattr(r, "chunk"):
                chunk_id = r.chunk.get("chunk_id")
                score = r.score
            else:
                chunk_id = r.get("chunk_id")
                score = r.get("score", 0.0)

            candidate = chunk_map.get(chunk_id)
            if candidate:
                results.append(
                    RerankResult(
                        chunk_id=chunk_id,
                        document_id=candidate.document_id,
                        chunk_index=candidate.chunk_index,
                        rerank_score=float(score),
                        rerank_rank=rank,
                        vector_score=candidate.vector_score,
                        source_type=candidate.source_type,
                    )
                )
            else:
                logger.debug(f"LLM returned unmapped chunk_id: {chunk_id}")

        return results

    def close(self, wait: bool = True) -> None:
        pass


# Module-level singletons
_cross_encoder_reranker: CrossEncoderReranker | None = None
_llm_reranker: LLMReranker | None = None


def get_reranker(config: dict) -> BaseReranker | None:
    """
    Get or create a reranker based on configuration.

    Args:
        config: Rerank configuration dict with:
            - enabled: bool
            - method: "cross_encoder" | "llm"
            - cross_encoder: {...} for CrossEncoderReranker config
            - llm: {...} for LLMReranker config

    Returns:
        Reranker instance or None if disabled
    """
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


def reset_rerankers() -> None:
    """Reset reranker singletons (for testing)."""
    global _cross_encoder_reranker, _llm_reranker

    if _cross_encoder_reranker is not None:
        _cross_encoder_reranker.close(wait=True)
        _cross_encoder_reranker = None

    if _llm_reranker is not None:
        _llm_reranker.close(wait=True)
        _llm_reranker = None
