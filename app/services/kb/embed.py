"""
KB-specific embedding adapter.

Wraps the base OllamaEmbedder with:
- Retry with binary split on batch failure
- Model ID and dimension tracking
- Normalized exception types
- Timeout handling
"""

import os
from dataclasses import dataclass
from typing import Optional

import httpx
import structlog

from app.services.embedder import OllamaEmbedder, get_embedder

logger = structlog.get_logger(__name__)

# Configuration
KB_EMBED_BATCH_SIZE = int(os.environ.get("KB_EMBED_BATCH_SIZE", "32"))
KB_EMBED_TIMEOUT = int(os.environ.get("KB_EMBED_TIMEOUT", "30"))
KB_EMBED_MAX_RETRIES = int(os.environ.get("KB_EMBED_MAX_RETRIES", "3"))
KB_EMBED_MIN_BATCH_SIZE = int(os.environ.get("KB_EMBED_MIN_BATCH_SIZE", "1"))


class EmbeddingError(Exception):
    """Base exception for embedding errors."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class EmbeddingTimeoutError(EmbeddingError):
    """Timeout during embedding."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)


class EmbeddingBatchError(EmbeddingError):
    """Batch embedding failed (may contain bad inputs)."""

    def __init__(self, message: str, failed_indices: list[int] = None):
        super().__init__(message, retryable=True)
        self.failed_indices = failed_indices or []


class EmbeddingServiceError(EmbeddingError):
    """Embedding service unavailable."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""

    vectors: list[list[float]]
    model_id: str
    vector_dim: int
    failed_indices: list[int]  # Indices that failed to embed


class KBEmbeddingAdapter:
    """
    KB-specific embedding adapter with retry and tracking.

    Features:
    - Batch embedding with configurable size
    - Retry with binary split to isolate bad inputs
    - Model ID and dimension tracking
    - Timeout handling
    """

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        batch_size: int = KB_EMBED_BATCH_SIZE,
        timeout: int = KB_EMBED_TIMEOUT,
        max_retries: int = KB_EMBED_MAX_RETRIES,
        min_batch_size: int = KB_EMBED_MIN_BATCH_SIZE,
    ):
        """
        Initialize KB embedding adapter.

        Args:
            embedder: Base embedder (default: get_embedder())
            batch_size: Batch size for embedding requests
            timeout: Request timeout in seconds
            max_retries: Max retries per batch
            min_batch_size: Minimum batch size before giving up
        """
        self._embedder = embedder
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_batch_size = min_batch_size
        self._model_id: Optional[str] = None
        self._vector_dim: Optional[int] = None

    @property
    def embedder(self) -> OllamaEmbedder:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def model_id(self) -> str:
        """Get model ID (e.g., 'nomic-embed-text')."""
        if self._model_id is None:
            self._model_id = self.embedder.model
        return self._model_id

    async def get_vector_dim(self) -> int:
        """Get vector dimension for current model."""
        if self._vector_dim is None:
            self._vector_dim = await self.embedder.get_dimension()
        return self._vector_dim

    def get_collection_name(self, base_name: str = "trading_kb_trials") -> str:
        """
        Generate versioned collection name.

        Format: {base_name}__{model_id}__{dim}
        Example: trading_kb_trials__nomic-embed-text__768

        Args:
            base_name: Base collection name

        Returns:
            Versioned collection name
        """
        model_slug = self.model_id.replace("/", "_").replace(":", "_")
        dim = self._vector_dim or "unknown"
        return f"{base_name}__{model_slug}__{dim}"

    async def embed_texts(
        self,
        texts: list[str],
        skip_failures: bool = True,
    ) -> EmbeddingResult:
        """
        Embed a list of texts with retry and failure isolation.

        Args:
            texts: List of texts to embed
            skip_failures: If True, skip failed texts; if False, raise on failure

        Returns:
            EmbeddingResult with vectors and failed indices

        Raises:
            EmbeddingError: If embedding fails and skip_failures=False
        """
        if not texts:
            return EmbeddingResult(
                vectors=[],
                model_id=self.model_id,
                vector_dim=await self.get_vector_dim(),
                failed_indices=[],
            )

        vectors: list[list[float]] = []
        failed_indices: list[int] = []

        # Process in batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch = texts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            try:
                batch_vectors = await self._embed_batch_with_retry(
                    batch, batch_indices
                )
                vectors.extend(batch_vectors)
            except EmbeddingError as e:
                if skip_failures:
                    # Mark all in batch as failed, use empty vectors
                    for idx in batch_indices:
                        failed_indices.append(idx)
                        vectors.append([])
                    logger.warning(
                        "Batch failed, skipping",
                        batch_start=batch_start,
                        batch_size=len(batch),
                        error=e.message,
                    )
                else:
                    raise

        return EmbeddingResult(
            vectors=vectors,
            model_id=self.model_id,
            vector_dim=await self.get_vector_dim(),
            failed_indices=failed_indices,
        )

    async def _embed_batch_with_retry(
        self,
        texts: list[str],
        indices: list[int],
        retry_count: int = 0,
    ) -> list[list[float]]:
        """
        Embed batch with retry and binary split on failure.

        If a batch fails:
        1. Retry up to max_retries
        2. If still failing, split batch in half and try each half
        3. Continue splitting until min_batch_size reached

        Args:
            texts: Texts to embed
            indices: Original indices (for logging)
            retry_count: Current retry count

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If all retries exhausted
        """
        try:
            return await self._embed_batch_raw(texts)

        except httpx.TimeoutException as e:
            if retry_count < self.max_retries:
                logger.warning(
                    "Batch timeout, retrying",
                    batch_size=len(texts),
                    retry=retry_count + 1,
                )
                return await self._embed_batch_with_retry(
                    texts, indices, retry_count + 1
                )

            # Try binary split
            if len(texts) > self.min_batch_size:
                return await self._binary_split_embed(texts, indices)

            raise EmbeddingTimeoutError(
                f"Timeout embedding batch of {len(texts)} texts after {retry_count} retries"
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                # Server error - retry
                if retry_count < self.max_retries:
                    logger.warning(
                        "Server error, retrying",
                        status=e.response.status_code,
                        retry=retry_count + 1,
                    )
                    return await self._embed_batch_with_retry(
                        texts, indices, retry_count + 1
                    )
                raise EmbeddingServiceError(
                    f"Embedding service error: {e.response.status_code}"
                )

            # Client error (4xx) - likely bad input, try binary split
            if len(texts) > self.min_batch_size:
                return await self._binary_split_embed(texts, indices)

            raise EmbeddingBatchError(
                f"Batch embedding failed: {e.response.status_code}",
                failed_indices=indices,
            )

        except Exception as e:
            # Unexpected error
            if retry_count < self.max_retries:
                logger.warning(
                    "Unexpected error, retrying",
                    error=str(e),
                    retry=retry_count + 1,
                )
                return await self._embed_batch_with_retry(
                    texts, indices, retry_count + 1
                )

            raise EmbeddingError(f"Embedding failed: {str(e)}", retryable=False)

    async def _binary_split_embed(
        self,
        texts: list[str],
        indices: list[int],
    ) -> list[list[float]]:
        """
        Split batch in half and embed each half separately.

        Used to isolate problematic inputs.
        """
        if len(texts) <= self.min_batch_size:
            raise EmbeddingBatchError(
                f"Cannot split batch further (size={len(texts)})",
                failed_indices=indices,
            )

        mid = len(texts) // 2
        left_texts, right_texts = texts[:mid], texts[mid:]
        left_indices, right_indices = indices[:mid], indices[mid:]

        logger.debug(
            "Binary split batch",
            original_size=len(texts),
            left_size=len(left_texts),
            right_size=len(right_texts),
        )

        left_vectors = await self._embed_batch_with_retry(left_texts, left_indices)
        right_vectors = await self._embed_batch_with_retry(right_texts, right_indices)

        return left_vectors + right_vectors

    async def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        """
        Raw batch embedding without retry logic.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.embedder.base_url}/api/embed",
                json={
                    "model": self.model_id,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}"
                )

            return embeddings

    async def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        result = await self.embed_texts([text], skip_failures=False)
        return result.vectors[0]

    async def health_check(self) -> bool:
        """Check if embedding service is healthy."""
        return await self.embedder.health_check()


# Module-level instance
_kb_embedder: Optional[KBEmbeddingAdapter] = None


def get_kb_embedder() -> KBEmbeddingAdapter:
    """Get or create KB embedding adapter instance."""
    global _kb_embedder
    if _kb_embedder is None:
        _kb_embedder = KBEmbeddingAdapter()
    return _kb_embedder
