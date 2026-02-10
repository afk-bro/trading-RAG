"""Embedding service using Ollama."""

from typing import Optional

import httpx
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


class OllamaEmbedder:
    """Embedding service using local Ollama."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize Ollama embedder.

        Args:
            base_url: Ollama base URL (default from settings)
            model: Model name (default from settings)
            timeout: Request timeout in seconds
            batch_size: Batch size for embedding requests
        """
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.embed_model
        self.timeout = timeout or settings.ollama_timeout
        self.batch_size = batch_size or settings.embed_batch_size
        self._dimension: Optional[int] = None
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def get_dimension(self) -> int:
        """
        Get embedding dimension for the current model.

        Caches the dimension after first call.
        """
        if self._dimension is not None:
            return self._dimension

        # Embed a sample text to detect dimension
        embedding = await self.embed("test")
        self._dimension = len(embedding)
        logger.info(
            "Detected embedding dimension",
            model=self.model,
            dimension=self._dimension,
        )
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = await self._client.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text,
            },
        )
        response.raise_for_status()
        data = response.json()

        # Ollama returns embeddings in "embeddings" array
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]

        raise ValueError("No embedding returned from Ollama")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]  # noqa: E203

            response = await self._client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": batch,
                },
            )
            response.raise_for_status()
            data = response.json()

            batch_embeddings = data.get("embeddings", [])
            embeddings.extend(batch_embeddings)

            logger.debug(
                "Embedded batch",
                batch_size=len(batch),
                total_embedded=len(embeddings),
                total_remaining=len(texts) - len(embeddings),
            )

        return embeddings

    async def health_check(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return self.model in models
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


# Singleton instance
_embedder: Optional[OllamaEmbedder] = None


def get_embedder() -> OllamaEmbedder:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbedder()
    return _embedder
