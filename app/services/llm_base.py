"""Base LLM client interface and shared types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypedDict

import structlog

logger = structlog.get_logger(__name__)

# Message types
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    """Chat message structure."""

    role: Role
    content: str


# Response types
@dataclass
class LLMResponse:
    """Response from an LLM generation call."""

    text: str
    model: str
    provider: str
    usage: dict | None = None  # {input_tokens, output_tokens, cache_*}
    latency_ms: float | None = None


@dataclass
class RankedChunk:
    """A chunk with its relevance score and rank."""

    chunk: dict
    score: float
    rank: int
    original_index: int  # For stable tiebreaker


# Errors
class LLMError(Exception):
    """Error from LLM provider."""

    def __init__(self, message: str, provider: str, model: str | None = None):
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMNotConfiguredError(Exception):
    """Raised when LLM generation is requested but no provider is configured."""

    pass


# Base client
class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, answer_model: str, rerank_model: str | None):
        """
        Initialize the LLM client.

        Args:
            answer_model: Model to use for answer generation
            rerank_model: Model to use for reranking (None = use answer_model)
        """
        self.answer_model = answer_model
        self._rerank_model = rerank_model
        self.provider: str = "base"  # Override in subclasses

    @property
    def effective_rerank_model(self) -> str:
        """Get the effective rerank model (falls back to answer_model)."""
        model = (self._rerank_model or "").strip() or None
        return model or self.answer_model

    @abstractmethod
    async def generate(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to answer_model)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with text, model, provider, usage, latency

        Raises:
            LLMError: On provider API errors
        """
        ...

    async def generate_text(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> str:
        """
        Convenience method: generate text from a simple prompt.

        Args:
            prompt: User prompt
            system: Optional system prompt
            model: Model to use
            max_tokens: Maximum tokens

        Returns:
            Generated text string
        """
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
        return response.text

    async def generate_answer(
        self,
        question: str,
        chunks: list[dict],
        max_context_tokens: int = 8000,
    ) -> LLMResponse:
        """
        Generate an answer with citations from context chunks.

        Args:
            question: User's question
            chunks: List of chunks with 'content' and optional 'title'/'source_url'
            max_context_tokens: Maximum tokens for context

        Returns:
            LLMResponse with answer text
        """
        # Build context with citation numbers
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = chunk.get("title", chunk.get("source_url", f"Source {i}"))
            context_parts.append(f"[{i}] {source_info}:\n{chunk['content']}\n")

        context = "\n".join(context_parts)

        system = """You are a knowledgeable analyst assistant.
Answer questions based on the provided context. Always cite your sources using
the format [1], [2], etc. to reference the numbered sources.

If the context doesn't contain enough information to fully answer the question,
say so and provide what information you can from the available sources.

Be concise but thorough. Focus on actionable insights when relevant."""

        prompt = f"""Context:
{context}

Question: {question}

Answer the question based on the context above. Use citations like [1], [2] to reference sources."""

        logger.info(
            "Generating answer",
            question=question[:50],
            num_chunks=len(chunks),
            model=self.answer_model,
        )

        return await self.generate(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=self.answer_model,
        )

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[RankedChunk]:
        """
        Rerank chunks by relevance to query using LLM.

        Args:
            query: Original query
            chunks: Chunks to rerank (must have 'content' key)
            top_k: Number of top chunks to return

        Returns:
            List of RankedChunk sorted by score desc, tiebreaker = original order
        """
        if len(chunks) <= top_k:
            # No need to rerank if we're returning all
            return [
                RankedChunk(chunk=c, score=1.0, rank=i, original_index=i)
                for i, c in enumerate(chunks)
            ]

        # Build prompt for reranking
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = chunk["content"][:200]  # Truncate for prompt
            chunk_summaries.append(f"{i}. {summary}...")

        prompt = f"""Given the query: "{query}"

Rate the relevance of each chunk on a scale of 1-10.
Return only the chunk numbers in order of relevance (most relevant first).

Chunks:
{chr(10).join(chunk_summaries)}

Return the top {top_k} most relevant chunk numbers as a comma-separated list."""

        try:
            response = await self.generate(
                messages=[{"role": "user", "content": prompt}],
                model=self.effective_rerank_model,
                max_tokens=100,
            )

            # Parse response to get ranking
            import re

            numbers = [int(n) for n in re.findall(r"\d+", response.text)]
            # Filter valid indices and take top_k
            valid_indices = [n for n in numbers if 0 <= n < len(chunks)][:top_k]

            if valid_indices:
                return [
                    RankedChunk(
                        chunk=chunks[idx],
                        score=1.0 - (rank * 0.1),  # Decay score by rank
                        rank=rank,
                        original_index=idx,
                    )
                    for rank, idx in enumerate(valid_indices)
                ]
        except Exception as e:
            logger.warning("Failed to parse reranking response", error=str(e))

        # Fallback: return original top_k with stable ordering
        return [
            RankedChunk(chunk=c, score=1.0, rank=i, original_index=i)
            for i, c in enumerate(chunks[:top_k])
        ]
