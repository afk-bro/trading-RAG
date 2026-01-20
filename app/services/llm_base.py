"""Base LLM client interface and shared types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypedDict

import structlog
import tiktoken

logger = structlog.get_logger(__name__)

# Token counter for context truncation
_token_encoding = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base encoding."""
    return len(_token_encoding.encode(text))

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


# ===========================================
# Errors - Provider-agnostic exception hierarchy
# Providers should map their errors to these in their adapters
# ===========================================


class LLMError(Exception):
    """Base error from LLM provider."""

    def __init__(self, message: str, provider: str, model: str | None = None):
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMNotConfiguredError(Exception):
    """Raised when LLM generation is requested but no provider is configured."""


class LLMTimeoutError(LLMError):
    """Request timed out waiting for LLM response."""

    def __init__(
        self,
        message: str = "LLM request timed out",
        provider: str = "unknown",
        model: str | None = None,
        timeout_seconds: float | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, provider, model)


class LLMRateLimitError(LLMError):
    """Rate limited by LLM provider."""

    def __init__(
        self,
        message: str = "Rate limited by LLM provider",
        provider: str = "unknown",
        model: str | None = None,
        retry_after_seconds: int | None = None,
    ):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message, provider, model)


class LLMAPIError(LLMError):
    """General API error from LLM provider (non-rate-limit)."""

    def __init__(
        self,
        message: str = "LLM provider API error",
        provider: str = "unknown",
        model: str | None = None,
        status_code: int | None = None,
    ):
        self.status_code = status_code
        super().__init__(message, provider, model)


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

    # Grounding contract - enforces RAG discipline
    GROUNDING_CONTRACT = """GROUNDING CONTRACT (you must follow these rules):
- Use ONLY the provided context chunks as your source of truth.
- If the answer is not explicitly supported by the context, say: "The provided context does not specify this."  # noqa: E501
- Do NOT use general knowledge, assumptions, or outside facts.
- If the question asks for definitions and the context lacks a definition, state that clearly.
- Cite sources using [1], [2], etc. to reference the numbered context chunks.
- Prefer accuracy over completeness. It is OK to answer partially.
- Be honest about what the context does and does not contain."""

    async def generate_answer(
        self,
        question: str,
        chunks: list[dict],
        max_context_tokens: int = 8000,
        model: str | None = None,
    ) -> LLMResponse:
        """
        Generate an answer with citations from context chunks.

        Uses grounding contract to ensure answers are strictly based on
        provided context, not general knowledge.

        Args:
            question: User's question
            chunks: List of chunks with 'content' and optional 'title'/'source_url'
            max_context_tokens: Maximum tokens for context
            model: Override model for answer generation (defaults to answer_model)

        Returns:
            LLMResponse with answer text
        """
        # Build context with citation numbers, truncating by whole chunks
        # to stay within max_context_tokens
        context_parts = []
        total_tokens = 0
        chunks_included = 0

        for i, chunk in enumerate(chunks, 1):
            source_info = chunk.get("title", chunk.get("source_url", f"Source {i}"))
            locator = chunk.get("locator_label", "")
            if locator:
                source_info = f"{source_info} ({locator})"
            chunk_text = f"[{i}] {source_info}:\n{chunk['content']}\n"
            chunk_tokens = _count_tokens(chunk_text)

            # Check if adding this chunk would exceed the limit
            if total_tokens + chunk_tokens > max_context_tokens:
                logger.info(
                    "Truncating context to respect max_context_tokens",
                    max_context_tokens=max_context_tokens,
                    chunks_included=chunks_included,
                    chunks_total=len(chunks),
                    tokens_used=total_tokens,
                )
                break

            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
            chunks_included += 1

        context = "\n".join(context_parts)

        system = f"""{self.GROUNDING_CONTRACT}

You are a research assistant analyzing provided documents.
Your job is to answer questions using ONLY the context chunks below.
Always cite your sources using [1], [2], etc."""

        prompt = f"""Context chunks:
{context}

Question: {question}

Provide your response in this format:

**Answer:**
[Your answer based strictly on the context, with citations]

**Supported by context:**
- [Key facts from the context that support your answer]

**Not specified in context:**
- [Aspects of the question the context does not address, if any]"""

        effective_model = model or self.answer_model
        logger.info(
            "Generating grounded answer",
            question=question[:50],
            chunks_provided=len(chunks),
            chunks_included=chunks_included,
            context_tokens=total_tokens,
            max_context_tokens=max_context_tokens,
            model=effective_model,
        )

        return await self.generate(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model=effective_model,
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
