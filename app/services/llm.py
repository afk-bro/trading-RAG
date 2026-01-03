"""LLM service using OpenRouter API."""

from typing import Optional

import httpx
import structlog

from app.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class OpenRouterLLM:
    """LLM service using OpenRouter API."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (default from settings)
            model: Default model to use
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.answer_model
        self.timeout = timeout or settings.openrouter_timeout

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use (overrides default)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://trading-rag.local",
                    "X-Title": "Trading RAG Pipeline",
                },
                json={
                    "model": model or self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract response text
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")

            raise ValueError("No response from LLM")

    async def generate_answer(
        self,
        question: str,
        context_chunks: list[dict],
        max_context_tokens: int = 8000,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate an answer with citations from context chunks.

        Args:
            question: User's question
            context_chunks: List of chunks with content and metadata
            max_context_tokens: Maximum tokens for context
            model: Model to use

        Returns:
            Answer with citations
        """
        # Build context with citation numbers
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = chunk.get("title", chunk.get("source_url", f"Source {i}"))
            context_parts.append(f"[{i}] {source_info}:\n{chunk['content']}\n")

        context = "\n".join(context_parts)

        system_prompt = """You are a knowledgeable financial analyst assistant.
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
            num_chunks=len(context_chunks),
            model=model or self.model,
        )

        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
        )

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank chunks using LLM as a reranker.

        Args:
            query: Original query
            chunks: Chunks to rerank
            top_k: Number of top chunks to return

        Returns:
            Reranked and filtered chunks
        """
        if len(chunks) <= top_k:
            return chunks

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

        response = await self.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.0,
        )

        # Parse response to get ranking
        try:
            # Extract numbers from response
            import re

            numbers = [int(n) for n in re.findall(r"\d+", response)]
            # Filter valid indices and take top_k
            valid_indices = [n for n in numbers if 0 <= n < len(chunks)][:top_k]

            if valid_indices:
                return [chunks[i] for i in valid_indices]
        except Exception as e:
            logger.warning("Failed to parse reranking response", error=str(e))

        # Fallback: return original top_k
        return chunks[:top_k]


# Singleton instance
_llm: Optional[OpenRouterLLM] = None


def get_llm() -> OpenRouterLLM:
    """Get or create LLM instance."""
    global _llm
    if _llm is None:
        _llm = OpenRouterLLM()
    return _llm
