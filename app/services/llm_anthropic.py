"""Anthropic LLM client using the official SDK."""

import time

import structlog
from anthropic import APIError, AsyncAnthropic

from app.services.llm_base import BaseLLMClient, LLMError, LLMResponse, Message

logger = structlog.get_logger(__name__)


class AnthropicLLMClient(BaseLLMClient):
    """LLM client using the Anthropic API directly."""

    def __init__(
        self,
        api_key: str,
        answer_model: str,
        rerank_model: str | None,
        timeout: int = 60,
    ):
        """
        Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key
            answer_model: Model for answer generation
            rerank_model: Model for reranking (None = use answer_model)
            timeout: Request timeout in seconds
        """
        super().__init__(answer_model, rerank_model)
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.provider = "anthropic"

    async def generate(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """
        Generate a response using the Anthropic API.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to answer_model)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with text, model, provider, usage, latency

        Raises:
            LLMError: On Anthropic API errors
        """
        model = model or self.answer_model

        # Separate system messages from chat messages
        # Anthropic requires system to be passed separately, not as a message
        system_parts: list[str] = []
        chat_messages: list[dict] = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

        system = "\n\n".join(system_parts) if system_parts else ""

        try:
            start = time.perf_counter()
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=chat_messages,
            )
            latency_ms = (time.perf_counter() - start) * 1000
        except APIError as e:
            logger.error(
                "Anthropic API error",
                model=model,
                error=str(e),
            )
            raise LLMError(
                f"Anthropic API error: {e}",
                provider=self.provider,
                model=model,
            )

        # Extract text from all text blocks (handles multi-block responses)
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

        # Build usage dict (include cache fields if present)
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        # Optional cache token fields (may be present in some SDK versions)
        if hasattr(response.usage, "cache_creation_input_tokens"):
            usage["cache_creation_input_tokens"] = (
                response.usage.cache_creation_input_tokens
            )
        if hasattr(response.usage, "cache_read_input_tokens"):
            usage["cache_read_input_tokens"] = response.usage.cache_read_input_tokens

        logger.debug(
            "Anthropic generation complete",
            model=response.model,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            latency_ms=round(latency_ms, 2),
        )

        return LLMResponse(
            text=text,
            model=response.model,
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
        )
