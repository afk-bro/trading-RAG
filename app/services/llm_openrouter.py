"""OpenRouter LLM client for accessing multiple providers via proxy."""

import time

import httpx
import structlog

from app.services.llm_base import BaseLLMClient, LLMError, LLMResponse, Message

logger = structlog.get_logger(__name__)


class OpenRouterLLMClient(BaseLLMClient):
    """LLM client using OpenRouter API (proxy to multiple providers)."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str,
        answer_model: str,
        rerank_model: str | None,
        timeout: int = 60,
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key
            answer_model: Model for answer generation
            rerank_model: Model for reranking (None = use answer_model)
            timeout: Request timeout in seconds
        """
        super().__init__(answer_model, rerank_model)
        self.api_key = api_key
        self.timeout = timeout
        self.provider = "openrouter"

    async def generate(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """
        Generate a response using the OpenRouter API.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to answer_model)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with text, model, provider, usage, latency

        Raises:
            LLMError: On OpenRouter API errors
        """
        model = model or self.answer_model

        # OpenRouter accepts standard OpenAI-style messages
        api_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://rag-core.local",
                        "X-Title": "RAG Core Pipeline",
                    },
                    json={
                        "model": model,
                        "messages": api_messages,
                        "max_tokens": max_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()
            latency_ms = (time.perf_counter() - start) * 1000
        except httpx.HTTPStatusError as e:
            logger.error(
                "OpenRouter API HTTP error",
                model=model,
                status_code=e.response.status_code,
                error=str(e),
            )
            raise LLMError(
                f"OpenRouter API error: {e.response.status_code} - {e}",
                provider=self.provider,
                model=model,
            )
        except httpx.RequestError as e:
            logger.error(
                "OpenRouter request error",
                model=model,
                error=str(e),
            )
            raise LLMError(
                f"OpenRouter request failed: {e}",
                provider=self.provider,
                model=model,
            )

        # Extract response text
        choices = data.get("choices", [])
        if not choices:
            raise LLMError(
                "No response from OpenRouter",
                provider=self.provider,
                model=model,
            )

        text = choices[0].get("message", {}).get("content", "")

        # Extract usage if present
        usage_data = data.get("usage", {})
        usage = None
        if usage_data:
            usage = {
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
            }

        # Get actual model used (may differ from requested)
        actual_model = data.get("model", model)

        logger.debug(
            "OpenRouter generation complete",
            model=actual_model,
            input_tokens=usage.get("input_tokens") if usage else None,
            output_tokens=usage.get("output_tokens") if usage else None,
            latency_ms=round(latency_ms, 2),
        )

        return LLMResponse(
            text=text,
            model=actual_model,
            provider=self.provider,
            usage=usage,
            latency_ms=latency_ms,
        )
