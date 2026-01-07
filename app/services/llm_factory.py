"""LLM provider factory and status management."""

from dataclasses import dataclass
from typing import Literal

import structlog

from app.config import Settings, get_settings
from app.services.llm_base import BaseLLMClient, LLMNotConfiguredError

logger = structlog.get_logger(__name__)

# Type aliases
ProviderConfig = Literal["auto", "anthropic", "openrouter"]
ProviderResolved = Literal["anthropic", "openrouter"]


@dataclass
class LLMStatus:
    """LLM configuration status."""

    enabled: bool
    provider_config: ProviderConfig
    provider_resolved: ProviderResolved | None
    answer_model: str | None
    rerank_model_effective: str | None


# Module-level singletons
_llm_client: BaseLLMClient | None = None
_llm_status: LLMStatus | None = None
_initialized: bool = False


class LLMStartupError(Exception):
    """Raised when LLM is required but no provider key is configured."""

    pass


def _get_effective_rerank_model(settings: Settings) -> str:
    """Get the effective rerank model (with fallback to answer_model)."""
    model = (settings.rerank_model or "").strip() or None
    return model or settings.answer_model


def _resolve_provider(settings: Settings) -> tuple[ProviderResolved | None, str | None]:
    """
    Resolve which provider to use based on settings.

    Returns:
        (provider_resolved, api_key) tuple, or (None, None) if disabled
    """
    # Kill switch
    if not settings.llm_enabled:
        return None, None

    provider = settings.llm_provider

    # Helper to get Anthropic key (API key or OAuth token)
    def get_anthropic_key() -> str | None:
        key = (settings.anthropic_api_key or "").strip() or None
        if key:
            return key
        # Fall back to Claude Code OAuth token
        return (settings.claude_code_oauth_token or "").strip() or None

    # Explicit provider selection
    if provider == "anthropic":
        key = get_anthropic_key()
        if not key:
            raise LLMStartupError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY (or CLAUDE_CODE_OAUTH_TOKEN) not set"
            )
        return "anthropic", key

    if provider == "openrouter":
        key = (settings.openrouter_api_key or "").strip() or None
        if not key:
            raise LLMStartupError(
                "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY not set"
            )
        return "openrouter", key

    # Auto: prefer Anthropic, fall back to OpenRouter
    anthropic_key = get_anthropic_key()
    if anthropic_key:
        return "anthropic", anthropic_key

    openrouter_key = (settings.openrouter_api_key or "").strip() or None
    if openrouter_key:
        return "openrouter", openrouter_key

    # No keys available
    if settings.llm_required:
        raise LLMStartupError(
            "LLM_REQUIRED=true but no API key configured. "
            "Set ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN, or OPENROUTER_API_KEY"
        )

    return None, None


def _create_client(
    provider: ProviderResolved, api_key: str, settings: Settings
) -> BaseLLMClient:
    """Create the appropriate LLM client."""
    if provider == "anthropic":
        from app.services.llm_anthropic import AnthropicLLMClient

        return AnthropicLLMClient(
            api_key=api_key,
            answer_model=settings.answer_model,
            rerank_model=settings.rerank_model,
            timeout=settings.llm_timeout,
        )
    else:
        from app.services.llm_openrouter import OpenRouterLLMClient

        return OpenRouterLLMClient(
            api_key=api_key,
            answer_model=settings.answer_model,
            rerank_model=settings.rerank_model,
            timeout=settings.llm_timeout,
        )


def _initialize() -> None:
    """Initialize the LLM subsystem (idempotent)."""
    global _llm_client, _llm_status, _initialized

    if _initialized:
        return

    settings = get_settings()
    provider_config = settings.llm_provider

    try:
        provider_resolved, api_key = _resolve_provider(settings)
    except LLMStartupError:
        raise

    if provider_resolved and api_key:
        _llm_client = _create_client(provider_resolved, api_key, settings)
        _llm_status = LLMStatus(
            enabled=True,
            provider_config=provider_config,
            provider_resolved=provider_resolved,
            answer_model=settings.answer_model,
            rerank_model_effective=_get_effective_rerank_model(settings),
        )
        logger.info(
            "LLM initialized",
            provider_config=provider_config,
            provider_resolved=provider_resolved,
            answer_model=settings.answer_model,
            rerank_model_effective=_get_effective_rerank_model(settings),
            llm_enabled=True,
        )
    else:
        _llm_client = None
        _llm_status = LLMStatus(
            enabled=False,
            provider_config=provider_config,
            provider_resolved=None,
            answer_model=settings.answer_model,
            rerank_model_effective=_get_effective_rerank_model(settings),
        )
        logger.info(
            "LLM disabled",
            provider_config=provider_config,
            reason="no API key configured" if settings.llm_enabled else "kill switch",
        )

    _initialized = True


def get_llm_status() -> LLMStatus:
    """
    Get LLM configuration status.

    This parses config and resolves provider without creating a client,
    useful for checking availability before making requests.
    """
    _initialize()
    assert _llm_status is not None
    return _llm_status


def get_llm() -> BaseLLMClient | None:
    """
    Get the cached LLM client.

    Returns:
        BaseLLMClient if LLM is enabled and configured, None otherwise

    Raises:
        LLMStartupError: If LLM_REQUIRED=true and no key configured
    """
    _initialize()
    return _llm_client


def reset_llm() -> None:
    """
    Reset the LLM singleton (for testing).

    This allows re-initialization with different settings.
    """
    global _llm_client, _llm_status, _initialized
    _llm_client = None
    _llm_status = None
    _initialized = False
