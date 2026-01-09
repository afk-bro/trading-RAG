"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service Configuration
    service_host: str = Field(default="0.0.0.0", description="Service host")
    service_port: int = Field(default=8000, description="Service port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Supabase Configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_role_key: str = Field(
        ..., description="Supabase service role key"
    )
    supabase_db_password: Optional[str] = Field(
        default=None, description="Supabase database password for direct PostgreSQL connection"
    )
    database_url: Optional[str] = Field(
        default=None, description="Direct PostgreSQL connection URL (overrides Supabase URL construction)"
    )

    # LLM Provider Configuration
    llm_provider: Literal["auto", "anthropic", "openai", "openrouter"] = Field(
        default="auto",
        description="LLM provider: auto prefers Anthropic > OpenAI > OpenRouter",
    )
    llm_required: bool = Field(
        default=False,
        description="If true, fail startup when no LLM provider key configured",
    )
    llm_enabled: bool = Field(
        default=True,
        description="Kill switch to disable LLM regardless of keys",
    )

    # LLM API Keys
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key (sk-ant-* format, preferred provider)"
    )
    # NOTE: claude_code_oauth_token does NOT work with Anthropic API.
    # It's reserved for future Claude Code CLI proxy provider.
    # Use ANTHROPIC_API_KEY from console.anthropic.com instead.
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key (second preference)"
    )
    openrouter_api_key: Optional[str] = Field(
        default=None, description="OpenRouter API key (fallback provider)"
    )

    # LLM Model Configuration
    answer_model: str = Field(
        default="claude-sonnet-4",
        description="Model for answer generation (user-facing output)",
    )
    rerank_model: Optional[str] = Field(
        default="claude-haiku-3-5",
        description="Model for reranking/scoring (set to null/empty to reuse answer_model)",
    )
    max_context_tokens: int = Field(
        default=8000, description="Maximum context tokens for LLM"
    )
    llm_timeout: int = Field(
        default=60, description="LLM request timeout in seconds"
    )

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection_active: str = Field(
        default="kb_nomic_embed_text_v1", description="Active Qdrant collection"
    )
    qdrant_timeout: int = Field(default=30, description="Qdrant timeout in seconds")

    # Ollama Configuration
    ollama_host: str = Field(default="localhost", description="Ollama host")
    ollama_port: int = Field(default=11434, description="Ollama port")
    embed_model: str = Field(
        default="nomic-embed-text", description="Embedding model name"
    )
    embed_batch_size: int = Field(default=32, description="Embedding batch size")
    ollama_timeout: int = Field(default=120, description="Ollama timeout in seconds")

    # Database Connection Pool
    db_pool_min_size: int = Field(default=5, description="Minimum connection pool size")
    db_pool_max_size: int = Field(default=20, description="Maximum connection pool size")

    # Optional YouTube API
    youtube_api_key: Optional[str] = Field(
        default=None, description="YouTube Data API key"
    )


    # Chunking configuration
    chunk_max_tokens: int = Field(
        default=512, description="Maximum tokens per chunk"
    )
    chunk_overlap_tokens: int = Field(
        default=50, description="Overlap tokens between chunks for context preservation"
    )
    chunk_tokenizer_encoding: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding name for tokenization (e.g., cl100k_base, p50k_base, r50k_base)"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True, description="Enable rate limiting"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60, description="Maximum requests per minute per IP"
    )
    rate_limit_burst: int = Field(
        default=10, description="Burst limit for rate limiting"
    )

    # Request size limits
    max_request_body_size: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum request body size in bytes"
    )

    # API Key Authentication
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for authentication. If set, all requests must include X-API-Key header"
    )
    api_key_header_name: str = Field(
        default="X-API-Key",
        description="Header name for API key"
    )

    # Cross-encoder Reranker
    warmup_reranker: bool = Field(
        default=False,
        description="Pre-load cross-encoder model at startup (uses GPU memory)"
    )
    rerank_timeout_s: float = Field(
        default=10.0,
        description="Rerank timeout in seconds. Fails open to vector fallback on timeout."
    )

    # Evaluation Persistence (PR3)
    eval_persist_enabled: bool = Field(
        default=False,
        description="Persist /query/compare evaluations to query_compare_evals table"
    )
    eval_store_question_preview: bool = Field(
        default=False,
        description="Store first 80 chars of question (otherwise hash only)"
    )

    # Sentry Observability
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking and performance monitoring"
    )
    sentry_environment: str = Field(
        default="development",
        description="Sentry environment tag (development, staging, production)"
    )
    sentry_traces_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sentry performance tracing sample rate (0.0-1.0)"
    )
    sentry_profiles_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sentry profiling sample rate (0.0-1.0)"
    )

    @property
    def ollama_base_url(self) -> str:
        """Get the Ollama base URL."""
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def qdrant_url(self) -> str:
        """Get the Qdrant URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
