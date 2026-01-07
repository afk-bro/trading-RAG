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
    llm_provider: Literal["auto", "anthropic", "openrouter"] = Field(
        default="auto",
        description="LLM provider: auto prefers Anthropic when key is available",
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
        default=None, description="Anthropic API key (preferred provider)"
    )
    claude_code_oauth_token: Optional[str] = Field(
        default=None, description="Claude Code OAuth token (alternative to API key)"
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
