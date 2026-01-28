"""Application lifespan management - startup and shutdown logic."""

import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
import structlog
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from app import __version__
from app.config import get_settings
from app.routers import (
    backtests,
    execution,
    forward_metrics,
    ingest,
    intents,
    kb,
    kb_trials,
    query,
    reembed,
    testing,
)
from app.admin import set_db_pool as set_admin_db_pool
from app.admin import set_qdrant_client as set_admin_qdrant_client
from app.admin.data import set_db_pool as set_admin_data_db_pool
from app.services.pine.poller import PineRepoPoller, set_poller
from app.services.market_data.poller import (
    LivePricePoller,
    set_poller as set_price_poller,
)
from app.services.discord_bot import start_bot as start_discord_bot, stop_bot as stop_discord_bot

logger = structlog.get_logger(__name__)

# Global clients - accessed by other modules
_db_pool: Optional[asyncpg.Pool] = None
_qdrant_client: Optional[AsyncQdrantClient] = None
_pine_poller: Optional[PineRepoPoller] = None
_price_poller: Optional[LivePricePoller] = None


def get_db_pool() -> Optional[asyncpg.Pool]:
    """Get the database connection pool."""
    return _db_pool


def get_qdrant_client() -> Optional[AsyncQdrantClient]:
    """Get the Qdrant client."""
    return _qdrant_client


def get_pine_poller() -> Optional[PineRepoPoller]:
    """Get the Pine repo poller instance."""
    return _pine_poller


def get_price_poller() -> Optional[LivePricePoller]:
    """Get the live price poller instance."""
    return _price_poller


async def _init_qdrant(settings) -> Optional[AsyncQdrantClient]:
    """Initialize Qdrant client."""
    try:
        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=settings.qdrant_timeout,
        )
        logger.info(
            "Qdrant client initialized",
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Wire up Qdrant client to routers
        ingest.set_qdrant_client(client)
        query.set_qdrant_client(client)
        reembed.set_qdrant_client(client)
        kb_trials.set_qdrant_client(client)
        set_admin_qdrant_client(client)

        return client
    except Exception as e:
        logger.error("Failed to initialize Qdrant client", error=str(e))
        return None


async def _init_database(settings) -> Optional[asyncpg.Pool]:
    """Initialize asyncpg connection pool for Supabase."""
    import traceback

    try:
        postgres_url = None

        # Option 1: Direct DATABASE_URL takes precedence
        if settings.database_url:
            postgres_url = settings.database_url
            logger.info("Using direct DATABASE_URL for database connection")
        # Option 2: Construct URL from Supabase settings with DB password
        elif settings.supabase_db_password:
            supabase_url = settings.supabase_url
            if supabase_url.startswith("https://"):
                # Extract project ID from URL
                match = re.match(r"https://([^.]+)\.supabase\.co", supabase_url)
                if match:
                    project_id = match.group(1)
                    # Supabase Postgres connection format with actual DB password
                    postgres_url = f"postgresql://postgres:{settings.supabase_db_password}@db.{project_id}.supabase.co:5432/postgres"  # noqa: E501
                    logger.info(
                        "Constructed database URL from Supabase project settings"
                    )
        # Option 3: If supabase_url is already a postgres URL, use it directly
        elif settings.supabase_url.startswith("postgresql://"):
            postgres_url = settings.supabase_url
            logger.info("Using supabase_url as direct PostgreSQL connection")
        else:
            logger.warning(
                "Database connection not configured. Set DATABASE_URL or SUPABASE_DB_PASSWORD in .env"  # noqa: E501
            )
            return None

        if postgres_url:
            logger.info(
                "Attempting database connection",
                url_prefix=postgres_url[:50] + "...",
            )

            # Try with short timeout - don't block startup if DB is unreachable
            # The service can still operate in degraded mode without DB
            pool = await asyncpg.create_pool(
                postgres_url,
                min_size=0,  # Don't require any connections at startup
                max_size=settings.db_pool_max_size,
                ssl="require",
                timeout=10,  # Short connection timeout to avoid blocking startup
                command_timeout=30,  # Query timeout
                statement_cache_size=0,  # Disable for pgbouncer transaction mode
            )
            logger.info(
                "Database pool initialized",
                min_size=0,
                max_size=settings.db_pool_max_size,
            )

            # Wire up database pool to routers
            ingest.set_db_pool(pool)
            query.set_db_pool(pool)
            reembed.set_db_pool(pool)
            kb.set_db_pool(pool)
            kb_trials.set_db_pool(pool)
            backtests.set_db_pool(pool)
            forward_metrics.set_db_pool(pool)
            intents.set_db_pool(pool)
            execution.set_db_pool(pool)
            testing.set_db_pool(pool)
            set_admin_db_pool(pool)
            set_admin_data_db_pool(pool)

            return pool

    except Exception as e:
        logger.error(
            "Failed to initialize database pool - endpoints requiring DB will be unavailable",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        return None

    return None


async def _validate_ollama(settings) -> None:
    """Validate Ollama model availability."""
    try:
        from app.services.embedder import get_embedder

        embedder = get_embedder()
        if await embedder.health_check():
            logger.info(
                "Ollama embedder validated",
                model=settings.embed_model,
            )
        else:
            logger.warning(
                "Ollama model not available",
                model=settings.embed_model,
            )
    except Exception as e:
        logger.warning("Failed to validate Ollama embedder", error=str(e))


async def _validate_qdrant_collection(client: AsyncQdrantClient, settings) -> None:
    """Validate/create Qdrant collection."""
    try:
        from app.repositories.vectors import VectorRepository
        from app.services.embedder import get_embedder

        embedder = get_embedder()
        dimension = await embedder.get_dimension()

        vector_repo = VectorRepository(
            client=client,
            collection=settings.qdrant_collection_active,
        )
        await vector_repo.ensure_collection(dimension=dimension)
        logger.info(
            "Qdrant collection validated",
            collection=settings.qdrant_collection_active,
            dimension=dimension,
        )
    except Exception as e:
        logger.warning("Failed to validate Qdrant collection", error=str(e))


def _init_llm_subsystem() -> None:
    """Initialize LLM subsystem and log configuration."""
    try:
        from app.services.llm_factory import get_llm_status, LLMStartupError

        llm_status = get_llm_status()
        logger.info(
            "LLM configuration",
            provider_config=llm_status.provider_config,
            provider_resolved=llm_status.provider_resolved,
            answer_model=llm_status.answer_model,
            rerank_model_effective=llm_status.rerank_model_effective,
            llm_enabled=llm_status.enabled,
        )
    except Exception as e:
        # Check if it's a startup error that should be raised
        from app.services.llm_factory import LLMStartupError

        if isinstance(e, LLMStartupError):
            logger.error("LLM startup failed", error=str(e))
            raise
        logger.warning("Failed to initialize LLM subsystem", error=str(e))


async def _warmup_reranker(settings) -> None:
    """
    Optional: Warm up cross-encoder reranker (pre-load model to GPU).

    IMPORTANT: This only warms the DEFAULT model (BAAI/bge-reranker-v2-m3).
    If workspaces configure different models, those will load on first query.
    We intentionally do NOT iterate workspaces to warm multiple models,
    as this could exhaust GPU memory with multiple large models.
    """
    if not settings.warmup_reranker:
        return

    try:
        from app.services.reranker import get_reranker, RerankCandidate

        # Only warm the default model - do not iterate workspace configs
        warmup_config = {
            "enabled": True,
            "cross_encoder": {"device": "cuda"},  # Uses DEFAULT_MODEL
        }
        reranker = get_reranker(warmup_config)

        if reranker and reranker.method == "cross_encoder":
            dummy = [
                RerankCandidate(
                    chunk_id="warmup",
                    document_id="warmup",
                    chunk_index=0,
                    text="warmup text for model loading",
                    vector_score=1.0,
                    workspace_id="warmup",
                )
            ]
            await reranker.rerank("warmup query", dummy, top_k=1)
            logger.info(
                "Cross-encoder reranker warmed up",
                model=reranker.model_id,
            )
    except Exception as e:
        logger.warning("Failed to warm up reranker", error=str(e))


async def _shutdown_reranker() -> None:
    """Close reranker resources."""
    try:
        from app.services import reranker as reranker_module

        if reranker_module._cross_encoder_reranker is not None:
            reranker_module._cross_encoder_reranker.close(wait=True)
            reranker_module._cross_encoder_reranker = None
            logger.info("CrossEncoderReranker closed")

        if reranker_module._llm_reranker is not None:
            reranker_module._llm_reranker.close()
            reranker_module._llm_reranker = None
            logger.info("LLMReranker closed")
    except Exception as e:
        logger.warning("Error closing reranker", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global _db_pool, _qdrant_client, _pine_poller, _price_poller

    settings = get_settings()
    logger.info(
        "Starting Trading RAG Service",
        version=__version__,
        git_sha=settings.git_sha or "unknown",
        build_time=settings.build_time or "unknown",
        config_profile=settings.config_profile,
        host=settings.service_host,
        port=settings.service_port,
        qdrant_collection=settings.qdrant_collection_active,
        embed_model=settings.embed_model,
    )

    # Initialize Qdrant client
    _qdrant_client = await _init_qdrant(settings)

    # Initialize database pool
    _db_pool = await _init_database(settings)

    # Validate Ollama model availability
    await _validate_ollama(settings)

    # Validate/create Qdrant collection
    if _qdrant_client:
        await _validate_qdrant_collection(_qdrant_client, settings)

    # Initialize LLM subsystem
    _init_llm_subsystem()

    # Optional reranker warmup
    await _warmup_reranker(settings)

    # Start Pine repo poller (if enabled and DB available)
    if _db_pool and settings.pine_repo_poll_enabled:
        _pine_poller = PineRepoPoller(_db_pool, settings, _qdrant_client)
        set_poller(_pine_poller)
        await _pine_poller.start()
    elif not settings.pine_repo_poll_enabled:
        logger.info("Pine repo polling disabled (PINE_REPO_POLL_ENABLED=false)")

    # Start live price poller (if enabled and DB available)
    if _db_pool and settings.live_price_poll_enabled:
        _price_poller = LivePricePoller(_db_pool, settings)
        set_price_poller(_price_poller)
        await _price_poller.start()
    elif not settings.live_price_poll_enabled:
        logger.info("Live price polling disabled (LIVE_PRICE_POLL_ENABLED=false)")

    # Start Discord bot (if token configured)
    await start_discord_bot()

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Trading RAG Service")

    # Stop Discord bot
    await stop_discord_bot()

    # Stop pollers first (before DB pool closes)
    if _price_poller:
        await _price_poller.stop()
        set_price_poller(None)
        logger.info("Live price poller stopped")

    if _pine_poller:
        await _pine_poller.stop()
        set_poller(None)
        logger.info("Pine repo poller stopped")

    await _shutdown_reranker()

    if _db_pool:
        await _db_pool.close()
        logger.info("Database pool closed")

    if _qdrant_client:
        await _qdrant_client.close()
        logger.info("Qdrant client closed")
