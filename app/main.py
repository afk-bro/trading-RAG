"""Trading RAG Pipeline - FastAPI Application."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
import sentry_sdk
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import __version__
from app.config import get_settings
from app.routers import (
    health,
    ingest,
    jobs,
    kb,
    kb_trials,
    metrics,
    pdf,
    query,
    reembed,
    youtube,
    backtests,
    forward_metrics,
    intents,
    execution,
    testing,
)
from app.admin import router as admin_router, set_db_pool as set_admin_db_pool

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Initialize Sentry (if configured)
settings = get_settings()
if settings.sentry_dsn:
    from sentry_sdk.integrations.logging import LoggingIntegration
    import os

    # Only send ERROR-level logs as Sentry events
    sentry_logging = LoggingIntegration(
        level=None,  # Keep normal log levels
        event_level="ERROR",  # Only ERROR+ become Sentry events
    )

    def before_send(event, hint):
        """
        Filter out 4xx client errors from Sentry events.

        We don't want to track user errors (401, 403, 404, 422, 429) as exceptions.
        Only 5xx server errors should be captured.
        """
        # Check if this is an HTTP exception
        if "exc_info" in hint:
            exc_type, exc_value, tb = hint["exc_info"]
            # Filter FastAPI/Starlette HTTP exceptions
            if hasattr(exc_value, "status_code"):
                status_code = exc_value.status_code
                if 400 <= status_code < 500:
                    return None  # Drop 4xx errors

        # Check response context for status code
        if "contexts" in event:
            response = event.get("contexts", {}).get("response", {})
            status_code = response.get("status_code", 0)
            if 400 <= status_code < 500:
                return None

        return event

    def traces_sampler(sampling_context: dict) -> float:
        """
        Route-aware sampling for KB recommend critical path.

        - 100% for KB recommend (filter by kb_status tag in Sentry dashboards)
        - Inherits parent sampling decision if available
        - Default rate for everything else

        Note: We can't sample by response status (degraded/none) at trace start
        since traces_sampler runs before the request. Instead we sample 100%
        for KB recommend and use kb_status/mode tags to filter in Sentry UI.
        """
        # Check for transaction context
        tx_context = sampling_context.get("transaction_context", {})
        tx_name = tx_context.get("name", "")

        # KB recommend - 100% sampling (filter by kb_status/mode tags in Sentry)
        if "/kb/trials/recommend" in tx_name:
            return 1.0

        # Check parent sampling decision
        parent = sampling_context.get("parent_sampled")
        if parent is not None:
            return float(parent)

        # Default sampling rate
        return settings.sentry_traces_sample_rate

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=os.environ.get("GIT_SHA", f"trading-rag@{__version__}"),
        integrations=[
            sentry_logging,
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        enable_tracing=True,
        traces_sampler=traces_sampler,
        profiles_sample_rate=settings.sentry_profiles_sample_rate,
        send_default_pii=False,
        attach_stacktrace=True,
        before_send=before_send,
    )

    # Set global tags for service metadata
    sentry_sdk.set_tag("service", "trading-rag")
    sentry_sdk.set_tag("collection", settings.qdrant_collection_active)
    sentry_sdk.set_tag("embed_model", settings.embed_model)
    sentry_sdk.set_tag("vector_dim", settings.embed_dim)

    logger.info(
        "Sentry initialized",
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
    )

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_requests_per_minute}/minute"],
    enabled=settings.rate_limit_enabled,
)

# Global clients
_db_pool = None
_qdrant_client = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global _db_pool, _qdrant_client

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
    try:
        _qdrant_client = AsyncQdrantClient(
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
        ingest.set_qdrant_client(_qdrant_client)
        query.set_qdrant_client(_qdrant_client)
        reembed.set_qdrant_client(_qdrant_client)

    except Exception as e:
        logger.error("Failed to initialize Qdrant client", error=str(e))
        _qdrant_client = None

    # Initialize asyncpg connection pool for Supabase
    try:
        import re

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
                    postgres_url = f"postgresql://postgres:{settings.supabase_db_password}@db.{project_id}.supabase.co:5432/postgres"
                    logger.info(
                        "Constructed database URL from Supabase project settings"
                    )
        # Option 3: If supabase_url is already a postgres URL, use it directly
        elif settings.supabase_url.startswith("postgresql://"):
            postgres_url = settings.supabase_url
            logger.info("Using supabase_url as direct PostgreSQL connection")
        else:
            logger.warning(
                "Database connection not configured. Set DATABASE_URL or SUPABASE_DB_PASSWORD in .env"
            )

        if postgres_url:
            logger.info(
                "Attempting database connection",
                url_prefix=postgres_url[:50] + "...",
            )

            # Try with short timeout - don't block startup if DB is unreachable
            # The service can still operate in degraded mode without DB
            _db_pool = await asyncpg.create_pool(
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
            ingest.set_db_pool(_db_pool)
            query.set_db_pool(_db_pool)
            reembed.set_db_pool(_db_pool)
            kb.set_db_pool(_db_pool)
            kb_trials.set_db_pool(_db_pool)
            backtests.set_db_pool(_db_pool)
            forward_metrics.set_db_pool(_db_pool)
            intents.set_db_pool(_db_pool)
            execution.set_db_pool(_db_pool)
            testing.set_db_pool(_db_pool)
            set_admin_db_pool(_db_pool)

    except Exception as e:
        import traceback

        logger.error(
            "Failed to initialize database pool - endpoints requiring DB will be unavailable",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        _db_pool = None

    # Validate Ollama model availability
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

    # Validate/create Qdrant collection
    if _qdrant_client:
        try:
            from app.repositories.vectors import VectorRepository
            from app.services.embedder import get_embedder

            embedder = get_embedder()
            dimension = await embedder.get_dimension()

            vector_repo = VectorRepository(
                client=_qdrant_client,
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

    # Initialize LLM subsystem and log configuration
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
    except LLMStartupError as e:
        logger.error("LLM startup failed", error=str(e))
        raise
    except Exception as e:
        logger.warning("Failed to initialize LLM subsystem", error=str(e))

    # Optional: Warm up cross-encoder reranker (pre-load model to GPU)
    # This is disabled by default to avoid GPU memory usage at startup.
    # Enable via WARMUP_RERANKER=true in .env if desired.
    #
    # IMPORTANT: This only warms the DEFAULT model (BAAI/bge-reranker-v2-m3).
    # If workspaces configure different models, those will load on first query.
    # We intentionally do NOT iterate workspaces to warm multiple models,
    # as this could exhaust GPU memory with multiple large models.
    if settings.warmup_reranker:
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

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Trading RAG Service")

    # Close reranker resources
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

    if _db_pool:
        await _db_pool.close()
        logger.info("Database pool closed")

    if _qdrant_client:
        await _qdrant_client.close()
        logger.info("Qdrant client closed")


# Create FastAPI app
# Conditionally disable docs in production (set DOCS_ENABLED=false)
app = FastAPI(
    title="Trading RAG Pipeline",
    description="RAG pipeline for finance and trading knowledge",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

if not settings.docs_enabled:
    logger.info("API docs disabled (DOCS_ENABLED=false)")

# Add rate limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
# In production, set CORS_ORIGINS to comma-separated list of allowed origins
import os

cors_origins_str = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_str == "*":
    # Development mode - allow all (will log warning)
    cors_origins = ["*"]
    logger.warning("CORS_ORIGINS not set, allowing all origins (not for production)")
else:
    # Production mode - explicit allowlist
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
    logger.info("CORS origins configured", origins=cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)

    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"

    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"

    # Control referrer information
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Add HSTS if behind TLS (check X-Forwarded-Proto)
    if request.headers.get("X-Forwarded-Proto") == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    return response


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID, timing, size limits, and API key validation to all requests."""
    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Bind request context to logger
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    # Check API key authentication (if configured)
    # Skip auth for health, metrics, docs, and openapi endpoints
    public_paths = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/"}
    if settings.api_key and request.url.path not in public_paths:
        provided_key = request.headers.get(settings.api_key_header_name)
        if not provided_key:
            logger.warning(
                "API key missing",
                path=request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": f"API key required. Provide key in {settings.api_key_header_name} header",
                    "retryable": False,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-API-Version": __version__,
                },
            )
        if provided_key != settings.api_key:
            logger.warning(
                "Invalid API key",
                path=request.url.path,
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Invalid API key",
                    "retryable": False,
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-API-Version": __version__,
                },
            )

    # Check request body size limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_request_body_size:
        logger.warning(
            "Request body too large",
            content_length=int(content_length),
            max_size=settings.max_request_body_size,
        )
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"Request body too large. Maximum size is {settings.max_request_body_size // (1024 * 1024)}MB",
                "retryable": False,
            },
            headers={
                "X-Request-ID": request_id,
                "X-API-Version": __version__,
            },
        )

    # Time the request
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("Request failed", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "retryable": True},
            headers={
                "X-Request-ID": request_id,
                "X-API-Version": __version__,
            },
        )

    # Calculate duration
    duration_ms = (time.perf_counter() - start_time) * 1000
    duration_seconds = duration_ms / 1000

    # Add headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    response.headers["X-API-Version"] = __version__

    # Record metrics (skip /metrics endpoint to avoid recursion)
    if request.url.path != "/metrics":
        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration_seconds,
        )

    # Log request completion
    logger.info(
        "Request completed",
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )

    return response


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(youtube.router, prefix="/sources/youtube", tags=["YouTube"])
app.include_router(pdf.router, tags=["PDF"])
app.include_router(query.router, tags=["Query"])
app.include_router(kb.router)  # KB endpoints with /kb prefix
app.include_router(kb_trials.router)  # Trading KB trial endpoints
app.include_router(backtests.router)  # Backtest runner endpoints
app.include_router(reembed.router, tags=["Re-embed"])
app.include_router(jobs.router, tags=["Jobs"])
app.include_router(metrics.router)  # Metrics endpoint (excluded from OpenAPI)
app.include_router(forward_metrics.router)  # Forward metrics endpoints
app.include_router(intents.router)  # Trade intent evaluation
app.include_router(execution.router)  # Paper execution endpoints
app.include_router(testing.router)  # Testing endpoints (run plan generation/execution)
app.include_router(admin_router)  # Admin UI (not in OpenAPI docs)


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Trading RAG Pipeline",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
