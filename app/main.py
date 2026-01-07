"""Trading RAG Pipeline - FastAPI Application."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import __version__
from app.config import get_settings
from app.routers import health, ingest, jobs, metrics, pdf, query, reembed, youtube

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

# Initialize rate limiter
settings = get_settings()
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
                    logger.info("Constructed database URL from Supabase project settings")
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
                ssl='require',
                timeout=10,  # Short connection timeout to avoid blocking startup
                command_timeout=30,  # Query timeout
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

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Trading RAG Service")

    if _db_pool:
        await _db_pool.close()
        logger.info("Database pool closed")

    if _qdrant_client:
        await _qdrant_client.close()
        logger.info("Qdrant client closed")


# Create FastAPI app
app = FastAPI(
    title="Trading RAG Pipeline",
    description="RAG pipeline for finance and trading knowledge",
    version=__version__,
    lifespan=lifespan,
)

# Add rate limiter state and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
app.include_router(reembed.router, tags=["Re-embed"])
app.include_router(jobs.router, tags=["Jobs"])
app.include_router(metrics.router)  # Metrics endpoint (excluded from OpenAPI)


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Trading RAG Pipeline",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
