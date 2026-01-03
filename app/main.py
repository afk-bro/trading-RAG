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

from app import __version__
from app.config import get_settings
from app.routers import health, ingest, jobs, query, reembed, youtube

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
        # Parse Supabase URL to get connection string
        # Supabase URLs are like: https://xxx.supabase.co
        # The Postgres connection is: postgresql://postgres:password@db.xxx.supabase.co:5432/postgres
        supabase_url = settings.supabase_url

        # Try to construct Postgres URL if not provided directly
        # This assumes the user has set up the database connection string
        # For now, we'll try to connect using the Supabase REST API format
        if supabase_url.startswith("https://"):
            # Extract project ID from URL
            import re
            match = re.match(r"https://([^.]+)\.supabase\.co", supabase_url)
            if match:
                project_id = match.group(1)
                # Supabase Postgres connection format
                postgres_url = f"postgresql://postgres:{settings.supabase_service_role_key}@db.{project_id}.supabase.co:5432/postgres"

                _db_pool = await asyncpg.create_pool(
                    postgres_url,
                    min_size=settings.db_pool_min_size,
                    max_size=settings.db_pool_max_size,
                )
                logger.info(
                    "Database pool initialized",
                    min_size=settings.db_pool_min_size,
                    max_size=settings.db_pool_max_size,
                )

                # Wire up database pool to routers
                ingest.set_db_pool(_db_pool)
                query.set_db_pool(_db_pool)
                reembed.set_db_pool(_db_pool)
        else:
            # Assume it's already a Postgres connection string
            _db_pool = await asyncpg.create_pool(
                supabase_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
            )

            # Wire up database pool to routers
            ingest.set_db_pool(_db_pool)
            query.set_db_pool(_db_pool)
            reembed.set_db_pool(_db_pool)

            logger.info("Database pool initialized")

    except Exception as e:
        logger.warning(
            "Failed to initialize database pool - endpoints requiring DB will be unavailable",
            error=str(e),
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
    """Add request ID and timing to all requests."""
    # Get or generate request ID
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Bind request context to logger
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
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
            headers={"X-Request-ID": request_id},
        )

    # Calculate duration
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Add headers to response
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

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
app.include_router(query.router, tags=["Query"])
app.include_router(reembed.router, tags=["Re-embed"])
app.include_router(jobs.router, tags=["Jobs"])


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Trading RAG Pipeline",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
