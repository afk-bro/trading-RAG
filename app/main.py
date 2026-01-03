"""Trading RAG Pipeline - FastAPI Application."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    logger.info(
        "Starting Trading RAG Service",
        version=__version__,
        host=settings.service_host,
        port=settings.service_port,
        qdrant_collection=settings.qdrant_collection_active,
        embed_model=settings.embed_model,
    )

    # Initialize connections here (Qdrant, Supabase pools, etc.)
    # TODO: Initialize Qdrant client
    # TODO: Initialize Supabase/asyncpg pool
    # TODO: Validate Ollama model availability
    # TODO: Validate Qdrant collection exists/create

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Trading RAG Service")
    # TODO: Close connection pools


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
