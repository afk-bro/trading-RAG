"""Middleware configuration for the FastAPI application."""

import os
import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import __version__
from app.config import Settings
from app.routers import metrics

logger = structlog.get_logger(__name__)


def setup_rate_limiter(app: FastAPI, settings: Settings) -> Limiter:
    """Set up rate limiter and attach to app."""
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.rate_limit_requests_per_minute}/minute"],
        enabled=settings.rate_limit_enabled,
    )
    app.state.limiter = limiter
    # mypy: slowapi handler signature differs from FastAPI expected type
    app.add_exception_handler(
        RateLimitExceeded, _rate_limit_exceeded_handler  # type: ignore[arg-type]
    )
    return limiter


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware."""
    cors_origins_str = os.environ.get("CORS_ORIGINS", "*")
    if cors_origins_str == "*":
        # Development mode - allow all (will log warning)
        cors_origins = ["*"]
        logger.warning(
            "CORS_ORIGINS not set, allowing all origins (not for production)"
        )
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


def create_request_middleware(settings: Settings):
    """Create request middleware with settings closure."""

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
                        "detail": f"API key required. Provide key in {settings.api_key_header_name} header",  # noqa: E501
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
                    "detail": f"Request body too large. Maximum size is {settings.max_request_body_size // (1024 * 1024)}MB",  # noqa: E501
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

    return request_middleware


def setup_middleware(app: FastAPI, settings: Settings) -> None:
    """Set up all middleware for the application."""
    # Rate limiter
    setup_rate_limiter(app, settings)

    # CORS
    setup_cors(app)

    # Security headers (added first, runs last in middleware stack)
    app.middleware("http")(security_headers_middleware)

    # Request middleware (request ID, timing, auth, size limits)
    app.middleware("http")(create_request_middleware(settings))
