"""Trading RAG Pipeline - FastAPI Application.

This is the application assembly file. All logic is delegated to:
- app/core/logging.py - Structured logging configuration
- app/core/sentry.py - Sentry initialization
- app/core/lifespan.py - Startup/shutdown lifecycle
- app/core/middleware.py - Request middleware stack
- app/api/router.py - Router aggregation
"""

from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app import __version__
from app.api import api_router
from app.config import get_settings
from app.core.lifespan import lifespan
from app.core.logging import configure_logging
from app.core.middleware import setup_middleware
from app.core.sentry import init_sentry

# Configure structured logging
configure_logging()
logger = structlog.get_logger(__name__)

# Load settings
settings = get_settings()

# Initialize Sentry (if configured)
init_sentry(settings)

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

# Set up middleware (rate limiter, CORS, security headers, request tracking)
setup_middleware(app, settings)

# Include all API routers
app.include_router(api_router)


# Setup templates for landing page
_templates_dir = Path(__file__).parent / "admin" / "templates"
_templates = Jinja2Templates(directory=str(_templates_dir))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Landing page with system overview."""
    return _templates.TemplateResponse("landing.html", {"request": request})


@app.get("/api")
async def api_info():
    """API info endpoint (JSON)."""
    return {
        "service": "Trading RAG Pipeline",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
