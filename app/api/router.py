"""API router aggregation - includes all application routers."""

from fastapi import APIRouter

from app.routers import (
    backtests,
    dashboards,
    execution,
    forward_metrics,
    health,
    ingest,
    intents,
    jobs,
    kb,
    kb_trials,
    metrics,
    pdf,
    pine,
    query,
    reembed,
    sources,
    strategies,
    testing,
    unified_ingest,
    workspaces,
    youtube,
    youtube_pine,
)
from app.admin import router as admin_router
from app.admin.data import router as admin_data_router

# Main API router
api_router = APIRouter()

# Health and metrics
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(metrics.router)  # Metrics endpoint (excluded from OpenAPI)

# Core RAG functionality
api_router.include_router(ingest.router, tags=["Ingestion"])
api_router.include_router(unified_ingest.router, tags=["Ingestion"])
api_router.include_router(youtube.router, prefix="/sources/youtube", tags=["YouTube"])
api_router.include_router(
    youtube_pine.router, prefix="/sources/youtube", tags=["YouTube"]
)
api_router.include_router(pdf.router, tags=["PDF"])
api_router.include_router(pine.router, prefix="/sources/pine", tags=["Pine Script"])
api_router.include_router(sources.router, prefix="/sources", tags=["Sources"])
api_router.include_router(query.router, tags=["Query"])
api_router.include_router(reembed.router, tags=["Re-embed"])
api_router.include_router(jobs.router, tags=["Jobs"])

# Knowledge base
api_router.include_router(kb.router)  # KB endpoints with /kb prefix
api_router.include_router(kb_trials.router)  # Trading KB trial endpoints

# Backtesting
api_router.include_router(backtests.router)  # Backtest runner endpoints
api_router.include_router(forward_metrics.router)  # Forward metrics endpoints

# Execution
api_router.include_router(intents.router)  # Trade intent evaluation
api_router.include_router(execution.router)  # Paper execution endpoints

# Testing
api_router.include_router(
    testing.router
)  # Testing endpoints (run plan generation/execution)

# Strategy Registry
api_router.include_router(strategies.router, prefix="/strategies", tags=["Strategies"])

# Workspaces
api_router.include_router(workspaces.router, prefix="/workspaces", tags=["Workspaces"])

# Dashboards (read-only views)
api_router.include_router(dashboards.router, tags=["Dashboards"])

# Admin
api_router.include_router(admin_router)  # Admin UI (not in OpenAPI docs)
api_router.include_router(admin_data_router)  # Admin data management endpoints
