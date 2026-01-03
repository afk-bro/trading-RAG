"""Health check endpoint."""

import time
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Depends

from app import __version__
from app.config import Settings, get_settings
from app.schemas import DependencyHealth, HealthResponse

router = APIRouter()
logger = structlog.get_logger(__name__)


async def check_qdrant_health(settings: Settings) -> DependencyHealth:
    """Check Qdrant connectivity and health."""
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.qdrant_timeout) as client:
            response = await client.get(f"{settings.qdrant_url}/health")
            latency = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                return DependencyHealth(status="ok", latency_ms=latency)
            return DependencyHealth(
                status="error",
                latency_ms=latency,
                error=f"Unexpected status: {response.status_code}",
            )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(status="error", latency_ms=latency, error=str(e))


async def check_ollama_health(settings: Settings) -> DependencyHealth:
    """Check Ollama connectivity and model availability."""
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            latency = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                # Check if the embed model is available
                data = response.json()
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                if settings.embed_model in models:
                    return DependencyHealth(status="ok", latency_ms=latency)
                return DependencyHealth(
                    status="error",
                    latency_ms=latency,
                    error=f"Model {settings.embed_model} not found",
                )
            return DependencyHealth(
                status="error",
                latency_ms=latency,
                error=f"Unexpected status: {response.status_code}",
            )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(status="error", latency_ms=latency, error=str(e))


async def check_supabase_health(settings: Settings) -> DependencyHealth:
    """Check Supabase connectivity."""
    start = time.perf_counter()
    try:
        # Simple health check - try to reach the Supabase REST API
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{settings.supabase_url}/rest/v1/",
                headers={
                    "apikey": settings.supabase_service_role_key,
                    "Authorization": f"Bearer {settings.supabase_service_role_key}",
                },
            )
            latency = (time.perf_counter() - start) * 1000
            # Supabase returns 200 or 404 for valid connection
            if response.status_code in (200, 404):
                return DependencyHealth(status="ok", latency_ms=latency)
            return DependencyHealth(
                status="error",
                latency_ms=latency,
                error=f"Unexpected status: {response.status_code}",
            )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(status="error", latency_ms=latency, error=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """
    Check health of service and all dependencies.

    Returns status of Qdrant, Supabase, and Ollama along with latency metrics.
    """
    logger.info("Health check requested")

    # Check all dependencies concurrently
    import asyncio

    qdrant_task = check_qdrant_health(settings)
    ollama_task = check_ollama_health(settings)
    supabase_task = check_supabase_health(settings)

    qdrant_health, ollama_health, supabase_health = await asyncio.gather(
        qdrant_task, ollama_task, supabase_task
    )

    # Determine overall status
    all_healthy = all(
        h.status == "ok" for h in [qdrant_health, ollama_health, supabase_health]
    )
    overall_status = "ok" if all_healthy else "degraded"

    # Build latency map
    latency_ms: dict[str, float] = {}
    if qdrant_health.latency_ms is not None:
        latency_ms["qdrant"] = qdrant_health.latency_ms
    if ollama_health.latency_ms is not None:
        latency_ms["ollama"] = ollama_health.latency_ms
    if supabase_health.latency_ms is not None:
        latency_ms["supabase"] = supabase_health.latency_ms

    response = HealthResponse(
        status=overall_status,
        qdrant=qdrant_health,
        supabase=supabase_health,
        ollama=ollama_health,
        active_collection=settings.qdrant_collection_active,
        embed_model=settings.embed_model,
        latency_ms=latency_ms,
        version=__version__,
    )

    logger.info(
        "Health check completed",
        status=overall_status,
        qdrant=qdrant_health.status,
        ollama=ollama_health.status,
        supabase=supabase_health.status,
    )

    return response
