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
            # Qdrant uses /healthz for health checks (not /health)
            response = await client.get(f"{settings.qdrant_url}/healthz")
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


@router.get("/debug/db")
async def debug_db_pool(settings: Settings = Depends(get_settings)):
    """Debug endpoint to check database pool status."""
    from app.routers import ingest as ingest_router

    pool = ingest_router._db_pool

    return {
        "database_url_configured": bool(settings.database_url),
        "database_url_prefix": settings.database_url[:50] + "..." if settings.database_url else None,
        "supabase_db_password_configured": bool(settings.supabase_db_password),
        "pool_initialized": pool is not None,
        "pool_size": pool.get_size() if pool else None,
        "pool_free_size": pool.get_idle_size() if pool else None,
    }


@router.get("/debug/db/test")
async def debug_db_test(settings: Settings = Depends(get_settings)):
    """Test database connection on-demand."""
    import asyncpg
    import traceback

    if not settings.database_url:
        return {"success": False, "error": "DATABASE_URL not configured"}

    try:
        # Try to connect directly (disable statement cache for pgbouncer)
        conn = await asyncpg.connect(
            settings.database_url,
            ssl='require',
            timeout=30,
            statement_cache_size=0,
        )
        # Test the connection
        result = await conn.fetchval("SELECT 1")
        await conn.close()
        return {"success": True, "test_result": result}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()[:1000],
        }


@router.get("/debug/network")
async def debug_network(settings: Settings = Depends(get_settings)):
    """Debug network connectivity from container."""
    import socket
    import re

    results = {
        "dns_resolv_conf": None,
        "dns_tests": {},
        "tcp_tests": {},
    }

    # Read resolv.conf
    try:
        with open("/etc/resolv.conf") as f:
            results["dns_resolv_conf"] = f.read()
    except Exception as e:
        results["dns_resolv_conf"] = f"Error: {e}"

    # Extract host from database URL
    db_host = None
    if settings.database_url:
        match = re.search(r"@([^:/@]+)", settings.database_url)
        if match:
            db_host = match.group(1)

    # Test hosts
    test_hosts = [
        ("google.com", 443),
        ("8.8.8.8", 53),
    ]
    if db_host:
        test_hosts.append((db_host, 5432))

    for host, port in test_hosts:
        # DNS test
        try:
            ip = socket.gethostbyname(host)
            results["dns_tests"][host] = {"resolved": True, "ip": ip}
        except Exception as e:
            results["dns_tests"][host] = {"resolved": False, "error": str(e)}

        # TCP test
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))
            sock.close()
            results["tcp_tests"][f"{host}:{port}"] = {"connected": True}
        except Exception as e:
            results["tcp_tests"][f"{host}:{port}"] = {"connected": False, "error": str(e)}

    return results
