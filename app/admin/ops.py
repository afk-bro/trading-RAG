"""Ops Snapshot admin endpoint."""

import json
from datetime import datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for ops routes."""
    global _db_pool
    _db_pool = pool


@router.get("/ops/snapshot")
async def ops_snapshot(
    _: bool = Depends(require_admin_token),
    settings: Settings = Depends(get_settings),
):
    """
    Operational snapshot for go-live verification.

    Aggregates:
    - Service readiness + release metadata
    - Active collection and model configuration
    - Ingestion coverage metrics
    - Last smoke run timestamp (if available)

    Use this endpoint to verify deployment health before enabling features.
    """
    import asyncio

    from app import __version__
    from app.routers.health import (
        _timed_check,
        check_database_health,
        check_embed_service,
        check_qdrant_collection,
    )

    # Run readiness checks
    db_check, qdrant_check, embed_check = await asyncio.gather(
        _timed_check("database", check_database_health(settings)),
        _timed_check("qdrant_collection", check_qdrant_collection(settings)),
        _timed_check("embed_service", check_embed_service(settings)),
    )

    readiness = {
        "ready": all(c.status == "ok" for c in [db_check, qdrant_check, embed_check]),
        "checks": {
            "database": {
                "status": db_check.status,
                "latency_ms": db_check.latency_ms,
                "error": db_check.error,
            },
            "qdrant_collection": {
                "status": qdrant_check.status,
                "latency_ms": qdrant_check.latency_ms,
                "error": qdrant_check.error,
            },
            "embed_service": {
                "status": embed_check.status,
                "latency_ms": embed_check.latency_ms,
                "error": embed_check.error,
            },
        },
    }

    # Release metadata
    release = {
        "version": __version__,
        "git_sha": settings.git_sha or "unknown",
        "build_time": settings.build_time or "unknown",
        "config_profile": settings.config_profile,
        "environment": settings.sentry_environment,
    }

    # Configuration snapshot
    config = {
        "active_collection": settings.qdrant_collection_active,
        "embed_model": settings.embed_model,
        "embed_dim": getattr(settings, "embed_dim", 768),
        "docs_enabled": settings.docs_enabled,
        "rate_limit_enabled": settings.rate_limit_enabled,
        "llm_enabled": settings.llm_enabled,
    }

    # Ingestion coverage metrics (from database)
    coverage = {
        "total_kb_trials": 0,
        "trials_with_vectors": 0,
        "missing_vectors": 0,
        "by_strategy": {},
        "by_workspace": {},
        "stale_text_hash_count": 0,
    }

    if _db_pool:
        try:
            async with _db_pool.acquire() as conn:
                # Total KB trials
                total = await conn.fetchval("SELECT COUNT(*) FROM kb_trials")
                coverage["total_kb_trials"] = total or 0

                # Trials with vectors (have point_id)
                with_vectors = await conn.fetchval(
                    "SELECT COUNT(*) FROM kb_trials WHERE point_id IS NOT NULL"
                )
                coverage["trials_with_vectors"] = with_vectors or 0
                coverage["missing_vectors"] = (total or 0) - (with_vectors or 0)

                # By strategy
                strategy_rows = await conn.fetch(
                    """
                    SELECT strategy_name, COUNT(*) as count,
                           COUNT(*) FILTER (WHERE point_id IS NOT NULL) as indexed
                    FROM kb_trials
                    GROUP BY strategy_name
                    ORDER BY count DESC
                    """
                )
                coverage["by_strategy"] = {
                    row["strategy_name"]: {
                        "total": row["count"],
                        "indexed": row["indexed"],
                    }
                    for row in strategy_rows
                }

                # By workspace
                workspace_rows = await conn.fetch(
                    """
                    SELECT workspace_id, COUNT(*) as count,
                           COUNT(*) FILTER (WHERE point_id IS NOT NULL) as indexed
                    FROM kb_trials
                    GROUP BY workspace_id
                    ORDER BY count DESC
                    """
                )
                coverage["by_workspace"] = {
                    str(row["workspace_id"]): {
                        "total": row["count"],
                        "indexed": row["indexed"],
                    }
                    for row in workspace_rows
                }

                # Check for stale text hashes
                stale = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM kb_trials
                    WHERE point_id IS NOT NULL
                    AND updated_at > indexed_at
                    """
                )
                coverage["stale_text_hash_count"] = stale or 0

        except Exception as e:
            logger.warning("Failed to fetch coverage metrics", error=str(e))
            coverage["error"] = str(e)

    # Last smoke run timestamp (check for smoke test results file)
    smoke_run = {
        "last_run": None,
        "status": "unknown",
        "results_available": False,
    }

    # Check common smoke test output locations
    smoke_paths = [
        Path("tests/smoke/results/latest.json"),
        Path("smoke_results.json"),
        Path("/tmp/smoke_results.json"),
    ]

    for smoke_path in smoke_paths:
        if smoke_path.exists():
            try:
                with open(smoke_path) as f:
                    smoke_data = json.load(f)
                smoke_run["last_run"] = smoke_data.get("timestamp") or smoke_data.get(
                    "run_at"
                )
                smoke_run["status"] = smoke_data.get("status", "unknown")
                smoke_run["results_available"] = True
                break
            except Exception:
                pass

    return {
        "snapshot_time": datetime.utcnow().isoformat() + "Z",
        "readiness": readiness,
        "release": release,
        "config": config,
        "coverage": coverage,
        "smoke_run": smoke_run,
    }
