"""Query Compare Evals admin endpoints (PR3: Evaluation Collector)."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for evals routes."""
    global _db_pool
    _db_pool = pool


def _get_eval_repo():
    """Get EvalRepository instance."""
    from app.repositories.evals import EvalRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return EvalRepository(_db_pool)


@router.get("/evals/query-compare/summary")
async def eval_summary(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace to get stats for"),
    since: str = Query("24h", description="Time window: 1h, 24h, 7d, 30d"),
    _: None = Depends(require_admin_token),
):
    """
    Get summary stats for query compare evaluations.

    Returns impact rate, latency percentiles, and fallback rate.
    """
    # Parse time window
    since_hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}.get(since, 24)

    repo = _get_eval_repo()
    summary = await repo.get_summary(workspace_id, since_hours)

    return {
        "workspace_id": str(workspace_id),
        "since": since,
        "total": summary.total,
        "impacted_count": summary.impacted_count,
        "impact_rate": round(summary.impact_rate, 4),
        "p50_rerank_ms": (
            round(summary.p50_rerank_ms, 1) if summary.p50_rerank_ms else None
        ),
        "p95_rerank_ms": (
            round(summary.p95_rerank_ms, 1) if summary.p95_rerank_ms else None
        ),
        "fallback_count": summary.fallback_count,
        "fallback_rate": round(summary.fallback_rate, 4),
        "timeout_count": summary.timeout_count,
        "timeout_rate": round(summary.timeout_rate, 4),
    }


@router.get("/evals/query-compare/by-config")
async def eval_by_config(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace to get stats for"),
    since: str = Query("7d", description="Time window: 1h, 24h, 7d, 30d"),
    _: None = Depends(require_admin_token),
):
    """
    Get stats grouped by config fingerprint.

    Useful for comparing different retrieve_k/top_k combinations.
    """
    since_hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}.get(since, 168)

    repo = _get_eval_repo()
    configs = await repo.get_by_config(workspace_id, since_hours)

    return {
        "workspace_id": str(workspace_id),
        "since": since,
        "configs": [
            {
                "config_fingerprint": c.config_fingerprint,
                "rerank_method": c.rerank_method,
                "rerank_model": c.rerank_model,
                "candidates_k": c.candidates_k,
                "top_k": c.top_k,
                "total": c.total,
                "impact_rate": round(c.impact_rate, 4),
                "p50_rerank_ms": round(c.p50_rerank_ms, 1) if c.p50_rerank_ms else None,
                "p95_rerank_ms": round(c.p95_rerank_ms, 1) if c.p95_rerank_ms else None,
                "fallback_rate": round(c.fallback_rate, 4),
            }
            for c in configs
        ],
    }


@router.get("/evals/query-compare/most-impacted")
async def eval_most_impacted(
    request: Request,
    workspace_id: UUID = Query(..., description="Workspace to get stats for"),
    since: str = Query("7d", description="Time window: 1h, 24h, 7d, 30d"),
    limit: int = Query(20, ge=1, le=100, description="Number of results"),
    _: None = Depends(require_admin_token),
):
    """
    Get queries with highest reranking impact (lowest jaccard).

    Use for manual spot-checking of rerank quality.
    """
    since_hours = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}.get(since, 168)

    repo = _get_eval_repo()
    impacted = await repo.get_most_impacted(workspace_id, since_hours, limit)

    return {
        "workspace_id": str(workspace_id),
        "since": since,
        "count": len(impacted),
        "queries": [
            {
                "created_at": q.created_at.isoformat(),
                "question_hash": q.question_hash,
                "question_preview": q.question_preview,
                "jaccard": round(q.jaccard, 4),
                "spearman": round(q.spearman, 4) if q.spearman is not None else None,
                "rank_delta_mean": (
                    round(q.rank_delta_mean, 2) if q.rank_delta_mean else None
                ),
                "rank_delta_max": q.rank_delta_max,
                "vector_top5_ids": q.vector_top5_ids,
                "reranked_top5_ids": q.reranked_top5_ids,
            }
            for q in impacted
        ],
    }


@router.delete("/evals/query-compare/cleanup")
async def eval_cleanup(
    request: Request,
    days: int = Query(90, ge=1, le=365, description="Delete evals older than N days"),
    _: None = Depends(require_admin_token),
):
    """
    Delete evaluations older than N days.

    Use for retention management.
    """
    repo = _get_eval_repo()
    deleted = await repo.delete_older_than(days)

    return {
        "deleted": deleted,
        "retention_days": days,
    }
