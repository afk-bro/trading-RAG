"""Admin UI router for KB inspection and curation."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog


def _json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)
    return obj


from fastapi import (  # noqa: E402
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

from app.deps.security import require_admin_token  # noqa: E402
from app.admin import analytics as analytics_router  # noqa: E402
from app.admin import alerts as alerts_router  # noqa: E402
from app.admin import coverage as coverage_router  # noqa: E402
from app.admin import retention as retention_router  # noqa: E402
from app.admin import events as events_router  # noqa: E402
from app.admin import system_health as system_health_router  # noqa: E402
from app.admin import ops as ops_router  # noqa: E402
from app.admin import trade_events as trade_events_router  # noqa: E402
from app.admin import evals as evals_router  # noqa: E402
from app.admin import kb_admin as kb_admin_router  # noqa: E402
from app.admin import backtests as backtests_router  # noqa: E402
from app.admin import run_plans as run_plans_router  # noqa: E402
from app.admin import jobs as jobs_router  # noqa: E402

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger(__name__)

# Include analytics sub-router
router.include_router(analytics_router.router)

# Include alerts sub-router
router.include_router(alerts_router.router)

# Include coverage sub-router
router.include_router(coverage_router.router)

# Include retention sub-router
router.include_router(retention_router.router)

# Include events (SSE) sub-router
router.include_router(events_router.router)

# Include system health sub-router
router.include_router(system_health_router.router)

# Include ops sub-router
router.include_router(ops_router.router)

# Include trade events sub-router
router.include_router(trade_events_router.router)

# Include evals sub-router
router.include_router(evals_router.router)

# Include KB admin sub-router
router.include_router(kb_admin_router.router)

# Include backtests sub-router
router.include_router(backtests_router.router)

# Include run_plans sub-router
router.include_router(run_plans_router.router)

# Include jobs sub-router
router.include_router(jobs_router.router)

# Setup Jinja2 templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Global connection pool (set during app startup)
_db_pool = None
_qdrant_client = None


def set_db_pool(pool):
    """Set the database pool for admin routes."""
    global _db_pool
    _db_pool = pool
    # Also set pool for analytics router
    analytics_router.set_db_pool(pool)
    # Also set pool for alerts router
    alerts_router.set_db_pool(pool)
    # Also set pool for coverage router
    coverage_router.set_db_pool(pool)
    # Also set pool for retention router
    retention_router.set_db_pool(pool)
    # Also set pool for system health router
    system_health_router.set_db_pool(pool)
    # Also set pool for ops router
    ops_router.set_db_pool(pool)
    # Also set pool for trade events router
    trade_events_router.set_db_pool(pool)
    # Also set pool for evals router
    evals_router.set_db_pool(pool)
    # Also set pool for KB admin router
    kb_admin_router.set_db_pool(pool)
    # Also set pool for backtests router
    backtests_router.set_db_pool(pool)
    # Also set pool for run_plans router
    run_plans_router.set_db_pool(pool)
    # Also set pool for jobs router
    jobs_router.set_db_pool(pool)


def set_qdrant_client(client):
    """Set the Qdrant client for admin routes."""
    global _qdrant_client
    _qdrant_client = client


def _get_kb_repo():
    """Get KnowledgeBaseRepository instance."""
    from app.repositories.kb import KnowledgeBaseRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return KnowledgeBaseRepository(_db_pool)


# =============================================================================
# KB Trials Admin Endpoints
# =============================================================================


def _get_kb_trial_repo():
    """Get KBTrialRepository instance."""
    from app.repositories.kb_trials import KBTrialRepository

    if _qdrant_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant client not available",
        )
    return KBTrialRepository(_qdrant_client)


@router.get("/kb/trials/stats")
async def kb_trials_stats(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    since: Optional[datetime] = Query(
        None, description="Only count trials after this time"
    ),
    window_days: Optional[int] = Query(
        None, ge=1, le=365, description="Time window in days (7, 30, etc.)"
    ),
    _: None = Depends(require_admin_token),
):
    """
    Get KB trials statistics for a workspace.

    Returns point-in-time stats plus trend deltas when window_days is specified.

    **Query params:**
    - `since`: Only count trials created after this timestamp
    - `window_days`: Compare current vs previous window (e.g., 7 days)

    **Coverage metrics** explain why recommend might return none:
    - pct_with_regime_is: Has in-sample regime snapshot
    - pct_with_regime_oos: Has OOS regime snapshot (when has_oos=true)
    - pct_with_objective_score: Has objective_score set
    - pct_with_sharpe_oos: Has sharpe_oos metric (when has_oos=true)
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    # Calculate time boundaries
    now = datetime.utcnow()
    if window_days:
        since = now - timedelta(days=window_days)
        prev_since = since - timedelta(days=window_days)
    else:
        prev_since = None

    async with _db_pool.acquire() as conn:
        # Build query conditions
        workspace_cond = "AND t.workspace_id = $1" if workspace_id else ""
        base_params = [workspace_id] if workspace_id else []

        async def get_stats(time_filter: str = "", time_params: list = []):
            """Get stats with optional time filter."""
            params = base_params + time_params

            # Total trials
            total = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE 1=1 {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            if total == 0:
                return {
                    "total": 0,
                    "with_oos": 0,
                    "valid": 0,
                    "stale": 0,
                    "with_regime_is": 0,
                    "with_regime_oos": 0,
                    "with_objective_score": 0,
                    "with_sharpe_oos": 0,
                }

            # Core metrics
            with_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            valid = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.is_valid = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            # Stale count
            try:
                stale = (
                    await conn.fetchval(
                        f"""
                    SELECT COUNT(*) FROM kb_trial_vectors t
                    WHERE t.needs_reembed = true {workspace_cond} {time_filter}
                """,
                        *params,
                    )
                    or 0
                )
            except Exception:
                stale = 0

            # Coverage metrics
            with_regime_is = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_is IS NOT NULL {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_regime_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_objective = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.objective_score IS NOT NULL {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_sharpe_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.sharpe_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            return {
                "total": total,
                "with_oos": with_oos,
                "valid": valid,
                "stale": stale,
                "with_regime_is": with_regime_is,
                "with_regime_oos": with_regime_oos,
                "with_objective_score": with_objective,
                "with_sharpe_oos": with_sharpe_oos,
            }

        # Get current stats
        if since:
            param_idx = len(base_params) + 1
            time_filter = f"AND t.created_at >= ${param_idx}"
            current = await get_stats(time_filter, [since])
        else:
            current = await get_stats()

        # Get previous window stats for deltas
        deltas = None
        if prev_since and window_days:
            param_idx = len(base_params) + 1
            time_filter = (
                f"AND t.created_at >= ${param_idx} AND t.created_at < ${param_idx + 1}"
            )
            previous = await get_stats(time_filter, [prev_since, since])

            # Calculate deltas
            deltas = {
                "trials_added": current["total"] - previous["total"],
                "valid_added": current["valid"] - previous["valid"],
                "stale_added": current["stale"] - previous["stale"],
                "pct_valid_delta": round(
                    (
                        current["valid"] / current["total"] * 100
                        if current["total"] > 0
                        else 0
                    )
                    - (
                        previous["valid"] / previous["total"] * 100
                        if previous["total"] > 0
                        else 0
                    ),
                    1,
                ),
                "window_days": window_days,
                "prev_window_start": prev_since.isoformat(),
                "prev_window_end": since.isoformat(),
            }

        # Last ingestion timestamp
        last_ts = await conn.fetchval(
            f"""
            SELECT MAX(t.created_at) FROM kb_trial_vectors t
            WHERE 1=1 {workspace_cond}
        """,
            *base_params,
        )

        # Workspace config for embedding info
        embed_model = "nomic-embed-text"
        embed_dim = 768
        collection_name = "trading_kb_trials__nomic-embed-text__768"

        if workspace_id:
            config_row = await conn.fetchrow(
                """
                SELECT config FROM workspaces WHERE id = $1
            """,
                workspace_id,
            )
            if config_row and config_row["config"]:
                config = config_row["config"]
                if isinstance(config, str):
                    config = json.loads(config)
                kb_config = config.get("kb", {})
                embed_model = kb_config.get("embed_model", embed_model)
                embed_dim = kb_config.get("embed_dim", embed_dim)
                collection_name = kb_config.get("collection_name", collection_name)

    # Calculate percentages
    total = current["total"]
    oos_count = current["with_oos"]

    def pct(num, denom):
        return round(num / denom * 100, 1) if denom > 0 else 0

    result = {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "total_trials": total,
        "trials_with_oos": oos_count,
        "trials_valid": current["valid"],
        "pct_with_oos": pct(oos_count, total),
        "pct_valid": pct(current["valid"], total),
        # Coverage metrics (explains why recommend might fail)
        "coverage": {
            "pct_with_regime_is": pct(current["with_regime_is"], total),
            "pct_with_regime_oos": pct(current["with_regime_oos"], oos_count),
            "pct_with_objective_score": pct(current["with_objective_score"], total),
            "pct_with_sharpe_oos": pct(current["with_sharpe_oos"], oos_count),
        },
        # Embedding config
        "embedding_model": embed_model,
        "embedding_dim": embed_dim,
        "collection_name": collection_name,
        "last_ingestion_ts": last_ts.isoformat() if last_ts else None,
        "stale_text_hash_count": current["stale"],
    }

    # Add time window info
    if since:
        result["since"] = since.isoformat()
    if deltas:
        result["deltas"] = deltas

    return result


@router.get("/kb/trials/ingestion-status")
async def kb_ingestion_status(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    _: None = Depends(require_admin_token),
):
    """
    Get KB ingestion health status.

    Returns:
    - trials_missing_vectors: Trials without embeddings
    - trials_missing_regime: Trials without regime snapshots
    - warning_counts: Top warning types and counts
    - recent_ingestion_runs: Last 10 ingestion runs
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        workspace_filter = "AND workspace_id = $1" if workspace_id else ""
        params = [workspace_id] if workspace_id else []

        # Missing vectors
        missing_vectors_query = f"""
            SELECT COUNT(*) as count
            FROM kb_trial_vectors
            WHERE vector IS NULL {workspace_filter}
        """
        try:
            missing_vectors = await conn.fetchval(missing_vectors_query, *params)
        except Exception:
            missing_vectors = 0

        # Missing regime
        missing_regime_query = f"""
            SELECT COUNT(*) as count
            FROM kb_trial_vectors
            WHERE regime_snapshot IS NULL {workspace_filter}
        """
        try:
            missing_regime = await conn.fetchval(missing_regime_query, *params)
        except Exception:
            missing_regime = 0

        # Warning counts
        warning_counts_query = f"""
            SELECT
                warning,
                COUNT(*) as count
            FROM kb_trial_vectors,
                 LATERAL unnest(warnings) as warning
            WHERE 1=1 {workspace_filter}
            GROUP BY warning
            ORDER BY count DESC
            LIMIT 20
        """
        try:
            warning_rows = await conn.fetch(warning_counts_query, *params)
            warning_counts = {row["warning"]: row["count"] for row in warning_rows}
        except Exception:
            warning_counts = {}

        # Recent ingestion runs (if tracked)
        recent_runs = []
        try:
            runs_query = f"""
                SELECT
                    id,
                    created_at,
                    ingested_count,
                    skipped_count,
                    error_count,
                    duration_ms
                FROM kb_ingestion_runs
                WHERE 1=1 {workspace_filter}
                ORDER BY created_at DESC
                LIMIT 10
            """
            run_rows = await conn.fetch(runs_query, *params)
            recent_runs = [
                {
                    "id": str(row["id"]),
                    "created_at": row["created_at"].isoformat(),
                    "ingested_count": row["ingested_count"],
                    "skipped_count": row["skipped_count"],
                    "error_count": row["error_count"],
                    "duration_ms": row["duration_ms"],
                }
                for row in run_rows
            ]
        except Exception:
            # Table might not exist yet
            pass

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "trials_missing_vectors": missing_vectors or 0,
        "trials_missing_regime": missing_regime or 0,
        "warning_counts": warning_counts,
        "recent_ingestion_runs": recent_runs,
    }


@router.get("/kb/collections")
async def kb_collections(
    request: Request,
    _: None = Depends(require_admin_token),
):
    """
    Get list of KB collections in Qdrant with health info.

    Returns:
    - Collection names and point counts
    - Vector dimension and distance metric
    - Payload index counts
    - Optimizer status
    """
    from qdrant_client import AsyncQdrantClient
    from app.config import get_settings

    settings = get_settings()

    try:
        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Get all collections
        collections_response = await client.get_collections()

        result = []
        for coll in collections_response.collections:
            try:
                info = await client.get_collection(coll.name)

                # Parse embedding model from collection name if encoded
                # Format: trading_kb_trials__{model}__{dim}
                parts = coll.name.split("__")
                embedding_model = parts[1] if len(parts) >= 2 else None
                embedding_dim = int(parts[2]) if len(parts) >= 3 else None

                # Vector config
                vec_cfg = info.config.params.vectors
                vector_size = vec_cfg.size if vec_cfg else None
                distance = (
                    vec_cfg.distance.value if vec_cfg and vec_cfg.distance else None
                )

                # Payload indexes count
                payload_indexes = 0
                if info.payload_schema:
                    payload_indexes = len(info.payload_schema)

                # Optimizer status
                optimizer_status = "unknown"
                if info.optimizer_status:
                    optimizer_status = (
                        info.optimizer_status.status.value
                        if hasattr(info.optimizer_status, "status")
                        else str(info.optimizer_status)
                    )

                result.append(
                    {
                        "name": coll.name,
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count,
                        "status": info.status.value if info.status else "unknown",
                        "vector_size": vector_size or embedding_dim,
                        "distance": distance,
                        "embedding_model_id": embedding_model,
                        "payload_indexes_count": payload_indexes,
                        "optimizer_status": optimizer_status,
                        "segments_count": (
                            len(info.segments or [])
                            if hasattr(info, "segments")
                            else None
                        ),
                    }
                )
            except Exception as e:
                result.append(
                    {
                        "name": coll.name,
                        "error": str(e),
                    }
                )

        await client.close()

        return {
            "collections": result,
            "qdrant_host": settings.qdrant_host,
            "qdrant_port": settings.qdrant_port,
            "total_collections": len(result),
        }

    except Exception as e:
        logger.error("Failed to get Qdrant collections", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Qdrant: {str(e)}",
        )


@router.get("/kb/warnings/top")
async def kb_top_warnings(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    limit: int = Query(20, ge=1, le=100, description="Number of warnings to return"),
    _: None = Depends(require_admin_token),
):
    """
    Get top warning types across KB trials.

    Useful for identifying systematic data quality issues.
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        workspace_filter = "AND workspace_id = $1" if workspace_id else ""
        params = [workspace_id, limit] if workspace_id else [limit]
        limit_param = "$2" if workspace_id else "$1"

        query = f"""
            SELECT
                warning,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM kb_trial_vectors WHERE 1=1 {workspace_filter}) as pct  # noqa: E501
            FROM kb_trial_vectors,
                 LATERAL unnest(warnings) as warning
            WHERE 1=1 {workspace_filter}
            GROUP BY warning
            ORDER BY count DESC
            LIMIT {limit_param}
        """

        try:
            rows = await conn.fetch(query, *params)
            warnings = [
                {
                    "warning": row["warning"],
                    "count": row["count"],
                    "pct": round(row["pct"], 2) if row["pct"] else 0,
                }
                for row in rows
            ]
        except Exception:
            warnings = []

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "warnings": warnings,
    }


@router.get("/kb/trials/sample")
async def kb_trials_sample(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    warning: Optional[str] = Query(
        None, description="Filter by warning type (e.g., high_overfit)"
    ),
    is_valid: Optional[bool] = Query(None, description="Filter by validity"),
    has_oos: Optional[bool] = Query(None, description="Filter by OOS availability"),
    strategy_name: Optional[str] = Query(None, description="Filter by strategy"),
    limit: int = Query(20, ge=1, le=100, description="Number of samples to return"),
    _: None = Depends(require_admin_token),
):
    """
    Get sample trials for debugging quality issues.

    Returns safe fields only - no sensitive internals.
    Useful for inspecting what's actually in the KB.

    **Example use cases:**
    - `?warning=high_overfit` - See trials flagged as overfit
    - `?is_valid=false` - See invalid trials
    - `?has_oos=true&strategy_name=mean_reversion` - See valid OOS trials for a strategy
    """
    if not _db_pool:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )

    async with _db_pool.acquire() as conn:
        # Build dynamic WHERE clause
        conditions = ["1=1"]
        params = []
        param_idx = 1

        if workspace_id:
            conditions.append(f"workspace_id = ${param_idx}")
            params.append(workspace_id)
            param_idx += 1

        if warning:
            conditions.append(f"${param_idx} = ANY(warnings)")
            params.append(warning)
            param_idx += 1

        if is_valid is not None:
            conditions.append(f"is_valid = ${param_idx}")
            params.append(is_valid)
            param_idx += 1

        if has_oos is not None:
            conditions.append(f"has_oos_metrics = ${param_idx}")
            params.append(has_oos)
            param_idx += 1

        if strategy_name:
            conditions.append(f"strategy_name = ${param_idx}")
            params.append(strategy_name)
            param_idx += 1

        # Add limit
        params.append(limit)
        limit_param = f"${param_idx}"

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                id,
                point_id,
                tune_run_id,
                strategy_name,
                objective_type,
                objective_score,
                is_valid,
                has_oos_metrics,
                overfit_gap,
                sharpe_is,
                sharpe_oos,
                trades_is,
                trades_oos,
                max_drawdown_is,
                max_drawdown_oos,
                regime_tags_is,
                regime_tags_oos,
                warnings,
                created_at
            FROM kb_trial_vectors
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit_param}
        """

        try:
            rows = await conn.fetch(query, *params)
            samples = [
                {
                    "id": str(row["id"]),
                    "point_id": row["point_id"],
                    "tune_run_id": (
                        str(row["tune_run_id"]) if row["tune_run_id"] else None
                    ),
                    "strategy_name": row["strategy_name"],
                    "objective_type": row["objective_type"],
                    "objective_score": row["objective_score"],
                    "is_valid": row["is_valid"],
                    "has_oos_metrics": row["has_oos_metrics"],
                    "overfit_gap": row["overfit_gap"],
                    "metrics": {
                        "sharpe_is": row["sharpe_is"],
                        "sharpe_oos": row["sharpe_oos"],
                        "trades_is": row["trades_is"],
                        "trades_oos": row["trades_oos"],
                        "max_drawdown_is": row["max_drawdown_is"],
                        "max_drawdown_oos": row["max_drawdown_oos"],
                    },
                    "regime_tags_is": row["regime_tags_is"],
                    "regime_tags_oos": row["regime_tags_oos"],
                    "warnings": row["warnings"],
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to fetch KB samples", error=str(e))
            samples = []

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "filters": {
            "warning": warning,
            "is_valid": is_valid,
            "has_oos": has_oos,
            "strategy_name": strategy_name,
        },
        "count": len(samples),
        "samples": samples,
    }


# ===========================================
# KB Trials Promotion Endpoints
# ===========================================


def _get_status_service():
    """Get KBStatusService instance."""
    from app.services.kb.status_service import KBStatusService

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    # Create repository adapters
    from app.admin.kb_trials_repos import (
        AdminKBStatusRepository,
        AdminKBIndexRepository,
    )

    status_repo = AdminKBStatusRepository(_db_pool)
    index_repo = AdminKBIndexRepository(_db_pool)

    return KBStatusService(
        status_repo=status_repo,
        index_repo=index_repo,
    )


@router.get("/kb/trials/promotion-preview")
async def kb_trials_promotion_preview(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    group_id: Optional[UUID] = Query(
        None, description="Filter by group (tune_id or run_plan_id)"
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: str = Query("sharpe_oos", description="Sort field"),
    include_ineligible: bool = Query(False, description="Include ineligible trials"),
    _: bool = Depends(require_admin_token),
):
    """
    Preview trials for promotion consideration.

    Returns trials that could be promoted with eligibility analysis.
    Uses the same candidacy logic as auto-promotion to ensure consistency.
    """
    from app.admin.kb_trials_schemas import (
        PromotionPreviewResponse,
        PromotionPreviewSummary,
        TrialPreviewItem,
    )
    from app.services.kb.candidacy import (
        CandidacyConfig,
        is_candidate,
        VariantMetricsForCandidacy,
        KNOWN_EXPERIMENT_TYPES,
    )
    from app.services.kb.types import RegimeSnapshot

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )

    # Build query for eligible trials view
    query = """
        SELECT
            source_type,
            experiment_type,
            source_id,
            group_id,
            workspace_id,
            strategy_name,
            params,
            trial_status,
            regime_is,
            regime_oos,
            regime_schema_version,
            sharpe_oos,
            return_frac_oos,
            max_dd_frac_oos,
            n_trades_oos,
            sharpe_is,
            kb_status,
            kb_promoted_at,
            objective_type,
            objective_score,
            created_at
        FROM kb_eligible_trials
        WHERE workspace_id = $1
    """
    params = [workspace_id]
    param_idx = 2

    if source_type:
        query += f" AND source_type = ${param_idx}"
        params.append(source_type)
        param_idx += 1

    if group_id:
        query += f" AND group_id = ${param_idx}"
        params.append(group_id)
        param_idx += 1

    # Also include excluded/candidate trials if include_ineligible
    if include_ineligible:
        # Get trials not in eligible view too
        query = f"""
            WITH eligible AS ({query})
            SELECT * FROM eligible
            UNION ALL
            SELECT
                'test_variant'::TEXT AS source_type,
                COALESCE(r.summary->>'experiment_type', 'sweep')::TEXT AS experiment_type,
                r.id AS source_id,
                r.run_plan_id AS group_id,
                r.workspace_id,
                COALESCE(r.summary->>'strategy_name', e.name) AS strategy_name,
                r.params,
                r.status AS trial_status,
                r.regime_is,
                r.regime_oos,
                r.regime_schema_version,
                (r.summary->>'sharpe')::FLOAT AS sharpe_oos,
                (r.summary->>'return_pct')::FLOAT AS return_frac_oos,
                (r.summary->>'max_drawdown_pct')::FLOAT AS max_dd_frac_oos,
                (r.summary->>'trade_count')::INT AS n_trades_oos,
                NULL::FLOAT AS sharpe_is,
                r.kb_status,
                r.kb_promoted_at,
                COALESCE(rp.objective_name, 'sharpe') AS objective_type,
                r.objective_score,
                r.created_at
            FROM backtest_runs r
            LEFT JOIN kb_entities e ON r.strategy_entity_id = e.id
            LEFT JOIN run_plans rp ON r.run_plan_id = rp.id
            WHERE r.workspace_id = $1
              AND r.run_kind = 'test_variant'
              AND r.kb_status = 'excluded'
              AND r.status IN ('completed', 'success')
        """

    # Add sorting
    sort_column = "sharpe_oos"
    if sort in ("sharpe_oos", "return_frac_oos", "max_dd_frac_oos", "created_at"):
        sort_column = sort

    query += f" ORDER BY {sort_column} DESC NULLS LAST, created_at DESC, source_id"
    query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
    params.extend([limit, offset])

    async with _db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        # Get total count
        count_query = """
            SELECT COUNT(*) FROM kb_eligible_trials
            WHERE workspace_id = $1
        """
        count_params = [workspace_id]
        if source_type:
            count_query += " AND source_type = $2"
            count_params.append(source_type)
        if group_id:
            count_query += f" AND group_id = ${len(count_params) + 1}"
            count_params.append(group_id)

        total = await conn.fetchval(count_query, *count_params)

    # Process rows with candidacy check
    trials = []
    summary = PromotionPreviewSummary()
    config = CandidacyConfig()

    for row in rows:
        regime_is = None
        regime_oos = None
        if row["regime_is"]:
            regime_is = RegimeSnapshot.from_dict(row["regime_is"])
        if row["regime_oos"]:
            regime_oos = RegimeSnapshot.from_dict(row["regime_oos"])

        # Check candidacy gates
        ineligibility_reasons = []
        passes_gates = False

        experiment_type = row["experiment_type"] or "sweep"

        if experiment_type not in KNOWN_EXPERIMENT_TYPES:
            ineligibility_reasons.append("unknown_experiment_type")
        elif experiment_type == "manual":
            ineligibility_reasons.append("manual_experiment_excluded")
        else:
            # Build metrics for candidacy check
            metrics = VariantMetricsForCandidacy(
                sharpe_oos=row["sharpe_oos"],
                return_frac_oos=row["return_frac_oos"],
                max_dd_frac_oos=row["max_dd_frac_oos"],
                n_trades_oos=row["n_trades_oos"],
                overfit_gap=(
                    max(0, (row["sharpe_is"] or 0) - (row["sharpe_oos"] or 0))
                    if row["sharpe_is"] is not None and row["sharpe_oos"] is not None
                    else None
                ),
            )

            decision = is_candidate(
                metrics=metrics,
                regime_oos=regime_oos,
                experiment_type=experiment_type,
                config=config,
            )

            passes_gates = decision.eligible
            if not decision.eligible:
                ineligibility_reasons.append(decision.reason)

        # Determine promotion eligibility
        kb_status = row["kb_status"]
        can_promote = kb_status not in ("promoted", "rejected")
        is_eligible = passes_gates and can_promote

        # Update summary counts
        if kb_status == "promoted":
            summary.already_promoted += 1
        elif not regime_oos and kb_status != "promoted":
            summary.missing_regime += 1
        elif is_eligible:
            summary.would_promote += 1
        else:
            summary.would_skip += 1

        trials.append(
            TrialPreviewItem(
                source_type=row["source_type"],
                source_id=row["source_id"],
                group_id=row["group_id"],
                experiment_type=experiment_type,
                strategy_name=row["strategy_name"],
                kb_status=kb_status,
                sharpe_oos=row["sharpe_oos"],
                return_frac_oos=row["return_frac_oos"],
                max_dd_frac_oos=row["max_dd_frac_oos"],
                n_trades_oos=row["n_trades_oos"],
                passes_auto_gates=passes_gates,
                can_promote=can_promote,
                is_eligible=is_eligible,
                ineligibility_reasons=ineligibility_reasons,
                has_regime_is=regime_is is not None,
                has_regime_oos=regime_oos is not None,
                regime_schema_version=row["regime_schema_version"],
                created_at=row["created_at"],
            )
        )

    return PromotionPreviewResponse(
        summary=summary,
        pagination={
            "limit": limit,
            "offset": offset,
            "total": total or 0,
        },
        trials=trials,
    )


@router.post("/kb/trials/promote")
async def kb_trials_promote(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """
    Bulk promote trials to 'promoted' status.

    Transitions trials from excluded/candidate to promoted.
    Optionally triggers ingestion for newly promoted trials.
    """
    from app.admin.kb_trials_schemas import (
        BulkStatusRequest,
        BulkStatusResponse,
        StatusChangeResult,
    )

    body = await request.json()
    req = BulkStatusRequest(**body)

    service = _get_status_service()
    results = []
    updated = 0
    skipped = 0
    ingested = 0
    errors = 0

    for source_id in req.source_ids:
        try:
            result = await service.transition(
                source_type=req.source_type,
                source_id=source_id,
                to_status="promoted",
                actor_type="admin",
                actor_id="admin",  # Could be extracted from auth token
                reason=req.reason,
                trigger_ingest=req.trigger_ingest,
            )

            if result.transitioned:
                updated += 1
                if req.trigger_ingest:
                    ingested += 1
            else:
                skipped += 1

            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    group_id=result.group_id if hasattr(result, "group_id") else None,
                    status="promoted",
                )
            )
        except Exception as e:
            errors += 1
            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="error",
                    error=str(e),
                )
            )
            logger.warning(
                "Failed to promote trial",
                source_type=req.source_type,
                source_id=str(source_id),
                error=str(e),
            )

    return BulkStatusResponse(
        updated=updated,
        skipped=skipped,
        ingested=ingested,
        errors=errors,
        results=results,
    )


@router.post("/kb/trials/reject")
async def kb_trials_reject(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """
    Bulk reject trials.

    Transitions trials to 'rejected' status. Requires a reason.
    Archives trials from Qdrant index.
    """
    from app.admin.kb_trials_schemas import (
        BulkStatusRequest,
        BulkStatusResponse,
        StatusChangeResult,
    )

    body = await request.json()
    req = BulkStatusRequest(**body)

    if not req.reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reason is required for rejection",
        )

    service = _get_status_service()
    results = []
    updated = 0
    skipped = 0
    errors = 0

    for source_id in req.source_ids:
        try:
            result = await service.transition(
                source_type=req.source_type,
                source_id=source_id,
                to_status="rejected",
                actor_type="admin",
                actor_id="admin",
                reason=req.reason,
            )

            if result.transitioned:
                updated += 1
            else:
                skipped += 1

            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="rejected",
                )
            )
        except Exception as e:
            errors += 1
            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="error",
                    error=str(e),
                )
            )
            logger.warning(
                "Failed to reject trial",
                source_type=req.source_type,
                source_id=str(source_id),
                error=str(e),
            )

    return BulkStatusResponse(
        updated=updated,
        skipped=skipped,
        ingested=0,
        errors=errors,
        results=results,
    )


@router.post("/kb/trials/mark-candidate")
async def kb_trials_mark_candidate(
    request: Request,
    _: bool = Depends(require_admin_token),
):
    """
    Mark trials as candidates.

    Transitions excluded trials to 'candidate' status.
    Does not trigger ingestion - use the ingestion endpoint for that.
    """
    from app.admin.kb_trials_schemas import (
        MarkCandidateRequest,
        BulkStatusResponse,
        StatusChangeResult,
    )

    body = await request.json()
    req = MarkCandidateRequest(**body)

    service = _get_status_service()
    results = []
    updated = 0
    skipped = 0
    errors = 0

    for source_id in req.source_ids:
        try:
            result = await service.transition(
                source_type=req.source_type,
                source_id=source_id,
                to_status="candidate",
                actor_type="admin",
                actor_id="admin",
            )

            if result.transitioned:
                updated += 1
            else:
                skipped += 1

            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="candidate",
                )
            )
        except Exception as e:
            errors += 1
            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status="error",
                    error=str(e),
                )
            )
            logger.warning(
                "Failed to mark candidate",
                source_type=req.source_type,
                source_id=str(source_id),
                error=str(e),
            )

    return BulkStatusResponse(
        updated=updated,
        skipped=skipped,
        ingested=0,
        errors=errors,
        results=results,
    )
