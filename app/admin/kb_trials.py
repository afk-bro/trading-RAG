"""KB Trials Admin Endpoints (stats, collections, promotion).

Thin routing layer - business logic lives in app/services/kb_trials_admin.py
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.admin.utils import PaginationDefaults, require_db_pool
from app.deps.security import require_admin_token

router = APIRouter(tags=["admin"])
logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

ACTOR_TYPE_ADMIN = "admin"
ACTOR_ID_ADMIN = "admin"

# KB Status values
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"
STATUS_CANDIDATE = "candidate"
STATUS_ERROR = "error"

# Global connection pool (set during app startup)
_db_pool = None
_qdrant_client = None


def set_db_pool(pool):
    """Set the database pool for KB trials routes."""
    global _db_pool
    _db_pool = pool


def set_qdrant_client(client):
    """Set the Qdrant client for KB trials routes."""
    global _qdrant_client
    _qdrant_client = client


def _get_kb_trial_repo():
    """Get KBTrialRepository instance."""
    from app.repositories.kb_trials import KBTrialRepository

    if _qdrant_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant client not available",
        )
    return KBTrialRepository(_qdrant_client)


def _get_status_service():
    """Get KBStatusService instance."""
    from app.services.kb.status_service import KBStatusService

    require_db_pool(_db_pool)

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


# =============================================================================
# Bulk Transition Helper
# =============================================================================


@dataclass
class BulkTransitionConfig:
    """Configuration for bulk status transition."""

    to_status: str
    log_action: str
    require_reason: bool = False
    trigger_ingest: bool = False


async def _bulk_transition_handler(
    request: Request,
    config: BulkTransitionConfig,
    request_class: type,
    get_reason: Optional[Callable] = None,
    get_trigger_ingest: Optional[Callable] = None,
):
    """Handle bulk status transition for promote/reject/mark_candidate endpoints.

    Args:
        request: FastAPI request object
        config: Configuration for the transition
        request_class: Pydantic model class to parse request body
        get_reason: Optional function to extract reason from parsed request
        get_trigger_ingest: Optional function to extract trigger_ingest from request

    Returns:
        BulkStatusResponse with results
    """
    from app.admin.kb_trials_schemas import BulkStatusResponse, StatusChangeResult

    body = await request.json()
    req = request_class(**body)

    reason = get_reason(req) if get_reason else None
    trigger_ingest = get_trigger_ingest(req) if get_trigger_ingest else False

    if config.require_reason and not reason:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reason is required for rejection",
        )

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
                to_status=config.to_status,
                actor_type=ACTOR_TYPE_ADMIN,
                actor_id=ACTOR_ID_ADMIN,
                reason=reason,
                trigger_ingest=trigger_ingest if config.trigger_ingest else False,
            )

            if result.transitioned:
                updated += 1
                if config.trigger_ingest and trigger_ingest:
                    ingested += 1
            else:
                skipped += 1

            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    group_id=getattr(result, "group_id", None),
                    status=config.to_status,
                )
            )
        except Exception as e:
            errors += 1
            results.append(
                StatusChangeResult(
                    source_id=source_id,
                    status=STATUS_ERROR,
                    error=str(e),
                )
            )
            logger.warning(
                f"Failed to {config.log_action}",
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


# =============================================================================
# KB Trials Stats & Status Endpoints
# =============================================================================


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
    require_db_pool(_db_pool)

    from app.services.kb_trials_admin import compute_kb_trials_stats

    return await compute_kb_trials_stats(_db_pool, workspace_id, since, window_days)


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
    require_db_pool(_db_pool)

    from app.services.kb_trials_admin import compute_ingestion_status

    return await compute_ingestion_status(_db_pool, workspace_id)


# =============================================================================
# KB Collections Endpoints
# =============================================================================


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
    from app.config import get_settings
    from app.services.kb_trials_admin import get_qdrant_collections

    settings = get_settings()

    try:
        return await get_qdrant_collections(settings.qdrant_host, settings.qdrant_port)
    except Exception as e:
        logger.error("Failed to get Qdrant collections", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Qdrant: {str(e)}",
        )


# =============================================================================
# KB Warnings Endpoints
# =============================================================================


@router.get("/kb/warnings/top")
async def kb_top_warnings(
    request: Request,
    workspace_id: Optional[UUID] = Query(None, description="Filter by workspace"),
    limit: int = Query(
        PaginationDefaults.DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.MAX_LIMIT,
        description="Number of warnings to return",
    ),
    _: None = Depends(require_admin_token),
):
    """
    Get top warning types across KB trials.

    Useful for identifying systematic data quality issues.
    """
    require_db_pool(_db_pool)

    from app.services.kb_trials_admin import get_top_warnings

    return await get_top_warnings(_db_pool, workspace_id, limit)


# =============================================================================
# KB Trials Sample Endpoints
# =============================================================================


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
    limit: int = Query(
        PaginationDefaults.DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.MAX_LIMIT,
        description="Number of samples to return",
    ),
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
    require_db_pool(_db_pool)

    from app.services.kb_trials_admin import get_trial_samples

    return await get_trial_samples(
        _db_pool, workspace_id, warning, is_valid, has_oos, strategy_name, limit
    )


# =============================================================================
# KB Trials Promotion Endpoints
# =============================================================================


@router.get("/kb/trials/promotion-preview")
async def kb_trials_promotion_preview(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    source_type: Optional[str] = Query(None, description="Filter by source type"),
    group_id: Optional[UUID] = Query(
        None, description="Filter by group (tune_id or run_plan_id)"
    ),
    limit: int = Query(
        PaginationDefaults.DETAIL_DEFAULT_LIMIT,
        ge=1,
        le=PaginationDefaults.DETAIL_MAX_LIMIT,
    ),
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
    require_db_pool(_db_pool)

    from app.services.kb_trials_admin import compute_promotion_preview

    return await compute_promotion_preview(
        _db_pool,
        workspace_id,
        source_type,
        group_id,
        limit,
        offset,
        sort,
        include_ineligible,
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
    from app.admin.kb_trials_schemas import BulkStatusRequest

    config = BulkTransitionConfig(
        to_status=STATUS_PROMOTED,
        log_action="promote trial",
        trigger_ingest=True,
    )
    return await _bulk_transition_handler(
        request,
        config,
        BulkStatusRequest,
        get_reason=lambda req: req.reason,
        get_trigger_ingest=lambda req: req.trigger_ingest,
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
    from app.admin.kb_trials_schemas import BulkStatusRequest

    config = BulkTransitionConfig(
        to_status=STATUS_REJECTED,
        log_action="reject trial",
        require_reason=True,
    )
    return await _bulk_transition_handler(
        request,
        config,
        BulkStatusRequest,
        get_reason=lambda req: req.reason,
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
    from app.admin.kb_trials_schemas import MarkCandidateRequest

    config = BulkTransitionConfig(
        to_status=STATUS_CANDIDATE,
        log_action="mark candidate",
    )
    return await _bulk_transition_handler(
        request,
        config,
        MarkCandidateRequest,
    )
