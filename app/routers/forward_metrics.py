"""
Forward metrics endpoint for v1.5 expected-vs-realized tracking.

Receives streaming realized metrics from forward runs (paper/shadow/live).
Designed per v1.5 Live Intelligence design doc.
"""

from datetime import datetime
from uuid import UUID

import structlog
from asyncpg.exceptions import UniqueViolationError
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.repositories.recommendation_records import (
    RecommendationRecordsRepository,
    RecommendationObservation,
    RecordStatus,
)

router = APIRouter(prefix="/forward", tags=["forward-metrics"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_repository() -> RecommendationRecordsRepository:
    """Get repository instance."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return RecommendationRecordsRepository(_db_pool)


# =============================================================================
# Request/Response Schemas
# =============================================================================


class ForwardMetricsRequest(BaseModel):
    """
    Request to submit forward metrics.

    Expected metrics fields in realized_metrics:
    - return_pct: float
    - sharpe_proxy: float
    - hit_rate: float
    - max_drawdown_pct: float
    - expectancy: float
    """

    workspace_id: UUID = Field(..., description="Workspace ID")
    record_id: UUID = Field(..., description="Recommendation record ID")
    ts: datetime = Field(..., description="Timestamp of the observation")
    bars_seen: int = Field(..., ge=0, description="Number of bars seen in forward run")
    trades_seen: int = Field(
        ..., ge=0, description="Number of trades executed in forward run"
    )
    realized_metrics: dict = Field(
        ...,
        description="Realized metrics (return_pct, sharpe_proxy, hit_rate, max_drawdown_pct, expectancy)",
    )

    model_config = {"extra": "forbid"}


class ForwardMetricsResponse(BaseModel):
    """Response from forward metrics submission."""

    status: str = Field(..., description="Acceptance status")
    observation_id: str = Field(..., description="ID of created observation")


# =============================================================================
# Endpoint
# =============================================================================


@router.post(
    "/metrics",
    response_model=ForwardMetricsResponse,
    responses={
        200: {"description": "Metrics accepted"},
        404: {
            "description": "Record not found or not active",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "error": "Record not found",
                            "code": "RECORD_NOT_FOUND",
                        }
                    }
                }
            },
        },
        409: {
            "description": "Duplicate observation (record_id, ts)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "error": "Duplicate observation for (record_id, ts)",
                            "code": "DUPLICATE_OBSERVATION",
                        }
                    }
                }
            },
        },
        503: {"description": "Service unavailable (database not connected)"},
    },
    summary="Submit forward run metrics",
    description="""
Submit realized metrics from a forward run (paper/shadow/live).

**Guarantees:**
- Idempotency via (record_id, ts) unique constraint
- Returns 409 Conflict if duplicate
- Returns 404 if record_id not found or not active

**Metrics fields:**
- `return_pct` - Return percentage
- `sharpe_proxy` - Rolling Sharpe approximation
- `hit_rate` - Win rate fraction
- `max_drawdown_pct` - Maximum drawdown percentage
- `expectancy` - Expected value per trade
""",
)
async def submit_metrics(request: ForwardMetricsRequest) -> ForwardMetricsResponse:
    """Submit forward run metrics."""
    repo = _get_repository()

    logger.debug(
        "forward_metrics_request",
        record_id=str(request.record_id),
        ts=request.ts.isoformat(),
        bars_seen=request.bars_seen,
        trades_seen=request.trades_seen,
    )

    # Check record exists and is active
    record = await repo.get_record_by_id(request.record_id)
    if record is None:
        logger.warning(
            "forward_metrics_record_not_found",
            record_id=str(request.record_id),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Record not found", "code": "RECORD_NOT_FOUND"},
        )

    if record.status != RecordStatus.ACTIVE:
        logger.warning(
            "forward_metrics_record_not_active",
            record_id=str(request.record_id),
            status=(
                record.status.value
                if hasattr(record.status, "value")
                else str(record.status)
            ),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": f"Record is not active (status={record.status.value if hasattr(record.status, 'value') else record.status})",
                "code": "RECORD_NOT_ACTIVE",
            },
        )

    # Create observation
    obs = RecommendationObservation(
        record_id=request.record_id,
        ts=request.ts,
        bars_seen=request.bars_seen,
        trades_seen=request.trades_seen,
        realized_metrics_json=request.realized_metrics,
    )

    try:
        obs_id = await repo.add_observation(obs)
    except UniqueViolationError:
        logger.warning(
            "forward_metrics_duplicate",
            record_id=str(request.record_id),
            ts=request.ts.isoformat(),
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "Duplicate observation for (record_id, ts)",
                "code": "DUPLICATE_OBSERVATION",
            },
        )
    except Exception as e:
        # Check for unique violation in error message (generic exception)
        error_str = str(e).lower()
        if "unique" in error_str or "duplicate" in error_str:
            logger.warning(
                "forward_metrics_duplicate",
                record_id=str(request.record_id),
                ts=request.ts.isoformat(),
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": "Duplicate observation for (record_id, ts)",
                    "code": "DUPLICATE_OBSERVATION",
                },
            )
        # Re-raise other exceptions
        logger.error(
            "forward_metrics_error",
            record_id=str(request.record_id),
            ts=request.ts.isoformat(),
            error=str(e),
        )
        raise

    logger.info(
        "forward_metrics_accepted",
        observation_id=str(obs_id),
        record_id=str(request.record_id),
        ts=request.ts.isoformat(),
        bars_seen=request.bars_seen,
        trades_seen=request.trades_seen,
    )

    return ForwardMetricsResponse(
        status="accepted",
        observation_id=str(obs_id),
    )
