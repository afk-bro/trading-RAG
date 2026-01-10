"""Trade execution endpoints."""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.schemas import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionMode,
    PaperState,
    PaperPosition,
    ReconciliationResult,
)
from app.deps.security import require_admin_token
from app.services.execution.factory import get_paper_broker
from app.repositories.trade_events import TradeEventsRepository


router = APIRouter(prefix="/execute", tags=["execution"])
logger = structlog.get_logger(__name__)

# Global state
_db_pool = None


def set_db_pool(pool):
    """Set database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_events_repo() -> TradeEventsRepository:
    """Get events repository."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return TradeEventsRepository(_db_pool)


@router.post(
    "/intents",
    response_model=ExecutionResult,
    responses={
        200: {"description": "Intent executed successfully"},
        400: {
            "description": "Validation error (unsupported action, invalid quantity, etc.)"
        },
        409: {"description": "Intent already executed (idempotency)"},
        503: {"description": "Service unavailable"},
    },
)
async def execute_intent(
    request: ExecutionRequest,
    _: None = Depends(require_admin_token),
) -> ExecutionResult:
    """
    Execute a trade intent.

    The execution layer:
    1. Validates action is supported (OPEN_LONG, CLOSE_LONG only)
    2. Checks idempotency (rejects if intent_id already executed)
    3. Re-evaluates policy internally (does NOT trust caller)
    4. Executes and journals ORDER_FILLED + POSITION_* events

    The caller must provide fill_price - execution emits facts, not guesses.

    Returns 409 Conflict if the intent was already executed.
    """
    log = logger.bind(
        intent_id=str(request.intent.id),
        correlation_id=request.intent.correlation_id,
        action=request.intent.action.value,
        symbol=request.intent.symbol,
        mode=request.mode.value,
    )
    log.info("execute_intent_request")

    # Only paper mode supported in PR1
    if request.mode != ExecutionMode.PAPER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Only paper mode supported",
                "error_code": "UNSUPPORTED_MODE",
            },
        )

    # Get paper broker
    events_repo = _get_events_repo()
    broker = get_paper_broker(events_repo)

    # Execute
    result = await broker.execute_intent(
        intent=request.intent,
        fill_price=request.fill_price,
    )

    # Map error codes to HTTP status
    if not result.success:
        if result.error_code == "ALREADY_EXECUTED":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": result.error,
                    "error_code": result.error_code,
                    "prior_correlation_id": result.correlation_id,
                },
            )
        elif result.error_code in {
            "UNSUPPORTED_ACTION",
            "INVALID_FILL_PRICE",
            "INVALID_QUANTITY",
            "NO_POSITION",
            "PARTIAL_CLOSE_NOT_SUPPORTED",
        }:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": result.error,
                    "error_code": result.error_code,
                },
            )
        elif result.error_code == "POLICY_REJECTED":
            # Policy rejection is not an HTTP error - return success=false
            log.info("policy_rejected", error=result.error)
            return result
        else:
            # Catch-all for unhandled error codes
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": result.error,
                    "error_code": result.error_code,
                },
            )

    log.info(
        "execute_intent_complete",
        success=result.success,
        order_id=str(result.order_id) if result.order_id else None,
        fill_price=result.fill_price,
        position_action=result.position_action,
    )

    return result


@router.get(
    "/paper/state/{workspace_id}",
    response_model=PaperState,
)
async def get_paper_state(
    workspace_id: UUID,
    _: None = Depends(require_admin_token),
) -> PaperState:
    """
    Get current paper trading state for workspace.

    Returns cash, positions, and tracking info.
    Creates fresh state if workspace hasn't been used yet.
    """
    events_repo = _get_events_repo()
    broker = get_paper_broker(events_repo)
    return await broker.get_state(workspace_id)


@router.get(
    "/paper/positions/{workspace_id}",
    response_model=list[PaperPosition],
)
async def get_paper_positions(
    workspace_id: UUID,
    _: None = Depends(require_admin_token),
) -> list[PaperPosition]:
    """
    Get open paper positions for workspace.

    Only returns positions with quantity > 0.
    """
    events_repo = _get_events_repo()
    broker = get_paper_broker(events_repo)
    return await broker.get_positions(workspace_id)


@router.post(
    "/paper/reconcile/{workspace_id}",
    response_model=ReconciliationResult,
)
async def reconcile_paper_state(
    workspace_id: UUID,
    _: None = Depends(require_admin_token),
) -> ReconciliationResult:
    """
    Reconcile paper state from journal.

    Rebuilds in-memory state by replaying ORDER_FILLED events.
    POSITION_* events are observability breadcrumbs only.

    Deduplicates by order_id to handle duplicate journal entries.

    Returns errors for any inconsistencies (e.g., SELL qty != position qty).
    """
    log = logger.bind(workspace_id=str(workspace_id))
    log.info("reconcile_request")

    events_repo = _get_events_repo()
    broker = get_paper_broker(events_repo)
    result = await broker.reconcile_from_journal(workspace_id)

    log.info(
        "reconcile_complete",
        success=result.success,
        events_replayed=result.events_replayed,
        positions_rebuilt=result.positions_rebuilt,
        errors_count=len(result.errors),
    )

    return result


@router.post(
    "/paper/reset/{workspace_id}",
    responses={
        200: {"description": "State reset"},
        403: {"description": "Not allowed in production"},
    },
)
async def reset_paper_state(
    workspace_id: UUID,
    _: None = Depends(require_admin_token),
) -> dict:
    """
    Reset paper state (development only).

    WARNING: This only clears in-memory state. Journal events
    are immutable and will cause state to be rebuilt on next
    reconciliation.

    Disabled in production (config_profile == "production").
    """
    settings = get_settings()

    if settings.config_profile == "production":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reset not allowed in production",
        )

    events_repo = _get_events_repo()
    broker = get_paper_broker(events_repo)
    await broker.reset(workspace_id)

    logger.warning("paper_state_reset_via_api", workspace_id=str(workspace_id))

    return {"status": "reset", "workspace_id": str(workspace_id)}
