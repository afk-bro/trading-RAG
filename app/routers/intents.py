"""
Trade Intent endpoints for policy evaluation.

The intent evaluation flow:
1. Brain emits a TradeIntent (what it wants to do)
2. Policy Engine evaluates against rules (KillSwitch, DriftGuard, etc.)
3. Events are journaled (INTENT_EMITTED, POLICY_EVALUATED, APPROVED/REJECTED)
4. Response indicates if intent is approved for execution
"""

import time
import uuid as uuid_module
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas import (
    TradeIntent,
    TradeEvent,
    TradeEventType,
    PolicyDecision,
    CurrentState,
    IntentEvaluateRequest,
    IntentEvaluateResponse,
)
from app.services.policy_engine import PolicyEngine
from app.deps.security import require_admin_token


router = APIRouter(prefix="/intents", tags=["intents"])
logger = structlog.get_logger(__name__)

# Global state (set during app startup)
_db_pool = None
_policy_engine: Optional[PolicyEngine] = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def get_policy_engine() -> PolicyEngine:
    """Get or create the policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
    return _policy_engine


def _get_events_repo():
    """Get TradeEventsRepository instance."""
    from app.repositories.trade_events import TradeEventsRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return TradeEventsRepository(_db_pool)


async def _journal_events(
    intent: TradeIntent,
    decision: PolicyDecision,
    dry_run: bool = False,
) -> int:
    """
    Journal the intent evaluation events.

    Records:
    1. INTENT_EMITTED - The intent was created
    2. POLICY_EVALUATED - The policy engine evaluated it
    3. INTENT_APPROVED or INTENT_REJECTED - The outcome

    Returns count of events written.
    """
    if dry_run:
        return 0

    repo = _get_events_repo()
    events = []

    # Event 1: Intent emitted
    events.append(
        TradeEvent(
            correlation_id=intent.correlation_id,
            workspace_id=intent.workspace_id,
            event_type=TradeEventType.INTENT_EMITTED,
            strategy_entity_id=intent.strategy_entity_id,
            symbol=intent.symbol,
            timeframe=intent.timeframe,
            intent_id=intent.id,
            payload={
                "action": intent.action.value,
                "quantity": intent.quantity,
                "price": intent.price,
                "stop_loss": intent.stop_loss,
                "take_profit": intent.take_profit,
                "signal_strength": intent.signal_strength,
                "reason": intent.reason,
            },
        )
    )

    # Event 2: Policy evaluated
    events.append(
        TradeEvent(
            correlation_id=intent.correlation_id,
            workspace_id=intent.workspace_id,
            event_type=TradeEventType.POLICY_EVALUATED,
            strategy_entity_id=intent.strategy_entity_id,
            symbol=intent.symbol,
            timeframe=intent.timeframe,
            intent_id=intent.id,
            payload={
                "rules_evaluated": decision.rules_evaluated,
                "rules_passed": decision.rules_passed,
                "rules_failed": decision.rules_failed,
                "warnings": decision.warnings,
                "evaluation_ms": decision.evaluation_ms,
            },
        )
    )

    # Event 3: Outcome
    outcome_type = (
        TradeEventType.INTENT_APPROVED
        if decision.approved
        else TradeEventType.INTENT_REJECTED
    )
    events.append(
        TradeEvent(
            correlation_id=intent.correlation_id,
            workspace_id=intent.workspace_id,
            event_type=outcome_type,
            strategy_entity_id=intent.strategy_entity_id,
            symbol=intent.symbol,
            timeframe=intent.timeframe,
            intent_id=intent.id,
            payload={
                "approved": decision.approved,
                "reason": decision.reason.value,
                "reason_details": decision.reason_details,
                "modified_quantity": decision.modified_quantity,
            },
        )
    )

    return await repo.insert_many(events)


@router.post(
    "/evaluate",
    response_model=IntentEvaluateResponse,
    responses={
        200: {"description": "Intent evaluated successfully"},
        503: {"description": "Service unavailable"},
    },
)
async def evaluate_intent(
    request: IntentEvaluateRequest,
    _: None = Depends(require_admin_token),
) -> IntentEvaluateResponse:
    """
    Evaluate a trade intent against policy rules.

    This is the main entry point for the policy engine. It:
    1. Validates the intent
    2. Evaluates all policy rules (KillSwitch, DriftGuard, etc.)
    3. Journals the decision (unless dry_run=true)
    4. Returns the decision

    The caller should only proceed with execution if `decision.approved` is true.

    Use `dry_run=true` for testing without writing to the journal.
    """
    log = logger.bind(
        intent_id=str(request.intent.id),
        correlation_id=request.intent.correlation_id,
        action=request.intent.action.value,
        symbol=request.intent.symbol,
        dry_run=request.dry_run,
    )
    log.info("Evaluating intent")

    # Get current state (use provided or defaults)
    state = request.state or CurrentState()

    # Evaluate against policy rules
    engine = get_policy_engine()
    decision = engine.evaluate(request.intent, state)

    log.info(
        "Intent evaluation complete",
        approved=decision.approved,
        reason=decision.reason.value,
        rules_passed=decision.rules_passed,
        rules_failed=decision.rules_failed,
    )

    # Journal the events
    events_recorded = await _journal_events(
        request.intent,
        decision,
        dry_run=request.dry_run,
    )

    return IntentEvaluateResponse(
        intent_id=request.intent.id,
        decision=decision,
        events_recorded=events_recorded,
        correlation_id=request.intent.correlation_id,
    )


@router.get(
    "/rules",
    responses={
        200: {"description": "List of active policy rules"},
    },
)
async def list_rules(
    _: None = Depends(require_admin_token),
) -> dict:
    """
    List all active policy rules.

    Returns the rule name, priority, and enabled status for each rule.
    Useful for debugging and monitoring the policy configuration.
    """
    engine = get_policy_engine()
    rules = []
    for rule in engine.rules:
        rules.append({
            "name": rule.name,
            "priority": rule.priority,
            "enabled": rule.enabled,
        })

    return {
        "rules": rules,
        "count": len(rules),
    }
