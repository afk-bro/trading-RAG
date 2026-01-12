"""Alert transition layer - handles state changes and DB operations."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog

from app.services.alerts.models import EvalResult, RuleType, Severity

logger = structlog.get_logger(__name__)

SEVERITY_MAP = {
    RuleType.DRIFT_SPIKE: Severity.MEDIUM,
    RuleType.CONFIDENCE_DROP: Severity.MEDIUM,
    RuleType.COMBO: Severity.HIGH,
}


class AlertTransitionManager:
    """Manages alert state transitions."""

    def __init__(self, repo):
        """Initialize with alerts repository."""
        self.repo = repo

    async def process_evaluation(
        self,
        eval_result: EvalResult,
        workspace_id: UUID,
        rule_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
        rule_type: RuleType,
        fingerprint: str,
        cooldown_minutes: int,
    ) -> dict[str, Any]:
        """
        Process evaluation result and update DB state.

        Returns dict with action taken and details.
        """
        now = datetime.now(timezone.utc)

        # Insufficient data: no action
        if eval_result.insufficient_data:
            return {"action": "no_change", "reason": "insufficient_data"}

        # Get existing event
        existing = await self.repo.get_existing_event(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            regime_key=regime_key,
            timeframe=timeframe,
            rule_type=rule_type,
            fingerprint=fingerprint,
        )

        # Condition met: activate or update
        if eval_result.condition_met:
            if existing and existing["status"] == "active":
                # Still active: just update last_seen
                await self.repo.update_last_seen(existing["id"])
                return {"action": "updated_last_seen", "event_id": existing["id"]}

            # Potential activation (new or reactivation)
            if existing:
                activated_at = existing["activated_at"]
                if isinstance(activated_at, str):
                    activated_at = datetime.fromisoformat(activated_at)
                elapsed = (now - activated_at).total_seconds()
                if elapsed < cooldown_minutes * 60:
                    return {
                        "action": "suppressed_cooldown",
                        "reason": f"elapsed {elapsed:.0f}s < cooldown {cooldown_minutes * 60}s",
                    }

            # Activate
            severity = SEVERITY_MAP.get(rule_type, Severity.MEDIUM)
            context_json = {
                **eval_result.context,
                "deep_link": {
                    "strategy_entity_id": str(strategy_entity_id),
                    "timeframe": timeframe,
                    "regime_key": regime_key,
                },
            }

            result = await self.repo.upsert_activate(
                workspace_id=workspace_id,
                rule_id=rule_id,
                strategy_entity_id=strategy_entity_id,
                regime_key=regime_key,
                timeframe=timeframe,
                rule_type=rule_type,
                severity=severity,
                context_json=context_json,
                fingerprint=fingerprint,
            )

            return {"action": "activated", "event_id": result["id"]}

        # Condition clear: resolve if active
        elif eval_result.condition_clear:
            if existing and existing["status"] == "active":
                await self.repo.resolve(existing["id"])
                return {"action": "resolved", "event_id": existing["id"]}

        return {"action": "no_change", "reason": "unchanged"}
