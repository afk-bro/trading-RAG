"""Alert transition layer - handles state changes and DB operations."""

import json
from datetime import datetime, timezone
from typing import Any, Optional
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

    def __init__(
        self,
        repo,
        webhook_enabled: bool = False,
        slack_webhook_url: Optional[str] = None,
        alert_webhook_url: Optional[str] = None,
        alert_webhook_headers: Optional[str] = None,
    ):
        """
        Initialize with alerts repository and webhook configuration.

        Args:
            repo: AlertsRepository instance
            webhook_enabled: Enable webhook delivery
            slack_webhook_url: Optional Slack webhook URL
            alert_webhook_url: Optional generic webhook URL
            alert_webhook_headers: Optional JSON string of headers for generic webhook
        """
        self.repo = repo
        self.webhook_enabled = webhook_enabled
        self.slack_webhook_url = slack_webhook_url
        self.alert_webhook_url = alert_webhook_url

        # Parse webhook headers if provided
        self.webhook_headers = None
        if alert_webhook_headers:
            try:
                self.webhook_headers = json.loads(alert_webhook_headers)
            except json.JSONDecodeError:
                logger.warning(
                    "Invalid JSON in alert_webhook_headers, ignoring",
                    headers=alert_webhook_headers,
                )

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

            # Send webhooks for NEW activations (not updates to existing active alerts)
            is_new_activation = not existing or existing["status"] != "active"
            if is_new_activation and self.webhook_enabled:
                await self._send_webhooks(
                    result, workspace_id, rule_type, severity, context_json, timeframe
                )

            return {"action": "activated", "event_id": result["id"]}

        # Condition clear: resolve if active
        elif eval_result.condition_clear:
            if existing and existing["status"] == "active":
                await self.repo.resolve(existing["id"])
                return {"action": "resolved", "event_id": existing["id"]}

        return {"action": "no_change", "reason": "unchanged"}

    async def _send_webhooks(
        self,
        result: dict[str, Any],
        workspace_id: UUID,
        rule_type: RuleType,
        severity: Severity,
        context_json: dict[str, Any],
        timeframe: str,
    ) -> None:
        """
        Send webhooks for new alert activation (fire and forget).

        Args:
            result: Result from upsert_activate
            workspace_id: Workspace ID
            rule_type: Alert rule type
            severity: Alert severity
            context_json: Alert context
            timeframe: Timeframe
        """
        try:
            from app.services.ops_alerts.webhook_sink import send_alert_webhooks

            # Build alert event dict for webhook
            alert_event = {
                "id": result["id"],
                "workspace_id": workspace_id,
                "rule_type": rule_type.value,
                "severity": severity.value,
                "status": "active",
                "activated_at": result.get("activated_at"),
                "last_seen": result.get("last_seen"),
                "context_json": context_json,
                "timeframe": timeframe,
            }

            await send_alert_webhooks(
                alert_event=alert_event,
                slack_webhook_url=self.slack_webhook_url,
                generic_webhook_url=self.alert_webhook_url,
                generic_webhook_headers=self.webhook_headers,
            )

        except Exception as e:
            # Never let webhook failures block alert processing
            logger.exception(
                "Failed to send alert webhooks",
                alert_id=result.get("id"),
                error=str(e),
            )
