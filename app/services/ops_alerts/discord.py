"""Discord notifier for operational alerts."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from app.repositories.ops_alerts import OpsAlert

logger = structlog.get_logger(__name__)


@dataclass
class SendResult:
    """Result from sending a Discord notification."""

    ok: bool
    message_id: Optional[str] = None


# Severity to Discord embed color mapping (decimal RGB)
SEVERITY_COLORS = {
    "critical": 0xED4245,  # Red
    "high": 0xFFA500,  # Orange
    "medium": 0xFEE75C,  # Yellow
    "low": 0x5865F2,  # Blurple
}

RECOVERY_COLOR = 0x57F287  # Green


class DiscordNotifier:
    """
    Sends operational alerts to Discord via webhooks.

    Features:
    - Rich embeds with severity colors
    - Retry on transient errors (5xx, timeouts)
    - Deep links to admin UI when base_url configured
    - Channel routing by alert category
    """

    MAX_EMBED_DESCRIPTION = 4000  # Discord limit is 4096

    # Rule type to channel category mapping
    RULE_CHANNEL_MAP = {
        "health_degraded": "health",
        "weak_coverage:P1": "strategy",
        "weak_coverage:P2": "strategy",
        "drift_spike": "strategy",
        "confidence_drop": "strategy",
    }

    def __init__(
        self,
        webhook_url: str,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        enabled: bool = True,
        username: str = "Trading RAG Alerts",
        avatar_url: Optional[str] = None,
        webhook_health: Optional[str] = None,
        webhook_strategy: Optional[str] = None,
    ):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Default Discord webhook URL
            base_url: Admin UI base URL for deep links (optional)
            timeout: Request timeout in seconds
            enabled: Master kill switch
            username: Bot display name
            avatar_url: Bot avatar URL (optional)
            webhook_health: Webhook URL for health alerts (optional)
            webhook_strategy: Webhook URL for strategy alerts (optional)
        """
        self.webhook_url = webhook_url
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.enabled = enabled
        self.username = username
        self.avatar_url = avatar_url
        self.webhook_map = {
            "health": webhook_health,
            "strategy": webhook_strategy,
        }

    async def send_alert(
        self,
        alert: OpsAlert,
        is_recovery: bool = False,
        is_escalation: bool = False,
    ) -> SendResult:
        """
        Send alert notification to Discord.

        Args:
            alert: The OpsAlert to notify about
            is_recovery: True if this is a recovery notification
            is_escalation: True if severity escalated

        Returns:
            SendResult with ok=True and message_id if successful
        """
        if not self.enabled:
            logger.debug("discord_disabled", alert_id=str(alert.id))
            return SendResult(ok=False)

        embed = self._build_embed(alert, is_recovery, is_escalation)
        webhook = self._get_webhook_for_rule(alert.rule_type)

        return await self._send(webhook, embed, alert.id)

    def _get_webhook_for_rule(self, rule_type: str) -> str:
        """Get the webhook URL for a given rule type."""
        category = self.RULE_CHANNEL_MAP.get(rule_type)
        if category:
            webhook = self.webhook_map.get(category)
            if webhook:
                return webhook
        return self.webhook_url

    async def send_batch(
        self,
        alerts: list[tuple[OpsAlert, bool, bool]],
    ) -> int:
        """
        Send multiple alert notifications.

        Returns count of successfully sent messages.
        """
        if not self.enabled or not alerts:
            return 0

        sent = 0
        for alert, is_recovery, is_escalation in alerts:
            result = await self.send_alert(alert, is_recovery, is_escalation)
            if result.ok:
                sent += 1
            # Brief delay between messages to avoid rate limiting
            if len(alerts) > 1:
                await asyncio.sleep(0.5)

        return sent

    def _build_embed(
        self,
        alert: OpsAlert,
        is_recovery: bool,
        is_escalation: bool,
    ) -> dict:
        """Build Discord embed for alert."""
        # Determine action and color
        if is_recovery:
            action = "RECOVERED"
            color = RECOVERY_COLOR
        elif is_escalation:
            action = "ESCALATED"
            color = SEVERITY_COLORS.get(alert.severity, 0x99AAB5)
        else:
            action = "ALERT"
            color = SEVERITY_COLORS.get(alert.severity, 0x99AAB5)

        # Title
        rule_display = self._format_rule_type(alert.rule_type)
        title = f"{action}: {rule_display}"

        # Fields
        fields = [
            {"name": "Severity", "value": f"`{alert.severity}`", "inline": True},
            {"name": "Rule", "value": f"`{alert.rule_type}`", "inline": True},
        ]

        # Extract date from dedupe_key
        parts = alert.dedupe_key.split(":")
        if len(parts) >= 2:
            date_str = parts[-1]
            fields.append({"name": "Date", "value": date_str, "inline": True})

        # Payload details
        payload_text = self._format_payload(alert.rule_type, alert.payload, is_recovery)
        if payload_text:
            fields.append({"name": "Details", "value": payload_text, "inline": False})

        # Build embed
        embed = {
            "title": title,
            "color": color,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"Event: {str(alert.id)[:8]}"},
        }

        # Add link to job run if available
        if self.base_url and alert.job_run_id:
            url = f"{self.base_url}/admin/jobs/runs/{alert.job_run_id}/detail"
            embed["url"] = url

        return embed

    def _format_rule_type(self, rule_type: str) -> str:
        """Format rule_type for display."""
        mapping = {
            "health_degraded": "Health Degraded",
            "weak_coverage:P1": "Weak Coverage (P1)",
            "weak_coverage:P2": "Weak Coverage (P2)",
            "drift_spike": "Drift Spike",
            "confidence_drop": "Confidence Drop",
        }
        return mapping.get(rule_type, rule_type)

    def _format_payload(self, rule_type: str, payload: dict, is_recovery: bool) -> str:
        """Format payload details for a specific rule type."""
        if is_recovery:
            return "*Condition cleared.*"

        lines: list[str] = []

        if rule_type == "health_degraded":
            status = payload.get("overall_status", "unknown")
            lines.append(f"**Status:** {status}")
            issues = payload.get("issues", [])
            if issues:
                lines.append("**Issues:**")
                for issue in issues[:5]:
                    lines.append(f"• {str(issue)[:80]}")
                if len(issues) > 5:
                    lines.append(f"*(+{len(issues) - 5} more…)*")

        elif rule_type.startswith("weak_coverage"):
            count = payload.get("count", 0)
            lines.append(f"**Open gaps:** {count}")
            worst = payload.get("worst_score")
            if worst is not None:
                lines.append(f"**Worst score:** {worst:.3f}")

        elif rule_type == "drift_spike":
            reason = payload.get("trigger_reason", "unknown")
            lines.append(f"**Trigger:** {reason}")
            weak_15m = payload.get("weak_rate_15m", 0)
            weak_24h = payload.get("weak_rate_24h", 0)
            lines.append(f"**Weak rate:** {weak_15m:.1%} (15m) vs {weak_24h:.1%} (24h)")
            score_15m = payload.get("avg_score_15m")
            score_24h = payload.get("avg_score_24h")
            if score_15m is not None and score_24h is not None:
                lines.append(
                    f"**Avg score:** {score_15m:.3f} (15m) vs {score_24h:.3f} (24h)"
                )

        elif rule_type == "confidence_drop":
            reason = payload.get("trigger_reason", "unknown")
            lines.append(f"**Trigger:** {reason}")
            score_15m = payload.get("avg_score_15m")
            score_24h = payload.get("avg_score_24h")
            if score_15m is not None:
                lines.append(f"**Avg score (15m):** {score_15m:.3f}")
            if score_24h is not None:
                lines.append(f"**Avg score (24h):** {score_24h:.3f}")

        text = "\n".join(lines)

        # Truncate if too long
        if len(text) > 1000:
            text = text[:950] + "\n*(truncated…)*"

        return text

    async def _send(self, webhook_url: str, embed: dict, alert_id) -> SendResult:
        """Send embed to Discord with retry."""
        payload = {
            "username": self.username,
            "embeds": [embed],
        }
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        for attempt in range(2):  # Retry once
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Use ?wait=true to get message ID in response
                    resp = await client.post(f"{webhook_url}?wait=true", json=payload)

                    if resp.status_code in (200, 204):
                        message_id = None
                        try:
                            data = resp.json()
                            message_id = data.get("id")
                        except Exception:
                            pass

                        logger.info(
                            "discord_sent",
                            alert_id=str(alert_id),
                            message_id=message_id,
                        )
                        return SendResult(ok=True, message_id=message_id)

                    if resp.status_code == 400:
                        # Bad request - don't retry
                        logger.warning(
                            "discord_bad_request",
                            alert_id=str(alert_id),
                            response=resp.text[:200],
                        )
                        return SendResult(ok=False)

                    # 429 or 5xx - retry
                    logger.warning(
                        "discord_error",
                        alert_id=str(alert_id),
                        status=resp.status_code,
                        attempt=attempt,
                    )

            except Exception as e:
                logger.warning(
                    "discord_exception",
                    alert_id=str(alert_id),
                    error=str(e),
                    attempt=attempt,
                )

            if attempt == 0:
                await asyncio.sleep(1)  # Brief backoff before retry

        logger.error("discord_failed", alert_id=str(alert_id))
        return SendResult(ok=False)


def get_discord_notifier() -> Optional[DiscordNotifier]:
    """
    Get configured Discord notifier from settings.

    Returns None if Discord is not configured.
    """
    from app.config import get_settings

    settings = get_settings()

    webhook_url = getattr(settings, "discord_webhook_url", None)

    if not webhook_url:
        return None

    enabled = getattr(settings, "discord_enabled", True)
    timeout = getattr(settings, "discord_timeout_secs", 10.0)
    base_url = getattr(settings, "admin_base_url", None)
    username = getattr(settings, "discord_username", "Trading RAG Alerts")
    avatar_url = getattr(settings, "discord_avatar_url", None)
    webhook_health = getattr(settings, "discord_webhook_health", None)
    webhook_strategy = getattr(settings, "discord_webhook_strategy", None)

    return DiscordNotifier(
        webhook_url=webhook_url,
        base_url=base_url,
        timeout=timeout,
        enabled=enabled,
        username=username,
        avatar_url=avatar_url,
        webhook_health=webhook_health,
        webhook_strategy=webhook_strategy,
    )
