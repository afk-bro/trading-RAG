"""Telegram notifier for operational alerts."""

import asyncio
from typing import Optional

import httpx
import structlog

from app.repositories.ops_alerts import OpsAlert

logger = structlog.get_logger(__name__)


# Severity emoji mapping
SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
}

RECOVERY_EMOJI = "🟢"


class TelegramNotifier:
    """
    Sends operational alerts to Telegram.

    Features:
    - HTML formatting with severity emojis
    - Retry on transient errors (5xx, timeouts)
    - Message truncation for Telegram limit
    - Deep links to admin UI when base_url configured
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
    MAX_MESSAGE_LENGTH = 4000  # Telegram limit is ~4096

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        base_url: Optional[str] = None,
        timeout: float = 10.0,
        enabled: bool = True,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token
            chat_id: Target chat/group ID
            base_url: Admin UI base URL for deep links (optional)
            timeout: Request timeout in seconds
            enabled: Master kill switch
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self.enabled = enabled

    async def send_alert(
        self,
        alert: OpsAlert,
        is_recovery: bool = False,
        is_escalation: bool = False,
    ) -> bool:
        """
        Send alert notification to Telegram.

        Args:
            alert: The OpsAlert to notify about
            is_recovery: True if this is a recovery notification
            is_escalation: True if severity escalated

        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("telegram_disabled", alert_id=str(alert.id))
            return False

        message = self._format_message(alert, is_recovery, is_escalation)

        # Truncate if too long
        if len(message) > self.MAX_MESSAGE_LENGTH:
            message = (
                message[: self.MAX_MESSAGE_LENGTH - 50] + "\n\n<i>(truncated...)</i>"
            )

        return await self._send(message, alert.id)

    async def send_batch(
        self,
        alerts: list[
            tuple[OpsAlert, bool, bool]
        ],  # (alert, is_recovery, is_escalation)
    ) -> int:
        """
        Send multiple alert notifications.

        Returns count of successfully sent messages.
        """
        if not self.enabled or not alerts:
            return 0

        sent = 0
        for alert, is_recovery, is_escalation in alerts:
            if await self.send_alert(alert, is_recovery, is_escalation):
                sent += 1
            # Brief delay between messages to avoid rate limiting
            if len(alerts) > 1:
                await asyncio.sleep(0.5)

        return sent

    def _format_message(
        self,
        alert: OpsAlert,
        is_recovery: bool,
        is_escalation: bool,
    ) -> str:
        """Format alert as HTML message for Telegram."""
        lines: list[str] = []

        # Header with emoji
        if is_recovery:
            emoji = RECOVERY_EMOJI
            action = "RECOVERED"
        elif is_escalation:
            emoji = SEVERITY_EMOJI.get(alert.severity, "⚪")
            action = "ESCALATED"
        else:
            emoji = SEVERITY_EMOJI.get(alert.severity, "⚪")
            action = "ALERT"

        # Title - format rule_type nicely
        rule_display = self._format_rule_type(alert.rule_type)
        lines.append(f"{emoji} <b>{action}: {rule_display}</b>")
        lines.append("━" * 20)

        # Metadata
        lines.append(f"Severity: <code>{alert.severity}</code>")
        lines.append(f"Rule: <code>{alert.rule_type}</code>")

        # Extract date from dedupe_key (format: rule_type:date or rule_type:extra:date)
        parts = alert.dedupe_key.split(":")
        if len(parts) >= 2:
            date_str = parts[-1]
            lines.append(f"Date: {date_str}")

        # Payload details (rule-specific)
        payload_lines = self._format_payload(
            alert.rule_type, alert.payload, is_recovery
        )
        if payload_lines:
            lines.append("")
            lines.extend(payload_lines)

        # Links
        lines.append("")
        if self.base_url and alert.job_run_id:
            url = f"{self.base_url}/admin/jobs/runs/{alert.job_run_id}/detail"
            lines.append(f'<a href="{url}">View Job Run</a>')
        lines.append(f"Event: <code>{str(alert.id)[:8]}</code>")

        return "\n".join(lines)

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

    def _format_payload(
        self, rule_type: str, payload: dict, is_recovery: bool
    ) -> list[str]:
        """Format payload details for a specific rule type."""
        lines: list[str] = []

        if is_recovery:
            lines.append("<i>Condition cleared.</i>")
            return lines

        if rule_type == "health_degraded":
            status = payload.get("overall_status", "unknown")
            lines.append(f"<b>Status:</b> {status}")
            issues = payload.get("issues", [])
            if issues:
                lines.append("<b>Issues:</b>")
                for issue in issues[:5]:
                    lines.append(f"• {self._escape_html(str(issue)[:80])}")
                if len(issues) > 5:
                    lines.append(f"<i>(+{len(issues) - 5} more…)</i>")

        elif rule_type.startswith("weak_coverage"):
            count = payload.get("count", 0)
            lines.append(f"<b>Open gaps:</b> {count}")
            worst = payload.get("worst_score")
            if worst is not None:
                lines.append(f"<b>Worst score:</b> {worst:.3f}")

        elif rule_type == "drift_spike":
            reason = payload.get("trigger_reason", "unknown")
            lines.append(f"<b>Trigger:</b> {reason}")
            weak_15m = payload.get("weak_rate_15m", 0)
            weak_24h = payload.get("weak_rate_24h", 0)
            lines.append(
                f"<b>Weak rate:</b> {weak_15m:.1%} (15m) vs {weak_24h:.1%} (24h)"
            )
            score_15m = payload.get("avg_score_15m")
            score_24h = payload.get("avg_score_24h")
            if score_15m is not None and score_24h is not None:
                lines.append(
                    f"<b>Avg score:</b> {score_15m:.3f} (15m) vs {score_24h:.3f} (24h)"
                )

        elif rule_type == "confidence_drop":
            reason = payload.get("trigger_reason", "unknown")
            lines.append(f"<b>Trigger:</b> {reason}")
            score_15m = payload.get("avg_score_15m")
            score_24h = payload.get("avg_score_24h")
            if score_15m is not None:
                lines.append(f"<b>Avg score (15m):</b> {score_15m:.3f}")
            if score_24h is not None:
                lines.append(f"<b>Avg score (24h):</b> {score_24h:.3f}")

        return lines

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    async def _send(self, message: str, alert_id) -> bool:
        """Send message to Telegram with retry."""
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        url = self.TELEGRAM_API.format(token=self.bot_token)

        for attempt in range(2):  # Retry once
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)

                    if resp.status_code == 200:
                        logger.info(
                            "telegram_sent",
                            alert_id=str(alert_id),
                        )
                        return True

                    if resp.status_code == 400:
                        # Bad request - don't retry
                        logger.warning(
                            "telegram_bad_request",
                            alert_id=str(alert_id),
                            response=resp.text[:200],
                        )
                        return False

                    # 429 or 5xx - retry
                    logger.warning(
                        "telegram_error",
                        alert_id=str(alert_id),
                        status=resp.status_code,
                        attempt=attempt,
                    )

            except Exception as e:
                logger.warning(
                    "telegram_exception",
                    alert_id=str(alert_id),
                    error=str(e),
                    attempt=attempt,
                )

            if attempt == 0:
                await asyncio.sleep(1)  # Brief backoff before retry

        logger.error("telegram_failed", alert_id=str(alert_id))
        return False


def get_telegram_notifier() -> Optional[TelegramNotifier]:
    """
    Get configured Telegram notifier from settings.

    Returns None if Telegram is not configured.
    """
    from app.config import get_settings

    settings = get_settings()

    bot_token = getattr(settings, "telegram_bot_token", None)
    chat_id = getattr(settings, "telegram_chat_id", None)

    if not bot_token or not chat_id:
        return None

    enabled = getattr(settings, "telegram_enabled", True)
    timeout = getattr(settings, "telegram_timeout_secs", 10.0)
    base_url = getattr(settings, "admin_base_url", None)

    return TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        base_url=base_url,
        timeout=timeout,
        enabled=enabled,
    )
