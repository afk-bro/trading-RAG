"""Webhook delivery sinks for ops alerts."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import structlog

from app.services.alerts.models import Severity

logger = structlog.get_logger(__name__)


class WebhookDeliveryError(Exception):
    """Raised when webhook delivery fails after retries."""

    pass


class SlackWebhookSink:
    """Delivers alerts to Slack via webhook with retry logic."""

    SEVERITY_COLORS = {
        Severity.HIGH: "danger",  # Red
        Severity.MEDIUM: "warning",  # Yellow
        Severity.LOW: "good",  # Green
    }

    def __init__(
        self,
        webhook_url: str,
        max_retries: int = 3,
        timeout: float = 10.0,
        admin_ui_base_url: Optional[str] = None,
    ):
        """
        Initialize Slack webhook sink.

        Args:
            webhook_url: Slack incoming webhook URL
            max_retries: Maximum retry attempts (default 3)
            timeout: Request timeout in seconds (default 10)
            admin_ui_base_url: Base URL for admin UI links (optional)
        """
        self.webhook_url = webhook_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.admin_ui_base_url = admin_ui_base_url

    def _format_message(self, alert_event: dict[str, Any]) -> dict[str, Any]:
        """
        Format alert event as Slack message blocks.

        Args:
            alert_event: Alert event dictionary from database

        Returns:
            Slack webhook payload
        """
        severity = Severity(alert_event["severity"])
        color = self.SEVERITY_COLORS.get(severity, "warning")

        # Build context message from context_json
        context_json = alert_event.get("context_json", {})
        details_lines = []
        for key, value in context_json.items():
            if isinstance(value, (list, dict)):
                continue  # Skip complex nested structures
            details_lines.append(f"• {key}: {value}")
        details = "\n".join(details_lines) if details_lines else "No additional details"

        # Format timestamp
        activated_at = alert_event.get("activated_at")
        if activated_at:
            if isinstance(activated_at, str):
                activated_str = activated_at
            else:
                activated_str = activated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            activated_str = "Unknown"

        # Build fields
        fields = [
            {
                "title": "Rule Type",
                "value": alert_event["rule_type"],
                "short": True,
            },
            {
                "title": "Severity",
                "value": alert_event["severity"],
                "short": True,
            },
            {
                "title": "Workspace",
                "value": str(alert_event["workspace_id"]),
                "short": True,
            },
            {
                "title": "Timeframe",
                "value": alert_event.get("timeframe", "N/A"),
                "short": True,
            },
            {
                "title": "Activated",
                "value": activated_str,
                "short": False,
            },
            {
                "title": "Details",
                "value": details,
                "short": False,
            },
        ]

        # Add admin UI link if configured
        if self.admin_ui_base_url:
            alert_id = alert_event.get("id")
            if alert_id:
                admin_url = (
                    f"{self.admin_ui_base_url}/admin/ops-alerts?alert_id={alert_id}"
                )
                fields.append(
                    {
                        "title": "Admin Link",
                        "value": f"<{admin_url}|View in Admin UI>",
                        "short": False,
                    }
                )

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"Alert: {alert_event['rule_type']}",
                    "fields": fields,
                    "footer": "Trading RAG Ops Alerts",
                    "ts": int(datetime.now(timezone.utc).timestamp()),
                }
            ]
        }

        return payload

    async def send(self, alert_event: dict[str, Any]) -> None:
        """
        Send alert to Slack with retry logic.

        Args:
            alert_event: Alert event dictionary

        Raises:
            WebhookDeliveryError: If delivery fails after retries
        """
        payload = self._format_message(alert_event)

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.webhook_url, json=payload)
                    response.raise_for_status()

                logger.info(
                    "Slack webhook delivered",
                    alert_id=alert_event.get("id"),
                    attempt=attempt + 1,
                )
                return

            except httpx.TimeoutException as e:
                logger.warning(
                    "Slack webhook timeout",
                    alert_id=alert_event.get("id"),
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt + 1 >= self.max_retries:
                    raise WebhookDeliveryError(
                        f"Slack webhook timeout after {self.max_retries} attempts"
                    ) from e

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "Slack webhook HTTP error",
                    alert_id=alert_event.get("id"),
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    status_code=e.response.status_code,
                    error=str(e),
                )
                if attempt + 1 >= self.max_retries:
                    raise WebhookDeliveryError(
                        f"Slack webhook failed with status {e.response.status_code} "
                        f"after {self.max_retries} attempts"
                    ) from e

            except Exception as e:
                logger.error(
                    "Slack webhook unexpected error",
                    alert_id=alert_event.get("id"),
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt + 1 >= self.max_retries:
                    raise WebhookDeliveryError(
                        f"Slack webhook failed after {self.max_retries} attempts: {e}"
                    ) from e

            # Exponential backoff: 1s, 2s, 4s
            if attempt + 1 < self.max_retries:
                backoff = 2**attempt
                await asyncio.sleep(backoff)


class GenericWebhookSink:
    """Delivers alerts to arbitrary webhook endpoints."""

    def __init__(
        self,
        webhook_url: str,
        max_retries: int = 3,
        timeout: float = 10.0,
        headers: Optional[dict[str, str]] = None,
    ):
        """
        Initialize generic webhook sink.

        Args:
            webhook_url: Webhook URL
            max_retries: Maximum retry attempts (default 3)
            timeout: Request timeout in seconds (default 10)
            headers: Optional custom headers (e.g., API keys)
        """
        self.webhook_url = webhook_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = headers or {}

    def _format_payload(self, alert_event: dict[str, Any]) -> dict[str, Any]:
        """
        Format alert event as generic JSON payload.

        Args:
            alert_event: Alert event dictionary

        Returns:
            JSON payload for webhook
        """
        # Convert datetime to ISO string if present
        activated_at = alert_event.get("activated_at")
        if activated_at and not isinstance(activated_at, str):
            activated_at_str = activated_at.isoformat()
        else:
            activated_at_str = activated_at

        return {
            "event_type": "alert.activated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workspace_id": str(alert_event["workspace_id"]),
            "alert": {
                "id": str(alert_event.get("id", "")),
                "rule_type": alert_event["rule_type"],
                "severity": alert_event["severity"],
                "status": alert_event.get("status", "active"),
                "activated_at": activated_at_str,
                "strategy_entity_id": str(alert_event.get("strategy_entity_id", "")),
                "regime_key": alert_event.get("regime_key"),
                "timeframe": alert_event.get("timeframe"),
                "context": alert_event.get("context_json", {}),
                "fingerprint": alert_event.get("fingerprint", ""),
            },
        }

    async def send(self, alert_event: dict[str, Any]) -> None:
        """
        Send alert to webhook with retry logic.

        Args:
            alert_event: Alert event dictionary

        Raises:
            WebhookDeliveryError: If delivery fails after retries
        """
        payload = self._format_payload(alert_event)
        headers = {
            "Content-Type": "application/json",
            **self.headers,
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.webhook_url, json=payload, headers=headers
                    )
                    response.raise_for_status()

                logger.info(
                    "Generic webhook delivered",
                    alert_id=alert_event.get("id"),
                    webhook_url=self.webhook_url,
                    attempt=attempt + 1,
                )
                return

            except httpx.TimeoutException as e:
                logger.warning(
                    "Generic webhook timeout",
                    alert_id=alert_event.get("id"),
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt + 1 >= self.max_retries:
                    raise WebhookDeliveryError(
                        f"Generic webhook timeout after {self.max_retries} attempts"
                    ) from e

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "Generic webhook HTTP error",
                    alert_id=alert_event.get("id"),
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    status_code=e.response.status_code,
                    error=str(e),
                )
                if attempt + 1 >= self.max_retries:
                    raise WebhookDeliveryError(
                        f"Generic webhook failed with status {e.response.status_code} "
                        f"after {self.max_retries} attempts"
                    ) from e

            except Exception as e:
                logger.error(
                    "Generic webhook unexpected error",
                    alert_id=alert_event.get("id"),
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt + 1 >= self.max_retries:
                    raise WebhookDeliveryError(
                        f"Generic webhook failed after {self.max_retries} attempts: {e}"
                    ) from e

            # Exponential backoff: 1s, 2s, 4s
            if attempt + 1 < self.max_retries:
                backoff = 2**attempt
                await asyncio.sleep(backoff)


async def send_alert_webhooks(
    alert_event: dict[str, Any],
    slack_webhook_url: Optional[str] = None,
    generic_webhook_url: Optional[str] = None,
    generic_webhook_headers: Optional[dict[str, str]] = None,
) -> None:
    """
    Fire-and-forget webhook delivery for alert events.

    This function logs errors but doesn't raise exceptions, allowing
    the caller to continue without blocking on webhook delivery.

    Args:
        alert_event: Alert event dictionary
        slack_webhook_url: Optional Slack webhook URL
        generic_webhook_url: Optional generic webhook URL
        generic_webhook_headers: Optional headers for generic webhook
    """
    tasks = []

    if slack_webhook_url:
        sink = SlackWebhookSink(slack_webhook_url)
        tasks.append(_safe_send(sink, alert_event, "Slack"))

    if generic_webhook_url:
        sink = GenericWebhookSink(generic_webhook_url, headers=generic_webhook_headers)
        tasks.append(_safe_send(sink, alert_event, "Generic"))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _safe_send(sink: Any, alert_event: dict[str, Any], sink_name: str) -> None:
    """
    Wrapper that catches and logs webhook errors without propagating.

    Args:
        sink: Webhook sink instance
        alert_event: Alert event to send
        sink_name: Name for logging
    """
    try:
        await sink.send(alert_event)
    except WebhookDeliveryError as e:
        logger.error(
            f"{sink_name} webhook delivery failed",
            alert_id=alert_event.get("id"),
            error=str(e),
        )
    except Exception as e:
        logger.exception(
            f"{sink_name} webhook unexpected failure",
            alert_id=alert_event.get("id"),
            error=str(e),
        )
