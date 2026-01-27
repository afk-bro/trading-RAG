"""Unit tests for Discord notifier.

Tests the DiscordNotifier class:
- Embed formatting for different alert types
- Retry logic on transient errors
- Webhook routing by alert category
- SendResult dataclass

Run with: pytest tests/unit/test_ops_alert_discord_delivery.py -v
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

# Set required env vars before importing app modules
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-key")

from app.repositories.ops_alerts import OpsAlert  # noqa: E402
from app.services.ops_alerts.discord import (  # noqa: E402
    DiscordNotifier,
    SendResult,
    SEVERITY_COLORS,
    RECOVERY_COLOR,
    get_discord_notifier,
)


def make_alert(
    rule_type: str = "health_degraded",
    severity: str = "high",
    status: str = "active",
    payload: dict = None,
) -> OpsAlert:
    """Factory for test alerts."""
    return OpsAlert(
        id=uuid4(),
        workspace_id=uuid4(),
        rule_type=rule_type,
        severity=severity,
        status=status,
        rule_version="v1",
        dedupe_key=f"{rule_type}:2026-01-27",
        payload=payload or {"test": "value"},
        source="alert_evaluator",
        job_run_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        last_seen_at=datetime.now(timezone.utc),
        resolved_at=None,
        acknowledged_at=None,
        acknowledged_by=None,
        occurrence_count=1,
        notified_at=None,
        recovery_notified_at=None,
        escalated_at=None,
        escalation_notified_at=None,
    )


class TestSendResult:
    """Test SendResult dataclass."""

    def test_ok_with_message_id(self):
        """SendResult captures success with message_id."""
        result = SendResult(ok=True, message_id="12345")
        assert result.ok is True
        assert result.message_id == "12345"

    def test_not_ok(self):
        """SendResult captures failure."""
        result = SendResult(ok=False)
        assert result.ok is False
        assert result.message_id is None


class TestDiscordNotifierEmbed:
    """Test embed building logic."""

    @pytest.fixture
    def notifier(self):
        """Create notifier instance."""
        return DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test/test",
            base_url="https://admin.example.com",
        )

    def test_build_embed_alert(self, notifier):
        """Alert embed has correct structure."""
        alert = make_alert(severity="critical")
        embed = notifier._build_embed(alert, is_recovery=False, is_escalation=False)

        assert "ALERT" in embed["title"]
        assert embed["color"] == SEVERITY_COLORS["critical"]
        assert len(embed["fields"]) >= 3
        assert embed["footer"]["text"].startswith("Event:")

    def test_build_embed_recovery(self, notifier):
        """Recovery embed uses green color."""
        alert = make_alert()
        embed = notifier._build_embed(alert, is_recovery=True, is_escalation=False)

        assert "RECOVERED" in embed["title"]
        assert embed["color"] == RECOVERY_COLOR

    def test_build_embed_escalation(self, notifier):
        """Escalation embed shows ESCALATED."""
        alert = make_alert(severity="critical")
        embed = notifier._build_embed(alert, is_recovery=False, is_escalation=True)

        assert "ESCALATED" in embed["title"]
        assert embed["color"] == SEVERITY_COLORS["critical"]

    def test_build_embed_with_job_link(self, notifier):
        """Embed includes job run link when base_url configured."""
        alert = make_alert()
        embed = notifier._build_embed(alert, is_recovery=False, is_escalation=False)

        assert "url" in embed
        assert "/admin/jobs/runs/" in embed["url"]

    def test_format_rule_type(self, notifier):
        """Rule types are formatted nicely."""
        assert notifier._format_rule_type("health_degraded") == "Health Degraded"
        assert notifier._format_rule_type("drift_spike") == "Drift Spike"
        assert notifier._format_rule_type("unknown_rule") == "unknown_rule"


class TestDiscordNotifierPayloadFormatting:
    """Test payload formatting for different rule types."""

    @pytest.fixture
    def notifier(self):
        return DiscordNotifier(webhook_url="https://discord.com/api/webhooks/test/test")

    def test_format_health_degraded(self, notifier):
        """Health degraded payload formatted correctly."""
        payload = {
            "overall_status": "degraded",
            "issues": ["DB slow", "High latency", "Memory warning"],
        }
        text = notifier._format_payload("health_degraded", payload, is_recovery=False)

        assert "**Status:** degraded" in text
        assert "DB slow" in text

    def test_format_weak_coverage(self, notifier):
        """Weak coverage payload formatted correctly."""
        payload = {"count": 5, "worst_score": 0.123}
        text = notifier._format_payload("weak_coverage:P1", payload, is_recovery=False)

        assert "**Open gaps:** 5" in text
        assert "0.123" in text

    def test_format_drift_spike(self, notifier):
        """Drift spike payload formatted correctly."""
        payload = {
            "trigger_reason": "weak_rate_spike",
            "weak_rate_15m": 0.25,
            "weak_rate_24h": 0.10,
            "avg_score_15m": 0.65,
            "avg_score_24h": 0.85,
        }
        text = notifier._format_payload("drift_spike", payload, is_recovery=False)

        assert "**Trigger:** weak_rate_spike" in text
        assert "25.0%" in text or "25%" in text

    def test_format_recovery(self, notifier):
        """Recovery shows cleared message."""
        text = notifier._format_payload("health_degraded", {}, is_recovery=True)
        assert "Condition cleared" in text


class TestDiscordNotifierWebhookRouting:
    """Test webhook URL selection by rule type."""

    def test_default_webhook(self):
        """Uses default webhook when no category-specific one set."""
        notifier = DiscordNotifier(
            webhook_url="https://default.webhook",
        )
        assert (
            notifier._get_webhook_for_rule("health_degraded")
            == "https://default.webhook"
        )

    def test_health_webhook(self):
        """Uses health webhook for health alerts."""
        notifier = DiscordNotifier(
            webhook_url="https://default.webhook",
            webhook_health="https://health.webhook",
        )
        assert (
            notifier._get_webhook_for_rule("health_degraded")
            == "https://health.webhook"
        )

    def test_strategy_webhook(self):
        """Uses strategy webhook for strategy alerts."""
        notifier = DiscordNotifier(
            webhook_url="https://default.webhook",
            webhook_strategy="https://strategy.webhook",
        )
        assert (
            notifier._get_webhook_for_rule("drift_spike") == "https://strategy.webhook"
        )
        assert (
            notifier._get_webhook_for_rule("confidence_drop")
            == "https://strategy.webhook"
        )
        assert (
            notifier._get_webhook_for_rule("weak_coverage:P1")
            == "https://strategy.webhook"
        )


class TestDiscordNotifierSend:
    """Test actual send behavior."""

    @pytest.fixture
    def notifier(self):
        return DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test/test",
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_disabled_notifier_returns_false(self):
        """Disabled notifier returns ok=False without sending."""
        notifier = DiscordNotifier(
            webhook_url="https://test.webhook",
            enabled=False,
        )
        alert = make_alert()
        result = await notifier.send_alert(alert)

        assert result.ok is False

    @pytest.mark.asyncio
    async def test_successful_send(self, notifier):
        """Successful send returns ok=True with message_id."""
        alert = make_alert()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        # json() is a regular method that returns dict, not async
        mock_response.json = lambda: {"id": "123456789"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await notifier.send_alert(alert)

        assert result.ok is True
        assert result.message_id == "123456789"

    @pytest.mark.asyncio
    async def test_bad_request_no_retry(self, notifier):
        """400 errors don't retry."""
        alert = make_alert()

        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await notifier.send_alert(alert)

        assert result.ok is False
        # Should only call once (no retry on 400)
        assert mock_instance.post.call_count == 1

    @pytest.mark.asyncio
    async def test_server_error_retries(self, notifier):
        """5xx errors trigger retry."""
        alert = make_alert()

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await notifier.send_alert(alert)

        assert result.ok is False
        # Should retry once (2 total attempts)
        assert mock_instance.post.call_count == 2


class TestGetDiscordNotifier:
    """Test factory function."""

    def test_returns_none_without_webhook(self):
        """Returns None when no webhook configured."""
        with patch("app.config.get_settings") as mock_settings:
            settings = type("Settings", (), {"discord_webhook_url": None})()
            mock_settings.return_value = settings

            notifier = get_discord_notifier()
            assert notifier is None

    def test_returns_notifier_with_webhook(self):
        """Returns configured notifier when webhook set."""
        with patch("app.config.get_settings") as mock_settings:
            settings = type(
                "Settings",
                (),
                {
                    "discord_webhook_url": "https://test.webhook",
                    "discord_enabled": True,
                    "discord_timeout_secs": 15.0,
                    "admin_base_url": "https://admin.test",
                    "discord_username": "Test Bot",
                    "discord_avatar_url": None,
                    "discord_webhook_health": None,
                    "discord_webhook_strategy": None,
                },
            )()
            mock_settings.return_value = settings

            notifier = get_discord_notifier()
            assert notifier is not None
            assert notifier.webhook_url == "https://test.webhook"
            assert notifier.username == "Test Bot"
