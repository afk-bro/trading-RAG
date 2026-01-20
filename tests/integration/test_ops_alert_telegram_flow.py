"""Integration tests for ops alert Telegram delivery flow.

Tests verify end-to-end notification delivery scenarios:
1. New alert → activate → send → no resend on rerun
2. Resolved alert → recovery notification
3. Severity escalation → escalation notification

Run with: pytest tests/integration/test_ops_alert_telegram_flow.py -v
"""

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Mock ccxt before importing handlers
sys.modules["ccxt"] = MagicMock()
sys.modules["ccxt.async_support"] = MagicMock()

from app.jobs.handlers.ops_alert_eval import _send_notifications  # noqa: E402
from app.repositories.ops_alerts import OpsAlert, OpsAlertsRepository  # noqa: E402
from app.services.ops_alerts.telegram import (  # noqa: E402
    SendResult,
    TelegramNotifier,
)

pytestmark = [pytest.mark.integration]


def make_alert(
    rule_type: str = "health_degraded",
    severity: str = "high",
    status: str = "active",
    notified_at=None,
    recovery_notified_at=None,
    escalated_at=None,
    escalation_notified_at=None,
) -> OpsAlert:
    """Factory for test alerts."""
    return OpsAlert(
        id=uuid4(),
        workspace_id=uuid4(),
        rule_type=rule_type,
        severity=severity,
        status=status,
        rule_version="v1",
        dedupe_key=f"{rule_type}:2026-01-19",
        payload={"test": "value"},
        source="alert_evaluator",
        job_run_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        last_seen_at=datetime.now(timezone.utc),
        resolved_at=datetime.now(timezone.utc) if status == "resolved" else None,
        acknowledged_at=None,
        acknowledged_by=None,
        occurrence_count=1,
        notified_at=notified_at,
        recovery_notified_at=recovery_notified_at,
        escalated_at=escalated_at,
        escalation_notified_at=escalation_notified_at,
    )


class TestActivationNotificationFlow:
    """Test activation notification flow (new alerts)."""

    @pytest.fixture
    def mock_notifier(self):
        """Create mock notifier that succeeds."""
        notifier = AsyncMock(spec=TelegramNotifier)
        notifier.send_alert.return_value = SendResult(ok=True, message_id="tg_msg_123")
        return notifier

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = AsyncMock(spec=OpsAlertsRepository)
        repo.mark_notified.return_value = True
        return repo

    @pytest.mark.asyncio
    async def test_new_alert_sends_activation_notification(
        self, mock_notifier, mock_repo
    ):
        """New active alert triggers activation notification."""
        alert = make_alert(status="active", notified_at=None)

        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert],
            "recoveries": [],
            "escalations": [],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        # Should send exactly one activation notification
        assert sent == 1
        mock_notifier.send_alert.assert_called_once_with(
            alert, is_recovery=False, is_escalation=False
        )
        mock_repo.mark_notified.assert_called_once_with(
            alert.id, "activation", "tg_msg_123"
        )

    @pytest.mark.asyncio
    async def test_already_notified_alert_not_resent(self, mock_notifier, mock_repo):
        """Alert with notified_at set is not resent on rerun."""
        # Repository returns no pending activations (because notified_at IS NOT NULL)
        mock_repo.get_pending_notifications.return_value = {
            "activations": [],  # Already notified, not returned
            "recoveries": [],
            "escalations": [],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        # Nothing sent
        assert sent == 0
        mock_notifier.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_workers_only_one_wins(self, mock_notifier, mock_repo):
        """When two workers try to notify same alert, only one succeeds."""
        alert1 = make_alert()
        alert2 = make_alert()

        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert1, alert2],
            "recoveries": [],
            "escalations": [],
        }

        # First mark succeeds, second loses the race
        mock_repo.mark_notified.side_effect = [True, False]

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        # Both alerts were attempted to send
        assert mock_notifier.send_alert.call_count == 2
        # But only one was counted as sent (won the mark race)
        assert sent == 1


class TestRecoveryNotificationFlow:
    """Test recovery notification flow (resolved alerts)."""

    @pytest.fixture
    def mock_notifier(self):
        """Create mock notifier."""
        notifier = AsyncMock(spec=TelegramNotifier)
        notifier.send_alert.return_value = SendResult(ok=True, message_id="tg_recovery")
        return notifier

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = AsyncMock(spec=OpsAlertsRepository)
        repo.mark_notified.return_value = True
        return repo

    @pytest.mark.asyncio
    async def test_resolved_alert_sends_recovery_notification(
        self, mock_notifier, mock_repo
    ):
        """Resolved alert triggers recovery notification."""
        alert = make_alert(
            status="resolved",
            notified_at=datetime.now(timezone.utc),  # Was previously activated
            recovery_notified_at=None,  # Not yet notified about recovery
        )

        mock_repo.get_pending_notifications.return_value = {
            "activations": [],
            "recoveries": [alert],
            "escalations": [],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        assert sent == 1
        mock_notifier.send_alert.assert_called_once_with(
            alert, is_recovery=True, is_escalation=False
        )
        mock_repo.mark_notified.assert_called_once_with(
            alert.id, "recovery", "tg_recovery"
        )

    @pytest.mark.asyncio
    async def test_already_recovery_notified_not_resent(self, mock_notifier, mock_repo):
        """Alert with recovery_notified_at set is not resent."""
        mock_repo.get_pending_notifications.return_value = {
            "activations": [],
            "recoveries": [],  # Already notified, not returned
            "escalations": [],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        assert sent == 0
        mock_notifier.send_alert.assert_not_called()


class TestEscalationNotificationFlow:
    """Test escalation notification flow (severity bump)."""

    @pytest.fixture
    def mock_notifier(self):
        """Create mock notifier."""
        notifier = AsyncMock(spec=TelegramNotifier)
        notifier.send_alert.return_value = SendResult(
            ok=True, message_id="tg_escalation"
        )
        return notifier

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = AsyncMock(spec=OpsAlertsRepository)
        repo.mark_notified.return_value = True
        return repo

    @pytest.mark.asyncio
    async def test_escalated_alert_sends_escalation_notification(
        self, mock_notifier, mock_repo
    ):
        """Escalated alert triggers escalation notification."""
        alert = make_alert(
            status="active",
            severity="critical",  # Escalated from medium
            notified_at=datetime.now(timezone.utc),  # Already activated
            escalated_at=datetime.now(timezone.utc),  # Just escalated
            escalation_notified_at=None,  # Not yet notified about escalation
        )

        mock_repo.get_pending_notifications.return_value = {
            "activations": [],
            "recoveries": [],
            "escalations": [alert],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        assert sent == 1
        mock_notifier.send_alert.assert_called_once_with(
            alert, is_recovery=False, is_escalation=True
        )
        mock_repo.mark_notified.assert_called_once_with(
            alert.id, "escalation", "tg_escalation"
        )

    @pytest.mark.asyncio
    async def test_escalation_requires_prior_activation(self, mock_notifier, mock_repo):
        """Escalation notification only sent for already-activated alerts."""
        # Alert was escalated but not yet activated (shouldn't happen normally)
        alert = make_alert(
            status="active",
            notified_at=None,  # Not yet activated
            escalated_at=datetime.now(timezone.utc),  # But was escalated
            escalation_notified_at=None,
        )

        # Repository should NOT return this in escalations because notified_at IS NULL
        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert],  # Would be in activations
            "recoveries": [],
            "escalations": [],  # NOT here because notified_at IS NULL
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        # Only activation is sent, not escalation
        assert sent == 1
        mock_notifier.send_alert.assert_called_once_with(
            alert, is_recovery=False, is_escalation=False
        )


class TestDeliveryFailureHandling:
    """Test delivery failure scenarios."""

    @pytest.fixture
    def mock_notifier(self):
        """Create mock notifier."""
        notifier = AsyncMock(spec=TelegramNotifier)
        return notifier

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = AsyncMock(spec=OpsAlertsRepository)
        return repo

    @pytest.mark.asyncio
    async def test_failed_send_records_error(self, mock_notifier, mock_repo):
        """Failed send records error in repository."""
        alert = make_alert()

        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert],
            "recoveries": [],
            "escalations": [],
        }
        mock_notifier.send_alert.return_value = SendResult(ok=False)

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        # Send failed, not counted
        assert sent == 0
        # mark_notified not called on failure
        mock_repo.mark_notified.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_records_delivery_failed(self, mock_notifier, mock_repo):
        """Exception during send records delivery failure."""
        alert = make_alert()

        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert],
            "recoveries": [],
            "escalations": [],
        }
        mock_notifier.send_alert.side_effect = Exception("Connection timeout")

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        assert sent == 0
        mock_repo.mark_delivery_failed.assert_called_once_with(
            alert.id, "activation", "Connection timeout"
        )


class TestFullNotificationCycle:
    """Test complete notification cycle: activate → resolve → recover."""

    @pytest.fixture
    def mock_notifier(self):
        """Create mock notifier."""
        notifier = AsyncMock(spec=TelegramNotifier)
        notifier.send_alert.return_value = SendResult(ok=True, message_id="tg_cycle")
        return notifier

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = AsyncMock(spec=OpsAlertsRepository)
        repo.mark_notified.return_value = True
        return repo

    @pytest.mark.asyncio
    async def test_all_three_notification_types_in_batch(
        self, mock_notifier, mock_repo
    ):
        """Single evaluation can trigger all three notification types."""
        activation_alert = make_alert(status="active", notified_at=None)
        recovery_alert = make_alert(
            status="resolved",
            notified_at=datetime.now(timezone.utc),
            recovery_notified_at=None,
        )
        escalation_alert = make_alert(
            status="active",
            severity="critical",
            notified_at=datetime.now(timezone.utc),
            escalated_at=datetime.now(timezone.utc),
            escalation_notified_at=None,
        )

        mock_repo.get_pending_notifications.return_value = {
            "activations": [activation_alert],
            "recoveries": [recovery_alert],
            "escalations": [escalation_alert],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        # All three types sent
        assert sent == 3
        assert mock_notifier.send_alert.call_count == 3
        assert mock_repo.mark_notified.call_count == 3

        # Verify correct notification types
        calls = mock_notifier.send_alert.call_args_list
        # Activation
        assert calls[0].kwargs == {"is_recovery": False, "is_escalation": False}
        # Recovery
        assert calls[1].kwargs == {"is_recovery": True, "is_escalation": False}
        # Escalation
        assert calls[2].kwargs == {"is_recovery": False, "is_escalation": True}
