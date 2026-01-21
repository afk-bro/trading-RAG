"""Unit tests for ops alert Telegram delivery idempotency.

Tests the idempotent notification delivery system:
- DB-driven pending queries
- Conditional mark_notified() for race prevention
- Escalation reset on severity bump
- Handler wiring with _send_notifications()

Run with: pytest tests/unit/test_ops_alert_telegram_delivery.py -v
"""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Set required env vars before importing app modules
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-key")

from app.repositories.ops_alerts import OpsAlert, OpsAlertsRepository  # noqa: E402
from app.services.ops_alerts.telegram import SendResult, TelegramNotifier  # noqa: E402

# Try to import handler - may fail if ccxt not installed
try:
    from app.jobs.handlers.ops_alert_eval import _send_notifications

    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    _send_notifications = None


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
        resolved_at=None,
        acknowledged_at=None,
        acknowledged_by=None,
        occurrence_count=1,
        notified_at=notified_at,
        recovery_notified_at=recovery_notified_at,
        escalated_at=escalated_at,
        escalation_notified_at=escalation_notified_at,
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


class TestGetPendingNotifications:
    """Test repository get_pending_notifications() query logic."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool, conn

    @pytest.mark.asyncio
    async def test_returns_activations_recoveries_escalations(self, mock_pool):
        """get_pending_notifications returns all three categories."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        # Mock the three queries returning different alert rows
        activation_row = {
            "id": uuid4(),
            "workspace_id": uuid4(),
            "rule_type": "health_degraded",
            "severity": "high",
            "status": "active",
            "rule_version": "v1",
            "dedupe_key": "health_degraded:2026-01-19",
            "payload": {},
            "source": "alert_evaluator",
            "job_run_id": None,
            "created_at": datetime.now(timezone.utc),
            "last_seen_at": datetime.now(timezone.utc),
            "resolved_at": None,
            "acknowledged_at": None,
            "acknowledged_by": None,
            "occurrence_count": 1,
            "notified_at": None,
            "recovery_notified_at": None,
            "escalated_at": None,
            "escalation_notified_at": None,
            "telegram_message_id": None,
            "delivery_attempts": 0,
            "last_delivery_error": None,
        }
        recovery_row = {**activation_row, "id": uuid4(), "status": "resolved"}
        escalation_row = {
            **activation_row,
            "id": uuid4(),
            "escalated_at": datetime.now(timezone.utc),
            "notified_at": datetime.now(timezone.utc),  # Already activated
        }

        conn.fetch.side_effect = [
            [activation_row],  # activations query
            [recovery_row],  # recoveries query
            [escalation_row],  # escalations query
        ]

        result = await repo.get_pending_notifications(uuid4())

        assert len(result["activations"]) == 1
        assert len(result["recoveries"]) == 1
        assert len(result["escalations"]) == 1

    @pytest.mark.asyncio
    async def test_escalations_require_notified_at(self, mock_pool):
        """Escalations only include already-activated alerts."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        # All empty returns
        conn.fetch.return_value = []

        workspace_id = uuid4()
        await repo.get_pending_notifications(workspace_id)

        # Check the escalation query includes notified_at IS NOT NULL
        calls = conn.fetch.call_args_list
        escalation_call = calls[2]  # Third call is escalations
        query = escalation_call[0][0]

        assert "notified_at IS NOT NULL" in query


class TestMarkNotified:
    """Test repository mark_notified() conditional update."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool, conn

    @pytest.mark.asyncio
    async def test_mark_notified_returns_true_on_first_call(self, mock_pool):
        """mark_notified returns True when it wins the race."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        alert_id = uuid4()
        conn.fetchrow.return_value = {"id": alert_id}  # Row returned = won race

        result = await repo.mark_notified(alert_id, "activation", "msg123")

        assert result is True

    @pytest.mark.asyncio
    async def test_mark_notified_returns_false_on_race_loss(self, mock_pool):
        """mark_notified returns False when another worker already marked it."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        alert_id = uuid4()
        conn.fetchrow.return_value = None  # No row = lost race

        result = await repo.mark_notified(alert_id, "activation", "msg456")

        assert result is False

    @pytest.mark.asyncio
    async def test_mark_notified_activation_uses_notified_at(self, mock_pool):
        """activation type updates notified_at column."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        conn.fetchrow.return_value = {"id": uuid4()}
        await repo.mark_notified(uuid4(), "activation", None)

        query = conn.fetchrow.call_args[0][0]
        assert "notified_at = NOW()" in query
        assert "notified_at IS NULL" in query

    @pytest.mark.asyncio
    async def test_mark_notified_recovery_uses_recovery_notified_at(self, mock_pool):
        """recovery type updates recovery_notified_at column."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        conn.fetchrow.return_value = {"id": uuid4()}
        await repo.mark_notified(uuid4(), "recovery", None)

        query = conn.fetchrow.call_args[0][0]
        assert "recovery_notified_at = NOW()" in query
        assert "recovery_notified_at IS NULL" in query

    @pytest.mark.asyncio
    async def test_mark_notified_escalation_uses_escalation_notified_at(
        self, mock_pool
    ):
        """escalation type updates escalation_notified_at column."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        conn.fetchrow.return_value = {"id": uuid4()}
        await repo.mark_notified(uuid4(), "escalation", None)

        query = conn.fetchrow.call_args[0][0]
        assert "escalation_notified_at = NOW()" in query
        assert "escalation_notified_at IS NULL" in query

    @pytest.mark.asyncio
    async def test_mark_notified_invalid_type_raises(self, mock_pool):
        """Invalid notification_type raises ValueError."""
        pool, _ = mock_pool
        repo = OpsAlertsRepository(pool)

        with pytest.raises(ValueError, match="Invalid notification_type"):
            await repo.mark_notified(uuid4(), "invalid_type", None)


class TestMarkDeliveryFailed:
    """Test repository mark_delivery_failed() conditional update."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool, conn

    @pytest.mark.asyncio
    async def test_mark_delivery_failed_only_if_not_notified(self, mock_pool):
        """mark_delivery_failed only updates if not already notified."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        await repo.mark_delivery_failed(uuid4(), "activation", "Connection timeout")

        query = conn.execute.call_args[0][0]
        assert "notified_at IS NULL" in query
        assert "delivery_attempts = delivery_attempts + 1" in query
        assert "last_delivery_error" in query


@pytest.mark.skipif(not HANDLER_AVAILABLE, reason="ccxt not installed")
class TestSendNotificationsHandler:
    """Test _send_notifications handler wiring."""

    @pytest.fixture
    def mock_notifier(self):
        """Create mock notifier."""
        notifier = AsyncMock(spec=TelegramNotifier)
        notifier.send_alert.return_value = SendResult(ok=True, message_id="tg123")
        return notifier

    @pytest.fixture
    def mock_repo(self):
        """Create mock repository."""
        repo = AsyncMock(spec=OpsAlertsRepository)
        repo.get_pending_notifications.return_value = {
            "activations": [],
            "recoveries": [],
            "escalations": [],
        }
        repo.mark_notified.return_value = True
        return repo

    @pytest.mark.asyncio
    async def test_send_notifications_queries_db_not_eval_result(
        self, mock_notifier, mock_repo
    ):
        """_send_notifications queries DB instead of using eval_result."""

        workspace_id = uuid4()
        await _send_notifications(mock_notifier, mock_repo, workspace_id)

        mock_repo.get_pending_notifications.assert_called_once_with(workspace_id)

    @pytest.mark.asyncio
    async def test_send_notifications_sends_activations(self, mock_notifier, mock_repo):
        """_send_notifications sends activation notifications."""
        alert = make_alert(status="active", notified_at=None)
        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert],
            "recoveries": [],
            "escalations": [],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        mock_notifier.send_alert.assert_called_once_with(
            alert, is_recovery=False, is_escalation=False
        )
        mock_repo.mark_notified.assert_called_once_with(alert.id, "activation", "tg123")
        assert sent == 1

    @pytest.mark.asyncio
    async def test_send_notifications_sends_recoveries(self, mock_notifier, mock_repo):
        """_send_notifications sends recovery notifications."""
        alert = make_alert(status="resolved", recovery_notified_at=None)
        mock_repo.get_pending_notifications.return_value = {
            "activations": [],
            "recoveries": [alert],
            "escalations": [],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        mock_notifier.send_alert.assert_called_once_with(
            alert, is_recovery=True, is_escalation=False
        )
        mock_repo.mark_notified.assert_called_once_with(alert.id, "recovery", "tg123")
        assert sent == 1

    @pytest.mark.asyncio
    async def test_send_notifications_sends_escalations(self, mock_notifier, mock_repo):
        """_send_notifications sends escalation notifications."""
        alert = make_alert(
            status="active",
            notified_at=datetime.now(timezone.utc),
            escalated_at=datetime.now(timezone.utc),
            escalation_notified_at=None,
        )
        mock_repo.get_pending_notifications.return_value = {
            "activations": [],
            "recoveries": [],
            "escalations": [alert],
        }

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        mock_notifier.send_alert.assert_called_once_with(
            alert, is_recovery=False, is_escalation=True
        )
        mock_repo.mark_notified.assert_called_once_with(alert.id, "escalation", "tg123")
        assert sent == 1

    @pytest.mark.asyncio
    async def test_send_notifications_counts_only_won_races(
        self, mock_notifier, mock_repo
    ):
        """_send_notifications only counts messages where mark_notified returns True."""
        alert1 = make_alert()
        alert2 = make_alert()
        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert1, alert2],
            "recoveries": [],
            "escalations": [],
        }
        # First call wins, second loses the race
        mock_repo.mark_notified.side_effect = [True, False]

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        assert mock_notifier.send_alert.call_count == 2
        assert mock_repo.mark_notified.call_count == 2
        assert sent == 1  # Only the winner counts

    @pytest.mark.asyncio
    async def test_send_notifications_records_failures(self, mock_notifier, mock_repo):
        """_send_notifications records delivery failures."""
        alert = make_alert()
        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert],
            "recoveries": [],
            "escalations": [],
        }
        mock_notifier.send_alert.return_value = SendResult(ok=False)

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        # No mark_notified since send failed
        mock_repo.mark_notified.assert_not_called()
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_notifications_handles_exceptions(
        self, mock_notifier, mock_repo
    ):
        """_send_notifications handles exceptions gracefully."""
        alert = make_alert()
        mock_repo.get_pending_notifications.return_value = {
            "activations": [alert],
            "recoveries": [],
            "escalations": [],
        }
        mock_notifier.send_alert.side_effect = Exception("Network error")

        workspace_id = uuid4()
        sent = await _send_notifications(mock_notifier, mock_repo, workspace_id)

        mock_repo.mark_delivery_failed.assert_called_once_with(
            alert.id, "activation", "Network error"
        )
        assert sent == 0


class TestUpsertEscalationTracking:
    """Test upsert() escalation tracking in SQL."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock()
        return pool, conn

    @pytest.mark.asyncio
    async def test_upsert_escalation_clears_escalation_notified_at(self, mock_pool):
        """upsert() clears escalation_notified_at when severity increases."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        # Mock existing alert with medium severity
        conn.fetchrow.side_effect = [
            {"severity": "medium"},  # existing_query
            {
                "id": uuid4(),
                "is_new": False,
                "current_severity": "critical",
            },  # upsert RETURNING
        ]

        await repo.upsert(
            workspace_id=uuid4(),
            rule_type="health_degraded",
            severity="critical",  # Escalating from medium
            dedupe_key="health_degraded:2026-01-19",
            payload={},
        )

        # Check the upsert query has escalation tracking
        upsert_call = conn.fetchrow.call_args_list[1]
        query = upsert_call[0][0]

        # Should set escalated_at to NOW() and clear escalation_notified_at
        assert "escalated_at = CASE" in query
        assert "escalation_notified_at = CASE" in query
        assert "THEN NOW()" in query
        assert "THEN NULL" in query

    @pytest.mark.asyncio
    async def test_upsert_severity_rank_uses_numeric_comparison(self, mock_pool):
        """upsert() uses numeric severity rank for escalation detection."""
        pool, conn = mock_pool
        repo = OpsAlertsRepository(pool)

        conn.fetchrow.side_effect = [
            None,  # No existing alert
            {"id": uuid4(), "is_new": True, "current_severity": "high"},
        ]

        await repo.upsert(
            workspace_id=uuid4(),
            rule_type="health_degraded",
            severity="high",
            dedupe_key="health_degraded:2026-01-19",
            payload={},
        )

        # Check the severity ranking SQL
        upsert_call = conn.fetchrow.call_args_list[1]
        query = upsert_call[0][0]

        # Should use numeric ranking: critical=4, high=3, medium=2, low=1
        assert "WHEN 'critical' THEN 4" in query
        assert "WHEN 'high' THEN 3" in query
        assert "WHEN 'medium' THEN 2" in query
