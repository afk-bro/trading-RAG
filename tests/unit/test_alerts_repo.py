"""Tests for alerts repository."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.repositories.alerts import AlertsRepository
from app.services.alerts.models import RuleType, Severity, AlertStatus


class TestAlertRulesRepository:
    """Tests for alert rules operations."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        pool.acquire = MagicMock()
        return pool

    @pytest.mark.asyncio
    async def test_list_rules_for_workspace(self, mock_pool):
        """List enabled rules for workspace."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": rule_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": None,
                    "regime_key": None,
                    "timeframe": "1h",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30},
                    "cooldown_minutes": 60,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        rules = await repo.list_rules(workspace_id, enabled_only=True)

        assert len(rules) == 1
        assert rules[0]["rule_type"] == "drift_spike"
        mock_conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_rule(self, mock_pool):
        """Create new alert rule."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": rule_id,
                "workspace_id": workspace_id,
                "rule_type": "drift_spike",
                "enabled": True,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        result = await repo.create_rule(
            workspace_id=workspace_id,
            rule_type=RuleType.DRIFT_SPIKE,
            config={"drift_threshold": 0.30},
        )

        assert result["id"] == rule_id
        mock_conn.fetchrow.assert_called_once()


class TestAlertEventsRepository:
    """Tests for alert events operations."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        pool.acquire = MagicMock()
        return pool

    @pytest.mark.asyncio
    async def test_list_events_with_filters(self, mock_pool):
        """List events with status and severity filters."""
        workspace_id = uuid4()
        event_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": event_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "status": "active",
                    "severity": "medium",
                    "acknowledged": False,
                    "last_seen": datetime.now(timezone.utc),
                }
            ]
        )
        mock_conn.fetchval = AsyncMock(return_value=1)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        events, total = await repo.list_events(
            workspace_id=workspace_id,
            status=AlertStatus.ACTIVE,
            acknowledged=False,
        )

        assert len(events) == 1
        assert events[0]["status"] == "active"
        assert total == 1

    @pytest.mark.asyncio
    async def test_upsert_activate(self, mock_pool):
        """Upsert activates alert event."""
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()
        event_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": event_id})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        result = await repo.upsert_activate(
            workspace_id=workspace_id,
            rule_id=rule_id,
            strategy_entity_id=strategy_id,
            regime_key="high_vol/uptrend",
            timeframe="1h",
            rule_type=RuleType.DRIFT_SPIKE,
            severity=Severity.MEDIUM,
            context_json={"threshold": 0.30},
            fingerprint="v1:high_vol/uptrend:1h",
        )

        assert result["id"] == event_id

    @pytest.mark.asyncio
    async def test_acknowledge_event(self, mock_pool):
        """Acknowledge alert event."""
        event_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        repo = AlertsRepository(mock_pool)
        success = await repo.acknowledge(event_id, acknowledged_by="admin")

        assert success is True
        mock_conn.execute.assert_called_once()
