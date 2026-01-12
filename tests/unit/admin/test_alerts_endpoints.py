"""Tests for admin alerts endpoints (rules and events)."""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Set required environment variables for tests before importing app
os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    return MagicMock()


# =============================================================================
# Alert Rules - List
# =============================================================================


class TestListAlertRulesEndpoint:
    """Tests for GET /admin/alerts/rules endpoint."""

    def test_list_rules_requires_admin_token(self, client):
        """List rules requires admin auth."""
        response = client.get("/admin/alerts/rules?workspace_id=" + str(uuid4()))
        assert response.status_code in (401, 403)

    def test_list_rules_requires_workspace_id(self, client):
        """List rules requires workspace_id parameter."""
        response = client.get(
            "/admin/alerts/rules",
            headers={"X-Admin-Token": "test-token"},
        )
        assert response.status_code == 422

    def test_list_rules_success(self, client, mock_db_pool):
        """List rules returns rules for workspace."""
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_rules = AsyncMock(
            return_value=[
                {
                    "id": rule_id,
                    "workspace_id": workspace_id,
                    "rule_type": "drift_spike",
                    "strategy_entity_id": strategy_id,
                    "regime_key": "btc_high_vol",
                    "timeframe": "1d",
                    "enabled": True,
                    "config": {"drift_threshold": 0.30},
                    "cooldown_minutes": 60,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            ]
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/rules?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "rules" in data
        assert len(data["rules"]) == 1
        assert data["rules"][0]["rule_type"] == "drift_spike"
        assert data["rules"][0]["enabled"] is True

    def test_list_rules_with_enabled_filter(self, client, mock_db_pool):
        """List rules accepts enabled_only filter."""
        workspace_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_rules = AsyncMock(return_value=[])

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/rules?workspace_id={workspace_id}&enabled_only=true",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        mock_repo.list_rules.assert_called_once()
        call_kwargs = mock_repo.list_rules.call_args[1]
        assert call_kwargs["enabled_only"] is True


# =============================================================================
# Alert Rules - Create
# =============================================================================


class TestCreateAlertRuleEndpoint:
    """Tests for POST /admin/alerts/rules endpoint."""

    def test_create_rule_requires_admin_token(self, client):
        """Create rule requires admin auth."""
        response = client.post(
            f"/admin/alerts/rules?workspace_id={uuid4()}",
            json={"rule_type": "drift_spike", "config": {}},
        )
        assert response.status_code in (401, 403)

    def test_create_rule_success(self, client, mock_db_pool):
        """Create rule creates and returns rule."""
        workspace_id = uuid4()
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.create_rule = AsyncMock(
            return_value={
                "id": rule_id,
                "workspace_id": workspace_id,
                "rule_type": "drift_spike",
                "strategy_entity_id": None,
                "regime_key": None,
                "timeframe": None,
                "enabled": True,
                "config": {"drift_threshold": 0.30},
                "cooldown_minutes": 60,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/rules?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
                json={
                    "rule_type": "drift_spike",
                    "config": {"drift_threshold": 0.30},
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["rule_type"] == "drift_spike"
        assert data["config"]["drift_threshold"] == 0.30

    def test_create_rule_with_all_fields(self, client, mock_db_pool):
        """Create rule accepts all optional fields."""
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.create_rule = AsyncMock(
            return_value={
                "id": rule_id,
                "workspace_id": workspace_id,
                "rule_type": "combo",
                "strategy_entity_id": strategy_id,
                "regime_key": "eth_low_vol",
                "timeframe": "4h",
                "enabled": True,
                "config": {"drift_threshold": 0.25},
                "cooldown_minutes": 120,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/rules?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
                json={
                    "rule_type": "combo",
                    "strategy_entity_id": str(strategy_id),
                    "regime_key": "eth_low_vol",
                    "timeframe": "4h",
                    "config": {"drift_threshold": 0.25},
                    "cooldown_minutes": 120,
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["rule_type"] == "combo"
        assert data["regime_key"] == "eth_low_vol"
        assert data["cooldown_minutes"] == 120

    def test_create_rule_invalid_type(self, client, mock_db_pool):
        """Create rule validates rule type."""
        workspace_id = uuid4()

        response = client.post(
            f"/admin/alerts/rules?workspace_id={workspace_id}",
            headers={"X-Admin-Token": "test-token"},
            json={"rule_type": "invalid_type", "config": {}},
        )

        assert response.status_code == 422


# =============================================================================
# Alert Rules - Get
# =============================================================================


class TestGetAlertRuleEndpoint:
    """Tests for GET /admin/alerts/rules/{id} endpoint."""

    def test_get_rule_requires_admin_token(self, client):
        """Get rule requires admin auth."""
        rule_id = uuid4()
        response = client.get(f"/admin/alerts/rules/{rule_id}")
        assert response.status_code in (401, 403)

    def test_get_rule_success(self, client, mock_db_pool):
        """Get rule returns rule details."""
        rule_id = uuid4()
        workspace_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_rule = AsyncMock(
            return_value={
                "id": rule_id,
                "workspace_id": workspace_id,
                "rule_type": "confidence_drop",
                "strategy_entity_id": None,
                "regime_key": None,
                "timeframe": None,
                "enabled": True,
                "config": {"trend_threshold": 0.05},
                "cooldown_minutes": 60,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/rules/{rule_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rule_type"] == "confidence_drop"
        assert data["config"]["trend_threshold"] == 0.05

    def test_get_rule_not_found(self, client, mock_db_pool):
        """Get rule returns 404 when not found."""
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_rule = AsyncMock(return_value=None)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/rules/{rule_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# Alert Rules - Update
# =============================================================================


class TestUpdateAlertRuleEndpoint:
    """Tests for PATCH /admin/alerts/rules/{id} endpoint."""

    def test_update_rule_requires_admin_token(self, client):
        """Update rule requires admin auth."""
        rule_id = uuid4()
        response = client.patch(
            f"/admin/alerts/rules/{rule_id}", json={"enabled": False}
        )
        assert response.status_code in (401, 403)

    def test_update_rule_success(self, client, mock_db_pool):
        """Update rule updates and returns rule."""
        rule_id = uuid4()
        workspace_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.update_rule = AsyncMock(
            return_value={
                "id": rule_id,
                "workspace_id": workspace_id,
                "rule_type": "drift_spike",
                "strategy_entity_id": None,
                "regime_key": None,
                "timeframe": None,
                "enabled": False,
                "config": {"drift_threshold": 0.40},
                "cooldown_minutes": 90,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.patch(
                f"/admin/alerts/rules/{rule_id}",
                headers={"X-Admin-Token": "test-token"},
                json={
                    "enabled": False,
                    "config": {"drift_threshold": 0.40},
                    "cooldown_minutes": 90,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["config"]["drift_threshold"] == 0.40
        assert data["cooldown_minutes"] == 90

    def test_update_rule_not_found(self, client, mock_db_pool):
        """Update rule returns 404 when not found."""
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.update_rule = AsyncMock(return_value=None)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.patch(
                f"/admin/alerts/rules/{rule_id}",
                headers={"X-Admin-Token": "test-token"},
                json={"enabled": False},
            )

        assert response.status_code == 404


# =============================================================================
# Alert Rules - Delete
# =============================================================================


class TestDeleteAlertRuleEndpoint:
    """Tests for DELETE /admin/alerts/rules/{id} endpoint."""

    def test_delete_rule_requires_admin_token(self, client):
        """Delete rule requires admin auth."""
        rule_id = uuid4()
        response = client.delete(f"/admin/alerts/rules/{rule_id}")
        assert response.status_code in (401, 403)

    def test_delete_rule_success(self, client, mock_db_pool):
        """Delete rule deletes and returns success."""
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.delete_rule = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.delete(
                f"/admin/alerts/rules/{rule_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 204

    def test_delete_rule_not_found(self, client, mock_db_pool):
        """Delete rule returns 404 when not found."""
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.delete_rule = AsyncMock(return_value=False)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.delete(
                f"/admin/alerts/rules/{rule_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404


# =============================================================================
# Alert Events - List
# =============================================================================


class TestListAlertEventsEndpoint:
    """Tests for GET /admin/alerts endpoint."""

    def test_list_events_requires_admin_token(self, client):
        """List events requires admin auth."""
        response = client.get("/admin/alerts?workspace_id=" + str(uuid4()))
        assert response.status_code in (401, 403)

    def test_list_events_requires_workspace_id(self, client):
        """List events requires workspace_id parameter."""
        response = client.get(
            "/admin/alerts",
            headers={"X-Admin-Token": "test-token"},
        )
        assert response.status_code == 422

    def test_list_events_success(self, client, mock_db_pool):
        """List events returns events for workspace."""
        workspace_id = uuid4()
        event_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_events = AsyncMock(
            return_value=(
                [
                    {
                        "id": event_id,
                        "workspace_id": workspace_id,
                        "rule_id": rule_id,
                        "strategy_entity_id": strategy_id,
                        "regime_key": "btc_high_vol",
                        "timeframe": "1d",
                        "rule_type": "drift_spike",
                        "status": "active",
                        "severity": "high",
                        "acknowledged": False,
                        "acknowledged_at": None,
                        "acknowledged_by": None,
                        "first_seen": datetime.now(timezone.utc),
                        "activated_at": datetime.now(timezone.utc),
                        "last_seen": datetime.now(timezone.utc),
                        "resolved_at": None,
                        "context_json": {"drift_value": 0.35},
                        "fingerprint": "abc123",
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    }
                ],
                1,
            )
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert len(data["events"]) == 1
        assert data["events"][0]["status"] == "active"
        assert data["events"][0]["severity"] == "high"
        assert data["total"] == 1

    def test_list_events_with_filters(self, client, mock_db_pool):
        """List events accepts multiple filters."""
        workspace_id = uuid4()
        strategy_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.list_events = AsyncMock(return_value=([], 0))

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts?workspace_id={workspace_id}"
                f"&status=active&severity=high&acknowledged=false"
                f"&rule_type=drift_spike&strategy_entity_id={strategy_id}"
                f"&limit=25&offset=10",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        mock_repo.list_events.assert_called_once()
        call_kwargs = mock_repo.list_events.call_args[1]
        assert call_kwargs["status"].value == "active"
        assert call_kwargs["severity"].value == "high"
        assert call_kwargs["acknowledged"] is False
        assert call_kwargs["rule_type"].value == "drift_spike"
        assert call_kwargs["strategy_entity_id"] == strategy_id
        assert call_kwargs["limit"] == 25
        assert call_kwargs["offset"] == 10


# =============================================================================
# Alert Events - Get
# =============================================================================


class TestGetAlertEventEndpoint:
    """Tests for GET /admin/alerts/{id} endpoint."""

    def test_get_event_requires_admin_token(self, client):
        """Get event requires admin auth."""
        event_id = uuid4()
        response = client.get(f"/admin/alerts/{event_id}")
        assert response.status_code in (401, 403)

    def test_get_event_success(self, client, mock_db_pool):
        """Get event returns event details."""
        event_id = uuid4()
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_event = AsyncMock(
            return_value={
                "id": event_id,
                "workspace_id": workspace_id,
                "rule_id": rule_id,
                "strategy_entity_id": strategy_id,
                "regime_key": "btc_high_vol",
                "timeframe": "1d",
                "rule_type": "drift_spike",
                "status": "active",
                "severity": "medium",
                "acknowledged": True,
                "acknowledged_at": datetime.now(timezone.utc),
                "acknowledged_by": "admin@example.com",
                "first_seen": datetime.now(timezone.utc),
                "activated_at": datetime.now(timezone.utc),
                "last_seen": datetime.now(timezone.utc),
                "resolved_at": None,
                "context_json": {"drift_value": 0.32},
                "fingerprint": "xyz789",
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }
        )

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/{event_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert data["acknowledged"] is True
        assert data["acknowledged_by"] == "admin@example.com"

    def test_get_event_not_found(self, client, mock_db_pool):
        """Get event returns 404 when not found."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_event = AsyncMock(return_value=None)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/{event_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# Alert Events - Acknowledge
# =============================================================================


class TestAcknowledgeAlertEventEndpoint:
    """Tests for POST /admin/alerts/{id}/acknowledge endpoint."""

    def test_acknowledge_requires_admin_token(self, client):
        """Acknowledge requires admin auth."""
        event_id = uuid4()
        response = client.post(f"/admin/alerts/{event_id}/acknowledge")
        assert response.status_code in (401, 403)

    def test_acknowledge_success(self, client, mock_db_pool):
        """Acknowledge event sets acknowledged flag."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.acknowledge = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/acknowledge",
                headers={"X-Admin-Token": "test-token"},
                json={"acknowledged_by": "admin@example.com"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is True
        mock_repo.acknowledge.assert_called_once_with(
            event_id, acknowledged_by="admin@example.com"
        )

    def test_acknowledge_without_user(self, client, mock_db_pool):
        """Acknowledge event works without acknowledged_by."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.acknowledge = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/acknowledge",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        mock_repo.acknowledge.assert_called_once_with(event_id, acknowledged_by=None)

    def test_acknowledge_not_found(self, client, mock_db_pool):
        """Acknowledge returns 404 when event not found."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.acknowledge = AsyncMock(return_value=False)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/acknowledge",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404


# =============================================================================
# Alert Events - Unacknowledge
# =============================================================================


class TestUnacknowledgeAlertEventEndpoint:
    """Tests for POST /admin/alerts/{id}/unacknowledge endpoint."""

    def test_unacknowledge_requires_admin_token(self, client):
        """Unacknowledge requires admin auth."""
        event_id = uuid4()
        response = client.post(f"/admin/alerts/{event_id}/unacknowledge")
        assert response.status_code in (401, 403)

    def test_unacknowledge_success(self, client, mock_db_pool):
        """Unacknowledge event clears acknowledged flag."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.unacknowledge = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/unacknowledge",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is False

    def test_unacknowledge_not_found(self, client, mock_db_pool):
        """Unacknowledge returns 404 when event not found."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.unacknowledge = AsyncMock(return_value=False)

        with patch("app.admin.alerts._db_pool", mock_db_pool), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/unacknowledge",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == 404
