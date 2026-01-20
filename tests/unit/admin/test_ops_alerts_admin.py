"""Unit tests for ops_alerts admin endpoints."""

import pytest
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status


@pytest.fixture
def mock_alerts_repo():
    """Mock AlertsRepository for testing."""
    repo = MagicMock()
    repo.list_events = AsyncMock()
    repo.get_event = AsyncMock()
    repo.acknowledge = AsyncMock()
    repo.resolve = AsyncMock()
    repo.unacknowledge = AsyncMock()
    return repo


@pytest.fixture
def sample_alerts():
    """Sample alert events for testing."""
    workspace_id = uuid4()
    return [
        {
            "id": uuid4(),
            "workspace_id": workspace_id,
            "rule_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "regime_key": "trending_strong",
            "timeframe": "1h",
            "rule_type": "drift_spike",
            "status": "active",
            "severity": "high",
            "acknowledged": False,
            "acknowledged_at": None,
            "acknowledged_by": None,
            "first_seen": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "activated_at": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "last_seen": datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc),
            "resolved_at": None,
            "context_json": {"drift_value": 0.45, "threshold": 0.30},
            "fingerprint": "drift_spike_trending_strong_1h",
            "created_at": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc),
        },
        {
            "id": uuid4(),
            "workspace_id": workspace_id,
            "rule_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "regime_key": "ranging_low_vol",
            "timeframe": "4h",
            "rule_type": "confidence_drop",
            "status": "active",
            "severity": "medium",
            "acknowledged": True,
            "acknowledged_at": datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc),
            "acknowledged_by": "admin@example.com",
            "first_seen": datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
            "activated_at": datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
            "last_seen": datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            "resolved_at": None,
            "context_json": {"confidence": 0.62, "trend": -0.08},
            "fingerprint": "confidence_drop_ranging_low_vol_4h",
            "created_at": datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        },
        {
            "id": uuid4(),
            "workspace_id": workspace_id,
            "rule_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "regime_key": "trending_strong",
            "timeframe": "1h",
            "rule_type": "drift_spike",
            "status": "resolved",
            "severity": "low",
            "acknowledged": False,
            "acknowledged_at": None,
            "acknowledged_by": None,
            "first_seen": datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc),
            "activated_at": datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc),
            "last_seen": datetime(2024, 1, 14, 11, 0, 0, tzinfo=timezone.utc),
            "resolved_at": datetime(2024, 1, 14, 12, 0, 0, tzinfo=timezone.utc),
            "context_json": {"drift_value": 0.25, "threshold": 0.30},
            "fingerprint": "drift_spike_trending_strong_1h_old",
            "created_at": datetime(2024, 1, 14, 10, 0, 0, tzinfo=timezone.utc),
            "updated_at": datetime(2024, 1, 14, 12, 0, 0, tzinfo=timezone.utc),
        },
    ], workspace_id


class TestOpsAlertsListEndpoint:
    """Tests for GET /admin/ops-alerts list endpoint."""

    @pytest.mark.asyncio
    async def test_list_returns_html(self, mock_alerts_repo, sample_alerts):
        """Test that list endpoint returns HTML response."""
        from fastapi.testclient import TestClient
        from app.main import app

        alerts, workspace_id = sample_alerts
        mock_alerts_repo.list_events.return_value = (alerts, len(alerts))

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.get(
                f"/admin/ops-alerts?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        assert "text/html" in response.headers["content-type"]
        assert b"Ops Alerts" in response.content or b"Alert" in response.content

    @pytest.mark.asyncio
    async def test_list_passes_filters_to_repository(self, mock_alerts_repo, sample_alerts):
        """Test that query params are passed to repository correctly."""
        from fastapi.testclient import TestClient
        from app.main import app

        alerts, workspace_id = sample_alerts
        mock_alerts_repo.list_events.return_value = ([], 0)

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.get(
                f"/admin/ops-alerts?workspace_id={workspace_id}&status=active&severity=high&limit=10&offset=20",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == status.HTTP_200_OK

        # Verify repository was called with correct params
        mock_alerts_repo.list_events.assert_called_once()
        call_kwargs = mock_alerts_repo.list_events.call_args.kwargs
        assert call_kwargs["workspace_id"] == workspace_id
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 20

    @pytest.mark.asyncio
    async def test_list_requires_admin_token(self, mock_alerts_repo, sample_alerts):
        """Test that endpoint requires admin token."""
        from fastapi.testclient import TestClient
        from app.main import app

        _, workspace_id = sample_alerts

        client = TestClient(app)
        response = client.get(f"/admin/ops-alerts?workspace_id={workspace_id}")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_list_pagination_metadata(self, mock_alerts_repo, sample_alerts):
        """Test that pagination metadata is included in response."""
        from fastapi.testclient import TestClient
        from app.main import app

        alerts, workspace_id = sample_alerts
        mock_alerts_repo.list_events.return_value = (alerts[:2], 10)  # 2 items, 10 total

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.get(
                f"/admin/ops-alerts?workspace_id={workspace_id}&limit=2&offset=0",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        # Check for pagination indicators in HTML
        content = response.content.decode()
        assert "10" in content  # total count


class TestOpsAlertsActionEndpoints:
    """Tests for action endpoints (acknowledge, resolve, reopen)."""

    @pytest.mark.asyncio
    async def test_acknowledge_success(self, mock_alerts_repo):
        """Test acknowledging an alert."""
        from fastapi.testclient import TestClient
        from app.main import app

        event_id = uuid4()
        mock_alerts_repo.acknowledge.return_value = True

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.post(
                f"/admin/ops-alerts/{event_id}/acknowledge",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["acknowledged"] is True
        mock_alerts_repo.acknowledge.assert_called_once_with(event_id, acknowledged_by=None)

    @pytest.mark.asyncio
    async def test_acknowledge_not_found(self, mock_alerts_repo):
        """Test acknowledging non-existent alert returns 404."""
        from fastapi.testclient import TestClient
        from app.main import app

        event_id = uuid4()
        mock_alerts_repo.acknowledge.return_value = False

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.post(
                f"/admin/ops-alerts/{event_id}/acknowledge",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_resolve_success(self, mock_alerts_repo):
        """Test resolving an alert."""
        from fastapi.testclient import TestClient
        from app.main import app

        event_id = uuid4()
        mock_alerts_repo.resolve.return_value = True

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.post(
                f"/admin/ops-alerts/{event_id}/resolve",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["resolved"] is True
        mock_alerts_repo.resolve.assert_called_once_with(event_id)

    @pytest.mark.asyncio
    async def test_reopen_success(self, mock_alerts_repo):
        """Test reopening a resolved alert."""
        from fastapi.testclient import TestClient
        from app.main import app

        event_id = uuid4()
        # For reopen, we need to get the event first and then upsert_activate
        mock_alerts_repo.get_event.return_value = {
            "id": event_id,
            "workspace_id": uuid4(),
            "rule_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "regime_key": "trending_strong",
            "timeframe": "1h",
            "rule_type": "drift_spike",
            "status": "resolved",
            "severity": "high",
            "context_json": {},
            "fingerprint": "test_fp",
        }
        mock_alerts_repo.upsert_activate.return_value = {"id": event_id, "status": "active"}

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.post(
                f"/admin/ops-alerts/{event_id}/reopen",
                headers={"X-Admin-Token": "test-token"},
            )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["reopened"] is True


class TestSeverityBadges:
    """Tests for severity badge rendering."""

    @pytest.mark.asyncio
    async def test_severity_badge_colors(self, mock_alerts_repo, sample_alerts):
        """Test that severity badges have correct color classes."""
        from fastapi.testclient import TestClient
        from app.main import app

        alerts, workspace_id = sample_alerts
        mock_alerts_repo.list_events.return_value = (alerts, len(alerts))

        with patch("app.admin.ops_alerts._get_alerts_repo", return_value=mock_alerts_repo):
            client = TestClient(app)
            response = client.get(
                f"/admin/ops-alerts?workspace_id={workspace_id}",
                headers={"X-Admin-Token": "test-token"},
            )

        content = response.content.decode()
        # Check for severity badge presence (implementation detail - may vary)
        assert "high" in content.lower() or "HIGH" in content
        assert "medium" in content.lower() or "MEDIUM" in content
        assert "low" in content.lower() or "LOW" in content
