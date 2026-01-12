"""Integration tests for alerts system.

Tests full flows including:
- Rule creation -> job evaluation -> alert creation
- Alert lifecycle: active -> ack -> resolve
- API endpoints integration
"""

import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

# Set admin token for tests
os.environ.setdefault("ADMIN_TOKEN", "test-token")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockBucket:
    """Mock bucket for evaluator tests."""

    drift_score: float
    avg_confidence: float


@pytest.fixture
def mock_db_pool():
    """Create mock database pool with async context manager."""
    pool = MagicMock()
    conn = MagicMock()

    # Mock async context manager for pool.acquire()
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=conn)
    async_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = async_cm

    # Mock base methods
    conn.fetchval = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="UPDATE 0")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)

    return pool, conn


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def admin_headers():
    """Headers with admin token."""
    return {"X-Admin-Token": "test-token"}


# =============================================================================
# Full Flow Tests: Create Rule -> Run Job -> Verify Alert
# =============================================================================


class TestFullAlertFlow:
    """Tests for complete alert flow from rule creation to alert activation."""

    @pytest.mark.asyncio
    async def test_drift_spike_rule_triggers_alert(self, mock_db_pool):
        """
        Full flow test:
        1. Create drift_spike rule
        2. Mock bucket data that exceeds threshold
        3. Run evaluate_alerts job
        4. Verify alert event was created
        """
        pool, conn = mock_db_pool
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()
        event_id = uuid4()

        # Setup: Mock advisory lock acquisition
        async def mock_fetchval(query, *args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            return uuid4()

        conn.fetchval = AsyncMock(side_effect=mock_fetchval)

        # Setup: Mock rule exists and is enabled
        mock_rule = {
            "id": rule_id,
            "workspace_id": workspace_id,
            "rule_type": "drift_spike",
            "strategy_entity_id": strategy_id,
            "regime_key": "btc_high_vol",
            "timeframe": "1h",
            "enabled": True,
            "config": {"drift_threshold": 0.30, "consecutive_buckets": 2},
            "cooldown_minutes": 60,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        conn.fetch = AsyncMock(return_value=[mock_rule])

        # Setup: No existing event
        conn.fetchrow = AsyncMock(return_value=None)

        # Mock upsert returns new event
        upsert_result = {
            "id": event_id,
            "workspace_id": workspace_id,
            "status": "active",
            "activated_at": datetime.now(timezone.utc),
            "last_seen": datetime.now(timezone.utc),
        }

        # Track upsert calls
        upsert_calls = []

        async def mock_fetchrow(query, *args):
            if "INSERT INTO alert_events" in query:
                upsert_calls.append(args)
                return upsert_result
            return None

        conn.fetchrow = AsyncMock(side_effect=mock_fetchrow)

        # Create mock buckets that trigger alert (drift > threshold for 2 consecutive)
        mock_buckets = [
            MockBucket(drift_score=0.35, avg_confidence=0.80),
            MockBucket(drift_score=0.38, avg_confidence=0.78),
        ]

        # Run job with mocked bucket fetching
        from app.services.alerts.job import AlertEvaluatorJob

        job = AlertEvaluatorJob(pool)

        # Patch _fetch_buckets to return our mock data
        with patch.object(job, "_fetch_buckets", AsyncMock(return_value=mock_buckets)):
            result = await job.run(workspace_id=workspace_id)

        # Verify job completed
        assert result["lock_acquired"] is True
        assert result["status"] == "completed"

        # Verify metrics
        metrics = result["metrics"]
        assert metrics["rules_loaded"] == 1
        assert metrics["tuples_evaluated"] == 1
        assert metrics["alerts_activated"] == 1
        assert metrics["db_upserts"] == 1

    @pytest.mark.asyncio
    async def test_confidence_drop_rule_triggers_alert(self, mock_db_pool):
        """
        Test confidence drop rule triggers alert when trend exceeds threshold.
        """
        pool, conn = mock_db_pool
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()
        event_id = uuid4()

        # Setup mocks
        async def mock_fetchval(query, *args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            return uuid4()

        conn.fetchval = AsyncMock(side_effect=mock_fetchval)

        mock_rule = {
            "id": rule_id,
            "workspace_id": workspace_id,
            "rule_type": "confidence_drop",
            "strategy_entity_id": strategy_id,
            "regime_key": "eth_low_vol",
            "timeframe": "4h",
            "enabled": True,
            "config": {"trend_threshold": 0.05, "hysteresis": 0.02},
            "cooldown_minutes": 120,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        conn.fetch = AsyncMock(return_value=[mock_rule])

        async def mock_fetchrow(query, *args):
            if "INSERT INTO alert_events" in query:
                return {
                    "id": event_id,
                    "workspace_id": workspace_id,
                    "status": "active",
                    "activated_at": datetime.now(timezone.utc),
                    "last_seen": datetime.now(timezone.utc),
                }
            return None

        conn.fetchrow = AsyncMock(side_effect=mock_fetchrow)

        # Create buckets with confidence dropping significantly
        # First half avg: 0.80, Second half avg: 0.70 -> delta = -0.10 (exceeds -0.05)
        mock_buckets = [
            MockBucket(drift_score=0.10, avg_confidence=0.82),
            MockBucket(drift_score=0.12, avg_confidence=0.78),
            MockBucket(drift_score=0.15, avg_confidence=0.72),
            MockBucket(drift_score=0.18, avg_confidence=0.68),
        ]

        from app.services.alerts.job import AlertEvaluatorJob

        job = AlertEvaluatorJob(pool)

        with patch.object(job, "_fetch_buckets", AsyncMock(return_value=mock_buckets)):
            result = await job.run(workspace_id=workspace_id)

        assert result["status"] == "completed"
        assert result["metrics"]["alerts_activated"] == 1

    @pytest.mark.asyncio
    async def test_insufficient_data_skips_evaluation(self, mock_db_pool):
        """
        Test that insufficient bucket data is properly tracked.
        """
        pool, conn = mock_db_pool
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()

        async def mock_fetchval(query, *args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            return uuid4()

        conn.fetchval = AsyncMock(side_effect=mock_fetchval)

        # Rule requires 2 consecutive buckets
        mock_rule = {
            "id": rule_id,
            "workspace_id": workspace_id,
            "rule_type": "drift_spike",
            "strategy_entity_id": strategy_id,
            "regime_key": "btc_high_vol",
            "timeframe": "1h",
            "enabled": True,
            "config": {"drift_threshold": 0.30, "consecutive_buckets": 2},
            "cooldown_minutes": 60,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        conn.fetch = AsyncMock(return_value=[mock_rule])

        # Only 1 bucket - insufficient data
        mock_buckets = [MockBucket(drift_score=0.35, avg_confidence=0.80)]

        from app.services.alerts.job import AlertEvaluatorJob

        job = AlertEvaluatorJob(pool)

        with patch.object(job, "_fetch_buckets", AsyncMock(return_value=mock_buckets)):
            result = await job.run(workspace_id=workspace_id)

        assert result["status"] == "completed"
        assert result["metrics"]["tuples_skipped_insufficient_data"] == 1
        assert result["metrics"]["alerts_activated"] == 0


# =============================================================================
# Alert Lifecycle Tests: Active -> Acknowledge -> Resolve
# =============================================================================


class TestAlertLifecycle:
    """Tests for alert lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_alert_lifecycle_active_ack_resolve(self, mock_db_pool):
        """
        Test full alert lifecycle:
        1. Create active alert
        2. Acknowledge it
        3. Resolve it
        """
        pool, conn = mock_db_pool
        event_id = uuid4()

        from app.repositories.alerts import AlertsRepository

        repo = AlertsRepository(pool)

        # Step 1: Acknowledge active alert
        conn.execute = AsyncMock(return_value="UPDATE 1")

        acknowledged = await repo.acknowledge(event_id, acknowledged_by="admin@test.com")
        assert acknowledged is True

        # Verify ack query was called
        conn.execute.assert_called()
        call_args = str(conn.execute.call_args)
        assert "acknowledged = true" in call_args
        assert "acknowledged_by" in call_args

        # Step 2: Unacknowledge
        conn.execute = AsyncMock(return_value="UPDATE 1")
        unacked = await repo.unacknowledge(event_id)
        assert unacked is True

        # Verify unack query
        call_args = str(conn.execute.call_args)
        assert "acknowledged = false" in call_args

        # Step 3: Resolve
        conn.execute = AsyncMock(return_value="UPDATE 1")
        resolved = await repo.resolve(event_id)
        assert resolved is True

        # Verify resolve query
        call_args = str(conn.execute.call_args)
        assert "resolved" in call_args

    @pytest.mark.asyncio
    async def test_alert_reactivation_after_resolve(self, mock_db_pool):
        """
        Test that resolved alert can be reactivated when condition recurs.
        """
        pool, conn = mock_db_pool
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()
        event_id = uuid4()

        async def mock_fetchval(query, *args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            return uuid4()

        conn.fetchval = AsyncMock(side_effect=mock_fetchval)

        # Setup resolved event that can be reactivated (old enough)
        resolved_at = datetime.now(timezone.utc) - timedelta(hours=2)

        mock_rule = {
            "id": rule_id,
            "workspace_id": workspace_id,
            "rule_type": "drift_spike",
            "strategy_entity_id": strategy_id,
            "regime_key": "btc_high_vol",
            "timeframe": "1h",
            "enabled": True,
            "config": {"drift_threshold": 0.30, "consecutive_buckets": 2},
            "cooldown_minutes": 60,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        conn.fetch = AsyncMock(return_value=[mock_rule])

        # Existing resolved event (old activation, should not block reactivation)
        existing_event = {
            "id": event_id,
            "status": "resolved",
            "activated_at": datetime.now(timezone.utc) - timedelta(hours=3),
            "last_seen": resolved_at,
        }

        call_count = 0

        async def mock_fetchrow(query, *args):
            nonlocal call_count
            call_count += 1
            if "SELECT id, status, activated_at" in query:
                return existing_event
            if "INSERT INTO alert_events" in query:
                return {
                    "id": event_id,
                    "workspace_id": workspace_id,
                    "status": "active",
                    "activated_at": datetime.now(timezone.utc),
                    "last_seen": datetime.now(timezone.utc),
                }
            return None

        conn.fetchrow = AsyncMock(side_effect=mock_fetchrow)

        mock_buckets = [
            MockBucket(drift_score=0.35, avg_confidence=0.80),
            MockBucket(drift_score=0.38, avg_confidence=0.78),
        ]

        from app.services.alerts.job import AlertEvaluatorJob

        job = AlertEvaluatorJob(pool)

        with patch.object(job, "_fetch_buckets", AsyncMock(return_value=mock_buckets)):
            result = await job.run(workspace_id=workspace_id)

        # Should reactivate the alert
        assert result["status"] == "completed"
        assert result["metrics"]["alerts_activated"] == 1

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_reactivation(self, mock_db_pool):
        """
        Test that cooldown period suppresses alert reactivation.
        """
        pool, conn = mock_db_pool
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()
        event_id = uuid4()

        async def mock_fetchval(query, *args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            return uuid4()

        conn.fetchval = AsyncMock(side_effect=mock_fetchval)

        # Rule with 60 minute cooldown
        mock_rule = {
            "id": rule_id,
            "workspace_id": workspace_id,
            "rule_type": "drift_spike",
            "strategy_entity_id": strategy_id,
            "regime_key": "btc_high_vol",
            "timeframe": "1h",
            "enabled": True,
            "config": {"drift_threshold": 0.30, "consecutive_buckets": 2},
            "cooldown_minutes": 60,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        conn.fetch = AsyncMock(return_value=[mock_rule])

        # Existing resolved event that was activated recently (within cooldown)
        existing_event = {
            "id": event_id,
            "status": "resolved",
            "activated_at": datetime.now(timezone.utc) - timedelta(minutes=30),
            "last_seen": datetime.now(timezone.utc) - timedelta(minutes=15),
        }

        async def mock_fetchrow(query, *args):
            if "SELECT id, status, activated_at" in query:
                return existing_event
            return None

        conn.fetchrow = AsyncMock(side_effect=mock_fetchrow)

        mock_buckets = [
            MockBucket(drift_score=0.35, avg_confidence=0.80),
            MockBucket(drift_score=0.38, avg_confidence=0.78),
        ]

        from app.services.alerts.job import AlertEvaluatorJob

        job = AlertEvaluatorJob(pool)

        with patch.object(job, "_fetch_buckets", AsyncMock(return_value=mock_buckets)):
            result = await job.run(workspace_id=workspace_id)

        assert result["status"] == "completed"
        assert result["metrics"]["activations_suppressed_cooldown"] == 1
        assert result["metrics"]["alerts_activated"] == 0


# =============================================================================
# API Endpoints Integration Tests
# =============================================================================


class TestAlertRulesApiIntegration:
    """Integration tests for alert rules CRUD operations."""

    def test_create_list_update_delete_rule_flow(self, client, admin_headers):
        """
        Test full CRUD flow for alert rules:
        1. Create rule
        2. List rules
        3. Update rule
        4. Delete rule
        """
        workspace_id = uuid4()
        rule_id = uuid4()
        strategy_id = uuid4()

        mock_repo = MagicMock()

        # Step 1: Create rule
        created_rule = {
            "id": rule_id,
            "workspace_id": workspace_id,
            "rule_type": "drift_spike",
            "strategy_entity_id": strategy_id,
            "regime_key": "btc_high_vol",
            "timeframe": "1h",
            "enabled": True,
            "config": {"drift_threshold": 0.30},
            "cooldown_minutes": 60,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        mock_repo.create_rule = AsyncMock(return_value=created_rule)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/rules?workspace_id={workspace_id}",
                headers=admin_headers,
                json={
                    "rule_type": "drift_spike",
                    "strategy_entity_id": str(strategy_id),
                    "regime_key": "btc_high_vol",
                    "timeframe": "1h",
                    "config": {"drift_threshold": 0.30},
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["rule_type"] == "drift_spike"
        assert data["enabled"] is True

        # Step 2: List rules
        mock_repo.list_rules = AsyncMock(return_value=[created_rule])

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/rules?workspace_id={workspace_id}",
                headers=admin_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["rules"][0]["id"] == str(rule_id)

        # Step 3: Update rule (disable it)
        updated_rule = {**created_rule, "enabled": False}
        mock_repo.update_rule = AsyncMock(return_value=updated_rule)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.patch(
                f"/admin/alerts/rules/{rule_id}",
                headers=admin_headers,
                json={"enabled": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False

        # Step 4: Delete rule
        mock_repo.delete_rule = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.delete(
                f"/admin/alerts/rules/{rule_id}",
                headers=admin_headers,
            )

        assert response.status_code == 204


class TestAlertEventsApiIntegration:
    """Integration tests for alert events operations."""

    def test_list_events_with_all_filters(self, client, admin_headers):
        """
        Test listing events with various filter combinations.
        """
        workspace_id = uuid4()
        event_id = uuid4()
        strategy_id = uuid4()

        mock_event = {
            "id": event_id,
            "workspace_id": workspace_id,
            "rule_id": uuid4(),
            "strategy_entity_id": strategy_id,
            "regime_key": "btc_high_vol",
            "timeframe": "1h",
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
            "context_json": {"threshold": 0.30},
            "fingerprint": "v1:btc_high_vol:1h",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        mock_repo = MagicMock()
        mock_repo.list_events = AsyncMock(return_value=([mock_event], 1))

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts?workspace_id={workspace_id}"
                "&status=active"
                "&severity=high"
                "&acknowledged=false"
                "&rule_type=drift_spike"
                f"&strategy_entity_id={strategy_id}"
                "&timeframe=1h"
                "&regime_key=btc_high_vol"
                "&limit=25&offset=0",
                headers=admin_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["status"] == "active"
        assert data["items"][0]["severity"] == "high"

    def test_acknowledge_and_unacknowledge_flow(self, client, admin_headers):
        """
        Test acknowledge/unacknowledge event flow.
        """
        event_id = uuid4()

        mock_repo = MagicMock()

        # Acknowledge
        mock_repo.acknowledge = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/acknowledge",
                headers=admin_headers,
                json={"acknowledged_by": "operator@test.com"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is True

        # Verify acknowledge was called with correct args
        mock_repo.acknowledge.assert_called_once_with(
            event_id, acknowledged_by="operator@test.com"
        )

        # Unacknowledge
        mock_repo.unacknowledge = AsyncMock(return_value=True)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/unacknowledge",
                headers=admin_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["acknowledged"] is False

    def test_get_event_detail(self, client, admin_headers):
        """
        Test fetching single event details.
        """
        event_id = uuid4()
        workspace_id = uuid4()
        strategy_id = uuid4()

        mock_event = {
            "id": event_id,
            "workspace_id": workspace_id,
            "rule_id": uuid4(),
            "strategy_entity_id": strategy_id,
            "regime_key": "btc_high_vol",
            "timeframe": "1h",
            "rule_type": "combo",
            "status": "active",
            "severity": "high",
            "acknowledged": True,
            "acknowledged_at": datetime.now(timezone.utc),
            "acknowledged_by": "admin@test.com",
            "first_seen": datetime.now(timezone.utc),
            "activated_at": datetime.now(timezone.utc),
            "last_seen": datetime.now(timezone.utc),
            "resolved_at": None,
            "context_json": {
                "drift": {"threshold": 0.30, "current_drift": 0.35},
                "confidence": {"trend_delta": -0.08},
            },
            "fingerprint": "v1:btc_high_vol:1h",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        mock_repo = MagicMock()
        mock_repo.get_event = AsyncMock(return_value=mock_event)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/{event_id}",
                headers=admin_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rule_type"] == "combo"
        assert data["severity"] == "high"
        assert data["acknowledged"] is True
        assert "drift" in data["context_json"]
        assert "confidence" in data["context_json"]


class TestEvaluateAlertsJobEndpoint:
    """Integration tests for the evaluate alerts job endpoint."""

    def test_evaluate_alerts_job_success(self, client, admin_headers):
        """
        Test successful execution of evaluate alerts job.
        """
        workspace_id = uuid4()

        mock_job = MagicMock()
        mock_job.run = AsyncMock(
            return_value={
                "lock_acquired": True,
                "status": "completed",
                "metrics": {
                    "rules_loaded": 3,
                    "tuples_evaluated": 3,
                    "tuples_skipped_insufficient_data": 1,
                    "activations_suppressed_cooldown": 0,
                    "alerts_activated": 2,
                    "alerts_resolved": 0,
                    "db_upserts": 2,
                    "db_updates": 0,
                    "evaluation_errors": 0,
                },
            }
        )

        with patch("app.admin.router._db_pool", MagicMock()), patch(
            "app.services.alerts.job.AlertEvaluatorJob", return_value=mock_job
        ):
            response = client.post(
                f"/admin/jobs/evaluate-alerts?workspace_id={workspace_id}",
                headers=admin_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["metrics"]["rules_loaded"] == 3
        assert data["metrics"]["alerts_activated"] == 2

    def test_evaluate_alerts_dry_run(self, client, admin_headers):
        """
        Test dry run mode doesn't create alerts.
        """
        workspace_id = uuid4()

        mock_job = MagicMock()
        mock_job.run = AsyncMock(
            return_value={
                "lock_acquired": True,
                "status": "completed",
                "metrics": {
                    "rules_loaded": 3,
                    "tuples_evaluated": 3,
                    "tuples_skipped_insufficient_data": 0,
                    "activations_suppressed_cooldown": 0,
                    "alerts_activated": 0,
                    "alerts_resolved": 0,
                    "db_upserts": 0,
                    "db_updates": 0,
                    "evaluation_errors": 0,
                },
            }
        )

        with patch("app.admin.router._db_pool", MagicMock()), patch(
            "app.services.alerts.job.AlertEvaluatorJob", return_value=mock_job
        ):
            response = client.post(
                f"/admin/jobs/evaluate-alerts?workspace_id={workspace_id}&dry_run=true",
                headers=admin_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["dry_run"] is True
        assert data["metrics"]["db_upserts"] == 0

        # Verify dry_run was passed to job
        mock_job.run.assert_called_once()
        call_kwargs = mock_job.run.call_args[1]
        assert call_kwargs["dry_run"] is True

    def test_evaluate_alerts_lock_conflict(self, client, admin_headers):
        """
        Test that concurrent job execution returns 409.
        """
        workspace_id = uuid4()

        mock_job = MagicMock()
        mock_job.run = AsyncMock(
            return_value={
                "lock_acquired": False,
                "status": "already_running",
                "metrics": {
                    "rules_loaded": 0,
                    "tuples_evaluated": 0,
                    "tuples_skipped_insufficient_data": 0,
                    "activations_suppressed_cooldown": 0,
                    "alerts_activated": 0,
                    "alerts_resolved": 0,
                    "db_upserts": 0,
                    "db_updates": 0,
                    "evaluation_errors": 0,
                },
            }
        )

        with patch("app.admin.router._db_pool", MagicMock()), patch(
            "app.services.alerts.job.AlertEvaluatorJob", return_value=mock_job
        ):
            response = client.post(
                f"/admin/jobs/evaluate-alerts?workspace_id={workspace_id}",
                headers=admin_headers,
            )

        assert response.status_code == 409
        data = response.json()
        assert data["status"] == "already_running"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestAlertEdgeCases:
    """Tests for edge cases and error handling."""

    def test_rule_not_found_returns_404(self, client, admin_headers):
        """Test that missing rule returns 404."""
        rule_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_rule = AsyncMock(return_value=None)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/rules/{rule_id}",
                headers=admin_headers,
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_event_not_found_returns_404(self, client, admin_headers):
        """Test that missing event returns 404."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.get_event = AsyncMock(return_value=None)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.get(
                f"/admin/alerts/{event_id}",
                headers=admin_headers,
            )

        assert response.status_code == 404

    def test_invalid_rule_type_returns_422(self, client, admin_headers):
        """Test that invalid rule type returns validation error."""
        workspace_id = uuid4()

        response = client.post(
            f"/admin/alerts/rules?workspace_id={workspace_id}",
            headers=admin_headers,
            json={
                "rule_type": "invalid_type",
                "config": {},
            },
        )

        assert response.status_code == 422

    def test_acknowledge_already_acknowledged_returns_404(self, client, admin_headers):
        """Test that acknowledging already-acknowledged event returns 404."""
        event_id = uuid4()

        mock_repo = MagicMock()
        mock_repo.acknowledge = AsyncMock(return_value=False)

        with patch("app.admin.alerts._db_pool", MagicMock()), patch(
            "app.repositories.alerts.AlertsRepository", return_value=mock_repo
        ):
            response = client.post(
                f"/admin/alerts/{event_id}/acknowledge",
                headers=admin_headers,
            )

        assert response.status_code == 404

    def test_db_pool_unavailable_returns_503(self, client, admin_headers):
        """Test that missing DB pool returns 503."""
        workspace_id = uuid4()

        with patch("app.admin.router._db_pool", None):
            response = client.post(
                f"/admin/jobs/evaluate-alerts?workspace_id={workspace_id}",
                headers=admin_headers,
            )

        assert response.status_code == 503
        assert "Database" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_job_handles_evaluation_errors(self, mock_db_pool):
        """Test that job continues after individual rule evaluation errors."""
        pool, conn = mock_db_pool
        workspace_id = uuid4()

        async def mock_fetchval(query, *args):
            if "pg_try_advisory_lock" in query:
                return True
            if "pg_advisory_unlock" in query:
                return True
            return uuid4()

        conn.fetchval = AsyncMock(side_effect=mock_fetchval)

        # Two rules - one will cause error
        mock_rules = [
            {
                "id": uuid4(),
                "workspace_id": workspace_id,
                "rule_type": "drift_spike",
                "strategy_entity_id": uuid4(),
                "regime_key": "rule1",
                "timeframe": "1h",
                "enabled": True,
                "config": {"drift_threshold": 0.30, "consecutive_buckets": 2},
                "cooldown_minutes": 60,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
            {
                "id": uuid4(),
                "workspace_id": workspace_id,
                "rule_type": "drift_spike",
                "strategy_entity_id": uuid4(),
                "regime_key": "rule2",
                "timeframe": "1h",
                "enabled": True,
                "config": {"drift_threshold": 0.30, "consecutive_buckets": 2},
                "cooldown_minutes": 60,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
        ]
        conn.fetch = AsyncMock(return_value=mock_rules)

        # Track how many times _fetch_buckets is called
        fetch_count = 0

        async def mock_fetch_buckets(workspace_id, strategy_entity_id, regime_key, timeframe):
            nonlocal fetch_count
            fetch_count += 1
            if fetch_count == 1:
                raise ValueError("Simulated bucket fetch error")
            # Second call succeeds but returns insufficient data
            return [MockBucket(drift_score=0.10, avg_confidence=0.80)]

        from app.services.alerts.job import AlertEvaluatorJob

        job = AlertEvaluatorJob(pool)
        # Patch the method on the instance (bound method)
        job._fetch_buckets = mock_fetch_buckets

        result = await job.run(workspace_id=workspace_id)

        # Job should complete despite error in first rule
        assert result["status"] == "completed"
        # Error should be recorded
        assert result["metrics"]["evaluation_errors"] == 1
        # Both rules were attempted (2 rules loaded)
        assert result["metrics"]["rules_loaded"] == 2
        # Second rule should be skipped due to insufficient data (only 1 bucket)
        assert result["metrics"]["tuples_skipped_insufficient_data"] == 1
        # Both fetch attempts happened
        assert fetch_count == 2
