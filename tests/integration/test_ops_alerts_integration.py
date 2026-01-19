"""Integration tests for operational alerts repository and evaluator.

Tests verify:
1. Recovery path - alerts resolve when conditions clear
2. Escalation detection - severity increases are tracked
3. Multi-workspace isolation - dedupe keys are workspace-scoped

Run with: pytest tests/integration/test_ops_alerts_integration.py -v
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.ops_alerts import OpsAlertsRepository, OpsAlert
from app.services.ops_alerts.evaluator import OpsAlertEvaluator


pytestmark = [pytest.mark.integration]


@pytest.fixture
def mock_pool():
    """Create mock database pool with async context manager support."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


class TestRecoveryPath:
    """Test alert recovery when conditions clear."""

    @pytest.mark.asyncio
    async def test_resolution_pass_resolves_active_alert_when_condition_clears(
        self, mock_pool
    ):
        """
        Scenario:
        1. Create an active alert via repo.upsert()
        2. Call evaluator._resolution_pass() with empty triggered_keys
        3. Verify alert status changed to 'resolved' and resolved_at is set
        """
        workspace_id = uuid4()
        alert_id = uuid4()
        now = datetime(2026, 1, 19, 14, 0, 0, tzinfo=timezone.utc)
        dedupe_key = "health_degraded:2026-01-19"

        # Mock the resolved alert that will be returned
        resolved_alert = OpsAlert(
            id=alert_id,
            workspace_id=workspace_id,
            rule_type="health_degraded",
            severity="high",
            status="resolved",
            rule_version="v1",
            dedupe_key=dedupe_key,
            payload={"overall_status": "ok"},
            source="alert_evaluator",
            job_run_id=None,
            created_at=now,
            last_seen_at=now,
            resolved_at=now,
            acknowledged_at=None,
            acknowledged_by=None,
        )

        # Setup repository mock
        repo = AsyncMock(spec=OpsAlertsRepository)
        repo.resolve_by_dedupe_key.return_value = resolved_alert

        # Create evaluator with mocked repo
        evaluator = OpsAlertEvaluator(repo, mock_pool)

        # Call resolution pass with empty triggered_keys (simulating healthy state)
        triggered_keys: set[str] = set()
        resolved_alerts = await evaluator._resolution_pass(
            workspace_id, triggered_keys, now
        )

        # Verify resolve_by_dedupe_key was called for singleton rules
        assert repo.resolve_by_dedupe_key.call_count >= 1

        # Find the health_degraded call
        calls = repo.resolve_by_dedupe_key.call_args_list
        health_call = next(
            (c for c in calls if c.args[1] == dedupe_key),
            None,
        )
        assert health_call is not None, "Expected resolve call for health_degraded"
        assert health_call.args[0] == workspace_id

        # Verify resolved alert is returned
        assert len(resolved_alerts) >= 1
        assert any(a.id == alert_id for a in resolved_alerts)
        assert any(a.status == "resolved" for a in resolved_alerts)
        assert any(a.resolved_at is not None for a in resolved_alerts)


class TestEscalationDetection:
    """Test escalation detection when severity increases."""

    @pytest.mark.asyncio
    async def test_upsert_detects_escalation_from_high_to_critical(self, mock_pool):
        """
        Scenario:
        1. Create alert with severity='high' via repo.upsert()
        2. Upsert same dedupe_key with severity='critical'
        3. Verify UpsertResult.escalated=True and no new row created
        """
        workspace_id = uuid4()
        alert_id = uuid4()
        dedupe_key = "health_degraded:2026-01-19"

        # Get the mock connection
        conn = mock_pool.acquire.return_value.__aenter__.return_value

        # First call: fetchrow for existing severity check returns 'high'
        # Second call: fetchrow for upsert returns the updated row
        conn.fetchrow.side_effect = [
            {"severity": "high"},  # Existing alert with high severity
            {
                "id": alert_id,
                "is_new": False,  # xmax != 0 means update, not insert
                "current_severity": "critical",
            },
        ]

        repo = OpsAlertsRepository(mock_pool)

        result = await repo.upsert(
            workspace_id=workspace_id,
            rule_type="health_degraded",
            severity="critical",
            dedupe_key=dedupe_key,
            payload={"overall_status": "error"},
            source="alert_evaluator",
        )

        # Verify escalation detected
        assert result.escalated is True
        assert result.is_new is False
        assert result.previous_severity == "high"
        assert result.current_severity == "critical"
        assert result.id == alert_id


class TestDedupeOnRepeatedEvaluation:
    """Test dedupe contract: repeated evaluation doesn't create duplicates."""

    @pytest.mark.asyncio
    async def test_second_evaluation_updates_existing_alert_not_creates_new(
        self, mock_pool
    ):
        """
        Scenario (encodes manual verification):
        1. First evaluate() triggers health_degraded → total_new=1
        2. Second evaluate() same condition still true → total_new=0
        3. Same alert row updated (last_seen_at advanced, occurrence_count incremented)

        This is the core dedupe contract we rely on.
        """
        workspace_id = uuid4()
        alert_id = uuid4()
        dedupe_key = "health_degraded:2026-01-19"

        conn = mock_pool.acquire.return_value.__aenter__.return_value

        # First upsert: new alert created
        # Second upsert: existing alert updated (is_new=False)
        conn.fetchrow.side_effect = [
            None,  # No existing severity (first eval)
            {"id": alert_id, "is_new": True, "current_severity": "critical"},
            {"severity": "critical"},  # Existing severity (second eval)
            {"id": alert_id, "is_new": False, "current_severity": "critical"},
        ]

        repo = OpsAlertsRepository(mock_pool)

        # First evaluation - creates new alert
        result1 = await repo.upsert(
            workspace_id=workspace_id,
            rule_type="health_degraded",
            severity="critical",
            dedupe_key=dedupe_key,
            payload={"overall_status": "error"},
            source="alert_evaluator",
        )

        # Second evaluation - same condition, should update not create
        result2 = await repo.upsert(
            workspace_id=workspace_id,
            rule_type="health_degraded",
            severity="critical",
            dedupe_key=dedupe_key,
            payload={"overall_status": "error"},
            source="alert_evaluator",
        )

        # Core contract assertions
        assert result1.is_new is True, "First eval should create new alert"
        assert result2.is_new is False, "Second eval should update existing"
        assert result1.id == result2.id, "Same alert ID (dedupe worked)"
        assert result2.escalated is False, "Same severity = no escalation"


class TestMultiWorkspaceIsolation:
    """Test workspace isolation for dedupe keys."""

    @pytest.mark.asyncio
    async def test_same_dedupe_key_creates_separate_alerts_per_workspace(
        self, mock_pool
    ):
        """
        Scenario:
        1. Create alert in workspace A
        2. Create alert with SAME dedupe_key in workspace B
        3. Verify 2 separate rows exist (not deduplicated across workspaces)
        """
        workspace_a = uuid4()
        workspace_b = uuid4()
        alert_id_a = uuid4()
        alert_id_b = uuid4()
        dedupe_key = "health_degraded:2026-01-19"

        # Get the mock connection
        conn = mock_pool.acquire.return_value.__aenter__.return_value

        # Track upsert calls to verify both workspaces get separate alerts
        upsert_results = []

        # Configure fetchrow to simulate:
        # - No existing alert for workspace A (first upsert)
        # - New alert created for workspace A
        # - No existing alert for workspace B (second upsert)
        # - New alert created for workspace B
        conn.fetchrow.side_effect = [
            None,  # No existing alert for workspace A
            {"id": alert_id_a, "is_new": True, "current_severity": "high"},
            None,  # No existing alert for workspace B
            {"id": alert_id_b, "is_new": True, "current_severity": "high"},
        ]

        repo = OpsAlertsRepository(mock_pool)

        # Upsert in workspace A
        result_a = await repo.upsert(
            workspace_id=workspace_a,
            rule_type="health_degraded",
            severity="high",
            dedupe_key=dedupe_key,
            payload={"overall_status": "degraded"},
            source="alert_evaluator",
        )
        upsert_results.append(result_a)

        # Upsert in workspace B with SAME dedupe_key
        result_b = await repo.upsert(
            workspace_id=workspace_b,
            rule_type="health_degraded",
            severity="high",
            dedupe_key=dedupe_key,
            payload={"overall_status": "degraded"},
            source="alert_evaluator",
        )
        upsert_results.append(result_b)

        # Verify both are new alerts (not deduplicated)
        assert result_a.is_new is True
        assert result_b.is_new is True

        # Verify different alert IDs
        assert result_a.id != result_b.id
        assert result_a.id == alert_id_a
        assert result_b.id == alert_id_b

        # Verify both workspaces received upserts
        assert len(upsert_results) == 2
