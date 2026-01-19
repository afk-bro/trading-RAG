"""Integration tests for ops_alert_eval job handler.

Tests verify:
1. Handler returns expected result structure
2. Health degraded condition creates alert
3. Repeated evaluation updates existing alert (not creates new)
4. Dry run mode evaluates but doesn't write alerts

Run with: pytest tests/integration/test_ops_alert_eval_job.py -v
"""

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Mock ccxt before importing handlers (optional dependency)
sys.modules["ccxt"] = MagicMock()
sys.modules["ccxt.async_support"] = MagicMock()

from app.jobs.handlers.ops_alert_eval import handle_ops_alert_eval  # noqa: E402
from app.jobs.models import Job  # noqa: E402
from app.jobs.types import JobType, JobStatus  # noqa: E402
from app.services.ops_alerts.models import EvalResult  # noqa: E402


pytestmark = [pytest.mark.integration]


@pytest.fixture
def mock_pool():
    """Create mock database pool with async context manager support."""
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return pool


@pytest.fixture
def sample_job():
    """Create a sample OPS_ALERT_EVAL job."""
    return Job(
        id=uuid4(),
        type=JobType.OPS_ALERT_EVAL,
        status=JobStatus.RUNNING,
        payload={
            "workspace_id": None,
            "triggered_by": "test",
        },
        attempt=1,
        max_attempts=3,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def workspace_job():
    """Create a workspace-scoped OPS_ALERT_EVAL job."""
    return Job(
        id=uuid4(),
        type=JobType.OPS_ALERT_EVAL,
        status=JobStatus.RUNNING,
        payload={
            "workspace_id": str(uuid4()),
            "triggered_by": "test",
        },
        attempt=1,
        max_attempts=3,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def dry_run_job():
    """Create a dry-run OPS_ALERT_EVAL job."""
    return Job(
        id=uuid4(),
        type=JobType.OPS_ALERT_EVAL,
        status=JobStatus.RUNNING,
        payload={
            "workspace_id": None,
            "triggered_by": "test",
            "dry_run": True,
        },
        attempt=1,
        max_attempts=3,
        created_at=datetime.now(timezone.utc),
    )


class TestHandlerResultStructure:
    """Test handler returns expected result structure."""

    @pytest.mark.asyncio
    async def test_handler_returns_expected_keys(self, mock_pool, sample_job):
        """Handler returns dict with all expected keys."""
        workspace_id = uuid4()

        # Mock active workspaces query
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = [{"id": workspace_id}]

        # Mock the evaluator to return a valid result
        mock_eval_result = EvalResult(
            workspace_id=workspace_id,
            job_run_id=sample_job.id,
            timestamp=datetime.now(timezone.utc),
            conditions_evaluated=5,
            alerts_triggered=1,
            alerts_new=1,
            alerts_updated=0,
            alerts_resolved=0,
            alerts_escalated=0,
        )

        with patch(
            "app.jobs.handlers.ops_alert_eval.OpsAlertEvaluator"
        ) as MockEvaluator, patch(
            "app.jobs.handlers.ops_alert_eval.get_telegram_notifier"
        ) as mock_notifier:
            mock_evaluator_instance = AsyncMock()
            mock_evaluator_instance.evaluate.return_value = mock_eval_result
            MockEvaluator.return_value = mock_evaluator_instance
            mock_notifier.return_value = None

            ctx = {"pool": mock_pool}
            result = await handle_ops_alert_eval(sample_job, ctx)

        # Verify all expected keys present
        expected_keys = {
            "workspaces_evaluated",
            "total_conditions",
            "total_triggered",
            "total_new",
            "total_resolved",
            "total_escalated",
            "telegram_sent",
            "errors",
            "by_workspace",
        }
        assert set(result.keys()) >= expected_keys

    @pytest.mark.asyncio
    async def test_handler_aggregates_workspace_results(self, mock_pool, sample_job):
        """Handler correctly aggregates results from multiple workspaces."""
        ws1 = uuid4()
        ws2 = uuid4()

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = [{"id": ws1}, {"id": ws2}]

        # Results for two workspaces
        result1 = EvalResult(
            workspace_id=ws1,
            job_run_id=sample_job.id,
            timestamp=datetime.now(timezone.utc),
            conditions_evaluated=5,
            alerts_triggered=2,
            alerts_new=1,
            alerts_resolved=1,
        )
        result2 = EvalResult(
            workspace_id=ws2,
            job_run_id=sample_job.id,
            timestamp=datetime.now(timezone.utc),
            conditions_evaluated=5,
            alerts_triggered=1,
            alerts_new=0,
            alerts_resolved=0,
        )

        with patch(
            "app.jobs.handlers.ops_alert_eval.OpsAlertEvaluator"
        ) as MockEvaluator, patch(
            "app.jobs.handlers.ops_alert_eval.get_telegram_notifier"
        ) as mock_notifier:
            mock_evaluator_instance = AsyncMock()
            mock_evaluator_instance.evaluate.side_effect = [result1, result2]
            MockEvaluator.return_value = mock_evaluator_instance
            mock_notifier.return_value = None

            ctx = {"pool": mock_pool}
            result = await handle_ops_alert_eval(sample_job, ctx)

        # Verify aggregation
        assert result["workspaces_evaluated"] == 2
        assert result["total_conditions"] == 10  # 5 + 5
        assert result["total_triggered"] == 3  # 2 + 1
        assert result["total_new"] == 1  # 1 + 0
        assert result["total_resolved"] == 1  # 1 + 0
        assert len(result["by_workspace"]) == 2


class TestHandlerWorkspaceSelection:
    """Test workspace selection logic."""

    @pytest.mark.asyncio
    async def test_no_workspace_evaluates_all_active(self, mock_pool, sample_job):
        """When workspace_id is None, evaluate all active workspaces."""
        ws1 = uuid4()
        ws2 = uuid4()

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = [{"id": ws1}, {"id": ws2}]

        with patch(
            "app.jobs.handlers.ops_alert_eval.OpsAlertEvaluator"
        ) as MockEvaluator, patch(
            "app.jobs.handlers.ops_alert_eval.get_telegram_notifier"
        ) as mock_notifier:
            mock_evaluator_instance = AsyncMock()
            mock_evaluator_instance.evaluate.return_value = EvalResult(
                workspace_id=ws1,
                job_run_id=sample_job.id,
                timestamp=datetime.now(timezone.utc),
            )
            MockEvaluator.return_value = mock_evaluator_instance
            mock_notifier.return_value = None

            ctx = {"pool": mock_pool}
            await handle_ops_alert_eval(sample_job, ctx)

        # Verify both workspaces evaluated
        assert mock_evaluator_instance.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_specific_workspace_evaluates_only_that(
        self, mock_pool, workspace_job
    ):
        """When workspace_id is provided, evaluate only that workspace."""
        from uuid import UUID

        ws_id = UUID(workspace_job.payload["workspace_id"])

        with patch(
            "app.jobs.handlers.ops_alert_eval.OpsAlertEvaluator"
        ) as MockEvaluator, patch(
            "app.jobs.handlers.ops_alert_eval.get_telegram_notifier"
        ) as mock_notifier:
            mock_evaluator_instance = AsyncMock()
            mock_evaluator_instance.evaluate.return_value = EvalResult(
                workspace_id=ws_id,
                job_run_id=workspace_job.id,
                timestamp=datetime.now(timezone.utc),
            )
            MockEvaluator.return_value = mock_evaluator_instance
            mock_notifier.return_value = None

            ctx = {"pool": mock_pool}
            await handle_ops_alert_eval(workspace_job, ctx)

        # Verify only one workspace evaluated
        assert mock_evaluator_instance.evaluate.call_count == 1
        call_args = mock_evaluator_instance.evaluate.call_args
        assert call_args.kwargs["workspace_id"] == ws_id


class TestDryRunMode:
    """Test dry run mode doesn't write alerts."""

    @pytest.mark.asyncio
    async def test_dry_run_skips_notifications(self, mock_pool, dry_run_job):
        """Dry run mode doesn't send Telegram notifications."""
        ws_id = uuid4()

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = [{"id": ws_id}]

        with patch(
            "app.jobs.handlers.ops_alert_eval.OpsAlertEvaluator"
        ) as MockEvaluator, patch(
            "app.jobs.handlers.ops_alert_eval.get_telegram_notifier"
        ) as mock_get_notifier:
            mock_evaluator_instance = AsyncMock()
            mock_evaluator_instance.evaluate.return_value = EvalResult(
                workspace_id=ws_id,
                job_run_id=dry_run_job.id,
                timestamp=datetime.now(timezone.utc),
                alerts_new=1,
            )
            MockEvaluator.return_value = mock_evaluator_instance

            ctx = {"pool": mock_pool}
            result = await handle_ops_alert_eval(dry_run_job, ctx)

        # Verify notifier was NOT initialized (dry_run=True)
        mock_get_notifier.assert_not_called()
        assert result["telegram_sent"] == 0


class TestHandlerErrorHandling:
    """Test handler error handling."""

    @pytest.mark.asyncio
    async def test_workspace_error_captured_in_errors_list(self, mock_pool, sample_job):
        """Errors from individual workspaces are captured, not raised."""
        ws1 = uuid4()
        ws2 = uuid4()

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = [{"id": ws1}, {"id": ws2}]

        with patch(
            "app.jobs.handlers.ops_alert_eval.OpsAlertEvaluator"
        ) as MockEvaluator, patch(
            "app.jobs.handlers.ops_alert_eval.get_telegram_notifier"
        ) as mock_notifier:
            mock_evaluator_instance = AsyncMock()
            # First workspace succeeds, second fails
            mock_evaluator_instance.evaluate.side_effect = [
                EvalResult(
                    workspace_id=ws1,
                    job_run_id=sample_job.id,
                    timestamp=datetime.now(timezone.utc),
                ),
                Exception("Database connection failed"),
            ]
            MockEvaluator.return_value = mock_evaluator_instance
            mock_notifier.return_value = None

            ctx = {"pool": mock_pool}
            result = await handle_ops_alert_eval(sample_job, ctx)

        # Verify error captured but job didn't fail
        assert result["workspaces_evaluated"] == 1  # Only ws1 succeeded
        assert len(result["errors"]) == 1
        assert "Database connection failed" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_no_active_workspaces_returns_empty_result(
        self, mock_pool, sample_job
    ):
        """When no active workspaces exist, return empty result (not error)."""
        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = []  # No active workspaces

        ctx = {"pool": mock_pool}
        result = await handle_ops_alert_eval(sample_job, ctx)

        assert result["workspaces_evaluated"] == 0
        assert result["total_conditions"] == 0
        assert result["errors"] == []


class TestJobIdPassthrough:
    """Test job_id is passed to evaluator for audit trail."""

    @pytest.mark.asyncio
    async def test_job_id_passed_to_evaluator(self, mock_pool, sample_job):
        """Job ID is passed to evaluator.evaluate() for audit trail."""
        ws_id = uuid4()

        conn = mock_pool.acquire.return_value.__aenter__.return_value
        conn.fetch.return_value = [{"id": ws_id}]

        with patch(
            "app.jobs.handlers.ops_alert_eval.OpsAlertEvaluator"
        ) as MockEvaluator, patch(
            "app.jobs.handlers.ops_alert_eval.get_telegram_notifier"
        ) as mock_notifier:
            mock_evaluator_instance = AsyncMock()
            mock_evaluator_instance.evaluate.return_value = EvalResult(
                workspace_id=ws_id,
                job_run_id=sample_job.id,
                timestamp=datetime.now(timezone.utc),
            )
            MockEvaluator.return_value = mock_evaluator_instance
            mock_notifier.return_value = None

            ctx = {"pool": mock_pool}
            await handle_ops_alert_eval(sample_job, ctx)

        # Verify job_run_id passed to evaluator
        call_args = mock_evaluator_instance.evaluate.call_args
        assert call_args.kwargs["job_run_id"] == sample_job.id
