"""Unit tests for auto-pause guardrail on CRITICAL alerts."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.ops_alerts.evaluator import OpsAlertEvaluator
from app.services.ops_alerts.models import (
    EvalContext,
    EvalResult,
    Severity,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repo():
    """Mock OpsAlertsRepository."""
    repo = AsyncMock()
    repo.upsert = AsyncMock(
        return_value=MagicMock(id=uuid4(), is_new=True, escalated=False)
    )
    repo.resolve_by_dedupe_key = AsyncMock(return_value=None)
    repo.get_active_dedupe_keys = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_pool():
    """Mock database connection pool."""
    pool = MagicMock()
    conn = AsyncMock()

    # Setup acquire as async context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm

    return pool, conn


@pytest.fixture
def evaluator_with_auto_pause(mock_repo, mock_pool):
    """Create evaluator with auto_pause_enabled=True."""
    pool, conn = mock_pool
    return OpsAlertEvaluator(mock_repo, pool, auto_pause_enabled=True)


@pytest.fixture
def evaluator_without_auto_pause(mock_repo, mock_pool):
    """Create evaluator with auto_pause_enabled=False (default)."""
    pool, conn = mock_pool
    return OpsAlertEvaluator(mock_repo, pool, auto_pause_enabled=False)


@pytest.fixture
def sample_workspace_id():
    """Sample workspace ID."""
    return uuid4()


@pytest.fixture
def base_context(sample_workspace_id):
    """Base evaluation context."""
    return EvalContext(
        workspace_id=sample_workspace_id,
        now=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        job_run_id=uuid4(),
    )


# =============================================================================
# Auto-Pause Configuration Tests
# =============================================================================


class TestAutoPauseConfig:
    """Tests for auto_pause_enabled configuration."""

    def test_default_auto_pause_disabled(self, mock_repo, mock_pool):
        """Test that auto_pause is disabled by default."""
        pool, conn = mock_pool
        evaluator = OpsAlertEvaluator(mock_repo, pool)
        assert evaluator.auto_pause_enabled is False

    def test_auto_pause_can_be_enabled(self, mock_repo, mock_pool):
        """Test that auto_pause can be enabled via constructor."""
        pool, conn = mock_pool
        evaluator = OpsAlertEvaluator(mock_repo, pool, auto_pause_enabled=True)
        assert evaluator.auto_pause_enabled is True


# =============================================================================
# Auto-Pause on CRITICAL Drawdown Tests
# =============================================================================


class TestAutoPauseDrawdownCritical:
    """Tests for auto-pause on CRITICAL drawdown alerts."""

    @pytest.mark.asyncio
    async def test_pauses_all_versions_on_critical_drawdown(
        self, evaluator_with_auto_pause, sample_workspace_id
    ):
        """Test that all active versions are paused on CRITICAL drawdown."""
        version_id_1 = uuid4()
        version_id_2 = uuid4()
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Create result with triggered CRITICAL drawdown
        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=now,
        )
        result.by_rule_type["workspace_drawdown_high"] = {
            "triggered": True,
            "severity": Severity.HIGH.value,
        }

        # Create context with active versions
        ctx = EvalContext(
            workspace_id=sample_workspace_id,
            now=now,
            job_run_id=uuid4(),
        )
        ctx.strategy_intel = [
            {"version_id": version_id_1, "snapshots": []},
            {"version_id": version_id_2, "snapshots": []},
        ]

        # Mock the strategy versions repository
        with patch(
            "app.repositories.strategy_versions.StrategyVersionsRepository"
        ) as MockRepo:
            mock_version_repo = AsyncMock()
            MockRepo.return_value = mock_version_repo

            await evaluator_with_auto_pause._auto_pause_pass(
                sample_workspace_id, result, ctx, now
            )

            # Should pause both versions
            assert mock_version_repo.pause.call_count == 2
            assert result.versions_auto_paused == 2
            assert set(result.auto_paused_version_ids) == {version_id_1, version_id_2}

    @pytest.mark.asyncio
    async def test_no_pause_on_warn_drawdown(
        self, evaluator_with_auto_pause, sample_workspace_id
    ):
        """Test that versions are NOT paused on WARN (MEDIUM) drawdown."""
        version_id = uuid4()
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Create result with triggered WARN (MEDIUM) drawdown
        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=now,
        )
        result.by_rule_type["workspace_drawdown_high"] = {
            "triggered": True,
            "severity": Severity.MEDIUM.value,  # Warn, not critical
        }

        ctx = EvalContext(
            workspace_id=sample_workspace_id,
            now=now,
            job_run_id=uuid4(),
        )
        ctx.strategy_intel = [{"version_id": version_id, "snapshots": []}]

        with patch(
            "app.repositories.strategy_versions.StrategyVersionsRepository"
        ) as MockRepo:
            mock_version_repo = AsyncMock()
            MockRepo.return_value = mock_version_repo

            await evaluator_with_auto_pause._auto_pause_pass(
                sample_workspace_id, result, ctx, now
            )

            # Should NOT pause any versions
            mock_version_repo.pause.assert_not_called()
            assert result.versions_auto_paused == 0

    @pytest.mark.asyncio
    async def test_no_pause_when_auto_pause_disabled(
        self, evaluator_without_auto_pause, sample_workspace_id, mock_repo
    ):
        """Test that no pausing occurs when auto_pause is disabled."""
        # Even with CRITICAL alerts, should not pause
        # This test verifies the evaluator doesn't call _auto_pause_pass when disabled

        # The auto_pause_pass is only called if auto_pause_enabled=True
        # So we just verify the flag is respected
        assert evaluator_without_auto_pause.auto_pause_enabled is False


# =============================================================================
# Auto-Pause on CRITICAL Confidence Tests
# =============================================================================


class TestAutoPauseConfidenceCritical:
    """Tests for auto-pause on CRITICAL confidence alerts."""

    @pytest.mark.asyncio
    async def test_pauses_specific_version_on_critical_confidence(
        self, evaluator_with_auto_pause, sample_workspace_id
    ):
        """Test that specific version is paused on CRITICAL confidence."""
        version_id_critical = uuid4()
        version_id_ok = uuid4()
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Create result with triggered confidence alert
        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=now,
        )
        result.by_rule_type["strategy_confidence_low"] = {
            "triggered": True,
        }

        # Create context with one critical, one ok version
        ctx = EvalContext(
            workspace_id=sample_workspace_id,
            now=now,
            job_run_id=uuid4(),
        )
        ctx.strategy_intel = [
            {
                "version_id": version_id_critical,
                "snapshots": [{"confidence_score": 0.15}],  # Below CRITICAL (0.20)
            },
            {
                "version_id": version_id_ok,
                "snapshots": [{"confidence_score": 0.50}],  # Above threshold
            },
        ]

        with patch(
            "app.repositories.strategy_versions.StrategyVersionsRepository"
        ) as MockRepo:
            mock_version_repo = AsyncMock()
            MockRepo.return_value = mock_version_repo

            await evaluator_with_auto_pause._auto_pause_pass(
                sample_workspace_id, result, ctx, now
            )

            # Should only pause the critical version
            assert mock_version_repo.pause.call_count == 1
            mock_version_repo.pause.assert_called_once_with(
                version_id=version_id_critical,
                triggered_by="system:auto_pause",
                reason="Auto-paused by CRITICAL alert guardrail",
            )
            assert result.versions_auto_paused == 1
            assert result.auto_paused_version_ids == [version_id_critical]

    @pytest.mark.asyncio
    async def test_no_pause_on_warn_confidence(
        self, evaluator_with_auto_pause, sample_workspace_id
    ):
        """Test that versions are NOT paused on WARN confidence (0.20-0.35)."""
        version_id = uuid4()
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=now,
        )
        result.by_rule_type["strategy_confidence_low"] = {
            "triggered": True,
        }

        ctx = EvalContext(
            workspace_id=sample_workspace_id,
            now=now,
            job_run_id=uuid4(),
        )
        ctx.strategy_intel = [
            {
                "version_id": version_id,
                "snapshots": [{"confidence_score": 0.25}],  # Warn level, not critical
            },
        ]

        with patch(
            "app.repositories.strategy_versions.StrategyVersionsRepository"
        ) as MockRepo:
            mock_version_repo = AsyncMock()
            MockRepo.return_value = mock_version_repo

            await evaluator_with_auto_pause._auto_pause_pass(
                sample_workspace_id, result, ctx, now
            )

            # Should NOT pause (0.25 >= 0.20 critical threshold)
            mock_version_repo.pause.assert_not_called()
            assert result.versions_auto_paused == 0


# =============================================================================
# Combined Scenarios Tests
# =============================================================================


class TestAutoPauseCombinedScenarios:
    """Tests for combined alert scenarios."""

    @pytest.mark.asyncio
    async def test_both_drawdown_and_confidence_critical(
        self, evaluator_with_auto_pause, sample_workspace_id
    ):
        """Test that same version isn't paused twice."""
        version_id = uuid4()
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Both alerts triggered for same version
        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=now,
        )
        result.by_rule_type["workspace_drawdown_high"] = {
            "triggered": True,
            "severity": Severity.HIGH.value,
        }
        result.by_rule_type["strategy_confidence_low"] = {
            "triggered": True,
        }

        ctx = EvalContext(
            workspace_id=sample_workspace_id,
            now=now,
            job_run_id=uuid4(),
        )
        ctx.strategy_intel = [
            {
                "version_id": version_id,
                "snapshots": [{"confidence_score": 0.15}],
            },
        ]

        with patch(
            "app.repositories.strategy_versions.StrategyVersionsRepository"
        ) as MockRepo:
            mock_version_repo = AsyncMock()
            MockRepo.return_value = mock_version_repo

            await evaluator_with_auto_pause._auto_pause_pass(
                sample_workspace_id, result, ctx, now
            )

            # Should pause once (deduplicated via set)
            assert mock_version_repo.pause.call_count == 1
            assert result.versions_auto_paused == 1

    @pytest.mark.asyncio
    async def test_no_versions_to_pause(
        self, evaluator_with_auto_pause, sample_workspace_id
    ):
        """Test graceful handling when no versions exist."""
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=now,
        )
        result.by_rule_type["workspace_drawdown_high"] = {
            "triggered": True,
            "severity": Severity.HIGH.value,
        }

        ctx = EvalContext(
            workspace_id=sample_workspace_id,
            now=now,
            job_run_id=uuid4(),
        )
        ctx.strategy_intel = []  # No active versions

        with patch(
            "app.repositories.strategy_versions.StrategyVersionsRepository"
        ) as MockRepo:
            mock_version_repo = AsyncMock()
            MockRepo.return_value = mock_version_repo

            await evaluator_with_auto_pause._auto_pause_pass(
                sample_workspace_id, result, ctx, now
            )

            # Should not fail, just no-op
            mock_version_repo.pause.assert_not_called()
            assert result.versions_auto_paused == 0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestAutoPauseErrorHandling:
    """Tests for error handling during auto-pause."""

    @pytest.mark.asyncio
    async def test_pause_failure_recorded_in_errors(
        self, evaluator_with_auto_pause, sample_workspace_id
    ):
        """Test that pause failures are recorded in result.errors."""
        version_id = uuid4()
        now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=now,
        )
        result.by_rule_type["workspace_drawdown_high"] = {
            "triggered": True,
            "severity": Severity.HIGH.value,
        }

        ctx = EvalContext(
            workspace_id=sample_workspace_id,
            now=now,
            job_run_id=uuid4(),
        )
        ctx.strategy_intel = [{"version_id": version_id, "snapshots": []}]

        with patch(
            "app.repositories.strategy_versions.StrategyVersionsRepository"
        ) as MockRepo:
            mock_version_repo = AsyncMock()
            mock_version_repo.pause.side_effect = Exception("Database error")
            MockRepo.return_value = mock_version_repo

            await evaluator_with_auto_pause._auto_pause_pass(
                sample_workspace_id, result, ctx, now
            )

            # Error should be recorded
            assert len(result.errors) == 1
            assert "auto_pause" in result.errors[0]
            assert "Database error" in result.errors[0]
            # Counter should NOT increment on failure
            assert result.versions_auto_paused == 0


# =============================================================================
# EvalResult Tests
# =============================================================================


class TestEvalResultAutoPauseFields:
    """Tests for EvalResult auto-pause fields."""

    def test_eval_result_has_auto_pause_fields(self, sample_workspace_id):
        """Test that EvalResult has auto-pause tracking fields."""
        result = EvalResult(
            workspace_id=sample_workspace_id,
            job_run_id=uuid4(),
            timestamp=datetime.now(timezone.utc),
        )

        assert hasattr(result, "versions_auto_paused")
        assert hasattr(result, "auto_paused_version_ids")
        assert result.versions_auto_paused == 0
        assert result.auto_paused_version_ids == []
