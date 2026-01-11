"""Unit tests for RunOrchestrator persistence."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.testing.models import RunPlan, RunVariant
from app.services.testing.run_orchestrator import RunOrchestrator


@pytest.fixture
def mock_events_repo():
    """Create mock events repository."""
    repo = MagicMock()
    repo.insert = AsyncMock()
    return repo


@pytest.fixture
def mock_run_plans_repo():
    """Create mock run plans repository."""
    repo = MagicMock()
    repo.create_run_plan = AsyncMock(return_value=uuid4())
    repo.update_run_plan_status = AsyncMock()
    repo.complete_run_plan = AsyncMock()
    return repo


@pytest.fixture
def mock_backtest_repo():
    """Create mock backtest repository."""
    repo = MagicMock()
    repo.create_variant_run = AsyncMock(return_value=uuid4())
    repo.update_variant_completed = AsyncMock()
    repo.update_variant_skipped = AsyncMock()
    repo.update_variant_failed = AsyncMock()
    # KB status methods for candidacy evaluation
    repo.update_variant_kb_status = AsyncMock()
    repo.get_breaker_state = AsyncMock(return_value={"kb_auto_candidacy_state": "enabled"})
    repo.get_recent_candidacy_decisions = AsyncMock(return_value=[])
    repo.get_candidate_count_rolling_24h = AsyncMock(return_value=0)
    return repo


@pytest.fixture
def mock_runner():
    """Create mock strategy runner."""
    return MagicMock()


@pytest.fixture
def orchestrator(
    mock_events_repo, mock_run_plans_repo, mock_backtest_repo, mock_runner
):
    """Create orchestrator with all mocked repos."""
    return RunOrchestrator(
        events_repo=mock_events_repo,
        runner=mock_runner,
        run_plans_repo=mock_run_plans_repo,
        backtest_repo=mock_backtest_repo,
    )


@pytest.fixture
def simple_csv_content():
    """Minimal valid OHLCV CSV."""
    return (
        b"ts,open,high,low,close,volume\n"
        b"2024-01-01T00:00:00Z,100,101,99,100.5,1000\n"
        b"2024-01-02T00:00:00Z,100.5,102,100,101,1100"
    )


class TestOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_init_with_all_repos(
        self, mock_events_repo, mock_run_plans_repo, mock_backtest_repo, mock_runner
    ):
        """Orchestrator accepts all repositories."""
        orch = RunOrchestrator(
            events_repo=mock_events_repo,
            runner=mock_runner,
            run_plans_repo=mock_run_plans_repo,
            backtest_repo=mock_backtest_repo,
        )
        assert orch._run_plans_repo is mock_run_plans_repo
        assert orch._backtest_repo is mock_backtest_repo

    def test_init_repos_optional(self, mock_events_repo, mock_runner):
        """Orchestrator works without optional repos."""
        orch = RunOrchestrator(
            events_repo=mock_events_repo,
            runner=mock_runner,
        )
        assert orch._run_plans_repo is None
        assert orch._backtest_repo is None


class TestExecutePersistence:
    """Tests for execute method persistence."""

    @pytest.mark.asyncio
    async def test_execute_creates_run_plan(
        self, orchestrator, mock_run_plans_repo, simple_csv_content
    ):
        """execute creates run_plan in DB at start."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[],
            dataset_ref="test.csv",
        )

        await orchestrator.execute(run_plan, simple_csv_content)

        mock_run_plans_repo.create_run_plan.assert_called_once()
        call_kwargs = mock_run_plans_repo.create_run_plan.call_args[1]
        assert call_kwargs["workspace_id"] == workspace_id
        assert call_kwargs["n_variants"] == 0

    @pytest.mark.asyncio
    async def test_execute_updates_status_to_running(
        self, orchestrator, mock_run_plans_repo, simple_csv_content
    ):
        """execute updates run_plan status to running after creation."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[],
            dataset_ref="test.csv",
        )

        await orchestrator.execute(run_plan, simple_csv_content)

        mock_run_plans_repo.update_run_plan_status.assert_called_once()
        call_args = mock_run_plans_repo.update_run_plan_status.call_args[0]
        assert call_args[1] == "running"

    @pytest.mark.asyncio
    async def test_execute_completes_run_plan(
        self, orchestrator, mock_run_plans_repo, simple_csv_content
    ):
        """execute calls complete_run_plan at end."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[
                RunVariant(
                    variant_id="abc123def456",
                    label="baseline",
                    spec_overrides={},
                )
            ],
            dataset_ref="test.csv",
        )

        await orchestrator.execute(run_plan, simple_csv_content)

        mock_run_plans_repo.complete_run_plan.assert_called_once()
        call_kwargs = mock_run_plans_repo.complete_run_plan.call_args[1]
        assert call_kwargs["status"] == "completed"

    @pytest.mark.asyncio
    async def test_execute_creates_variant_runs(
        self, orchestrator, mock_backtest_repo, simple_csv_content
    ):
        """execute creates backtest_run for each variant."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[
                RunVariant(variant_id="variant1", label="v1", spec_overrides={}),
                RunVariant(variant_id="variant2", label="v2", spec_overrides={}),
            ],
            dataset_ref="test.csv",
        )

        await orchestrator.execute(run_plan, simple_csv_content)

        # Should create 2 variant runs
        assert mock_backtest_repo.create_variant_run.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_updates_variant_completed(
        self, orchestrator, mock_backtest_repo, simple_csv_content
    ):
        """execute updates variant with completed status."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[
                RunVariant(variant_id="abc123", label="baseline", spec_overrides={}),
            ],
            dataset_ref="test.csv",
        )

        await orchestrator.execute(run_plan, simple_csv_content)

        # Should update the variant as completed
        mock_backtest_repo.update_variant_completed.assert_called_once()


class TestExecuteWithoutPersistence:
    """Tests for execute without persistence repos."""

    @pytest.mark.asyncio
    async def test_execute_works_without_repos(
        self, mock_events_repo, mock_runner, simple_csv_content
    ):
        """execute works when repos are not provided."""
        orchestrator = RunOrchestrator(
            events_repo=mock_events_repo,
            runner=mock_runner,
        )

        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[],
            dataset_ref="test.csv",
        )

        # Should not raise
        results = await orchestrator.execute(run_plan, simple_csv_content)
        assert results == []


class TestCompleteRunPlanAggregates:
    """Tests for complete_run_plan aggregate values."""

    @pytest.mark.asyncio
    async def test_complete_run_plan_counts_success(
        self, orchestrator, mock_run_plans_repo, simple_csv_content
    ):
        """complete_run_plan receives correct success count."""
        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[
                RunVariant(variant_id="v1", label="v1", spec_overrides={}),
                RunVariant(variant_id="v2", label="v2", spec_overrides={}),
            ],
            dataset_ref="test.csv",
        )

        await orchestrator.execute(run_plan, simple_csv_content)

        call_kwargs = mock_run_plans_repo.complete_run_plan.call_args[1]
        # Both variants should succeed (placeholder implementation)
        assert call_kwargs["n_completed"] == 2
        assert call_kwargs["n_failed"] == 0
        assert call_kwargs["n_skipped"] == 0


class TestFinalizeInFinally:
    """Tests that plan is finalized even on unexpected exceptions."""

    @pytest.mark.asyncio
    async def test_unexpected_error_still_finalizes_plan(
        self, mock_events_repo, mock_run_plans_repo, mock_backtest_repo, mock_runner
    ):
        """Plan is marked failed (not stuck in running) on unexpected exception."""
        # Make create_variant_run raise an unexpected error
        mock_backtest_repo.create_variant_run = AsyncMock(
            side_effect=RuntimeError("Database connection lost")
        )

        orchestrator = RunOrchestrator(
            events_repo=mock_events_repo,
            runner=mock_runner,
            run_plans_repo=mock_run_plans_repo,
            backtest_repo=mock_backtest_repo,
        )

        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[
                RunVariant(variant_id="v1", label="v1", spec_overrides={}),
            ],
            dataset_ref="test.csv",
        )

        # Execute should raise (error is re-raised after cleanup)
        csv_data = (
            b"ts,open,high,low,close,volume\n"
            b"2024-01-01T00:00:00Z,100,101,99,100.5,1000\n"
            b"2024-01-02T00:00:00Z,100.5,102,100,101,1100"
        )
        with pytest.raises(RuntimeError, match="Run plan failed unexpectedly"):
            await orchestrator.execute(run_plan, csv_data)

        # CRITICAL: complete_run_plan should still have been called with failed status
        mock_run_plans_repo.complete_run_plan.assert_called_once()
        call_kwargs = mock_run_plans_repo.complete_run_plan.call_args[1]
        assert call_kwargs["status"] == "failed"
        assert "Database connection lost" in call_kwargs["error_summary"]

    @pytest.mark.asyncio
    async def test_run_failed_event_journaled_on_unexpected_error(
        self, mock_events_repo, mock_run_plans_repo, mock_backtest_repo, mock_runner
    ):
        """RUN_FAILED event is journaled on unexpected exception."""
        from app.schemas import TradeEventType

        mock_backtest_repo.create_variant_run = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        orchestrator = RunOrchestrator(
            events_repo=mock_events_repo,
            runner=mock_runner,
            run_plans_repo=mock_run_plans_repo,
            backtest_repo=mock_backtest_repo,
        )

        workspace_id = uuid4()
        run_plan = RunPlan(
            workspace_id=workspace_id,
            base_spec={
                "strategy_name": "test",
                "risk": {"dollars_per_trade": 100, "max_positions": 1},
            },
            variants=[
                RunVariant(variant_id="v1", label="v1", spec_overrides={}),
            ],
            dataset_ref="test.csv",
        )

        csv_data = (
            b"ts,open,high,low,close,volume\n"
            b"2024-01-01T00:00:00Z,100,101,99,100.5,1000\n"
            b"2024-01-02T00:00:00Z,100.5,102,100,101,1100"
        )
        with pytest.raises(RuntimeError):
            await orchestrator.execute(run_plan, csv_data)

        # Check that RUN_FAILED event was journaled
        insert_calls = mock_events_repo.insert.call_args_list
        assert len(insert_calls) >= 2  # RUN_STARTED and RUN_FAILED

        # Last event should be RUN_FAILED
        last_event = insert_calls[-1][0][0]
        assert last_event.event_type == TradeEventType.RUN_FAILED
        assert "Unexpected error" in last_event.payload.get("error", "")
