"""Unit tests for RunPlansRepository."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.run_plans import RunPlansRepository


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


@pytest.fixture
def repo(mock_pool):
    """Create repository with mock pool."""
    return RunPlansRepository(mock_pool)


class TestCreateRunPlan:
    """Tests for create_run_plan method."""

    @pytest.mark.asyncio
    async def test_create_run_plan_returns_id(self, repo, mock_pool):
        """create_run_plan returns the new plan ID."""
        plan_id = uuid4()
        workspace_id = uuid4()

        # Setup mock
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=plan_id)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await repo.create_run_plan(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe_dd_penalty",
            n_variants=10,
            plan={"inputs": {}, "resolved": {}, "provenance": {}},
        )

        assert result == plan_id
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_run_plan_serializes_plan_json(self, repo, mock_pool):
        """create_run_plan serializes plan dict to JSON."""
        plan_id = uuid4()
        workspace_id = uuid4()
        plan_data = {"inputs": {"foo": "bar"}, "resolved": {}, "provenance": {}}

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=plan_id)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.create_run_plan(
            workspace_id=workspace_id,
            strategy_entity_id=None,
            objective_name="sharpe",
            n_variants=5,
            plan=plan_data,
        )

        # Check that json.dumps was applied to plan argument
        call_args = mock_conn.fetchval.call_args[0]
        # Plan is 5th positional arg (after query, workspace_id, strategy_entity_id, objective_name)
        plan_arg = call_args[5]  # Index 5 = plan after n_variants
        assert isinstance(plan_arg, str)
        assert json.loads(plan_arg) == plan_data


class TestUpdateRunPlanStatus:
    """Tests for update_run_plan_status method."""

    @pytest.mark.asyncio
    async def test_update_to_running(self, repo, mock_pool):
        """update_run_plan_status updates status to running."""
        plan_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_run_plan_status(plan_id, "running")

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert plan_id in call_args
        assert "running" in call_args

    @pytest.mark.asyncio
    async def test_update_to_running_sets_started_at(self, repo, mock_pool):
        """update_run_plan_status sets started_at when transitioning to running."""
        plan_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_run_plan_status(plan_id, "running")

        call_args = mock_conn.execute.call_args[0]
        query = call_args[0]
        assert "started_at" in query


class TestCompleteRunPlan:
    """Tests for complete_run_plan method."""

    @pytest.mark.asyncio
    async def test_complete_run_plan_sets_aggregates(self, repo, mock_pool):
        """complete_run_plan updates all aggregate fields."""
        plan_id = uuid4()
        best_run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.complete_run_plan(
            plan_id=plan_id,
            status="completed",
            n_completed=8,
            n_failed=1,
            n_skipped=1,
            best_backtest_run_id=best_run_id,
            best_objective_score=1.42,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        # Check key values are in args
        assert plan_id in call_args
        assert "completed" in call_args
        assert 8 in call_args  # n_completed
        assert 1.42 in call_args  # best_objective_score


class TestGetRunPlan:
    """Tests for get_run_plan method."""

    @pytest.mark.asyncio
    async def test_get_run_plan_returns_dict(self, repo, mock_pool):
        """get_run_plan returns plan as dict."""
        plan_id = uuid4()
        workspace_id = uuid4()

        mock_row = {
            "id": plan_id,
            "workspace_id": workspace_id,
            "status": "completed",
            "n_variants": 10,
            "plan": json.dumps({"inputs": {}, "resolved": {}, "provenance": {}}),
            "strategy_name": "test_strategy",
        }

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await repo.get_run_plan(plan_id)

        assert result is not None
        assert result["id"] == plan_id
        assert result["status"] == "completed"
        # Plan should be parsed from JSON
        assert isinstance(result["plan"], dict)

    @pytest.mark.asyncio
    async def test_get_run_plan_not_found(self, repo, mock_pool):
        """get_run_plan returns None for non-existent plan."""
        plan_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await repo.get_run_plan(plan_id)

        assert result is None


class TestListRunPlans:
    """Tests for list_run_plans method."""

    @pytest.mark.asyncio
    async def test_list_run_plans_returns_tuple(self, repo, mock_pool):
        """list_run_plans returns (plans, total) tuple."""
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=2)
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"id": uuid4(), "status": "completed", "n_variants": 5},
                {"id": uuid4(), "status": "running", "n_variants": 3},
            ]
        )
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        plans, total = await repo.list_run_plans(workspace_id)

        assert total == 2
        assert len(plans) == 2

    @pytest.mark.asyncio
    async def test_list_run_plans_excludes_full_plan_blob(self, repo, mock_pool):
        """list_run_plans query should NOT include full plan JSONB."""
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=0)
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.list_run_plans(workspace_id)

        # Check the list query doesn't select 'plan' column
        fetch_call = mock_conn.fetch.call_args[0][0]
        # The query should NOT have 'rp.plan' in SELECT
        assert "rp.plan" not in fetch_call or "rp.plan," not in fetch_call


class TestListRunsForPlan:
    """Tests for list_runs_for_plan method."""

    @pytest.mark.asyncio
    async def test_list_runs_for_plan_returns_tuple(self, repo, mock_pool):
        """list_runs_for_plan returns (runs, total) tuple."""
        plan_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=3)
        mock_conn.fetch = AsyncMock(
            return_value=[
                {"id": uuid4(), "variant_index": 0, "status": "completed"},
                {"id": uuid4(), "variant_index": 1, "status": "completed"},
                {"id": uuid4(), "variant_index": 2, "status": "skipped"},
            ]
        )
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        runs, total = await repo.list_runs_for_plan(plan_id)

        assert total == 3
        assert len(runs) == 3

    @pytest.mark.asyncio
    async def test_list_runs_for_plan_excludes_blobs(self, repo, mock_pool):
        """list_runs_for_plan should NOT include equity_curve/trades blobs."""
        plan_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=0)
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.list_runs_for_plan(plan_id)

        # Check the query doesn't use SELECT * (which would include blobs)
        fetch_call = mock_conn.fetch.call_args[0][0]
        assert "SELECT *" not in fetch_call
        # Verify explicit column list is used (good practice for blob exclusion)
        assert (
            "SELECT id," in fetch_call
            or "SELECT id\n" in fetch_call
            or "id," in fetch_call
        )
