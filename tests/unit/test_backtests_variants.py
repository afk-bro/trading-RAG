"""Unit tests for BacktestRepository variant methods."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.repositories.backtests import BacktestRepository


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


@pytest.fixture
def repo(mock_pool):
    """Create repository with mock pool."""
    return BacktestRepository(mock_pool)


class TestCreateVariantRun:
    """Tests for create_variant_run method."""

    @pytest.mark.asyncio
    async def test_create_variant_run_returns_id(self, repo, mock_pool):
        """create_variant_run returns the new run ID."""
        run_id = uuid4()
        run_plan_id = uuid4()
        workspace_id = uuid4()
        strategy_entity_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=run_id)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        result = await repo.create_variant_run(
            run_plan_id=run_plan_id,
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            variant_index=0,
            variant_fingerprint="abc123def456",
            params={"lookback_days": 200},
            dataset_meta={"filename": "BTC_1h.csv"},
        )

        assert result == run_id
        mock_conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_variant_run_sets_run_kind(self, repo, mock_pool):
        """create_variant_run sets run_kind to test_variant by default."""
        run_id = uuid4()
        run_plan_id = uuid4()
        workspace_id = uuid4()
        strategy_entity_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=run_id)
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.create_variant_run(
            run_plan_id=run_plan_id,
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            variant_index=0,
            variant_fingerprint="abc123def456",
            params={},
            dataset_meta={},
        )

        call_args = mock_conn.fetchval.call_args[0]
        assert "test_variant" in call_args


class TestUpdateVariantCompleted:
    """Tests for update_variant_completed method."""

    @pytest.mark.asyncio
    async def test_update_variant_completed_sets_fields(self, repo, mock_pool):
        """update_variant_completed updates all result fields."""
        run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_variant_completed(
            run_id=run_id,
            summary={"sharpe": 1.42, "return_pct": 12.5},
            equity_curve=[{"t": "2024-01-01", "equity": 10000}],
            trades=[],
            objective_score=1.42,
            has_equity_curve=True,
            has_trades=False,
            equity_points=100,
            trade_count=0,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert run_id in call_args
        assert 1.42 in call_args  # objective_score


class TestUpdateVariantSkipped:
    """Tests for update_variant_skipped method."""

    @pytest.mark.asyncio
    async def test_update_variant_skipped_sets_reason(self, repo, mock_pool):
        """update_variant_skipped sets skip_reason."""
        run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_variant_skipped(
            run_id=run_id,
            skip_reason="invalid_params",
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert run_id in call_args
        assert "invalid_params" in call_args


class TestUpdateVariantFailed:
    """Tests for update_variant_failed method."""

    @pytest.mark.asyncio
    async def test_update_variant_failed_sets_error(self, repo, mock_pool):
        """update_variant_failed sets error message."""
        run_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        await repo.update_variant_failed(
            run_id=run_id,
            error="Simulation crashed: out of memory",
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert run_id in call_args
        assert "Simulation crashed: out of memory" in call_args
