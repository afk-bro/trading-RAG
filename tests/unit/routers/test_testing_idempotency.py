# tests/unit/routers/test_testing_idempotency.py
"""Tests for run plan creation idempotency."""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import HTTPException


class TestRunPlanIdempotency:
    """Tests for idempotency handling in run plan creation."""

    @pytest.mark.asyncio
    async def test_returns_existing_on_idempotency_key_match(self):
        """Returns existing plan when idempotency key matches."""
        from app.services.testing.idempotency import create_run_plan_with_idempotency

        existing_plan = {
            "id": uuid4(),
            "status": "pending",
            "idempotency_key": "test-key",
        }

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = existing_plan

        result = await create_run_plan_with_idempotency(
            workspace_id=uuid4(),
            strategy_entity_id=None,
            objective_name="sharpe",
            plan={"test": True},
            idempotency_key="test-key",
            repo=mock_repo,
        )

        assert result["id"] == existing_plan["id"]
        assert result["status"] == "existing"
        mock_repo.create_run_plan.assert_not_called()

    @pytest.mark.asyncio
    async def test_409_on_idempotency_key_match_non_pending(self):
        """Returns 409 when idempotency key matches non-pending plan."""
        from app.services.testing.idempotency import create_run_plan_with_idempotency

        existing_plan = {
            "id": uuid4(),
            "status": "running",  # Not pending
            "idempotency_key": "test-key",
        }

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = existing_plan

        with pytest.raises(HTTPException) as exc_info:
            await create_run_plan_with_idempotency(
                workspace_id=uuid4(),
                strategy_entity_id=None,
                objective_name="sharpe",
                plan={"test": True},
                idempotency_key="test-key",
                repo=mock_repo,
            )

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_409_on_request_hash_match(self):
        """Returns 409 when request hash matches (duplicate request)."""
        from app.services.testing.idempotency import create_run_plan_with_idempotency

        existing_plan = {
            "id": uuid4(),
            "status": "pending",
            "request_hash": "abc123",
        }

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = None
        mock_repo.get_by_request_hash.return_value = existing_plan

        with pytest.raises(HTTPException) as exc_info:
            await create_run_plan_with_idempotency(
                workspace_id=uuid4(),
                strategy_entity_id=None,
                objective_name="sharpe",
                plan={"test": True},
                idempotency_key=None,
                repo=mock_repo,
            )

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio
    async def test_creates_new_plan_when_no_match(self):
        """Creates new plan when no idempotency or hash match."""
        from app.services.testing.idempotency import create_run_plan_with_idempotency

        new_plan_id = uuid4()

        mock_repo = AsyncMock()
        mock_repo.get_by_idempotency_key.return_value = None
        mock_repo.get_by_request_hash.return_value = None
        mock_repo.create_run_plan.return_value = new_plan_id

        result = await create_run_plan_with_idempotency(
            workspace_id=uuid4(),
            strategy_entity_id=None,
            objective_name="sharpe",
            plan={"test": True},
            idempotency_key="new-key",
            repo=mock_repo,
        )

        assert result["id"] == new_plan_id
        assert result["status"] == "created"
        mock_repo.create_run_plan.assert_called_once()
