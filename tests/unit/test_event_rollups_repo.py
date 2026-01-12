"""Tests for event rollups repository."""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.repositories.event_rollups import EventRollupsRepository


class TestEventRollupsRepository:
    """Tests for rollup operations."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        return pool, conn

    @pytest.mark.asyncio
    async def test_run_daily_rollup(self, mock_pool):
        """Aggregates events into rollups."""
        pool, conn = mock_pool
        conn.execute.return_value = "INSERT 0 5"

        repo = EventRollupsRepository(pool)
        count = await repo.run_daily_rollup(date(2026, 1, 10))

        assert count == 5
        conn.execute.assert_called_once()
        # Verify query includes ON CONFLICT
        query = conn.execute.call_args[0][0]
        assert "ON CONFLICT" in query

    @pytest.mark.asyncio
    async def test_get_rollups_for_workspace(self, mock_pool):
        """Returns rollups for workspace in date range."""
        pool, conn = mock_pool
        workspace_id = uuid4()
        conn.fetch.return_value = [
            {
                "event_type": "ORDER_FILLED",
                "rollup_date": date(2026, 1, 10),
                "event_count": 50,
                "error_count": 2,
            }
        ]

        repo = EventRollupsRepository(pool)
        rollups = await repo.get_rollups(
            workspace_id=workspace_id,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )

        assert len(rollups) == 1
        assert rollups[0]["event_type"] == "ORDER_FILLED"
