"""Tests for event rollups repository."""

import pytest
from datetime import date
from unittest.mock import AsyncMock
from uuid import uuid4

from app.repositories.event_rollups import EventRollupsRepository


class TestEventRollupsRepository:
    """Tests for rollup operations."""

    @pytest.fixture
    def mock_conn(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_run_daily_rollup(self, mock_conn):
        """Aggregates events into rollups."""
        mock_conn.execute.return_value = "INSERT 0 5"
        workspace_id = uuid4()

        repo = EventRollupsRepository()
        count = await repo.run_daily_rollup(mock_conn, workspace_id, date(2026, 1, 10))

        assert count == 5
        mock_conn.execute.assert_called_once()
        # Verify query includes ON CONFLICT
        query = mock_conn.execute.call_args[0][0]
        assert "ON CONFLICT" in query

    @pytest.mark.asyncio
    async def test_get_rollups_for_workspace(self, mock_conn):
        """Returns rollups for workspace in date range."""
        workspace_id = uuid4()
        mock_conn.fetch.return_value = [
            {
                "event_type": "ORDER_FILLED",
                "rollup_date": date(2026, 1, 10),
                "event_count": 50,
                "error_count": 2,
            }
        ]

        repo = EventRollupsRepository()
        rollups = await repo.get_rollups(
            conn=mock_conn,
            workspace_id=workspace_id,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )

        assert len(rollups) == 1
        assert rollups[0]["event_type"] == "ORDER_FILLED"

    @pytest.mark.asyncio
    async def test_preview_daily_rollup(self, mock_conn):
        """Preview returns counts without aggregating."""
        mock_conn.fetchval.side_effect = [100, 15]
        workspace_id = uuid4()

        repo = EventRollupsRepository()
        result = await repo.preview_daily_rollup(mock_conn, workspace_id, date(2026, 1, 10))

        assert result["events_to_aggregate"] == 100
        assert result["rollup_rows_to_create"] == 15
        # fetchval for SELECT COUNT, not execute for INSERT
        assert mock_conn.execute.call_count == 0
