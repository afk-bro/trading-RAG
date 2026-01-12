"""Tests for event retention service."""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.services.retention import RetentionService


class TestRetentionService:
    """Tests for retention cleanup."""

    @pytest.fixture
    def mock_conn(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_cleanup_respects_severity_tiers(self, mock_conn):
        """Deletes info/debug at 30 days, warn/error at 90 days."""
        mock_conn.execute.side_effect = ["DELETE 100", "DELETE 50"]
        workspace_id = uuid4()

        service = RetentionService()
        result = await service.run_cleanup(mock_conn, workspace_id)

        assert result["info_debug_deleted"] == 100
        assert result["warn_error_deleted"] == 50
        assert mock_conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_preserves_pinned(self, mock_conn):
        """Pinned events are never deleted."""
        mock_conn.execute.side_effect = ["DELETE 0", "DELETE 0"]
        workspace_id = uuid4()

        service = RetentionService()
        await service.run_cleanup(mock_conn, workspace_id)

        # Verify both queries include pinned = FALSE
        for call in mock_conn.execute.call_args_list:
            query = call[0][0]
            assert "pinned = FALSE" in query

    @pytest.mark.asyncio
    async def test_cleanup_scopes_to_workspace(self, mock_conn):
        """Cleanup only affects specified workspace."""
        mock_conn.execute.side_effect = ["DELETE 10", "DELETE 5"]
        workspace_id = uuid4()

        service = RetentionService()
        await service.run_cleanup(mock_conn, workspace_id)

        # Verify workspace_id is in both queries
        for call in mock_conn.execute.call_args_list:
            query = call[0][0]
            assert "workspace_id = $1" in query
            assert call[0][1] == workspace_id

    @pytest.mark.asyncio
    async def test_preview_cleanup_returns_counts(self, mock_conn):
        """Preview returns counts without deleting."""
        mock_conn.fetchval.side_effect = [25, 10]
        workspace_id = uuid4()

        service = RetentionService()
        result = await service.preview_cleanup(mock_conn, workspace_id)

        assert result["info_debug_would_delete"] == 25
        assert result["warn_error_would_delete"] == 10
        # fetchval for SELECT COUNT, not execute for DELETE
        assert mock_conn.execute.call_count == 0
