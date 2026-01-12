"""Tests for event retention service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.retention import RetentionService


class TestRetentionService:
    """Tests for retention cleanup."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = conn
        return pool, conn

    @pytest.mark.asyncio
    async def test_cleanup_respects_severity_tiers(self, mock_pool):
        """Deletes info/debug at 30 days, warn/error at 90 days."""
        pool, conn = mock_pool
        conn.execute.side_effect = ["DELETE 100", "DELETE 50"]

        service = RetentionService(pool)
        result = await service.run_cleanup()

        assert result["info_debug_deleted"] == 100
        assert result["warn_error_deleted"] == 50
        assert conn.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_preserves_pinned(self, mock_pool):
        """Pinned events are never deleted."""
        pool, conn = mock_pool
        conn.execute.side_effect = ["DELETE 0", "DELETE 0"]

        service = RetentionService(pool)
        await service.run_cleanup()

        # Verify both queries include pinned = FALSE
        for call in conn.execute.call_args_list:
            query = call[0][0]
            assert "pinned = FALSE" in query
