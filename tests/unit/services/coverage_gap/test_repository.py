"""Unit tests for MatchRunRepository coverage status updates."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.services.coverage_gap.repository import MatchRunRepository


class TestUpdateCoverageStatusGuard:
    """Tests for resolution guard logic."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        pool = MagicMock()
        pool.acquire = MagicMock()
        return pool

    @pytest.fixture
    def repo(self, mock_pool):
        """Create repository with mock pool."""
        return MatchRunRepository(mock_pool)

    @pytest.mark.asyncio
    async def test_resolve_with_note_succeeds(self, repo, mock_pool):
        """Resolving with a note should succeed without checking candidates."""
        run_id = uuid4()
        workspace_id = uuid4()

        # Mock the connection context manager and fetchrow for the UPDATE
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": run_id,
                "coverage_status": "resolved",
                "acknowledged_at": None,
                "acknowledged_by": None,
                "resolved_at": "2026-01-16T00:00:00Z",
                "resolved_by": "admin",
                "resolution_note": "False positive",
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repo.update_coverage_status(
            run_id=run_id,
            workspace_id=workspace_id,
            status="resolved",
            note="False positive",
            updated_by="admin",
        )

        assert result is not None
        assert result["coverage_status"] == "resolved"
        # Should only call fetchrow once (for the UPDATE, not for checking candidates)
        assert mock_conn.fetchrow.call_count == 1

    @pytest.mark.asyncio
    async def test_resolve_without_note_checks_candidates(self, repo, mock_pool):
        """Resolving without note should check for candidates."""
        run_id = uuid4()
        workspace_id = uuid4()

        # First call returns candidates, second call is the UPDATE
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            side_effect=[
                # First call: check for candidates (has some)
                {"candidate_strategy_ids": [uuid4(), uuid4()]},
                # Second call: UPDATE returns the row
                {
                    "id": run_id,
                    "coverage_status": "resolved",
                    "acknowledged_at": None,
                    "acknowledged_by": None,
                    "resolved_at": "2026-01-16T00:00:00Z",
                    "resolved_by": "admin",
                    "resolution_note": None,
                },
            ]
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repo.update_coverage_status(
            run_id=run_id,
            workspace_id=workspace_id,
            status="resolved",
            note=None,  # No note provided
            updated_by="admin",
        )

        assert result is not None
        assert result["coverage_status"] == "resolved"
        # Should call fetchrow twice (check candidates + UPDATE)
        assert mock_conn.fetchrow.call_count == 2

    @pytest.mark.asyncio
    async def test_resolve_without_note_or_candidates_fails(self, repo, mock_pool):
        """Resolving without note AND without candidates should raise ValueError."""
        run_id = uuid4()
        workspace_id = uuid4()

        # First call returns empty candidates
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"candidate_strategy_ids": []}  # No candidates
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(ValueError) as exc_info:
            await repo.update_coverage_status(
                run_id=run_id,
                workspace_id=workspace_id,
                status="resolved",
                note=None,  # No note provided
                updated_by="admin",
            )

        assert "candidate_strategy_ids or resolution_note" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_without_note_null_candidates_fails(self, repo, mock_pool):
        """Resolving without note when candidates is NULL should raise ValueError."""
        run_id = uuid4()
        workspace_id = uuid4()

        # First call returns NULL candidates
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={"candidate_strategy_ids": None}  # NULL in DB
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(ValueError) as exc_info:
            await repo.update_coverage_status(
                run_id=run_id,
                workspace_id=workspace_id,
                status="resolved",
                note=None,
                updated_by="admin",
            )

        assert "candidate_strategy_ids or resolution_note" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resolve_not_found_returns_none(self, repo, mock_pool):
        """Resolving non-existent run should return None."""
        run_id = uuid4()
        workspace_id = uuid4()

        # First call returns None (not found)
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repo.update_coverage_status(
            run_id=run_id,
            workspace_id=workspace_id,
            status="resolved",
            note=None,
            updated_by="admin",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_acknowledge_does_not_check_candidates(self, repo, mock_pool):
        """Acknowledging should not check candidates (only resolve has guard)."""
        run_id = uuid4()
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": run_id,
                "coverage_status": "acknowledged",
                "acknowledged_at": "2026-01-16T00:00:00Z",
                "acknowledged_by": "admin",
                "resolved_at": None,
                "resolved_by": None,
                "resolution_note": None,
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repo.update_coverage_status(
            run_id=run_id,
            workspace_id=workspace_id,
            status="acknowledged",
            note=None,
            updated_by="admin",
        )

        assert result is not None
        assert result["coverage_status"] == "acknowledged"
        # Should only call fetchrow once (no guard check)
        assert mock_conn.fetchrow.call_count == 1

    @pytest.mark.asyncio
    async def test_reopen_does_not_check_candidates(self, repo, mock_pool):
        """Reopening should not check candidates."""
        run_id = uuid4()
        workspace_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "id": run_id,
                "coverage_status": "open",
                "acknowledged_at": None,
                "acknowledged_by": None,
                "resolved_at": None,
                "resolved_by": None,
                "resolution_note": "Previous note",
            }
        )
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await repo.update_coverage_status(
            run_id=run_id,
            workspace_id=workspace_id,
            status="open",
            note=None,
            updated_by="admin",
        )

        assert result is not None
        assert result["coverage_status"] == "open"
        # Should only call fetchrow once (no guard check)
        assert mock_conn.fetchrow.call_count == 1


class TestAutoResolveByIntentSignature:
    """Tests for auto-resolve logic."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock database pool."""
        pool = MagicMock()
        pool.acquire = MagicMock()
        return pool

    @pytest.fixture
    def repo(self, mock_pool):
        """Create repository with mock pool."""
        return MatchRunRepository(mock_pool)

    @pytest.mark.asyncio
    async def test_auto_resolve_returns_count(self, repo, mock_pool):
        """Auto-resolve returns count of resolved runs."""
        workspace_id = uuid4()
        intent_sig = "abc123" * 10

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 3")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        count = await repo.auto_resolve_by_intent_signature(
            workspace_id=workspace_id,
            intent_signature=intent_sig,
        )

        assert count == 3

    @pytest.mark.asyncio
    async def test_auto_resolve_returns_zero_when_none_found(self, repo, mock_pool):
        """Auto-resolve returns 0 when no matching runs."""
        workspace_id = uuid4()
        intent_sig = "xyz789" * 10

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 0")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        count = await repo.auto_resolve_by_intent_signature(
            workspace_id=workspace_id,
            intent_signature=intent_sig,
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_auto_resolve_excludes_current_run(self, repo, mock_pool):
        """Auto-resolve can exclude the current run from resolution."""
        workspace_id = uuid4()
        intent_sig = "abc123" * 10
        exclude_id = uuid4()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 2")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        count = await repo.auto_resolve_by_intent_signature(
            workspace_id=workspace_id,
            intent_signature=intent_sig,
            exclude_run_id=exclude_id,
        )

        assert count == 2
        # Verify the query was called with 5 args (query + 4 params including exclude_id)
        call_args = mock_conn.execute.call_args
        # call_args[0] contains positional args: (query, ws_id, intent_sig, now, exclude_id)
        assert len(call_args[0]) == 5
        # The query should contain "id != $4"
        assert "id != $4" in call_args[0][0]
        # exclude_id should be in the args
        assert call_args[0][4] == exclude_id

    @pytest.mark.asyncio
    async def test_auto_resolve_sets_system_resolved_by(self, repo, mock_pool):
        """Auto-resolve sets resolved_by to 'system'."""
        workspace_id = uuid4()
        intent_sig = "abc123" * 10

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="UPDATE 1")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        await repo.auto_resolve_by_intent_signature(
            workspace_id=workspace_id,
            intent_signature=intent_sig,
        )

        # Verify the query sets resolved_by = 'system'
        call_args = mock_conn.execute.call_args
        assert "resolved_by = 'system'" in call_args[0][0]
        assert "Auto-resolved by successful match" in call_args[0][0]
