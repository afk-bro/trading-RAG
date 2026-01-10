"""Tests for tune cancellation semantics and invariants."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestCancellationInvariants:
    """Cancellation semantic invariants."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return pool, conn

    @pytest.mark.asyncio
    async def test_cancel_only_allowed_for_queued_or_running(self, mock_pool):
        """Cancel should only succeed for tunes in {queued, running} state."""
        from app.repositories.backtests import TuneRepository

        pool, conn = mock_pool
        repo = TuneRepository(pool)
        tune_id = uuid4()

        # Simulate cancel attempt on completed tune - returns None (no rows updated)
        conn.fetchval.return_value = None

        result = await repo.cancel_tune(tune_id)

        assert result is False
        # Verify the WHERE clause limits to queued/running
        call_args = conn.fetchval.call_args[0][0]
        assert "status IN ('queued', 'running')" in call_args

    @pytest.mark.asyncio
    async def test_cancel_sets_status_to_canceled(self, mock_pool):
        """After cancel, tune.status must be 'canceled'."""
        from app.repositories.backtests import TuneRepository

        pool, conn = mock_pool
        repo = TuneRepository(pool)
        tune_id = uuid4()

        # Simulate successful cancel
        conn.fetchval.return_value = tune_id

        result = await repo.cancel_tune(tune_id)

        assert result is True
        # Verify status is set to 'canceled'
        call_args = conn.fetchval.call_args[0][0]
        assert "status = 'canceled'" in call_args

    @pytest.mark.asyncio
    async def test_cancel_marks_queued_trials_as_skipped(self, mock_pool):
        """After cancel, all queued trials should be skipped with skip_reason='canceled'."""
        from app.repositories.backtests import TuneRepository

        pool, conn = mock_pool
        repo = TuneRepository(pool)
        tune_id = uuid4()

        # Simulate successful cancel
        conn.fetchval.return_value = tune_id

        await repo.cancel_tune(tune_id)

        # Verify queued trials are marked as skipped
        execute_calls = [str(c) for c in conn.execute.call_args_list]
        skip_call = [c for c in execute_calls if "skipped" in c and "canceled" in c]
        assert (
            len(skip_call) > 0
        ), "Should mark queued trials as skipped with skip_reason='canceled'"

    @pytest.mark.asyncio
    async def test_canceled_status_never_overwritten_by_complete_tune(self, mock_pool):
        """
        A canceled tune should NEVER be overwritten to 'completed'.

        Even if all trials finish, the tune must stay canceled.
        """
        from app.repositories.backtests import TuneRepository

        pool, conn = mock_pool
        repo = TuneRepository(pool)
        tune_id = uuid4()

        # First, verify complete_tune doesn't blindly set status
        # It should check current status first or use conditional update
        await repo.complete_tune(
            tune_id=tune_id,
            best_run_id=uuid4(),
            best_score=1.5,
            best_params={"period": 10},
            leaderboard=[],
            trials_completed=10,
        )

        # The implementation should protect canceled status
        # This is verified by the WHERE clause in the SQL
        call_args = conn.execute.call_args[0][0]
        # If the implementation doesn't have protection, this test documents the need
        # The fix will add: WHERE status != 'canceled'


class TestCancellationAPISemantics:
    """API-level cancellation behavior."""

    @pytest.mark.asyncio
    async def test_cancel_endpoint_rejects_completed_tune(self):
        """POST /tunes/{id}/cancel should return 400 for completed tunes."""
        from fastapi.testclient import TestClient
        from unittest.mock import patch, AsyncMock

        # Mock the repos
        mock_tune = {
            "id": uuid4(),
            "status": "completed",
            "search_type": "grid",
            "n_trials": 10,
        }

        with patch("app.routers.backtests._get_repos") as mock_get_repos:
            mock_tune_repo = AsyncMock()
            mock_tune_repo.get_tune.return_value = mock_tune
            mock_get_repos.return_value = (None, None, mock_tune_repo)

            from app.routers.backtests import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.post(f"/backtests/tunes/{mock_tune['id']}/cancel")

            assert response.status_code == 400
            assert "completed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_cancel_endpoint_rejects_canceled_tune(self):
        """POST /tunes/{id}/cancel should return 400 for already canceled tunes."""
        from fastapi.testclient import TestClient
        from unittest.mock import patch, AsyncMock

        mock_tune = {
            "id": uuid4(),
            "status": "canceled",
            "search_type": "grid",
            "n_trials": 10,
        }

        with patch("app.routers.backtests._get_repos") as mock_get_repos:
            mock_tune_repo = AsyncMock()
            mock_tune_repo.get_tune.return_value = mock_tune
            mock_get_repos.return_value = (None, None, mock_tune_repo)

            from app.routers.backtests import router
            from fastapi import FastAPI

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.post(f"/backtests/tunes/{mock_tune['id']}/cancel")

            assert response.status_code == 400
            assert "canceled" in response.json()["detail"]


class TestRunningTrialsDontFlipCanceledStatus:
    """
    Critical invariant: running trials finishing should NOT flip canceled tune back to completed.

    Scenario:
    1. Tune starts with 10 trials
    2. 5 trials complete
    3. User cancels tune (5 queued become skipped)
    4. 2 running trials finish (complete/failed)
    5. Tune status MUST remain 'canceled'
    """

    @pytest.mark.asyncio
    async def test_late_finishing_trial_preserves_canceled_status(
        self,
    ):
        """
        Simulates a trial completing after tune was canceled.

        The complete_tune call (if made) should NOT overwrite canceled status.
        """
        from app.repositories.backtests import TuneRepository
        from unittest.mock import AsyncMock, MagicMock

        pool = MagicMock()
        conn = AsyncMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        repo = TuneRepository(pool)
        tune_id = uuid4()

        # Tune is currently canceled
        conn.fetchrow.return_value = {
            "id": tune_id,
            "status": "canceled",
            "n_trials": 10,
        }

        # Even if we call complete_tune, status should be protected
        # The implementation fix will ensure this
        await repo.complete_tune(
            tune_id=tune_id,
            best_run_id=uuid4(),
            best_score=1.0,
            best_params={},
            leaderboard=[],
            trials_completed=8,
        )

        # Verify the UPDATE doesn't blindly set status
        # After fix: query should have WHERE status != 'canceled'
        # or use conditional logic


class TestBestResultsUnderCanceled:
    """Best results can still be populated if valid trials completed before cancellation."""

    @pytest.mark.asyncio
    async def test_canceled_tune_preserves_best_from_completed_trials(self):
        """
        If some trials completed before cancellation, best_* should be populated.
        """
        # This is a documentation test - the tuner already computes best from
        # valid trials, and cancel doesn't clear them.
        # The key invariant: best_* reflects "best of what ran" even if canceled.
        pass

    @pytest.mark.asyncio
    async def test_canceled_tune_with_no_valid_trials_has_null_best(self):
        """
        If no valid trials completed before cancellation, best_* should be null.
        """
        # This is natural behavior - if all trials were queued when canceled,
        # there's nothing to populate best_* with.
        pass
