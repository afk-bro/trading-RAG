"""Tests for tune regime attribution backfill job.

TDD tests for backfilling regime columns on existing tunes:
1. Populated all regime columns correctly
2. Skips already populated tunes
3. Handles tunes with no completed runs
4. Handles runs without regime_oos snapshot
5. Handles runs without metrics_oos (still populates regime)
6. Dry run mode doesn't modify DB
7. Limit parameter works
8. Returns correct stats (processed/skipped/errors)
"""

import json
import pytest
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from app.services.kb.types import RegimeSnapshot
from app.jobs.backfill_tune_regime import (
    run_tune_regime_backfill,
    BackfillResult,
    BackfillTuneRegimeJob,
)


# =============================================================================
# Mock Data Builders
# =============================================================================


def make_regime_snapshot(
    schema_version: str = "regime_v1_1",
    regime_tags: Optional[list[str]] = None,
    atr_pct: float = 0.02,
    trend_strength: float = 0.7,
    trend_dir: int = 1,
) -> dict:
    """Create a mock regime snapshot dict (as stored in DB)."""
    tags = regime_tags or ["uptrend", "high_vol", "efficient"]
    return {
        "schema_version": schema_version,
        "regime_tags": tags,
        "atr_pct": atr_pct,
        "trend_strength": trend_strength,
        "trend_dir": trend_dir,
        "zscore": 0.5,
        "rsi": 55.0,
        "efficiency_ratio": 0.75,
        "bb_width_pct": 0.03,
    }


def make_tune_row(
    tune_id: Optional[UUID] = None,
    workspace_id: Optional[UUID] = None,
    strategy_entity_id: Optional[UUID] = None,
    status: str = "completed",
    regime_key: Optional[str] = None,
) -> dict:
    """Create a mock tune row from DB."""
    return {
        "id": tune_id or uuid4(),
        "workspace_id": workspace_id or uuid4(),
        "strategy_entity_id": strategy_entity_id or uuid4(),
        "status": status,
        "regime_key": regime_key,
    }


def make_tune_run_row(
    tune_id: UUID,
    run_id: Optional[UUID] = None,
    status: str = "completed",
    score_oos: Optional[float] = None,
    objective_score: Optional[float] = None,
    metrics_oos: Optional[dict] = None,
    params: Optional[dict] = None,
) -> dict:
    """Create a mock tune_run row from DB (as returned by fetchrow)."""
    return {
        "run_id": run_id or uuid4(),
        "params": params,  # Already parsed dict (simulating JSONB)
        "objective_score": objective_score,
        "score_oos": score_oos,
        "metrics_oos": metrics_oos,  # Already parsed dict (simulating JSONB)
    }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_db_pool():
    """Create mock database pool with async context manager support."""
    pool = MagicMock()

    # Create a mock connection that will be returned by acquire()
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock()

    # Make acquire() return an async context manager
    async_cm = AsyncMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    async_cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=async_cm)

    # Store the connection for test access
    pool._mock_conn = mock_conn

    return pool


@pytest.fixture
def sample_regime_snapshot():
    """Create a sample regime snapshot dict for testing."""
    return make_regime_snapshot(
        schema_version="regime_v1_1",
        regime_tags=["uptrend", "high_vol", "efficient"],
    )


@pytest.fixture
def sample_tune_id():
    """Create a sample tune ID."""
    return uuid4()


# =============================================================================
# Test Classes
# =============================================================================


class TestBackfillPopulatesRegimeColumns:
    """Test that backfill correctly populates all regime columns."""

    @pytest.mark.asyncio
    async def test_backfill_populates_regime_columns(
        self, mock_db_pool, sample_tune_id, sample_regime_snapshot
    ):
        """Verify all regime columns are populated correctly."""
        # Setup: tune without regime columns
        tune_row = make_tune_row(tune_id=sample_tune_id, status="completed")

        # Best run with regime snapshot in metrics_oos
        best_run = make_tune_run_row(
            tune_id=sample_tune_id,
            score_oos=1.5,
            metrics_oos={"regime": sample_regime_snapshot, "sharpe": 1.5},
            params={"lookback": 20, "threshold": 0.5},
        )

        # Track updates
        updates_called = []

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=[tune_row])
        conn.fetchrow = AsyncMock(return_value=best_run)

        async def capture_execute(query, *args):
            if "UPDATE backtest_tunes" in query:
                updates_called.append(
                    {
                        "tune_id": args[0],
                        "regime_schema_version": args[1],
                        "tag_ruleset_id": args[2],
                        "regime_key": args[3],
                        "regime_fingerprint": args[4],
                        "trend_tag": args[5],
                        "vol_tag": args[6],
                        "efficiency_tag": args[7],
                        "best_oos_score": args[8],
                        "best_oos_params": args[9],
                    }
                )

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=False,
        )

        # Verify result
        assert isinstance(result, BackfillResult)
        assert result.processed == 1
        assert result.updated == 1
        assert result.errors == []

        # Verify update was called with correct values
        assert len(updates_called) == 1
        update = updates_called[0]

        assert update["tune_id"] == sample_tune_id
        assert update["regime_schema_version"] == "regime_v1_1"
        assert update["tag_ruleset_id"] == "default_v1"
        assert "uptrend" in update["regime_key"]
        assert "high_vol" in update["regime_key"]
        assert "efficient" in update["regime_key"]
        assert len(update["regime_fingerprint"]) == 64  # SHA256 hex
        assert update["trend_tag"] == "uptrend"
        assert update["vol_tag"] == "high_vol"
        assert update["efficiency_tag"] == "efficient"
        assert update["best_oos_score"] == 1.5


class TestBackfillSkipsAlreadyPopulated:
    """Test that backfill skips tunes with regime_key already set."""

    @pytest.mark.asyncio
    async def test_backfill_skips_already_populated(
        self, mock_db_pool, sample_tune_id, sample_regime_snapshot
    ):
        """Tunes with regime_key are filtered by SQL query (WHERE regime_key IS NULL)."""
        # The query should filter out already populated tunes
        # Return empty since WHERE regime_key IS NULL excludes them
        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=[])

        updates_called = []

        async def capture_execute(query, *args):
            if "UPDATE" in query:
                updates_called.append(args)

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=False,
        )

        # Already-populated tunes are filtered at query level, so processed=0
        assert result.processed == 0
        assert result.updated == 0
        assert len(updates_called) == 0


class TestBackfillHandlesNoCompletedRuns:
    """Test handling tunes with no completed runs."""

    @pytest.mark.asyncio
    async def test_backfill_handles_no_completed_runs(
        self, mock_db_pool, sample_tune_id
    ):
        """Tune with no completed runs is skipped gracefully."""
        # Setup: tune without regime columns
        tune_row = make_tune_row(tune_id=sample_tune_id, status="completed")

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=[tune_row])
        conn.fetchrow = AsyncMock(return_value=None)  # No best run found

        updates_called = []

        async def capture_execute(query, *args):
            if "UPDATE" in query:
                updates_called.append(args)

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=False,
        )

        assert result.processed == 1
        assert result.skipped_no_runs == 1
        assert result.updated == 0
        assert len(updates_called) == 0


class TestBackfillHandlesMissingRegimeOos:
    """Test handling runs without regime snapshot."""

    @pytest.mark.asyncio
    async def test_backfill_handles_missing_regime_oos(
        self, mock_db_pool, sample_tune_id
    ):
        """Run without regime snapshot in metrics_oos is skipped."""
        tune_row = make_tune_row(tune_id=sample_tune_id, status="completed")

        # Best run WITHOUT regime snapshot
        best_run = make_tune_run_row(
            tune_id=sample_tune_id,
            score_oos=1.5,
            metrics_oos={"sharpe": 1.5, "return_pct": 10.0},  # No 'regime' key
            params={"lookback": 20},
        )

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=[tune_row])
        conn.fetchrow = AsyncMock(return_value=best_run)

        updates_called = []

        async def capture_execute(query, *args):
            if "UPDATE" in query:
                updates_called.append(args)

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=False,
        )

        assert result.processed == 1
        assert result.skipped_no_regime == 1
        assert result.updated == 0
        assert len(updates_called) == 0


class TestBackfillHandlesMissingMetricsOos:
    """Test handling runs with regime but no metrics."""

    @pytest.mark.asyncio
    async def test_backfill_handles_missing_metrics_oos(
        self, mock_db_pool, sample_tune_id, sample_regime_snapshot
    ):
        """Still populates regime even without other metrics (best_oos_score may be None)."""
        tune_row = make_tune_row(tune_id=sample_tune_id, status="completed")

        # Best run WITH regime but minimal metrics_oos (no sharpe)
        best_run = make_tune_run_row(
            tune_id=sample_tune_id,
            score_oos=None,  # No score
            metrics_oos={"regime": sample_regime_snapshot},  # Only regime, no sharpe
            params={"lookback": 20},
        )

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=[tune_row])
        conn.fetchrow = AsyncMock(return_value=best_run)

        updates_called = []

        async def capture_execute(query, *args):
            if "UPDATE backtest_tunes" in query:
                updates_called.append(
                    {
                        "tune_id": args[0],
                        "regime_key": args[3],
                        "best_oos_score": args[8],
                    }
                )

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=False,
        )

        assert result.processed == 1
        assert result.updated == 1

        # Regime columns populated even without score
        assert len(updates_called) == 1
        assert updates_called[0]["regime_key"] is not None
        assert updates_called[0]["best_oos_score"] is None  # Gracefully handles None


class TestBackfillDryRunNoChanges:
    """Test dry run mode doesn't modify DB."""

    @pytest.mark.asyncio
    async def test_backfill_dry_run_no_changes(
        self, mock_db_pool, sample_tune_id, sample_regime_snapshot
    ):
        """Dry run mode doesn't write to database."""
        tune_row = make_tune_row(tune_id=sample_tune_id, status="completed")

        best_run = make_tune_run_row(
            tune_id=sample_tune_id,
            score_oos=1.5,
            metrics_oos={"regime": sample_regime_snapshot, "sharpe": 1.5},
            params={"lookback": 20},
        )

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=[tune_row])
        conn.fetchrow = AsyncMock(return_value=best_run)

        execute_called = []

        async def capture_execute(query, *args):
            execute_called.append(query)

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=True,
        )

        assert result.dry_run is True
        assert result.processed == 1
        assert result.updated == 0  # No actual updates in dry run
        assert result.would_update == 1  # Would have updated 1

        # No UPDATE queries should be executed
        update_queries = [q for q in execute_called if "UPDATE" in q]
        assert len(update_queries) == 0


class TestBackfillRespectsLimit:
    """Test limit parameter works correctly."""

    @pytest.mark.asyncio
    async def test_backfill_respects_limit(self, mock_db_pool, sample_regime_snapshot):
        """Limit parameter restricts number of tunes processed."""
        # Create multiple tunes
        tune_ids = [uuid4() for _ in range(5)]
        tune_rows = [make_tune_row(tune_id=tid, status="completed") for tid in tune_ids]

        def make_best_run_for(tune_id):
            return make_tune_run_row(
                tune_id=tune_id,
                score_oos=1.5,
                metrics_oos={"regime": sample_regime_snapshot, "sharpe": 1.5},
                params={"lookback": 20},
            )

        # Only return first 2 tunes (simulating LIMIT 2 in SQL)
        limited_tune_rows = tune_rows[:2]

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=limited_tune_rows)
        conn.fetchrow = AsyncMock(
            side_effect=[
                make_best_run_for(tune_ids[0]),
                make_best_run_for(tune_ids[1]),
            ]
        )

        updates_called = []

        async def capture_execute(query, *args):
            if "UPDATE backtest_tunes" in query:
                updates_called.append(args[0])  # tune_id

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            limit=2,
            dry_run=False,
        )

        assert result.processed == 2  # Limited to 2
        assert result.updated == 2


class TestBackfillReturnsStats:
    """Test backfill returns correct statistics."""

    @pytest.mark.asyncio
    async def test_backfill_returns_stats(self, mock_db_pool, sample_regime_snapshot):
        """Returns processed/skipped/errors counts."""
        tune_ids = [uuid4() for _ in range(4)]

        # Mix of scenarios:
        # 1. Successfully updated
        # 2. No completed runs
        # 3. Missing regime
        # 4. Successfully updated

        tune_rows = [make_tune_row(tune_id=tid, status="completed") for tid in tune_ids]

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=tune_rows)

        # Create side effects for each tune's best run query
        conn.fetchrow = AsyncMock(
            side_effect=[
                # Tune 1: has regime
                make_tune_run_row(
                    tune_id=tune_ids[0],
                    score_oos=1.5,
                    metrics_oos={"regime": sample_regime_snapshot, "sharpe": 1.5},
                    params={"lookback": 20},
                ),
                # Tune 2: no runs
                None,
                # Tune 3: no regime in metrics
                make_tune_run_row(
                    tune_id=tune_ids[2],
                    score_oos=1.0,
                    metrics_oos={"sharpe": 1.0},  # No regime
                    params={"lookback": 20},
                ),
                # Tune 4: has regime
                make_tune_run_row(
                    tune_id=tune_ids[3],
                    score_oos=2.0,
                    metrics_oos={"regime": sample_regime_snapshot, "sharpe": 2.0},
                    params={"lookback": 30},
                ),
            ]
        )

        updates_called = []

        async def capture_execute(query, *args):
            if "UPDATE backtest_tunes" in query:
                updates_called.append(args[0])

        conn.execute = capture_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=False,
        )

        assert result.processed == 4
        assert result.updated == 2  # Tunes 1 and 4
        assert result.skipped_no_runs == 1  # Tune 2
        assert result.skipped_no_regime == 1  # Tune 3
        assert result.errors == []


class TestBackfillHandlesErrors:
    """Test backfill handles errors gracefully."""

    @pytest.mark.asyncio
    async def test_backfill_handles_db_errors(
        self, mock_db_pool, sample_tune_id, sample_regime_snapshot
    ):
        """Database errors are captured and reported."""
        tune_row = make_tune_row(tune_id=sample_tune_id, status="completed")

        best_run = make_tune_run_row(
            tune_id=sample_tune_id,
            score_oos=1.5,
            metrics_oos={"regime": sample_regime_snapshot, "sharpe": 1.5},
            params={"lookback": 20},
        )

        conn = mock_db_pool._mock_conn
        conn.fetch = AsyncMock(return_value=[tune_row])
        conn.fetchrow = AsyncMock(return_value=best_run)

        async def failing_execute(query, *args):
            if "UPDATE" in query:
                raise Exception("Database connection lost")

        conn.execute = failing_execute

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            dry_run=False,
        )

        assert result.processed == 1
        assert result.updated == 0
        assert len(result.errors) == 1
        assert "Database connection lost" in result.errors[0]


class TestRegimeKeyComputation:
    """Test regime key is computed correctly from snapshot."""

    def test_regime_key_format(self):
        """Regime key has correct format from snapshot tags."""
        from app.services.kb.regime import (
            compute_regime_key,
            compute_regime_fingerprint,
            extract_regime_tags_for_attribution,
        )

        # Verify regime functions work as expected
        snapshot = RegimeSnapshot(
            schema_version="regime_v1_1",
            regime_tags=["downtrend", "low_vol", "noisy"],
        )

        key = compute_regime_key(snapshot)
        assert "regime_v1_1" in key
        assert "default_v1" in key
        assert "downtrend" in key
        assert "low_vol" in key
        assert "noisy" in key

        fingerprint = compute_regime_fingerprint(key)
        assert len(fingerprint) == 64

        trend, vol, eff = extract_regime_tags_for_attribution(snapshot)
        assert trend == "downtrend"
        assert vol == "low_vol"
        assert eff == "noisy"


class TestBackfillWithWorkspaceFilter:
    """Test backfill with workspace ID filter."""

    @pytest.mark.asyncio
    async def test_backfill_with_workspace_filter(
        self, mock_db_pool, sample_tune_id, sample_regime_snapshot
    ):
        """Workspace filter is applied to query."""
        workspace_id = uuid4()

        # Track queries to verify filter
        queries_executed = []

        conn = mock_db_pool._mock_conn

        original_fetch = conn.fetch

        async def tracking_fetch(query, *args, **kwargs):
            queries_executed.append((query, args))
            return []

        conn.fetch = tracking_fetch

        result = await run_tune_regime_backfill(
            db_pool=mock_db_pool,
            workspace_id=workspace_id,
            dry_run=False,
        )

        assert result.processed == 0
        # Verify workspace filter was in query
        assert len(queries_executed) > 0
        query, args = queries_executed[0]
        assert "workspace_id" in query
        assert workspace_id in args
