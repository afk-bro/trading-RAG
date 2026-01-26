"""Unit tests for intel runner module."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pandas as pd
import pytest

from app.services.intel.runner import IntelRunner, compute_and_store_snapshot


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool():
    """Create mock database pool."""
    pool = MagicMock()
    conn = AsyncMock()

    # Setup acquire as async context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire.return_value = cm

    return pool, conn


@pytest.fixture
def sample_version_row():
    """Sample version data from database."""
    return {
        "id": uuid4(),
        "strategy_id": uuid4(),
        "strategy_entity_id": uuid4(),
        "workspace_id": uuid4(),
        "config_snapshot": json.dumps({"symbol": "BTC/USDT", "timeframe": "1h"}),
        "regime_awareness": json.dumps({"good_regimes": ["trend_low_vol"]}),
        "state": "active",
    }


@pytest.fixture
def sample_backtest_summary():
    """Sample backtest summary from database."""
    return {
        "sharpe": 1.5,
        "return_pct": 25.0,
        "max_drawdown_pct": 12.0,
        "trades": 45,
        "win_rate": 0.55,
    }


@pytest.fixture
def sample_snapshot_row():
    """Sample snapshot row for repository mock."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "strategy_version_id": uuid4(),
        "as_of_ts": datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        "computed_at": datetime(2024, 1, 15, 10, 0, 5, tzinfo=timezone.utc),
        "regime": "trend_low_vol",
        "confidence_score": 0.72,
        "confidence_components": {"performance": 0.8, "drawdown": 0.7},
        "features": {},
        "explain": {},
        "engine_version": "test",
        "inputs_hash": "a" * 64,
        "run_id": None,
    }


@pytest.fixture
def sample_wfo_best_candidate():
    """Sample WFO best candidate metrics."""
    return {
        "params": {"lookback": 20, "threshold": 0.5},
        "params_hash": "abc123",
        "mean_oos": 1.25,  # Out-of-sample sharpe
        "median_oos": 1.1,
        "worst_fold_oos": 0.3,
        "stddev_oos": 0.4,
        "pct_top_k": 0.85,
        "fold_count": 5,
        "total_folds": 6,
        "coverage": 0.833,
        "regime_tags": ["trend_low_vol", "range_mid_vol"],
    }


@pytest.fixture
def sample_wfo_config():
    """Sample WFO configuration."""
    return {
        "train_days": 90,
        "test_days": 30,
        "step_days": 15,
        "min_folds": 3,
        "leaderboard_top_k": 10,
        "allow_partial": False,
    }


# =============================================================================
# IntelRunner Tests
# =============================================================================


class TestIntelRunnerInit:
    """Tests for IntelRunner initialization."""

    def test_init_basic(self, mock_pool):
        """Test basic initialization."""
        pool, _ = mock_pool
        runner = IntelRunner(pool)

        assert runner._pool is pool
        assert runner._ohlcv_provider is None
        assert runner._engine_version == "intel_runner_v0.2"

    def test_init_with_provider(self, mock_pool):
        """Test initialization with OHLCV provider."""
        pool, _ = mock_pool
        mock_provider = MagicMock()

        runner = IntelRunner(pool, ohlcv_provider=mock_provider)

        assert runner._ohlcv_provider is mock_provider

    def test_init_custom_engine_version(self, mock_pool):
        """Test initialization with custom engine version."""
        pool, _ = mock_pool
        runner = IntelRunner(pool, engine_version="custom_v1.0")

        assert runner._engine_version == "custom_v1.0"


class TestRunForVersion:
    """Tests for run_for_version()."""

    @pytest.mark.asyncio
    async def test_version_not_found_returns_none(self, mock_pool):
        """Test that missing version returns None."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        runner = IntelRunner(pool)
        result = await runner.run_for_version(
            version_id=uuid4(),
            as_of_ts=datetime.now(timezone.utc),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_creates_snapshot(
        self,
        mock_pool,
        sample_version_row,
        sample_backtest_summary,
        sample_snapshot_row,
    ):
        """Test that snapshot is created successfully."""
        pool, conn = mock_pool

        version_id = sample_version_row["id"]
        workspace_id = sample_version_row["workspace_id"]

        # Mock version fetch
        conn.fetchrow = AsyncMock(
            side_effect=[
                sample_version_row,  # version data
                {
                    "summary": json.dumps(sample_backtest_summary)
                },  # backtest (version-linked fails)
                None,  # get_latest_snapshot returns None
            ]
        )
        conn.fetch = AsyncMock(return_value=[])

        # Mock the intel repository
        with patch.object(
            IntelRunner,
            "_fetch_version_data",
            new_callable=AsyncMock,
            return_value={
                **sample_version_row,
                "config_snapshot": {"symbol": "BTC/USDT"},
                "regime_awareness": {"good_regimes": ["trend_low_vol"]},
            },
        ), patch.object(
            IntelRunner,
            "_fetch_backtest_metrics",
            new_callable=AsyncMock,
            return_value=sample_backtest_summary,
        ), patch.object(
            IntelRunner,
            "_fetch_wfo_metrics",
            new_callable=AsyncMock,
            return_value=None,
        ):
            runner = IntelRunner(pool)

            # Mock the repository insert
            with patch.object(
                runner._intel_repo,
                "insert_snapshot",
                new_callable=AsyncMock,
            ) as mock_insert, patch.object(
                runner._intel_repo,
                "get_latest_snapshot",
                new_callable=AsyncMock,
                return_value=None,
            ):
                from app.repositories.strategy_intel import IntelSnapshot

                mock_insert.return_value = IntelSnapshot(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    strategy_version_id=version_id,
                    as_of_ts=datetime.now(timezone.utc),
                    computed_at=datetime.now(timezone.utc),
                    regime="unknown",
                    confidence_score=0.5,
                    confidence_components={},
                    features={},
                    explain={},
                    engine_version="test",
                    inputs_hash="a" * 64,
                    run_id=None,
                )

                result = await runner.run_for_version(
                    version_id=version_id,
                    as_of_ts=datetime.now(timezone.utc),
                    workspace_id=workspace_id,
                )

                assert result is not None
                mock_insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_dedupe_skips_same_hash(
        self, mock_pool, sample_version_row, sample_backtest_summary
    ):
        """Test that deduplication skips when inputs_hash matches."""
        pool, conn = mock_pool

        version_id = sample_version_row["id"]
        workspace_id = sample_version_row["workspace_id"]

        # Use fixed timestamp for deterministic hashing
        fixed_ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

        # Pre-compute what hash will be generated
        from app.services.intel.confidence import compute_confidence, ConfidenceContext

        ctx = ConfidenceContext(
            version_id=version_id,
            as_of_ts=fixed_ts,
            backtest_metrics=sample_backtest_summary,
        )
        expected_result = compute_confidence(ctx)

        with patch.object(
            IntelRunner,
            "_fetch_version_data",
            new_callable=AsyncMock,
            return_value={
                **sample_version_row,
                "config_snapshot": {"symbol": "BTC/USDT"},
                "regime_awareness": None,
            },
        ), patch.object(
            IntelRunner,
            "_fetch_backtest_metrics",
            new_callable=AsyncMock,
            return_value=sample_backtest_summary,
        ), patch.object(
            IntelRunner,
            "_fetch_wfo_metrics",
            new_callable=AsyncMock,
            return_value=None,
        ):
            runner = IntelRunner(pool)

            # Mock existing snapshot with same inputs_hash
            from app.repositories.strategy_intel import IntelSnapshot

            existing_snapshot = IntelSnapshot(
                id=uuid4(),
                workspace_id=workspace_id,
                strategy_version_id=version_id,
                as_of_ts=fixed_ts,
                computed_at=fixed_ts,
                regime="unknown",
                confidence_score=0.5,
                confidence_components={},
                features={},
                explain={},
                engine_version="test",
                inputs_hash=expected_result.inputs_hash,  # Same hash
                run_id=None,
            )

            with patch.object(
                runner._intel_repo,
                "get_latest_snapshot",
                new_callable=AsyncMock,
                return_value=existing_snapshot,
            ), patch.object(
                runner._intel_repo,
                "insert_snapshot",
                new_callable=AsyncMock,
            ) as mock_insert:
                result = await runner.run_for_version(
                    version_id=version_id,
                    as_of_ts=fixed_ts,
                    workspace_id=workspace_id,
                    force=False,
                )

                # Should return None due to dedupe
                assert result is None
                mock_insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_bypasses_dedupe(
        self, mock_pool, sample_version_row, sample_backtest_summary
    ):
        """Test that force=True bypasses deduplication."""
        pool, conn = mock_pool

        version_id = sample_version_row["id"]
        workspace_id = sample_version_row["workspace_id"]

        with patch.object(
            IntelRunner,
            "_fetch_version_data",
            new_callable=AsyncMock,
            return_value={
                **sample_version_row,
                "config_snapshot": {"symbol": "BTC/USDT"},
                "regime_awareness": None,
            },
        ), patch.object(
            IntelRunner,
            "_fetch_backtest_metrics",
            new_callable=AsyncMock,
            return_value=sample_backtest_summary,
        ), patch.object(
            IntelRunner,
            "_fetch_wfo_metrics",
            new_callable=AsyncMock,
            return_value=None,
        ):
            runner = IntelRunner(pool)

            from app.repositories.strategy_intel import IntelSnapshot

            with patch.object(
                runner._intel_repo,
                "get_latest_snapshot",
                new_callable=AsyncMock,
                return_value=None,  # Would have same hash
            ), patch.object(
                runner._intel_repo,
                "insert_snapshot",
                new_callable=AsyncMock,
            ) as mock_insert:
                mock_insert.return_value = IntelSnapshot(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    strategy_version_id=version_id,
                    as_of_ts=datetime.now(timezone.utc),
                    computed_at=datetime.now(timezone.utc),
                    regime="unknown",
                    confidence_score=0.5,
                    confidence_components={},
                    features={},
                    explain={},
                    engine_version="test",
                    inputs_hash="a" * 64,
                    run_id=None,
                )

                result = await runner.run_for_version(
                    version_id=version_id,
                    as_of_ts=datetime.now(timezone.utc),
                    workspace_id=workspace_id,
                    force=True,
                )

                # Should create snapshot despite potential dedupe
                assert result is not None
                mock_insert.assert_called_once()


class TestRunForWorkspaceActive:
    """Tests for run_for_workspace_active()."""

    @pytest.mark.asyncio
    async def test_no_active_versions_returns_empty(self, mock_pool):
        """Test that no active versions returns empty list."""
        pool, conn = mock_pool

        runner = IntelRunner(pool)

        with patch.object(
            runner,
            "_fetch_active_versions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await runner.run_for_workspace_active(
                workspace_id=uuid4(),
                as_of_ts=datetime.now(timezone.utc),
            )

            assert result == []

    @pytest.mark.asyncio
    async def test_processes_all_active_versions(
        self, mock_pool, sample_version_row, sample_backtest_summary
    ):
        """Test that all active versions are processed."""
        pool, conn = mock_pool
        workspace_id = uuid4()

        # Create multiple version data
        versions = [
            {**sample_version_row, "id": uuid4(), "workspace_id": workspace_id},
            {**sample_version_row, "id": uuid4(), "workspace_id": workspace_id},
            {**sample_version_row, "id": uuid4(), "workspace_id": workspace_id},
        ]

        runner = IntelRunner(pool)

        with patch.object(
            runner,
            "_fetch_active_versions",
            new_callable=AsyncMock,
            return_value=versions,
        ), patch.object(
            runner,
            "run_for_version",
            new_callable=AsyncMock,
        ) as mock_run:
            from app.repositories.strategy_intel import IntelSnapshot

            # Return snapshot for first two, None for third (dedupe)
            mock_run.side_effect = [
                IntelSnapshot(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    strategy_version_id=versions[0]["id"],
                    as_of_ts=datetime.now(timezone.utc),
                    computed_at=datetime.now(timezone.utc),
                    regime="trend_low_vol",
                    confidence_score=0.7,
                    confidence_components={},
                    features={},
                    explain={},
                    engine_version="test",
                    inputs_hash="a" * 64,
                    run_id=None,
                ),
                IntelSnapshot(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    strategy_version_id=versions[1]["id"],
                    as_of_ts=datetime.now(timezone.utc),
                    computed_at=datetime.now(timezone.utc),
                    regime="range_mid_vol",
                    confidence_score=0.5,
                    confidence_components={},
                    features={},
                    explain={},
                    engine_version="test",
                    inputs_hash="b" * 64,
                    run_id=None,
                ),
                None,  # Third deduplicated
            ]

            result = await runner.run_for_workspace_active(
                workspace_id=workspace_id,
                as_of_ts=datetime.now(timezone.utc),
            )

            assert len(result) == 2
            assert mock_run.call_count == 3

    @pytest.mark.asyncio
    async def test_continues_on_error(self, mock_pool, sample_version_row):
        """Test that errors in one version don't stop others."""
        pool, conn = mock_pool
        workspace_id = uuid4()

        versions = [
            {**sample_version_row, "id": uuid4(), "workspace_id": workspace_id},
            {**sample_version_row, "id": uuid4(), "workspace_id": workspace_id},
        ]

        runner = IntelRunner(pool)

        with patch.object(
            runner,
            "_fetch_active_versions",
            new_callable=AsyncMock,
            return_value=versions,
        ), patch.object(
            runner,
            "run_for_version",
            new_callable=AsyncMock,
        ) as mock_run:
            from app.repositories.strategy_intel import IntelSnapshot

            # First raises error, second succeeds
            mock_run.side_effect = [
                Exception("Database error"),
                IntelSnapshot(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    strategy_version_id=versions[1]["id"],
                    as_of_ts=datetime.now(timezone.utc),
                    computed_at=datetime.now(timezone.utc),
                    regime="trend_low_vol",
                    confidence_score=0.7,
                    confidence_components={},
                    features={},
                    explain={},
                    engine_version="test",
                    inputs_hash="a" * 64,
                    run_id=None,
                ),
            ]

            result = await runner.run_for_workspace_active(
                workspace_id=workspace_id,
                as_of_ts=datetime.now(timezone.utc),
            )

            # Should have one result despite error
            assert len(result) == 1
            assert mock_run.call_count == 2


class TestFetchHelpers:
    """Tests for data fetching helper methods."""

    @pytest.mark.asyncio
    async def test_fetch_version_data_parses_json(self, mock_pool, sample_version_row):
        """Test that JSON fields are parsed correctly."""
        pool, conn = mock_pool

        conn.fetchrow = AsyncMock(return_value=sample_version_row)

        runner = IntelRunner(pool)
        result = await runner._fetch_version_data(sample_version_row["id"])

        assert result is not None
        assert isinstance(result["config_snapshot"], dict)
        assert isinstance(result["regime_awareness"], dict)

    @pytest.mark.asyncio
    async def test_fetch_version_data_returns_none_when_not_found(self, mock_pool):
        """Test that None is returned when version not found."""
        pool, conn = mock_pool
        conn.fetchrow = AsyncMock(return_value=None)

        runner = IntelRunner(pool)
        result = await runner._fetch_version_data(uuid4())

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_backtest_metrics_maps_fields(
        self, mock_pool, sample_backtest_summary
    ):
        """Test that backtest summary fields are mapped correctly."""
        pool, conn = mock_pool

        conn.fetchrow = AsyncMock(
            return_value={"summary": json.dumps(sample_backtest_summary)}
        )

        runner = IntelRunner(pool)
        result = await runner._fetch_backtest_metrics(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
            version_id=uuid4(),
        )

        assert result is not None
        assert result["sharpe"] == 1.5
        assert result["return_pct"] == 25.0
        assert result["max_drawdown_pct"] == 12.0
        assert result["trades"] == 45

    def test_get_latest_candle_ts_from_index(self, mock_pool):
        """Test extracting timestamp from DataFrame index."""
        pool, _ = mock_pool
        runner = IntelRunner(pool)

        dates = pd.date_range(start="2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame({"close": range(10)}, index=dates)

        ts = runner._get_latest_candle_ts(df)
        assert ts is not None

    def test_get_latest_candle_ts_none_for_empty(self, mock_pool):
        """Test that None is returned for empty DataFrame."""
        pool, _ = mock_pool
        runner = IntelRunner(pool)

        assert runner._get_latest_candle_ts(None) is None
        assert runner._get_latest_candle_ts(pd.DataFrame()) is None


class TestWFOMetrics:
    """Tests for WFO metrics fetching and mapping."""

    @pytest.mark.asyncio
    async def test_fetch_wfo_metrics_success(
        self, mock_pool, sample_wfo_best_candidate, sample_wfo_config
    ):
        """Test successful WFO metrics fetch."""
        pool, conn = mock_pool

        conn.fetchrow = AsyncMock(
            return_value={
                "best_candidate": sample_wfo_best_candidate,
                "wfo_config": sample_wfo_config,
            }
        )

        runner = IntelRunner(pool)
        result = await runner._fetch_wfo_metrics(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
        )

        assert result is not None
        assert result["oos_sharpe"] == 1.25
        assert result["num_folds"] == 5
        assert result["fold_variance"] is not None
        # fold_variance = |0.4 / 1.25| = 0.32, capped at 1.0
        assert 0 <= result["fold_variance"] <= 1.0

    @pytest.mark.asyncio
    async def test_fetch_wfo_metrics_returns_none_when_no_wfo(self, mock_pool):
        """Test that None is returned when no WFO runs exist."""
        pool, conn = mock_pool

        conn.fetchrow = AsyncMock(return_value=None)

        runner = IntelRunner(pool)
        result = await runner._fetch_wfo_metrics(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_wfo_metrics_returns_none_without_strategy_entity(
        self, mock_pool
    ):
        """Test that None is returned when strategy_entity_id is None."""
        pool, _ = mock_pool

        runner = IntelRunner(pool)
        result = await runner._fetch_wfo_metrics(
            workspace_id=uuid4(),
            strategy_entity_id=None,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_wfo_metrics_parses_json_string(
        self, mock_pool, sample_wfo_best_candidate, sample_wfo_config
    ):
        """Test that JSON string fields are parsed correctly."""
        pool, conn = mock_pool

        # Return as JSON strings (simulating raw JSONB that wasn't auto-parsed)
        conn.fetchrow = AsyncMock(
            return_value={
                "best_candidate": json.dumps(sample_wfo_best_candidate),
                "wfo_config": json.dumps(sample_wfo_config),
            }
        )

        runner = IntelRunner(pool)
        result = await runner._fetch_wfo_metrics(
            workspace_id=uuid4(),
            strategy_entity_id=uuid4(),
        )

        assert result is not None
        assert result["oos_sharpe"] == 1.25

    def test_map_wfo_to_confidence_metrics_basic(
        self, mock_pool, sample_wfo_best_candidate
    ):
        """Test basic WFO metrics mapping."""
        pool, _ = mock_pool
        runner = IntelRunner(pool)

        result = runner._map_wfo_to_confidence_metrics(sample_wfo_best_candidate)

        assert result["oos_sharpe"] == 1.25
        assert result["num_folds"] == 5
        assert result["oos_return_pct"] is None  # Not available in WFO candidate
        assert result["max_drawdown_pct"] is None  # Not available

    def test_map_wfo_to_confidence_metrics_fold_variance_calculation(self, mock_pool):
        """Test fold variance calculation from stddev and mean."""
        pool, _ = mock_pool
        runner = IntelRunner(pool)

        # Test case: mean=2.0, stddev=0.5 -> CV = 0.25
        candidate = {"mean_oos": 2.0, "stddev_oos": 0.5, "fold_count": 4}
        result = runner._map_wfo_to_confidence_metrics(candidate)
        assert result["fold_variance"] == pytest.approx(0.25, rel=0.01)

        # Test case: high variance (stddev > mean) -> capped at 1.0
        candidate = {"mean_oos": 0.5, "stddev_oos": 1.0, "fold_count": 4}
        result = runner._map_wfo_to_confidence_metrics(candidate)
        assert result["fold_variance"] == 1.0

        # Test case: near-zero mean with variance -> high instability
        candidate = {"mean_oos": 0.0005, "stddev_oos": 0.5, "fold_count": 4}
        result = runner._map_wfo_to_confidence_metrics(candidate)
        assert result["fold_variance"] == 1.0

    def test_map_wfo_to_confidence_metrics_empty_candidate(self, mock_pool):
        """Test mapping with empty or None candidate."""
        pool, _ = mock_pool
        runner = IntelRunner(pool)

        assert runner._map_wfo_to_confidence_metrics(None) is None
        assert runner._map_wfo_to_confidence_metrics({}) is None

    def test_map_wfo_to_confidence_metrics_negative_sharpe(self, mock_pool):
        """Test mapping with negative sharpe (poor OOS performance)."""
        pool, _ = mock_pool
        runner = IntelRunner(pool)

        candidate = {"mean_oos": -0.5, "stddev_oos": 0.3, "fold_count": 3}
        result = runner._map_wfo_to_confidence_metrics(candidate)

        assert result["oos_sharpe"] == -0.5
        # CV = |0.3 / -0.5| = 0.6
        assert result["fold_variance"] == pytest.approx(0.6, rel=0.01)


class TestWFOIntegration:
    """Tests for WFO metrics integration in run_for_version."""

    @pytest.mark.asyncio
    async def test_wfo_metrics_used_in_context(
        self,
        mock_pool,
        sample_version_row,
        sample_backtest_summary,
        sample_wfo_best_candidate,
        sample_wfo_config,
    ):
        """Test that WFO metrics are passed to confidence context."""
        pool, conn = mock_pool

        version_id = sample_version_row["id"]
        workspace_id = sample_version_row["workspace_id"]

        with patch.object(
            IntelRunner,
            "_fetch_version_data",
            new_callable=AsyncMock,
            return_value={
                **sample_version_row,
                "config_snapshot": {"symbol": "BTC/USDT"},
                "regime_awareness": None,
            },
        ), patch.object(
            IntelRunner,
            "_fetch_backtest_metrics",
            new_callable=AsyncMock,
            return_value=sample_backtest_summary,
        ), patch.object(
            IntelRunner,
            "_fetch_wfo_metrics",
            new_callable=AsyncMock,
            return_value={
                "oos_sharpe": 1.25,
                "oos_return_pct": None,
                "fold_variance": 0.32,
                "num_folds": 5,
                "max_drawdown_pct": None,
            },
        ):
            runner = IntelRunner(pool)

            with patch.object(
                runner._intel_repo,
                "get_latest_snapshot",
                new_callable=AsyncMock,
                return_value=None,
            ), patch.object(
                runner._intel_repo,
                "insert_snapshot",
                new_callable=AsyncMock,
            ) as mock_insert:
                from app.repositories.strategy_intel import IntelSnapshot

                mock_insert.return_value = IntelSnapshot(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    strategy_version_id=version_id,
                    as_of_ts=datetime.now(timezone.utc),
                    computed_at=datetime.now(timezone.utc),
                    regime="unknown",
                    confidence_score=0.65,
                    confidence_components={},
                    features={"metrics_source": "wfo"},
                    explain={},
                    engine_version="intel_runner_v0.2",
                    inputs_hash="a" * 64,
                    run_id=None,
                )

                result = await runner.run_for_version(
                    version_id=version_id,
                    as_of_ts=datetime.now(timezone.utc),
                    workspace_id=workspace_id,
                )

                assert result is not None
                # Verify insert was called with WFO-influenced features
                call_kwargs = mock_insert.call_args[1]
                assert "features" in call_kwargs
                assert call_kwargs["features"]["metrics_source"] == "wfo"


class TestConvenienceFunction:
    """Tests for compute_and_store_snapshot convenience function."""

    @pytest.mark.asyncio
    async def test_creates_runner_and_calls(
        self, mock_pool, sample_version_row, sample_backtest_summary
    ):
        """Test that convenience function works correctly."""
        pool, conn = mock_pool

        version_id = uuid4()
        workspace_id = uuid4()

        with patch("app.services.intel.runner.IntelRunner") as MockRunner:
            mock_runner_instance = AsyncMock()
            mock_runner_instance.run_for_version = AsyncMock(return_value=None)
            MockRunner.return_value = mock_runner_instance

            await compute_and_store_snapshot(
                pool=pool,
                version_id=version_id,
                as_of_ts=datetime.now(timezone.utc),
                workspace_id=workspace_id,
            )

            MockRunner.assert_called_once()
            mock_runner_instance.run_for_version.assert_called_once()
