"""Tests for cluster stats backfill job.

TDD tests following the v1.5 implementation plan:
1. Empty trials -> no stats
2. Single trial -> valid stats with zero variance
3. Multiple trials same key -> aggregated correctly (Welford's)
4. Multiple regime keys -> separate stats per key
5. Missing regime_features -> skipped
6. Dry-run mode -> no DB writes
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from app.jobs.backfill_cluster_stats import (
    run_cluster_stats_backfill,
    BackfillResult,
    aggregate_trial_features,
    WelfordAccumulator,
)
from app.repositories.cluster_stats import ClusterStats


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant async client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_db_pool():
    """Create mock database pool."""
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=AsyncMock())
    return pool


def make_trial_point(
    strategy_entity_id: str,
    timeframe: str,
    regime_key: str,
    regime_dims: dict,
    regime_features: dict | None,
    point_id: str | None = None,
) -> dict:
    """Helper to create a mock Qdrant point record."""
    return {
        "id": point_id or str(uuid4()),
        "payload": {
            "strategy_entity_id": strategy_entity_id,
            "timeframe": timeframe,
            "regime_key": regime_key,
            "regime_dims": regime_dims,
            "regime_features": regime_features,
            "is_valid": True,
        },
    }


# =============================================================================
# Welford Accumulator Unit Tests
# =============================================================================


class TestWelfordAccumulator:
    """Tests for Welford's online variance algorithm."""

    def test_empty_accumulator(self):
        """Empty accumulator has n=0."""
        acc = WelfordAccumulator()
        assert acc.n == 0
        assert acc.mean == {}
        assert acc.variance == {}

    def test_single_sample_zero_variance(self):
        """Single sample has zero variance."""
        acc = WelfordAccumulator()
        acc.update({"atr_pct": 0.02, "rsi": 50.0})

        assert acc.n == 1
        assert acc.mean["atr_pct"] == pytest.approx(0.02)
        assert acc.mean["rsi"] == pytest.approx(50.0)
        assert acc.variance["atr_pct"] == pytest.approx(0.0)
        assert acc.variance["rsi"] == pytest.approx(0.0)

    def test_two_samples_correct_variance(self):
        """Two samples compute correct variance."""
        acc = WelfordAccumulator()
        acc.update({"x": 0.0})
        acc.update({"x": 2.0})

        assert acc.n == 2
        assert acc.mean["x"] == pytest.approx(1.0)
        # Variance of [0, 2] with n-1 denominator = 2.0
        assert acc.variance["x"] == pytest.approx(2.0)

    def test_multiple_samples_numerically_stable(self):
        """Multiple samples remain numerically stable (Welford's advantage)."""
        acc = WelfordAccumulator()
        # Large values that would lose precision with naive variance
        base = 1e10
        values = [base + 1, base + 2, base + 3, base + 4, base + 5]

        for v in values:
            acc.update({"x": v})

        assert acc.n == 5
        # Mean of [1, 2, 3, 4, 5] shifted by base
        assert acc.mean["x"] == pytest.approx(base + 3.0, rel=1e-9)
        # Variance of [1, 2, 3, 4, 5] = 2.5 (sample variance)
        assert acc.variance["x"] == pytest.approx(2.5, rel=1e-6)

    def test_min_max_tracking(self):
        """Min and max are tracked correctly."""
        acc = WelfordAccumulator()
        acc.update({"x": 5.0})
        acc.update({"x": 1.0})
        acc.update({"x": 10.0})
        acc.update({"x": 3.0})

        assert acc.min["x"] == pytest.approx(1.0)
        assert acc.max["x"] == pytest.approx(10.0)

    def test_multiple_features_independent(self):
        """Different features tracked independently."""
        acc = WelfordAccumulator()
        acc.update({"a": 1.0, "b": 100.0})
        acc.update({"a": 3.0, "b": 200.0})

        assert acc.mean["a"] == pytest.approx(2.0)
        assert acc.mean["b"] == pytest.approx(150.0)
        assert acc.variance["a"] == pytest.approx(2.0)
        assert acc.variance["b"] == pytest.approx(5000.0)

    def test_missing_feature_in_update_ignored(self):
        """Missing features in an update don't affect other features."""
        acc = WelfordAccumulator()
        acc.update({"a": 1.0, "b": 10.0})
        acc.update({"a": 3.0})  # b missing

        assert acc.n == 2
        # a updated twice
        assert acc.mean["a"] == pytest.approx(2.0)
        # b only has one sample (from first update)
        # Note: this tests that we handle sparse features


# =============================================================================
# Aggregation Unit Tests
# =============================================================================


class TestAggregateTrialFeatures:
    """Tests for aggregate_trial_features helper."""

    def test_empty_trials_returns_empty_dict(self):
        """Empty trial list returns empty aggregation dict."""
        result = aggregate_trial_features([])
        assert result == {}

    def test_groups_by_key(self):
        """Trials grouped by (strategy_entity_id, timeframe, regime_key)."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.02},
            ),
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=low_vol",  # Different key
                regime_dims={"trend": "uptrend", "vol": "low_vol"},
                regime_features={"atr_pct": 0.01},
            ),
        ]

        result = aggregate_trial_features(trials)

        # Should have 2 separate groups
        assert len(result) == 2

    def test_aggregates_features_within_group(self):
        """Features aggregated correctly within a group."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.01, "rsi": 40.0},
            ),
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.03, "rsi": 60.0},
            ),
        ]

        result = aggregate_trial_features(trials)

        assert len(result) == 1
        key = (strategy_id, "5m", "trend=uptrend|vol=high_vol")
        assert key in result

        stats = result[key]
        assert stats.n == 2
        assert stats.feature_mean["atr_pct"] == pytest.approx(0.02)
        assert stats.feature_mean["rsi"] == pytest.approx(50.0)

    def test_skips_trials_without_regime_features(self):
        """Trials without regime_features are skipped."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.02},
            ),
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features=None,  # Missing features
            ),
        ]

        result = aggregate_trial_features(trials)

        key = (strategy_id, "5m", "trend=uptrend|vol=high_vol")
        assert result[key].n == 1  # Only one trial counted

    def test_skips_trials_with_empty_regime_features(self):
        """Trials with empty regime_features dict are skipped."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={},  # Empty features
            ),
        ]

        result = aggregate_trial_features(trials)

        # No valid trials, no stats
        assert len(result) == 0


# =============================================================================
# Backfill Job Integration Tests
# =============================================================================


class TestBackfillJobEmptyTrials:
    """Tests for backfill job with empty trials."""

    @pytest.mark.asyncio
    async def test_empty_collection_returns_zero_stats(
        self, mock_qdrant_client, mock_db_pool
    ):
        """Empty Qdrant collection results in zero stats written."""
        # Mock scroll returning empty
        mock_qdrant_client.scroll = AsyncMock(return_value=([], None))

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=False,
        )

        assert isinstance(result, BackfillResult)
        assert result.trials_processed == 0
        assert result.stats_written == 0
        assert result.errors == []


class TestBackfillJobSingleTrial:
    """Tests for backfill job with single trial."""

    @pytest.mark.asyncio
    async def test_single_trial_creates_stats_with_zero_variance(
        self, mock_qdrant_client, mock_db_pool
    ):
        """Single trial creates stats with n=1 and zero variance."""
        strategy_id = str(uuid4())
        trial = make_trial_point(
            strategy_entity_id=strategy_id,
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            regime_dims={"trend": "uptrend", "vol": "high_vol"},
            regime_features={"atr_pct": 0.02, "rsi": 50.0},
        )

        # Mock scroll returning single trial
        mock_point = MagicMock()
        mock_point.id = trial["id"]
        mock_point.payload = trial["payload"]
        mock_qdrant_client.scroll = AsyncMock(return_value=([mock_point], None))

        # Track upserted stats
        upserted_stats = []

        async def mock_execute(query, *args):
            # Capture the ClusterStats arguments
            upserted_stats.append({
                "strategy_entity_id": args[0],
                "timeframe": args[1],
                "regime_key": args[2],
                "n": args[4],
                "feature_mean": args[6],
                "feature_var": args[7],
            })

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=False,
        )

        assert result.trials_processed == 1
        assert result.stats_written == 1
        assert len(upserted_stats) == 1

        stats = upserted_stats[0]
        assert stats["n"] == 1
        # Zero variance for single sample
        import json
        var = json.loads(stats["feature_var"])
        assert var["atr_pct"] == pytest.approx(0.0)
        assert var["rsi"] == pytest.approx(0.0)


class TestBackfillJobMultipleTrials:
    """Tests for backfill job with multiple trials."""

    @pytest.mark.asyncio
    async def test_multiple_trials_same_key_aggregated(
        self, mock_qdrant_client, mock_db_pool
    ):
        """Multiple trials with same key are aggregated correctly."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.01},
            ),
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.03},
            ),
        ]

        mock_points = []
        for t in trials:
            point = MagicMock()
            point.id = t["id"]
            point.payload = t["payload"]
            mock_points.append(point)

        mock_qdrant_client.scroll = AsyncMock(return_value=(mock_points, None))

        upserted_stats = []

        async def mock_execute(query, *args):
            upserted_stats.append({
                "strategy_entity_id": args[0],
                "timeframe": args[1],
                "regime_key": args[2],
                "n": args[4],
                "feature_mean": args[6],
                "feature_var": args[7],
            })

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=False,
        )

        assert result.trials_processed == 2
        assert result.stats_written == 1

        import json
        stats = upserted_stats[0]
        assert stats["n"] == 2
        mean = json.loads(stats["feature_mean"])
        assert mean["atr_pct"] == pytest.approx(0.02)


class TestBackfillJobMultipleKeys:
    """Tests for backfill job with multiple regime keys."""

    @pytest.mark.asyncio
    async def test_multiple_regime_keys_separate_stats(
        self, mock_qdrant_client, mock_db_pool
    ):
        """Different regime keys produce separate stats entries."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.02},
            ),
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=downtrend|vol=low_vol",
                regime_dims={"trend": "downtrend", "vol": "low_vol"},
                regime_features={"atr_pct": 0.01},
            ),
        ]

        mock_points = []
        for t in trials:
            point = MagicMock()
            point.id = t["id"]
            point.payload = t["payload"]
            mock_points.append(point)

        mock_qdrant_client.scroll = AsyncMock(return_value=(mock_points, None))

        upserted_stats = []

        async def mock_execute(query, *args):
            upserted_stats.append({
                "regime_key": args[2],
                "n": args[4],
            })

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=False,
        )

        assert result.trials_processed == 2
        assert result.stats_written == 2

        regime_keys = [s["regime_key"] for s in upserted_stats]
        assert "trend=uptrend|vol=high_vol" in regime_keys
        assert "trend=downtrend|vol=low_vol" in regime_keys


class TestBackfillJobMissingFeatures:
    """Tests for backfill job handling missing regime_features."""

    @pytest.mark.asyncio
    async def test_missing_regime_features_skipped(
        self, mock_qdrant_client, mock_db_pool
    ):
        """Trials without regime_features are skipped gracefully."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.02},  # Valid
            ),
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features=None,  # Missing - should be skipped
            ),
        ]

        mock_points = []
        for t in trials:
            point = MagicMock()
            point.id = t["id"]
            point.payload = t["payload"]
            mock_points.append(point)

        mock_qdrant_client.scroll = AsyncMock(return_value=(mock_points, None))

        upserted_stats = []

        async def mock_execute(query, *args):
            upserted_stats.append({"n": args[4]})

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=False,
        )

        assert result.trials_processed == 2  # Both trials read
        assert result.trials_skipped == 1  # One skipped
        assert result.stats_written == 1
        assert upserted_stats[0]["n"] == 1  # Only one valid trial


class TestBackfillJobDryRun:
    """Tests for backfill job dry-run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_no_db_writes(self, mock_qdrant_client, mock_db_pool):
        """Dry run mode doesn't write to database."""
        strategy_id = str(uuid4())
        trial = make_trial_point(
            strategy_entity_id=strategy_id,
            timeframe="5m",
            regime_key="trend=uptrend|vol=high_vol",
            regime_dims={"trend": "uptrend", "vol": "high_vol"},
            regime_features={"atr_pct": 0.02},
        )

        mock_point = MagicMock()
        mock_point.id = trial["id"]
        mock_point.payload = trial["payload"]
        mock_qdrant_client.scroll = AsyncMock(return_value=([mock_point], None))

        conn = AsyncMock()
        conn.execute = AsyncMock()
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=True,  # Dry run!
        )

        assert result.trials_processed == 1
        assert result.stats_written == 0  # No writes in dry run
        assert result.dry_run is True
        # DB execute should NOT have been called
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_returns_would_write_count(
        self, mock_qdrant_client, mock_db_pool
    ):
        """Dry run returns count of stats that would be written."""
        strategy_id = str(uuid4())
        trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.02},
            ),
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=downtrend|vol=low_vol",
                regime_dims={"trend": "downtrend", "vol": "low_vol"},
                regime_features={"atr_pct": 0.01},
            ),
        ]

        mock_points = []
        for t in trials:
            point = MagicMock()
            point.id = t["id"]
            point.payload = t["payload"]
            mock_points.append(point)

        mock_qdrant_client.scroll = AsyncMock(return_value=(mock_points, None))

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=True,
        )

        assert result.trials_processed == 2
        assert result.stats_would_write == 2  # Would write 2 stats


class TestBackfillJobStrategyFilter:
    """Tests for backfill job strategy_entity_id filtering."""

    @pytest.mark.asyncio
    async def test_filters_by_strategy_entity_id(
        self, mock_qdrant_client, mock_db_pool
    ):
        """Backfill can filter by specific strategy_entity_id."""
        target_strategy_id = str(uuid4())
        other_strategy_id = str(uuid4())

        trials = [
            make_trial_point(
                strategy_entity_id=target_strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.02},
            ),
            make_trial_point(
                strategy_entity_id=other_strategy_id,  # Different strategy
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.01},
            ),
        ]

        mock_points = []
        for t in trials:
            point = MagicMock()
            point.id = t["id"]
            point.payload = t["payload"]
            mock_points.append(point)

        # When filtering, only target strategy returned by Qdrant
        async def mock_scroll(collection_name, scroll_filter=None, **kwargs):
            if scroll_filter:
                # Return only matching trials
                return ([mock_points[0]], None)
            return (mock_points, None)

        mock_qdrant_client.scroll = mock_scroll

        upserted_stats = []

        async def mock_execute(query, *args):
            upserted_stats.append({"strategy_entity_id": args[0]})

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            strategy_entity_id=target_strategy_id,
            dry_run=False,
        )

        assert result.stats_written == 1
        # Should only write stats for target strategy
        from uuid import UUID
        assert str(upserted_stats[0]["strategy_entity_id"]) == target_strategy_id


class TestBackfillJobPagination:
    """Tests for backfill job Qdrant pagination."""

    @pytest.mark.asyncio
    async def test_handles_pagination(self, mock_qdrant_client, mock_db_pool):
        """Backfill handles Qdrant scroll pagination correctly."""
        strategy_id = str(uuid4())

        # Create trials for two pages
        page1_trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.01},
            ),
        ]
        page2_trials = [
            make_trial_point(
                strategy_entity_id=strategy_id,
                timeframe="5m",
                regime_key="trend=uptrend|vol=high_vol",
                regime_dims={"trend": "uptrend", "vol": "high_vol"},
                regime_features={"atr_pct": 0.03},
            ),
        ]

        page1_points = []
        for t in page1_trials:
            point = MagicMock()
            point.id = t["id"]
            point.payload = t["payload"]
            page1_points.append(point)

        page2_points = []
        for t in page2_trials:
            point = MagicMock()
            point.id = t["id"]
            point.payload = t["payload"]
            page2_points.append(point)

        # Mock pagination: first call returns page1 with offset, second returns page2 with None
        call_count = 0
        async def mock_scroll(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (page1_points, "offset_token")  # More pages
            else:
                return (page2_points, None)  # Last page

        mock_qdrant_client.scroll = mock_scroll

        upserted_stats = []

        async def mock_execute(query, *args):
            upserted_stats.append({"n": args[4]})

        conn = AsyncMock()
        conn.execute = mock_execute
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock()

        result = await run_cluster_stats_backfill(
            qdrant_client=mock_qdrant_client,
            db_pool=mock_db_pool,
            collection_name="kb_trials",
            dry_run=False,
        )

        assert result.trials_processed == 2  # Both pages processed
        assert result.stats_written == 1
        # Both trials should be aggregated (n=2)
        import json
        assert upserted_stats[0]["n"] == 2
