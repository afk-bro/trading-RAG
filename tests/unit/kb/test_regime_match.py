"""
Unit tests for tiered regime matching.

Tests the matching ladder: Exact → Partial → Distance → Global
with proper fallback logic and metadata tracking.
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.services.kb.regime_match import (
    RegimeMatchTier,
    MatchDetail,
    SampleContext,
    RegimeMatchCandidate,
    TieredRegimeMatcher,
    extract_regime_features,
    compute_regime_distance,
)
from app.services.kb.types import RegimeSnapshot


# =============================================================================
# Test extract_regime_features
# =============================================================================


class TestExtractRegimeFeatures:
    """Test regime feature extraction from RegimeSnapshot."""

    def test_extracts_all_features(self):
        """Should extract all defined numeric features."""
        snapshot = RegimeSnapshot(
            atr_pct=0.03,
            std_pct=0.02,
            bb_width_pct=0.08,
            trend_strength=0.7,
            trend_dir=1,
            zscore=1.5,
            rsi=65.0,
            efficiency_ratio=0.6,
        )

        features = extract_regime_features(snapshot)

        assert features["atr_pct"] == 0.03
        assert features["std_pct"] == 0.02
        assert features["bb_width_pct"] == 0.08
        assert features["trend_strength"] == 0.7
        assert features["trend_dir"] == 1.0
        assert features["zscore"] == 1.5
        assert features["rsi"] == 65.0
        assert features["efficiency_ratio"] == 0.6

    def test_excludes_none_values(self):
        """Should exclude None values from features."""
        snapshot = RegimeSnapshot(
            atr_pct=0.03,
            std_pct=None,  # type: ignore
            trend_strength=0.7,
        )

        features = extract_regime_features(snapshot)

        assert "atr_pct" in features
        assert "std_pct" not in features
        assert "trend_strength" in features

    def test_excludes_nan_values(self):
        """Should exclude NaN values from features."""
        snapshot = RegimeSnapshot(
            atr_pct=0.03,
            std_pct=float("nan"),
            trend_strength=0.7,
        )

        features = extract_regime_features(snapshot)

        assert "atr_pct" in features
        assert "std_pct" not in features
        assert "trend_strength" in features


# =============================================================================
# Test compute_regime_distance
# =============================================================================


class TestComputeRegimeDistance:
    """Test Euclidean distance computation between regime feature vectors."""

    def test_identical_features_zero_distance(self):
        """Identical feature vectors should have zero distance."""
        features = {"atr_pct": 0.03, "trend_strength": 0.7, "rsi": 50.0}

        distance = compute_regime_distance(features, features)

        assert distance == 0.0

    def test_different_features_positive_distance(self):
        """Different features should have positive distance."""
        query = {"atr_pct": 0.03, "trend_strength": 0.7}
        candidate = {"atr_pct": 0.05, "trend_strength": 0.5}

        distance = compute_regime_distance(query, candidate)

        # sqrt((0.03-0.05)^2 + (0.7-0.5)^2) = sqrt(0.0004 + 0.04) = sqrt(0.0404)
        assert distance > 0
        assert abs(distance - 0.201) < 0.01

    def test_no_common_features_inf_distance(self):
        """No common features should return infinite distance."""
        query = {"atr_pct": 0.03}
        candidate = {"rsi": 50.0}

        distance = compute_regime_distance(query, candidate)

        assert distance == float("inf")

    def test_partial_overlap_uses_common_only(self):
        """Should only compare features present in both vectors."""
        query = {"atr_pct": 0.03, "trend_strength": 0.7, "rsi": 50.0}
        candidate = {"atr_pct": 0.03, "trend_strength": 0.7}  # rsi missing

        distance = compute_regime_distance(query, candidate)

        # Only compares atr_pct and trend_strength, both equal
        assert distance == 0.0


# =============================================================================
# Test RegimeMatchCandidate
# =============================================================================


class TestRegimeMatchCandidate:
    """Test RegimeMatchCandidate dataclass."""

    def test_candidate_with_match_detail(self):
        """Should create candidate with match detail."""
        candidate = RegimeMatchCandidate(
            tune_id=uuid4(),
            run_id=uuid4(),
            strategy_entity_id=uuid4(),
            best_params={"lookback": 20},
            best_oos_score=1.5,
            regime_key="trend=uptrend|vol=high_vol",
            trend_tag="uptrend",
            vol_tag="high_vol",
            match_detail=MatchDetail(
                tier=RegimeMatchTier.EXACT,
            ),
        )

        assert candidate.match_detail.tier == RegimeMatchTier.EXACT
        assert candidate.best_oos_score == 1.5


# =============================================================================
# Test TieredRegimeMatcher
# =============================================================================


class TestTieredRegimeMatcher:
    """Test tiered regime matching logic."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock())
        return pool

    @pytest.fixture
    def matcher(self, mock_pool):
        """Create matcher with mock pool."""
        return TieredRegimeMatcher(mock_pool)

    def _make_candidate(
        self,
        tune_id=None,
        regime_key="trend=uptrend|vol=high_vol",
        trend_tag="uptrend",
        vol_tag="high_vol",
        best_oos_score=1.0,
    ) -> dict[str, Any]:
        """Create a mock DB row for a candidate."""
        return {
            "tune_id": tune_id or uuid4(),
            "run_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "best_params": {"lookback": 20},
            "best_oos_score": best_oos_score,
            "regime_key": regime_key,
            "trend_tag": trend_tag,
            "vol_tag": vol_tag,
            "efficiency_tag": "efficient",
        }

    @pytest.mark.asyncio
    async def test_exact_match_sufficient_returns_exact_only(self, matcher, mock_pool):
        """When exact match has enough samples, should return only exact."""
        workspace_id = uuid4()
        regime_key = "trend=uptrend|vol=high_vol"

        # Mock DB to return 10 exact matches
        exact_rows = [self._make_candidate() for _ in range(10)]

        async def mock_fetch(query, *params):
            if "regime_key = $2" in query:  # Exact query
                return exact_rows
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        result = await matcher.match(
            workspace_id=workspace_id,
            regime_key=regime_key,
            trend_tag="uptrend",
            vol_tag="high_vol",
            min_samples=5,
            k=20,
        )

        assert len(result.candidates) == 10
        assert result.sample_context.tier_used == RegimeMatchTier.EXACT
        assert result.sample_context.exact_count == 10
        assert all(
            c.match_detail.tier == RegimeMatchTier.EXACT for c in result.candidates
        )

    @pytest.mark.asyncio
    async def test_exact_sparse_falls_back_to_partial(self, matcher, mock_pool):
        """When exact is sparse, should fall back to partial."""
        workspace_id = uuid4()

        # Mock DB: 2 exact, 5 partial_trend
        exact_rows = [self._make_candidate() for _ in range(2)]
        partial_trend_rows = [
            self._make_candidate(trend_tag="uptrend", vol_tag="low_vol")
            for _ in range(5)
        ]

        call_count = [0]

        async def mock_fetch(query, *params):
            call_count[0] += 1
            if "regime_key = $2" in query:  # Exact
                return exact_rows
            elif "trend_tag = $2" in query:  # Partial trend
                return partial_trend_rows
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        result = await matcher.match(
            workspace_id=workspace_id,
            regime_key="trend=uptrend|vol=high_vol",
            trend_tag="uptrend",
            vol_tag="high_vol",
            min_samples=5,
            k=20,
        )

        # 2 exact + 5 partial = 7 total, partial tier used
        assert len(result.candidates) == 7
        assert result.sample_context.tier_used == RegimeMatchTier.PARTIAL_TREND
        assert result.sample_context.exact_count == 2
        assert result.sample_context.partial_trend_count == 5
        assert "partial" in result.sample_context.tiers_attempted

    @pytest.mark.asyncio
    async def test_partial_sparse_falls_back_to_global(self, matcher, mock_pool):
        """When partial is sparse and no snapshot, should fall back to global."""
        workspace_id = uuid4()

        # Mock DB: 1 exact, 1 partial, 5 global
        exact_rows = [self._make_candidate()]
        partial_trend_rows = [self._make_candidate(vol_tag="low_vol")]
        global_rows = [
            self._make_candidate(regime_key=f"other_{i}", trend_tag="downtrend")
            for i in range(5)
        ]

        async def mock_fetch(query, *params):
            if "regime_key = $2" in query:
                return exact_rows
            elif "trend_tag = $2" in query and "vol_tag = $2" not in query:
                return partial_trend_rows
            elif "vol_tag = $2" in query and "trend_tag = $2" not in query:
                return []
            elif "best_oos_score IS NOT NULL" in query and "regime_key" not in query:
                return global_rows
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        result = await matcher.match(
            workspace_id=workspace_id,
            regime_key="trend=uptrend|vol=high_vol",
            trend_tag="uptrend",
            vol_tag="high_vol",
            min_samples=5,
            k=20,
            query_snapshot=None,  # No snapshot = skip distance tier
        )

        assert result.sample_context.tier_used == RegimeMatchTier.GLOBAL_BEST
        assert "global" in result.sample_context.tiers_attempted

    @pytest.mark.asyncio
    async def test_match_detail_populated_for_partial(self, matcher, mock_pool):
        """Partial match should have matched_field and matched_value."""
        workspace_id = uuid4()

        partial_trend_rows = [
            self._make_candidate(trend_tag="uptrend", vol_tag="low_vol")
            for _ in range(10)
        ]

        async def mock_fetch(query, *params):
            if "regime_key = $2" in query:
                return []
            elif "trend_tag = $2" in query:
                return partial_trend_rows
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        result = await matcher.match(
            workspace_id=workspace_id,
            trend_tag="uptrend",
            min_samples=5,
            k=20,
        )

        for c in result.candidates:
            assert c.match_detail.tier == RegimeMatchTier.PARTIAL_TREND
            assert c.match_detail.matched_field == "trend_tag"
            assert c.match_detail.matched_value == "uptrend"

    @pytest.mark.asyncio
    async def test_sample_context_tracks_tiers_attempted(self, matcher, mock_pool):
        """Sample context should track all attempted tiers."""
        workspace_id = uuid4()

        # All tiers sparse, global fallback
        async def mock_fetch(query, *params):
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        result = await matcher.match(
            workspace_id=workspace_id,
            regime_key="trend=uptrend|vol=high_vol",
            trend_tag="uptrend",
            vol_tag="high_vol",
            min_samples=5,
            k=20,
        )

        assert "exact" in result.sample_context.tiers_attempted
        assert "partial" in result.sample_context.tiers_attempted
        assert "global" in result.sample_context.tiers_attempted

    @pytest.mark.asyncio
    async def test_strategy_scoped_matching(self, matcher, mock_pool):
        """Should filter by strategy_entity_id when provided."""
        workspace_id = uuid4()
        strategy_id = uuid4()

        query_with_strategy = []

        async def mock_fetch(query, *params):
            if "strategy_entity_id" in query:
                query_with_strategy.append(query)
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        await matcher.match(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_id,
            regime_key="trend=uptrend|vol=high_vol",
            min_samples=5,
            k=20,
        )

        # All queries should include strategy filter
        assert len(query_with_strategy) > 0

    @pytest.mark.asyncio
    async def test_no_regime_key_skips_exact(self, matcher, mock_pool):
        """Should skip exact tier if no regime_key provided."""
        workspace_id = uuid4()

        partial_rows = [self._make_candidate() for _ in range(10)]

        async def mock_fetch(query, *params):
            if "trend_tag = $2" in query:
                return partial_rows
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        result = await matcher.match(
            workspace_id=workspace_id,
            trend_tag="uptrend",  # No regime_key
            min_samples=5,
            k=20,
        )

        # Should still work with partial matching
        assert result.sample_context.exact_count == 0
        assert len(result.candidates) == 10


# =============================================================================
# Test Distance Matching
# =============================================================================


class TestDistanceMatching:
    """Test distance-based matching (Tier 2)."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock())
        return pool

    @pytest.fixture
    def matcher(self, mock_pool):
        """Create matcher with mock pool."""
        return TieredRegimeMatcher(mock_pool)

    @pytest.mark.asyncio
    async def test_distance_tier_with_snapshot(self, matcher, mock_pool):
        """Should compute distance when snapshot provided."""
        workspace_id = uuid4()

        # Create query snapshot
        query_snapshot = RegimeSnapshot(
            atr_pct=0.03,
            trend_strength=0.7,
            rsi=60.0,
            efficiency_ratio=0.5,
        )

        # Mock rows with regime data
        rows_with_regime = [
            {
                "tune_id": uuid4(),
                "run_id": uuid4(),
                "strategy_entity_id": uuid4(),
                "best_params": {"lookback": 20},
                "best_oos_score": 1.0 + i * 0.1,
                "regime_key": f"regime_{i}",
                "trend_tag": "uptrend",
                "vol_tag": "high_vol",
                "efficiency_tag": "efficient",
                "metrics_oos": {
                    "regime": {
                        "atr_pct": 0.03 + i * 0.01,
                        "trend_strength": 0.7 - i * 0.05,
                        "rsi": 60.0 + i * 2,
                        "efficiency_ratio": 0.5,
                    }
                },
            }
            for i in range(5)
        ]

        async def mock_fetch(query, *params):
            if "regime_key = $2" in query:  # Exact
                return []
            elif "trend_tag = $2" in query:  # Partial
                return []
            elif "vol_tag = $2" in query:  # Partial
                return []
            elif "metrics_oos" in query:  # Distance
                return rows_with_regime
            return []

        conn = AsyncMock()
        conn.fetch = mock_fetch
        mock_pool.acquire.return_value.__aenter__.return_value = conn

        result = await matcher.match(
            workspace_id=workspace_id,
            trend_tag="uptrend",
            vol_tag="high_vol",
            query_snapshot=query_snapshot,
            min_samples=3,
            k=20,
        )

        assert result.sample_context.distance_count == 5
        # Distance candidates should have distance_score
        for c in result.candidates:
            if c.match_detail.tier == RegimeMatchTier.DISTANCE:
                assert c.match_detail.distance_score is not None
                assert c.match_detail.distance_rank is not None


# =============================================================================
# Test SampleContext
# =============================================================================


class TestSampleContext:
    """Test SampleContext dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        context = SampleContext()

        assert context.exact_count == 0
        assert context.partial_trend_count == 0
        assert context.partial_vol_count == 0
        assert context.distance_count == 0
        assert context.global_count == 0
        assert context.min_samples == 5
        assert context.k == 20
        assert context.tiers_attempted == []

    def test_fallback_reason_set(self):
        """Should track fallback reason."""
        context = SampleContext(
            tier_used=RegimeMatchTier.PARTIAL_TREND,
            fallback_reason="exact_count=2 < min_samples=5",
        )

        assert context.fallback_reason == "exact_count=2 < min_samples=5"
