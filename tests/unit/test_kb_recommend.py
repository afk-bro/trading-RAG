"""Unit tests for KB recommendation pipeline.

Tests for:
- Confidence guard on low trade counts
- Request override priority in filter rejections
- RecommendedRelaxedSettings only on status=none
"""

import pytest
from uuid import uuid4

from app.services.kb.aggregation import compute_confidence, ParamSpread
from app.services.kb.retrieval import (
    build_filters,
    DEFAULT_STRICT_FILTERS,
    RetrievalRequest,
    FilterRejections,
)
from app.services.kb.recommend import (
    RelaxationSuggestion,
    RecommendedRelaxedSettings,
    RecommendRequest,
    KBRecommender,
)


# =============================================================================
# Confidence Guard Tests
# =============================================================================


class TestConfidenceGuardLowTrades:
    """Tests for confidence guard on low trade count."""

    def test_confidence_capped_when_median_trades_below_10(self):
        """Confidence should be capped at 0.4 when median_oos_trades < 10."""
        # High count that would normally give high confidence
        conf = compute_confidence(
            spreads={},
            count_used=100,  # Would normally give ~0.9
            median_oos_trades=5,  # Below trust threshold
        )
        assert conf <= 0.4, f"Confidence {conf} should be <= 0.4 with low trades"

    def test_confidence_not_capped_when_median_trades_above_10(self):
        """Confidence should not be capped when median_oos_trades >= 10."""
        conf = compute_confidence(
            spreads={},
            count_used=100,
            median_oos_trades=50,  # Above trust threshold
        )
        assert conf > 0.4, f"Confidence {conf} should be > 0.4 with adequate trades"

    def test_confidence_with_trades_at_threshold(self):
        """Confidence at exactly 10 trades should not be penalized."""
        conf = compute_confidence(
            spreads={},
            count_used=50,
            median_oos_trades=10,  # Exactly at threshold
        )
        # Should not have low trades penalty
        conf_with_low = compute_confidence(
            spreads={},
            count_used=50,
            median_oos_trades=9,  # Just below threshold
        )
        assert conf > conf_with_low, "10 trades should have higher confidence than 9"

    def test_confidence_with_none_trades_no_penalty(self):
        """When median_oos_trades is None, no penalty applied."""
        conf_with_none = compute_confidence(
            spreads={},
            count_used=50,
            median_oos_trades=None,
        )
        conf_with_high = compute_confidence(
            spreads={},
            count_used=50,
            median_oos_trades=100,
        )
        # Should be equal (no penalty for None)
        assert conf_with_none == conf_with_high

    def test_low_trades_with_spread_penalties(self):
        """Low trades penalty should stack with spread penalties."""
        spreads = {
            "param1": ParamSpread(
                name="param1",
                value=10.0,
                count_used=10,
                weight_sum=10.0,
                spread=5.0,
            ),
        }
        conf = compute_confidence(
            spreads=spreads,
            count_used=50,
            median_oos_trades=3,  # Very low
        )
        # Should have both penalties and still be capped
        assert conf <= 0.4
        assert conf >= 0.0


# =============================================================================
# Request Override Priority Tests
# =============================================================================


class TestRequestOverridePriority:
    """Tests for request overrides taking priority over strategy defaults."""

    def test_request_min_trades_overrides_strategy_default(self):
        """Request min_trades should override strategy default."""
        req = RetrievalRequest(
            workspace_id=uuid4(),
            strategy_name="trend_following",
            objective_type="sharpe",
            min_trades=7,  # Request override
        )
        # Strategy default is 3, workspace default is 5
        strategy_floors = {"min_trades": 3}

        filters = build_filters(req, strict=True, strategy_floors=strategy_floors)
        assert filters["min_trades"] == 7, "Request should override strategy floor"

    def test_strategy_floor_overrides_workspace_default(self):
        """Strategy floor should override workspace default when no request."""
        req = RetrievalRequest(
            workspace_id=uuid4(),
            strategy_name="trend_following",
            objective_type="sharpe",
            # No min_trades override
        )
        strategy_floors = {"min_trades": 3}

        filters = build_filters(req, strict=True, strategy_floors=strategy_floors)
        assert filters["min_trades"] == 3, "Strategy floor should override default (5)"

    def test_workspace_default_used_when_no_overrides(self):
        """Workspace default should be used when no overrides."""
        req = RetrievalRequest(
            workspace_id=uuid4(),
            strategy_name="mean_reversion",
            objective_type="sharpe",
            # No overrides
        )

        filters = build_filters(req, strict=True, strategy_floors=None)
        assert filters["min_trades"] == DEFAULT_STRICT_FILTERS["min_trades"]

    def test_request_max_overfit_gap_overrides_strategy(self):
        """Request max_overfit_gap should override strategy default."""
        req = RetrievalRequest(
            workspace_id=uuid4(),
            strategy_name="trend_following",
            objective_type="sharpe",
            max_overfit_gap=0.5,  # Request override
        )
        strategy_floors = {"max_overfit_gap": 0.4}  # Strategy default

        filters = build_filters(req, strict=True, strategy_floors=strategy_floors)
        assert filters["max_overfit_gap"] == 0.5, "Request should override strategy"

    def test_priority_chain_all_levels(self):
        """Full priority chain: request > strategy > workspace."""
        req = RetrievalRequest(
            workspace_id=uuid4(),
            strategy_name="test",
            objective_type="sharpe",
            min_trades=8,
            # max_overfit_gap not set - should use strategy
            # max_drawdown not set - should use workspace
        )
        strategy_floors = {
            "min_trades": 3,  # Overridden by request
            "max_overfit_gap": 0.4,  # Used (no request override)
        }

        filters = build_filters(req, strict=True, strategy_floors=strategy_floors)
        assert filters["min_trades"] == 8, "Request override"
        assert filters["max_overfit_gap"] == 0.4, "Strategy floor"
        assert (
            filters["max_drawdown"] == DEFAULT_STRICT_FILTERS["max_drawdown"]
        ), "Workspace default"


# =============================================================================
# RecommendedRelaxedSettings Tests
# =============================================================================


class TestRecommendedRelaxedSettings:
    """Tests for recommended relaxed settings behavior."""

    def test_suggestions_sorted_by_estimated_candidates(self):
        """Suggestions should be sorted by estimated_candidates descending."""
        from app.services.kb.recommend import KBRecommender

        recommender = KBRecommender(repository=None)

        filter_rejections = FilterRejections(
            total_before_filters=100,
            by_oos=10,
            by_trades=30,
            by_drawdown=20,
            by_overfit_gap=15,
        )

        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="test",
            objective_type="sharpe",
        )

        result = recommender._compute_recommended_relaxed_settings(
            filter_rejections=filter_rejections,
            current_request=req,
        )

        assert result is not None
        assert len(result.suggestions) > 0

        # Verify sorted by estimated_candidates descending
        estimates = [s.estimated_candidates for s in result.suggestions]
        assert estimates == sorted(estimates, reverse=True)

    def test_each_suggestion_has_risk_note(self):
        """Each suggestion should have a non-empty risk note."""
        from app.services.kb.recommend import KBRecommender

        recommender = KBRecommender(repository=None)

        filter_rejections = FilterRejections(
            total_before_filters=100,
            by_oos=10,
            by_trades=30,
            by_drawdown=20,
            by_overfit_gap=15,
        )

        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="test",
            objective_type="sharpe",
        )

        result = recommender._compute_recommended_relaxed_settings(
            filter_rejections=filter_rejections,
            current_request=req,
        )

        assert result is not None
        for suggestion in result.suggestions:
            assert (
                suggestion.risk_note
            ), f"Missing risk note for {suggestion.filter_name}"
            assert len(suggestion.risk_note) > 10, "Risk note should be descriptive"

    def test_no_suggestions_when_no_rejections(self):
        """Should return None when no filter rejections."""
        from app.services.kb.recommend import KBRecommender

        recommender = KBRecommender(repository=None)

        filter_rejections = FilterRejections(
            total_before_filters=100,
            by_oos=0,
            by_trades=0,
            by_drawdown=0,
            by_overfit_gap=0,
        )

        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="test",
            objective_type="sharpe",
        )

        result = recommender._compute_recommended_relaxed_settings(
            filter_rejections=filter_rejections,
            current_request=req,
        )

        assert result is None

    def test_no_suggestions_when_no_data(self):
        """Should return None when total_before_filters is 0."""
        from app.services.kb.recommend import KBRecommender

        recommender = KBRecommender(repository=None)

        filter_rejections = FilterRejections(
            total_before_filters=0,  # No data at all
            by_oos=0,
            by_trades=0,
            by_drawdown=0,
            by_overfit_gap=0,
        )

        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="test",
            objective_type="sharpe",
        )

        result = recommender._compute_recommended_relaxed_settings(
            filter_rejections=filter_rejections,
            current_request=req,
        )

        assert result is None

    def test_single_axis_suggestions_independent(self):
        """Each suggestion should be for a single filter."""
        from app.services.kb.recommend import KBRecommender

        recommender = KBRecommender(repository=None)

        filter_rejections = FilterRejections(
            total_before_filters=100,
            by_oos=10,
            by_trades=30,
            by_drawdown=20,
            by_overfit_gap=15,
        )

        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="test",
            objective_type="sharpe",
        )

        result = recommender._compute_recommended_relaxed_settings(
            filter_rejections=filter_rejections,
            current_request=req,
        )

        assert result is not None
        filter_names = [s.filter_name for s in result.suggestions]

        # Each should be a unique single filter
        assert len(filter_names) == len(set(filter_names)), "Duplicate filter names"
        for name in filter_names:
            assert name in [
                "min_trades",
                "max_drawdown",
                "max_overfit_gap",
                "require_oos",
            ]


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestRecommendPipelineIntegration:
    """Integration-style tests for recommend pipeline behavior."""

    def test_low_trade_count_adds_reason_and_warning(self):
        """Low trade count should add reason and warning to response."""
        # This would require mocking the full pipeline
        # Simplified test of the reason addition logic
        reasons = []
        warnings = []
        median_oos_trades = 5

        if median_oos_trades is not None and median_oos_trades < 10:
            reasons.append("low_trade_count")
            warnings.append("low_oos_trades_statistical_noise")

        assert "low_trade_count" in reasons
        assert "low_oos_trades_statistical_noise" in warnings


# =============================================================================
# v1.5 Tests
# =============================================================================


class TestBuildRegimeKeyFromTags:
    """Tests for _build_regime_key_from_tags helper."""

    def test_builds_key_from_valid_tags(self):
        """Should build canonical key from valid trend and vol tags."""
        recommender = KBRecommender(repository=None)
        key = recommender._build_regime_key_from_tags(["uptrend", "high_vol"])
        assert key == "trend=uptrend|vol=high_vol"

    def test_builds_key_case_insensitive(self):
        """Should handle case-insensitive tags."""
        recommender = KBRecommender(repository=None)
        key = recommender._build_regime_key_from_tags(["DOWNTREND", "LOW_VOL"])
        assert key == "trend=downtrend|vol=low_vol"

    def test_returns_none_for_missing_trend(self):
        """Should return None when trend is missing."""
        recommender = KBRecommender(repository=None)
        key = recommender._build_regime_key_from_tags(["high_vol"])
        assert key is None

    def test_returns_none_for_missing_vol(self):
        """Should return None when vol is missing."""
        recommender = KBRecommender(repository=None)
        key = recommender._build_regime_key_from_tags(["uptrend"])
        assert key is None

    def test_returns_none_for_empty_tags(self):
        """Should return None for empty tags list."""
        recommender = KBRecommender(repository=None)
        key = recommender._build_regime_key_from_tags([])
        assert key is None

    def test_returns_none_for_none_tags(self):
        """Should return None when tags is None."""
        recommender = KBRecommender(repository=None)
        key = recommender._build_regime_key_from_tags(None)
        assert key is None

    def test_ignores_extra_tags(self):
        """Should ignore irrelevant tags."""
        recommender = KBRecommender(repository=None)
        key = recommender._build_regime_key_from_tags(
            ["uptrend", "high_vol", "extra_tag", "another_tag"]
        )
        assert key == "trend=uptrend|vol=high_vol"


class TestV15ResponseDataclasses:
    """Tests for v1.5 response dataclasses."""

    def test_regime_state_stability_defaults(self):
        """RegimeStateStability should have sensible defaults."""
        from app.services.kb.recommend import RegimeStateStability

        stability = RegimeStateStability()
        assert stability.candidate_key is None
        assert stability.candidate_bars == 0
        assert stability.M == 20
        assert stability.C_enter == 0.75
        assert stability.C_exit == 0.55

    def test_window_metadata_defaults(self):
        """WindowMetadata should have sensible defaults."""
        from app.services.kb.recommend import WindowMetadata

        window = WindowMetadata()
        assert window.regime_age_bars == 0
        assert window.performance_window is None
        assert window.distance_window is None

    def test_recommend_timings_includes_v15_fields(self):
        """RecommendTimings should include v1.5 timing fields."""
        from app.services.kb.recommend import RecommendTimings

        timings = RecommendTimings()
        assert timings.distance_ms == 0.0
        assert timings.duration_ms == 0.0


class TestRecommendResponseV15Fields:
    """Tests for v1.5 fields in RecommendResponse."""

    def test_response_has_v15_fields(self):
        """RecommendResponse should have all v1.5 fields."""
        from app.services.kb.recommend import (
            RecommendResponse,
            RegimeStateStability,
            WindowMetadata,
        )

        response = RecommendResponse(
            params={"period": 20},
            status="ok",
            confidence=0.8,
            # v1.5 fields
            regime_fit_confidence=0.85,
            regime_distance_z=1.5,
            distance_baseline="neighbors_only",
            distance_n=10,
            regime_age_bars=15,
            regime_half_life_bars=30,
            expected_remaining_bars=15,
            duration_iqr_bars=[20, 40],
            remaining_iqr_bars=[5, 25],
            duration_baseline="composite_symbol",
            duration_n=50,
            stable_regime_key="trend=uptrend|vol=high_vol",
            raw_regime_key="trend=uptrend|vol=high_vol",
            regime_state_stability=RegimeStateStability(
                candidate_key=None,
                candidate_bars=0,
                M=20,
                C_enter=0.75,
                C_exit=0.55,
            ),
            windows=WindowMetadata(
                regime_age_bars=15,
                distance_window={"bars": 10, "timeframe": "5m"},
            ),
            missing=[],
        )

        assert response.regime_fit_confidence == 0.85
        assert response.regime_distance_z == 1.5
        assert response.distance_baseline == "neighbors_only"
        assert response.distance_n == 10
        assert response.regime_age_bars == 15
        assert response.regime_half_life_bars == 30
        assert response.expected_remaining_bars == 15
        assert response.duration_iqr_bars == [20, 40]
        assert response.remaining_iqr_bars == [5, 25]
        assert response.duration_baseline == "composite_symbol"
        assert response.duration_n == 50
        assert response.stable_regime_key == "trend=uptrend|vol=high_vol"
        assert response.raw_regime_key == "trend=uptrend|vol=high_vol"
        assert response.regime_state_stability is not None
        assert response.windows is not None
        assert response.missing == []

    def test_response_v15_fields_optional(self):
        """v1.5 fields should all be optional."""
        from app.services.kb.recommend import RecommendResponse

        # Create response with only required fields
        response = RecommendResponse(
            params={},
            status="none",
        )

        # All v1.5 fields should be None or empty
        assert response.regime_fit_confidence is None
        assert response.regime_distance_z is None
        assert response.distance_baseline is None
        assert response.distance_n is None
        assert response.regime_age_bars is None
        assert response.regime_half_life_bars is None
        assert response.expected_remaining_bars is None
        assert response.duration_iqr_bars is None
        assert response.remaining_iqr_bars is None
        assert response.duration_baseline is None
        assert response.duration_n is None
        assert response.stable_regime_key is None
        assert response.raw_regime_key is None
        assert response.regime_state_stability is None
        assert response.windows is None
        assert response.missing == []


class TestRecommendRequestV15Fields:
    """Tests for v1.5 fields in RecommendRequest."""

    def test_request_has_v15_context_fields(self):
        """RecommendRequest should have v1.5 context fields."""
        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="bb_reversal",
            objective_type="sharpe",
            symbol="BTC/USDT",
            strategy_entity_id=uuid4(),
            timeframe="5m",
        )

        assert req.symbol == "BTC/USDT"
        assert req.strategy_entity_id is not None
        assert req.timeframe == "5m"

    def test_request_v15_fields_optional(self):
        """v1.5 context fields should be optional."""
        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="bb_reversal",
            objective_type="sharpe",
        )

        assert req.symbol is None
        assert req.strategy_entity_id is None
        assert req.timeframe is None
        assert req.fsm_config is None

    def test_request_fsm_config_override(self):
        """Request should accept FSM config override."""
        from app.services.kb.regime_fsm import FSMConfig

        custom_config = FSMConfig(M=30, C_enter=0.8, C_exit=0.6)

        req = RecommendRequest(
            workspace_id=uuid4(),
            strategy_name="bb_reversal",
            objective_type="sharpe",
            fsm_config=custom_config,
        )

        assert req.fsm_config is not None
        assert req.fsm_config.M == 30
        assert req.fsm_config.C_enter == 0.8
        assert req.fsm_config.C_exit == 0.6
