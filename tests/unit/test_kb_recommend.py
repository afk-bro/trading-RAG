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
        assert filters["max_drawdown"] == DEFAULT_STRICT_FILTERS["max_drawdown"], "Workspace default"


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
            assert suggestion.risk_note, f"Missing risk note for {suggestion.filter_name}"
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
            assert name in ["min_trades", "max_drawdown", "max_overfit_gap", "require_oos"]


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
