# tests/unit/test_query_builder.py
"""Unit tests for query builder."""

from uuid import uuid4

from app.services.intent.models import MatchIntent
from app.services.intent.query_builder import (
    build_query_string,
    build_filters,
)


class MockRequest:
    """Mock request for testing."""

    def __init__(
        self,
        workspace_id=None,
        symbols=None,
        script_type=None,
        lint_ok=True,
    ):
        self.workspace_id = workspace_id or uuid4()
        self.symbols = symbols
        self.script_type = script_type
        self.lint_ok = lint_ok


class TestBuildQueryString:
    """Tests for query string building."""

    def test_archetypes_first(self):
        """Test archetypes appear first in query."""
        intent = MatchIntent(
            strategy_archetypes=["breakout", "momentum"],
            indicators=["rsi"],
        )
        query = build_query_string(intent, MockRequest())
        parts = query.split()
        assert parts[0] == "breakout"
        assert parts[1] == "momentum"

    def test_indicators_after_archetypes(self):
        """Test indicators appear after archetypes."""
        intent = MatchIntent(
            strategy_archetypes=["breakout"],
            indicators=["rsi", "macd", "bollinger"],
        )
        query = build_query_string(intent, MockRequest())
        assert "rsi" in query
        assert "macd" in query
        assert "bollinger" in query

    def test_timeframe_explicit_beats_bucket(self):
        """Test explicit timeframes take priority over buckets."""
        intent = MatchIntent(
            timeframe_explicit=["4h"],
            timeframe_buckets=["swing"],
        )
        query = build_query_string(intent, MockRequest())
        assert "4h" in query

    def test_high_signal_topic_included(self):
        """Test high-signal topics are included."""
        intent = MatchIntent(
            topics=["crypto", "tech"],
            indicators=["rsi"],
        )
        query = build_query_string(intent, MockRequest())
        assert "crypto" in query

    def test_risk_terms_when_confident(self):
        """Test risk terms included when confidence high."""
        intent = MatchIntent(
            indicators=["rsi"],
            risk_terms=["stop_loss"],
            overall_confidence=0.6,
        )
        query = build_query_string(intent, MockRequest())
        assert "stop_loss" in query

    def test_risk_terms_excluded_when_low_confidence(self):
        """Test risk terms excluded when confidence low."""
        intent = MatchIntent(
            risk_terms=["stop_loss"],
            overall_confidence=0.3,
        )
        query = build_query_string(intent, MockRequest())
        assert "stop_loss" not in query

    def test_request_symbols_appended(self):
        """Test request symbols are appended to query."""
        intent = MatchIntent(indicators=["rsi"])
        request = MockRequest(symbols=["BTC"])
        query = build_query_string(intent, request)
        assert "BTC" in query

    def test_deduplication(self):
        """Test duplicate terms are removed."""
        intent = MatchIntent(
            strategy_archetypes=["breakout"],
            topics=["breakout"],
        )
        query = build_query_string(intent, MockRequest())
        assert query.count("breakout") == 1

    def test_fallback_when_empty(self):
        """Test fallback when no extraction."""
        intent = MatchIntent()
        query = build_query_string(intent, MockRequest())
        assert query == "trading"

    def test_fallback_uses_topic(self):
        """Test fallback uses first topic if available."""
        intent = MatchIntent(topics=["tech"])
        query = build_query_string(intent, MockRequest())
        assert query == "tech"


class TestBuildFilters:
    """Tests for filter building."""

    def test_request_symbols_override(self):
        """Test request symbols override intent symbols."""
        intent = MatchIntent(symbols=["AAPL"], overall_confidence=0.8)
        request = MockRequest(symbols=["BTC", "ETH"])
        filters = build_filters(intent, request)
        assert filters.symbols == ["BTC", "ETH"]

    def test_single_symbol_used(self):
        """Test single extracted symbol is used."""
        intent = MatchIntent(symbols=["AAPL"])
        filters = build_filters(intent, MockRequest())
        assert filters.symbols == ["AAPL"]

    def test_multiple_symbols_need_confidence(self):
        """Test multiple symbols require confidence threshold."""
        intent = MatchIntent(symbols=["AAPL", "MSFT"], overall_confidence=0.4)
        filters = build_filters(intent, MockRequest())
        assert filters.symbols == []

    def test_multiple_symbols_with_confidence(self):
        """Test multiple symbols work with confidence."""
        intent = MatchIntent(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"], overall_confidence=0.7
        )
        filters = build_filters(intent, MockRequest())
        assert len(filters.symbols) == 3

    def test_script_type_from_request(self):
        """Test script type from request takes priority."""
        intent = MatchIntent(
            inferred_script_type="indicator",
            script_type_confidence=0.9,
        )
        request = MockRequest(script_type="strategy")
        filters = build_filters(intent, request)
        assert filters.script_type == "strategy"

    def test_script_type_from_intent_confident(self):
        """Test script type from intent when confident."""
        intent = MatchIntent(
            inferred_script_type="strategy",
            script_type_confidence=0.7,
        )
        filters = build_filters(intent, MockRequest())
        assert filters.script_type == "strategy"

    def test_script_type_none_when_uncertain(self):
        """Test script type is None when uncertain."""
        intent = MatchIntent(
            inferred_script_type="strategy",
            script_type_confidence=0.5,
        )
        filters = build_filters(intent, MockRequest())
        assert filters.script_type is None

    def test_lint_ok_from_request(self):
        """Test lint_ok comes from request."""
        filters = build_filters(MatchIntent(), MockRequest(lint_ok=False))
        assert filters.lint_ok is False
