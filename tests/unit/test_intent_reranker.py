# tests/unit/test_intent_reranker.py
"""Unit tests for intent-based reranker."""

from uuid import uuid4

from app.services.intent.models import MatchIntent
from app.services.intent.reranker import rerank


class MockPineMatchResult:
    """Mock PineMatchResult for testing."""

    def __init__(self, id=None, title="", inputs_preview=None, score=0.5):
        self.id = id or uuid4()
        self.title = title
        self.inputs_preview = inputs_preview or []
        self.score = score
        self.rel_path = "test.pine"
        self.script_type = "strategy"
        self.pine_version = "5"
        self.match_reasons = []
        self.snippet = None
        self.lint_ok = True


class TestReranker:
    """Tests for reranking logic."""

    def test_indicator_boost(self):
        """Test indicator overlap adds boost."""
        intent = MatchIntent(indicators=["rsi", "macd"])
        results = [MockPineMatchResult(title="RSI Strategy", score=0.5)]
        ranked = rerank(results, intent)
        assert ranked[0].boost > 0
        assert ranked[0].final_score > 0.5

    def test_indicator_boost_capped_at_2(self):
        """Test indicator boost capped at 2 matches."""
        intent = MatchIntent(indicators=["rsi", "macd", "bollinger", "atr"])
        results = [
            MockPineMatchResult(
                title="RSI MACD Bollinger ATR Strategy",
                score=0.5,
            )
        ]
        ranked = rerank(results, intent)
        # 0.15 * min(2, 4) = 0.30, not 0.60
        assert ranked[0].boost <= 0.35  # Allow small margin for other boosts

    def test_timeframe_explicit_boost(self):
        """Test explicit timeframe match adds boost."""
        intent = MatchIntent(timeframe_explicit=["4h"])
        results = [
            MockPineMatchResult(
                title="4H Breakout",
                inputs_preview=["timeframe"],
                score=0.5,
            )
        ]
        ranked = rerank(results, intent)
        assert ranked[0].boost >= 0.12

    def test_timeframe_bucket_boost_lower(self):
        """Test bucket timeframe boost is lower than explicit."""
        intent = MatchIntent(timeframe_buckets=["swing"])
        results = [MockPineMatchResult(title="Swing Trader", score=0.5)]
        ranked = rerank(results, intent)
        assert 0.09 <= ranked[0].boost <= 0.11

    def test_archetype_boost(self):
        """Test archetype match adds boost."""
        intent = MatchIntent(strategy_archetypes=["breakout"])
        results = [MockPineMatchResult(title="Breakout Strategy", score=0.5)]
        ranked = rerank(results, intent)
        assert ranked[0].boost >= 0.10

    def test_risk_term_boost(self):
        """Test risk term adds small boost."""
        intent = MatchIntent(risk_terms=["stop_loss"])
        results = [
            MockPineMatchResult(
                title="Strategy",
                inputs_preview=["stop_loss_pct"],
                score=0.5,
            )
        ]
        ranked = rerank(results, intent)
        assert ranked[0].boost >= 0.05

    def test_total_boost_capped(self):
        """Test total boost is capped at 0.4."""
        intent = MatchIntent(
            indicators=["rsi", "macd", "bollinger"],
            strategy_archetypes=["breakout"],
            timeframe_explicit=["4h"],
            risk_terms=["stop_loss"],
        )
        results = [
            MockPineMatchResult(
                title="RSI MACD Breakout 4H Stop Loss",
                inputs_preview=["stop_loss"],
                score=0.5,
            )
        ]
        ranked = rerank(results, intent)
        assert ranked[0].boost <= 0.4
        assert ranked[0].final_score <= 0.9

    def test_preserves_base_score(self):
        """Test base score is preserved."""
        intent = MatchIntent(indicators=["rsi"])
        results = [MockPineMatchResult(title="RSI", score=0.7)]
        ranked = rerank(results, intent)
        assert ranked[0].base_score == 0.7

    def test_sorted_by_final_score(self):
        """Test results sorted by final score descending."""
        intent = MatchIntent(indicators=["rsi"])
        results = [
            MockPineMatchResult(title="No match", score=0.8),
            MockPineMatchResult(title="RSI Strategy", score=0.6),
        ]
        ranked = rerank(results, intent)
        assert ranked[0].final_score >= ranked[1].final_score

    def test_deterministic_tiebreaker(self):
        """Test deterministic ordering on ties."""
        intent = MatchIntent()
        id1, id2 = uuid4(), uuid4()
        results = [
            MockPineMatchResult(id=id1, score=0.5),
            MockPineMatchResult(id=id2, score=0.5),
        ]
        ranked = rerank(results, intent)
        assert str(ranked[0].result.id) < str(ranked[1].result.id)

    def test_empty_results(self):
        """Test empty results handled."""
        ranked = rerank([], MatchIntent())
        assert ranked == []

    def test_empty_intent(self):
        """Test empty intent gives no boost."""
        results = [MockPineMatchResult(score=0.5)]
        ranked = rerank(results, MatchIntent())
        assert ranked[0].boost == 0.0
        assert ranked[0].final_score == 0.5
