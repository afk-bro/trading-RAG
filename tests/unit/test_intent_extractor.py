# tests/unit/test_intent_extractor.py
"""Unit tests for intent extractor service."""

import pytest

from app.services.intent.models import MatchIntent


class TestMatchIntentModel:
    """Tests for MatchIntent model validation."""

    def test_empty_intent(self):
        """Test creating empty intent."""
        intent = MatchIntent()
        assert intent.symbols == []
        assert intent.strategy_archetypes == []
        assert intent.overall_confidence == 0.0

    def test_valid_archetypes_pass_validation(self):
        """Test that valid archetypes pass validation."""
        intent = MatchIntent(strategy_archetypes=["breakout", "momentum"])
        assert intent.strategy_archetypes == ["breakout", "momentum"]

    def test_invalid_archetypes_filtered(self):
        """Test that invalid archetypes are filtered out."""
        intent = MatchIntent(strategy_archetypes=["breakout", "invalid_tag", "momentum"])
        assert "breakout" in intent.strategy_archetypes
        assert "momentum" in intent.strategy_archetypes
        assert "invalid_tag" not in intent.strategy_archetypes

    def test_valid_indicators_pass(self):
        """Test that valid indicators pass validation."""
        intent = MatchIntent(indicators=["rsi", "macd", "bollinger"])
        assert intent.indicators == ["rsi", "macd", "bollinger"]

    def test_invalid_indicators_filtered(self):
        """Test that invalid indicators are filtered out."""
        intent = MatchIntent(indicators=["rsi", "fake_indicator"])
        assert "rsi" in intent.indicators
        assert "fake_indicator" not in intent.indicators

    def test_timeframe_explicit_validation(self):
        """Test explicit timeframe validation."""
        intent = MatchIntent(timeframe_explicit=["1h", "4h", "invalid"])
        assert "1h" in intent.timeframe_explicit
        assert "4h" in intent.timeframe_explicit
        assert "invalid" not in intent.timeframe_explicit

    def test_confidence_bounds(self):
        """Test confidence values are bounded 0-1."""
        intent = MatchIntent(overall_confidence=0.5, script_type_confidence=0.8)
        assert intent.overall_confidence == 0.5
        assert intent.script_type_confidence == 0.8

    def test_deduplication(self):
        """Test that duplicate tags are removed."""
        intent = MatchIntent(indicators=["rsi", "rsi", "macd", "rsi"])
        assert intent.indicators == ["rsi", "macd"]


# Tests for RuleBasedIntentExtractor (will fail until Task 3)
class TestRuleBasedIntentExtractor:
    """Tests for rule-based intent extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        from app.services.intent.extractor import RuleBasedIntentExtractor
        return RuleBasedIntentExtractor()

    def test_extract_strategy_archetypes(self, extractor):
        """Test extracting strategy archetypes."""
        text = "Looking for breakout opportunities with momentum confirmation"
        intent = extractor.extract(text)
        assert "breakout" in intent.strategy_archetypes
        assert "momentum" in intent.strategy_archetypes

    def test_extract_indicators(self, extractor):
        """Test extracting indicators."""
        text = "Using RSI and MACD for entry signals with bollinger bands"
        intent = extractor.extract(text)
        assert "rsi" in intent.indicators
        assert "macd" in intent.indicators
        assert "bollinger" in intent.indicators

    def test_extract_timeframe_bucket(self, extractor):
        """Test extracting timeframe buckets."""
        text = "This is a swing trading strategy for the 4h chart"
        intent = extractor.extract(text)
        assert "swing" in intent.timeframe_buckets

    def test_extract_timeframe_explicit(self, extractor):
        """Test extracting explicit timeframes."""
        text = "Trading on the 1h and 4h timeframes"
        intent = extractor.extract(text)
        assert "1h" in intent.timeframe_explicit
        assert "4h" in intent.timeframe_explicit

    def test_extract_risk_terms(self, extractor):
        """Test extracting risk terms."""
        text = "Using a stop loss at 2% with take profit targets"
        intent = extractor.extract(text)
        assert "stop_loss" in intent.risk_terms
        assert "take_profit" in intent.risk_terms

    def test_infer_strategy_script_type(self, extractor):
        """Test inferring strategy script type."""
        text = "Backtesting this breakout strategy with entry and exit rules"
        intent = extractor.extract(text)
        assert intent.inferred_script_type == "strategy"
        assert intent.script_type_confidence >= 0.5

    def test_infer_indicator_script_type(self, extractor):
        """Test inferring indicator script type."""
        text = "This indicator plots an oscillator with divergence signals"
        intent = extractor.extract(text)
        assert intent.inferred_script_type == "indicator"
        assert intent.script_type_confidence >= 0.5

    def test_uses_existing_metadata(self, extractor):
        """Test that extractor uses existing MetadataExtractor output."""
        from app.services.extractor import ExtractedMetadata
        text = "Trading $AAPL with RSI"
        metadata = ExtractedMetadata(symbols=["AAPL"], topics=["tech"])
        intent = extractor.extract(text, metadata=metadata)
        assert "AAPL" in intent.symbols
        assert "tech" in intent.topics

    def test_overall_confidence_weighted(self, extractor):
        """Test overall confidence is weighted by signal type."""
        # High-signal content
        text = "BTC breakout strategy using RSI on the 4h timeframe"
        intent = extractor.extract(text)
        assert intent.overall_confidence > 0.3  # Has multiple high-signal fields

    def test_dedupe_preserves_order(self, extractor):
        """Test that extraction dedupes while preserving order."""
        text = "RSI RSI RSI using RSI indicator"
        intent = extractor.extract(text)
        assert intent.indicators.count("rsi") == 1

    def test_empty_text(self, extractor):
        """Test extracting from empty text."""
        intent = extractor.extract("")
        assert intent.symbols == []
        assert intent.strategy_archetypes == []
        assert intent.overall_confidence == 0.0

    def test_no_trading_content(self, extractor):
        """Test text with no trading content."""
        text = "The quick brown fox jumps over the lazy dog."
        intent = extractor.extract(text)
        assert len(intent.strategy_archetypes) == 0
        assert len(intent.indicators) == 0
