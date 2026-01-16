# YouTube to Pine Script Match - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `POST /sources/youtube/match-pine` endpoint that extracts trading intent from YouTube transcripts and returns matching Pine scripts.

**Architecture:** Rule-based intent extraction → query building → Pine match → reranking. Hybrid KB/transient flow.

**Tech Stack:** FastAPI, Pydantic, pytest, existing MetadataExtractor patterns

---

## Task 1: Create Intent Models

**Files:**
- Create: `app/services/intent/__init__.py`
- Create: `app/services/intent/models.py`

**Step 1: Create package init**

```python
# app/services/intent/__init__.py
"""Intent extraction for trading content."""

from app.services.intent.models import MatchIntent

__all__ = ["MatchIntent"]
```

**Step 2: Create MatchIntent model with canonical tag mappings**

```python
# app/services/intent/models.py
"""Intent models and canonical tag mappings for trading content extraction."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Canonical tag mappings: pattern -> tag
# All tags are lowercase for consistency

STRATEGY_ARCHETYPE_PATTERNS: dict[str, list[str]] = {
    "trend_following": ["trend", "ride the trend", "trend continuation", "higher highs", "higher lows"],
    "mean_reversion": ["mean revert", "reversal", "snap back", "oversold bounce", "overbought"],
    "breakout": ["breakout", "range break", "ath breakout", "support break", "resistance break", "52 week high"],
    "momentum": ["momentum", "impulse", "strength", "relative strength"],
    "range_bound": ["range", "chop", "sideways", "box", "consolidation"],
    "volatility": ["volatility expansion", "squeeze", "atr", "iv expansion"],
}

INDICATOR_PATTERNS: dict[str, list[str]] = {
    "rsi": ["rsi", "relative strength index"],
    "macd": ["macd"],
    "bollinger": ["bollinger", "bbands", "b bands"],
    "moving_average": ["sma", "ema", "wma", "moving average", "ma crossover"],
    "vwap": ["vwap"],
    "atr": ["atr", "average true range"],
    "volume": ["volume", "obv", "volume profile"],
    "stochastic": ["stoch", "stochastic"],
    "adx": ["adx"],
    "ichimoku": ["ichimoku", "cloud"],
    "pivot": ["pivot"],
    "support_resistance": ["support", "resistance", "s/r"],
}

# Timeframe bucket patterns
TIMEFRAME_BUCKET_PATTERNS: dict[str, list[str]] = {
    "scalp": ["scalp", "scalping"],
    "intraday": ["day trade", "daytrade", "intraday"],
    "swing": ["swing", "swing trade"],
    "position": ["position", "long-term", "long term"],
}

# Explicit timeframes (normalized format)
TIMEFRAME_EXPLICIT_PATTERNS: dict[str, list[str]] = {
    "1m": ["1m", "1 min", "1 minute", "one minute"],
    "5m": ["5m", "5 min", "5 minute", "five minute"],
    "15m": ["15m", "15 min", "15 minute", "fifteen minute"],
    "30m": ["30m", "30 min", "30 minute"],
    "1h": ["1h", "1 hour", "one hour", "hourly"],
    "4h": ["4h", "4 hour", "four hour"],
    "1d": ["1d", "daily", "1 day"],
    "1w": ["1w", "weekly", "1 week"],
}

RISK_TERM_PATTERNS: dict[str, list[str]] = {
    "stop_loss": ["stop loss", "stop-loss", "stoploss", " sl "],
    "take_profit": ["take profit", "take-profit", "takeprofit", " tp ", "target"],
    "trailing_stop": ["trailing stop", "trailing"],
    "position_sizing": ["position size", "position sizing", "risk per trade", "r-multiple", "risk reward"],
    "dca": ["dca", "dollar cost average", "pyramiding", "scaling in", "scaling out"],
}

# Script type inference cues
STRATEGY_CUES = [
    "strategy", "strategy.entry", "strategy.exit", "backtest", "backtesting",
    "drawdown", "win rate", "entry", "exit", "pnl", "profit", "trade",
]
INDICATOR_CUES = [
    "indicator", "plot", "alert", "overlay", "oscillator", "histogram",
    "signal line", "divergence", "tradingview indicator",
]

# Valid canonical tags for validation
VALID_ARCHETYPES = set(STRATEGY_ARCHETYPE_PATTERNS.keys())
VALID_INDICATORS = set(INDICATOR_PATTERNS.keys())
VALID_TIMEFRAME_BUCKETS = set(TIMEFRAME_BUCKET_PATTERNS.keys())
VALID_TIMEFRAME_EXPLICIT = set(TIMEFRAME_EXPLICIT_PATTERNS.keys())
VALID_RISK_TERMS = set(RISK_TERM_PATTERNS.keys())


class MatchIntent(BaseModel):
    """Extracted trading intent from content."""

    # From existing MetadataExtractor
    symbols: list[str] = Field(default_factory=list, description="Ticker symbols (uppercase)")
    topics: list[str] = Field(default_factory=list, description="Topic tags (lowercase)")
    entities: list[str] = Field(default_factory=list, description="Named entities")

    # Trading-specific (lowercase canonical tags, deduped, order-preserved)
    strategy_archetypes: list[str] = Field(default_factory=list, description="Strategy types")
    indicators: list[str] = Field(default_factory=list, description="Technical indicators")
    timeframe_buckets: list[str] = Field(default_factory=list, description="Timeframe categories")
    timeframe_explicit: list[str] = Field(default_factory=list, description="Explicit timeframes")
    risk_terms: list[str] = Field(default_factory=list, description="Risk management terms")

    # Script type inference
    inferred_script_type: Optional[Literal["strategy", "indicator"]] = Field(
        None, description="Inferred Pine script type"
    )
    script_type_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence in script type (Laplace smoothed)"
    )
    overall_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall extraction confidence"
    )

    @field_validator("strategy_archetypes")
    @classmethod
    def validate_archetypes(cls, v: list[str]) -> list[str]:
        """Filter to valid archetype tags."""
        return [tag for tag in v if tag in VALID_ARCHETYPES]

    @field_validator("indicators")
    @classmethod
    def validate_indicators(cls, v: list[str]) -> list[str]:
        """Filter to valid indicator tags."""
        return [tag for tag in v if tag in VALID_INDICATORS]

    @field_validator("timeframe_buckets")
    @classmethod
    def validate_timeframe_buckets(cls, v: list[str]) -> list[str]:
        """Filter to valid timeframe bucket tags."""
        return [tag for tag in v if tag in VALID_TIMEFRAME_BUCKETS]

    @field_validator("timeframe_explicit")
    @classmethod
    def validate_timeframe_explicit(cls, v: list[str]) -> list[str]:
        """Filter to valid explicit timeframe tags."""
        return [tag for tag in v if tag in VALID_TIMEFRAME_EXPLICIT]

    @field_validator("risk_terms")
    @classmethod
    def validate_risk_terms(cls, v: list[str]) -> list[str]:
        """Filter to valid risk term tags."""
        return [tag for tag in v if tag in VALID_RISK_TERMS]
```

**Step 3: Run linter to verify syntax**

```bash
cd /home/x/dev/automation-infra/trading-RAG && python -c "from app.services.intent.models import MatchIntent; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add app/services/intent/
git commit -m "feat(intent): add MatchIntent model with canonical tag mappings"
```

---

## Task 2: Create Intent Extractor Tests

**Files:**
- Create: `tests/unit/test_intent_extractor.py`

**Step 1: Write failing tests for extractor**

```python
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


# Tests for RuleBasedIntentExtractor (will fail until implemented)
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
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/test_intent_extractor.py -v
```

Expected: `TestMatchIntentModel` tests pass, `TestRuleBasedIntentExtractor` tests fail with import error

**Step 3: Commit test file**

```bash
git add tests/unit/test_intent_extractor.py
git commit -m "test(intent): add failing tests for intent extractor"
```

---

## Task 3: Implement Rule-Based Intent Extractor

**Files:**
- Create: `app/services/intent/extractor.py`
- Modify: `app/services/intent/__init__.py`

**Step 1: Implement extractor**

```python
# app/services/intent/extractor.py
"""Intent extraction from trading content."""

import re
from abc import ABC, abstractmethod
from typing import Optional

from app.services.extractor import ExtractedMetadata, get_extractor
from app.services.intent.models import (
    INDICATOR_PATTERNS,
    INDICATOR_CUES,
    RISK_TERM_PATTERNS,
    STRATEGY_ARCHETYPE_PATTERNS,
    STRATEGY_CUES,
    TIMEFRAME_BUCKET_PATTERNS,
    TIMEFRAME_EXPLICIT_PATTERNS,
    MatchIntent,
)


class IntentExtractor(ABC):
    """Abstract base class for intent extraction."""

    @abstractmethod
    def extract(
        self, text: str, metadata: Optional[ExtractedMetadata] = None
    ) -> MatchIntent:
        """
        Extract trading intent from text.

        Args:
            text: Content to analyze
            metadata: Optional pre-extracted metadata from MetadataExtractor

        Returns:
            MatchIntent with extracted trading signals
        """
        pass


class RuleBasedIntentExtractor(IntentExtractor):
    """Rule-based intent extraction using regex and keyword matching."""

    def __init__(self):
        """Initialize extractor with compiled patterns."""
        self._metadata_extractor = get_extractor()

    def extract(
        self, text: str, metadata: Optional[ExtractedMetadata] = None
    ) -> MatchIntent:
        """Extract trading intent using rule-based matching."""
        if not text:
            return MatchIntent()

        text_lower = text.lower()

        # Get base metadata from existing extractor if not provided
        if metadata is None:
            metadata = self._metadata_extractor.extract(text)

        # Extract trading-specific fields
        archetypes = self._extract_with_patterns(text_lower, STRATEGY_ARCHETYPE_PATTERNS)
        indicators = self._extract_with_patterns(text_lower, INDICATOR_PATTERNS)
        tf_buckets = self._extract_with_patterns(text_lower, TIMEFRAME_BUCKET_PATTERNS)
        tf_explicit = self._extract_with_patterns(text_lower, TIMEFRAME_EXPLICIT_PATTERNS)
        risk_terms = self._extract_with_patterns(text_lower, RISK_TERM_PATTERNS)

        # Infer script type
        script_type, script_confidence = self._infer_script_type(text_lower, archetypes)

        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(
            symbols=metadata.symbols,
            archetypes=archetypes,
            indicators=indicators,
            tf_buckets=tf_buckets,
            tf_explicit=tf_explicit,
            topics=metadata.topics,
            entities=metadata.entities,
            risk_terms=risk_terms,
        )

        return MatchIntent(
            symbols=metadata.symbols,
            topics=metadata.topics,
            entities=metadata.entities,
            strategy_archetypes=archetypes,
            indicators=indicators,
            timeframe_buckets=tf_buckets,
            timeframe_explicit=tf_explicit,
            risk_terms=risk_terms,
            inferred_script_type=script_type,
            script_type_confidence=script_confidence,
            overall_confidence=overall_confidence,
        )

    def _extract_with_patterns(
        self, text_lower: str, patterns: dict[str, list[str]]
    ) -> list[str]:
        """
        Extract canonical tags by matching patterns.

        Returns deduped list preserving first-occurrence order.
        """
        found: list[str] = []
        seen: set[str] = set()

        for tag, keywords in patterns.items():
            if tag in seen:
                continue
            for keyword in keywords:
                if keyword in text_lower:
                    found.append(tag)
                    seen.add(tag)
                    break

        return found

    def _infer_script_type(
        self, text_lower: str, archetypes: list[str]
    ) -> tuple[Optional[str], float]:
        """
        Infer whether content describes a strategy or indicator.

        Uses Laplace smoothing: (cues + 1) / (total + 2)
        """
        strategy_count = sum(1 for cue in STRATEGY_CUES if cue in text_lower)
        indicator_count = sum(1 for cue in INDICATOR_CUES if cue in text_lower)

        # Archetypes are strong strategy signals
        if archetypes:
            strategy_count += 1

        # Laplace smoothed confidence
        total = strategy_count + indicator_count
        if total == 0:
            return None, 0.5

        confidence = (strategy_count + 1) / (total + 2)

        if confidence >= 0.6:
            return "strategy", confidence
        elif confidence <= 0.4:
            return "indicator", 1 - confidence
        else:
            return None, 0.5

    def _calculate_confidence(
        self,
        symbols: list,
        archetypes: list,
        indicators: list,
        tf_buckets: list,
        tf_explicit: list,
        topics: list,
        entities: list,
        risk_terms: list,
    ) -> float:
        """
        Calculate overall extraction confidence using weighted signals.

        High-signal: symbols, archetypes, indicators, timeframes (1.0 each)
        Low-signal: topics, entities, risk_terms (0.5 each)
        """
        high_signal = [symbols, archetypes, indicators, tf_buckets, tf_explicit]
        low_signal = [topics, entities, risk_terms]

        weighted_sum = sum(1.0 for f in high_signal if f) + sum(0.5 for f in low_signal if f)
        return min(1.0, weighted_sum / 6.0)


# Singleton instance
_extractor: RuleBasedIntentExtractor | None = None


def get_intent_extractor() -> RuleBasedIntentExtractor:
    """Get or create intent extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = RuleBasedIntentExtractor()
    return _extractor
```

**Step 2: Update package init**

```python
# app/services/intent/__init__.py
"""Intent extraction for trading content."""

from app.services.intent.models import MatchIntent
from app.services.intent.extractor import (
    IntentExtractor,
    RuleBasedIntentExtractor,
    get_intent_extractor,
)

__all__ = [
    "MatchIntent",
    "IntentExtractor",
    "RuleBasedIntentExtractor",
    "get_intent_extractor",
]
```

**Step 3: Run tests to verify they pass**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/test_intent_extractor.py -v
```

Expected: All tests pass

**Step 4: Commit**

```bash
git add app/services/intent/
git commit -m "feat(intent): implement RuleBasedIntentExtractor"
```

---

## Task 4: Implement Query Builder

**Files:**
- Create: `app/services/intent/query_builder.py`
- Create: `tests/unit/test_query_builder.py`
- Modify: `app/services/intent/__init__.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_query_builder.py
"""Unit tests for query builder."""

import pytest
from uuid import uuid4

from app.services.intent.models import MatchIntent
from app.services.intent.query_builder import (
    build_query_string,
    build_filters,
    MatchFiltersApplied,
    HIGH_SIGNAL_TOPICS,
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
        # bucket may or may not be included, but explicit should be there

    def test_high_signal_topic_included(self):
        """Test high-signal topics are included."""
        intent = MatchIntent(
            topics=["crypto", "tech"],  # crypto is high-signal
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
            topics=["breakout"],  # duplicate
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
        assert filters.symbols == []  # Not confident enough

    def test_multiple_symbols_with_confidence(self):
        """Test multiple symbols work with confidence."""
        intent = MatchIntent(symbols=["AAPL", "MSFT", "GOOGL", "AMZN"], overall_confidence=0.7)
        filters = build_filters(intent, MockRequest())
        assert len(filters.symbols) == 3  # Capped at 3

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
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/test_query_builder.py -v
```

Expected: Import error (query_builder.py doesn't exist)

**Step 3: Implement query builder**

```python
# app/services/intent/query_builder.py
"""Query and filter building for Pine script matching."""

from typing import Literal, Optional, Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from app.services.intent.models import MatchIntent


# High-signal topics worth including in query
HIGH_SIGNAL_TOPICS = {"options", "crypto", "forex", "macro"}


class MatchFiltersApplied(BaseModel):
    """Filters applied to Pine script matching."""

    script_type: Optional[Literal["strategy", "indicator"]] = Field(
        None, description="Script type filter"
    )
    symbols: list[str] = Field(default_factory=list, description="Symbol filters")
    lint_ok: bool = Field(True, description="Lint status filter")


class YouTubeMatchRequest(Protocol):
    """Protocol for request objects (allows testing with mocks)."""

    workspace_id: UUID
    symbols: Optional[list[str]]
    script_type: Optional[Literal["strategy", "indicator"]]
    lint_ok: bool


def build_query_string(intent: MatchIntent, request: YouTubeMatchRequest) -> str:
    """
    Build query string for Pine script matching.

    Priority order:
    1. Archetypes (top 2)
    2. Indicators (top 3)
    3. Timeframe: explicit beats bucket
    4. One high-signal topic
    5. Risk terms (if confident)
    6. Request symbols (first one)

    Returns deduped query string, or fallback if empty.
    """
    parts: list[str] = []

    # 1. Archetypes (top 2)
    parts.extend(intent.strategy_archetypes[:2])

    # 2. Indicators (top 3)
    parts.extend(intent.indicators[:3])

    # 3. Timeframe: explicit beats bucket
    if intent.timeframe_explicit:
        parts.extend(intent.timeframe_explicit[:1])
    elif intent.timeframe_buckets:
        parts.extend(intent.timeframe_buckets[:1])

    # 4. One high-signal topic (always, if present)
    high_signal = [t for t in intent.topics if t in HIGH_SIGNAL_TOPICS]
    if high_signal:
        parts.append(high_signal[0])

    # 5. Risk terms (if confident)
    if intent.overall_confidence >= 0.5 and intent.risk_terms:
        parts.extend(intent.risk_terms[:1])

    # 6. Request symbol override (first one)
    if request.symbols:
        parts.append(request.symbols[0])

    # Order-preserving dedupe
    seen: set[str] = set()
    deduped: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    # Fallback if empty
    if not deduped:
        if intent.topics:
            return intent.topics[0]
        return "trading"

    return " ".join(deduped)


def build_filters(intent: MatchIntent, request: YouTubeMatchRequest) -> MatchFiltersApplied:
    """
    Build filters for Pine script matching.

    Symbol filter logic:
    - Request symbols override always
    - Single extracted symbol used
    - Multiple symbols require confidence >= 0.6
    - Empty otherwise (avoid over-constraining)

    Script type logic:
    - Request override takes priority
    - Use inferred type if confidence >= 0.6
    - None otherwise
    """
    # Symbol filter
    if request.symbols:
        symbols = request.symbols
    elif len(intent.symbols) == 1:
        symbols = intent.symbols
    elif intent.overall_confidence >= 0.6 and intent.symbols:
        symbols = intent.symbols[:3]  # Cap at 3
    else:
        symbols = []

    # Script type
    if request.script_type:
        script_type = request.script_type
    elif intent.script_type_confidence >= 0.6 and intent.inferred_script_type:
        script_type = intent.inferred_script_type
    else:
        script_type = None

    return MatchFiltersApplied(
        symbols=symbols,
        script_type=script_type,
        lint_ok=request.lint_ok,
    )
```

**Step 4: Update package init**

```python
# app/services/intent/__init__.py
"""Intent extraction for trading content."""

from app.services.intent.models import MatchIntent
from app.services.intent.extractor import (
    IntentExtractor,
    RuleBasedIntentExtractor,
    get_intent_extractor,
)
from app.services.intent.query_builder import (
    build_query_string,
    build_filters,
    MatchFiltersApplied,
    HIGH_SIGNAL_TOPICS,
)

__all__ = [
    "MatchIntent",
    "IntentExtractor",
    "RuleBasedIntentExtractor",
    "get_intent_extractor",
    "build_query_string",
    "build_filters",
    "MatchFiltersApplied",
    "HIGH_SIGNAL_TOPICS",
]
```

**Step 5: Run tests**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/test_query_builder.py -v
```

Expected: All tests pass

**Step 6: Commit**

```bash
git add app/services/intent/ tests/unit/test_query_builder.py
git commit -m "feat(intent): implement query builder and filters"
```

---

## Task 5: Implement Reranker

**Files:**
- Create: `app/services/intent/reranker.py`
- Create: `tests/unit/test_intent_reranker.py`
- Modify: `app/services/intent/__init__.py`

**Step 1: Write failing tests**

```python
# tests/unit/test_intent_reranker.py
"""Unit tests for intent-based reranker."""

import pytest
from uuid import uuid4

from app.services.intent.models import MatchIntent
from app.services.intent.reranker import rerank, RankedResult


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
        results = [MockPineMatchResult(
            title="RSI MACD Bollinger ATR Strategy",
            score=0.5,
        )]
        ranked = rerank(results, intent)
        # 0.15 * min(2, 4) = 0.30, not 0.60
        assert ranked[0].boost <= 0.30 + 0.1  # +0.1 for possible other boosts

    def test_timeframe_explicit_boost(self):
        """Test explicit timeframe match adds boost."""
        intent = MatchIntent(timeframe_explicit=["4h"])
        results = [MockPineMatchResult(
            title="4H Breakout",
            inputs_preview=["timeframe"],
            score=0.5,
        )]
        ranked = rerank(results, intent)
        assert ranked[0].boost >= 0.12

    def test_timeframe_bucket_boost_lower(self):
        """Test bucket timeframe boost is lower than explicit."""
        intent = MatchIntent(timeframe_buckets=["swing"])
        results = [MockPineMatchResult(title="Swing Trader", score=0.5)]
        ranked = rerank(results, intent)
        # Bucket boost is 0.10, explicit would be 0.12
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
        results = [MockPineMatchResult(
            title="Strategy",
            inputs_preview=["stop_loss_pct"],
            score=0.5,
        )]
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
        results = [MockPineMatchResult(
            title="RSI MACD Breakout 4H Stop Loss",
            inputs_preview=["stop_loss"],
            score=0.5,
        )]
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
        # RSI Strategy gets boost, may overtake
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
        # Should be sorted by id ascending for ties
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
```

**Step 2: Run tests**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/test_intent_reranker.py -v
```

Expected: Import error

**Step 3: Implement reranker**

```python
# app/services/intent/reranker.py
"""Intent-based reranking for Pine script matches."""

from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.services.intent.models import MatchIntent


class PineMatchResultProtocol(Protocol):
    """Protocol for PineMatchResult (allows duck typing)."""

    id: Any
    title: str
    inputs_preview: list[str]
    score: float


class RankedResult(BaseModel):
    """Reranked result with score breakdown."""

    result: Any = Field(..., description="Original PineMatchResult")
    base_score: float = Field(..., description="Original match score")
    boost: float = Field(..., description="Intent-based boost applied")
    final_score: float = Field(..., description="Final score after boost")

    class Config:
        arbitrary_types_allowed = True


def rerank(
    results: list[PineMatchResultProtocol],
    intent: MatchIntent,
) -> list[RankedResult]:
    """
    Rerank Pine match results based on intent overlap.

    Boosts:
    - Indicator overlap: +0.15 * min(2, matches)
    - Timeframe explicit match: +0.12
    - Timeframe bucket match: +0.10
    - Archetype in title: +0.10 (no stack)
    - Risk term mention: +0.05 (no stack)

    Total boost capped at 0.4.
    """
    if not results:
        return []

    ranked: list[RankedResult] = []

    for r in results:
        boost = 0.0

        # Build haystack from title and inputs
        haystack = f"{r.title} {' '.join(r.inputs_preview)}".lower()

        # Indicator overlap: +0.15 * min(2, n)
        ind_matches = sum(1 for i in intent.indicators if i in haystack)
        boost += 0.15 * min(2, ind_matches)

        # Timeframe match: explicit +0.12, bucket +0.10
        tf_matched = False
        for tf in intent.timeframe_explicit[:1]:
            if tf in haystack:
                boost += 0.12
                tf_matched = True
                break

        if not tf_matched:
            for tf in intent.timeframe_buckets[:1]:
                if tf in haystack:
                    boost += 0.10
                    break

        # Archetype in title: +0.10 (no stack)
        if any(a in haystack for a in intent.strategy_archetypes):
            boost += 0.10

        # Risk mention: +0.05 (no stack)
        if any(rt in haystack for rt in intent.risk_terms):
            boost += 0.05

        # Cap total boost at 0.4
        boost = min(0.4, boost)
        final_score = min(1.0, r.score + boost)

        ranked.append(RankedResult(
            result=r,
            base_score=r.score,
            boost=round(boost, 3),
            final_score=round(final_score, 3),
        ))

    # Sort: final_score desc, then id asc (deterministic)
    ranked.sort(key=lambda x: (-x.final_score, str(x.result.id)))

    return ranked
```

**Step 4: Update package init**

Add to `app/services/intent/__init__.py`:
```python
from app.services.intent.reranker import rerank, RankedResult
```

And update `__all__`:
```python
__all__ = [
    # ... existing exports
    "rerank",
    "RankedResult",
]
```

**Step 5: Run tests**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/test_intent_reranker.py -v
```

Expected: All tests pass

**Step 6: Commit**

```bash
git add app/services/intent/ tests/unit/test_intent_reranker.py
git commit -m "feat(intent): implement intent-based reranker"
```

---

## Task 6: Add API Schemas

**Files:**
- Modify: `app/schemas.py`

**Step 1: Add new schema models**

Add to `app/schemas.py` (after PineMatchResponse):

```python
# YouTube to Pine Match models

class IngestRequestHint(BaseModel):
    """Hint for ingesting the video."""

    workspace_id: UUID = Field(..., description="Workspace to ingest into")
    url: str = Field(..., description="YouTube URL")


class PineMatchRankedResult(PineMatchResult):
    """Pine match result with reranking score breakdown."""

    base_score: float = Field(..., description="Original match score")
    boost: float = Field(..., description="Intent-based boost")
    final_score: float = Field(..., description="Final score after boost")


class YouTubeMatchPineRequest(BaseModel):
    """Request for YouTube to Pine script matching."""

    workspace_id: UUID = Field(..., description="Workspace ID")
    url: str = Field(..., description="YouTube video URL")
    symbols: Optional[list[str]] = Field(None, description="Override extracted symbols")
    script_type: Optional[Literal["strategy", "indicator"]] = Field(
        None, description="Filter by script type"
    )
    lint_ok: bool = Field(True, description="Filter to clean scripts")
    top_k: int = Field(10, ge=1, le=50, description="Max results")
    force_transient: bool = Field(False, description="Bypass KB, fetch live")


class YouTubeMatchPineResponse(BaseModel):
    """Response for YouTube to Pine script matching."""

    # Source metadata
    video_id: str = Field(..., description="YouTube video ID")
    title: Optional[str] = Field(None, description="Video title")
    channel: Optional[str] = Field(None, description="Channel name")

    # KB status
    in_knowledge_base: bool = Field(..., description="Whether video is in KB")
    transcript_source: Literal["kb", "transient"] = Field(
        ..., description="Where transcript came from"
    )
    transcript_chars_used: int = Field(..., description="Characters of transcript used")

    # Extraction
    match_intent: dict = Field(..., description="Extracted trading intent")
    extraction_method: Literal["rule_based", "llm"] = Field(
        "rule_based", description="Extraction method used"
    )

    # Match results
    results: list[PineMatchRankedResult] = Field(
        default_factory=list, description="Matched scripts"
    )
    total_searched: int = Field(..., description="Total scripts searched")
    query_used: str = Field(..., description="Query string used for matching")
    filters_applied: dict = Field(..., description="Filters applied")

    # Next actions
    ingest_available: bool = Field(..., description="Whether ingest is available")
    ingest_request_hint: Optional[IngestRequestHint] = Field(
        None, description="Hint for ingesting video"
    )
```

**Step 2: Verify syntax**

```bash
cd /home/x/dev/automation-infra/trading-RAG && python -c "from app.schemas import YouTubeMatchPineRequest, YouTubeMatchPineResponse; print('OK')"
```

**Step 3: Commit**

```bash
git add app/schemas.py
git commit -m "feat(schemas): add YouTube to Pine match request/response models"
```

---

## Task 7: Implement YouTube Pine Match Endpoint

**Files:**
- Create: `app/routers/youtube_pine.py`
- Modify: `app/main.py` (add router)

**Step 1: Implement endpoint**

```python
# app/routers/youtube_pine.py
"""YouTube to Pine Script matching endpoint."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.deps.security import require_admin_token
from app.routers.ingest import _db_pool
from app.routers.pine import match_pine_scripts
from app.routers.youtube import (
    fetch_transcript,
    fetch_video_metadata,
    parse_youtube_url,
)
from app.schemas import (
    IngestRequestHint,
    PineMatchRankedResult,
    PineScriptType,
    YouTubeMatchPineRequest,
    YouTubeMatchPineResponse,
)
from app.services.chunker import normalize_transcript
from app.services.intent import (
    MatchIntent,
    build_filters,
    build_query_string,
    get_intent_extractor,
    rerank,
)

router = APIRouter()
logger = structlog.get_logger(__name__)

# Max characters to use for extraction
MAX_EXTRACTION_CHARS = 50_000


async def check_kb_for_video(
    workspace_id: UUID,
    video_id: str,
) -> tuple[bool, Optional[str], Optional[list[str]]]:
    """
    Check if video is already in knowledge base.

    Returns:
        (in_kb, title, chunk_texts)
    """
    if _db_pool is None:
        return False, None, None

    async with _db_pool.acquire() as conn:
        # Check for document
        doc = await conn.fetchrow(
            """
            SELECT id, title
            FROM documents
            WHERE workspace_id = $1
              AND source_type = 'youtube'
              AND video_id = $2
              AND status = 'active'
            """,
            workspace_id,
            video_id,
        )

        if not doc:
            return False, None, None

        # Load top chunks
        rows = await conn.fetch(
            """
            SELECT content
            FROM chunks
            WHERE doc_id = $1
            ORDER BY chunk_index
            LIMIT 20
            """,
            doc["id"],
        )

        chunk_texts = [r["content"] for r in rows]
        return True, doc["title"], chunk_texts


@router.post(
    "/match-pine",
    response_model=YouTubeMatchPineResponse,
    responses={
        200: {"description": "Match results"},
        401: {"description": "Admin token required"},
        403: {"description": "Invalid admin token"},
        404: {"description": "Video not found or no transcript"},
        422: {"description": "Invalid request parameters"},
        502: {"description": "Upstream service failure"},
    },
    summary="Match YouTube video to Pine scripts",
    description="Extract trading intent from YouTube transcript and find matching Pine scripts.",
)
async def youtube_match_pine(
    request: YouTubeMatchPineRequest,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_token),
) -> YouTubeMatchPineResponse:
    """
    Match YouTube video content to Pine scripts.

    Hybrid flow:
    1. Check if video is already in KB
    2. If in KB (and not force_transient): use stored chunks
    3. Otherwise: fetch transcript transiently
    4. Extract trading intent
    5. Build query and filters
    6. Match against Pine scripts
    7. Rerank results
    """
    log = logger.bind(
        workspace_id=str(request.workspace_id),
        url=request.url,
        force_transient=request.force_transient,
    )
    log.info("youtube_pine_match_started")

    # Parse URL
    parsed = parse_youtube_url(request.url)
    video_id = parsed.get("video_id")

    if not video_id:
        raise HTTPException(422, "Invalid YouTube URL: could not extract video ID")

    # Check KB
    in_kb, kb_title, kb_chunks = await check_kb_for_video(
        request.workspace_id, video_id
    )

    # Determine transcript source
    transcript_text: str = ""
    title: Optional[str] = None
    channel: Optional[str] = None
    transcript_source: str = "transient"

    if in_kb and not request.force_transient and kb_chunks:
        # Use KB chunks
        transcript_source = "kb"
        transcript_text = " ".join(kb_chunks)[:MAX_EXTRACTION_CHARS]
        title = kb_title
        log.info("using_kb_chunks", chunk_count=len(kb_chunks))
    else:
        # Fetch transcript transiently
        try:
            # Fetch metadata
            metadata = await fetch_video_metadata(
                video_id, api_key=settings.youtube_api_key
            )
            title = metadata.get("title")
            channel = metadata.get("channel")

            # Fetch transcript
            transcript_segments = await fetch_transcript(video_id)
            if not transcript_segments:
                raise HTTPException(404, "No transcript available for this video")

            # Normalize and join
            raw_text = " ".join(seg.get("text", "") for seg in transcript_segments)
            transcript_text = normalize_transcript(raw_text)[:MAX_EXTRACTION_CHARS]

            log.info(
                "fetched_transcript",
                chars=len(transcript_text),
                segments=len(transcript_segments),
            )

        except HTTPException:
            raise
        except Exception as e:
            log.error("transcript_fetch_failed", error=str(e))
            raise HTTPException(502, f"Failed to fetch transcript: {e}")

    # Extract intent
    extractor = get_intent_extractor()
    intent = extractor.extract(transcript_text)

    log.info(
        "intent_extracted",
        archetypes=intent.strategy_archetypes,
        indicators=intent.indicators,
        confidence=intent.overall_confidence,
    )

    # Build query and filters
    query_string = build_query_string(intent, request)
    filters = build_filters(intent, request)

    log.info(
        "query_built",
        query=query_string,
        symbols=filters.symbols,
        script_type=filters.script_type,
    )

    # Call Pine match (internal)
    try:
        match_response = await match_pine_scripts(
            workspace_id=request.workspace_id,
            q=query_string,
            symbol=filters.symbols[0] if filters.symbols else None,
            script_type=PineScriptType(filters.script_type) if filters.script_type else None,
            lint_ok=filters.lint_ok if filters.lint_ok else None,
            top_k=request.top_k * 2,  # Get more for reranking
        )
    except Exception as e:
        log.error("pine_match_failed", error=str(e))
        raise HTTPException(500, f"Pine match failed: {e}")

    # Rerank results
    ranked = rerank(match_response.results, intent)

    # Convert to response format
    ranked_results = [
        PineMatchRankedResult(
            id=r.result.id,
            rel_path=r.result.rel_path,
            title=r.result.title,
            script_type=r.result.script_type,
            pine_version=r.result.pine_version,
            score=r.result.score,
            match_reasons=r.result.match_reasons,
            snippet=r.result.snippet,
            inputs_preview=r.result.inputs_preview,
            lint_ok=r.result.lint_ok,
            base_score=r.base_score,
            boost=r.boost,
            final_score=r.final_score,
        )
        for r in ranked[:request.top_k]
    ]

    # Build response
    return YouTubeMatchPineResponse(
        video_id=video_id,
        title=title,
        channel=channel,
        in_knowledge_base=in_kb,
        transcript_source=transcript_source,
        transcript_chars_used=len(transcript_text),
        match_intent=intent.model_dump(),
        extraction_method="rule_based",
        results=ranked_results,
        total_searched=match_response.total_searched,
        query_used=query_string,
        filters_applied=filters.model_dump(),
        ingest_available=not in_kb,
        ingest_request_hint=(
            IngestRequestHint(workspace_id=request.workspace_id, url=request.url)
            if not in_kb
            else None
        ),
    )
```

**Step 2: Register router in main.py**

Find the router registration section in `app/main.py` and add:

```python
from app.routers import youtube_pine

# In the router registration section:
app.include_router(
    youtube_pine.router,
    prefix="/sources/youtube",
    tags=["YouTube"],
)
```

**Step 3: Verify syntax**

```bash
cd /home/x/dev/automation-infra/trading-RAG && python -c "from app.routers.youtube_pine import router; print('OK')"
```

**Step 4: Run all unit tests**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/test_intent*.py tests/unit/test_query_builder.py -v
```

**Step 5: Commit**

```bash
git add app/routers/youtube_pine.py app/main.py
git commit -m "feat(api): implement POST /sources/youtube/match-pine endpoint"
```

---

## Task 8: Integration Verification

**Step 1: Run full test suite**

```bash
cd /home/x/dev/automation-infra/trading-RAG && pytest tests/unit/ -v --tb=short
```

**Step 2: Run linters**

```bash
cd /home/x/dev/automation-infra/trading-RAG && black --check app/services/intent/ app/routers/youtube_pine.py tests/unit/test_intent*.py tests/unit/test_query_builder.py
cd /home/x/dev/automation-infra/trading-RAG && flake8 app/services/intent/ app/routers/youtube_pine.py --max-line-length=100
```

**Step 3: Fix any issues and commit**

```bash
git add -A
git commit -m "chore: lint fixes for youtube-pine-match feature"
```

---

## Summary

This plan implements:

1. **Task 1**: MatchIntent model with canonical tag mappings and validators
2. **Task 2**: Unit tests for intent extractor (TDD)
3. **Task 3**: RuleBasedIntentExtractor implementation
4. **Task 4**: Query builder with priority ordering and filters
5. **Task 5**: Intent-based reranker with score preservation
6. **Task 6**: API schemas for request/response
7. **Task 7**: Endpoint implementation with hybrid KB/transient flow
8. **Task 8**: Integration verification

Each task is atomic with clear test → implement → verify → commit cycle.
