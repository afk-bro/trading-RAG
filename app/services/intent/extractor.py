# app/services/intent/extractor.py
"""Intent extraction from trading content."""

from abc import ABC, abstractmethod
from typing import Optional

from app.services.extractor import ExtractedMetadata, get_extractor
from app.services.intent.models import (
    INDICATOR_CUES,
    INDICATOR_PATTERNS,
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

        weighted_sum = sum(1.0 for f in high_signal if f) + sum(
            0.5 for f in low_signal if f
        )
        return min(1.0, weighted_sum / 6.0)


# Singleton instance
_extractor: RuleBasedIntentExtractor | None = None


def get_intent_extractor() -> RuleBasedIntentExtractor:
    """Get or create intent extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = RuleBasedIntentExtractor()
    return _extractor
