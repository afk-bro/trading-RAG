"""Coverage gap detection logic."""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.services.intent.models import MatchIntent


class CoverageReasonCode(str, Enum):
    """Reason codes for coverage gaps."""

    NO_RESULTS_ABOVE_THRESHOLD = "NO_RESULTS_ABOVE_THRESHOLD"
    LOW_BEST_SCORE = "LOW_BEST_SCORE"
    LOW_SIGNAL_INPUT = "LOW_SIGNAL_INPUT"
    NO_RESULTS = "NO_RESULTS"


# Default thresholds
DEFAULT_SCORE_THRESHOLD = 0.55  # Score considered "good match"
DEFAULT_WEAK_BEST_SCORE = 0.45  # Below this = weak coverage
DEFAULT_LOW_CONFIDENCE = 0.35  # Below this = low signal input


@dataclass
class CoverageAssessment:
    """Result of coverage gap assessment."""

    weak: bool
    best_score: Optional[float]
    avg_top_k_score: Optional[float]
    num_above_threshold: int
    threshold: float
    reason_codes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def compute_intent_signature(intent: MatchIntent) -> str:
    """
    Compute deterministic signature hash for intent deduplication.

    Canonical format: archetypes|indicators|timeframes|topics|risk_terms
    Each field sorted alphabetically for stability.

    Args:
        intent: MatchIntent to compute signature for

    Returns:
        SHA256 hex digest (64 chars)
    """
    # Sort each field alphabetically for deterministic ordering
    archetypes = ",".join(sorted(intent.strategy_archetypes))
    indicators = ",".join(sorted(intent.indicators))
    timeframes = ",".join(sorted(intent.timeframe_buckets + intent.timeframe_explicit))
    topics = ",".join(sorted(intent.topics))
    risk = ",".join(sorted(intent.risk_terms))

    # Canonical format with pipe separators
    canonical = f"{archetypes}|{indicators}|{timeframes}|{topics}|{risk}"

    # SHA256 hash
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assess_coverage(
    scores: list[float],
    intent: MatchIntent,
    top_k: int = 10,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    weak_best_score: float = DEFAULT_WEAK_BEST_SCORE,
    low_confidence: float = DEFAULT_LOW_CONFIDENCE,
) -> CoverageAssessment:
    """
    Assess coverage quality based on match scores and intent.

    Args:
        scores: List of match scores from results (descending order)
        intent: Extracted MatchIntent
        top_k: Number of results requested
        threshold: Score threshold for "good" match
        weak_best_score: Score below which coverage is considered weak
        low_confidence: Confidence below which to flag as low signal

    Returns:
        CoverageAssessment with weak flag, metrics, and reason codes
    """
    reason_codes: list[str] = []

    # Calculate metrics
    best_score = scores[0] if scores else None
    top_k_scores = scores[:top_k] if scores else []
    avg_top_k_score = sum(top_k_scores) / len(top_k_scores) if top_k_scores else None
    num_above_threshold = sum(1 for s in scores if s >= threshold)

    # Check for low signal input first
    if intent.overall_confidence < low_confidence:
        reason_codes.append(CoverageReasonCode.LOW_SIGNAL_INPUT.value)
        # Don't flag as weak coverage for low signal - just note it
        return CoverageAssessment(
            weak=False,
            best_score=best_score,
            avg_top_k_score=avg_top_k_score,
            num_above_threshold=num_above_threshold,
            threshold=threshold,
            reason_codes=reason_codes,
            suggestions=["Input has low signal - try more specific content"],
        )

    # No results at all
    if not scores:
        reason_codes.append(CoverageReasonCode.NO_RESULTS.value)
        return CoverageAssessment(
            weak=True,
            best_score=None,
            avg_top_k_score=None,
            num_above_threshold=0,
            threshold=threshold,
            reason_codes=reason_codes,
            suggestions=generate_suggestions(intent),
        )

    # Determine weak coverage
    weak = False

    # Rule 1: best_score < 0.45
    if best_score is not None and best_score < weak_best_score:
        weak = True
        reason_codes.append(CoverageReasonCode.LOW_BEST_SCORE.value)

    # Rule 2: no results above threshold
    if num_above_threshold == 0:
        weak = True
        reason_codes.append(CoverageReasonCode.NO_RESULTS_ABOVE_THRESHOLD.value)

    # Generate suggestions if weak
    suggestions = generate_suggestions(intent) if weak else []

    return CoverageAssessment(
        weak=weak,
        best_score=best_score,
        avg_top_k_score=avg_top_k_score,
        num_above_threshold=num_above_threshold,
        threshold=threshold,
        reason_codes=reason_codes,
        suggestions=suggestions,
    )


def generate_suggestions(intent: MatchIntent) -> list[str]:
    """
    Generate actionable suggestions based on extracted intent.

    Rule-based suggestions from MatchIntent fields.

    Args:
        intent: MatchIntent with extracted trading concepts

    Returns:
        List of suggestion strings
    """
    suggestions: list[str] = []

    # Archetype-based suggestions
    archetypes = set(intent.strategy_archetypes)

    if "breakout" in archetypes:
        suggestions.append("Add breakout/opening-range breakout strategies")

    if "range_bound" in archetypes:
        suggestions.append("Add range-bound/consolidation strategies")

    if "trend_following" in archetypes:
        suggestions.append("Add trend-following/moving average strategies")

    if "mean_reversion" in archetypes:
        suggestions.append("Add mean reversion/oversold bounce strategies")

    if "momentum" in archetypes:
        suggestions.append("Add momentum/relative strength strategies")

    if "volatility" in archetypes:
        suggestions.append("Add volatility expansion/squeeze strategies")

    # Indicator-based suggestions
    indicators = set(intent.indicators)

    if "volume" in indicators:
        suggestions.append("Add volume-based confirmation scripts")

    if "bollinger" in indicators:
        suggestions.append("Add Bollinger Bands strategies")

    if "rsi" in indicators:
        suggestions.append("Add RSI-based strategies")

    if "vwap" in indicators:
        suggestions.append("Add VWAP strategies")

    # Topic-based suggestions
    topics = set(t.lower() for t in intent.topics)

    if "options" in topics or "option" in topics:
        suggestions.append("Add options-related strategies")

    if "crypto" in topics or "bitcoin" in topics:
        suggestions.append("Add crypto-specific strategies")

    if "forex" in topics or "fx" in topics:
        suggestions.append("Add forex strategies")

    # Timeframe-based suggestions
    timeframes = set(intent.timeframe_buckets)

    if "scalp" in timeframes:
        suggestions.append("Add scalping strategies (1m-5m timeframes)")

    if "swing" in timeframes:
        suggestions.append("Add swing trading strategies")

    # Combination suggestions
    if "breakout" in archetypes and "volume" in indicators:
        # Remove individual suggestions if combination applies
        suggestions = [
            s
            for s in suggestions
            if s
            not in [
                "Add breakout/opening-range breakout strategies",
                "Add volume-based confirmation scripts",
            ]
        ]
        suggestions.insert(0, "Add breakout + volume confirmation strategies")

    if "range_bound" in archetypes and "volume" in indicators:
        suggestions = [
            s
            for s in suggestions
            if s
            not in [
                "Add range-bound/consolidation strategies",
                "Add volume-based confirmation scripts",
            ]
        ]
        suggestions.insert(0, "Add range-bound + volume confirmation scripts")

    # Fallback suggestion
    if not suggestions:
        suggestions.append("Import more Pine scripts into the store")

    # Limit to top 3 most relevant
    return suggestions[:3]
