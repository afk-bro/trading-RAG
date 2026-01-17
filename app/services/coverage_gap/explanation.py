"""LLM-powered explanation service for strategy recommendations."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import structlog

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None  # type: ignore

from app.services.llm_base import (
    LLMResponse,
    LLMError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAPIError,
)
from app.services.llm_factory import get_llm, get_llm_status

logger = structlog.get_logger(__name__)

# Verbosity types
Verbosity = Literal["short", "detailed"]

# LLM request timeout (seconds)
LLM_EXPLANATION_TIMEOUT = 15.0

# User-safe error messages (don't expose internal details)
USER_SAFE_ERRORS = {
    "llm_timeout": "LLM request timed out",
    "llm_rate_limit": "LLM rate limited, please retry",
    "llm_error": "LLM provider error",
    "llm_unconfigured": "LLM not configured",
}

# Reason codes for degraded responses
ReasonCode = Literal["llm_timeout", "llm_rate_limit", "llm_error", "llm_unconfigured"]


@dataclass
class StrategyExplanation:
    """Result of explaining why a strategy matches an intent."""

    strategy_id: str
    strategy_name: str
    explanation: str
    confidence_qualifier: str
    model: str
    provider: str
    verbosity: Verbosity = "short"
    latency_ms: Optional[float] = None
    cache_hit: bool = False
    generated_at: Optional[str] = None
    # Degraded mode fields
    degraded: bool = False
    reason_code: Optional[ReasonCode] = None
    error: Optional[str] = None


@dataclass
class CachedExplanation:
    """Cached explanation entry for storage in match_runs.explanations_cache."""

    explanation: str
    confidence_qualifier: str
    model: str
    provider: str
    verbosity: Verbosity
    latency_ms: Optional[float]
    generated_at: str
    strategy_updated_at: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSONB storage."""
        return {
            "explanation": self.explanation,
            "confidence_qualifier": self.confidence_qualifier,
            "model": self.model,
            "provider": self.provider,
            "verbosity": self.verbosity,
            "latency_ms": self.latency_ms,
            "generated_at": self.generated_at,
            "strategy_updated_at": self.strategy_updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CachedExplanation":
        """Create from JSONB dict."""
        return cls(
            explanation=data.get("explanation", ""),
            confidence_qualifier=data.get("confidence_qualifier", ""),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
            verbosity=data.get("verbosity", "short"),
            latency_ms=data.get("latency_ms"),
            generated_at=data.get("generated_at", ""),
            strategy_updated_at=data.get("strategy_updated_at"),
        )


class ExplanationError(Exception):
    """Error generating explanation."""


def _make_fallback_explanation(
    strategy_id: str,
    strategy_name: str,
    reason_code: ReasonCode,
    matched_tags: list[str],
    match_score: Optional[float] = None,
    verbosity: Verbosity = "short",
) -> StrategyExplanation:
    """
    Create a fallback explanation when LLM is unavailable.

    Returns a degraded response with deterministic confidence qualifier
    but fallback explanation text.
    """
    # Still compute confidence qualifier (it's deterministic, no LLM needed)
    confidence_qualifier = compute_confidence_qualifier(
        matched_tags=matched_tags,
        match_score=match_score,
        has_backtest=False,  # Conservative: assume no backtest info
        backtest_validated=False,
    )

    # Build fallback explanation based on available data
    if matched_tags:
        explanation = (
            f"This strategy matches your search based on {len(matched_tags)} "
            f"overlapping tag{'s' if len(matched_tags) > 1 else ''}: "
            f"{', '.join(matched_tags[:3])}{'...' if len(matched_tags) > 3 else ''}. "
            "Review the strategy details to assess fit for your trading approach."
        )
    elif match_score and match_score > 0.5:
        explanation = (
            f"This strategy has {match_score:.0%} semantic similarity to your search. "
            "Review the strategy details to assess fit for your trading approach."
        )
    else:
        explanation = (
            "This strategy was identified as a potential match. "
            "Review the strategy details to assess fit for your trading approach."
        )

    return StrategyExplanation(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        explanation=explanation,
        confidence_qualifier=confidence_qualifier,
        model="fallback",
        provider="fallback",
        verbosity=verbosity,
        latency_ms=None,
        cache_hit=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
        degraded=True,
        reason_code=reason_code,
        error=USER_SAFE_ERRORS.get(reason_code, "LLM unavailable"),
    )


def compute_confidence_qualifier(
    matched_tags: list[str],
    match_score: Optional[float],
    has_backtest: bool,
    backtest_validated: bool,
) -> str:
    """
    Compute a deterministic confidence qualifier for the explanation.

    Args:
        matched_tags: Tags that overlap between intent and strategy
        match_score: Similarity score (0-1)
        has_backtest: Whether backtest_summary exists
        backtest_validated: Whether backtest status is 'validated'

    Returns:
        Confidence qualifier string (e.g., "High — based on...")
    """
    # Score components
    tag_score = min(len(matched_tags) / 3.0, 1.0)  # 3+ tags = full credit
    match_score_val = match_score if match_score is not None else 0.5
    backtest_score = 0.3 if backtest_validated else (0.1 if has_backtest else 0.0)

    # Weighted average
    confidence = (tag_score * 0.4) + (match_score_val * 0.4) + (backtest_score * 0.2)

    # Determine level
    if confidence >= 0.7:
        level = "High"
    elif confidence >= 0.4:
        level = "Medium"
    else:
        level = "Low"

    # Build reasoning
    reasons = []
    if matched_tags:
        reasons.append(
            f"{len(matched_tags)} matching tag{'s' if len(matched_tags) > 1 else ''}"
        )
    elif match_score and match_score > 0.5:
        reasons.append("semantic similarity")
    if backtest_validated:
        reasons.append("validated backtest")
    elif has_backtest:
        reasons.append("backtest available")

    if not reasons:
        reasons.append("general alignment")

    return f"Confidence: {level} — based on {', '.join(reasons)}."


async def generate_strategy_explanation(
    intent_json: dict[str, Any],
    strategy: dict[str, Any],
    matched_tags: list[str],
    match_score: Optional[float] = None,
    verbosity: Verbosity = "short",
    timeout_seconds: float = LLM_EXPLANATION_TIMEOUT,
) -> StrategyExplanation:
    """
    Generate an LLM-powered explanation of why a strategy matches a user's intent.

    This function gracefully degrades to fallback templates when:
    - LLM is not configured
    - LLM request times out
    - LLM is rate limited
    - Any other LLM error occurs

    Args:
        intent_json: The MatchIntent data (archetypes, indicators, timeframes, etc.)
        strategy: Strategy dict with name, description, tags, backtest_summary
        matched_tags: List of tags that overlap between intent and strategy
        match_score: Optional similarity score (0-1)
        verbosity: "short" (2-4 sentences) or "detailed" (2-3 paragraphs)
        timeout_seconds: Timeout for LLM request (default 15s)

    Returns:
        StrategyExplanation with generated text and confidence qualifier.
        If LLM fails, returns degraded=True with fallback text.
    """
    # Extract strategy info for fallback
    strategy_name = strategy.get("name", "Unknown Strategy")
    strategy_id = str(strategy.get("id", ""))

    # Helper to log and optionally report to Sentry
    def _log_and_capture(
        reason_code: ReasonCode,
        error: Exception,
        log_level: str = "warning",
    ) -> None:
        log_fn = getattr(logger, log_level, logger.warning)
        log_fn(
            "explanation_fallback",
            reason_code=reason_code,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            error=str(error),
            exc_info=True,
        )
        if sentry_sdk:
            sentry_sdk.set_tag("llm_failure", reason_code)
            sentry_sdk.capture_exception(error)

    # Check LLM availability
    llm = get_llm()
    if not llm:
        status = get_llm_status()
        logger.warning(
            "explanation_fallback",
            reason_code="llm_unconfigured",
            enabled=status.enabled,
            provider_config=status.provider_config,
        )
        return _make_fallback_explanation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            reason_code="llm_unconfigured",
            matched_tags=matched_tags,
            match_score=match_score,
            verbosity=verbosity,
        )

    # Extract intent components
    archetypes = intent_json.get("strategy_archetypes", [])
    indicators = intent_json.get("indicators", [])
    timeframes = intent_json.get("timeframe_buckets", [])
    explicit_tf = intent_json.get("timeframe_explicit", [])
    symbols = intent_json.get("symbols", [])
    risk_terms = intent_json.get("risk_terms", [])
    topics = intent_json.get("topics", [])

    # Extract strategy components
    strategy_name = strategy.get("name", "Unknown Strategy")
    strategy_id = str(strategy.get("id", ""))
    description = strategy.get("description", "")
    strategy_tags = strategy.get("tags", {})
    backtest = strategy.get("backtest_summary")

    # Compute confidence qualifier (deterministic)
    has_backtest = backtest is not None
    backtest_validated = (
        has_backtest and backtest is not None and backtest.get("status") == "validated"
    )
    confidence_qualifier = compute_confidence_qualifier(
        matched_tags=matched_tags,
        match_score=match_score,
        has_backtest=has_backtest,
        backtest_validated=backtest_validated,
    )

    # Build intent summary
    intent_parts = []
    if archetypes:
        intent_parts.append(f"Strategy style: {', '.join(archetypes)}")
    if indicators:
        intent_parts.append(f"Indicators: {', '.join(indicators)}")
    if timeframes or explicit_tf:
        tf_list = timeframes + explicit_tf
        intent_parts.append(f"Timeframes: {', '.join(tf_list)}")
    if symbols:
        intent_parts.append(f"Symbols: {', '.join(symbols[:5])}")
    if risk_terms:
        intent_parts.append(f"Risk management: {', '.join(risk_terms)}")
    if topics:
        intent_parts.append(f"Topics: {', '.join(topics[:5])}")

    intent_summary = (
        "\n".join(intent_parts) if intent_parts else "General trading interest"
    )

    # Build strategy summary
    strategy_parts = [f"Name: {strategy_name}"]
    if description:
        strategy_parts.append(f"Description: {description}")

    # Extract strategy tags
    strat_archetypes = strategy_tags.get("strategy_archetypes", [])
    strat_indicators = strategy_tags.get("indicators", [])
    strat_timeframes = strategy_tags.get("timeframe_buckets", [])

    if strat_archetypes:
        strategy_parts.append(f"Style: {', '.join(strat_archetypes)}")
    if strat_indicators:
        strategy_parts.append(f"Uses indicators: {', '.join(strat_indicators)}")
    if strat_timeframes:
        strategy_parts.append(f"Timeframes: {', '.join(strat_timeframes)}")

    # Add backtest summary if available
    if backtest:
        bt_status = backtest.get("status", "unknown")
        best_score = backtest.get("best_oos_score")
        max_dd = backtest.get("max_drawdown")
        if bt_status == "validated" and best_score is not None:
            strategy_parts.append(f"Backtest: validated (OOS score: {best_score:.2f})")
        if max_dd is not None:
            strategy_parts.append(f"Max drawdown: {max_dd:.1%}")

    strategy_summary = "\n".join(strategy_parts)

    # Build matched tags section
    matched_summary = (
        f"Matched tags: {', '.join(matched_tags)}"
        if matched_tags
        else "No explicit tag matches (matched by semantic similarity)"
    )
    if match_score is not None:
        matched_summary += f"\nMatch score: {match_score:.0%}"

    # Build prompt based on verbosity
    if verbosity == "detailed":
        length_instruction = """Write a detailed explanation (2-3 paragraphs) covering:
1. How the strategy's approach aligns with the user's trading style
2. Specific technical aspects that match (indicators, timeframes, risk management)
3. Any caveats or considerations the user should be aware of"""
        max_tokens = 600
    else:
        length_instruction = (
            "Write a brief (2-4 sentence) explanation of why this strategy aligns "
            "with what the user is looking for. Focus on the practical fit."
        )
        max_tokens = 300

    system_prompt = f"""You are a trading strategy analyst helping users understand
why a particular strategy might fit their trading needs.

Your explanations should be:
- {"Thorough and educational" if verbosity == "detailed" else "Concise"}
- Focused on the practical alignment between user intent and strategy capabilities
- Honest about any limitations or caveats
- Written for a trader, not a developer

Do NOT make up features the strategy doesn't have.
Do NOT promise specific returns or performance."""

    user_prompt = f"""Explain why this strategy might be a good match for the user's trading intent.

USER'S TRADING INTENT:
{intent_summary}

CANDIDATE STRATEGY:
{strategy_summary}

OVERLAP:
{matched_summary}

{length_instruction}"""

    log = logger.bind(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        matched_tags=matched_tags,
        intent_archetypes=archetypes,
        verbosity=verbosity,
    )
    log.info("generating_strategy_explanation")

    try:
        # Wrap LLM call with timeout
        response: LLMResponse = await asyncio.wait_for(
            llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            ),
            timeout=timeout_seconds,
        )

        log.info(
            "explanation_generated",
            model=response.model,
            provider=response.provider,
            latency_ms=response.latency_ms,
        )

        return StrategyExplanation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            explanation=response.text.strip(),
            confidence_qualifier=confidence_qualifier,
            model=response.model,
            provider=response.provider,
            verbosity=verbosity,
            latency_ms=response.latency_ms,
            cache_hit=False,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    except asyncio.TimeoutError as e:
        _log_and_capture("llm_timeout", e, log_level="warning")
        return _make_fallback_explanation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            reason_code="llm_timeout",
            matched_tags=matched_tags,
            match_score=match_score,
            verbosity=verbosity,
        )

    except LLMRateLimitError as e:
        _log_and_capture("llm_rate_limit", e, log_level="warning")
        return _make_fallback_explanation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            reason_code="llm_rate_limit",
            matched_tags=matched_tags,
            match_score=match_score,
            verbosity=verbosity,
        )

    except (LLMTimeoutError, LLMAPIError, LLMError) as e:
        _log_and_capture("llm_error", e, log_level="error")
        return _make_fallback_explanation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            reason_code="llm_error",
            matched_tags=matched_tags,
            match_score=match_score,
            verbosity=verbosity,
        )

    except Exception as e:
        # Unexpected errors - still fallback, but log as error
        log.exception("explanation_unexpected_error", error=str(e))
        if sentry_sdk:
            sentry_sdk.set_tag("llm_failure", "unexpected")
            sentry_sdk.capture_exception(e)
        return _make_fallback_explanation(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            reason_code="llm_error",
            matched_tags=matched_tags,
            match_score=match_score,
            verbosity=verbosity,
        )
