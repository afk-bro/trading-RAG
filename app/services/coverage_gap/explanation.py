"""LLM-powered explanation service for strategy recommendations."""

from dataclasses import dataclass
from typing import Any, Optional

import structlog

from app.services.llm_base import LLMResponse
from app.services.llm_factory import get_llm, get_llm_status

logger = structlog.get_logger(__name__)


@dataclass
class StrategyExplanation:
    """Result of explaining why a strategy matches an intent."""

    strategy_id: str
    strategy_name: str
    explanation: str
    model: str
    provider: str
    latency_ms: Optional[float] = None


class ExplanationError(Exception):
    """Error generating explanation."""


async def generate_strategy_explanation(
    intent_json: dict[str, Any],
    strategy: dict[str, Any],
    matched_tags: list[str],
    match_score: Optional[float] = None,
) -> StrategyExplanation:
    """
    Generate an LLM-powered explanation of why a strategy matches a user's intent.

    Args:
        intent_json: The MatchIntent data (archetypes, indicators, timeframes, etc.)
        strategy: Strategy dict with name, description, tags, backtest_summary
        matched_tags: List of tags that overlap between intent and strategy
        match_score: Optional similarity score (0-1)

    Returns:
        StrategyExplanation with generated text

    Raises:
        ExplanationError: If LLM is not configured or generation fails
    """
    llm = get_llm()
    if not llm:
        status = get_llm_status()
        raise ExplanationError(
            f"LLM not configured (enabled={status.enabled}, "
            f"provider={status.provider_config})"
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

    # Build prompt
    system_prompt = """You are a trading strategy analyst helping users understand
why a particular strategy might fit their trading needs.

Your explanations should be:
- Concise (2-4 sentences)
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

Write a brief (2-4 sentence) explanation of why this strategy aligns with what the user
is looking for. Focus on the practical fit."""

    log = logger.bind(
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        matched_tags=matched_tags,
        intent_archetypes=archetypes,
    )
    log.info("generating_strategy_explanation")

    try:
        response: LLMResponse = await llm.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
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
            model=response.model,
            provider=response.provider,
            latency_ms=response.latency_ms,
        )

    except Exception as e:
        log.error("explanation_generation_failed", error=str(e))
        raise ExplanationError(f"Failed to generate explanation: {e}") from e
