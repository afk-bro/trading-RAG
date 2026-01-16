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


def build_filters(
    intent: MatchIntent, request: YouTubeMatchRequest
) -> MatchFiltersApplied:
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
        symbols = intent.symbols[:3]
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
