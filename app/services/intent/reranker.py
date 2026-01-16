# app/services/intent/reranker.py
"""Intent-based reranking for Pine script matches."""

from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.services.intent.models import MatchIntent


class PineMatchResultProtocol(Protocol):
    """Protocol for PineMatchResult (allows duck typing)."""

    id: Any
    title: str
    inputs_preview: list[str]
    score: float


class RankedResult(BaseModel):
    """Reranked result with score breakdown."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: Any = Field(..., description="Original PineMatchResult")
    base_score: float = Field(..., description="Original match score")
    boost: float = Field(..., description="Intent-based boost applied")
    final_score: float = Field(..., description="Final score after boost")


def rerank(
    results: Sequence[PineMatchResultProtocol],
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

        ranked.append(
            RankedResult(
                result=r,
                base_score=r.score,
                boost=round(boost, 3),
                final_score=round(final_score, 3),
            )
        )

    # Sort: final_score desc, then id asc (deterministic)
    ranked.sort(key=lambda x: (-x.final_score, str(x.result.id)))

    return ranked
