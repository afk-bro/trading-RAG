"""
KB reranking module for scoring candidates.

Handles:
- Jaccard similarity between regime tags
- Regime distance (optional)
- Combined rerank scoring
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import structlog

from app.services.kb.types import RegimeSnapshot
from app.services.kb.retrieval import RetrievalCandidate

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Weights for rerank scoring
SIMILARITY_WEIGHT = 0.5  # Vector similarity
JACCARD_WEIGHT = 0.3  # Tag overlap
OBJECTIVE_WEIGHT = 0.2  # Performance score

# Fallback when both query and candidate have empty tags
EMPTY_TAGS_JACCARD = 1.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RerankedCandidate:
    """A candidate with reranking scores attached."""

    point_id: str
    payload: dict
    similarity_score: float  # Original vector similarity
    jaccard_score: float  # Tag overlap score
    rerank_score: float  # Combined rerank score
    used_regime_source: Literal["oos", "is", "none"]  # Which regime was used
    _relaxed: bool = False
    _metadata_only: bool = False


@dataclass
class RerankResult:
    """Result of reranking operation."""

    candidates: list[RerankedCandidate]
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Similarity Functions
# =============================================================================


def jaccard(tags_a: list[str], tags_b: list[str]) -> float:
    """
    Compute Jaccard similarity between two tag sets.

    Args:
        tags_a: First tag list
        tags_b: Second tag list

    Returns:
        Jaccard similarity (0-1), or 1.0 if both empty
    """
    if not tags_a and not tags_b:
        return EMPTY_TAGS_JACCARD

    set_a = set(tags_a)
    set_b = set(tags_b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return EMPTY_TAGS_JACCARD

    return intersection / union


def regime_distance(
    regime_a: Optional[RegimeSnapshot],
    regime_b: Optional[RegimeSnapshot],
) -> float:
    """
    Compute distance between two regime snapshots.

    Uses key numeric features for similarity:
    - atr_pct, trend_strength, trend_dir, rsi, efficiency_ratio

    Args:
        regime_a: First regime snapshot
        regime_b: Second regime snapshot

    Returns:
        Distance (0 = identical, higher = more different)
        Returns 1.0 if either regime is None
    """
    if regime_a is None or regime_b is None:
        return 1.0

    # Normalize features to 0-1 range
    def normalize_atr(val: float) -> float:
        # ATR typically 0-0.10, clip and normalize
        return min(val / 0.10, 1.0)

    def normalize_trend(val: int) -> float:
        # -1, 0, 1 -> 0, 0.5, 1
        return (val + 1) / 2

    def normalize_rsi(val: float) -> float:
        return val / 100

    # Feature vectors
    features_a = [
        normalize_atr(regime_a.atr_pct),
        regime_a.trend_strength,
        normalize_trend(regime_a.trend_dir),
        normalize_rsi(regime_a.rsi),
        regime_a.efficiency_ratio,
    ]

    features_b = [
        normalize_atr(regime_b.atr_pct),
        regime_b.trend_strength,
        normalize_trend(regime_b.trend_dir),
        normalize_rsi(regime_b.rsi),
        regime_b.efficiency_ratio,
    ]

    # Euclidean distance normalized by sqrt(n_features)
    import math
    sq_diff = sum((a - b) ** 2 for a, b in zip(features_a, features_b))
    distance = math.sqrt(sq_diff / len(features_a))

    return distance


# =============================================================================
# Reranking Logic
# =============================================================================


def get_candidate_regime_tags(
    payload: dict,
) -> tuple[list[str], Literal["oos", "is", "none"]]:
    """
    Get regime tags from candidate payload.

    Prefers OOS regime, falls back to IS.

    Args:
        payload: Candidate payload dict

    Returns:
        (tags, source) where source is "oos", "is", or "none"
    """
    # Try OOS first
    regime_oos = payload.get("regime_oos")
    if regime_oos and isinstance(regime_oos, dict):
        tags = regime_oos.get("regime_tags", [])
        if tags:
            return tags, "oos"

    # Fall back to IS
    regime_is = payload.get("regime_is")
    if regime_is and isinstance(regime_is, dict):
        tags = regime_is.get("regime_tags", [])
        if tags:
            return tags, "is"

    # Check top-level regime_tags
    top_tags = payload.get("regime_tags", [])
    if top_tags:
        # Determine source from has_oos flag
        source = "oos" if payload.get("has_oos") else "is"
        return top_tags, source

    return [], "none"


def compute_rerank_score(
    similarity: float,
    jaccard_score: float,
    objective_score: float,
    max_objective: float,
    is_relaxed: bool = False,
    is_metadata_only: bool = False,
) -> float:
    """
    Compute combined rerank score.

    Args:
        similarity: Vector similarity (0-1)
        jaccard_score: Tag overlap (0-1)
        objective_score: Candidate's objective score
        max_objective: Maximum objective score for normalization
        is_relaxed: Penalty for relaxed filter candidates
        is_metadata_only: Penalty for metadata-only candidates

    Returns:
        Combined rerank score (higher = better)
    """
    # Normalize objective score
    norm_objective = 0.5  # Default if no max
    if max_objective > 0:
        norm_objective = min(objective_score / max_objective, 1.0) if objective_score else 0.0

    # Base score
    score = (
        SIMILARITY_WEIGHT * similarity +
        JACCARD_WEIGHT * jaccard_score +
        OBJECTIVE_WEIGHT * norm_objective
    )

    # Apply penalties
    if is_metadata_only:
        score *= 0.5  # Heavy penalty - no vector similarity
    elif is_relaxed:
        score *= 0.8  # Moderate penalty - lower quality

    return score


def rerank_candidates(
    candidates: list[RetrievalCandidate],
    query_tags: list[str],
    query_regime: Optional[RegimeSnapshot] = None,
) -> RerankResult:
    """
    Rerank candidates by combined score.

    Args:
        candidates: List of retrieval candidates
        query_tags: Query regime tags for Jaccard
        query_regime: Optional full regime for distance (unused in v1)

    Returns:
        RerankResult with scored and sorted candidates
    """
    warnings: list[str] = []

    if not candidates:
        return RerankResult(candidates=[], warnings=warnings)

    # Find max objective score for normalization
    max_objective = max(
        (c.payload.get("objective_score") or 0.0 for c in candidates),
        default=1.0,
    )
    if max_objective <= 0:
        max_objective = 1.0

    reranked: list[RerankedCandidate] = []

    for candidate in candidates:
        # Get candidate tags
        cand_tags, regime_source = get_candidate_regime_tags(candidate.payload)

        # Compute Jaccard
        jaccard_score = jaccard(query_tags, cand_tags)

        # Compute rerank score
        objective_score = candidate.payload.get("objective_score") or 0.0
        rerank_score = compute_rerank_score(
            similarity=candidate.similarity_score,
            jaccard_score=jaccard_score,
            objective_score=objective_score,
            max_objective=max_objective,
            is_relaxed=candidate._relaxed,
            is_metadata_only=candidate._metadata_only,
        )

        reranked.append(RerankedCandidate(
            point_id=candidate.point_id,
            payload=candidate.payload,
            similarity_score=candidate.similarity_score,
            jaccard_score=jaccard_score,
            rerank_score=rerank_score,
            used_regime_source=regime_source,
            _relaxed=candidate._relaxed,
            _metadata_only=candidate._metadata_only,
        ))

    # Sort by rerank score descending, with tiebreakers for stability
    # Tiebreakers: objective_score desc, point_id asc (guaranteed unique)
    reranked.sort(
        key=lambda x: (
            -x.rerank_score,  # Descending
            -(x.payload.get("objective_score") or 0),  # Descending
            x.point_id,  # Ascending for absolute determinism (unique)
        ),
    )

    # Add warning if many have empty tags
    empty_tag_count = sum(1 for c in reranked if c.used_regime_source == "none")
    if empty_tag_count > len(reranked) * 0.5:
        warnings.append("many_candidates_missing_regime_tags")

    logger.info(
        "Reranking complete",
        count=len(reranked),
        empty_tag_count=empty_tag_count,
        top_score=reranked[0].rerank_score if reranked else 0,
    )

    return RerankResult(candidates=reranked, warnings=warnings)
