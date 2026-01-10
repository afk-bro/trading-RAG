"""Epsilon-aware tie-break comparator for KB candidates.

Implements ranking rules for cases where primary scores are within epsilon.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cmp_to_key
from typing import Optional


EPSILON = 0.02
CURRENT_REGIME_SCHEMA = "regime_v1"


@dataclass
class ScoredCandidate:
    """A candidate with score and metadata for tie-breaking.

    Attributes:
        source_id: Unique identifier for the trial
        score: Primary objective score (e.g., sharpe_oos)
        kb_status: Current KB status (promoted, candidate)
        regime_schema_version: Schema version for regime data
        kb_promoted_at: When the trial was promoted (if ever)
        created_at: When the trial was created
    """

    source_id: str
    score: float
    kb_status: str
    regime_schema_version: Optional[str]
    kb_promoted_at: Optional[datetime]
    created_at: datetime


def _schema_rank(version: Optional[str]) -> int:
    """Rank schema versions for tie-breaking.

    Returns:
        0 for current schema (best)
        1 for other known schema
        2 for null/missing (worst)
    """
    if version == CURRENT_REGIME_SCHEMA:
        return 0
    if version is not None:
        return 1
    return 2


def compare_candidates(a: ScoredCandidate, b: ScoredCandidate) -> int:
    """Compare two candidates for ranking.

    Tie-break rules (applied when scores within epsilon):
    1. Primary score - higher is better
    2. promoted > candidate - human curation signal
    3. Current schema > other > null - prefer compatible
    4. Higher kb_promoted_at - recent curation
    5. Newer created_at - recency tiebreaker

    Args:
        a: First candidate
        b: Second candidate

    Returns:
        -1 if a should rank higher
        1 if b should rank higher
        0 if equal
    """
    # Rule 1: Primary score (higher is better)
    if abs(a.score - b.score) > EPSILON:
        return -1 if a.score > b.score else 1

    # Within epsilon - apply tie-breaks

    # Rule 2: promoted > candidate
    if a.kb_status == "promoted" and b.kb_status != "promoted":
        return -1
    if b.kb_status == "promoted" and a.kb_status != "promoted":
        return 1

    # Rule 3: Schema preference
    a_rank = _schema_rank(a.regime_schema_version)
    b_rank = _schema_rank(b.regime_schema_version)
    if a_rank != b_rank:
        return -1 if a_rank < b_rank else 1

    # Rule 4: Recent promotion (higher = more recent = better)
    min_dt = datetime.min.replace(tzinfo=timezone.utc)
    a_promoted = a.kb_promoted_at or min_dt
    b_promoted = b.kb_promoted_at or min_dt
    if a_promoted != b_promoted:
        return -1 if a_promoted > b_promoted else 1

    # Rule 5: Recency (newer = better)
    if a.created_at != b.created_at:
        return -1 if a.created_at > b.created_at else 1

    return 0


def rank_candidates(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Rank candidates using the epsilon-aware comparator.

    Args:
        candidates: List of scored candidates

    Returns:
        Sorted list with best candidates first
    """
    return sorted(candidates, key=cmp_to_key(compare_candidates))


def candidates_within_epsilon(a: ScoredCandidate, b: ScoredCandidate) -> bool:
    """Check if two candidates have scores within epsilon.

    Args:
        a: First candidate
        b: Second candidate

    Returns:
        True if scores are within epsilon
    """
    return abs(a.score - b.score) <= EPSILON
