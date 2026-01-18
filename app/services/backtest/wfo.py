"""Walk-Forward Optimization (WFO) for backtests.

WFO validates strategy robustness by running parameter tuning across
rolling time windows (folds), then aggregating results to find params
that perform consistently across different market conditions.

Key concepts:
- Fold: A train/test window pair
- Candidate: A param set that appears in fold winners
- Coverage: % of folds where a candidate appears in top-K
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)

# Minimum coverage requirement for candidate eligibility
MIN_COVERAGE_RATIO = 0.6  # Must appear in ≥60% of folds


class InsufficientDataError(Exception):
    """Raised when available data cannot produce minimum required folds."""

    pass


@dataclass
class WFOConfig:
    """Configuration for walk-forward optimization.

    Attributes:
        train_days: Days in training window (IS period)
        test_days: Days in test window (OOS period)
        step_days: Days to step forward between folds
        min_folds: Minimum number of folds required
        start_ts: Optional start of data range (uses available if None)
        end_ts: Optional end of data range (uses available if None)
        leaderboard_top_k: Number of top params to extract from each fold
        allow_partial: If True, continue even if some child tunes fail
    """

    train_days: int
    test_days: int
    step_days: int
    min_folds: int = 3
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    leaderboard_top_k: int = 10
    allow_partial: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if self.train_days <= 0:
            raise ValueError("train_days must be positive")
        if self.test_days <= 0:
            raise ValueError("test_days must be positive")
        if self.step_days <= 0:
            raise ValueError("step_days must be positive")
        if self.min_folds < 1:
            raise ValueError("min_folds must be at least 1")
        if self.leaderboard_top_k < 1:
            raise ValueError("leaderboard_top_k must be at least 1")


@dataclass
class Fold:
    """A single train/test window pair.

    Uses half-open intervals: [train_start, train_end) [test_start, test_end)
    This means train_end == test_start (contiguous windows).

    Attributes:
        index: Fold number (0-indexed)
        train_start: Start of training window (inclusive)
        train_end: End of training window (exclusive) = test_start
        test_start: Start of test window (inclusive) = train_end
        test_end: End of test window (exclusive)
    """

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self):
        """Validate fold boundaries."""
        if self.train_start >= self.train_end:
            raise ValueError("train_start must be before train_end")
        if self.test_start >= self.test_end:
            raise ValueError("test_start must be before test_end")
        if self.train_end != self.test_start:
            raise ValueError("train_end must equal test_start (contiguous)")

    @property
    def train_days(self) -> int:
        """Duration of training window in days."""
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        """Duration of test window in days."""
        return (self.test_end - self.test_start).days


@dataclass
class WFOCandidateMetrics:
    """Aggregated metrics for a parameter set across folds.

    Attributes:
        params: The parameter values
        params_hash: Deterministic hash for deduplication
        mean_oos: Mean OOS score across folds
        median_oos: Median OOS score across folds
        worst_fold_oos: Lowest OOS score in any fold
        stddev_oos: Standard deviation of OOS scores
        pct_top_k: Percentage of folds where this param was in top-K
        fold_count: Number of folds where this param appeared in top-K
        total_folds: Total folds in the WFO run
        regime_tags: Union of regime tags from all appearances
        fold_scores: Individual OOS scores per fold (for diagnostics)
    """

    params: dict[str, Any]
    params_hash: str
    mean_oos: float
    median_oos: float
    worst_fold_oos: float
    stddev_oos: float
    pct_top_k: float
    fold_count: int
    total_folds: int
    regime_tags: list[str] = field(default_factory=list)
    fold_scores: list[tuple[int, float]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Coverage ratio (fold_count / total_folds)."""
        return self.fold_count / self.total_folds if self.total_folds > 0 else 0.0

    @property
    def meets_coverage_threshold(self) -> bool:
        """Whether this candidate meets minimum coverage requirement."""
        return self.coverage >= MIN_COVERAGE_RATIO


@dataclass
class WFOResult:
    """Result of a walk-forward optimization run.

    Attributes:
        wfo_id: Unique identifier for this WFO run
        status: Current status (pending, running, succeeded, failed, canceled)
        n_folds: Total number of folds
        folds_completed: Number of folds that completed successfully
        folds_failed: Number of folds that failed
        candidates: Aggregated candidate metrics (sorted by selection_score)
        best_params: Parameters of top candidate (if any eligible)
        best_candidate: Full metrics of top candidate
        child_tune_ids: IDs of child tune jobs
        warnings: Any warnings generated during the run
    """

    wfo_id: UUID
    status: str
    n_folds: int
    folds_completed: int
    folds_failed: int
    candidates: list[WFOCandidateMetrics]
    best_params: Optional[dict[str, Any]]
    best_candidate: Optional[WFOCandidateMetrics]
    child_tune_ids: list[UUID] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generate_folds(
    config: WFOConfig,
    available_range: tuple[datetime, datetime],
) -> list[Fold]:
    """Generate rolling train/test folds for walk-forward optimization.

    Creates a sequence of contiguous train/test window pairs that step
    forward through the available data range.

    Args:
        config: WFO configuration with window sizes and step
        available_range: (start, end) of available OHLCV data

    Returns:
        List of Fold objects

    Raises:
        InsufficientDataError: If available data cannot produce min_folds
    """
    data_start, data_end = available_range

    # Determine effective range (respect config bounds if provided)
    effective_start = (
        max(config.start_ts, data_start) if config.start_ts else data_start
    )
    effective_end = min(config.end_ts, data_end) if config.end_ts else data_end

    # Ensure timezone-aware
    if effective_start.tzinfo is None:
        effective_start = effective_start.replace(tzinfo=timezone.utc)
    if effective_end.tzinfo is None:
        effective_end = effective_end.replace(tzinfo=timezone.utc)

    total_required_days = config.train_days + config.test_days
    available_days = (effective_end - effective_start).days

    if available_days < total_required_days:
        raise InsufficientDataError(
            f"Need {total_required_days} days for one fold, "
            f"but only {available_days} days available"
        )

    folds = []
    cursor = effective_start
    fold_index = 0

    while True:
        train_end = cursor + timedelta(days=config.train_days)
        test_end = train_end + timedelta(days=config.test_days)

        # Stop if this fold would exceed available data
        if test_end > effective_end:
            break

        folds.append(
            Fold(
                index=fold_index,
                train_start=cursor,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
        )

        fold_index += 1
        cursor += timedelta(days=config.step_days)

    if len(folds) < config.min_folds:
        raise InsufficientDataError(
            f"Only {len(folds)} folds possible with current config, "
            f"but min_folds={config.min_folds} required. "
            f"Consider reducing train_days ({config.train_days}), "
            f"test_days ({config.test_days}), or min_folds."
        )

    logger.info(
        "Generated WFO folds",
        n_folds=len(folds),
        train_days=config.train_days,
        test_days=config.test_days,
        step_days=config.step_days,
        effective_start=effective_start.isoformat(),
        effective_end=effective_end.isoformat(),
    )

    return folds


def compute_params_hash(params: dict[str, Any]) -> str:
    """Compute deterministic hash for parameter dict.

    Used to identify same param set across different folds.
    """
    import hashlib
    import json

    # Canonical JSON with sorted keys
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclass
class FoldLeaderboardEntry:
    """Entry from a fold's leaderboard (top-K results)."""

    fold_index: int
    params: dict[str, Any]
    params_hash: str
    oos_score: float
    is_score: Optional[float] = None
    regime_tags: list[str] = field(default_factory=list)


@dataclass
class FoldResult:
    """Result of a single fold's tune job.

    Attributes:
        fold_index: Index of this fold
        tune_id: UUID of the child tune job
        status: Job status (succeeded, failed, etc.)
        leaderboard: Top-K results from this fold
        error: Error message if failed
    """

    fold_index: int
    tune_id: UUID
    status: str
    leaderboard: list[FoldLeaderboardEntry] = field(default_factory=list)
    error: Optional[str] = None


# Penalty threshold for worst-fold OOS score
WORST_FOLD_THRESHOLD = 0.0


def compute_selection_score(candidate: WFOCandidateMetrics) -> float:
    """Compute selection score for WFO candidate ranking.

    Score formula:
        score = (mean_oos - 0.5 * stddev_oos
                 - 0.3 * max(0, threshold - worst_fold_oos)) * coverage

    Higher is better. Penalizes:
    - High variance (via stddev)
    - Poor worst-case performance (via worst_fold penalty)
    - Low coverage (multiplicative factor)

    Args:
        candidate: Candidate metrics to score

    Returns:
        Selection score (higher = better)
    """
    # Base score from mean OOS
    score = candidate.mean_oos

    # Penalty for variance (consistency matters)
    score -= 0.5 * candidate.stddev_oos

    # Penalty for poor worst-fold performance
    worst_fold_penalty = max(0, WORST_FOLD_THRESHOLD - candidate.worst_fold_oos)
    score -= 0.3 * worst_fold_penalty

    # Scale by coverage (must appear in enough folds)
    score *= candidate.coverage

    return score


def aggregate_wfo_candidates(
    fold_results: list[FoldResult],
    top_k: int = 10,
) -> list[WFOCandidateMetrics]:
    """Aggregate fold results into WFO candidate metrics.

    Collects leaderboard entries from all folds, groups by params_hash,
    computes aggregate metrics, filters by coverage, and ranks by selection score.

    Args:
        fold_results: Results from all fold tune jobs
        top_k: Number of top candidates per fold that were extracted

    Returns:
        List of WFOCandidateMetrics sorted by selection_score descending
    """
    import statistics

    # Only consider successful folds
    successful_folds = [f for f in fold_results if f.status == "succeeded"]
    total_folds = len(successful_folds)

    if total_folds == 0:
        logger.warning("No successful folds to aggregate")
        return []

    # Collect all leaderboard entries, grouped by params_hash
    entries_by_hash: dict[str, list[FoldLeaderboardEntry]] = {}
    params_by_hash: dict[str, dict[str, Any]] = {}

    for fold in successful_folds:
        for entry in fold.leaderboard:
            h = entry.params_hash
            if h not in entries_by_hash:
                entries_by_hash[h] = []
                params_by_hash[h] = entry.params
            entries_by_hash[h].append(entry)

    # Compute metrics for each unique param set
    candidates: list[WFOCandidateMetrics] = []

    for params_hash, entries in entries_by_hash.items():
        oos_scores = [e.oos_score for e in entries]
        fold_count = len(entries)

        # Compute aggregate metrics
        mean_oos = statistics.mean(oos_scores)
        median_oos = statistics.median(oos_scores)
        worst_fold_oos = min(oos_scores)
        stddev_oos = statistics.stdev(oos_scores) if len(oos_scores) > 1 else 0.0

        # Coverage = fraction of folds where this param appeared in top-K
        pct_top_k = fold_count / total_folds

        # Collect regime tags (union of all folds)
        all_regime_tags: set[str] = set()
        for entry in entries:
            all_regime_tags.update(entry.regime_tags)

        # Collect fold scores for diagnostics
        fold_scores = [(e.fold_index, e.oos_score) for e in entries]

        candidate = WFOCandidateMetrics(
            params=params_by_hash[params_hash],
            params_hash=params_hash,
            mean_oos=mean_oos,
            median_oos=median_oos,
            worst_fold_oos=worst_fold_oos,
            stddev_oos=stddev_oos,
            pct_top_k=pct_top_k,
            fold_count=fold_count,
            total_folds=total_folds,
            regime_tags=sorted(all_regime_tags),
            fold_scores=fold_scores,
        )
        candidates.append(candidate)

    # Filter by coverage threshold
    eligible_candidates = [c for c in candidates if c.meets_coverage_threshold]

    logger.info(
        "WFO aggregation complete",
        total_unique_params=len(candidates),
        eligible_candidates=len(eligible_candidates),
        total_folds=total_folds,
        coverage_threshold=MIN_COVERAGE_RATIO,
    )

    # Sort by selection score (descending)
    eligible_candidates.sort(key=compute_selection_score, reverse=True)

    return eligible_candidates


def extract_leaderboard_from_tune_result(
    fold_index: int,
    tune_result: dict[str, Any],
    top_k: int = 10,
) -> list[FoldLeaderboardEntry]:
    """Extract top-K leaderboard entries from a tune result.

    Args:
        fold_index: Index of the fold
        tune_result: Result dict from TuneJob
        top_k: Number of top entries to extract

    Returns:
        List of FoldLeaderboardEntry objects
    """
    leaderboard = tune_result.get("leaderboard", [])
    entries: list[FoldLeaderboardEntry] = []

    for item in leaderboard[:top_k]:
        params = item.get("params", {})
        entries.append(
            FoldLeaderboardEntry(
                fold_index=fold_index,
                params=params,
                params_hash=compute_params_hash(params),
                oos_score=item.get("score_oos") or item.get("score", 0.0),
                is_score=item.get("score_is"),
                regime_tags=item.get("regime_tags", []),
            )
        )

    return entries
