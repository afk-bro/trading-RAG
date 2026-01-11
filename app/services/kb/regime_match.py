"""
Tiered regime matching for KB recommendations.

Provides SQL-based matching with graceful fallback:
  Tier 0: Exact regime_key match
  Tier 1: Partial match (same trend OR same vol)
  Tier 2: Distance match (regime feature similarity)
  Tier 3: Global fallback (top OOS scores)

Design: Fast SQL-first approach for exact/partial, Python distance for tier 2.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.kb.types import RegimeSnapshot

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class RegimeMatchTier(str, Enum):
    """Match tier indicating how the candidate was selected."""

    EXACT = "exact"  # regime_key == current_regime_key
    PARTIAL_TREND = "partial_trend"  # Same trend_tag, different vol_tag
    PARTIAL_VOL = "partial_vol"  # Same vol_tag, different trend_tag
    DISTANCE = "distance"  # Nearest by regime feature distance
    GLOBAL_BEST = "global_best"  # Top OOS scores regardless of regime


@dataclass
class DistanceFeatureDelta:
    """A feature's contribution to distance score."""

    name: str  # Feature name (e.g., "bb_width_pct")
    query_value: float  # Value in query snapshot
    candidate_value: float  # Value in candidate snapshot
    delta: float  # Absolute difference


@dataclass
class MatchDetail:
    """Details about how a candidate matched.

    Provides explainability for UI and debugging.
    """

    tier: RegimeMatchTier

    # For partial matches: which field matched
    matched_field: Optional[str] = None  # "trend_tag" | "vol_tag"
    matched_value: Optional[str] = None  # e.g., "uptrend"

    # For distance matches: distance metrics
    distance_score: Optional[float] = None  # Euclidean distance
    distance_rank: Optional[int] = None  # Rank among distance candidates
    distance_method: Optional[str] = None  # "euclidean" | "cosine" | "weighted"
    distance_features_version: Optional[str] = None  # e.g., "regime_v1"
    distance_top_features: Optional[list[DistanceFeatureDelta]] = (
        None  # Top contributing features
    )

    # For all: contributing features (optional, for explainability)
    contributing_features: Optional[list[str]] = None


@dataclass
class TierCaps:
    """Per-tier caps to avoid any single tier flooding the results.

    Keeps recommendations diverse and stable.
    """

    max_partial_trend: int = 10
    max_partial_vol: int = 10
    max_distance: int = 20


# Default tier caps
DEFAULT_TIER_CAPS = TierCaps()

# Distance feature version for reproducibility
DISTANCE_FEATURES_VERSION = "regime_v1"


@dataclass
class SampleContext:
    """Context about the matching process for transparency.

    Helps UI explain why a particular tier was used.
    """

    exact_count: int = 0
    partial_trend_count: int = 0
    partial_vol_count: int = 0
    distance_count: int = 0
    global_count: int = 0

    tier_used: RegimeMatchTier = RegimeMatchTier.EXACT
    tiers_attempted: list[str] = field(default_factory=list)

    # Thresholds used
    min_samples: int = 5
    k: int = 20

    # Per-tier caps applied
    tier_caps: Optional[TierCaps] = None

    # Why we fell back (if applicable)
    fallback_reason: Optional[str] = None


@dataclass
class RegimeMatchCandidate:
    """A candidate from tiered regime matching.

    Contains all fields needed for recommendation + match explainability.
    """

    # Identity
    tune_id: UUID
    run_id: Optional[UUID] = None
    strategy_entity_id: Optional[UUID] = None

    # Parameters
    best_params: Optional[dict] = None

    # Performance
    best_oos_score: Optional[float] = None

    # Regime info
    regime_key: Optional[str] = None
    trend_tag: Optional[str] = None
    vol_tag: Optional[str] = None
    efficiency_tag: Optional[str] = None

    # Match metadata
    match_detail: Optional[MatchDetail] = None


@dataclass
class TieredMatchResult:
    """Result of tiered matching operation."""

    candidates: list[RegimeMatchCandidate]
    sample_context: SampleContext
    query_regime_key: Optional[str] = None


# =============================================================================
# Regime Feature Extraction
# =============================================================================

# Features used for distance computation
# These are the numeric fields from RegimeSnapshot that define the regime space
REGIME_FEATURES = [
    "atr_pct",
    "std_pct",
    "bb_width_pct",
    "trend_strength",
    "trend_dir",
    "zscore",
    "rsi",
    "efficiency_ratio",
]


def extract_regime_features(snapshot: RegimeSnapshot) -> dict[str, float]:
    """Extract numeric features from RegimeSnapshot for distance computation.

    Returns dict of {feature_name: value} for all available features.
    NaN/None values are excluded.
    """
    features = {}
    for feat in REGIME_FEATURES:
        val = getattr(snapshot, feat, None)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            features[feat] = float(val)
    return features


def compute_regime_distance(
    query_features: dict[str, float],
    candidate_features: dict[str, float],
) -> float:
    """Compute Euclidean distance between two regime feature vectors.

    Uses standardized features (assumes features are already normalized).
    Only compares features present in both vectors.

    Returns:
        Euclidean distance (0 = identical, higher = more different)
    """
    common_features = set(query_features.keys()) & set(candidate_features.keys())
    if not common_features:
        return float("inf")

    sum_sq = 0.0
    for feat in common_features:
        diff = query_features[feat] - candidate_features[feat]
        sum_sq += diff * diff

    return math.sqrt(sum_sq)


def compute_regime_distance_with_features(
    query_features: dict[str, float],
    candidate_features: dict[str, float],
    top_n: int = 3,
) -> tuple[float, list[DistanceFeatureDelta]]:
    """Compute distance and return top contributing features.

    Args:
        query_features: Query regime features
        candidate_features: Candidate regime features
        top_n: Number of top contributing features to return

    Returns:
        Tuple of (distance, list of top feature deltas sorted by |delta| desc)
    """
    common_features = set(query_features.keys()) & set(candidate_features.keys())
    if not common_features:
        return float("inf"), []

    feature_deltas: list[DistanceFeatureDelta] = []
    sum_sq = 0.0

    for feat in common_features:
        q_val = query_features[feat]
        c_val = candidate_features[feat]
        diff = q_val - c_val
        sum_sq += diff * diff
        feature_deltas.append(
            DistanceFeatureDelta(
                name=feat,
                query_value=q_val,
                candidate_value=c_val,
                delta=abs(diff),
            )
        )

    distance = math.sqrt(sum_sq)

    # Sort by delta descending, take top N
    feature_deltas.sort(key=lambda x: x.delta, reverse=True)
    top_features = feature_deltas[:top_n]

    return distance, top_features


# =============================================================================
# Tiered Matching Implementation
# =============================================================================


class TieredRegimeMatcher:
    """
    SQL-based tiered regime matcher.

    Queries backtest_tunes table directly with fallback ladder:
    Exact → Partial → Distance → Global

    Design decision: Strategy-scoped first to avoid mixing unrelated strategies.
    """

    def __init__(self, pool):
        """Initialize with database connection pool.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def match(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
        regime_key: Optional[str] = None,
        trend_tag: Optional[str] = None,
        vol_tag: Optional[str] = None,
        query_snapshot: Optional[RegimeSnapshot] = None,
        min_samples: int = 5,
        k: int = 20,
        tier_caps: Optional[TierCaps] = None,
    ) -> TieredMatchResult:
        """
        Find candidates using tiered matching.

        Args:
            workspace_id: Workspace to search in
            strategy_entity_id: Optional strategy filter (recommended)
            regime_key: Current regime key for exact match
            trend_tag: Current trend tag for partial match
            vol_tag: Current vol tag for partial match
            query_snapshot: Current regime snapshot for distance match
            min_samples: Minimum candidates before falling back
            k: Maximum candidates to return
            tier_caps: Per-tier caps to avoid flooding (uses defaults if None)

        Returns:
            TieredMatchResult with candidates and context
        """
        caps = tier_caps or DEFAULT_TIER_CAPS
        context = SampleContext(min_samples=min_samples, k=k, tier_caps=caps)
        all_candidates: list[RegimeMatchCandidate] = []
        seen_tune_ids: set[UUID] = set()

        # Tier 0: Exact match
        context.tiers_attempted.append("exact")
        exact = await self._query_exact(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            regime_key=regime_key,
            limit=k,
        )
        context.exact_count = len(exact)

        for c in exact:
            if c.tune_id not in seen_tune_ids:
                c.match_detail = MatchDetail(tier=RegimeMatchTier.EXACT)
                all_candidates.append(c)
                seen_tune_ids.add(c.tune_id)

        if len(all_candidates) >= min_samples:
            context.tier_used = RegimeMatchTier.EXACT
            logger.debug(
                "regime_match_exact_sufficient",
                count=len(all_candidates),
                regime_key=regime_key,
            )
            return TieredMatchResult(
                candidates=all_candidates[:k],
                sample_context=context,
                query_regime_key=regime_key,
            )

        # Tier 1: Partial match
        context.tiers_attempted.append("partial")
        context.fallback_reason = (
            f"exact_count={context.exact_count} < min_samples={min_samples}"
        )

        # 1A: Same trend, any vol (capped)
        partial_trend = await self._query_partial_trend(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            trend_tag=trend_tag,
            exclude_ids=seen_tune_ids,
            limit=caps.max_partial_trend,  # Apply tier cap
        )
        context.partial_trend_count = len(partial_trend)

        added_trend = 0
        for c in partial_trend:
            if c.tune_id not in seen_tune_ids and added_trend < caps.max_partial_trend:
                c.match_detail = MatchDetail(
                    tier=RegimeMatchTier.PARTIAL_TREND,
                    matched_field="trend_tag",
                    matched_value=trend_tag,
                )
                all_candidates.append(c)
                seen_tune_ids.add(c.tune_id)
                added_trend += 1

        # 1B: Same vol, any trend (capped)
        partial_vol = await self._query_partial_vol(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            vol_tag=vol_tag,
            exclude_ids=seen_tune_ids,
            limit=caps.max_partial_vol,  # Apply tier cap
        )
        context.partial_vol_count = len(partial_vol)

        added_vol = 0
        for c in partial_vol:
            if c.tune_id not in seen_tune_ids and added_vol < caps.max_partial_vol:
                c.match_detail = MatchDetail(
                    tier=RegimeMatchTier.PARTIAL_VOL,
                    matched_field="vol_tag",
                    matched_value=vol_tag,
                )
                all_candidates.append(c)
                seen_tune_ids.add(c.tune_id)
                added_vol += 1

        if len(all_candidates) >= min_samples:
            context.tier_used = RegimeMatchTier.PARTIAL_TREND
            logger.debug(
                "regime_match_partial_sufficient",
                count=len(all_candidates),
                trend_count=context.partial_trend_count,
                vol_count=context.partial_vol_count,
            )
            return TieredMatchResult(
                candidates=all_candidates[:k],
                sample_context=context,
                query_regime_key=regime_key,
            )

        # Tier 2: Distance match (capped)
        if query_snapshot:
            context.tiers_attempted.append("distance")
            context.fallback_reason = (
                f"partial_count={len(all_candidates)} < min_samples={min_samples}"
            )

            distance_candidates = await self._query_distance(
                workspace_id=workspace_id,
                strategy_entity_id=strategy_entity_id,
                query_snapshot=query_snapshot,
                exclude_ids=seen_tune_ids,
                limit=caps.max_distance * 2,  # Fetch extra for ranking
            )
            context.distance_count = len(distance_candidates)

            added_distance = 0
            for i, c in enumerate(distance_candidates):
                if (
                    c.tune_id not in seen_tune_ids
                    and added_distance < caps.max_distance
                ):
                    # match_detail already set by _query_distance with full metadata
                    if c.match_detail:
                        c.match_detail.distance_rank = i + 1
                    all_candidates.append(c)
                    seen_tune_ids.add(c.tune_id)
                    added_distance += 1

            if len(all_candidates) >= min_samples:
                context.tier_used = RegimeMatchTier.DISTANCE
                logger.debug(
                    "regime_match_distance_sufficient",
                    count=len(all_candidates),
                    distance_count=context.distance_count,
                )
                return TieredMatchResult(
                    candidates=all_candidates[:k],
                    sample_context=context,
                    query_regime_key=regime_key,
                )

        # Tier 3: Global fallback
        context.tiers_attempted.append("global")
        context.fallback_reason = (
            f"distance_count={context.distance_count} insufficient, using global"
        )

        global_candidates = await self._query_global_best(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            exclude_ids=seen_tune_ids,
            limit=k,
        )
        context.global_count = len(global_candidates)

        for c in global_candidates:
            if c.tune_id not in seen_tune_ids:
                c.match_detail = MatchDetail(tier=RegimeMatchTier.GLOBAL_BEST)
                all_candidates.append(c)
                seen_tune_ids.add(c.tune_id)

        context.tier_used = RegimeMatchTier.GLOBAL_BEST
        logger.debug(
            "regime_match_global_fallback",
            count=len(all_candidates),
            global_count=context.global_count,
        )

        return TieredMatchResult(
            candidates=all_candidates[:k],
            sample_context=context,
            query_regime_key=regime_key,
        )

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def _query_exact(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        regime_key: Optional[str],
        limit: int,
    ) -> list[RegimeMatchCandidate]:
        """Query for exact regime_key matches."""
        if not regime_key:
            return []

        conditions = [
            "t.workspace_id = $1",
            "t.regime_key = $2",
            "t.status = 'completed'",
            "t.best_oos_score IS NOT NULL",
        ]
        params: list[Any] = [workspace_id, regime_key]
        param_idx = 3

        if strategy_entity_id:
            conditions.append(f"t.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        query = f"""
            SELECT
                t.id as tune_id,
                t.best_oos_run_id as run_id,
                t.strategy_entity_id,
                t.best_oos_params as best_params,
                t.best_oos_score,
                t.regime_key,
                t.trend_tag,
                t.vol_tag,
                t.efficiency_tag
            FROM backtest_tunes t
            WHERE {" AND ".join(conditions)}
            ORDER BY t.best_oos_score DESC NULLS LAST
            LIMIT ${param_idx}
        """
        params.append(limit)

        return await self._execute_query(query, params)

    async def _query_partial_trend(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        trend_tag: Optional[str],
        exclude_ids: set[UUID],
        limit: int,
    ) -> list[RegimeMatchCandidate]:
        """Query for same trend_tag (any vol_tag)."""
        if not trend_tag:
            return []

        conditions = [
            "t.workspace_id = $1",
            "t.trend_tag = $2",
            "t.status = 'completed'",
            "t.best_oos_score IS NOT NULL",
        ]
        params: list[Any] = [workspace_id, trend_tag]
        param_idx = 3

        if strategy_entity_id:
            conditions.append(f"t.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if exclude_ids:
            conditions.append(f"t.id != ALL(${param_idx})")
            params.append(list(exclude_ids))
            param_idx += 1

        query = f"""
            SELECT
                t.id as tune_id,
                t.best_oos_run_id as run_id,
                t.strategy_entity_id,
                t.best_oos_params as best_params,
                t.best_oos_score,
                t.regime_key,
                t.trend_tag,
                t.vol_tag,
                t.efficiency_tag
            FROM backtest_tunes t
            WHERE {" AND ".join(conditions)}
            ORDER BY t.best_oos_score DESC NULLS LAST
            LIMIT ${param_idx}
        """
        params.append(limit)

        return await self._execute_query(query, params)

    async def _query_partial_vol(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        vol_tag: Optional[str],
        exclude_ids: set[UUID],
        limit: int,
    ) -> list[RegimeMatchCandidate]:
        """Query for same vol_tag (any trend_tag)."""
        if not vol_tag:
            return []

        conditions = [
            "t.workspace_id = $1",
            "t.vol_tag = $2",
            "t.status = 'completed'",
            "t.best_oos_score IS NOT NULL",
        ]
        params: list[Any] = [workspace_id, vol_tag]
        param_idx = 3

        if strategy_entity_id:
            conditions.append(f"t.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if exclude_ids:
            conditions.append(f"t.id != ALL(${param_idx})")
            params.append(list(exclude_ids))
            param_idx += 1

        query = f"""
            SELECT
                t.id as tune_id,
                t.best_oos_run_id as run_id,
                t.strategy_entity_id,
                t.best_oos_params as best_params,
                t.best_oos_score,
                t.regime_key,
                t.trend_tag,
                t.vol_tag,
                t.efficiency_tag
            FROM backtest_tunes t
            WHERE {" AND ".join(conditions)}
            ORDER BY t.best_oos_score DESC NULLS LAST
            LIMIT ${param_idx}
        """
        params.append(limit)

        return await self._execute_query(query, params)

    async def _query_distance(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        query_snapshot: RegimeSnapshot,
        exclude_ids: set[UUID],
        limit: int,
    ) -> list[RegimeMatchCandidate]:
        """
        Query candidates and rank by regime feature distance.

        Fetches candidates with regime data, then computes distance in Python.
        """
        query_features = extract_regime_features(query_snapshot)
        if not query_features:
            return []

        # Fetch candidates that have regime data (via their best run)
        conditions = [
            "t.workspace_id = $1",
            "t.status = 'completed'",
            "t.best_oos_score IS NOT NULL",
            "t.best_oos_run_id IS NOT NULL",
        ]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if strategy_entity_id:
            conditions.append(f"t.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if exclude_ids:
            conditions.append(f"t.id != ALL(${param_idx})")
            params.append(list(exclude_ids))
            param_idx += 1

        # Join with tune_runs to get metrics_oos.regime
        query = f"""
            SELECT
                t.id as tune_id,
                t.best_oos_run_id as run_id,
                t.strategy_entity_id,
                t.best_oos_params as best_params,
                t.best_oos_score,
                t.regime_key,
                t.trend_tag,
                t.vol_tag,
                t.efficiency_tag,
                tr.metrics_oos
            FROM backtest_tunes t
            JOIN backtest_tune_runs tr ON tr.run_id = t.best_oos_run_id
            WHERE {" AND ".join(conditions)}
              AND tr.metrics_oos IS NOT NULL
              AND tr.metrics_oos->'regime' IS NOT NULL
            ORDER BY t.best_oos_score DESC NULLS LAST
            LIMIT ${param_idx}
        """
        params.append(limit * 3)  # Fetch extra for distance filtering

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        # Compute distances and rank
        candidates_with_distance: list[tuple[float, RegimeMatchCandidate]] = []

        for row in rows:
            metrics_oos = row.get("metrics_oos")
            if not metrics_oos or not isinstance(metrics_oos, dict):
                continue

            regime_data = metrics_oos.get("regime")
            if not regime_data:
                continue

            try:
                candidate_snapshot = RegimeSnapshot.from_dict(regime_data)
                candidate_features = extract_regime_features(candidate_snapshot)
                distance, top_features = compute_regime_distance_with_features(
                    query_features, candidate_features
                )

                if distance != float("inf"):
                    candidate = RegimeMatchCandidate(
                        tune_id=row["tune_id"],
                        run_id=row["run_id"],
                        strategy_entity_id=row["strategy_entity_id"],
                        best_params=row["best_params"],
                        best_oos_score=row["best_oos_score"],
                        regime_key=row["regime_key"],
                        trend_tag=row["trend_tag"],
                        vol_tag=row["vol_tag"],
                        efficiency_tag=row["efficiency_tag"],
                        match_detail=MatchDetail(
                            tier=RegimeMatchTier.DISTANCE,
                            distance_score=distance,
                            distance_method="euclidean",
                            distance_features_version=DISTANCE_FEATURES_VERSION,
                            distance_top_features=top_features,
                        ),
                    )
                    candidates_with_distance.append((distance, candidate))
            except Exception as e:
                logger.warning(
                    "regime_distance_computation_failed",
                    tune_id=str(row["tune_id"]),
                    error=str(e),
                )
                continue

        # Sort by distance (ascending) and return top limit
        candidates_with_distance.sort(key=lambda x: x[0])
        return [c for _, c in candidates_with_distance[:limit]]

    async def _query_global_best(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        exclude_ids: set[UUID],
        limit: int,
    ) -> list[RegimeMatchCandidate]:
        """Query top OOS scores regardless of regime."""
        conditions = [
            "t.workspace_id = $1",
            "t.status = 'completed'",
            "t.best_oos_score IS NOT NULL",
        ]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if strategy_entity_id:
            conditions.append(f"t.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if exclude_ids:
            conditions.append(f"t.id != ALL(${param_idx})")
            params.append(list(exclude_ids))
            param_idx += 1

        query = f"""
            SELECT
                t.id as tune_id,
                t.best_oos_run_id as run_id,
                t.strategy_entity_id,
                t.best_oos_params as best_params,
                t.best_oos_score,
                t.regime_key,
                t.trend_tag,
                t.vol_tag,
                t.efficiency_tag
            FROM backtest_tunes t
            WHERE {" AND ".join(conditions)}
            ORDER BY t.best_oos_score DESC NULLS LAST
            LIMIT ${param_idx}
        """
        params.append(limit)

        return await self._execute_query(query, params)

    async def _execute_query(
        self,
        query: str,
        params: list[Any],
    ) -> list[RegimeMatchCandidate]:
        """Execute query and convert rows to RegimeMatchCandidate."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        candidates = []
        for row in rows:
            candidates.append(
                RegimeMatchCandidate(
                    tune_id=row["tune_id"],
                    run_id=row["run_id"],
                    strategy_entity_id=row["strategy_entity_id"],
                    best_params=row["best_params"],
                    best_oos_score=row["best_oos_score"],
                    regime_key=row["regime_key"],
                    trend_tag=row["trend_tag"],
                    vol_tag=row["vol_tag"],
                    efficiency_tag=row["efficiency_tag"],
                )
            )
        return candidates
