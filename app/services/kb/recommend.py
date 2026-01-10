"""
KB recommendation orchestrator.

Ties together retrieval, reranking, and aggregation to produce
parameter recommendations based on similar historical trials.
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Literal, Optional
from uuid import UUID

import sentry_sdk
import structlog

from app.services.kb.types import RegimeSnapshot
from app.services.kb.regime import compute_regime_from_ohlcv
from app.services.kb.retrieval import (
    RetrievalRequest,
    KBRetriever,
    FilterRejections,
)
from app.services.kb.rerank import (
    rerank_candidates,
    RerankedCandidate,
)
from app.services.kb.aggregation import (
    aggregate_params,
    compute_confidence,
)
from app.services.kb.distance import compute_regime_distance_z
from app.services.kb.regime_fsm import RegimeFSM, FSMConfig
from app.services.kb.comparator import (
    ScoredCandidate,
    rank_candidates,
)
from app.services.strategies.registry import get_strategy
from app.repositories.kb_trials import KBTrialRepository
from app.repositories.cluster_stats import ClusterStatsRepository
from app.repositories.duration_stats import DurationStatsRepository

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Default limits
DEFAULT_RETRIEVE_LIMIT = 100  # Max to retrieve
DEFAULT_RERANK_TOP_M = 50  # Top M after reranking
DEFAULT_AGGREGATE_TOP_K = 20  # Top K for aggregation

# Status thresholds
MIN_CANDIDATES_FOR_OK = 10
MIN_CANDIDATES_FOR_DEGRADED = 3


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RecommendTimings:
    """Per-step timing breakdown for recommendation pipeline."""

    total_ms: float = 0.0
    regime_ms: float = 0.0
    embed_ms: float = 0.0
    qdrant_ms: float = 0.0
    rerank_ms: float = 0.0
    aggregate_ms: float = 0.0
    distance_ms: float = 0.0
    duration_ms: float = 0.0


@dataclass
class RegimeStateStability:
    """FSM state info for regime stability (v1.5)."""

    candidate_key: Optional[str] = None
    candidate_bars: int = 0
    M: int = 20
    C_enter: float = 0.75
    C_exit: float = 0.55


@dataclass
class WindowMetadata:
    """Window metadata for rolling computations (v1.5)."""

    regime_age_bars: int = 0
    performance_window: Optional[dict] = None  # {"bars": 500, "timeframe": "5m"}
    distance_window: Optional[dict] = None  # {"bars": 500, "timeframe": "5m"}


@dataclass
class RecommendRequest:
    """Request for parameter recommendations."""

    workspace_id: UUID
    strategy_name: str
    objective_type: str

    # Query context
    ohlcv_data: Optional[list[dict]] = None  # Raw OHLCV for regime computation
    query_regime: Optional[RegimeSnapshot] = None  # Pre-computed regime
    timeframe: Optional[str] = None

    # v1.5 context (for duration stats, cluster stats lookups)
    symbol: Optional[str] = None  # Trading symbol (e.g., "BTC/USDT")
    strategy_entity_id: Optional[UUID] = None  # Strategy entity ID for cluster stats

    # Filter options
    require_oos: Optional[bool] = None
    max_overfit_gap: Optional[float] = None
    min_trades: Optional[int] = None
    max_drawdown: Optional[float] = None

    # Retrieval limits
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT
    rerank_top_m: int = DEFAULT_RERANK_TOP_M
    aggregate_top_k: int = DEFAULT_AGGREGATE_TOP_K

    # Diagnostic mode: compute filter rejection counts (expensive)
    diagnostic: bool = False

    # v1.5 FSM configuration (optional, uses defaults if None)
    fsm_config: Optional[FSMConfig] = None


@dataclass
class RelaxationSuggestion:
    """Single-axis relaxation suggestion with risk note.

    Presents one filter adjustment at a time so users can
    understand the impact of each change individually.
    """

    filter_name: str  # e.g., "min_trades", "max_drawdown"
    current_value: Optional[float | int | bool] = None
    suggested_value: Optional[float | int | bool] = None
    estimated_candidates: int = 0  # How many would pass with this single change
    risk_note: str = ""  # Warning about the trade-off


@dataclass
class RecommendedRelaxedSettings:
    """Suggested relaxed filter settings that would yield candidates.

    Computed when status='none' to help users understand what
    constraints need to be loosened to get recommendations.

    Now returns single-axis suggestions so users can evaluate
    each trade-off independently.
    """

    suggestions: list[RelaxationSuggestion] = field(default_factory=list)


@dataclass
class TrialSummary:
    """Summary of a trial used in recommendation."""

    point_id: str
    strategy_name: str
    objective_score: float
    similarity_score: float
    jaccard_score: float
    rerank_score: float
    params: dict


@dataclass
class RecommendResponse:
    """Response from parameter recommendation."""

    # Core result
    params: dict  # Recommended parameters
    status: Literal["ok", "degraded", "none"]  # Recommendation quality

    # Confidence
    confidence: Optional[float] = None  # 0-1 confidence score

    # Transparency
    top_trials: list[TrialSummary] = field(default_factory=list)  # Trials used
    count_used: int = 0  # Total trials in aggregation

    # Quality indicators
    warnings: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)  # Why status is what it is
    suggested_actions: list[str] = field(default_factory=list)  # What user can do

    # Stats
    retrieval_strict_count: int = 0
    retrieval_relaxed_count: int = 0
    used_relaxed_filters: bool = False
    used_metadata_fallback: bool = False

    # Context
    query_regime_tags: list[str] = field(default_factory=list)
    collection_name: str = ""
    embedding_model: str = ""

    # Timings (for observability)
    timings: Optional[RecommendTimings] = None

    # Filter rejection counts (only in diagnostic mode)
    filter_rejections: Optional[FilterRejections] = None

    # Recommended relaxed settings (only when status='none')
    recommended_relaxed_settings: Optional[RecommendedRelaxedSettings] = None

    # ==========================================================================
    # v1.5 Live Intelligence Fields
    # ==========================================================================

    # Confidence decomposition (v1.5)
    regime_fit_confidence: Optional[float] = (
        None  # How well current market matches historical (0-1)
    )
    regime_distance_z: Optional[float] = None  # Z-score distance from neighborhood
    distance_baseline: Optional[str] = (
        None  # "composite" | "marginal" | "neighbors_only"
    )
    distance_n: Optional[int] = None  # Number of neighbors used for distance

    # Duration fields (v1.5)
    regime_age_bars: Optional[int] = None  # Bars since stable regime confirmed
    regime_half_life_bars: Optional[int] = None  # Median historical duration
    expected_remaining_bars: Optional[int] = None  # max(0, median - age)
    duration_iqr_bars: Optional[list[int]] = None  # [p25, p75]
    remaining_iqr_bars: Optional[list[int]] = None  # [max(0, p25-age), max(0, p75-age)]
    duration_baseline: Optional[str] = (
        None  # "composite_symbol" | "marginal" | "global_timeframe"
    )
    duration_n: Optional[int] = None  # Number of segments used

    # FSM state (v1.5)
    stable_regime_key: Optional[str] = None  # Confirmed stable regime
    raw_regime_key: Optional[str] = None  # Raw current classification
    regime_state_stability: Optional[RegimeStateStability] = None  # FSM state details

    # Window metadata (v1.5)
    windows: Optional[WindowMetadata] = None

    # Missing field reasons (v1.5)
    missing: list[str] = field(
        default_factory=list
    )  # ["no_duration_stats", "no_cluster_stats", etc.]


# =============================================================================
# Orchestrator
# =============================================================================


class KBRecommender:
    """
    Orchestrates KB-based parameter recommendations.

    Flow:
    1. Parse/load dataset and compute query regime
    2. Retrieve candidates (strict → relaxed → metadata fallback)
    3. Rerank by combined score
    4. Select top K by objective_score
    5. Aggregate and repair parameters
    6. Compute confidence and status
    7. (v1.5) Compute distance_z, duration stats, FSM state
    """

    def __init__(
        self,
        repository: KBTrialRepository,
        retriever: Optional[KBRetriever] = None,
        cluster_stats_repo: Optional[ClusterStatsRepository] = None,
        duration_stats_repo: Optional[DurationStatsRepository] = None,
    ):
        """
        Initialize recommender.

        Args:
            repository: KB trial repository
            retriever: Optional retriever (created if None)
            cluster_stats_repo: Optional cluster stats repository (v1.5)
            duration_stats_repo: Optional duration stats repository (v1.5)
        """
        self.repository = repository
        self._retriever = retriever
        self._cluster_stats_repo = cluster_stats_repo
        self._duration_stats_repo = duration_stats_repo

    @property
    def retriever(self) -> KBRetriever:
        """Lazy-load retriever."""
        if self._retriever is None:
            self._retriever = KBRetriever(repository=self.repository)
        return self._retriever

    async def recommend(
        self,
        req: RecommendRequest,
    ) -> RecommendResponse:
        """
        Generate parameter recommendations.

        Args:
            req: Recommendation request

        Returns:
            RecommendResponse with params, confidence, transparency, timings
        """
        start_total = time.perf_counter()
        timings = RecommendTimings()
        warnings: list[str] = []
        reasons: list[str] = []

        # Step 1: Compute query regime if needed
        start_regime = time.perf_counter()
        query_regime = req.query_regime

        if query_regime is None and req.ohlcv_data:
            with sentry_sdk.start_span(
                op="regime", description="Compute regime from OHLCV"
            ):
                try:
                    query_regime = compute_regime_from_ohlcv(
                        ohlcv=req.ohlcv_data,
                        timeframe=req.timeframe,
                        source="query",
                    )
                    logger.info(
                        "Computed query regime",
                        n_bars=query_regime.n_bars,
                        tags=query_regime.regime_tags,
                    )
                except Exception as e:
                    logger.warning("Failed to compute query regime", error=str(e))
                    warnings.append("query_regime_computation_failed")
        timings.regime_ms = (time.perf_counter() - start_regime) * 1000

        query_tags = query_regime.regime_tags if query_regime else []

        # Step 2: Retrieve candidates (includes embed + qdrant time)
        start_retrieval = time.perf_counter()
        retrieval_req = RetrievalRequest(
            workspace_id=req.workspace_id,
            strategy_name=req.strategy_name,
            objective_type=req.objective_type,
            query_regime=query_regime,
            require_oos=req.require_oos,
            max_overfit_gap=req.max_overfit_gap,
            min_trades=req.min_trades,
            max_drawdown=req.max_drawdown,
            regime_tags=query_tags if query_tags else None,
            limit=req.retrieve_limit,
            diagnostic=req.diagnostic,
        )

        with sentry_sdk.start_span(
            op="retrieve", description="Retrieve candidates from Qdrant"
        ) as span:
            retrieval_result = await self.retriever.retrieve(retrieval_req)
            if span:
                span.set_data("strict_count", retrieval_result.stats.strict_count)
                span.set_data("relaxed_count", retrieval_result.stats.relaxed_count)
                span.set_data("total_returned", retrieval_result.stats.total_returned)
                span.set_data(
                    "used_relaxed", retrieval_result.stats.used_relaxed_filters
                )
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000
        # Note: embed_ms and qdrant_ms are sub-components of retrieval
        # If retriever provides breakdown, use it; otherwise estimate
        timings.embed_ms = getattr(retrieval_result.stats, "embed_ms", 0.0)
        timings.qdrant_ms = getattr(retrieval_result.stats, "qdrant_ms", retrieval_ms)
        warnings.extend(retrieval_result.warnings)

        # Check if we have enough candidates
        if not retrieval_result.candidates:
            timings.total_ms = (time.perf_counter() - start_total) * 1000
            response = self._build_none_response(
                req=req,
                warnings=warnings,
                reasons=["no_candidates_found"],
                suggested_actions=[
                    "ingest_more_trials",
                    "relax_filter_criteria",
                    "check_strategy_name_spelling",
                ],
                collection_name=retrieval_result.stats.collection_name,
                embedding_model=retrieval_result.stats.embedding_model,
            )
            response.timings = timings
            return response

        # Step 3: Rerank candidates
        start_rerank = time.perf_counter()
        with sentry_sdk.start_span(
            op="rerank", description="Rerank candidates by combined score"
        ) as span:
            rerank_result = rerank_candidates(
                candidates=retrieval_result.candidates,
                query_tags=query_tags,
                query_regime=query_regime,
            )
            if span:
                span.set_data("candidates_in", len(retrieval_result.candidates))
                span.set_data("candidates_out", len(rerank_result.candidates))
        timings.rerank_ms = (time.perf_counter() - start_rerank) * 1000
        warnings.extend(rerank_result.warnings)

        # Step 4: Select top M reranked, then top K by objective_score
        # Uses epsilon-aware comparator for tie-breaking
        top_m = rerank_result.candidates[: req.rerank_top_m]
        top_k = self._rank_and_select_top_k(top_m, req.aggregate_top_k)

        # Step 5: Get strategy spec and aggregate
        start_aggregate = time.perf_counter()
        strategy_spec = get_strategy(req.strategy_name)

        with sentry_sdk.start_span(op="aggregate", description="Aggregate parameters"):
            aggregation_result = aggregate_params(
                candidates=top_k,
                strategy_spec=strategy_spec,
            )
        timings.aggregate_ms = (time.perf_counter() - start_aggregate) * 1000
        warnings.extend(aggregation_result.warnings)

        # Check 6: Final param validation (assert types are correct)
        if strategy_spec and aggregation_result.params:
            validation_result = strategy_spec.validate_params(aggregation_result.params)
            if not validation_result.is_valid:
                logger.warning(
                    "Final param validation failed after repair",
                    violations=validation_result.constraint_violations,
                )
                # Don't fail, but add warning
                warnings.append("final_param_validation_failed")

        # Check for repair warnings (clamping, type coercion, etc.)
        has_repair_warnings = any(
            w.startswith("param_") or w.startswith("constraint_")
            for w in aggregation_result.warnings
        )

        # Step 6: Determine status and compute confidence
        status = self._compute_status(
            strict_count=retrieval_result.stats.strict_count,
            total_count=retrieval_result.stats.total_returned,
            used_relaxed=retrieval_result.stats.used_relaxed_filters,
            used_metadata=retrieval_result.stats.used_metadata_fallback,
            count_used=aggregation_result.count_used,
            has_repair_warnings=has_repair_warnings,
        )

        if status == "degraded":
            if retrieval_result.stats.used_relaxed_filters:
                reasons.append("used_relaxed_filters")
            if retrieval_result.stats.used_metadata_fallback:
                reasons.append("used_metadata_fallback")
            if retrieval_result.stats.strict_count < MIN_CANDIDATES_FOR_OK:
                reasons.append("insufficient_strict_candidates")
            if has_repair_warnings:
                reasons.append("params_required_repair")

        # Compute median n_trades_oos for confidence guard
        # Low trades = statistically noisy metrics
        oos_trades_list = [
            c.payload.get("n_trades_oos")
            for c in top_k
            if c.payload.get("n_trades_oos") is not None
        ]
        median_oos_trades = (
            int(statistics.median(oos_trades_list)) if oos_trades_list else None
        )

        # Add low_trade_count reason if median < trust threshold
        if median_oos_trades is not None and median_oos_trades < 10:
            reasons.append("low_trade_count")
            warnings.append("low_oos_trades_statistical_noise")

        confidence = compute_confidence(
            spreads=aggregation_result.spreads,
            count_used=aggregation_result.count_used,
            has_warnings=len(warnings) > 0 or has_repair_warnings,
            used_relaxed=retrieval_result.stats.used_relaxed_filters,
            used_metadata_fallback=retrieval_result.stats.used_metadata_fallback,
            median_oos_trades=median_oos_trades,
        )

        # Build trial summaries for transparency
        top_trials = [
            TrialSummary(
                point_id=c.point_id,
                strategy_name=c.payload.get("strategy_name", ""),
                objective_score=c.payload.get("objective_score") or 0,
                similarity_score=c.similarity_score,
                jaccard_score=c.jaccard_score,
                rerank_score=c.rerank_score,
                params=c.payload.get("params", {}),
            )
            for c in top_k[:5]  # Only include top 5 for transparency
        ]

        # =================================================================
        # Step 7: v1.5 Computations (distance_z, duration, FSM state)
        # =================================================================
        v15_result = await self._compute_v15_fields(
            req=req,
            query_regime=query_regime,
            top_k=top_k,
            timings=timings,
        )
        missing = v15_result.get("missing", [])

        # Finalize timings
        timings.total_ms = (time.perf_counter() - start_total) * 1000

        logger.info(
            "Recommendation complete",
            status=status,
            confidence=confidence,
            count_used=aggregation_result.count_used,
            param_count=len(aggregation_result.params),
            total_ms=round(timings.total_ms, 1),
            regime_ms=round(timings.regime_ms, 1),
            qdrant_ms=round(timings.qdrant_ms, 1),
            rerank_ms=round(timings.rerank_ms, 1),
            aggregate_ms=round(timings.aggregate_ms, 1),
        )

        # Build conditional suggested_actions based on filter_rejections
        suggested_actions = self._build_suggested_actions(
            status=status,
            filter_rejections=retrieval_result.stats.filter_rejections,
            reasons=reasons,
            strict_count=retrieval_result.stats.strict_count,
        )

        # Compute recommended relaxed settings when status is 'none'
        recommended_relaxed = None
        if status == "none":
            recommended_relaxed = self._compute_recommended_relaxed_settings(
                filter_rejections=retrieval_result.stats.filter_rejections,
                current_request=req,
            )

        return RecommendResponse(
            params=aggregation_result.params,
            status=status,
            confidence=confidence,
            top_trials=top_trials,
            count_used=aggregation_result.count_used,
            warnings=warnings,
            reasons=reasons,
            suggested_actions=suggested_actions,
            retrieval_strict_count=retrieval_result.stats.strict_count,
            retrieval_relaxed_count=retrieval_result.stats.relaxed_count,
            used_relaxed_filters=retrieval_result.stats.used_relaxed_filters,
            used_metadata_fallback=retrieval_result.stats.used_metadata_fallback,
            query_regime_tags=query_tags,
            collection_name=retrieval_result.stats.collection_name,
            embedding_model=retrieval_result.stats.embedding_model,
            timings=timings,
            filter_rejections=retrieval_result.stats.filter_rejections,
            recommended_relaxed_settings=recommended_relaxed,
            # v1.5 fields
            regime_fit_confidence=v15_result.get("regime_fit_confidence"),
            regime_distance_z=v15_result.get("regime_distance_z"),
            distance_baseline=v15_result.get("distance_baseline"),
            distance_n=v15_result.get("distance_n"),
            regime_age_bars=v15_result.get("regime_age_bars"),
            regime_half_life_bars=v15_result.get("regime_half_life_bars"),
            expected_remaining_bars=v15_result.get("expected_remaining_bars"),
            duration_iqr_bars=v15_result.get("duration_iqr_bars"),
            remaining_iqr_bars=v15_result.get("remaining_iqr_bars"),
            duration_baseline=v15_result.get("duration_baseline"),
            duration_n=v15_result.get("duration_n"),
            stable_regime_key=v15_result.get("stable_regime_key"),
            raw_regime_key=v15_result.get("raw_regime_key"),
            regime_state_stability=v15_result.get("regime_state_stability"),
            windows=v15_result.get("windows"),
            missing=missing,
        )

    async def _compute_v15_fields(
        self,
        req: RecommendRequest,
        query_regime: Optional[RegimeSnapshot],
        top_k: list,
        timings: RecommendTimings,
    ) -> dict:
        """
        Compute v1.5 fields: distance_z, duration stats, FSM state.

        All v1.5 fields are optional - if data is unavailable, fields are
        omitted and reasons added to the 'missing' list.

        Args:
            req: Original recommend request
            query_regime: Computed regime snapshot
            top_k: Top K candidates from retrieval
            timings: Timing object to update

        Returns:
            Dict with v1.5 fields (all optional)
        """
        result: dict = {"missing": []}

        # Extract regime features from query regime (for distance computation)
        current_features: dict[str, float] = {}
        raw_regime_key: Optional[str] = None
        regime_fit_confidence: Optional[float] = None

        if query_regime:
            # Get features from regime snapshot
            if (
                hasattr(query_regime, "regime_features")
                and query_regime.regime_features
            ):
                current_features = query_regime.regime_features
            # Get raw regime key from tags
            raw_regime_key = self._build_regime_key_from_tags(query_regime.regime_tags)
            # Regime fit confidence from the regime computation
            if hasattr(query_regime, "confidence"):
                regime_fit_confidence = query_regime.confidence

        result["raw_regime_key"] = raw_regime_key
        result["regime_fit_confidence"] = regime_fit_confidence

        # Initialize FSM and compute stable regime key
        fsm_config = req.fsm_config or FSMConfig()
        fsm = RegimeFSM(config=fsm_config)

        # For live intelligence, FSM should be updated with historical bars
        # Here we just expose the current state based on raw_regime_key
        if raw_regime_key and regime_fit_confidence is not None:
            fsm.update(raw_regime_key, regime_fit_confidence)

        fsm_state = fsm.get_state()
        result["stable_regime_key"] = fsm_state.stable_regime_key
        result["regime_state_stability"] = RegimeStateStability(
            candidate_key=fsm_state.candidate_regime_key,
            candidate_bars=fsm_state.candidate_count,
            M=fsm_config.M,
            C_enter=fsm_config.C_enter,
            C_exit=fsm_config.C_exit,
        )

        # Regime age from FSM
        regime_age_bars = fsm_state.regime_age_bars
        result["regime_age_bars"] = regime_age_bars

        # =================================================================
        # Distance Z-Score Computation
        # =================================================================
        start_distance = time.perf_counter()

        # Extract neighbor features from top_k candidates
        neighbor_features: list[dict[str, float]] = []
        for candidate in top_k:
            payload = candidate.payload
            if payload and "regime_features" in payload:
                features = payload["regime_features"]
                if isinstance(features, dict):
                    neighbor_features.append(features)

        # Try to get cluster stats for variance scaling
        cluster_var: Optional[dict[str, float]] = None
        cluster_sigma_prior: Optional[float] = None
        _distance_baseline = "neighbors_only"

        if (
            self._cluster_stats_repo
            and req.strategy_entity_id
            and req.timeframe
            and fsm_state.stable_regime_key
        ):
            try:
                cluster_stats = await self._cluster_stats_repo.get_stats_with_backoff(
                    strategy_entity_id=req.strategy_entity_id,
                    timeframe=req.timeframe,
                    regime_key=fsm_state.stable_regime_key,
                )
                if cluster_stats:
                    cluster_var = cluster_stats.feature_var
                    _distance_baseline = cluster_stats.baseline  # noqa: F841
            except Exception as e:
                logger.warning("Failed to get cluster stats", error=str(e))
                result["missing"].append("cluster_stats_error")

        # Compute distance z-score
        if current_features and neighbor_features:
            distance_result = compute_regime_distance_z(
                current_features=current_features,
                neighbor_features=neighbor_features,
                cluster_var=cluster_var,
                cluster_sigma_prior=cluster_sigma_prior,
            )
            result["regime_distance_z"] = distance_result.z_score
            result["distance_baseline"] = distance_result.baseline
            result["distance_n"] = distance_result.n_neighbors
            if distance_result.missing:
                result["missing"].extend(distance_result.missing)
        else:
            if not current_features:
                result["missing"].append("no_query_regime_features")
            if not neighbor_features:
                result["missing"].append("no_neighbor_features")

        timings.distance_ms = (time.perf_counter() - start_distance) * 1000

        # =================================================================
        # Duration Stats Computation
        # =================================================================
        start_duration = time.perf_counter()

        if (
            self._duration_stats_repo
            and req.symbol
            and req.timeframe
            and fsm_state.stable_regime_key
        ):
            try:
                duration_stats = await self._duration_stats_repo.get_stats_with_backoff(
                    symbol=req.symbol,
                    timeframe=req.timeframe,
                    regime_key=fsm_state.stable_regime_key,
                )
                if duration_stats:
                    result["regime_half_life_bars"] = (
                        duration_stats.median_duration_bars
                    )
                    result["duration_iqr_bars"] = duration_stats.duration_iqr_bars
                    result["duration_baseline"] = duration_stats.baseline
                    result["duration_n"] = duration_stats.n_segments

                    # Compute expected remaining based on regime age
                    if regime_age_bars is not None:
                        remaining = duration_stats.compute_expected_remaining(
                            regime_age_bars
                        )
                        result["expected_remaining_bars"] = (
                            remaining.expected_remaining_bars
                        )
                        result["remaining_iqr_bars"] = remaining.remaining_iqr_bars
                else:
                    result["missing"].append("no_duration_stats")
            except Exception as e:
                logger.warning("Failed to get duration stats", error=str(e))
                result["missing"].append("duration_stats_error")
        else:
            if not self._duration_stats_repo:
                result["missing"].append("duration_stats_repo_unavailable")
            elif not req.symbol:
                result["missing"].append("no_symbol_provided")
            elif not req.timeframe:
                result["missing"].append("no_timeframe_provided")
            elif not fsm_state.stable_regime_key:
                result["missing"].append("no_stable_regime_key")

        timings.duration_ms = (time.perf_counter() - start_duration) * 1000

        # =================================================================
        # Window Metadata
        # =================================================================
        result["windows"] = WindowMetadata(
            regime_age_bars=regime_age_bars or 0,
            performance_window=None,  # To be populated by forward run system
            distance_window=(
                {
                    "bars": len(neighbor_features),
                    "timeframe": req.timeframe,
                }
                if neighbor_features
                else None
            ),
        )

        return result

    def _build_regime_key_from_tags(self, tags: list[str]) -> Optional[str]:
        """
        Build canonical regime key from regime tags.

        Attempts to extract trend and vol dimensions from tags.

        Args:
            tags: List of regime tags

        Returns:
            Canonical regime key or None if dimensions not found
        """
        if not tags:
            return None

        from app.services.kb.regime_key import (
            RegimeDims,
            canonicalize_regime_key,
            VALID_TREND_VALUES,
            VALID_VOL_VALUES,
        )

        trend = None
        vol = None

        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in VALID_TREND_VALUES:
                trend = tag_lower
            elif tag_lower in VALID_VOL_VALUES:
                vol = tag_lower

        if trend and vol:
            try:
                dims = RegimeDims(trend=trend, vol=vol)
                return canonicalize_regime_key(dims)
            except ValueError:
                return None

        return None

    def _rank_and_select_top_k(
        self,
        candidates: list[RerankedCandidate],
        top_k: int,
    ) -> list[RerankedCandidate]:
        """
        Rank candidates using epsilon-aware comparator and select top K.

        Uses tie-break rules when objective scores are within epsilon:
        1. Primary score (objective_score)
        2. promoted > candidate (human curation signal)
        3. Current schema > other > null
        4. Recent kb_promoted_at
        5. Recent created_at

        Args:
            candidates: Reranked candidates to sort
            top_k: Number of top candidates to return

        Returns:
            Top K candidates ranked by objective score with tie-breaks
        """
        from datetime import datetime, timezone

        if not candidates:
            return []

        # Build mapping from point_id to candidate for reconstruction
        candidate_map = {c.point_id: c for c in candidates}

        # Convert to ScoredCandidates
        scored: list[ScoredCandidate] = []
        for c in candidates:
            payload = c.payload
            score = payload.get("objective_score") or 0.0
            kb_status = payload.get("kb_status", "candidate")
            regime_schema_version = payload.get("regime_schema_version")

            # Parse timestamps
            kb_promoted_at = None
            if payload.get("kb_promoted_at"):
                try:
                    promoted_str = payload["kb_promoted_at"]
                    if isinstance(promoted_str, datetime):
                        kb_promoted_at = promoted_str
                    elif isinstance(promoted_str, str):
                        kb_promoted_at = datetime.fromisoformat(
                            promoted_str.replace("Z", "+00:00")
                        )
                except (ValueError, TypeError):
                    pass

            created_at = datetime.now(timezone.utc)  # Default
            if payload.get("created_at"):
                try:
                    created_str = payload["created_at"]
                    if isinstance(created_str, datetime):
                        created_at = created_str
                    elif isinstance(created_str, str):
                        created_at = datetime.fromisoformat(
                            created_str.replace("Z", "+00:00")
                        )
                except (ValueError, TypeError):
                    pass

            scored.append(
                ScoredCandidate(
                    source_id=c.point_id,
                    score=score,
                    kb_status=kb_status,
                    regime_schema_version=regime_schema_version,
                    kb_promoted_at=kb_promoted_at,
                    created_at=created_at,
                )
            )

        # Rank using epsilon-aware comparator
        ranked = rank_candidates(scored)

        # Reconstruct RerankedCandidates in ranked order
        result = []
        for sc in ranked[:top_k]:
            if sc.source_id in candidate_map:
                result.append(candidate_map[sc.source_id])

        return result

    def _compute_status(
        self,
        strict_count: int,
        total_count: int,
        used_relaxed: bool,
        used_metadata: bool,
        count_used: int,
        has_repair_warnings: bool = False,
    ) -> Literal["ok", "degraded", "none"]:
        """Compute recommendation status."""
        if count_used == 0:
            return "none"

        # Any fallback or repair triggers degraded
        if used_metadata:
            return "degraded"

        if has_repair_warnings:
            return "degraded"

        if used_relaxed and strict_count < MIN_CANDIDATES_FOR_DEGRADED:
            return "degraded"

        if strict_count >= MIN_CANDIDATES_FOR_OK:
            return "ok"

        if total_count >= MIN_CANDIDATES_FOR_DEGRADED:
            return "degraded"

        return "none"

    def _build_suggested_actions(
        self,
        status: str,
        filter_rejections: Optional[FilterRejections],
        reasons: list[str],
        strict_count: int,
    ) -> list[str]:
        """
        Build conditional suggested_actions based on filter rejection analysis.

        Actions are prioritized by the largest rejection source.
        """
        actions: list[str] = []

        if status == "ok":
            return []  # No actions needed

        # Check filter rejections for specific guidance
        if filter_rejections:
            # Sort rejection types by count (highest first)
            rejection_types = [
                (
                    "trades",
                    filter_rejections.by_trades,
                    "lower_min_trades_or_run_longer_backtest",
                ),
                (
                    "drawdown",
                    filter_rejections.by_drawdown,
                    "increase_max_drawdown_threshold",
                ),
                (
                    "overfit_gap",
                    filter_rejections.by_overfit_gap,
                    "increase_max_overfit_gap_or_longer_oos",
                ),
                ("oos", filter_rejections.by_oos, "enable_oos_split_in_tuner"),
                (
                    "regime",
                    filter_rejections.by_regime,
                    "backfill_regime_tags_for_strategy",
                ),
            ]

            # Add action for largest rejection source
            rejection_types.sort(key=lambda x: x[1], reverse=True)
            for name, count, action in rejection_types:
                if count > 0:
                    actions.append(action)
                    break  # Only add top action

            # If zero total_before_filters, suggest ingesting data
            if filter_rejections.total_before_filters == 0:
                actions.insert(0, "ingest_trials_for_strategy")

        elif status == "none":
            # Generic actions when no filter rejection data
            actions.extend(
                [
                    "ingest_more_trials",
                    "check_strategy_name_spelling",
                    "verify_objective_type_match",
                ]
            )

        elif status == "degraded":
            # Check specific reasons
            if "used_relaxed_filters" in reasons:
                actions.append("run_more_trials_to_improve_coverage")
            if "used_metadata_fallback" in reasons:
                actions.append("check_embedding_service_health")
            if "params_required_repair" in reasons:
                actions.append("review_param_spec_constraints")

        # Low trade count action applies to both ok and degraded
        if "low_trade_count" in reasons:
            actions.append("run_longer_backtests_for_more_trades")

        return actions

    def _compute_recommended_relaxed_settings(
        self,
        filter_rejections: Optional[FilterRejections],
        current_request: RecommendRequest,
    ) -> Optional[RecommendedRelaxedSettings]:
        """
        Compute single-axis relaxation suggestions with risk notes.

        Each suggestion relaxes ONE filter at a time so users can
        understand the impact of each change independently.

        Only computed when status='none' and filter_rejections are available.
        """
        if not filter_rejections:
            return None

        # If no data exists at all, relaxing won't help
        if filter_rejections.total_before_filters == 0:
            return None

        from app.services.kb.retrieval import DEFAULT_STRICT_FILTERS

        suggestions: list[RelaxationSuggestion] = []

        # Current values (from request or defaults)
        current_min_trades = (
            current_request.min_trades or DEFAULT_STRICT_FILTERS["min_trades"]
        )
        current_max_dd = (
            current_request.max_drawdown or DEFAULT_STRICT_FILTERS["max_drawdown"]
        )
        current_max_overfit = (
            current_request.max_overfit_gap or DEFAULT_STRICT_FILTERS["max_overfit_gap"]
        )
        current_require_oos = (
            current_request.require_oos
            if current_request.require_oos is not None
            else DEFAULT_STRICT_FILTERS["require_oos"]
        )

        # Baseline: total candidates before quality filters
        baseline = filter_rejections.total_before_filters

        # Single-axis suggestion 1: Lower min_trades
        if filter_rejections.by_trades > 0:
            # Estimate: baseline minus other rejections
            estimated = baseline - filter_rejections.by_oos
            suggestions.append(
                RelaxationSuggestion(
                    filter_name="min_trades",
                    current_value=current_min_trades,
                    suggested_value=1,
                    estimated_candidates=estimated,
                    risk_note="lowering min_trades increases statistical noise",
                )
            )

        # Single-axis suggestion 2: Increase max_drawdown
        if filter_rejections.by_drawdown > 0:
            estimated = (
                baseline - filter_rejections.by_oos - filter_rejections.by_trades
            )
            suggestions.append(
                RelaxationSuggestion(
                    filter_name="max_drawdown",
                    current_value=current_max_dd,
                    suggested_value=0.50,
                    estimated_candidates=estimated,
                    risk_note="increasing max_drawdown increases risk tolerance",
                )
            )

        # Single-axis suggestion 3: Increase max_overfit_gap
        if filter_rejections.by_overfit_gap > 0:
            estimated = (
                baseline
                - filter_rejections.by_oos
                - filter_rejections.by_trades
                - filter_rejections.by_drawdown
            )
            suggestions.append(
                RelaxationSuggestion(
                    filter_name="max_overfit_gap",
                    current_value=current_max_overfit,
                    suggested_value=1.0,
                    estimated_candidates=estimated,
                    risk_note="increasing max_overfit_gap increases overfit risk",
                )
            )

        # Single-axis suggestion 4: Disable OOS requirement
        if filter_rejections.by_oos > 0:
            # Disabling OOS recovers those rejected by OOS
            estimated = baseline
            suggestions.append(
                RelaxationSuggestion(
                    filter_name="require_oos",
                    current_value=current_require_oos,
                    suggested_value=False,
                    estimated_candidates=estimated,
                    risk_note="disabling OOS validation increases overfit risk",
                )
            )

        if not suggestions:
            return None

        # Sort by estimated_candidates descending (most impactful first)
        suggestions.sort(key=lambda s: s.estimated_candidates, reverse=True)

        return RecommendedRelaxedSettings(suggestions=suggestions)

    def _build_none_response(
        self,
        req: RecommendRequest,
        warnings: list[str],
        reasons: list[str],
        suggested_actions: list[str],
        collection_name: str,
        embedding_model: str,
    ) -> RecommendResponse:
        """Build response when no recommendations possible."""
        return RecommendResponse(
            params={},
            status="none",
            confidence=None,
            top_trials=[],
            count_used=0,
            warnings=warnings,
            reasons=reasons,
            suggested_actions=suggested_actions,
            retrieval_strict_count=0,
            retrieval_relaxed_count=0,
            used_relaxed_filters=False,
            used_metadata_fallback=False,
            query_regime_tags=[],
            collection_name=collection_name,
            embedding_model=embedding_model,
        )


# =============================================================================
# Module-level helpers
# =============================================================================


async def recommend_params(
    req: RecommendRequest,
    repository: KBTrialRepository,
) -> RecommendResponse:
    """
    Convenience function for parameter recommendation.

    Args:
        req: Recommendation request
        repository: KB trial repository

    Returns:
        RecommendResponse
    """
    recommender = KBRecommender(repository=repository)
    return await recommender.recommend(req)
