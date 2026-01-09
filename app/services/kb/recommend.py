"""
KB recommendation orchestrator.

Ties together retrieval, reranking, and aggregation to produce
parameter recommendations based on similar historical trials.
"""

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
    RetrievalResult,
    KBRetriever,
    FilterRejections,
)
from app.services.kb.rerank import (
    RerankResult,
    RerankedCandidate,
    rerank_candidates,
)
from app.services.kb.aggregation import (
    AggregationResult,
    aggregate_params,
    compute_confidence,
)
from app.services.strategies.registry import get_strategy, StrategySpec
from app.repositories.kb_trials import KBTrialRepository

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


@dataclass
class RecommendedRelaxedSettings:
    """Suggested relaxed filter settings that would yield candidates.

    Computed when status='none' to help users understand what
    constraints need to be loosened to get recommendations.
    """

    min_trades: Optional[int] = None
    max_drawdown: Optional[float] = None
    max_overfit_gap: Optional[float] = None
    require_oos: Optional[bool] = None
    estimated_candidates: int = 0  # How many would pass with these settings


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
    """

    def __init__(
        self,
        repository: KBTrialRepository,
        retriever: Optional[KBRetriever] = None,
    ):
        """
        Initialize recommender.

        Args:
            repository: KB trial repository
            retriever: Optional retriever (created if None)
        """
        self.repository = repository
        self._retriever = retriever

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
            with sentry_sdk.start_span(op="regime", description="Compute regime from OHLCV"):
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

        with sentry_sdk.start_span(op="retrieve", description="Retrieve candidates from Qdrant") as span:
            retrieval_result = await self.retriever.retrieve(retrieval_req)
            if span:
                span.set_data("strict_count", retrieval_result.stats.strict_count)
                span.set_data("relaxed_count", retrieval_result.stats.relaxed_count)
                span.set_data("total_returned", retrieval_result.stats.total_returned)
                span.set_data("used_relaxed", retrieval_result.stats.used_relaxed_filters)
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
        with sentry_sdk.start_span(op="rerank", description="Rerank candidates by combined score") as span:
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
        top_m = rerank_result.candidates[:req.rerank_top_m]

        # Sort top M by objective_score and take top K
        top_m_sorted = sorted(
            top_m,
            key=lambda c: c.payload.get("objective_score") or 0,
            reverse=True,
        )
        top_k = top_m_sorted[:req.aggregate_top_k]

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

        confidence = compute_confidence(
            spreads=aggregation_result.spreads,
            count_used=aggregation_result.count_used,
            has_warnings=len(warnings) > 0 or has_repair_warnings,
            used_relaxed=retrieval_result.stats.used_relaxed_filters,
            used_metadata_fallback=retrieval_result.stats.used_metadata_fallback,
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
        )

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
                ("trades", filter_rejections.by_trades, "lower_min_trades_or_run_longer_backtest"),
                ("drawdown", filter_rejections.by_drawdown, "increase_max_drawdown_threshold"),
                ("overfit_gap", filter_rejections.by_overfit_gap, "increase_max_overfit_gap_or_longer_oos"),
                ("oos", filter_rejections.by_oos, "enable_oos_split_in_tuner"),
                ("regime", filter_rejections.by_regime, "backfill_regime_tags_for_strategy"),
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
            actions.extend([
                "ingest_more_trials",
                "check_strategy_name_spelling",
                "verify_objective_type_match",
            ])

        elif status == "degraded":
            # Check specific reasons
            if "used_relaxed_filters" in reasons:
                actions.append("run_more_trials_to_improve_coverage")
            if "used_metadata_fallback" in reasons:
                actions.append("check_embedding_service_health")
            if "params_required_repair" in reasons:
                actions.append("review_param_spec_constraints")

        return actions

    def _compute_recommended_relaxed_settings(
        self,
        filter_rejections: Optional[FilterRejections],
        current_request: RecommendRequest,
    ) -> Optional[RecommendedRelaxedSettings]:
        """
        Compute relaxed settings that would yield candidates.

        Based on filter rejection analysis, suggest looser thresholds
        that would pass more candidates through. Only computed when
        status='none' and filter_rejections are available.
        """
        if not filter_rejections:
            return None

        # If no data exists at all, relaxing won't help
        if filter_rejections.total_before_filters == 0:
            return None

        # Determine which filters to relax based on rejection counts
        settings = RecommendedRelaxedSettings()

        # Progressive relaxation thresholds
        relaxed_min_trades = 1  # Absolute minimum
        relaxed_max_drawdown = 0.50  # 50% max DD
        relaxed_max_overfit_gap = 1.0  # Effectively disabled
        relaxed_require_oos = False  # Don't require OOS

        # Start with current request values (or defaults)
        from app.services.kb.retrieval import DEFAULT_STRICT_FILTERS

        # Compute cumulative candidates at each relaxation level
        cumulative = filter_rejections.total_before_filters

        # If by_oos is significant, suggest disabling OOS requirement
        if filter_rejections.by_oos > 0:
            settings.require_oos = relaxed_require_oos
            cumulative -= filter_rejections.by_oos  # Approximate remaining after relaxing

        # If by_trades is significant, suggest lower min_trades
        if filter_rejections.by_trades > 0:
            settings.min_trades = relaxed_min_trades

        # If by_drawdown is significant, suggest higher max_drawdown
        if filter_rejections.by_drawdown > 0:
            settings.max_drawdown = relaxed_max_drawdown

        # If by_overfit_gap is significant, suggest higher max_overfit_gap
        if filter_rejections.by_overfit_gap > 0:
            settings.max_overfit_gap = relaxed_max_overfit_gap

        # Estimate how many candidates would pass with relaxed settings
        # This is a rough estimate based on the rejection counts
        # (In diagnostic mode, the total_before_filters gives us the baseline)
        estimated = filter_rejections.total_before_filters
        settings.estimated_candidates = estimated

        # Only return if we actually suggested relaxations
        if (settings.min_trades is None and
            settings.max_drawdown is None and
            settings.max_overfit_gap is None and
            settings.require_oos is None):
            return None

        return settings

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
