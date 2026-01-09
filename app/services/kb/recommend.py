"""
KB recommendation orchestrator.

Ties together retrieval, reranking, and aggregation to produce
parameter recommendations based on similar historical trials.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from uuid import UUID

import structlog

from app.services.kb.types import RegimeSnapshot
from app.services.kb.regime import compute_regime_from_ohlcv
from app.services.kb.retrieval import (
    RetrievalRequest,
    RetrievalResult,
    KBRetriever,
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
            RecommendResponse with params, confidence, transparency
        """
        warnings: list[str] = []
        reasons: list[str] = []

        # Step 1: Compute query regime if needed
        query_regime = req.query_regime

        if query_regime is None and req.ohlcv_data:
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

        query_tags = query_regime.regime_tags if query_regime else []

        # Step 2: Retrieve candidates
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
        )

        retrieval_result = await self.retriever.retrieve(retrieval_req)
        warnings.extend(retrieval_result.warnings)

        # Check if we have enough candidates
        if not retrieval_result.candidates:
            return self._build_none_response(
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

        # Step 3: Rerank candidates
        rerank_result = rerank_candidates(
            candidates=retrieval_result.candidates,
            query_tags=query_tags,
            query_regime=query_regime,
        )
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
        strategy_spec = get_strategy(req.strategy_name)

        aggregation_result = aggregate_params(
            candidates=top_k,
            strategy_spec=strategy_spec,
        )
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

        logger.info(
            "Recommendation complete",
            status=status,
            confidence=confidence,
            count_used=aggregation_result.count_used,
            param_count=len(aggregation_result.params),
        )

        return RecommendResponse(
            params=aggregation_result.params,
            status=status,
            confidence=confidence,
            top_trials=top_trials,
            count_used=aggregation_result.count_used,
            warnings=warnings,
            reasons=reasons,
            suggested_actions=[],
            retrieval_strict_count=retrieval_result.stats.strict_count,
            retrieval_relaxed_count=retrieval_result.stats.relaxed_count,
            used_relaxed_filters=retrieval_result.stats.used_relaxed_filters,
            used_metadata_fallback=retrieval_result.stats.used_metadata_fallback,
            query_regime_tags=query_tags,
            collection_name=retrieval_result.stats.collection_name,
            embedding_model=retrieval_result.stats.embedding_model,
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
