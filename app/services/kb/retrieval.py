"""
KB retrieval module for finding similar trials.

Handles:
- Building Qdrant filters from request
- Strict → relaxed fallback
- Embed failure → metadata-only fallback
- Collection selection based on embedding model
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import structlog

from app.services.kb.types import RegimeSnapshot
from app.services.kb.embed import KBEmbeddingAdapter, EmbeddingError, get_kb_embedder
from app.repositories.kb_trials import KBTrialRepository
from app.services.strategies.registry import get_strategy

logger = structlog.get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Minimum candidates to proceed (triggers relaxation if below)
MIN_CANDIDATES_THRESHOLD = 10

# Default filter values (strict)
DEFAULT_STRICT_FILTERS = {
    "require_oos": True,
    "max_overfit_gap": 0.3,
    "min_trades": 5,
    "max_drawdown": 0.25,  # 25%
}

# Relaxed filters (only loosen overfit_gap, keep other quality gates)
DEFAULT_RELAXED_FILTERS = {
    "require_oos": True,  # Keep OOS requirement
    "max_overfit_gap": None,  # Only this is relaxed
    "min_trades": 5,  # Keep trade floor
    "max_drawdown": 0.25,  # Keep DD cap
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RetrievalRequest:
    """Request for retrieving similar trials."""

    workspace_id: UUID
    strategy_name: str
    objective_type: str

    # Query regime (from live data or user input)
    query_regime: Optional[RegimeSnapshot] = None

    # Optional pre-computed query embedding
    query_vector: Optional[list[float]] = None

    # Filter overrides
    require_oos: Optional[bool] = None
    max_overfit_gap: Optional[float] = None
    min_trades: Optional[int] = None
    max_drawdown: Optional[float] = None
    regime_tags: Optional[list[str]] = None

    # Retrieval limits
    limit: int = 100
    min_candidates: int = MIN_CANDIDATES_THRESHOLD

    # Model selection (defaults to workspace config)
    embedding_model: Optional[str] = None

    # Diagnostic mode: runs extra queries to count filter rejections
    diagnostic: bool = False


@dataclass
class RetrievalCandidate:
    """A candidate trial from retrieval."""

    point_id: str
    payload: dict
    similarity_score: float = 0.0  # Vector similarity (0-1)
    _relaxed: bool = False  # Retrieved via relaxed filters
    _metadata_only: bool = False  # Retrieved without embedding


@dataclass
class FilterRejections:
    """Counts of candidates rejected by each filter.

    These are computed by comparing counts with progressively relaxed filters.
    Only populated in diagnostic mode to avoid extra queries.
    """

    by_oos: int = 0  # Rejected due to require_oos=True
    by_trades: int = 0  # Rejected due to min_trades
    by_drawdown: int = 0  # Rejected due to max_drawdown
    by_overfit_gap: int = 0  # Rejected due to max_overfit_gap
    by_regime: int = 0  # Rejected due to regime_tags mismatch
    total_before_filters: int = 0  # Total candidates before any quality filters


@dataclass
class RetrievalStats:
    """Statistics from retrieval operation."""

    strict_count: int = 0
    relaxed_count: int = 0
    total_returned: int = 0
    used_relaxed_filters: bool = False
    used_metadata_fallback: bool = False
    collection_name: str = ""
    embedding_model: str = ""

    # Filter rejection counts (only in diagnostic mode)
    filter_rejections: Optional[FilterRejections] = None


@dataclass
class RetrievalResult:
    """Result of retrieval operation."""

    candidates: list[RetrievalCandidate]
    stats: RetrievalStats
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Filter Building
# =============================================================================


def build_filters(
    req: RetrievalRequest,
    strict: bool = True,
    strategy_floors: Optional[dict] = None,
) -> dict:
    """
    Build filter dict for Qdrant query.

    Priority: request overrides > strategy defaults > workspace defaults

    Args:
        req: Retrieval request
        strict: If True, use strict quality filters
        strategy_floors: Strategy-specific floors from StrategySpec.kb_floors

    Returns:
        Filter dict for repository search
    """
    if strict:
        base = DEFAULT_STRICT_FILTERS.copy()
    else:
        base = DEFAULT_RELAXED_FILTERS.copy()

    # Apply strategy-specific floors (if provided)
    if strategy_floors:
        if strategy_floors.get("require_oos") is not None:
            base["require_oos"] = strategy_floors["require_oos"]
        if strategy_floors.get("max_overfit_gap") is not None:
            base["max_overfit_gap"] = strategy_floors["max_overfit_gap"]
        if strategy_floors.get("min_trades") is not None:
            base["min_trades"] = strategy_floors["min_trades"]
        if strategy_floors.get("max_drawdown") is not None:
            base["max_drawdown"] = strategy_floors["max_drawdown"]

    # Apply request overrides (highest priority)
    if req.require_oos is not None:
        base["require_oos"] = req.require_oos
    if req.max_overfit_gap is not None:
        base["max_overfit_gap"] = req.max_overfit_gap
    if req.min_trades is not None:
        base["min_trades"] = req.min_trades
    if req.max_drawdown is not None:
        base["max_drawdown"] = req.max_drawdown
    if req.regime_tags:
        base["regime_tags"] = req.regime_tags

    return base


# =============================================================================
# Retrieval Pipeline
# =============================================================================


class KBRetriever:
    """
    Retrieves similar trials from the Knowledge Base.

    Features:
    - Strict → relaxed filter fallback
    - Embed failure → metadata-only fallback
    - Tags relaxed candidates for weight penalty
    """

    def __init__(
        self,
        repository: KBTrialRepository,
        embedder: Optional[KBEmbeddingAdapter] = None,
    ):
        """
        Initialize retriever.

        Args:
            repository: KB trial repository
            embedder: KB embedding adapter (lazy-loaded if None)
        """
        self.repository = repository
        self._embedder = embedder

    @property
    def embedder(self) -> KBEmbeddingAdapter:
        """Lazy-load embedder."""
        if self._embedder is None:
            self._embedder = get_kb_embedder()
        return self._embedder

    async def retrieve(
        self,
        req: RetrievalRequest,
    ) -> RetrievalResult:
        """
        Retrieve similar trials for recommendation.

        Flow:
        1. Try strict filters with embedding search
        2. If < min_candidates, relax filters and search again
        3. If embedding fails, fall back to metadata-only search

        Args:
            req: Retrieval request

        Returns:
            RetrievalResult with candidates, stats, and warnings
        """
        warnings: list[str] = []
        stats = RetrievalStats(
            collection_name=self.repository.collection,
            embedding_model=self.embedder.model_id,
        )

        # Lookup strategy-specific floors
        strategy_floors: Optional[dict] = None
        try:
            strategy_spec = get_strategy(req.strategy_name)
            if strategy_spec and strategy_spec.kb_floors:
                strategy_floors = {
                    "require_oos": strategy_spec.kb_floors.require_oos,
                    "max_overfit_gap": strategy_spec.kb_floors.max_overfit_gap,
                    "min_trades": strategy_spec.kb_floors.min_trades,
                    "max_drawdown": strategy_spec.kb_floors.max_drawdown,
                }
                # Remove None values to allow default fallback
                strategy_floors = {
                    k: v for k, v in strategy_floors.items() if v is not None
                }
                logger.debug(
                    "Using strategy-specific KB floors",
                    strategy=req.strategy_name,
                    floors=strategy_floors,
                )
        except Exception as e:
            logger.warning(
                "Failed to lookup strategy floors",
                strategy=req.strategy_name,
                error=str(e),
            )

        # Get query vector (embed if not provided)
        query_vector = req.query_vector
        embed_failed = False

        if query_vector is None and req.query_regime is not None:
            try:
                from app.services.kb.trial_doc import regime_to_text

                regime_text = regime_to_text(req.query_regime)
                query_vector = await self.embedder.embed_single(regime_text)
            except EmbeddingError as e:
                logger.warning("Query embedding failed", error=e.message)
                warnings.append("embedding_failed")
                embed_failed = True

        # Try strict search first
        candidates: list[RetrievalCandidate] = []

        if query_vector is not None:
            strict_filters = build_filters(
                req, strict=True, strategy_floors=strategy_floors
            )
            strict_results = await self.repository.search(
                vector=query_vector,
                workspace_id=req.workspace_id,
                strategy_name=req.strategy_name,
                objective_type=req.objective_type,
                filters=strict_filters,
                limit=req.limit,
            )

            for r in strict_results:
                candidates.append(
                    RetrievalCandidate(
                        point_id=r["id"],
                        payload=r["payload"],
                        similarity_score=r["score"],
                        _relaxed=False,
                        _metadata_only=False,
                    )
                )

            stats.strict_count = len(candidates)

            # Check if we need relaxed fallback
            if len(candidates) < req.min_candidates:
                logger.info(
                    "Strict search below threshold, trying relaxed",
                    strict_count=len(candidates),
                    threshold=req.min_candidates,
                )

                relaxed_filters = build_filters(
                    req, strict=False, strategy_floors=strategy_floors
                )
                relaxed_results = await self.repository.search(
                    vector=query_vector,
                    workspace_id=req.workspace_id,
                    strategy_name=req.strategy_name,
                    objective_type=req.objective_type,
                    filters=relaxed_filters,
                    limit=req.limit,
                )

                # Add relaxed results that weren't in strict
                strict_ids = {c.point_id for c in candidates}
                for r in relaxed_results:
                    if r["id"] not in strict_ids:
                        candidates.append(
                            RetrievalCandidate(
                                point_id=r["id"],
                                payload=r["payload"],
                                similarity_score=r["score"],
                                _relaxed=True,
                                _metadata_only=False,
                            )
                        )

                stats.relaxed_count = len(candidates) - stats.strict_count
                stats.used_relaxed_filters = stats.relaxed_count > 0

                if stats.used_relaxed_filters:
                    warnings.append("used_relaxed_filters")

                    # Add specific reason
                    if len(strict_results) < req.min_candidates:
                        warnings.append("relaxed_filters_insufficient_strict")

        # Metadata-only fallback
        if embed_failed or (query_vector is None and len(candidates) == 0):
            logger.info("Using metadata-only fallback")

            # Use relaxed filters for fallback
            relaxed_filters = build_filters(
                req, strict=False, strategy_floors=strategy_floors
            )
            fallback_results = await self.repository.search_by_filters(
                workspace_id=req.workspace_id,
                strategy_name=req.strategy_name,
                objective_type=req.objective_type,
                filters=relaxed_filters,
                limit=req.limit,
            )

            # Sort by objective_score descending
            fallback_results.sort(
                key=lambda x: x["payload"].get("objective_score") or 0,
                reverse=True,
            )

            for r in fallback_results:
                candidates.append(
                    RetrievalCandidate(
                        point_id=r["id"],
                        payload=r["payload"],
                        similarity_score=0.0,  # No similarity without embedding
                        _relaxed=True,
                        _metadata_only=True,
                    )
                )

            stats.used_metadata_fallback = True
            warnings.append("metadata_only_fallback")

        stats.total_returned = len(candidates)

        # Diagnostic mode: compute filter rejection counts
        if req.diagnostic and query_vector is not None:
            stats.filter_rejections = await self._compute_filter_rejections(
                req=req,
                query_vector=query_vector,
                strategy_floors=strategy_floors,
            )

        logger.info(
            "Retrieval complete",
            strict_count=stats.strict_count,
            relaxed_count=stats.relaxed_count,
            total=stats.total_returned,
            used_relaxed=stats.used_relaxed_filters,
            metadata_fallback=stats.used_metadata_fallback,
            diagnostic=req.diagnostic,
        )

        return RetrievalResult(
            candidates=candidates,
            stats=stats,
            warnings=warnings,
        )

    async def _compute_filter_rejections(
        self,
        req: RetrievalRequest,
        query_vector: list[float],
        strategy_floors: Optional[dict] = None,
    ) -> FilterRejections:
        """
        Compute how many candidates were rejected by each filter.

        Runs queries with progressively relaxed filters to compute deltas.
        This is expensive, so only used in diagnostic mode.

        Priority chain: request override > strategy default > workspace default
        """
        rejections = FilterRejections()
        sf = strategy_floors or {}

        # Query 1: No quality filters (baseline)
        no_filters_count = await self.repository.count(
            vector=query_vector,
            workspace_id=req.workspace_id,
            strategy_name=req.strategy_name,
            objective_type=req.objective_type,
            filters={},
            limit=req.limit,
        )
        rejections.total_before_filters = no_filters_count

        # Query 2: With require_oos only
        oos_count = await self.repository.count(
            vector=query_vector,
            workspace_id=req.workspace_id,
            strategy_name=req.strategy_name,
            objective_type=req.objective_type,
            filters={"require_oos": True},
            limit=req.limit,
        )
        rejections.by_oos = no_filters_count - oos_count

        # Query 3: With require_oos + min_trades
        # Priority: request override > strategy default > workspace default
        min_trades = (
            req.min_trades
            or sf.get("min_trades")
            or DEFAULT_STRICT_FILTERS["min_trades"]
        )
        trades_count = await self.repository.count(
            vector=query_vector,
            workspace_id=req.workspace_id,
            strategy_name=req.strategy_name,
            objective_type=req.objective_type,
            filters={"require_oos": True, "min_trades": min_trades},
            limit=req.limit,
        )
        rejections.by_trades = oos_count - trades_count

        # Query 4: With require_oos + min_trades + max_drawdown
        max_dd = (
            req.max_drawdown
            or sf.get("max_drawdown")
            or DEFAULT_STRICT_FILTERS["max_drawdown"]
        )
        dd_count = await self.repository.count(
            vector=query_vector,
            workspace_id=req.workspace_id,
            strategy_name=req.strategy_name,
            objective_type=req.objective_type,
            filters={
                "require_oos": True,
                "min_trades": min_trades,
                "max_drawdown": max_dd,
            },
            limit=req.limit,
        )
        rejections.by_drawdown = trades_count - dd_count

        # Query 5: All strict filters (including overfit_gap)
        max_overfit = (
            req.max_overfit_gap
            or sf.get("max_overfit_gap")
            or DEFAULT_STRICT_FILTERS["max_overfit_gap"]
        )
        strict_count = await self.repository.count(
            vector=query_vector,
            workspace_id=req.workspace_id,
            strategy_name=req.strategy_name,
            objective_type=req.objective_type,
            filters={
                "require_oos": True,
                "min_trades": min_trades,
                "max_drawdown": max_dd,
                "max_overfit_gap": max_overfit,
            },
            limit=req.limit,
        )
        rejections.by_overfit_gap = dd_count - strict_count

        # Query 6: With regime tags (if provided)
        if req.regime_tags:
            regime_count = await self.repository.count(
                vector=query_vector,
                workspace_id=req.workspace_id,
                strategy_name=req.strategy_name,
                objective_type=req.objective_type,
                filters={
                    "require_oos": True,
                    "min_trades": min_trades,
                    "max_drawdown": max_dd,
                    "max_overfit_gap": max_overfit,
                    "regime_tags": req.regime_tags,
                },
                limit=req.limit,
            )
            rejections.by_regime = strict_count - regime_count

        logger.info(
            "Filter rejection analysis",
            total_before=rejections.total_before_filters,
            by_oos=rejections.by_oos,
            by_trades=rejections.by_trades,
            by_drawdown=rejections.by_drawdown,
            by_overfit_gap=rejections.by_overfit_gap,
            by_regime=rejections.by_regime,
        )

        return rejections


# =============================================================================
# Module-level helpers
# =============================================================================


async def retrieve_candidates(
    req: RetrievalRequest,
    repository: KBTrialRepository,
    embedder: Optional[KBEmbeddingAdapter] = None,
) -> RetrievalResult:
    """
    Convenience function for retrieving candidates.

    Args:
        req: Retrieval request
        repository: KB trial repository
        embedder: Optional embedding adapter

    Returns:
        RetrievalResult
    """
    retriever = KBRetriever(repository=repository, embedder=embedder)
    return await retriever.retrieve(req)
