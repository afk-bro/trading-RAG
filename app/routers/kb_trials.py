"""
Trading Knowledge Base API endpoints.

Provides recommendation and ingestion endpoints for the Trading KB:
- POST /kb/trials/recommend - Get parameter recommendations
- POST /kb/trials/ingest - Ingest trials from tune runs (admin)
"""

import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional
from uuid import UUID

import anyio
import sentry_sdk
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field, model_validator

from app.config import Settings, get_settings
from app.deps.security import (
    require_admin_token,
    get_current_user,
    require_workspace_access,
    get_rate_limiter,
    get_workspace_semaphore,
    CurrentUser,
)
from app.routers.metrics import (
    record_kb_recommend,
    record_kb_embed_error,
    record_kb_qdrant_error,
)
from app.services.strategies.registry import (
    ObjectiveType,
    get_strategy,
    list_strategies,
    list_objectives,
)

router = APIRouter(prefix="/kb/trials", tags=["trading-kb"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


# =============================================================================
# Constants - Bounds
# =============================================================================

MAX_RETRIEVE_K = 500
MAX_RERANK_KEEP = 200
MAX_TOP_K = 50

DEFAULT_RETRIEVE_K = 100
DEFAULT_RERANK_KEEP = 50
DEFAULT_TOP_K = 20

# Timeouts (seconds)
RECOMMEND_TIMEOUT_S = 30
EMBED_TIMEOUT_S = 5
QDRANT_TIMEOUT_S = 10

# File upload limits
MAX_OHLCV_FILE_SIZE_MB = 20
MAX_OHLCV_FILE_SIZE_BYTES = MAX_OHLCV_FILE_SIZE_MB * 1024 * 1024
MAX_OHLCV_ROWS = 100_000  # Reject files with more rows

# Defensive request bounds
MAX_BARS_PROCESSED = 200_000  # Maximum OHLCV bars to process
MAX_PARAM_KEYS = 50  # Maximum unique param keys in payload
MAX_STRING_LENGTH = 256  # Max length for string fields (dataset_id, instrument, etc.)
MAX_REGIME_TAGS = 20  # Maximum regime tags in request

# Rate limiting defaults
RATE_LIMIT_UPLOAD_PER_MIN = 5  # uploads per minute per IP
RATE_LIMIT_RECOMMEND_PER_MIN_WS = 30  # recommend per minute per workspace
RATE_LIMIT_RECOMMEND_PER_MIN_IP = 60  # recommend per minute per IP
WORKSPACE_MAX_CONCURRENT = 2  # max concurrent recommend requests per workspace


# =============================================================================
# Enums
# =============================================================================


class RecommendMode(str, Enum):
    """Recommendation mode."""
    FULL = "full"  # Full pipeline with aggregation
    DEBUG = "debug"  # Return candidates without aggregation


class RecommendStatus(str, Enum):
    """Recommendation result status."""
    OK = "ok"
    DEGRADED = "degraded"
    NONE = "none"


# =============================================================================
# Request Schemas
# =============================================================================


class RecommendRequest(BaseModel):
    """Request for parameter recommendations."""

    # Required
    workspace_id: UUID = Field(..., description="Workspace ID")
    strategy_name: str = Field(..., min_length=1, max_length=100, description="Strategy name from registry")
    objective_type: str = Field(..., min_length=1, max_length=50, description="Objective function type")

    # Dataset source (exactly one required)
    dataset_id: Optional[str] = Field(
        None,
        max_length=MAX_STRING_LENGTH,
        description="Reference dataset ID for regime computation",
    )
    # Note: ohlcv_file handled separately via Form/File

    # Optional regime override
    regime_tags: Optional[list[str]] = Field(
        None,
        max_length=MAX_REGIME_TAGS,
        description="Override regime tags for query",
    )

    # Filter overrides (bounded by workspace defaults)
    require_oos: Optional[bool] = Field(None, description="Require OOS metrics")
    max_overfit_gap: Optional[float] = Field(None, ge=0, le=1, description="Max overfit gap (0-1)")
    min_trades: Optional[int] = Field(None, ge=1, le=1000, description="Minimum trade count")
    max_drawdown: Optional[float] = Field(None, ge=0, le=1, description="Max drawdown fraction (0-1)")

    # Retrieval knobs (bounded)
    retrieve_k: int = Field(DEFAULT_RETRIEVE_K, ge=1, le=MAX_RETRIEVE_K, description="Max candidates to retrieve")
    rerank_keep: int = Field(DEFAULT_RERANK_KEEP, ge=1, le=MAX_RERANK_KEEP, description="Top M after reranking")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Top K for aggregation")

    # v1.5 context (for duration stats, cluster stats lookups)
    symbol: Optional[str] = Field(
        None, max_length=50, description="Trading symbol (e.g., 'BTC/USDT') for duration stats"
    )
    strategy_entity_id: Optional[UUID] = Field(
        None, description="Strategy entity ID for cluster stats lookup"
    )
    timeframe: Optional[str] = Field(
        None, max_length=10, description="Timeframe (e.g., '5m', '1h') for stats lookups"
    )

    # Debug options
    include_candidates: bool = Field(False, description="Include top candidates in response (debug)")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_regime_tags(self):
        """Validate individual regime tag lengths."""
        if self.regime_tags:
            for tag in self.regime_tags:
                if len(tag) > MAX_STRING_LENGTH:
                    raise ValueError(f"Regime tag too long: max {MAX_STRING_LENGTH} chars")
        return self


class IngestRequest(BaseModel):
    """Request for ingesting trials from tune runs."""

    workspace_id: UUID = Field(..., description="Workspace ID")
    since: Optional[datetime] = Field(None, description="Only ingest runs after this time")
    tune_ids: Optional[list[UUID]] = Field(None, max_length=100, description="Specific tune IDs to ingest")
    dry_run: bool = Field(False, description="Preview without writing")
    reembed: bool = Field(False, description="Re-embed existing trials")
    batch_size: int = Field(50, ge=1, le=200, description="Ingestion batch size")

    model_config = {"extra": "forbid"}


# =============================================================================
# Response Schemas
# =============================================================================


class CandidateSummary(BaseModel):
    """Summary of a candidate trial (for debug/transparency)."""

    point_id: str
    tune_run_id: Optional[str] = None
    strategy_name: str
    objective_score: float
    similarity_score: float
    jaccard_score: float
    rerank_score: float
    used_regime_source: Literal["oos", "is", "none"]
    is_relaxed: bool = False
    is_metadata_only: bool = False
    params: dict = Field(default_factory=dict)
    regime_tags: list[str] = Field(default_factory=list)


class ParamSpreadInfo(BaseModel):
    """Spread/confidence info for a parameter."""

    name: str
    value: float | int | bool | str
    count_used: int
    spread: Optional[float] = None  # IQR for numeric
    mode_fraction: Optional[float] = None  # Fraction for categorical


class FilterRejectionsInfo(BaseModel):
    """Filter rejection counts for debug diagnostics."""

    total_before_filters: int = Field(0, description="Total candidates before quality filters")
    by_oos: int = Field(0, description="Rejected by require_oos=True")
    by_trades: int = Field(0, description="Rejected by min_trades")
    by_drawdown: int = Field(0, description="Rejected by max_drawdown")
    by_overfit_gap: int = Field(0, description="Rejected by max_overfit_gap")
    by_regime: int = Field(0, description="Rejected by regime_tags mismatch")


class RelaxationSuggestionInfo(BaseModel):
    """Single-axis relaxation suggestion with risk note."""

    filter_name: str = Field(..., description="Name of the filter to relax")
    current_value: Optional[float | int | bool] = Field(None, description="Current filter value")
    suggested_value: Optional[float | int | bool] = Field(None, description="Suggested relaxed value")
    estimated_candidates: int = Field(0, description="Estimated candidates with this single change")
    risk_note: str = Field("", description="Warning about the trade-off")


class RecommendedRelaxedSettingsInfo(BaseModel):
    """Suggested relaxed filter settings that would yield candidates.

    Only returned when status='none' to help users understand
    what constraints to loosen for recommendations.

    Each suggestion relaxes ONE filter at a time so users can
    evaluate trade-offs independently.
    """

    suggestions: list[RelaxationSuggestionInfo] = Field(
        default_factory=list,
        description="Single-axis relaxation suggestions, sorted by impact",
    )


# =============================================================================
# v1.5 Live Intelligence Response Models
# =============================================================================


class RegimeStateStabilityInfo(BaseModel):
    """FSM state info for regime stability (v1.5)."""

    candidate_key: Optional[str] = Field(None, description="Current candidate regime key")
    candidate_bars: int = Field(0, description="Bars candidate has persisted")
    M: int = Field(20, description="Persistence bars required for transition")
    C_enter: float = Field(0.75, description="Confidence threshold to confirm transition")
    C_exit: float = Field(0.55, description="Confidence threshold to consider change")


class WindowMetadataInfo(BaseModel):
    """Window metadata for rolling computations (v1.5)."""

    regime_age_bars: int = Field(0, description="Bars since stable regime confirmed")
    performance_window: Optional[dict] = Field(None, description="Performance window config")
    distance_window: Optional[dict] = Field(None, description="Distance computation window config")


class RecommendResponse(BaseModel):
    """Response from parameter recommendation."""

    # Request tracking
    request_id: str = Field(..., description="Unique request ID for logging")

    # Core result
    params: dict = Field(default_factory=dict, description="Recommended parameters")
    status: RecommendStatus = Field(..., description="Recommendation quality status")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score (0-1)")

    # Counts
    count_used: int = Field(0, description="Trials used in aggregation")
    retrieval_strict_count: int = Field(0, description="Trials from strict filters")
    retrieval_relaxed_count: int = Field(0, description="Trials from relaxed filters")

    # Flags
    used_relaxed_filters: bool = Field(False, description="Whether relaxed filters were used")
    used_metadata_fallback: bool = Field(False, description="Whether metadata-only fallback was used")

    # Quality indicators
    warnings: list[str] = Field(default_factory=list, description="Warning codes")
    reasons: list[str] = Field(default_factory=list, description="Reason codes for status")
    suggested_actions: list[str] = Field(default_factory=list, description="Suggested actions")

    # Context (for debugging)
    query_regime_tags: list[str] = Field(default_factory=list, description="Query regime tags used")
    active_collection: str = Field("", description="Qdrant collection used")
    embedding_model_id: str = Field("", description="Embedding model used")

    # Optional debug data
    top_candidates: Optional[list[CandidateSummary]] = Field(None, description="Top candidates (debug mode)")
    param_spreads: Optional[dict[str, ParamSpreadInfo]] = Field(None, description="Parameter spread info")
    filter_rejections: Optional[FilterRejectionsInfo] = Field(None, description="Filter rejection counts (debug mode)")

    # Self-healing guidance (only when status='none')
    recommended_relaxed_settings: Optional[RecommendedRelaxedSettingsInfo] = Field(
        None,
        description="Suggested relaxed filter settings that would yield candidates (status='none' only)",
    )

    # =========================================================================
    # v1.5 Live Intelligence Fields
    # =========================================================================

    # Confidence decomposition (v1.5)
    regime_fit_confidence: Optional[float] = Field(
        None, ge=0, le=1, description="How well current market matches historical regime (0-1)"
    )
    regime_distance_z: Optional[float] = Field(
        None, description="Z-score distance from neighborhood"
    )
    distance_baseline: Optional[str] = Field(
        None, description="Baseline used for distance: 'composite' | 'marginal' | 'neighbors_only'"
    )
    distance_n: Optional[int] = Field(
        None, ge=0, description="Number of neighbors used for distance computation"
    )

    # Duration fields (v1.5)
    regime_age_bars: Optional[int] = Field(
        None, ge=0, description="Bars since stable regime confirmed"
    )
    regime_half_life_bars: Optional[int] = Field(
        None, ge=0, description="Median historical duration for this regime"
    )
    expected_remaining_bars: Optional[int] = Field(
        None, ge=0, description="max(0, median - age)"
    )
    duration_iqr_bars: Optional[list[int]] = Field(
        None, description="[p25, p75] historical duration"
    )
    remaining_iqr_bars: Optional[list[int]] = Field(
        None, description="[max(0, p25-age), max(0, p75-age)]"
    )
    duration_baseline: Optional[str] = Field(
        None, description="Baseline: 'composite_symbol' | 'marginal' | 'global_timeframe'"
    )
    duration_n: Optional[int] = Field(
        None, ge=0, description="Number of segments used for duration stats"
    )

    # FSM state (v1.5)
    stable_regime_key: Optional[str] = Field(
        None, description="Confirmed stable regime key"
    )
    raw_regime_key: Optional[str] = Field(
        None, description="Raw current classification"
    )
    regime_state_stability: Optional[RegimeStateStabilityInfo] = Field(
        None, description="FSM state details"
    )

    # Window metadata (v1.5)
    windows: Optional[WindowMetadataInfo] = Field(
        None, description="Window metadata for rolling computations"
    )

    # Missing field reasons (v1.5)
    missing: list[str] = Field(
        default_factory=list, description="Reasons for unavailable v1.5 fields"
    )


class IngestResponse(BaseModel):
    """Response from trial ingestion."""

    request_id: str
    workspace_id: str
    ingested_count: int
    skipped_count: int
    error_count: int
    dry_run: bool
    collection: str
    embedding_model: str
    errors: list[str] = Field(default_factory=list, max_length=50)
    duration_ms: int


class StrategyListItem(BaseModel):
    """Strategy list item for error responses."""

    name: str
    display_name: str
    supported_objectives: list[str]


class ErrorDetail(BaseModel):
    """Detailed error response."""

    error: str
    code: str
    valid_options: Optional[list[str]] = None
    strategies: Optional[list[StrategyListItem]] = None


# =============================================================================
# Dependencies
# =============================================================================


def _get_repository():
    """Get KB trial repository."""
    from app.repositories.kb_trials import KBTrialRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return KBTrialRepository(_db_pool)


async def _get_recommender():
    """Get KB recommender with dependencies."""
    from app.services.kb.recommend import KBRecommender

    repo = _get_repository()
    return KBRecommender(repository=repo)


def _validate_strategy(strategy_name: str) -> None:
    """Validate strategy exists in registry."""
    spec = get_strategy(strategy_name)
    if spec is None:
        available_names = list_strategies()
        strategies = []
        for name in available_names:
            s = get_strategy(name)
            if s:
                strategies.append(
                    StrategyListItem(
                        name=s.name,
                        display_name=s.display_name,
                        supported_objectives=[o.value for o in s.supported_objectives],
                    ).model_dump()
                )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Strategy '{strategy_name}' not found in registry",
                "code": "INVALID_STRATEGY",
                "valid_options": available_names,
                "strategies": strategies,
            },
        )


def _validate_objective(strategy_name: str, objective_type: str) -> None:
    """Validate objective is supported for strategy."""
    spec = get_strategy(strategy_name)
    if spec is None:
        # Let strategy validation handle unknown strategies
        return

    valid_objectives = [o.value for o in spec.supported_objectives]

    if objective_type not in valid_objectives:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Objective '{objective_type}' not supported for strategy '{strategy_name}'",
                "code": "INVALID_OBJECTIVE",
                "valid_options": valid_objectives,
            },
        )


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    responses={
        200: {"description": "Recommendation generated (check status field)"},
        400: {"description": "Invalid strategy or objective", "model": ErrorDetail},
        403: {"description": "Access denied to workspace"},
        422: {"description": "Validation error (bounds exceeded)"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable (Qdrant/DB down)"},
    },
    summary="Get parameter recommendations",
    description="""
Generate parameter recommendations based on similar historical trials.

**Modes:**
- `mode=full` (default): Full pipeline with aggregation
- `mode=debug`: Returns top candidates without aggregation

**Status codes in response:**
- `ok`: Clean recommendation with good confidence
- `degraded`: Recommendation available but with caveats (relaxed filters, repairs)
- `none`: No recommendation possible (no matching trials)

**Note:** Returns 200 even when status=none. Use 503 only for infrastructure failures.
""",
)
async def recommend(
    http_request: Request,
    request: RecommendRequest,
    mode: RecommendMode = Query(RecommendMode.FULL, description="Recommendation mode"),
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(get_current_user),
    _rate_ip: None = Depends(get_rate_limiter().check("recommend_ip", RATE_LIMIT_RECOMMEND_PER_MIN_IP)),
) -> RecommendResponse:
    """Generate parameter recommendations."""
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    # Workspace access check (multi-tenant stub)
    require_workspace_access(request.workspace_id, user)

    # Per-workspace rate limit
    await get_rate_limiter().check(
        "recommend_ws",
        RATE_LIMIT_RECOMMEND_PER_MIN_WS,
        key_func=lambda _: str(request.workspace_id),
    )(http_request)

    # Set Sentry transaction tags for filtering/grouping
    sentry_sdk.set_tag("request_id", request_id)
    sentry_sdk.set_tag("workspace_id", str(request.workspace_id))
    sentry_sdk.set_tag("strategy", request.strategy_name)
    sentry_sdk.set_tag("objective", request.objective_type)
    sentry_sdk.set_tag("mode", mode.value)

    # Set detailed context (don't include large payloads like OHLCV)
    sentry_sdk.set_context("kb_recommend_request", {
        "request_id": request_id,
        "retrieve_k": request.retrieve_k,
        "rerank_keep": request.rerank_keep,
        "top_k": request.top_k,
        "require_oos": request.require_oos,
        "min_trades": request.min_trades,
        "max_drawdown": request.max_drawdown,
        "max_overfit_gap": request.max_overfit_gap,
        "has_regime_tags": bool(request.regime_tags),
        "has_dataset_id": bool(request.dataset_id),
    })

    logger.info(
        "kb_recommend_start",
        request_id=request_id,
        workspace_id=str(request.workspace_id),
        strategy_name=request.strategy_name,
        objective_type=request.objective_type,
        mode=mode.value,
        retrieve_k=request.retrieve_k,
        rerank_keep=request.rerank_keep,
        top_k=request.top_k,
        has_regime_tags=bool(request.regime_tags),
        has_dataset_id=bool(request.dataset_id),
    )

    # Validate strategy and objective
    with sentry_sdk.start_span(op="validate", description="Validate strategy and objective"):
        _validate_strategy(request.strategy_name)
        _validate_objective(request.strategy_name, request.objective_type)

    # Check for regime tags or dataset_id
    has_dataset_id = request.dataset_id is not None
    has_regime_tags = request.regime_tags is not None and len(request.regime_tags) > 0

    # Note: OHLCV file upload handled via separate endpoint /recommend/upload

    ohlcv_data = None
    if has_dataset_id:
        # TODO: Load dataset from storage by ID
        pass
    elif False:  # Placeholder for file handling
        try:
            content = b""  # Would be file content
            ohlcv_data = _parse_ohlcv_file(content, "data.json")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"Failed to parse OHLCV file: {str(e)}",
                    "code": "INVALID_OHLCV_FILE",
                },
            )

    # Build regime snapshot from various sources
    query_regime = None
    if ohlcv_data:
        try:
            from app.services.kb.regime import compute_regime_from_ohlcv

            query_regime = compute_regime_from_ohlcv(ohlcv_data, source="upload")
        except Exception as e:
            logger.warning("Failed to compute regime from OHLCV", error=str(e))

    # Override with explicit regime tags if provided
    if has_regime_tags:
        from app.services.kb.types import RegimeSnapshot

        if query_regime:
            query_regime.regime_tags = request.regime_tags
        else:
            query_regime = RegimeSnapshot(regime_tags=request.regime_tags)

    # Get recommender
    try:
        recommender = await _get_recommender()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to initialize recommender", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to initialize recommendation service",
        )

    # Build recommend request
    from app.services.kb.recommend import RecommendRequest as InternalRecommendRequest

    # Enable diagnostic mode in debug mode (computes filter rejection counts)
    diagnostic = mode == RecommendMode.DEBUG

    internal_req = InternalRecommendRequest(
        workspace_id=request.workspace_id,
        strategy_name=request.strategy_name,
        objective_type=request.objective_type,
        query_regime=query_regime,
        require_oos=request.require_oos,
        max_overfit_gap=request.max_overfit_gap,
        min_trades=request.min_trades,
        max_drawdown=request.max_drawdown,
        retrieve_limit=request.retrieve_k,
        rerank_top_m=request.rerank_keep,
        aggregate_top_k=request.top_k,
        diagnostic=diagnostic,
        # v1.5 context
        symbol=request.symbol,
        strategy_entity_id=request.strategy_entity_id,
        timeframe=request.timeframe,
    )

    # Execute with timeouts
    try:
        with sentry_sdk.start_span(op="recommend", description="KB recommend pipeline"):
            with anyio.fail_after(RECOMMEND_TIMEOUT_S):
                result = await recommender.recommend(internal_req)
    except TimeoutError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "kb_recommend_timeout",
            request_id=request_id,
            workspace_id=str(request.workspace_id),
            strategy_name=request.strategy_name,
            objective_type=request.objective_type,
            mode=mode.value,
            status="error",
            timeout_s=RECOMMEND_TIMEOUT_S,
            total_ms=round(elapsed_ms, 1),
        )
        record_kb_qdrant_error()  # Timeout usually means Qdrant issue

        # Capture with fingerprint for grouping
        # Note: embed_model/collection tags not available yet (failure occurred before result)
        # Use workspace_id tag (set at request start) to correlate with workspace config
        with sentry_sdk.push_scope() as scope:
            scope.fingerprint = ["kb", "recommend_timeout"]
            scope.set_context("kb_error", {
                "elapsed_ms": round(elapsed_ms, 1),
                "timeout_s": RECOMMEND_TIMEOUT_S,
                "retrieve_k": request.retrieve_k,
            })
            sentry_sdk.capture_exception(e)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation request timed out after {RECOMMEND_TIMEOUT_S}s",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        error_type = type(e).__name__
        logger.error(
            "kb_recommend_error",
            request_id=request_id,
            workspace_id=str(request.workspace_id),
            strategy_name=request.strategy_name,
            objective_type=request.objective_type,
            mode=mode.value,
            status="error",
            error=str(e),
            error_type=error_type,
            total_ms=round(elapsed_ms, 1),
        )

        # Record appropriate error metric and capture with fingerprint
        # Note: embed_model/collection tags not available yet (failure occurred before result)
        # Use workspace_id tag (set at request start) to correlate with workspace config
        error_context = {
            "elapsed_ms": round(elapsed_ms, 1),
            "error_type": error_type,
            "retrieve_k": request.retrieve_k,
        }
        if "embed" in str(e).lower() or "ollama" in str(e).lower():
            record_kb_embed_error()
            with sentry_sdk.push_scope() as scope:
                scope.fingerprint = ["kb", "embed_error"]
                scope.set_context("kb_error", error_context)
                sentry_sdk.capture_exception(e)
        elif "qdrant" in str(e).lower():
            record_kb_qdrant_error()
            with sentry_sdk.push_scope() as scope:
                scope.fingerprint = ["kb", "qdrant_error"]
                scope.set_context("kb_error", error_context)
                sentry_sdk.capture_exception(e)
        else:
            record_kb_qdrant_error()
            with sentry_sdk.push_scope() as scope:
                scope.fingerprint = ["kb", "recommend_error", error_type]
                scope.set_context("kb_error", error_context)
                sentry_sdk.capture_exception(e)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation failed: {str(e)}",
        )

    # Build response
    response = RecommendResponse(
        request_id=request_id,
        params=result.params if mode == RecommendMode.FULL else {},
        status=RecommendStatus(result.status),
        confidence=result.confidence,
        count_used=result.count_used,
        retrieval_strict_count=result.retrieval_strict_count,
        retrieval_relaxed_count=result.retrieval_relaxed_count,
        used_relaxed_filters=result.used_relaxed_filters,
        used_metadata_fallback=result.used_metadata_fallback,
        warnings=result.warnings,
        reasons=result.reasons,
        suggested_actions=result.suggested_actions,
        query_regime_tags=result.query_regime_tags,
        active_collection=result.collection_name,
        embedding_model_id=result.embedding_model,
        # v1.5 fields
        regime_fit_confidence=result.regime_fit_confidence,
        regime_distance_z=result.regime_distance_z,
        distance_baseline=result.distance_baseline,
        distance_n=result.distance_n,
        regime_age_bars=result.regime_age_bars,
        regime_half_life_bars=result.regime_half_life_bars,
        expected_remaining_bars=result.expected_remaining_bars,
        duration_iqr_bars=result.duration_iqr_bars,
        remaining_iqr_bars=result.remaining_iqr_bars,
        duration_baseline=result.duration_baseline,
        duration_n=result.duration_n,
        stable_regime_key=result.stable_regime_key,
        raw_regime_key=result.raw_regime_key,
        regime_state_stability=(
            RegimeStateStabilityInfo(
                candidate_key=result.regime_state_stability.candidate_key,
                candidate_bars=result.regime_state_stability.candidate_bars,
                M=result.regime_state_stability.M,
                C_enter=result.regime_state_stability.C_enter,
                C_exit=result.regime_state_stability.C_exit,
            )
            if result.regime_state_stability
            else None
        ),
        windows=(
            WindowMetadataInfo(
                regime_age_bars=result.windows.regime_age_bars,
                performance_window=result.windows.performance_window,
                distance_window=result.windows.distance_window,
            )
            if result.windows
            else None
        ),
        missing=result.missing,
    )

    # Add debug data if requested
    if mode == RecommendMode.DEBUG or request.include_candidates:
        response.top_candidates = [
            CandidateSummary(
                point_id=t.point_id,
                tune_run_id=t.params.get("tune_run_id"),
                strategy_name=t.strategy_name,
                objective_score=t.objective_score,
                similarity_score=t.similarity_score,
                jaccard_score=t.jaccard_score,
                rerank_score=t.rerank_score,
                used_regime_source="oos",  # TODO: get from candidate
                is_relaxed=False,  # TODO: get from candidate
                is_metadata_only=False,  # TODO: get from candidate
                params=t.params,
            )
            for t in result.top_trials
        ]

        # Add filter rejections in debug mode
        if result.filter_rejections:
            response.filter_rejections = FilterRejectionsInfo(
                total_before_filters=result.filter_rejections.total_before_filters,
                by_oos=result.filter_rejections.by_oos,
                by_trades=result.filter_rejections.by_trades,
                by_drawdown=result.filter_rejections.by_drawdown,
                by_overfit_gap=result.filter_rejections.by_overfit_gap,
                by_regime=result.filter_rejections.by_regime,
            )

    # Add recommended relaxed settings when status='none'
    if result.recommended_relaxed_settings and result.recommended_relaxed_settings.suggestions:
        response.recommended_relaxed_settings = RecommendedRelaxedSettingsInfo(
            suggestions=[
                RelaxationSuggestionInfo(
                    filter_name=s.filter_name,
                    current_value=s.current_value,
                    suggested_value=s.suggested_value,
                    estimated_candidates=s.estimated_candidates,
                    risk_note=s.risk_note,
                )
                for s in result.recommended_relaxed_settings.suggestions
            ]
        )

    # Extract timings from result
    timings = result.timings
    total_ms = timings.total_ms if timings else (time.perf_counter() - start_time) * 1000
    qdrant_ms = timings.qdrant_ms if timings else 0.0
    embed_ms = timings.embed_ms if timings else 0.0
    regime_ms = timings.regime_ms if timings else 0.0
    rerank_ms = timings.rerank_ms if timings else 0.0
    aggregate_ms = timings.aggregate_ms if timings else 0.0

    # Check for params repair (any warning starting with param_ or constraint_)
    params_repaired = any(
        w.startswith("param_") or w.startswith("constraint_")
        for w in result.warnings
    )
    incomplete_regime = "query_regime_computation_failed" in result.warnings

    # Comprehensive structured log record (9.1)
    logger.info(
        "kb_recommend_complete",
        # Identifiers
        request_id=request_id,
        workspace_id=str(request.workspace_id),
        strategy_name=request.strategy_name,
        objective_type=request.objective_type,
        mode=mode.value,
        # Status
        status=result.status,
        confidence=result.confidence,
        reasons=result.reasons,
        # Fallbacks
        strict_to_relaxed=result.used_relaxed_filters,
        metadata_only=result.used_metadata_fallback,
        repaired_params=params_repaired,
        # Counts
        candidates_returned=result.count_used,
        after_strict=result.retrieval_strict_count,
        after_relaxed=result.retrieval_relaxed_count,
        top_k=len(result.top_trials) if result.top_trials else 0,
        # Timings (ms)
        total_ms=round(total_ms, 1),
        regime_ms=round(regime_ms, 1),
        embed_ms=round(embed_ms, 1),
        qdrant_ms=round(qdrant_ms, 1),
        rerank_ms=round(rerank_ms, 1),
        aggregate_ms=round(aggregate_ms, 1),
        # Model info
        embedding_model_id=result.embedding_model,
        active_collection=result.collection_name,
        # Query context
        query_regime_tags=result.query_regime_tags,
    )

    # Record Prometheus metrics (9.3)
    record_kb_recommend(
        status=result.status,
        confidence=result.confidence,
        total_ms=total_ms,
        qdrant_ms=qdrant_ms,
        embed_ms=embed_ms,
        used_relaxed=result.used_relaxed_filters,
        used_metadata_fallback=result.used_metadata_fallback,
        params_repaired=params_repaired,
        incomplete_regime=incomplete_regime,
    )

    # Update Sentry with result status and model/collection tags (for filtering)
    sentry_sdk.set_tag("kb_status", result.status)
    sentry_sdk.set_tag("embed_model", result.embedding_model)
    sentry_sdk.set_tag("collection", result.collection_name)
    confidence_bucket = (
        "high" if result.confidence and result.confidence >= 0.7 else
        "medium" if result.confidence and result.confidence >= 0.4 else
        "low" if result.confidence else "none"
    )
    sentry_sdk.set_tag("kb_confidence", confidence_bucket)

    # Add ops breadcrumb for collection choice (helps debug mismatched environments)
    sentry_sdk.add_breadcrumb(
        category="kb.config",
        message=f"Using collection {result.collection_name}",
        level="info",
        data={
            "collection": result.collection_name,
            "embed_model": result.embedding_model,
            "workspace_id": str(request.workspace_id),
        },
    )

    # Add per-step duration measurements for dashboards (9.2 polish)
    sentry_sdk.set_measurement("kb.total_ms", total_ms, "millisecond")
    sentry_sdk.set_measurement("kb.regime_ms", regime_ms, "millisecond")
    sentry_sdk.set_measurement("kb.embed_ms", embed_ms, "millisecond")
    sentry_sdk.set_measurement("kb.qdrant_ms", qdrant_ms, "millisecond")
    sentry_sdk.set_measurement("kb.rerank_ms", rerank_ms, "millisecond")
    sentry_sdk.set_measurement("kb.aggregate_ms", aggregate_ms, "millisecond")

    # Add span data for counts
    span = sentry_sdk.get_current_span()
    if span:
        span.set_data("candidates_returned", result.count_used)
        span.set_data("strict_count", result.retrieval_strict_count)
        span.set_data("relaxed_count", result.retrieval_relaxed_count)
        span.set_data("used_relaxed", result.used_relaxed_filters)
        span.set_data("used_metadata_fallback", result.used_metadata_fallback)

    # Add breadcrumb for degraded status (searchable but not an error)
    if result.status == "degraded":
        sentry_sdk.add_breadcrumb(
            category="kb.recommend",
            message="Recommendation degraded",
            level="warning",
            data={
                "reasons": result.reasons,
                "status": result.status,
                "confidence": result.confidence,
                "used_relaxed": result.used_relaxed_filters,
                "used_metadata_fallback": result.used_metadata_fallback,
            },
        )
    elif result.status == "none":
        sentry_sdk.add_breadcrumb(
            category="kb.recommend",
            message="No recommendation available",
            level="info",
            data={
                "reasons": result.reasons,
                "suggested_actions": result.suggested_actions,
            },
        )

    return response


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        200: {"description": "Ingestion completed"},
        401: {"description": "Admin token required"},
        403: {"description": "Admin access denied"},
        503: {"description": "Service unavailable"},
    },
    summary="Ingest trials from tune runs (admin)",
    description="""
Ingest completed tune runs into the Trading KB vector store.

**Admin-only endpoint.** Requires X-Admin-Token header.

Options:
- `since`: Only ingest runs completed after this timestamp
- `tune_ids`: Ingest specific tunes only
- `dry_run`: Preview without writing
- `reembed`: Re-embed existing trials (for model updates)
""",
)
async def ingest(
    request: IngestRequest,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_token),
) -> IngestResponse:
    """Ingest trials from tune runs."""
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    logger.info(
        "Ingest request",
        request_id=request_id,
        workspace_id=str(request.workspace_id),
        since=request.since.isoformat() if request.since else None,
        tune_count=len(request.tune_ids) if request.tune_ids else "all",
        dry_run=request.dry_run,
        reembed=request.reembed,
    )

    try:
        from app.services.kb.ingest import ingest_trials

        result = await ingest_trials(
            workspace_id=request.workspace_id,
            since=request.since,
            tune_ids=request.tune_ids,
            dry_run=request.dry_run,
            reembed=request.reembed,
            batch_size=request.batch_size,
        )

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        logger.info(
            "Ingest complete",
            request_id=request_id,
            ingested=result.ingested_count,
            skipped=result.skipped_count,
            errors=result.error_count,
            duration_ms=duration_ms,
        )

        return IngestResponse(
            request_id=request_id,
            workspace_id=str(request.workspace_id),
            ingested_count=result.ingested_count,
            skipped_count=result.skipped_count,
            error_count=result.error_count,
            dry_run=request.dry_run,
            collection=result.collection_name,
            embedding_model=result.embedding_model,
            errors=result.errors[:50],  # Limit error messages
            duration_ms=duration_ms,
        )

    except Exception as e:
        logger.error("Ingest failed", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ingestion failed: {str(e)}",
        )


# =============================================================================
# File Upload Endpoint (Separate for memory safety)
# =============================================================================


class OHLCVUploadResponse(BaseModel):
    """Response from OHLCV file upload."""

    request_id: str
    row_count: int
    regime_tags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@router.post(
    "/upload-ohlcv",
    response_model=OHLCVUploadResponse,
    responses={
        200: {"description": "OHLCV parsed and regime computed"},
        400: {"description": "Invalid file format or content"},
        413: {"description": "File too large"},
        429: {"description": "Rate limit exceeded"},
    },
    summary="Upload OHLCV file for regime computation",
    description=f"""
Upload OHLCV data file (JSON/CSV) to compute regime tags for recommendation queries.

**Limits:**
- Max file size: {MAX_OHLCV_FILE_SIZE_MB}MB
- Max rows: {MAX_OHLCV_ROWS:,}
- Rate limit: {RATE_LIMIT_UPLOAD_PER_MIN}/min per IP

**Expected format (JSON):**
```json
[{{"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000}}, ...]
```

**Expected format (CSV):**
```
open,high,low,close,volume
100,105,99,103,1000
...
```
""",
)
async def upload_ohlcv(
    ohlcv_file: UploadFile = File(..., description="OHLCV data file (CSV/JSON)"),
    _rate: None = Depends(get_rate_limiter().check("upload_ohlcv", RATE_LIMIT_UPLOAD_PER_MIN)),
) -> OHLCVUploadResponse:
    """Upload OHLCV file and compute regime tags."""
    request_id = str(uuid.uuid4())
    warnings: list[str] = []

    logger.info(
        "OHLCV upload request",
        request_id=request_id,
        filename=ohlcv_file.filename,
        content_type=ohlcv_file.content_type,
    )

    # Check content-length header if available
    if ohlcv_file.size and ohlcv_file.size > MAX_OHLCV_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": f"File too large. Maximum size is {MAX_OHLCV_FILE_SIZE_MB}MB",
                "code": "FILE_TOO_LARGE",
                "max_bytes": MAX_OHLCV_FILE_SIZE_BYTES,
            },
        )

    # Stream-read with limit to prevent memory exhaustion
    try:
        chunks = []
        total_size = 0
        async for chunk in ohlcv_file:
            total_size += len(chunk)
            if total_size > MAX_OHLCV_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={
                        "error": f"File too large. Maximum size is {MAX_OHLCV_FILE_SIZE_MB}MB",
                        "code": "FILE_TOO_LARGE",
                        "max_bytes": MAX_OHLCV_FILE_SIZE_BYTES,
                    },
                )
            chunks.append(chunk)
        content = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Failed to read file: {str(e)}", "code": "READ_ERROR"},
        )

    # Parse OHLCV data
    try:
        ohlcv_data = _parse_ohlcv_file(content, ohlcv_file.filename or "data.json")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Failed to parse OHLCV: {str(e)}", "code": "PARSE_ERROR"},
        )

    # Check row count
    row_count = len(ohlcv_data)
    if row_count > MAX_OHLCV_ROWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"Too many rows ({row_count:,}). Maximum is {MAX_OHLCV_ROWS:,}",
                "code": "TOO_MANY_ROWS",
                "max_rows": MAX_OHLCV_ROWS,
            },
        )

    if row_count < 10:
        warnings.append("very_few_bars")

    # Compute regime
    regime_tags: list[str] = []
    try:
        from app.services.kb.regime import compute_regime_from_ohlcv

        regime = compute_regime_from_ohlcv(ohlcv_data, source="upload")
        regime_tags = regime.regime_tags
        warnings.extend(regime.warnings)
    except Exception as e:
        logger.warning(
            "Failed to compute regime from OHLCV",
            request_id=request_id,
            error=str(e),
        )
        warnings.append("regime_computation_failed")

    logger.info(
        "OHLCV upload complete",
        request_id=request_id,
        row_count=row_count,
        regime_tags=regime_tags,
        warning_count=len(warnings),
    )

    return OHLCVUploadResponse(
        request_id=request_id,
        row_count=row_count,
        regime_tags=regime_tags,
        warnings=warnings,
    )


# =============================================================================
# Helpers
# =============================================================================


def _parse_ohlcv_file(content: bytes, filename: str) -> list[dict]:
    """Parse OHLCV data from uploaded file."""
    import json
    import csv
    from io import StringIO

    text = content.decode("utf-8")

    if filename.endswith(".json"):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        else:
            raise ValueError("JSON must be array or object with 'data' key")

    elif filename.endswith(".csv"):
        reader = csv.DictReader(StringIO(text))
        rows = list(reader)

        # Convert numeric fields
        for row in rows:
            for key in ["open", "high", "low", "close", "volume"]:
                if key in row:
                    row[key] = float(row[key])

        return rows

    else:
        raise ValueError(f"Unsupported file format: {filename}. Use .json or .csv")
