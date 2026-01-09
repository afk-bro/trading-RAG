"""
Trading Knowledge Base API endpoints.

Provides recommendation and ingestion endpoints for the Trading KB:
- POST /kb/trials/recommend - Get parameter recommendations
- POST /kb/trials/ingest - Ingest trials from tune runs (admin)
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Optional
from uuid import UUID

import anyio
import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field, model_validator

from app.config import Settings, get_settings
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
    objective_type: str = Field(..., description="Objective function type")

    # Dataset source (exactly one required)
    dataset_id: Optional[str] = Field(None, description="Reference dataset ID for regime computation")
    # Note: ohlcv_file handled separately via Form/File

    # Optional regime override
    regime_tags: Optional[list[str]] = Field(None, max_length=20, description="Override regime tags for query")

    # Filter overrides (bounded by workspace defaults)
    require_oos: Optional[bool] = Field(None, description="Require OOS metrics")
    max_overfit_gap: Optional[float] = Field(None, ge=0, le=1, description="Max overfit gap (0-1)")
    min_trades: Optional[int] = Field(None, ge=1, le=1000, description="Minimum trade count")
    max_drawdown: Optional[float] = Field(None, ge=0, le=1, description="Max drawdown fraction (0-1)")

    # Retrieval knobs (bounded)
    retrieve_k: int = Field(DEFAULT_RETRIEVE_K, ge=1, le=MAX_RETRIEVE_K, description="Max candidates to retrieve")
    rerank_keep: int = Field(DEFAULT_RERANK_KEEP, ge=1, le=MAX_RERANK_KEEP, description="Top M after reranking")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Top K for aggregation")

    # Debug options
    include_candidates: bool = Field(False, description="Include top candidates in response (debug)")

    model_config = {"extra": "forbid"}


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
        422: {"description": "Validation error (bounds exceeded)"},
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
    request: RecommendRequest,
    mode: RecommendMode = Query(RecommendMode.FULL, description="Recommendation mode"),
    settings: Settings = Depends(get_settings),
) -> RecommendResponse:
    """Generate parameter recommendations."""
    request_id = str(uuid.uuid4())

    logger.info(
        "Recommend request",
        request_id=request_id,
        workspace_id=str(request.workspace_id),
        strategy=request.strategy_name,
        objective=request.objective_type,
        mode=mode.value,
    )

    # Validate strategy and objective
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
    )

    # Execute with timeouts
    try:
        with anyio.fail_after(RECOMMEND_TIMEOUT_S):
            result = await recommender.recommend(internal_req)
    except TimeoutError:
        logger.error(
            "Recommendation timed out",
            request_id=request_id,
            workspace_id=str(request.workspace_id),
            strategy=request.strategy_name,
            mode=mode.value,
            timeout_s=RECOMMEND_TIMEOUT_S,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation request timed out after {RECOMMEND_TIMEOUT_S}s",
        )
    except Exception as e:
        logger.error(
            "Recommendation failed",
            request_id=request_id,
            workspace_id=str(request.workspace_id),
            strategy=request.strategy_name,
            mode=mode.value,
            error=str(e),
        )
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

    logger.info(
        "Recommend complete",
        request_id=request_id,
        workspace_id=str(request.workspace_id),
        strategy=request.strategy_name,
        objective=request.objective_type,
        mode=mode.value,
        status=result.status,
        confidence=result.confidence,
        count_used=result.count_used,
        used_relaxed=result.used_relaxed_filters,
        used_metadata_fallback=result.used_metadata_fallback,
    )

    return response


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        200: {"description": "Ingestion completed"},
        403: {"description": "Admin access required"},
        503: {"description": "Service unavailable"},
    },
    summary="Ingest trials from tune runs (admin)",
    description="""
Ingest completed tune runs into the Trading KB vector store.

**Admin-only endpoint.**

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
    },
    summary="Upload OHLCV file for regime computation",
    description=f"""
Upload OHLCV data file (JSON/CSV) to compute regime tags for recommendation queries.

**Limits:**
- Max file size: {MAX_OHLCV_FILE_SIZE_MB}MB
- Max rows: {MAX_OHLCV_ROWS:,}

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
