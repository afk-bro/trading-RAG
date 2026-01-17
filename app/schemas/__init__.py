"""Pydantic models for request/response validation.

This package re-exports all schemas for backwards compatibility.
Imports like `from app.schemas import X` continue to work.
"""

# ===========================================
# Common: Enums, Health, Error, Config
# ===========================================
from app.schemas.common import (
    # Enums
    SourceType,
    DocumentStatus,
    VectorStatus,
    QueryMode,
    SymbolsMode,
    JobStatus,
    RerankState,
    PineIngestStatus,
    # Health & Error
    DependencyHealth,
    HealthResponse,
    ReadinessResponse,
    ErrorResponse,
    # Workspace Config
    CrossEncoderConfig,
    LLMRerankConfig,
    RerankConfig,
    NeighborConfig,
    RetrievalConfig,
    ChunkingConfig,
    WorkspaceConfig,
)

# ===========================================
# Ingest: Documents, Chunks, YouTube, PDF, Reembed
# ===========================================
from app.schemas.ingest import (
    SourceInfo,
    DocumentMetadata,
    ChunkInput,
    IngestRequest,
    IngestResponse,
    YouTubeIngestRequest,
    YouTubeIngestResponse,
    PDFBackendType,
    PDFConfig,
    PDFIngestRequest,
    PDFIngestResponse,
    ReembedRequest,
    ReembedResponse,
    JobResponse,
)

# ===========================================
# Query: Filters, Requests, Responses, Compare
# ===========================================
from app.schemas.query import (
    QueryFilters,
    QueryRequest,
    ChunkResultDebug,
    ChunkResult,
    KnowledgeExtractionStats,
    QueryMeta,
    QueryResponse,
    QueryCompareRequest,
    CompareMetrics,
    QueryCompareResponse,
)

# ===========================================
# KB: Entities, Claims, Evidence, Answers
# ===========================================
from app.schemas.kb import (
    KBEntityType,
    KBClaimType,
    KBClaimStatus,
    KBEntityStats,
    KBEntityItem,
    KBEntityListResponse,
    KBEntityDetailResponse,
    KBEvidenceItem,
    KBClaimItem,
    KBClaimListResponse,
    KBClaimDetailResponse,
    KBAnswerClaimRef,
    KBAnswerResponse,
)

# ===========================================
# Trading: Intents, Policy, Events, Execution, Strategies
# ===========================================
from app.schemas.trading import (
    # Strategy Spec
    StrategySpecStatus,
    StrategySpecResponse,
    StrategySpecRefreshRequest,
    StrategyCompileResponse,
    StrategySpecStatusUpdate,
    # Trade Intent & Policy
    IntentAction,
    TradeIntent,
    PolicyReason,
    PolicyDecision,
    PositionState,
    CurrentState,
    # Trade Events
    TradeEventType,
    TradeEvent,
    TradeEventListResponse,
    IntentEvaluateRequest,
    IntentEvaluateResponse,
    # Paper Execution
    OrderSide,
    OrderStatus,
    ExecutionMode,
    PaperOrder,
    PaperPosition,
    PaperState,
    ExecutionRequest,
    ExecutionResult,
    ReconciliationResult,
    # Strategy Registry
    StrategyEngine,
    StrategyStatus,
    StrategyReviewStatus,
    StrategyRiskLevel,
    BacktestSummaryStatus,
    StrategyTags,
    BacktestSummary,
    StrategySourceRef,
    StrategyCreateRequest,
    StrategyUpdateRequest,
    StrategyListItem,
    StrategyListResponse,
    StrategyDetailResponse,
    CandidateStrategy,
    StrategyCard,
)

# ===========================================
# Sources: Pine Script, YouTube Match, Generic Sources
# ===========================================
from app.schemas.sources import (
    # Pine Script Types (Literal aliases)
    PineScriptType,
    PineVersionType,
    PineDocStatus,
    PineLintSeverity,
    # Pine Ingest
    PineIngestRequest,
    PineIngestResponse,
    # Pine Read APIs
    PineLintSummary,
    PineLintFinding,
    PineInputSchema,
    PineImportSchema,
    PineChunkItem,
    PineScriptListItem,
    PineScriptListResponse,
    PineScriptDetailResponse,
    PineScriptLookupResponse,
    PineRebuildAndIngestRequest,
    PineBuildStats,
    PineRebuildAndIngestResponse,
    PineMatchResult,
    PineMatchResponse,
    # YouTube Match
    IngestRequestHint,
    PineMatchRankedResult,
    YouTubeMatchPineRequest,
    CoverageResponse,
    YouTubeMatchPineResponse,
    # Generic Sources
    SourceListItem,
    SourceListResponse,
    SourceChunkItem,
    SourceHealthCheck,
    SourceHealth,
    SourceDetailResponse,
)

# ===========================================
# Explicit __all__ for star imports
# ===========================================
__all__ = [
    # Common
    "SourceType",
    "DocumentStatus",
    "VectorStatus",
    "QueryMode",
    "SymbolsMode",
    "JobStatus",
    "RerankState",
    "PineIngestStatus",
    "DependencyHealth",
    "HealthResponse",
    "ReadinessResponse",
    "ErrorResponse",
    "CrossEncoderConfig",
    "LLMRerankConfig",
    "RerankConfig",
    "NeighborConfig",
    "RetrievalConfig",
    "ChunkingConfig",
    "WorkspaceConfig",
    # Ingest
    "SourceInfo",
    "DocumentMetadata",
    "ChunkInput",
    "IngestRequest",
    "IngestResponse",
    "YouTubeIngestRequest",
    "YouTubeIngestResponse",
    "PDFBackendType",
    "PDFConfig",
    "PDFIngestRequest",
    "PDFIngestResponse",
    "ReembedRequest",
    "ReembedResponse",
    "JobResponse",
    # Query
    "QueryFilters",
    "QueryRequest",
    "ChunkResultDebug",
    "ChunkResult",
    "KnowledgeExtractionStats",
    "QueryMeta",
    "QueryResponse",
    "QueryCompareRequest",
    "CompareMetrics",
    "QueryCompareResponse",
    # KB
    "KBEntityType",
    "KBClaimType",
    "KBClaimStatus",
    "KBEntityStats",
    "KBEntityItem",
    "KBEntityListResponse",
    "KBEntityDetailResponse",
    "KBEvidenceItem",
    "KBClaimItem",
    "KBClaimListResponse",
    "KBClaimDetailResponse",
    "KBAnswerClaimRef",
    "KBAnswerResponse",
    # Trading
    "StrategySpecStatus",
    "StrategySpecResponse",
    "StrategySpecRefreshRequest",
    "StrategyCompileResponse",
    "StrategySpecStatusUpdate",
    "IntentAction",
    "TradeIntent",
    "PolicyReason",
    "PolicyDecision",
    "PositionState",
    "CurrentState",
    "TradeEventType",
    "TradeEvent",
    "TradeEventListResponse",
    "IntentEvaluateRequest",
    "IntentEvaluateResponse",
    "OrderSide",
    "OrderStatus",
    "ExecutionMode",
    "PaperOrder",
    "PaperPosition",
    "PaperState",
    "ExecutionRequest",
    "ExecutionResult",
    "ReconciliationResult",
    "StrategyEngine",
    "StrategyStatus",
    "StrategyReviewStatus",
    "StrategyRiskLevel",
    "BacktestSummaryStatus",
    "StrategyTags",
    "BacktestSummary",
    "StrategySourceRef",
    "StrategyCreateRequest",
    "StrategyUpdateRequest",
    "StrategyListItem",
    "StrategyListResponse",
    "StrategyDetailResponse",
    "CandidateStrategy",
    "StrategyCard",
    # Sources
    "PineScriptType",
    "PineVersionType",
    "PineDocStatus",
    "PineLintSeverity",
    "PineIngestRequest",
    "PineIngestResponse",
    "PineLintSummary",
    "PineLintFinding",
    "PineInputSchema",
    "PineImportSchema",
    "PineChunkItem",
    "PineScriptListItem",
    "PineScriptListResponse",
    "PineScriptDetailResponse",
    "PineScriptLookupResponse",
    "PineRebuildAndIngestRequest",
    "PineBuildStats",
    "PineRebuildAndIngestResponse",
    "PineMatchResult",
    "PineMatchResponse",
    "IngestRequestHint",
    "PineMatchRankedResult",
    "YouTubeMatchPineRequest",
    "CoverageResponse",
    "YouTubeMatchPineResponse",
    "SourceListItem",
    "SourceListResponse",
    "SourceChunkItem",
    "SourceHealthCheck",
    "SourceHealth",
    "SourceDetailResponse",
]
