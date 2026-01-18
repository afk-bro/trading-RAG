"""Test schema exports to prevent drift after package split.

This test ensures backwards compatibility: `from app.schemas import X`
must continue to work for all canonical schema names.
"""


class TestSchemaExports:
    """Verify all canonical schemas are importable from app.schemas."""

    def test_common_enums_exported(self):
        """Core enums should be importable."""
        from app.schemas import (  # noqa: F401
            DocumentStatus,
            JobStatus,
            PineIngestStatus,
            QueryMode,
            RerankState,
            SourceType,
            SymbolsMode,
            VectorStatus,
        )

        # Spot check enum values exist
        assert SourceType.YOUTUBE == "youtube"
        assert QueryMode.RETRIEVE == "retrieve"

    def test_health_models_exported(self):
        """Health/error models should be importable."""
        from app.schemas import (  # noqa: F401
            DependencyHealth,
            ErrorResponse,
            HealthResponse,
            ReadinessResponse,
        )

        # Verify they're Pydantic models
        assert hasattr(DependencyHealth, "model_fields")
        assert hasattr(ErrorResponse, "model_fields")

    def test_workspace_config_exported(self):
        """Workspace configuration schemas should be importable."""
        from app.schemas import (  # noqa: F401
            ChunkingConfig,
            CrossEncoderConfig,
            LLMRerankConfig,
            NeighborConfig,
            RerankConfig,
            RetrievalConfig,
            WorkspaceConfig,
        )

        # Can construct with defaults
        config = WorkspaceConfig()
        assert config.chunking.size == 512

    def test_ingest_schemas_exported(self):
        """Ingest-related schemas should be importable."""
        from app.schemas import (  # noqa: F401
            ChunkInput,
            DocumentMetadata,
            IngestRequest,
            IngestResponse,
            JobResponse,
            PDFBackendType,
            PDFConfig,
            PDFIngestRequest,
            PDFIngestResponse,
            ReembedRequest,
            ReembedResponse,
            SourceInfo,
            YouTubeIngestRequest,
            YouTubeIngestResponse,
        )

        assert hasattr(IngestRequest, "model_fields")
        assert PDFBackendType.PYMUPDF == "pymupdf"

    def test_query_schemas_exported(self):
        """Query-related schemas should be importable."""
        from app.schemas import (  # noqa: F401
            ChunkResult,
            ChunkResultDebug,
            CompareMetrics,
            KnowledgeExtractionStats,
            QueryCompareRequest,
            QueryCompareResponse,
            QueryFilters,
            QueryMeta,
            QueryRequest,
            QueryResponse,
        )

        assert hasattr(QueryRequest, "model_fields")
        assert hasattr(QueryResponse, "model_fields")

    def test_kb_schemas_exported(self):
        """Knowledge base schemas should be importable."""
        from app.schemas import (  # noqa: F401
            KBAnswerClaimRef,
            KBAnswerResponse,
            KBClaimDetailResponse,
            KBClaimItem,
            KBClaimListResponse,
            KBClaimStatus,
            KBClaimType,
            KBEntityDetailResponse,
            KBEntityItem,
            KBEntityListResponse,
            KBEntityStats,
            KBEntityType,
            KBEvidenceItem,
        )

        assert hasattr(KBAnswerResponse, "model_fields")

    def test_trading_schemas_exported(self):
        """Trading-related schemas should be importable."""
        from app.schemas import (  # noqa: F401
            # Strategy Spec
            BacktestSummary,
            BacktestSummaryStatus,
            CandidateStrategy,
            CurrentState,
            ExecutionMode,
            ExecutionRequest,
            ExecutionResult,
            IntentAction,
            IntentEvaluateRequest,
            IntentEvaluateResponse,
            OrderSide,
            OrderStatus,
            PaperOrder,
            PaperPosition,
            PaperState,
            PolicyDecision,
            PolicyReason,
            PositionState,
            ReconciliationResult,
            StrategyCard,
            StrategyCompileResponse,
            StrategyCreateRequest,
            StrategyDetailResponse,
            StrategyEngine,
            StrategyListItem,
            StrategyListResponse,
            StrategyReviewStatus,
            StrategyRiskLevel,
            StrategySourceRef,
            StrategySpecRefreshRequest,
            StrategySpecResponse,
            StrategySpecStatus,
            StrategySpecStatusUpdate,
            StrategyStatus,
            StrategyTags,
            StrategyUpdateRequest,
            TradeEvent,
            TradeEventListResponse,
            TradeEventType,
            TradeIntent,
        )

        assert hasattr(TradeEvent, "model_fields")
        assert IntentAction.OPEN_LONG == "open_long"

    def test_sources_schemas_exported(self):
        """Source-related schemas should be importable."""
        from app.schemas import (  # noqa: F401
            # Pine Script
            CoverageResponse,
            IngestRequestHint,
            PineBuildStats,
            PineChunkItem,
            PineDocStatus,
            PineImportSchema,
            PineIngestRequest,
            PineIngestResponse,
            PineInputSchema,
            PineLintFinding,
            PineLintSeverity,
            PineLintSummary,
            PineMatchRankedResult,
            PineMatchResponse,
            PineMatchResult,
            PineRebuildAndIngestRequest,
            PineRebuildAndIngestResponse,
            PineScriptDetailResponse,
            PineScriptListItem,
            PineScriptListResponse,
            PineScriptLookupResponse,
            PineScriptType,
            PineVersionType,
            SourceChunkItem,
            SourceDetailResponse,
            SourceHealth,
            SourceHealthCheck,
            SourceListItem,
            SourceListResponse,
            YouTubeMatchPineRequest,
            YouTubeMatchPineResponse,
        )

        assert hasattr(PineIngestRequest, "model_fields")

    def test_all_exports_match_declared(self):
        """__all__ should match actual module exports."""
        import app.schemas as schemas

        # Get declared exports
        declared = set(schemas.__all__)

        # Verify each declared name is actually importable
        missing = []
        for name in declared:
            if not hasattr(schemas, name):
                missing.append(name)

        assert not missing, f"__all__ declares missing exports: {missing}"

    def test_no_undeclared_public_exports(self):
        """Public names should be in __all__ to prevent accidental exports."""
        import app.schemas as schemas

        declared = set(schemas.__all__)

        # Get all public names (not starting with _)
        public_names = {name for name in dir(schemas) if not name.startswith("_")}

        # Filter out modules and common builtins
        ignored = {"annotations", "datetime", "UUID", "Optional", "Field", "BaseModel"}
        public_names -= ignored

        undeclared = public_names - declared
        # Allow submodules to be present but not in __all__
        submodules = {"common", "ingest", "query", "kb", "trading", "sources"}
        undeclared -= submodules

        assert not undeclared, f"Public names not in __all__: {undeclared}"
