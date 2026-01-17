"""Source schemas: Pine Script, YouTube match, generic sources."""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.common import PineIngestStatus


# =============================================================================
# Pine Script Types
# =============================================================================

PineScriptType = Literal["indicator", "strategy", "library"]
PineVersionType = Literal["4", "5", "6"]
PineDocStatus = Literal["active", "superseded", "deleted"]
PineLintSeverity = Literal["error", "warning", "info"]


# =============================================================================
# Pine Script Ingest
# =============================================================================


class PineIngestRequest(BaseModel):
    """Request for Pine Script registry ingestion."""

    workspace_id: UUID = Field(..., description="Target workspace ID")
    registry_path: str = Field(
        ..., description="Server path to pine_registry.json (must be within DATA_DIR)"
    )
    lint_path: Optional[str] = Field(
        default=None,
        description="Server path to pine_lint_report.json (auto-derived if None)",
    )
    source_root: Optional[str] = Field(
        default=None,
        description="Directory containing .pine source files (required if include_source=True)",
    )
    include_source: bool = Field(
        default=True, description="Include source code in embedded content"
    )
    max_source_lines: int = Field(
        default=100, description="Maximum lines of source to include per script"
    )
    skip_lint_errors: bool = Field(
        default=False,
        description="Skip scripts with lint errors (runs inline lint if no report)",
    )
    update_existing: bool = Field(
        default=False,
        description="Update existing documents if sha256 changed (False=skip changed scripts)",
    )
    dry_run: bool = Field(
        default=False, description="Validate only, do not write to database"
    )


class PineIngestResponse(BaseModel):
    """Response for Pine Script registry ingestion."""

    status: PineIngestStatus = Field(..., description="Overall ingest status")
    scripts_processed: int = Field(default=0, description="Total scripts in registry")
    scripts_indexed: int = Field(
        default=0, description="Scripts newly indexed or updated"
    )
    scripts_already_indexed: int = Field(
        default=0, description="Scripts already indexed (unchanged sha256)"
    )
    scripts_skipped: int = Field(
        default=0, description="Scripts skipped (lint errors, changed but not updated)"
    )
    scripts_failed: int = Field(default=0, description="Scripts that failed to ingest")
    chunks_added: int = Field(default=0, description="Total chunks written to database")
    errors: list[str] = Field(
        default_factory=list, description="Error messages for failed scripts"
    )
    ingest_run_id: Optional[str] = Field(
        default=None, description="Run ID for log correlation"
    )


# =============================================================================
# Pine Script Read APIs
# =============================================================================


class PineLintSummary(BaseModel):
    """Lint summary for a Pine script."""

    errors: int = Field(default=0, description="Number of lint errors")
    warnings: int = Field(default=0, description="Number of lint warnings")
    info: int = Field(default=0, description="Number of lint info messages")


class PineLintFinding(BaseModel):
    """Single lint finding for a Pine script."""

    code: str = Field(..., description="Lint rule code (e.g., E001, W002)")
    severity: PineLintSeverity = Field(..., description="Finding severity")
    message: str = Field(..., description="Finding message")
    line: Optional[int] = Field(None, description="Line number (1-indexed)")
    column: Optional[int] = Field(None, description="Column number (1-indexed)")


class PineInputSchema(BaseModel):
    """Input parameter definition for a Pine script."""

    name: str = Field(..., description="Input parameter name")
    type: str = Field(..., description="Input type (int, float, bool, etc.)")
    default: Optional[str] = Field(None, description="Default value")
    tooltip: Optional[str] = Field(None, description="Tooltip description")


class PineImportSchema(BaseModel):
    """Import reference for a Pine script."""

    path: str = Field(..., description="Import path")
    alias: Optional[str] = Field(None, description="Import alias")


class PineChunkItem(BaseModel):
    """Single chunk of a Pine script document."""

    id: UUID = Field(..., description="Chunk ID")
    index: int = Field(..., description="Chunk index (0-based)")
    content: str = Field(..., description="Chunk content")
    token_count: int = Field(..., description="Token count")
    symbols: list[str] = Field(default_factory=list, description="Ticker symbols")


class PineScriptListItem(BaseModel):
    """Summary of a Pine script for list endpoints."""

    id: UUID = Field(..., description="Document ID")
    canonical_url: str = Field(..., description="Canonical URL (pine://source/path)")
    rel_path: str = Field(..., description="Relative file path")
    title: str = Field(..., description="Script title (fallback to basename)")
    script_type: Optional[PineScriptType] = Field(
        None, description="Script declaration type"
    )
    pine_version: Optional[PineVersionType] = Field(None, description="Pine version")
    symbols: list[str] = Field(default_factory=list, description="Ticker symbols")
    lint_summary: PineLintSummary = Field(
        default_factory=PineLintSummary, description="Lint summary"
    )
    lint_available: bool = Field(
        default=False, description="Whether lint data is available"
    )
    sha256: str = Field(..., description="Content hash")
    chunk_count: int = Field(default=0, description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: PineDocStatus = Field(default="active", description="Document status")


class PineScriptListResponse(BaseModel):
    """Response for Pine script list endpoint."""

    items: list[PineScriptListItem] = Field(
        default_factory=list, description="List of Pine scripts"
    )
    total: int = Field(default=0, description="Total matching scripts")
    limit: int = Field(..., description="Requested limit")
    offset: int = Field(default=0, description="Requested offset")
    has_more: bool = Field(default=False, description="More results available")
    next_offset: Optional[int] = Field(None, description="Next offset for pagination")


class PineScriptDetailResponse(BaseModel):
    """Response for Pine script detail endpoint."""

    id: UUID = Field(..., description="Document ID")
    canonical_url: str = Field(..., description="Canonical URL")
    rel_path: str = Field(..., description="Relative file path")
    title: str = Field(..., description="Script title")
    script_type: Optional[PineScriptType] = Field(None, description="Script type")
    pine_version: Optional[PineVersionType] = Field(None, description="Pine version")
    symbols: list[str] = Field(default_factory=list, description="Ticker symbols")
    lint_summary: PineLintSummary = Field(
        default_factory=PineLintSummary, description="Lint summary"
    )
    lint_available: bool = Field(default=False, description="Lint data available")
    lint_findings: Optional[list[PineLintFinding]] = Field(
        None, description="Lint findings (if include_lint_findings=true)"
    )
    sha256: str = Field(..., description="Content hash")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    status: PineDocStatus = Field(default="active", description="Document status")

    # Structured metadata
    inputs: Optional[list[PineInputSchema]] = Field(
        None, description="Input parameters"
    )
    imports: Optional[list[PineImportSchema]] = Field(None, description="Imports")
    features: Optional[dict[str, bool]] = Field(None, description="Feature flags")

    # Chunks
    chunk_total: int = Field(default=0, description="Total chunks in document")
    chunks: Optional[list[PineChunkItem]] = Field(
        None, description="Chunks (if include_chunks=true)"
    )
    chunk_has_more: Optional[bool] = Field(
        None, description="More chunks available for pagination"
    )
    chunk_next_offset: Optional[int] = Field(
        None, description="Next chunk offset for pagination"
    )


class PineScriptLookupResponse(BaseModel):
    """Response for Pine script lookup by rel_path."""

    id: UUID = Field(..., description="Document ID")
    canonical_url: str = Field(..., description="Canonical URL")
    rel_path: str = Field(..., description="Relative file path")
    title: str = Field(..., description="Script title")
    status: PineDocStatus = Field(default="active", description="Document status")
    script_type: Optional[PineScriptType] = Field(None, description="Script type")
    pine_version: Optional[PineVersionType] = Field(None, description="Pine version")


class PineRebuildAndIngestRequest(BaseModel):
    """Request for rebuild-and-ingest endpoint."""

    workspace_id: UUID = Field(..., description="Target workspace ID")
    scripts_root: str = Field(..., description="Root directory containing .pine files")
    output_dir: Optional[str] = Field(
        None,
        description="Output directory for registry files (defaults to scripts_root)",
    )

    # Ingest flags (passthrough)
    include_source: bool = Field(
        default=True, description="Include source code preview"
    )
    max_source_lines: int = Field(
        default=100, ge=0, le=500, description="Max source lines"
    )
    skip_lint_errors: bool = Field(
        default=False, description="Skip scripts with lint errors"
    )
    update_existing: bool = Field(default=False, description="Update if sha256 changed")

    dry_run: bool = Field(default=False, description="Validate only, no DB changes")


class PineBuildStats(BaseModel):
    """Statistics from registry build phase."""

    files_scanned: int = Field(default=0, description="Files found")
    files_parsed: int = Field(default=0, description="Files successfully parsed")
    parse_errors: int = Field(default=0, description="Files with parse errors")
    lint_errors: int = Field(
        default=0, description="Total lint errors across all files"
    )
    lint_warnings: int = Field(
        default=0, description="Total lint warnings across all files"
    )
    registry_path: Optional[str] = Field(
        default=None, description="Path to generated registry"
    )
    lint_report_path: Optional[str] = Field(
        default=None, description="Path to generated lint report"
    )


class PineRebuildAndIngestResponse(BaseModel):
    """Response for rebuild-and-ingest endpoint."""

    status: PineIngestStatus = Field(..., description="Overall status")

    # Build phase
    build: PineBuildStats = Field(
        default_factory=PineBuildStats, description="Build statistics"
    )

    # Ingest phase (mirrors PineIngestResponse)
    scripts_processed: int = Field(default=0, description="Total scripts in registry")
    scripts_indexed: int = Field(default=0, description="Scripts newly indexed")
    scripts_already_indexed: int = Field(default=0, description="Scripts unchanged")
    scripts_skipped: int = Field(
        default=0, description="Scripts skipped (lint errors, etc.)"
    )
    scripts_failed: int = Field(default=0, description="Scripts that failed to ingest")
    chunks_added: int = Field(default=0, description="New chunks created")

    errors: list[str] = Field(default_factory=list, description="Error messages")
    ingest_run_id: Optional[str] = Field(None, description="Run ID for log correlation")


class PineMatchResult(BaseModel):
    """A single match result from Pine script search."""

    id: UUID = Field(..., description="Document ID")
    rel_path: str = Field(..., description="Relative file path")
    title: str = Field(..., description="Script title")
    script_type: Optional[PineScriptType] = Field(None, description="Script type")
    pine_version: Optional[PineVersionType] = Field(None, description="Pine version")
    score: float = Field(..., description="Match score (0-1)")
    match_reasons: list[str] = Field(
        default_factory=list, description="Why this matched"
    )
    snippet: Optional[str] = Field(None, description="Relevant text snippet")
    inputs_preview: list[str] = Field(
        default_factory=list, description="First few input names"
    )
    lint_ok: bool = Field(default=True, description="No lint errors")


class PineMatchResponse(BaseModel):
    """Response for Pine script match endpoint."""

    results: list[PineMatchResult] = Field(
        default_factory=list, description="Matched scripts"
    )
    total_searched: int = Field(default=0, description="Total scripts searched")
    query: str = Field(..., description="Original query")
    filters_applied: dict = Field(
        default_factory=dict, description="Filters that were applied"
    )


# =============================================================================
# YouTube to Pine Match
# =============================================================================


class IngestRequestHint(BaseModel):
    """Hint for ingesting the video."""

    workspace_id: UUID = Field(..., description="Workspace to ingest into")
    url: str = Field(..., description="YouTube URL")


class PineMatchRankedResult(PineMatchResult):
    """Pine match result with reranking score breakdown."""

    base_score: float = Field(..., description="Original match score")
    boost: float = Field(..., description="Intent-based boost")
    final_score: float = Field(..., description="Final score after boost")


class YouTubeMatchPineRequest(BaseModel):
    """Request for YouTube to Pine script matching."""

    workspace_id: UUID = Field(..., description="Workspace ID")
    url: str = Field(..., description="YouTube video URL")
    symbols: Optional[list[str]] = Field(None, description="Override extracted symbols")
    script_type: Optional[Literal["strategy", "indicator"]] = Field(
        None, description="Filter by script type"
    )
    lint_ok: bool = Field(True, description="Filter to clean scripts")
    top_k: int = Field(10, ge=1, le=50, description="Max results")
    force_transient: bool = Field(False, description="Bypass KB, fetch live")


class CoverageResponse(BaseModel):
    """Coverage gap assessment for match results."""

    weak: bool = Field(..., description="True if coverage gap detected")
    best_score: Optional[float] = Field(None, description="Best match score")
    avg_top_k_score: Optional[float] = Field(None, description="Average top-k score")
    num_above_threshold: int = Field(..., description="Results above threshold")
    threshold: float = Field(..., description="Score threshold used")
    reason_codes: list[str] = Field(
        default_factory=list, description="Gap reason codes"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Actionable suggestions"
    )
    # Strategy registry integration
    intent_signature: Optional[str] = Field(
        None, description="SHA256 hash for candidates endpoint"
    )
    candidate_strategies: Optional[list] = Field(
        None, description="Strategies with tag overlap (when weak=true)"
    )


class YouTubeMatchPineResponse(BaseModel):
    """Response for YouTube to Pine script matching."""

    # Source metadata
    video_id: str = Field(..., description="YouTube video ID")
    title: Optional[str] = Field(None, description="Video title")
    channel: Optional[str] = Field(None, description="Channel name")

    # KB status
    in_knowledge_base: bool = Field(..., description="Whether video is in KB")
    source_id: Optional[UUID] = Field(None, description="Document ID if in KB")
    transcript_source: Literal["kb", "transient"] = Field(
        ..., description="Where transcript came from"
    )
    transcript_chars_used: int = Field(..., description="Characters of transcript used")

    # Extraction
    match_intent: dict = Field(..., description="Extracted trading intent")
    extraction_method: Literal["rule_based", "llm"] = Field(
        "rule_based", description="Extraction method used"
    )

    # Match results
    results: list[PineMatchRankedResult] = Field(
        default_factory=list, description="Matched scripts"
    )
    total_searched: int = Field(..., description="Total scripts searched")
    query_used: str = Field(..., description="Query string used for matching")
    filters_applied: dict = Field(..., description="Filters applied")

    # Coverage assessment
    coverage: CoverageResponse = Field(..., description="Coverage gap assessment")

    # Next actions
    ingest_available: bool = Field(..., description="Whether ingest is available")
    ingest_request_hint: Optional[IngestRequestHint] = Field(
        None, description="Hint for ingesting video"
    )


# =============================================================================
# Generic Sources Listing
# =============================================================================


class SourceListItem(BaseModel):
    """Single item in sources list response."""

    id: UUID = Field(..., description="Document ID")
    source_type: str = Field(..., description="Source type (youtube, pdf, pine_script)")
    canonical_url: str = Field(..., description="Canonical URL")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Author name")
    channel: Optional[str] = Field(None, description="Channel name (for YouTube)")
    video_id: Optional[str] = Field(None, description="YouTube video ID")
    status: str = Field(..., description="Document status")
    chunk_count: int = Field(..., description="Number of chunks")
    version: int = Field(default=1, description="Document version")
    created_at: datetime = Field(..., description="Created timestamp")
    updated_at: datetime = Field(..., description="Updated timestamp")
    last_indexed_at: Optional[datetime] = Field(
        None, description="Last indexed timestamp"
    )


class SourceListResponse(BaseModel):
    """Response for sources list endpoint."""

    items: list[SourceListItem] = Field(
        default_factory=list, description="Source items"
    )
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(..., description="More items available")
    next_offset: Optional[int] = Field(None, description="Next page offset")


class SourceChunkItem(BaseModel):
    """Chunk item in source detail response."""

    id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Chunk position")
    token_count: Optional[int] = Field(None, description="Token count")
    time_start_secs: Optional[int] = Field(None, description="Start time (video)")
    time_end_secs: Optional[int] = Field(None, description="End time (video)")
    page_start: Optional[int] = Field(None, description="Start page (PDF)")
    page_end: Optional[int] = Field(None, description="End page (PDF)")
    symbols: list[str] = Field(default_factory=list, description="Ticker symbols")
    entities: list[str] = Field(default_factory=list, description="Named entities")
    topics: list[str] = Field(default_factory=list, description="Topics")


class SourceHealthCheck(BaseModel):
    """Single health check result."""

    name: str = Field(..., description="Check name")
    passed: bool = Field(..., description="Whether check passed")
    message: str = Field(..., description="Check result message")


class SourceHealth(BaseModel):
    """Health status for a source."""

    status: Literal["ok", "degraded", "failed"] = Field(
        ..., description="Health status"
    )
    chunk_count_ok: bool = Field(..., description="Chunks exist")
    embeddings_ok: bool = Field(..., description="Embeddings match chunks")
    checks: list[SourceHealthCheck] = Field(
        default_factory=list, description="Individual check results"
    )


class SourceDetailResponse(BaseModel):
    """Response for source detail endpoint."""

    # Core fields
    id: UUID = Field(..., description="Document ID")
    source_type: str = Field(..., description="Source type")
    canonical_url: str = Field(..., description="Canonical URL")
    source_url: Optional[str] = Field(None, description="Original source URL")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Author name")
    channel: Optional[str] = Field(None, description="Channel name (YouTube)")
    video_id: Optional[str] = Field(None, description="YouTube video ID")
    playlist_id: Optional[str] = Field(None, description="YouTube playlist ID")
    published_at: Optional[datetime] = Field(None, description="Publication date")
    language: Optional[str] = Field(None, description="Content language")
    duration_secs: Optional[int] = Field(None, description="Duration (video)")
    content_hash: str = Field(..., description="SHA-256 content hash")

    # Status
    status: str = Field(..., description="Document status")
    version: int = Field(default=1, description="Document version")
    chunk_count: int = Field(..., description="Number of chunks")

    # Timestamps
    created_at: datetime = Field(..., description="Created timestamp")
    updated_at: datetime = Field(..., description="Updated timestamp")
    last_indexed_at: Optional[datetime] = Field(
        None, description="Last indexed timestamp"
    )

    # Health (optional)
    health: Optional[SourceHealth] = Field(None, description="Source health status")

    # Optional extras
    pine_metadata: Optional[dict] = Field(None, description="Pine script metadata")
    chunks: Optional[list[dict]] = Field(None, description="Chunk content")
    chunks_total: int = Field(default=0, description="Total chunks")
    chunks_has_more: bool = Field(default=False, description="More chunks available")
