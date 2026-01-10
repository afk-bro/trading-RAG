"""Pydantic schemas for KB trials admin endpoints."""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class BulkStatusRequest(BaseModel):
    """Request for bulk status change operations."""

    source_type: Literal["tune_run", "test_variant"]
    source_ids: list[UUID] = Field(..., min_length=1, max_length=100)
    reason: Optional[str] = Field(None, max_length=500)
    trigger_ingest: bool = Field(
        default=True,
        description="Whether to trigger ingestion after promotion",
    )


class StatusChangeResult(BaseModel):
    """Result for a single status change."""

    source_id: UUID
    group_id: Optional[UUID] = None
    status: str
    error: Optional[str] = None


class BulkStatusResponse(BaseModel):
    """Response for bulk status change operations."""

    updated: int
    skipped: int
    ingested: int
    errors: int
    results: list[StatusChangeResult]


class TrialPreviewItem(BaseModel):
    """Preview item for promotion consideration."""

    source_type: str
    source_id: UUID
    group_id: Optional[UUID] = None
    experiment_type: str
    strategy_name: Optional[str] = None
    kb_status: str
    sharpe_oos: Optional[float] = None
    return_frac_oos: Optional[float] = None
    max_dd_frac_oos: Optional[float] = None
    n_trades_oos: Optional[int] = None
    passes_auto_gates: bool = False
    can_promote: bool = False
    is_eligible: bool = False
    ineligibility_reasons: list[str] = Field(default_factory=list)
    has_regime_is: bool = False
    has_regime_oos: bool = False
    regime_schema_version: Optional[str] = None
    created_at: Optional[datetime] = None


class PromotionPreviewSummary(BaseModel):
    """Summary of promotion preview results."""

    would_promote: int = 0
    already_promoted: int = 0
    would_skip: int = 0
    missing_regime: int = 0


class PromotionPreviewResponse(BaseModel):
    """Response for promotion preview endpoint."""

    summary: PromotionPreviewSummary
    pagination: dict
    trials: list[TrialPreviewItem]


class MarkCandidateRequest(BaseModel):
    """Request to mark trials as candidates."""

    source_type: Literal["tune_run", "test_variant"]
    source_ids: list[UUID] = Field(..., min_length=1, max_length=100)
