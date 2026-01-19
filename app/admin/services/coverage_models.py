"""Pydantic models for coverage gap inspection."""

from enum import Enum
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas import StrategyCard


class CoverageStatusEnum(str, Enum):
    """Coverage status for triage workflow."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class WeakCoverageItem(BaseModel):
    """Single weak coverage run for cockpit display."""

    run_id: UUID = Field(..., description="Match run ID")
    created_at: str = Field(..., description="ISO timestamp")
    intent_signature: str = Field(..., description="SHA256 intent hash")
    script_type: Optional[str] = Field(None, description="Filtered script type")
    weak_reason_codes: list[str] = Field(
        default_factory=list, description="Coverage gap reasons"
    )
    best_score: Optional[float] = Field(None, description="Best match score")
    num_above_threshold: int = Field(..., description="Results above threshold")
    candidate_strategy_ids: list[UUID] = Field(
        default_factory=list, description="Strategy IDs with tag overlap"
    )
    candidate_scores: Optional[dict] = Field(
        None, description="Detailed scores per strategy"
    )
    query_preview: str = Field(..., description="First ~120 chars of query")
    source_ref: Optional[str] = Field(None, description="Source reference for display")
    coverage_status: CoverageStatusEnum = Field(
        default=CoverageStatusEnum.OPEN, description="Triage status"
    )
    priority_score: float = Field(
        default=0.0, description="Priority score for sorting (higher = more urgent)"
    )


class CoverageStatusUpdateRequest(BaseModel):
    """Request for PATCH /admin/coverage/weak/{run_id}."""

    status: CoverageStatusEnum = Field(..., description="New status")
    note: Optional[str] = Field(None, max_length=1000, description="Resolution note")


class CoverageStatusUpdateResponse(BaseModel):
    """Response for PATCH /admin/coverage/weak/{run_id}."""

    run_id: UUID = Field(..., description="Match run ID")
    coverage_status: CoverageStatusEnum = Field(..., description="New status")
    acknowledged_at: Optional[str] = Field(None, description="When acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged")
    resolved_at: Optional[str] = Field(None, description="When resolved")
    resolved_by: Optional[str] = Field(None, description="Who resolved")
    resolution_note: Optional[str] = Field(None, description="Resolution note")


class ExplainStrategyRequest(BaseModel):
    """Request for POST /admin/coverage/explain."""

    run_id: UUID = Field(..., description="Match run ID")
    strategy_id: UUID = Field(..., description="Strategy ID to explain")
    verbosity: Literal["short", "detailed"] = Field(
        "short", description="Explanation verbosity: short (2-4 sentences) or detailed"
    )


class ExplainStrategyResponse(BaseModel):
    """Response for POST /admin/coverage/explain."""

    run_id: UUID = Field(..., description="Match run ID")
    strategy_id: UUID = Field(..., description="Strategy ID")
    strategy_name: str = Field(..., description="Strategy name")
    explanation: str = Field(..., description="LLM-generated explanation")
    confidence_qualifier: str = Field(..., description="Deterministic confidence line")
    model: str = Field(..., description="LLM model used")
    provider: str = Field(..., description="LLM provider used")
    verbosity: Literal["short", "detailed"] = Field(..., description="Verbosity level")
    latency_ms: Optional[float] = Field(None, description="Generation latency")
    cache_hit: bool = Field(False, description="True if returned from cache")


class WeakCoverageResponse(BaseModel):
    """Response for weak coverage list endpoint."""

    items: list[WeakCoverageItem] = Field(default_factory=list)
    count: int = Field(..., description="Number of items returned")
    strategy_cards_by_id: Optional[dict[str, StrategyCard]] = Field(
        None,
        description="Hydrated strategy cards keyed by UUID (include_candidate_cards=true)",
    )
    missing_strategy_ids: list[UUID] = Field(
        default_factory=list,
        description="Strategy IDs referenced but not found (deleted/archived)",
    )


class SeedCoverageResponse(BaseModel):
    """Response for POST /admin/coverage/seed."""

    status: str = Field(..., description="success or error")
    workspace_id: UUID = Field(..., description="Workspace used/created")
    strategies_created: int = Field(..., description="Number of strategies seeded")
    match_runs_created: int = Field(..., description="Number of match_runs seeded")
    message: str = Field(..., description="Summary message")
