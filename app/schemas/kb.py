"""Knowledge Base (KB) schemas: entities, claims, evidence, answers."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ===========================================
# KB Enums
# ===========================================


class KBEntityType(str, Enum):
    """Entity types for KB filtering."""

    CONCEPT = "concept"
    INDICATOR = "indicator"
    STRATEGY = "strategy"
    EQUATION = "equation"
    TEST = "test"
    METRIC = "metric"
    ASSET = "asset"
    PATTERN = "pattern"
    PARAMETER = "parameter"
    OTHER = "other"


class KBClaimType(str, Enum):
    """Claim types for KB filtering."""

    DEFINITION = "definition"
    RULE = "rule"
    ASSUMPTION = "assumption"
    WARNING = "warning"
    PARAMETER = "parameter"
    EQUATION = "equation"
    OBSERVATION = "observation"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


class KBClaimStatus(str, Enum):
    """Claim verification status for KB filtering."""

    PENDING = "pending"
    VERIFIED = "verified"
    WEAK = "weak"
    REJECTED = "rejected"


# ===========================================
# KB Entity Models
# ===========================================


class KBEntityStats(BaseModel):
    """Statistics for a single entity."""

    verified_claims: int = Field(default=0, description="Verified claim count")
    weak_claims: int = Field(default=0, description="Weak claim count")
    total_claims: int = Field(default=0, description="Total claim count")
    relations_count: int = Field(default=0, description="Number of relations")


class KBEntityItem(BaseModel):
    """Single entity in KB list response."""

    id: UUID = Field(..., description="Entity ID")
    type: KBEntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(None, description="Entity description")
    verified_claim_count: Optional[int] = Field(
        None, description="Number of verified claims"
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class KBEntityListResponse(BaseModel):
    """Response for GET /kb/entities."""

    items: list[KBEntityItem] = Field(..., description="Entity list")
    total: int = Field(..., description="Total matching entities")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class KBEntityDetailResponse(BaseModel):
    """Response for GET /kb/entities/{entity_id}."""

    id: UUID = Field(..., description="Entity ID")
    type: KBEntityType = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(None, description="Entity description")
    stats: KBEntityStats = Field(..., description="Entity statistics")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# ===========================================
# KB Claim Models
# ===========================================


class KBEvidenceItem(BaseModel):
    """Evidence for a claim."""

    id: UUID = Field(..., description="Evidence ID")
    doc_id: UUID = Field(..., description="Source document ID")
    chunk_id: UUID = Field(..., description="Source chunk ID")
    quote: str = Field(..., description="Evidence quote")
    relevance_score: float = Field(..., description="Relevance score")
    doc_title: Optional[str] = Field(None, description="Document title")


class KBClaimItem(BaseModel):
    """Single claim in KB list response."""

    id: UUID = Field(..., description="Claim ID")
    claim_type: KBClaimType = Field(..., description="Claim type")
    text: str = Field(..., description="Claim text")
    status: KBClaimStatus = Field(..., description="Verification status")
    confidence: float = Field(..., description="Confidence score")
    entity_id: Optional[UUID] = Field(None, description="Linked entity ID")
    entity_name: Optional[str] = Field(None, description="Linked entity name")
    entity_type: Optional[KBEntityType] = Field(None, description="Linked entity type")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class KBClaimListResponse(BaseModel):
    """Response for GET /kb/claims."""

    items: list[KBClaimItem] = Field(..., description="Claim list")
    total: int = Field(..., description="Total matching claims")
    limit: int = Field(..., description="Page size")
    offset: int = Field(..., description="Page offset")


class KBClaimDetailResponse(BaseModel):
    """Response for GET /kb/claims/{claim_id}."""

    id: UUID = Field(..., description="Claim ID")
    claim_type: KBClaimType = Field(..., description="Claim type")
    text: str = Field(..., description="Claim text")
    status: KBClaimStatus = Field(..., description="Verification status")
    confidence: float = Field(..., description="Confidence score")
    entity_id: Optional[UUID] = Field(None, description="Linked entity ID")
    entity_name: Optional[str] = Field(None, description="Linked entity name")
    entity_type: Optional[KBEntityType] = Field(None, description="Linked entity type")
    evidence: list[KBEvidenceItem] = Field(
        default_factory=list, description="Supporting evidence"
    )
    extraction_model: Optional[str] = Field(None, description="Extraction model")
    verification_model: Optional[str] = Field(None, description="Verification model")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# ===========================================
# KB Answer Models
# ===========================================


class KBAnswerClaimRef(BaseModel):
    """Claim reference in kb_answer response."""

    id: str = Field(..., description="Short reference ID (e.g., C12)")
    claim_id: UUID = Field(..., description="Full claim UUID")
    confidence: float = Field(..., description="Claim confidence")


class KBAnswerResponse(BaseModel):
    """Response for mode=kb_answer queries."""

    mode: str = Field(default="kb_answer", description="Query mode")
    llm_enabled: bool = Field(..., description="Whether LLM was used")
    answer: Optional[str] = Field(None, description="Synthesized answer")
    supported: list[str] = Field(
        default_factory=list, description="Supported statements with claim refs"
    )
    not_specified: list[str] = Field(
        default_factory=list, description="Unanswered aspects"
    )
    claims_used: list[KBAnswerClaimRef] = Field(
        default_factory=list, description="Claims used in answer"
    )
    fallback_used: bool = Field(
        default=False, description="Whether chunk RAG fallback was used"
    )
    fallback_reason: Optional[str] = Field(
        None, description="Reason for fallback if used"
    )
