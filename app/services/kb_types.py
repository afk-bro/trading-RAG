"""Type definitions for knowledge base extraction/verification pipeline."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ===========================================
# Enums matching database constraints
# ===========================================

class EntityType(str, Enum):
    """Types of knowledge entities."""
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


class ClaimType(str, Enum):
    """Types of knowledge claims."""
    DEFINITION = "definition"
    RULE = "rule"
    ASSUMPTION = "assumption"
    WARNING = "warning"
    PARAMETER = "parameter"
    EQUATION = "equation"
    OBSERVATION = "observation"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


class RelationType(str, Enum):
    """Types of entity relationships."""
    USES = "uses"
    REQUIRES = "requires"
    DERIVED_FROM = "derived_from"
    VARIANT_OF = "variant_of"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    MENTIONS = "mentions"
    COMPONENT_OF = "component_of"
    INPUT_TO = "input_to"
    OUTPUT_OF = "output_of"
    PRECEDES = "precedes"
    FOLLOWS = "follows"


class VerificationStatus(str, Enum):
    """Verification status for claims."""
    PENDING = "pending"
    VERIFIED = "verified"
    WEAK = "weak"
    REJECTED = "rejected"


# ===========================================
# Pass 1: Extraction output models
# ===========================================

class EvidencePointer(BaseModel):
    """Reference to supporting evidence in a chunk."""
    chunk_index: int = Field(..., description="Index of chunk in provided context (0-based)")
    quote: str = Field(..., description="Short verbatim quote from chunk (max 200 chars)")
    relevance: float = Field(default=1.0, ge=0, le=1, description="Relevance score 0-1")


class ExtractedEntity(BaseModel):
    """An entity extracted from context."""
    type: EntityType
    name: str = Field(..., description="Primary name of the entity")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(None, description="Grounded description if found")
    evidence: list[EvidencePointer] = Field(default_factory=list)


class ExtractedClaim(BaseModel):
    """A claim extracted from context."""
    claim_type: ClaimType
    text: str = Field(..., description="Atomic truth statement")
    entity_name: Optional[str] = Field(None, description="Related entity name if applicable")
    entity_type: Optional[EntityType] = Field(None, description="Related entity type")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Initial confidence")
    evidence: list[EvidencePointer] = Field(..., min_length=1, description="Must have at least one evidence")


class ExtractedRelation(BaseModel):
    """A relationship between entities extracted from context."""
    from_entity: str = Field(..., description="Source entity name")
    from_type: EntityType
    relation: RelationType
    to_entity: str = Field(..., description="Target entity name")
    to_type: EntityType
    evidence: list[EvidencePointer] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Complete output of Pass 1 extraction."""
    entities: list[ExtractedEntity] = Field(default_factory=list)
    claims: list[ExtractedClaim] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


# ===========================================
# Pass 2: Verification output models
# ===========================================

class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim_index: int = Field(..., description="Index of claim in extraction result (0-based)")
    status: VerificationStatus
    confidence: float = Field(..., ge=0, le=1, description="Adjusted confidence after verification")
    reason: str = Field(..., description="Brief explanation of verdict")
    corrected_text: Optional[str] = Field(None, description="Rewritten claim if needed for accuracy")


class VerificationResult(BaseModel):
    """Complete output of Pass 2 verification."""
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


# ===========================================
# Pipeline result models
# ===========================================

class PersistenceStats(BaseModel):
    """Statistics from persisting to database."""
    entities_created: int = 0
    entities_updated: int = 0
    claims_created: int = 0
    evidence_created: int = 0
    relations_created: int = 0


class PipelineResult(BaseModel):
    """Complete result of the extraction → verification → persist pipeline."""
    extraction: ExtractionResult
    verification: VerificationResult
    persistence: PersistenceStats
    synthesized_answer: Optional[str] = None

    # For debugging/transparency
    verified_claims_count: int = 0
    weak_claims_count: int = 0
    rejected_claims_count: int = 0
