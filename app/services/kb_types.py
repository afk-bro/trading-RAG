"""Type definitions for knowledge base extraction/verification pipeline."""

import hashlib
import re
from enum import Enum
from typing import Optional
from uuid import UUID

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
    claims_skipped_duplicate: int = 0  # Skipped due to fingerprint match
    claims_skipped_invalid: int = 0  # Skipped due to evidence validation failure
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

    # Parse error tracking (allows user to see failures and retry)
    parse_errors: list[str] = Field(default_factory=list)
    had_extraction_error: bool = False
    had_verification_error: bool = False


# ===========================================
# Utility functions for deduplication and validation
# ===========================================

MAX_QUOTE_LENGTH = 300  # Max characters for evidence quotes


def normalize_text(text: str) -> str:
    """Normalize text for fingerprinting (lowercase, collapse whitespace)."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_claim_fingerprint(
    claim_text: str,
    claim_type: ClaimType | str,
    entity_name: str | None,
    workspace_id: UUID | str,
) -> str:
    """
    Compute deterministic fingerprint for claim deduplication.

    Args:
        claim_text: The claim text
        claim_type: Type of claim
        entity_name: Related entity name (or None)
        workspace_id: Workspace ID

    Returns:
        SHA-256 hex digest (64 chars)
    """
    # Normalize inputs
    normalized_text = normalize_text(claim_text)
    type_str = claim_type.value if isinstance(claim_type, ClaimType) else str(claim_type)
    entity_str = normalize_text(entity_name) if entity_name else ""
    workspace_str = str(workspace_id)

    # Create fingerprint
    fingerprint_input = f"{workspace_str}|{type_str}|{entity_str}|{normalized_text}"
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()


class EvidenceValidationResult(BaseModel):
    """Result of evidence validation."""
    is_valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sanitized_quote: str | None = None


def validate_evidence(
    evidence: EvidencePointer,
    available_chunk_indices: set[int],
    max_quote_length: int = MAX_QUOTE_LENGTH,
) -> EvidenceValidationResult:
    """
    Validate a single evidence pointer for integrity.

    Checks:
    - chunk_index exists in available chunks
    - quote is not empty
    - quote length is within limit

    Args:
        evidence: Evidence pointer to validate
        available_chunk_indices: Set of valid chunk indices
        max_quote_length: Maximum allowed quote length

    Returns:
        EvidenceValidationResult with validation status and sanitized quote
    """
    result = EvidenceValidationResult()

    # Check chunk index exists
    if evidence.chunk_index not in available_chunk_indices:
        result.is_valid = False
        result.errors.append(f"chunk_index {evidence.chunk_index} not in available chunks")
        return result

    # Check quote is not empty
    if not evidence.quote or not evidence.quote.strip():
        result.is_valid = False
        result.errors.append("quote is empty")
        return result

    # Sanitize and truncate quote
    sanitized = evidence.quote.strip()
    if len(sanitized) > max_quote_length:
        result.warnings.append(f"quote truncated from {len(sanitized)} to {max_quote_length} chars")
        sanitized = sanitized[:max_quote_length]

    result.sanitized_quote = sanitized
    return result


def validate_claim_evidence(
    claim: ExtractedClaim,
    available_chunk_indices: set[int],
) -> tuple[bool, list[EvidencePointer], list[str]]:
    """
    Validate all evidence for a claim.

    Returns:
        Tuple of (is_valid, validated_evidence, errors)
        - is_valid: True if claim has at least one valid evidence
        - validated_evidence: List of validated/sanitized evidence pointers
        - errors: List of validation errors
    """
    errors = []
    validated_evidence = []

    for ev in claim.evidence:
        result = validate_evidence(ev, available_chunk_indices)
        if result.is_valid:
            # Create sanitized evidence pointer
            validated_evidence.append(EvidencePointer(
                chunk_index=ev.chunk_index,
                quote=result.sanitized_quote or ev.quote[:MAX_QUOTE_LENGTH],
                relevance=ev.relevance,
            ))
        else:
            errors.extend(result.errors)

    is_valid = len(validated_evidence) >= 1
    if not is_valid:
        errors.append("no valid evidence pointers")

    return is_valid, validated_evidence, errors
