"""Knowledge Base endpoints for browsing entities and claims."""

import json
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import Settings, get_settings
from app.schemas import (
    KBEntityType,
    KBClaimType,
    KBClaimStatus,
    KBEntityItem,
    KBEntityListResponse,
    KBEntityDetailResponse,
    KBEntityStats,
    KBClaimItem,
    KBClaimListResponse,
    KBClaimDetailResponse,
    KBEvidenceItem,
)
from app.services.kb_types import EntityType, ClaimType

router = APIRouter(prefix="/kb", tags=["knowledge-base"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_kb_repo():
    """Get KnowledgeBaseRepository instance."""
    from app.repositories.kb import KnowledgeBaseRepository

    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return KnowledgeBaseRepository(_db_pool)


def _parse_aliases(aliases) -> list[str]:
    """Parse aliases from DB (can be JSON string or list)."""
    if aliases is None:
        return []
    if isinstance(aliases, list):
        return aliases
    if isinstance(aliases, str):
        try:
            return json.loads(aliases)
        except json.JSONDecodeError:
            return []
    return []


# ===========================================
# Entity Endpoints
# ===========================================


@router.get(
    "/entities",
    response_model=KBEntityListResponse,
    responses={
        200: {"description": "Entity list retrieved"},
        503: {"description": "Database unavailable"},
    },
)
async def list_entities(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    q: Optional[str] = Query(None, description="Search query (name/aliases)"),
    type: Optional[KBEntityType] = Query(None, description="Filter by entity type"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    include_counts: bool = Query(False, description="Include verified claim counts"),
) -> KBEntityListResponse:
    """
    List knowledge base entities with optional search and filtering.

    Use cases:
    - Browse all entities in a workspace
    - Search by name or alias
    - Filter by entity type (concept, indicator, strategy, etc.)
    - Include claim counts to see which entities have most knowledge
    """
    logger.info(
        "Listing entities",
        workspace_id=str(workspace_id),
        q=q,
        type=type,
        limit=limit,
        offset=offset,
        include_counts=include_counts,
    )

    kb_repo = _get_kb_repo()

    # Convert schema type to internal type
    entity_type = EntityType(type.value) if type else None

    entities, total = await kb_repo.list_entities(
        workspace_id=workspace_id,
        q=q,
        entity_type=entity_type,
        limit=limit,
        offset=offset,
        include_counts=include_counts,
    )

    items = [
        KBEntityItem(
            id=e["id"],
            type=KBEntityType(e["type"]),
            name=e["name"],
            aliases=_parse_aliases(e.get("aliases")),
            description=e.get("description"),
            verified_claim_count=e.get("verified_claim_count"),
            created_at=e.get("created_at"),
        )
        for e in entities
    ]

    return KBEntityListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/entities/{entity_id}",
    response_model=KBEntityDetailResponse,
    responses={
        200: {"description": "Entity details retrieved"},
        404: {"description": "Entity not found"},
        503: {"description": "Database unavailable"},
    },
)
async def get_entity(
    entity_id: UUID,
) -> KBEntityDetailResponse:
    """
    Get detailed information about a single entity.

    Includes summary statistics:
    - Number of verified/weak/total claims
    - Number of relations to other entities
    """
    logger.info("Getting entity", entity_id=str(entity_id))

    kb_repo = _get_kb_repo()
    entity = await kb_repo.get_entity_by_id(entity_id)

    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found",
        )

    return KBEntityDetailResponse(
        id=entity["id"],
        type=KBEntityType(entity["type"]),
        name=entity["name"],
        aliases=_parse_aliases(entity.get("aliases")),
        description=entity.get("description"),
        stats=KBEntityStats(
            verified_claims=entity.get("verified_claims", 0),
            weak_claims=entity.get("weak_claims", 0),
            total_claims=entity.get("total_claims", 0),
            relations_count=entity.get("relations_count", 0),
        ),
        created_at=entity.get("created_at"),
        updated_at=entity.get("updated_at"),
    )


# ===========================================
# Claim Endpoints
# ===========================================


@router.get(
    "/claims",
    response_model=KBClaimListResponse,
    responses={
        200: {"description": "Claim list retrieved"},
        503: {"description": "Database unavailable"},
    },
)
async def list_claims(
    workspace_id: UUID = Query(..., description="Workspace ID"),
    q: Optional[str] = Query(None, description="Search query (claim text)"),
    status: KBClaimStatus = Query(KBClaimStatus.VERIFIED, description="Filter by verification status"),
    claim_type: Optional[KBClaimType] = Query(None, description="Filter by claim type"),
    entity_id: Optional[UUID] = Query(None, description="Filter by entity"),
    source_id: Optional[UUID] = Query(None, description="Filter by source document"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> KBClaimListResponse:
    """
    List knowledge base claims with optional search and filtering.

    Default is status=verified (truth-first).

    Use cases:
    - Browse verified claims in a workspace
    - Search claim text
    - Filter by claim type (definition, rule, warning, etc.)
    - Get all claims for a specific entity
    - Get all claims from a specific source document
    """
    logger.info(
        "Listing claims",
        workspace_id=str(workspace_id),
        q=q,
        status=status,
        claim_type=claim_type,
        entity_id=str(entity_id) if entity_id else None,
        source_id=str(source_id) if source_id else None,
        limit=limit,
        offset=offset,
    )

    kb_repo = _get_kb_repo()

    # Convert schema types to internal types
    internal_claim_type = ClaimType(claim_type.value) if claim_type else None

    claims, total = await kb_repo.list_claims(
        workspace_id=workspace_id,
        q=q,
        status=status.value,
        claim_type=internal_claim_type,
        entity_id=entity_id,
        source_id=source_id,
        limit=limit,
        offset=offset,
    )

    items = [
        KBClaimItem(
            id=c["id"],
            claim_type=KBClaimType(c["claim_type"]),
            text=c["text"],
            status=KBClaimStatus(c["status"]),
            confidence=c["confidence"],
            entity_id=c.get("entity_id"),
            entity_name=c.get("entity_name"),
            entity_type=KBEntityType(c["entity_type"]) if c.get("entity_type") else None,
            created_at=c.get("created_at"),
        )
        for c in claims
    ]

    return KBClaimListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/claims/{claim_id}",
    response_model=KBClaimDetailResponse,
    responses={
        200: {"description": "Claim details retrieved"},
        404: {"description": "Claim not found"},
        503: {"description": "Database unavailable"},
    },
)
async def get_claim(
    claim_id: UUID,
) -> KBClaimDetailResponse:
    """
    Get detailed information about a single claim.

    Includes:
    - Full claim text and metadata
    - Linked entity information
    - All supporting evidence with quotes
    - Extraction/verification model info
    """
    logger.info("Getting claim", claim_id=str(claim_id))

    kb_repo = _get_kb_repo()
    claim = await kb_repo.get_claim_by_id(claim_id, include_evidence=True)

    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim {claim_id} not found",
        )

    evidence = [
        KBEvidenceItem(
            id=e["id"],
            doc_id=e["doc_id"],
            chunk_id=e["chunk_id"],
            quote=e["quote"],
            relevance_score=e["relevance_score"],
            doc_title=e.get("doc_title"),
        )
        for e in claim.get("evidence", [])
    ]

    return KBClaimDetailResponse(
        id=claim["id"],
        claim_type=KBClaimType(claim["claim_type"]),
        text=claim["text"],
        status=KBClaimStatus(claim["status"]),
        confidence=claim["confidence"],
        entity_id=claim.get("entity_id"),
        entity_name=claim.get("entity_name"),
        entity_type=KBEntityType(claim["entity_type"]) if claim.get("entity_type") else None,
        evidence=evidence,
        extraction_model=claim.get("extraction_model"),
        verification_model=claim.get("verification_model"),
        created_at=claim.get("created_at"),
        updated_at=claim.get("updated_at"),
    )
