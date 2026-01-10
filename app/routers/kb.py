"""Knowledge Base endpoints for browsing entities and claims."""

import json
from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, Query, status

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
    StrategySpecResponse,
    StrategySpecStatus,
    StrategyCompileResponse,
    StrategySpecStatusUpdate,
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
    status: KBClaimStatus = Query(
        KBClaimStatus.VERIFIED, description="Filter by verification status"
    ),
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
            entity_type=(
                KBEntityType(c["entity_type"]) if c.get("entity_type") else None
            ),
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
        entity_type=(
            KBEntityType(claim["entity_type"]) if claim.get("entity_type") else None
        ),
        evidence=evidence,
        extraction_model=claim.get("extraction_model"),
        verification_model=claim.get("verification_model"),
        created_at=claim.get("created_at"),
        updated_at=claim.get("updated_at"),
    )


# ============================================================================
# Strategy Spec Endpoints
# ============================================================================


@router.get(
    "/strategies/{entity_id}/spec",
    response_model=StrategySpecResponse,
    summary="Get strategy specification",
    description="Get the persisted strategy specification for a strategy entity.",
)
async def get_strategy_spec(
    entity_id: UUID,
) -> StrategySpecResponse:
    """Get the persisted strategy spec for an entity."""
    logger.info("Getting strategy spec", entity_id=str(entity_id))

    kb_repo = _get_kb_repo()
    spec = await kb_repo.get_strategy_spec(entity_id)

    if not spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy spec found for entity {entity_id}. Use POST /kb/strategies/{entity_id}/spec/refresh to create one.",  # noqa: E501
        )

    # Parse JSON fields
    spec_json = spec.get("spec_json", {})
    if isinstance(spec_json, str):
        spec_json = json.loads(spec_json)

    derived_claim_ids = spec.get("derived_from_claim_ids", [])
    if isinstance(derived_claim_ids, str):
        derived_claim_ids = json.loads(derived_claim_ids)

    return StrategySpecResponse(
        id=spec["id"],
        strategy_entity_id=spec["strategy_entity_id"],
        strategy_name=spec.get("strategy_name", "Unknown"),
        spec_json=spec_json,
        status=StrategySpecStatus(spec["status"]),
        version=spec.get("version", 1),
        derived_from_claim_ids=derived_claim_ids,
        created_at=spec.get("created_at"),
        updated_at=spec.get("updated_at"),
        approved_at=spec.get("approved_at"),
        approved_by=spec.get("approved_by"),
    )


@router.post(
    "/strategies/{entity_id}/spec/refresh",
    response_model=StrategySpecResponse,
    summary="Refresh strategy specification",
    description="Recompute strategy spec from verified claims and persist it.",
)
async def refresh_strategy_spec(
    entity_id: UUID,
) -> StrategySpecResponse:
    """Recompute and persist strategy spec from verified claims."""
    logger.info("Refreshing strategy spec", entity_id=str(entity_id))

    kb_repo = _get_kb_repo()

    # Get entity to find workspace_id
    entity = await kb_repo.get_entity_by_id(entity_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity {entity_id} not found",
        )

    if entity.get("type") != "strategy":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Entity {entity_id} is not a strategy type (type={entity.get('type')})",
        )

    try:
        spec = await kb_repo.refresh_strategy_spec(entity_id, entity["workspace_id"])
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Parse JSON fields
    spec_json = spec.get("spec_json", {})
    if isinstance(spec_json, str):
        spec_json = json.loads(spec_json)

    derived_claim_ids = spec.get("derived_from_claim_ids", [])
    if isinstance(derived_claim_ids, str):
        derived_claim_ids = json.loads(derived_claim_ids)

    logger.info(
        "Strategy spec refreshed",
        entity_id=str(entity_id),
        version=spec.get("version"),
        claim_count=len(derived_claim_ids),
    )

    return StrategySpecResponse(
        id=spec["id"],
        strategy_entity_id=spec["strategy_entity_id"],
        strategy_name=entity["name"],
        spec_json=spec_json,
        status=StrategySpecStatus(spec["status"]),
        version=spec.get("version", 1),
        derived_from_claim_ids=derived_claim_ids,
        created_at=spec.get("created_at"),
        updated_at=spec.get("updated_at"),
        approved_at=spec.get("approved_at"),
        approved_by=spec.get("approved_by"),
    )


@router.post(
    "/strategies/{entity_id}/compile",
    response_model=StrategyCompileResponse,
    summary="Compile strategy specification",
    description="Compile a strategy spec into actionable outputs: param schema, backtest config, pseudocode.",  # noqa: E501
    responses={
        200: {"description": "Compilation successful"},
        404: {"description": "No strategy spec found"},
        409: {"description": "Spec not approved (use allow_draft=true to override)"},
    },
)
async def compile_strategy_spec(
    entity_id: UUID,
    allow_draft: bool = Query(
        False, description="Allow compiling draft specs (dev/admin only)"
    ),
    force: bool = Query(False, description="Force recompilation (ignore cache)"),
) -> StrategyCompileResponse:
    """Compile strategy spec into actionable outputs.

    By default, only approved specs can be compiled.
    Use allow_draft=true to compile draft specs (for development/testing).
    Use force=true to recompile even if cached results exist.
    """
    logger.info(
        "Compiling strategy spec",
        entity_id=str(entity_id),
        allow_draft=allow_draft,
        force=force,
    )

    kb_repo = _get_kb_repo()

    # First check if spec exists and is approved
    spec = await kb_repo.get_strategy_spec(entity_id)
    if not spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy spec found for entity {entity_id}. Use POST /kb/strategies/{entity_id}/spec/refresh to create one first.",  # noqa: E501
        )

    # Enforce approval requirement unless allow_draft
    if spec.get("status") != "approved" and not allow_draft:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Strategy spec is '{spec.get('status')}', not approved. Use PATCH to approve first, or pass ?allow_draft=true for dev/testing.",  # noqa: E501
        )

    result = await kb_repo.compile_strategy_spec(entity_id, force=force)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy spec found for entity {entity_id}. Use POST /kb/strategies/{entity_id}/spec/refresh to create one first.",  # noqa: E501
        )

    logger.info(
        "Strategy spec compiled",
        entity_id=str(entity_id),
        spec_version=result.get("spec_version"),
        param_count=len(result.get("param_schema", {}).get("properties", {})),
    )

    return StrategyCompileResponse(
        spec_id=result["spec_id"],
        spec_version=result["spec_version"],
        spec_status=StrategySpecStatus(result["spec_status"]),
        param_schema=result["param_schema"],
        backtest_config=result["backtest_config"],
        pseudocode=result["pseudocode"],
        citations=result["citations"],
    )


@router.patch(
    "/strategies/{entity_id}/spec",
    response_model=StrategySpecResponse,
    summary="Update strategy spec status",
    description="Update the approval status of a strategy spec (draft, approved, deprecated).",
)
async def update_strategy_spec_status(
    entity_id: UUID,
    update: StrategySpecStatusUpdate,
) -> StrategySpecResponse:
    """Update strategy spec status for governance."""
    logger.info(
        "Updating strategy spec status",
        entity_id=str(entity_id),
        new_status=update.status.value,
    )

    kb_repo = _get_kb_repo()
    spec = await kb_repo.update_strategy_spec_status(
        entity_id,
        update.status.value,
        approved_by=update.approved_by,
    )

    if not spec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No strategy spec found for entity {entity_id}",
        )

    # Parse JSON fields
    spec_json = spec.get("spec_json", {})
    if isinstance(spec_json, str):
        spec_json = json.loads(spec_json)

    derived_claim_ids = spec.get("derived_from_claim_ids", [])
    if isinstance(derived_claim_ids, str):
        derived_claim_ids = json.loads(derived_claim_ids)

    # Get entity name
    entity = await kb_repo.get_entity_by_id(entity_id)
    strategy_name = entity["name"] if entity else "Unknown"

    logger.info(
        "Strategy spec status updated",
        entity_id=str(entity_id),
        status=spec["status"],
    )

    return StrategySpecResponse(
        id=spec["id"],
        strategy_entity_id=spec["strategy_entity_id"],
        strategy_name=strategy_name,
        spec_json=spec_json,
        status=StrategySpecStatus(spec["status"]),
        version=spec.get("version", 1),
        derived_from_claim_ids=derived_claim_ids,
        created_at=spec.get("created_at"),
        updated_at=spec.get("updated_at"),
        approved_at=spec.get("approved_at"),
        approved_by=spec.get("approved_by"),
    )
