"""Knowledge base repository for truth store persistence."""

import json
from typing import Optional
from uuid import UUID

import structlog

from app.services.kb_types import (
    EntityType,
    ClaimType,
    RelationType,
    VerificationStatus,
    ExtractedEntity,
    ExtractedClaim,
    ExtractedRelation,
    ClaimVerdict,
    PersistenceStats,
)

logger = structlog.get_logger(__name__)


class KnowledgeBaseRepository:
    """Repository for knowledge base CRUD operations."""

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    # ===========================================
    # Entity Operations
    # ===========================================

    async def upsert_entity(
        self,
        workspace_id: UUID,
        entity: ExtractedEntity,
    ) -> tuple[UUID, bool]:
        """
        Upsert an entity (insert or update if exists).

        Uses ON CONFLICT on (workspace_id, type, lower(name)) unique index.

        Args:
            workspace_id: Workspace ID
            entity: Extracted entity to persist

        Returns:
            Tuple of (entity_id, created) - created is True if new, False if updated
        """
        query = """
            INSERT INTO kb_entities (
                workspace_id, type, name, aliases, description
            ) VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (workspace_id, type, lower(name))
            DO UPDATE SET
                aliases = CASE
                    WHEN kb_entities.aliases IS NULL THEN EXCLUDED.aliases
                    ELSE kb_entities.aliases || EXCLUDED.aliases
                END,
                description = COALESCE(EXCLUDED.description, kb_entities.description),
                updated_at = NOW()
            RETURNING id, (xmax = 0) AS inserted
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                entity.type.value,
                entity.name,
                json.dumps(entity.aliases),
                entity.description,
            )

            entity_id = row["id"]
            created = row["inserted"]

            logger.debug(
                "Upserted entity",
                entity_id=str(entity_id),
                name=entity.name,
                type=entity.type.value,
                created=created,
            )

            return entity_id, created

    async def get_entity_by_name(
        self,
        workspace_id: UUID,
        entity_type: EntityType,
        name: str,
    ) -> Optional[dict]:
        """Get entity by workspace, type, and name (case-insensitive)."""
        query = """
            SELECT * FROM kb_entities
            WHERE workspace_id = $1
              AND type = $2
              AND lower(name) = lower($3)
        """

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                query,
                workspace_id,
                entity_type.value,
                name,
            )

    async def get_entity_id_by_name(
        self,
        workspace_id: UUID,
        entity_type: EntityType,
        name: str,
    ) -> Optional[UUID]:
        """Get entity ID by name, returns None if not found."""
        row = await self.get_entity_by_name(workspace_id, entity_type, name)
        return row["id"] if row else None

    async def find_or_create_entity(
        self,
        workspace_id: UUID,
        entity_type: EntityType,
        name: str,
    ) -> UUID:
        """Find entity by name or create a minimal one if not found."""
        existing = await self.get_entity_id_by_name(workspace_id, entity_type, name)
        if existing:
            return existing

        # Create minimal entity
        entity = ExtractedEntity(
            type=entity_type,
            name=name,
            aliases=[],
            description=None,
            evidence=[],
        )
        entity_id, _ = await self.upsert_entity(workspace_id, entity)
        return entity_id

    # ===========================================
    # Claim Operations
    # ===========================================

    async def create_claim(
        self,
        workspace_id: UUID,
        claim: ExtractedClaim,
        entity_id: Optional[UUID] = None,
        verdict: Optional[ClaimVerdict] = None,
        extraction_model: Optional[str] = None,
        verification_model: Optional[str] = None,
    ) -> UUID:
        """
        Create a new claim.

        Args:
            workspace_id: Workspace ID
            claim: Extracted claim
            entity_id: Optional linked entity ID
            verdict: Optional verification verdict (updates status/confidence)
            extraction_model: Model used for extraction
            verification_model: Model used for verification

        Returns:
            Created claim ID
        """
        # Determine status and confidence from verdict if provided
        if verdict:
            status = verdict.status.value
            confidence = verdict.confidence
            text = verdict.corrected_text or claim.text
        else:
            status = VerificationStatus.PENDING.value
            confidence = claim.confidence
            text = claim.text

        query = """
            INSERT INTO kb_claims (
                workspace_id, entity_id, claim_type, text,
                confidence, status, extraction_model, verification_model
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                entity_id,
                claim.claim_type.value,
                text,
                confidence,
                status,
                extraction_model,
                verification_model,
            )

            claim_id = row["id"]
            logger.debug(
                "Created claim",
                claim_id=str(claim_id),
                claim_type=claim.claim_type.value,
                status=status,
            )

            return claim_id

    async def update_claim_verdict(
        self,
        claim_id: UUID,
        verdict: ClaimVerdict,
        verification_model: Optional[str] = None,
    ) -> None:
        """Update a claim with verification verdict."""
        query = """
            UPDATE kb_claims
            SET status = $2,
                confidence = $3,
                text = COALESCE($4, text),
                verification_model = COALESCE($5, verification_model),
                updated_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                claim_id,
                verdict.status.value,
                verdict.confidence,
                verdict.corrected_text,
                verification_model,
            )

    async def get_verified_claims(
        self,
        workspace_id: UUID,
        entity_id: Optional[UUID] = None,
        claim_types: Optional[list[ClaimType]] = None,
        min_confidence: float = 0.5,
        limit: int = 100,
    ) -> list[dict]:
        """Get verified claims with optional filters."""
        conditions = ["workspace_id = $1", "status = 'verified'", "confidence >= $2"]
        params = [workspace_id, min_confidence]
        param_idx = 3

        if entity_id:
            conditions.append(f"entity_id = ${param_idx}")
            params.append(entity_id)
            param_idx += 1

        if claim_types:
            type_values = [ct.value for ct in claim_types]
            conditions.append(f"claim_type = ANY(${param_idx})")
            params.append(type_values)
            param_idx += 1

        conditions.append(f"LIMIT ${param_idx}")
        params.append(limit)

        query = f"""
            SELECT c.*, e.name as entity_name, e.type as entity_type
            FROM kb_claims c
            LEFT JOIN kb_entities e ON c.entity_id = e.id
            WHERE {' AND '.join(conditions[:-1])}
            ORDER BY c.confidence DESC, c.created_at DESC
            {conditions[-1].replace(f'${param_idx}', str(limit))}
        """

        # Fix: Use actual LIMIT syntax
        query = f"""
            SELECT c.*, e.name as entity_name, e.type as entity_type
            FROM kb_claims c
            LEFT JOIN kb_entities e ON c.entity_id = e.id
            WHERE {' AND '.join(conditions[:-1])}
            ORDER BY c.confidence DESC, c.created_at DESC
            LIMIT ${ param_idx}
        """

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *params)

    # ===========================================
    # Evidence Operations
    # ===========================================

    async def create_evidence(
        self,
        claim_id: UUID,
        doc_id: UUID,
        chunk_id: UUID,
        quote: str,
        relevance_score: float = 1.0,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
    ) -> UUID:
        """
        Create evidence linking a claim to a source chunk.

        Uses ON CONFLICT to handle duplicate (claim_id, chunk_id) pairs.
        """
        query = """
            INSERT INTO kb_evidence (
                claim_id, doc_id, chunk_id, quote,
                relevance_score, start_offset, end_offset
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (claim_id, chunk_id)
            DO UPDATE SET
                quote = EXCLUDED.quote,
                relevance_score = EXCLUDED.relevance_score
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                claim_id,
                doc_id,
                chunk_id,
                quote[:500],  # Truncate long quotes
                relevance_score,
                start_offset,
                end_offset,
            )
            return row["id"]

    async def get_evidence_for_claim(self, claim_id: UUID) -> list[dict]:
        """Get all evidence for a claim."""
        query = """
            SELECT e.*, c.text as chunk_text, d.title as doc_title
            FROM kb_evidence e
            JOIN chunks c ON e.chunk_id = c.id
            JOIN documents d ON e.doc_id = d.id
            WHERE e.claim_id = $1
            ORDER BY e.relevance_score DESC
        """

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, claim_id)

    # ===========================================
    # Relation Operations
    # ===========================================

    async def create_relation(
        self,
        workspace_id: UUID,
        relation: ExtractedRelation,
        from_entity_id: UUID,
        to_entity_id: UUID,
        claim_id: Optional[UUID] = None,
        weight: float = 1.0,
    ) -> Optional[UUID]:
        """
        Create a relation between entities.

        Uses ON CONFLICT to handle duplicate relations.
        Returns None if relation already exists (no update needed).
        """
        query = """
            INSERT INTO kb_relations (
                workspace_id, from_entity_id, relation, to_entity_id,
                claim_id, weight
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (from_entity_id, relation, to_entity_id)
            DO NOTHING
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                from_entity_id,
                relation.relation.value,
                to_entity_id,
                claim_id,
                weight,
            )
            return row["id"] if row else None

    async def get_entity_relations(
        self,
        entity_id: UUID,
        direction: str = "both",  # "from", "to", or "both"
    ) -> list[dict]:
        """Get all relations for an entity."""
        if direction == "from":
            condition = "r.from_entity_id = $1"
        elif direction == "to":
            condition = "r.to_entity_id = $1"
        else:
            condition = "(r.from_entity_id = $1 OR r.to_entity_id = $1)"

        query = f"""
            SELECT r.*,
                   f.name as from_name, f.type as from_type,
                   t.name as to_name, t.type as to_type
            FROM kb_relations r
            JOIN kb_entities f ON r.from_entity_id = f.id
            JOIN kb_entities t ON r.to_entity_id = t.id
            WHERE {condition}
            ORDER BY r.weight DESC
        """

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, entity_id)

    # ===========================================
    # Batch Persistence
    # ===========================================

    async def persist_extraction(
        self,
        workspace_id: UUID,
        entities: list[ExtractedEntity],
        claims: list[ExtractedClaim],
        relations: list[ExtractedRelation],
        verdicts: list[ClaimVerdict],
        chunk_ids: list[UUID],
        doc_id: UUID,
        extraction_model: Optional[str] = None,
        verification_model: Optional[str] = None,
    ) -> PersistenceStats:
        """
        Persist full extraction results to the truth store.

        Handles:
        1. Upserting entities
        2. Creating claims (only verified/weak, not rejected)
        3. Creating evidence links
        4. Creating relations

        Args:
            workspace_id: Workspace ID
            entities: Extracted entities
            claims: Extracted claims (parallel to verdicts by index)
            relations: Extracted relations
            verdicts: Verification verdicts (parallel to claims by index)
            chunk_ids: Source chunk IDs (parallel to extraction context)
            doc_id: Source document ID
            extraction_model: Model used for extraction
            verification_model: Model used for verification

        Returns:
            Statistics about what was persisted
        """
        stats = PersistenceStats()

        # Build verdict lookup by claim index
        verdict_map = {v.claim_index: v for v in verdicts}

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Upsert entities
                entity_id_map: dict[str, UUID] = {}  # name -> id
                for entity in entities:
                    entity_id, created = await self.upsert_entity(workspace_id, entity)
                    entity_id_map[entity.name.lower()] = entity_id
                    if created:
                        stats.entities_created += 1
                    else:
                        stats.entities_updated += 1

                # 2. Create claims and evidence (only verified/weak)
                for i, claim in enumerate(claims):
                    verdict = verdict_map.get(i)
                    if not verdict:
                        continue

                    # Skip rejected claims
                    if verdict.status == VerificationStatus.REJECTED:
                        continue

                    # Find linked entity if specified
                    entity_id = None
                    if claim.entity_name:
                        entity_id = entity_id_map.get(claim.entity_name.lower())
                        if not entity_id and claim.entity_type:
                            # Create entity if referenced but not extracted
                            entity_id = await self.find_or_create_entity(
                                workspace_id,
                                claim.entity_type,
                                claim.entity_name,
                            )
                            entity_id_map[claim.entity_name.lower()] = entity_id

                    # Create claim
                    claim_id = await self.create_claim(
                        workspace_id,
                        claim,
                        entity_id=entity_id,
                        verdict=verdict,
                        extraction_model=extraction_model,
                        verification_model=verification_model,
                    )
                    stats.claims_created += 1

                    # Create evidence for each evidence pointer
                    for ev in claim.evidence:
                        if ev.chunk_index < len(chunk_ids):
                            await self.create_evidence(
                                claim_id=claim_id,
                                doc_id=doc_id,
                                chunk_id=chunk_ids[ev.chunk_index],
                                quote=ev.quote,
                                relevance_score=ev.relevance,
                            )
                            stats.evidence_created += 1

                # 3. Create relations
                for relation in relations:
                    from_id = entity_id_map.get(relation.from_entity.lower())
                    to_id = entity_id_map.get(relation.to_entity.lower())

                    if not from_id:
                        from_id = await self.find_or_create_entity(
                            workspace_id,
                            relation.from_type,
                            relation.from_entity,
                        )
                    if not to_id:
                        to_id = await self.find_or_create_entity(
                            workspace_id,
                            relation.to_type,
                            relation.to_entity,
                        )

                    rel_id = await self.create_relation(
                        workspace_id,
                        relation,
                        from_entity_id=from_id,
                        to_entity_id=to_id,
                    )
                    if rel_id:
                        stats.relations_created += 1

        logger.info(
            "Persisted extraction to truth store",
            workspace_id=str(workspace_id),
            stats=stats.model_dump(),
        )

        return stats
