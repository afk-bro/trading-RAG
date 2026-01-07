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
    EvidencePointer,
    compute_claim_fingerprint,
    validate_claim_evidence,
    MAX_QUOTE_LENGTH,
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
    ) -> tuple[UUID | None, bool]:
        """
        Create a new claim with deduplication.

        Uses fingerprint-based deduplication to prevent duplicate claims.

        Args:
            workspace_id: Workspace ID
            claim: Extracted claim
            entity_id: Optional linked entity ID
            verdict: Optional verification verdict (updates status/confidence)
            extraction_model: Model used for extraction
            verification_model: Model used for verification

        Returns:
            Tuple of (claim_id, created) - claim_id is None if duplicate, created is False if skipped
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

        # Compute fingerprint for deduplication
        fingerprint = compute_claim_fingerprint(
            claim_text=text,
            claim_type=claim.claim_type,
            entity_name=claim.entity_name,
            workspace_id=workspace_id,
        )

        # Use INSERT ... ON CONFLICT DO NOTHING for idempotent insert
        query = """
            INSERT INTO kb_claims (
                workspace_id, entity_id, claim_type, text,
                confidence, status, extraction_model, verification_model,
                fingerprint
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (fingerprint) WHERE fingerprint IS NOT NULL
            DO NOTHING
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
                fingerprint,
            )

            if row:
                claim_id = row["id"]
                logger.debug(
                    "Created claim",
                    claim_id=str(claim_id),
                    claim_type=claim.claim_type.value,
                    status=status,
                    fingerprint=fingerprint[:16] + "...",
                )
                return claim_id, True
            else:
                logger.debug(
                    "Skipped duplicate claim",
                    claim_type=claim.claim_type.value,
                    fingerprint=fingerprint[:16] + "...",
                )
                return None, False

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
    # List/Search Operations (for KB endpoints)
    # ===========================================

    async def list_entities(
        self,
        workspace_id: UUID,
        q: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        limit: int = 50,
        offset: int = 0,
        include_counts: bool = False,
    ) -> tuple[list[dict], int]:
        """
        List entities with optional search and filtering.

        Args:
            workspace_id: Workspace ID
            q: Optional text search (ILIKE on name)
            entity_type: Optional filter by entity type
            limit: Max results (default 50, max 200)
            offset: Pagination offset
            include_counts: Include verified claim counts

        Returns:
            Tuple of (entities, total_count)
        """
        limit = min(limit, 200)  # Cap at 200

        # Build WHERE conditions
        conditions = ["e.workspace_id = $1"]
        params: list = [workspace_id]
        param_idx = 2

        if q:
            conditions.append(f"(e.name ILIKE ${param_idx} OR e.aliases::text ILIKE ${param_idx})")
            params.append(f"%{q}%")
            param_idx += 1

        if entity_type:
            conditions.append(f"e.type = ${param_idx}")
            params.append(entity_type.value)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"""
            SELECT COUNT(*) FROM kb_entities e
            WHERE {where_clause}
        """

        # Main query with optional claim count
        if include_counts:
            select_query = f"""
                SELECT e.*,
                       COALESCE(claim_counts.verified_count, 0) as verified_claim_count
                FROM kb_entities e
                LEFT JOIN (
                    SELECT entity_id, COUNT(*) as verified_count
                    FROM kb_claims
                    WHERE status = 'verified'
                    GROUP BY entity_id
                ) claim_counts ON claim_counts.entity_id = e.id
                WHERE {where_clause}
                ORDER BY e.name ASC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
        else:
            select_query = f"""
                SELECT e.*, NULL as verified_claim_count
                FROM kb_entities e
                WHERE {where_clause}
                ORDER BY e.name ASC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """

        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(select_query, *params)

        return [dict(r) for r in rows], total

    async def get_entity_by_id(
        self,
        entity_id: UUID,
    ) -> Optional[dict]:
        """Get a single entity by ID with stats."""
        query = """
            SELECT e.*,
                   COALESCE(verified.count, 0) as verified_claims,
                   COALESCE(weak.count, 0) as weak_claims,
                   COALESCE(total.count, 0) as total_claims,
                   COALESCE(rels.count, 0) as relations_count
            FROM kb_entities e
            LEFT JOIN (
                SELECT entity_id, COUNT(*) as count
                FROM kb_claims WHERE status = 'verified'
                GROUP BY entity_id
            ) verified ON verified.entity_id = e.id
            LEFT JOIN (
                SELECT entity_id, COUNT(*) as count
                FROM kb_claims WHERE status = 'weak'
                GROUP BY entity_id
            ) weak ON weak.entity_id = e.id
            LEFT JOIN (
                SELECT entity_id, COUNT(*) as count
                FROM kb_claims
                GROUP BY entity_id
            ) total ON total.entity_id = e.id
            LEFT JOIN (
                SELECT from_entity_id as entity_id, COUNT(*) as count
                FROM kb_relations GROUP BY from_entity_id
                UNION ALL
                SELECT to_entity_id as entity_id, COUNT(*) as count
                FROM kb_relations GROUP BY to_entity_id
            ) rels ON rels.entity_id = e.id
            WHERE e.id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, entity_id)
            return dict(row) if row else None

    async def list_claims(
        self,
        workspace_id: UUID,
        q: Optional[str] = None,
        status: Optional[str] = "verified",
        claim_type: Optional[ClaimType] = None,
        entity_id: Optional[UUID] = None,
        source_id: Optional[UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        List claims with optional search and filtering.

        Args:
            workspace_id: Workspace ID
            q: Optional text search (ILIKE on claim text)
            status: Filter by status (default 'verified')
            claim_type: Optional filter by claim type
            entity_id: Optional filter by entity
            source_id: Optional filter by source document
            limit: Max results (default 50, max 200)
            offset: Pagination offset

        Returns:
            Tuple of (claims, total_count)
        """
        limit = min(limit, 200)  # Cap at 200

        # Build WHERE conditions
        conditions = ["c.workspace_id = $1"]
        params: list = [workspace_id]
        param_idx = 2

        if q:
            conditions.append(f"c.text ILIKE ${param_idx}")
            params.append(f"%{q}%")
            param_idx += 1

        if status:
            conditions.append(f"c.status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if claim_type:
            conditions.append(f"c.claim_type = ${param_idx}")
            params.append(claim_type.value)
            param_idx += 1

        if entity_id:
            conditions.append(f"c.entity_id = ${param_idx}")
            params.append(entity_id)
            param_idx += 1

        if source_id:
            conditions.append(f"""
                EXISTS (
                    SELECT 1 FROM kb_evidence e
                    WHERE e.claim_id = c.id AND e.doc_id = ${param_idx}
                )
            """)
            params.append(source_id)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"""
            SELECT COUNT(*) FROM kb_claims c
            WHERE {where_clause}
        """

        # Main query with entity join and evidence stats
        select_query = f"""
            SELECT c.*,
                   e.name as entity_name,
                   e.type as entity_type,
                   (SELECT COUNT(*) FROM kb_evidence ev WHERE ev.claim_id = c.id) as evidence_count,
                   (SELECT LEFT(ev.quote, 100) FROM kb_evidence ev
                    WHERE ev.claim_id = c.id
                    ORDER BY ev.relevance_score DESC LIMIT 1) as first_quote
            FROM kb_claims c
            LEFT JOIN kb_entities e ON c.entity_id = e.id
            WHERE {where_clause}
            ORDER BY c.confidence DESC, c.created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """

        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(select_query, *params)

        return [dict(r) for r in rows], total

    async def get_claim_by_id(
        self,
        claim_id: UUID,
        include_evidence: bool = True,
    ) -> Optional[dict]:
        """Get a single claim by ID with optional evidence."""
        query = """
            SELECT c.*,
                   e.name as entity_name,
                   e.type as entity_type
            FROM kb_claims c
            LEFT JOIN kb_entities e ON c.entity_id = e.id
            WHERE c.id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, claim_id)
            if not row:
                return None

            result = dict(row)

            if include_evidence:
                evidence_query = """
                    SELECT ev.id, ev.doc_id, ev.chunk_id, ev.quote,
                           ev.relevance_score, d.title as doc_title
                    FROM kb_evidence ev
                    JOIN documents d ON ev.doc_id = d.id
                    WHERE ev.claim_id = $1
                    ORDER BY ev.relevance_score DESC
                """
                evidence_rows = await conn.fetch(evidence_query, claim_id)
                result["evidence"] = [dict(e) for e in evidence_rows]

            return result

    async def search_claims_for_answer(
        self,
        workspace_id: UUID,
        query_text: str,
        limit: int = 20,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """
        Search verified claims for kb_answer mode.

        Uses text search (ILIKE) on claim text and entity names.
        Ranks by confidence and recency.

        Args:
            workspace_id: Workspace ID
            query_text: Search query
            limit: Max results
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching claims with entity info
        """
        # Split query into words for better matching
        words = query_text.lower().split()
        word_conditions = []
        params: list = [workspace_id, min_confidence]
        param_idx = 3

        for word in words[:5]:  # Limit to first 5 words
            word_conditions.append(f"""
                (c.text ILIKE ${param_idx} OR e.name ILIKE ${param_idx} OR e.aliases::text ILIKE ${param_idx})
            """)
            params.append(f"%{word}%")
            param_idx += 1

        # At least one word must match
        word_clause = " OR ".join(word_conditions) if word_conditions else "TRUE"

        query = f"""
            SELECT c.*,
                   e.name as entity_name,
                   e.type as entity_type
            FROM kb_claims c
            LEFT JOIN kb_entities e ON c.entity_id = e.id
            WHERE c.workspace_id = $1
              AND c.status = 'verified'
              AND c.confidence >= $2
              AND ({word_clause})
            ORDER BY c.confidence DESC, c.created_at DESC
            LIMIT ${param_idx}
        """
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(r) for r in rows]

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
        2. Validating evidence integrity
        3. Creating claims with fingerprint deduplication (only verified/weak, not rejected)
        4. Creating evidence links
        5. Creating relations

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

        # Build set of available chunk indices for evidence validation
        available_chunk_indices = set(range(len(chunk_ids)))

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

                    # Validate evidence integrity before persisting
                    is_valid, validated_evidence, errors = validate_claim_evidence(
                        claim, available_chunk_indices
                    )

                    if not is_valid:
                        # Downgrade to invalid if VERIFIED claim has no valid evidence
                        logger.warning(
                            "Skipping claim with invalid evidence",
                            claim_text=claim.text[:50],
                            errors=errors,
                        )
                        stats.claims_skipped_invalid += 1
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

                    # Create claim (with fingerprint deduplication)
                    claim_id, created = await self.create_claim(
                        workspace_id,
                        claim,
                        entity_id=entity_id,
                        verdict=verdict,
                        extraction_model=extraction_model,
                        verification_model=verification_model,
                    )

                    if created and claim_id:
                        stats.claims_created += 1

                        # Create evidence for each validated evidence pointer
                        for ev in validated_evidence:
                            if ev.chunk_index < len(chunk_ids):
                                await self.create_evidence(
                                    claim_id=claim_id,
                                    doc_id=doc_id,
                                    chunk_id=chunk_ids[ev.chunk_index],
                                    quote=ev.quote[:MAX_QUOTE_LENGTH],  # Ensure truncation
                                    relevance_score=ev.relevance,
                                )
                                stats.evidence_created += 1
                    else:
                        stats.claims_skipped_duplicate += 1

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
