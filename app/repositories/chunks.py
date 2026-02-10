"""Chunk repository for Supabase Postgres."""

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NeighborChunk:
    """A chunk fetched as a neighbor of a seed chunk."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source_type: str | None
    seed_chunk_id: str  # Which seed this is a neighbor of


class ChunkRepository:
    """Repository for chunk CRUD operations."""

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def create_batch(
        self,
        doc_id: UUID,
        workspace_id: UUID,
        chunks: list[dict],
    ) -> list[UUID]:
        """
        Create multiple chunks for a document.

        Args:
            doc_id: Parent document ID
            workspace_id: Workspace ID
            chunks: List of chunk data dicts

        Returns:
            List of created chunk IDs
        """
        if not chunks:
            return []

        # Build multi-row INSERT for batch efficiency (avoids N round-trips)
        params: list = []
        value_clauses: list[str] = []
        for i, chunk in enumerate(chunks):
            offset = i * 17
            value_clauses.append(
                f"(${offset+1}, ${offset+2}, ${offset+3}, ${offset+4}, "
                f"${offset+5}, ${offset+6}, ${offset+7}, ${offset+8}, "
                f"${offset+9}, ${offset+10}, ${offset+11}, ${offset+12}, "
                f"${offset+13}, ${offset+14}, ${offset+15}, ${offset+16}, "
                f"${offset+17}, NOW(), NOW())"
            )
            params.extend([
                doc_id,
                workspace_id,
                chunk.get("chunk_index", 0),
                chunk["content"],
                chunk.get("content_hash", ""),
                chunk.get("token_count", 0),
                chunk.get("section"),
                chunk.get("time_start_secs"),
                chunk.get("time_end_secs"),
                chunk.get("page_start"),
                chunk.get("page_end"),
                chunk.get("locator_label"),
                chunk.get("speaker"),
                chunk.get("symbols", []),
                chunk.get("entities", []),
                chunk.get("topics", []),
                chunk.get("quality_score", 1.0),
            ])

        query = f"""
            INSERT INTO chunks (
                doc_id, workspace_id, chunk_index, content, content_hash,
                token_count, section, time_start_secs, time_end_secs,
                page_start, page_end, locator_label, speaker,
                symbols, entities, topics, quality_score,
                created_at, updated_at
            ) VALUES {", ".join(value_clauses)}
            RETURNING id
        """

        chunk_ids = []

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(query, *params)
                chunk_ids = [row["id"] for row in rows]

        logger.info(
            "Created chunks",
            doc_id=str(doc_id),
            count=len(chunk_ids),
        )

        return chunk_ids

    async def get_by_id(self, chunk_id: UUID) -> Optional[dict]:
        """Get chunk by ID."""
        query = "SELECT * FROM chunks WHERE id = $1"

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, chunk_id)

    async def get_by_doc_id(self, doc_id: UUID) -> list[dict]:
        """Get all chunks for a document."""
        query = """
            SELECT * FROM chunks
            WHERE doc_id = $1
            ORDER BY chunk_index
        """

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, doc_id)

    async def get_by_ids(
        self,
        chunk_ids: list[UUID],
        preserve_order: bool = True,
    ) -> list[dict]:
        """
        Get chunks by IDs.

        Args:
            chunk_ids: List of chunk IDs
            preserve_order: If True, preserve input order using array_position

        Returns:
            List of chunk records
        """
        if not chunk_ids:
            return []

        if preserve_order:
            query = """
                SELECT c.*, d.source_url, d.canonical_url, d.title, d.author,
                       d.channel, d.published_at, d.source_type, d.video_id
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.id = ANY($1::uuid[])
                ORDER BY array_position($1::uuid[], c.id)
            """
        else:
            query = """
                SELECT c.*, d.source_url, d.canonical_url, d.title, d.author,
                       d.channel, d.published_at, d.source_type, d.video_id
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.id = ANY($1::uuid[])
            """

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, chunk_ids)

    async def get_by_workspace(
        self,
        workspace_id: UUID,
        doc_ids: Optional[list[UUID]] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Get chunks for a workspace with document metadata.

        Args:
            workspace_id: Workspace ID
            doc_ids: Optional list of doc IDs to filter
            limit: Maximum chunks to return

        Returns:
            List of chunks with document metadata (author, published_at, source_type)
        """
        if doc_ids:
            query = """
                SELECT c.*, d.source_url, d.canonical_url, d.title, d.author,
                       d.channel, d.published_at, d.source_type, d.video_id
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.workspace_id = $1
                  AND c.doc_id = ANY($2::uuid[])
                ORDER BY c.doc_id, c.chunk_index
                LIMIT $3
            """
            params = (workspace_id, doc_ids, limit)
        else:
            query = """
                SELECT c.*, d.source_url, d.canonical_url, d.title, d.author,
                       d.channel, d.published_at, d.source_type, d.video_id
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.workspace_id = $1
                ORDER BY c.doc_id, c.chunk_index
                LIMIT $2
            """
            params = (workspace_id, limit)

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *params)

    async def delete_by_doc_id(self, doc_id: UUID) -> int:
        """
        Delete all chunks for a document.

        Returns:
            Number of deleted chunks
        """
        query = "DELETE FROM chunks WHERE doc_id = $1"

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, doc_id)
            # Parse "DELETE N" result
            count = int(result.split()[-1])

            logger.info(
                "Deleted chunks",
                doc_id=str(doc_id),
                count=count,
            )

            return count

    async def count_by_workspace(self, workspace_id: UUID) -> int:
        """Count chunks in a workspace."""
        query = "SELECT COUNT(*) FROM chunks WHERE workspace_id = $1"

        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, workspace_id)

    async def get_by_ids_map(
        self,
        chunk_ids: list[str],
    ) -> dict[str, dict]:
        """
        Get chunks by IDs as a dict keyed by chunk_id.

        Args:
            chunk_ids: List of chunk ID strings

        Returns:
            Dict mapping chunk_id -> chunk record with document metadata
        """
        if not chunk_ids:
            return {}

        # Convert string IDs to UUIDs
        uuid_ids = [UUID(cid) if isinstance(cid, str) else cid for cid in chunk_ids]

        query = """
            SELECT c.*, d.source_url, d.canonical_url, d.title, d.author,
                   d.channel, d.published_at, d.source_type, d.video_id
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE c.id = ANY($1::uuid[])
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, uuid_ids)

        return {str(row["id"]): dict(row) for row in rows}

    async def get_neighbors_by_doc_indices(
        self,
        requests: list[tuple[str, int, str]],
    ) -> list[NeighborChunk]:
        """
        Fetch chunks by (document_id, chunk_index) pairs with seed attribution.

        Args:
            requests: List of (document_id, chunk_index, seed_chunk_id) tuples

        Returns:
            List of NeighborChunk objects
        """
        if not requests:
            return []

        # Parse and prepare arrays for unnest
        doc_ids = [UUID(r[0]) if isinstance(r[0], str) else r[0] for r in requests]
        indices = [r[1] for r in requests]
        seed_ids = [UUID(r[2]) if isinstance(r[2], str) else r[2] for r in requests]

        query = """
            WITH req(document_id, chunk_index, seed_chunk_id) AS (
                SELECT * FROM unnest($1::uuid[], $2::int[], $3::uuid[])
            )
            SELECT
                c.id AS chunk_id,
                c.doc_id AS document_id,
                c.chunk_index,
                c.content AS text,
                d.source_type,
                req.seed_chunk_id
            FROM req
            JOIN chunks c ON c.doc_id = req.document_id
                         AND c.chunk_index = req.chunk_index
            JOIN documents d ON d.id = c.doc_id
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, doc_ids, indices, seed_ids)

        return [
            NeighborChunk(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                chunk_index=row["chunk_index"],
                text=row["text"],
                source_type=row["source_type"],
                seed_chunk_id=str(row["seed_chunk_id"]),
            )
            for row in rows
        ]
