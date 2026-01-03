"""Chunk repository for Supabase Postgres."""

from typing import Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


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

        query = """
            INSERT INTO chunks (
                doc_id, workspace_id, chunk_index, content, content_hash,
                token_count, section, time_start_secs, time_end_secs,
                page_start, page_end, locator_label, speaker,
                symbols, entities, topics, quality_score,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14, $15, $16, $17, NOW(), NOW()
            )
            RETURNING id
        """

        chunk_ids = []

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for chunk in chunks:
                    row = await conn.fetchrow(
                        query,
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
                    )
                    chunk_ids.append(row["id"])

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
        Get chunks for a workspace.

        Args:
            workspace_id: Workspace ID
            doc_ids: Optional list of doc IDs to filter
            limit: Maximum chunks to return
        """
        if doc_ids:
            query = """
                SELECT * FROM chunks
                WHERE workspace_id = $1
                  AND doc_id = ANY($2::uuid[])
                ORDER BY doc_id, chunk_index
                LIMIT $3
            """
            params = (workspace_id, doc_ids, limit)
        else:
            query = """
                SELECT * FROM chunks
                WHERE workspace_id = $1
                ORDER BY doc_id, chunk_index
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
