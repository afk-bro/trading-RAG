"""Document repository for Supabase Postgres."""

from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog

from app.schemas import DocumentStatus, SourceType

logger = structlog.get_logger(__name__)


class DocumentRepository:
    """Repository for document CRUD operations."""

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def create(
        self,
        workspace_id: UUID,
        source_url: Optional[str],
        canonical_url: str,
        source_type: SourceType,
        content_hash: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        channel: Optional[str] = None,
        published_at: Optional[datetime] = None,
        language: str = "en",
        duration_secs: Optional[int] = None,
        video_id: Optional[str] = None,
        playlist_id: Optional[str] = None,
    ) -> UUID:
        """
        Create a new document.

        Returns:
            Created document ID
        """
        query = """
            INSERT INTO documents (
                workspace_id, source_url, canonical_url, source_type,
                content_hash, title, author, channel, published_at,
                language, duration_secs, video_id, playlist_id,
                status, version, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                'active', 1, NOW(), NOW()
            )
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                source_url,
                canonical_url,
                source_type.value,
                content_hash,
                title,
                author,
                channel,
                published_at,
                language,
                duration_secs,
                video_id,
                playlist_id,
            )
            doc_id = row["id"]

            logger.info(
                "Created document",
                doc_id=str(doc_id),
                workspace_id=str(workspace_id),
                source_type=source_type.value,
            )

            return doc_id

    async def get_by_id(self, doc_id: UUID) -> Optional[dict]:
        """Get document by ID."""
        query = "SELECT * FROM documents WHERE id = $1"

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, doc_id)

    async def get_by_canonical_url(
        self,
        workspace_id: UUID,
        source_type: SourceType,
        canonical_url: str,
    ) -> Optional[dict]:
        """Get document by unique constraint fields."""
        query = """
            SELECT * FROM documents
            WHERE workspace_id = $1
              AND source_type = $2
              AND canonical_url = $3
              AND status = 'active'
        """

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                query,
                workspace_id,
                source_type.value,
                canonical_url,
            )

    async def get_by_content_hash(
        self,
        workspace_id: UUID,
        content_hash: str,
    ) -> Optional[dict]:
        """Get document by content hash (for deduplication)."""
        query = """
            SELECT * FROM documents
            WHERE workspace_id = $1
              AND content_hash = $2
              AND status = 'active'
            LIMIT 1
        """

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, workspace_id, content_hash)

    async def update_status(
        self,
        doc_id: UUID,
        status: DocumentStatus,
    ) -> None:
        """Update document status."""
        query = """
            UPDATE documents
            SET status = $2, updated_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, doc_id, status.value)

    async def supersede(self, doc_id: UUID) -> None:
        """Mark document as superseded."""
        await self.update_status(doc_id, DocumentStatus.SUPERSEDED)

    async def update_last_indexed(self, doc_id: UUID) -> None:
        """Update last_indexed_at timestamp."""
        query = """
            UPDATE documents
            SET last_indexed_at = NOW(), updated_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, doc_id)

    async def list_by_workspace(
        self,
        workspace_id: UUID,
        source_type: Optional[SourceType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List documents for a workspace."""
        if source_type:
            query = """
                SELECT * FROM documents
                WHERE workspace_id = $1
                  AND source_type = $2
                  AND status = 'active'
                ORDER BY published_at DESC NULLS LAST, created_at DESC
                LIMIT $3 OFFSET $4
            """
            params = (workspace_id, source_type.value, limit, offset)
        else:
            query = """
                SELECT * FROM documents
                WHERE workspace_id = $1
                  AND status = 'active'
                ORDER BY published_at DESC NULLS LAST, created_at DESC
                LIMIT $2 OFFSET $3
            """
            params = (workspace_id, limit, offset)

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *params)
