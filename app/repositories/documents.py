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

    async def get_version(self, doc_id: UUID) -> int:
        """Get the version number of a document."""
        query = "SELECT version FROM documents WHERE id = $1"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, doc_id)
            return row["version"] if row else 1

    async def supersede_and_create(
        self,
        old_doc_id: UUID,
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
    ) -> tuple[UUID, int]:
        """
        Supersede an existing document and create a new version.

        Uses write-new-first pattern:
        1. Get old document's version
        2. Mark old document as superseded (removes unique constraint conflict)
        3. Create new document with incremented version

        Returns:
            Tuple of (new_doc_id, new_version)
        """
        async with self.pool.acquire() as conn:
            # Start transaction for atomicity
            async with conn.transaction():
                # Get old document's version
                old_row = await conn.fetchrow(
                    "SELECT version FROM documents WHERE id = $1", old_doc_id
                )
                old_version = old_row["version"] if old_row else 1
                new_version = old_version + 1

                # Mark old document as superseded
                await conn.execute(
                    """
                    UPDATE documents
                    SET status = 'superseded', updated_at = NOW()
                    WHERE id = $1
                    """,
                    old_doc_id,
                )

                logger.info(
                    "Superseded old document",
                    old_doc_id=str(old_doc_id),
                    old_version=old_version,
                )

                # Create new document with incremented version
                query = """
                    INSERT INTO documents (
                        workspace_id, source_url, canonical_url, source_type,
                        content_hash, title, author, channel, published_at,
                        language, duration_secs, video_id, playlist_id,
                        status, version, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        'active', $14, NOW(), NOW()
                    )
                    RETURNING id
                """

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
                    new_version,
                )
                new_doc_id = row["id"]

                logger.info(
                    "Created new document version",
                    new_doc_id=str(new_doc_id),
                    new_version=new_version,
                    superseded_doc_id=str(old_doc_id),
                )

                return new_doc_id, new_version

    async def get_chunk_ids(self, doc_id: UUID) -> list[UUID]:
        """Get all chunk IDs for a document."""
        query = "SELECT id FROM chunks WHERE doc_id = $1"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, doc_id)
            return [row["id"] for row in rows]
