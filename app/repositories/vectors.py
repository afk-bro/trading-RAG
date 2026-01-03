"""Vector repository for Qdrant operations."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings, get_settings
from app.schemas import QueryFilters, SymbolsMode

logger = structlog.get_logger(__name__)


class VectorRepository:
    """Repository for Qdrant vector operations."""

    def __init__(
        self,
        client: Optional[AsyncQdrantClient] = None,
        collection: Optional[str] = None,
    ):
        """
        Initialize repository.

        Args:
            client: Qdrant async client
            collection: Collection name
        """
        settings = get_settings()
        self.client = client
        self.collection = collection or settings.qdrant_collection_active

    async def ensure_collection(
        self,
        dimension: int,
        recreate: bool = False,
    ) -> None:
        """
        Ensure collection exists with correct configuration.

        Args:
            dimension: Vector dimension
            recreate: If True, recreate collection
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        collections = await self.client.get_collections()
        exists = any(c.name == self.collection for c in collections.collections)

        if exists and recreate:
            await self.client.delete_collection(self.collection)
            exists = False

        if not exists:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

            # Create payload indexes
            await self._create_indexes()

            logger.info(
                "Created collection",
                collection=self.collection,
                dimension=dimension,
            )

    async def _create_indexes(self) -> None:
        """Create payload indexes for filtering."""
        indexes = [
            ("workspace_id", qmodels.PayloadSchemaType.KEYWORD),
            ("source_type", qmodels.PayloadSchemaType.KEYWORD),
            ("published_at", qmodels.PayloadSchemaType.INTEGER),
            ("symbols", qmodels.PayloadSchemaType.KEYWORD),
            ("topics", qmodels.PayloadSchemaType.KEYWORD),
            ("entities", qmodels.PayloadSchemaType.KEYWORD),
            ("author", qmodels.PayloadSchemaType.KEYWORD),
            ("channel", qmodels.PayloadSchemaType.KEYWORD),
        ]

        for field_name, schema_type in indexes:
            await self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field_name,
                field_schema=schema_type,
            )

    async def upsert_batch(
        self,
        points: list[dict],
    ) -> None:
        """
        Upsert vectors with payloads.

        Args:
            points: List of dicts with 'id', 'vector', 'payload'
        """
        if not points:
            return

        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        qdrant_points = [
            qmodels.PointStruct(
                id=str(p["id"]),
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points
        ]

        await self.client.upsert(
            collection_name=self.collection,
            points=qdrant_points,
        )

        logger.debug(
            "Upserted vectors",
            collection=self.collection,
            count=len(points),
        )

    async def delete_batch(self, point_ids: list[UUID]) -> None:
        """
        Delete vectors by IDs.

        Args:
            point_ids: List of point IDs to delete
        """
        if not point_ids:
            return

        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        await self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.PointIdsList(
                points=[str(pid) for pid in point_ids],
            ),
        )

        logger.debug(
            "Deleted vectors",
            collection=self.collection,
            count=len(point_ids),
        )

    async def search(
        self,
        vector: list[float],
        workspace_id: UUID,
        filters: Optional[QueryFilters] = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Search for similar vectors with filtering.

        Args:
            vector: Query vector
            workspace_id: Workspace to search
            filters: Optional query filters
            limit: Maximum results

        Returns:
            List of results with score and payload
        """
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

        # Build Qdrant filter
        filter_conditions = [
            qmodels.FieldCondition(
                key="workspace_id",
                match=qmodels.MatchValue(value=str(workspace_id)),
            )
        ]

        if filters:
            filter_conditions.extend(self._build_filter_conditions(filters))

        qdrant_filter = qmodels.Filter(must=filter_conditions) if filter_conditions else None

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "id": UUID(result.id) if isinstance(result.id, str) else result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

    def _build_filter_conditions(
        self,
        filters: QueryFilters,
    ) -> list[qmodels.Condition]:
        """Build Qdrant filter conditions from QueryFilters."""
        conditions = []

        # Source types filter
        if filters.source_types:
            conditions.append(
                qmodels.FieldCondition(
                    key="source_type",
                    match=qmodels.MatchAny(any=[st.value for st in filters.source_types]),
                )
            )

        # Symbols filter
        if filters.symbols:
            if filters.symbols_mode == SymbolsMode.ALL:
                # All symbols must match
                for symbol in filters.symbols:
                    conditions.append(
                        qmodels.FieldCondition(
                            key="symbols",
                            match=qmodels.MatchValue(value=symbol),
                        )
                    )
            else:
                # Any symbol matches
                conditions.append(
                    qmodels.FieldCondition(
                        key="symbols",
                        match=qmodels.MatchAny(any=filters.symbols),
                    )
                )

        # Topics filter
        if filters.topics:
            conditions.append(
                qmodels.FieldCondition(
                    key="topics",
                    match=qmodels.MatchAny(any=filters.topics),
                )
            )

        # Entities filter
        if filters.entities:
            conditions.append(
                qmodels.FieldCondition(
                    key="entities",
                    match=qmodels.MatchAny(any=filters.entities),
                )
            )

        # Authors filter
        if filters.authors:
            conditions.append(
                qmodels.FieldCondition(
                    key="author",
                    match=qmodels.MatchAny(any=filters.authors),
                )
            )

        # Date range filters
        if filters.published_from:
            timestamp = int(filters.published_from.timestamp())
            conditions.append(
                qmodels.FieldCondition(
                    key="published_at",
                    range=qmodels.Range(gte=timestamp),
                )
            )

        if filters.published_to:
            timestamp = int(filters.published_to.timestamp())
            conditions.append(
                qmodels.FieldCondition(
                    key="published_at",
                    range=qmodels.Range(lte=timestamp),
                )
            )

        return conditions

    async def get_collection_info(self) -> Optional[dict]:
        """Get collection information."""
        if not self.client:
            return None

        try:
            info = await self.client.get_collection(self.collection)
            return {
                "name": self.collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            return None


class ChunkVectorRepository:
    """Repository for chunk_vectors table in Postgres."""

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def create_batch(
        self,
        records: list[dict],
    ) -> list[UUID]:
        """
        Create chunk_vector records.

        Args:
            records: List of record dicts

        Returns:
            List of created record IDs
        """
        if not records:
            return []

        query = """
            INSERT INTO chunk_vectors (
                chunk_id, workspace_id, embed_provider, embed_model,
                collection, vector_dim, qdrant_point_id, status,
                indexed_at, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, 'indexed', NOW(), NOW()
            )
            ON CONFLICT (chunk_id, embed_provider, embed_model, collection)
            DO UPDATE SET
                status = 'indexed',
                indexed_at = NOW(),
                qdrant_point_id = EXCLUDED.qdrant_point_id
            RETURNING id
        """

        record_ids = []

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for record in records:
                    row = await conn.fetchrow(
                        query,
                        record["chunk_id"],
                        record["workspace_id"],
                        record["embed_provider"],
                        record["embed_model"],
                        record["collection"],
                        record["vector_dim"],
                        str(record["chunk_id"]),  # point_id = chunk_id
                    )
                    record_ids.append(row["id"])

        logger.debug(
            "Created chunk_vector records",
            count=len(record_ids),
        )

        return record_ids

    async def get_by_chunk_id(
        self,
        chunk_id: UUID,
        collection: Optional[str] = None,
    ) -> list[dict]:
        """Get vector records for a chunk."""
        if collection:
            query = """
                SELECT * FROM chunk_vectors
                WHERE chunk_id = $1 AND collection = $2
            """
            params = (chunk_id, collection)
        else:
            query = "SELECT * FROM chunk_vectors WHERE chunk_id = $1"
            params = (chunk_id,)

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *params)

    async def delete_by_chunk_ids(
        self,
        chunk_ids: list[UUID],
        collection: Optional[str] = None,
    ) -> int:
        """Delete vector records for chunks."""
        if collection:
            query = """
                DELETE FROM chunk_vectors
                WHERE chunk_id = ANY($1::uuid[]) AND collection = $2
            """
            params = (chunk_ids, collection)
        else:
            query = "DELETE FROM chunk_vectors WHERE chunk_id = ANY($1::uuid[])"
            params = (chunk_ids,)

        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return int(result.split()[-1])
