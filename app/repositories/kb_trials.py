"""
KB Trial Repository for Qdrant operations.

Handles storage and retrieval of trial documents for the
Trading Knowledge Base recommendation system.
"""

from typing import Any, Optional
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.services.kb.constants import (
    KB_TRIALS_COLLECTION_NAME,
    KB_TRIALS_DOC_TYPE,
    TIMEOUT_QDRANT_S,
)

logger = structlog.get_logger(__name__)


class KBTrialRepository:
    """Repository for KB trial vector operations in Qdrant."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: Optional[str] = None,
    ):
        """
        Initialize repository.

        Args:
            client: Qdrant async client
            collection: Collection name (default: trading_kb_trials)
        """
        self.client = client
        self.collection = collection or KB_TRIALS_COLLECTION_NAME

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
        collections = await self.client.get_collections()
        exists = any(c.name == self.collection for c in collections.collections)

        if exists and not recreate:
            # Validate existing collection dimension
            try:
                collection_info = await self.client.get_collection(self.collection)
                existing_dim = collection_info.config.params.vectors.size

                if existing_dim != dimension:
                    logger.warning(
                        "KB collection dimension mismatch",
                        collection=self.collection,
                        expected_dimension=dimension,
                        existing_dimension=existing_dim,
                    )
                    await self.client.delete_collection(self.collection)
                    exists = False
                else:
                    logger.info(
                        "KB collection validated",
                        collection=self.collection,
                        dimension=dimension,
                    )
                    return
            except Exception as e:
                logger.warning(
                    "Failed to validate KB collection",
                    collection=self.collection,
                    error=str(e),
                )

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

            # Create payload indexes for KB trial filtering
            await self._create_indexes()

            logger.info(
                "Created KB trials collection",
                collection=self.collection,
                dimension=dimension,
            )

    async def _create_indexes(self) -> None:
        """Create payload indexes for KB trial filtering."""
        indexes = [
            # Identity
            ("doc_type", qmodels.PayloadSchemaType.KEYWORD),
            ("workspace_id", qmodels.PayloadSchemaType.KEYWORD),
            ("tune_id", qmodels.PayloadSchemaType.KEYWORD),
            ("tune_run_id", qmodels.PayloadSchemaType.KEYWORD),
            ("dataset_id", qmodels.PayloadSchemaType.KEYWORD),
            ("instrument", qmodels.PayloadSchemaType.KEYWORD),
            ("timeframe", qmodels.PayloadSchemaType.KEYWORD),

            # Strategy
            ("strategy_name", qmodels.PayloadSchemaType.KEYWORD),
            ("objective_type", qmodels.PayloadSchemaType.KEYWORD),

            # Quality flags
            ("has_oos", qmodels.PayloadSchemaType.BOOL),
            ("is_valid", qmodels.PayloadSchemaType.BOOL),

            # Regime tags
            ("regime_tags", qmodels.PayloadSchemaType.KEYWORD),

            # Numeric (for range filters)
            ("sharpe_oos", qmodels.PayloadSchemaType.FLOAT),
            ("return_frac_oos", qmodels.PayloadSchemaType.FLOAT),
            ("max_dd_frac_oos", qmodels.PayloadSchemaType.FLOAT),
            ("n_trades_oos", qmodels.PayloadSchemaType.INTEGER),
            ("overfit_gap", qmodels.PayloadSchemaType.FLOAT),
            ("objective_score", qmodels.PayloadSchemaType.FLOAT),
        ]

        for field_name, schema_type in indexes:
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception as e:
                logger.warning(
                    "Failed to create index",
                    field_name=field_name,
                    error=str(e),
                )

    async def upsert_trial(
        self,
        point_id: str | UUID,
        vector: list[float],
        payload: dict,
    ) -> None:
        """
        Upsert a single trial document.

        Args:
            point_id: Unique point ID (typically tune_run_id)
            vector: Embedding vector
            payload: Trial metadata
        """
        point = qmodels.PointStruct(
            id=str(point_id),
            vector=vector,
            payload=payload,
        )

        await self.client.upsert(
            collection_name=self.collection,
            points=[point],
        )

        logger.debug(
            "Upserted KB trial",
            point_id=str(point_id),
            strategy=payload.get("strategy_name"),
        )

    async def upsert_batch(
        self,
        points: list[dict],
    ) -> int:
        """
        Upsert multiple trial documents.

        Args:
            points: List of dicts with 'id', 'vector', 'payload'

        Returns:
            Number of points upserted
        """
        if not points:
            return 0

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

        logger.info(
            "Batch upserted KB trials",
            collection=self.collection,
            count=len(points),
        )

        return len(points)

    async def delete_trial(self, point_id: str | UUID) -> None:
        """Delete a single trial document."""
        await self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.PointIdsList(
                points=[str(point_id)],
            ),
        )

    async def delete_by_tune_id(self, tune_id: UUID) -> int:
        """
        Delete all trials for a tune.

        Args:
            tune_id: Tune ID

        Returns:
            Number of points deleted
        """
        # Get count before delete
        count_result = await self.client.count(
            collection_name=self.collection,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="tune_id",
                        match=qmodels.MatchValue(value=str(tune_id)),
                    )
                ]
            ),
        )
        count = count_result.count

        if count > 0:
            await self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="tune_id",
                                match=qmodels.MatchValue(value=str(tune_id)),
                            )
                        ]
                    )
                ),
            )

            logger.info(
                "Deleted KB trials for tune",
                tune_id=str(tune_id),
                count=count,
            )

        return count

    async def search(
        self,
        vector: list[float],
        workspace_id: UUID,
        strategy_name: str,
        objective_type: str,
        filters: Optional[dict] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Search for similar trial documents.

        Args:
            vector: Query embedding vector
            workspace_id: Workspace to search
            strategy_name: Filter by strategy name
            objective_type: Filter by objective type
            filters: Additional filter conditions
            limit: Maximum results

        Returns:
            List of results with score and payload
        """
        # Build base filter conditions
        must_conditions = [
            qmodels.FieldCondition(
                key="doc_type",
                match=qmodels.MatchValue(value=KB_TRIALS_DOC_TYPE),
            ),
            qmodels.FieldCondition(
                key="workspace_id",
                match=qmodels.MatchValue(value=str(workspace_id)),
            ),
            qmodels.FieldCondition(
                key="strategy_name",
                match=qmodels.MatchValue(value=strategy_name),
            ),
            qmodels.FieldCondition(
                key="objective_type",
                match=qmodels.MatchValue(value=objective_type),
            ),
            qmodels.FieldCondition(
                key="is_valid",
                match=qmodels.MatchValue(value=True),
            ),
        ]

        # Add optional filters
        if filters:
            must_conditions.extend(self._build_filter_conditions(filters))

        qdrant_filter = qmodels.Filter(must=must_conditions)

        results = await self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

    async def search_by_filters(
        self,
        workspace_id: UUID,
        strategy_name: str,
        objective_type: str,
        filters: Optional[dict] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Search trials by filters only (no vector similarity).

        Useful for metadata-only fallback when embedding unavailable.

        Args:
            workspace_id: Workspace to search
            strategy_name: Filter by strategy name
            objective_type: Filter by objective type
            filters: Additional filter conditions
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of payloads matching filters
        """
        # Build filter conditions
        must_conditions = [
            qmodels.FieldCondition(
                key="doc_type",
                match=qmodels.MatchValue(value=KB_TRIALS_DOC_TYPE),
            ),
            qmodels.FieldCondition(
                key="workspace_id",
                match=qmodels.MatchValue(value=str(workspace_id)),
            ),
            qmodels.FieldCondition(
                key="strategy_name",
                match=qmodels.MatchValue(value=strategy_name),
            ),
            qmodels.FieldCondition(
                key="objective_type",
                match=qmodels.MatchValue(value=objective_type),
            ),
            qmodels.FieldCondition(
                key="is_valid",
                match=qmodels.MatchValue(value=True),
            ),
        ]

        if filters:
            must_conditions.extend(self._build_filter_conditions(filters))

        qdrant_filter = qmodels.Filter(must=must_conditions)

        results, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qdrant_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "id": result.id,
                "payload": result.payload,
            }
            for result in results
        ]

    def _build_filter_conditions(
        self,
        filters: dict,
    ) -> list[qmodels.FieldCondition]:
        """
        Build Qdrant filter conditions from filter dict.

        Args:
            filters: Dict with filter specifications:
                - require_oos: bool
                - min_sharpe: float
                - min_trades: int
                - max_drawdown: float
                - max_overfit_gap: float
                - regime_tags: list[str]

        Returns:
            List of Qdrant field conditions
        """
        conditions = []

        # Has OOS
        if filters.get("require_oos"):
            conditions.append(
                qmodels.FieldCondition(
                    key="has_oos",
                    match=qmodels.MatchValue(value=True),
                )
            )

        # Performance floors
        if "min_sharpe" in filters and filters["min_sharpe"] is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="sharpe_oos",
                    range=qmodels.Range(gte=filters["min_sharpe"]),
                )
            )

        if "min_trades" in filters and filters["min_trades"] is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="n_trades_oos",
                    range=qmodels.Range(gte=filters["min_trades"]),
                )
            )

        if "max_drawdown" in filters and filters["max_drawdown"] is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="max_dd_frac_oos",
                    range=qmodels.Range(lte=filters["max_drawdown"]),
                )
            )

        if "max_overfit_gap" in filters and filters["max_overfit_gap"] is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key="overfit_gap",
                    range=qmodels.Range(lte=filters["max_overfit_gap"]),
                )
            )

        # Regime tags (any match)
        if filters.get("regime_tags"):
            conditions.append(
                qmodels.FieldCondition(
                    key="regime_tags",
                    match=qmodels.MatchAny(any=filters["regime_tags"]),
                )
            )

        return conditions

    async def count(
        self,
        workspace_id: Optional[UUID] = None,
        strategy_name: Optional[str] = None,
    ) -> int:
        """
        Count trials matching filters.

        Args:
            workspace_id: Optional workspace filter
            strategy_name: Optional strategy filter

        Returns:
            Number of matching trials
        """
        conditions = [
            qmodels.FieldCondition(
                key="doc_type",
                match=qmodels.MatchValue(value=KB_TRIALS_DOC_TYPE),
            )
        ]

        if workspace_id:
            conditions.append(
                qmodels.FieldCondition(
                    key="workspace_id",
                    match=qmodels.MatchValue(value=str(workspace_id)),
                )
            )

        if strategy_name:
            conditions.append(
                qmodels.FieldCondition(
                    key="strategy_name",
                    match=qmodels.MatchValue(value=strategy_name),
                )
            )

        result = await self.client.count(
            collection_name=self.collection,
            count_filter=qmodels.Filter(must=conditions),
        )

        return result.count

    async def get_collection_info(self) -> dict:
        """Get collection information."""
        try:
            info = await self.client.get_collection(self.collection)
            return {
                "name": self.collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as e:
            return {
                "name": self.collection,
                "error": str(e),
            }

    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        collections = await self.client.get_collections()
        return any(c.name == self.collection for c in collections.collections)
