"""Repository for strategy registry operations."""

import json
import re
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


def _json_serial(obj: Any) -> str:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def slugify(name: str) -> str:
    """Convert name to URL-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:100]


class StrategyRepository:
    """Repository for strategies table operations."""

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
        name: str,
        engine: str = "pine",
        description: Optional[str] = None,
        source_ref: Optional[dict] = None,
        status: str = "draft",
        risk_level: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> dict:
        """
        Create a new strategy.

        Returns:
            Created strategy dict with id
        """
        slug = slugify(name)

        # Ensure unique slug within workspace
        base_slug = slug
        counter = 1
        async with self.pool.acquire() as conn:
            while True:
                exists = await conn.fetchval(
                    """
                    SELECT 1 FROM strategies
                    WHERE workspace_id = $1 AND slug = $2
                    """,
                    workspace_id,
                    slug,
                )
                if not exists:
                    break
                slug = f"{base_slug}-{counter}"
                counter += 1

            row = await conn.fetchrow(
                """
                INSERT INTO strategies (
                    workspace_id, name, slug, description, engine,
                    source_ref, status, risk_level, tags
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING *
                """,
                workspace_id,
                name,
                slug,
                description,
                engine,
                json.dumps(source_ref) if source_ref else "{}",
                status,
                risk_level,
                json.dumps(tags) if tags else "{}",
            )

        logger.info(
            "strategy_created",
            strategy_id=str(row["id"]),
            workspace_id=str(workspace_id),
            name=name,
            slug=slug,
            engine=engine,
        )

        return self._row_to_dict(row)

    async def get_by_id(
        self,
        strategy_id: UUID,
        workspace_id: UUID,
    ) -> Optional[dict]:
        """
        Get strategy by ID.

        Args:
            strategy_id: Strategy UUID
            workspace_id: Workspace UUID for scoping

        Returns:
            Strategy dict or None if not found
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM strategies
                WHERE id = $1 AND workspace_id = $2
                """,
                strategy_id,
                workspace_id,
            )

        return self._row_to_dict(row) if row else None

    async def list_strategies(
        self,
        workspace_id: UUID,
        engine: Optional[str] = None,
        status: Optional[str] = None,
        review_status: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        List strategies with filters.

        Args:
            workspace_id: Workspace to query
            engine: Filter by engine
            status: Filter by status
            review_status: Filter by review_status
            q: Search name/tags
            limit: Max results
            offset: Pagination offset

        Returns:
            (list of strategies, total count)
        """
        # Build WHERE clause
        conditions = ["workspace_id = $1"]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if engine:
            conditions.append(f"engine = ${param_idx}")
            params.append(engine)
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if review_status:
            conditions.append(f"review_status = ${param_idx}")
            params.append(review_status)
            param_idx += 1

        if q:
            # Search name and tags
            conditions.append(
                f"(name ILIKE ${param_idx} OR tags::text ILIKE ${param_idx})"
            )
            params.append(f"%{q}%")
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"SELECT COUNT(*) FROM strategies WHERE {where_clause}"

        # Data query
        data_query = f"""
            SELECT * FROM strategies
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(data_query, *params)

        return [self._row_to_dict(row) for row in rows], total or 0

    async def update(
        self,
        strategy_id: UUID,
        workspace_id: UUID,
        **updates,
    ) -> Optional[dict]:
        """
        Update strategy fields.

        Args:
            strategy_id: Strategy UUID
            workspace_id: Workspace UUID for scoping
            **updates: Fields to update

        Returns:
            Updated strategy dict or None if not found
        """
        if not updates:
            return await self.get_by_id(strategy_id, workspace_id)

        # Build SET clause
        set_parts = []
        params: list[Any] = []
        param_idx = 1

        # Map field names and handle JSONB
        for field, value in updates.items():
            if value is None and field not in ("description", "risk_level"):
                continue

            if field == "name":
                set_parts.append(f"name = ${param_idx}")
                params.append(value)
                param_idx += 1
                # Also update slug
                set_parts.append(f"slug = ${param_idx}")
                params.append(slugify(value))
                param_idx += 1
            elif field in ("source_ref", "tags", "backtest_summary"):
                set_parts.append(f"{field} = ${param_idx}")
                params.append(json.dumps(value, default=_json_serial) if value else None)
                param_idx += 1
            elif field in (
                "description",
                "status",
                "review_status",
                "risk_level",
                "engine",
            ):
                set_parts.append(f"{field} = ${param_idx}")
                params.append(value)
                param_idx += 1

        if not set_parts:
            return await self.get_by_id(strategy_id, workspace_id)

        set_parts.append("updated_at = NOW()")

        set_clause = ", ".join(set_parts)
        params.extend([strategy_id, workspace_id])

        query = f"""
            UPDATE strategies
            SET {set_clause}
            WHERE id = ${param_idx} AND workspace_id = ${param_idx + 1}
            RETURNING *
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row:
            logger.info(
                "strategy_updated",
                strategy_id=str(strategy_id),
                fields=list(updates.keys()),
            )

        return self._row_to_dict(row) if row else None

    async def find_candidates_by_tags(
        self,
        workspace_id: UUID,
        intent_tags: dict,
        limit: int = 10,
    ) -> list[dict]:
        """
        Find strategies whose tags overlap with intent.

        Args:
            workspace_id: Workspace to query
            intent_tags: MatchIntent-style tags dict
            limit: Max results

        Returns:
            List of {strategy_id, name, score, matched_tags}
        """
        # Extract all tags from intent
        all_tags = []
        for field in [
            "strategy_archetypes",
            "indicators",
            "timeframe_buckets",
            "topics",
            "risk_terms",
        ]:
            all_tags.extend(intent_tags.get(field, []))

        if not all_tags:
            return []

        # Query strategies with overlapping tags
        # Use JSONB containment and text search
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, slug, tags, status, review_status,
                       backtest_summary
                FROM strategies
                WHERE workspace_id = $1
                  AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 100
                """,
                workspace_id,
            )

        # Score each strategy by tag overlap
        candidates = []
        tag_set = set(t.lower() for t in all_tags)

        for row in rows:
            raw_tags = row["tags"]
            if isinstance(raw_tags, str):
                try:
                    tags = json.loads(raw_tags)
                except json.JSONDecodeError:
                    tags = {}
            elif isinstance(raw_tags, dict):
                tags = raw_tags
            else:
                tags = {}
            strategy_tags = set()
            for field in [
                "strategy_archetypes",
                "indicators",
                "timeframe_buckets",
                "topics",
                "risk_terms",
            ]:
                strategy_tags.update(t.lower() for t in tags.get(field, []))

            matched = tag_set & strategy_tags
            if matched:
                score = len(matched) / len(tag_set) if tag_set else 0
                candidates.append(
                    {
                        "strategy_id": row["id"],
                        "name": row["name"],
                        "score": round(score, 2),
                        "matched_tags": sorted(matched),
                    }
                )

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:limit]

    def _row_to_dict(self, row) -> dict:
        """Convert asyncpg Row to dict with JSONB parsing."""
        if not row:
            return {}

        d = dict(row)

        # Parse JSONB fields
        for field in ("source_ref", "tags", "backtest_summary"):
            if field in d and d[field]:
                if isinstance(d[field], str):
                    try:
                        d[field] = json.loads(d[field])
                    except json.JSONDecodeError:
                        d[field] = {}
            elif field in d:
                d[field] = {} if field != "backtest_summary" else None

        return d
