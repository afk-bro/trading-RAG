"""Repository for match run persistence."""

import json
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class MatchRunRepository:
    """Repository for match_runs table operations."""

    def __init__(self, pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool
        """
        self.pool = pool

    async def record_match_run(
        self,
        workspace_id: UUID,
        source_type: str,
        intent_signature: str,
        query_used: str,
        filters_applied: dict[str, Any],
        top_k: int,
        total_searched: int,
        best_score: Optional[float],
        avg_top_k_score: Optional[float],
        num_above_threshold: int,
        weak_coverage: bool,
        reason_codes: list[str],
        source_id: Optional[UUID] = None,
        video_id: Optional[str] = None,
        intent_json: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """
        Record a match run for analytics.

        Returns:
            Created match_run ID
        """
        query = """
            INSERT INTO match_runs (
                workspace_id, source_type, source_id, video_id,
                intent_signature, intent_json, query_used, filters_applied,
                top_k, total_searched, best_score, avg_top_k_score,
                num_above_threshold, weak_coverage, reason_codes
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            )
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                source_type,
                source_id,
                video_id,
                intent_signature,
                json.dumps(intent_json) if intent_json else None,
                query_used,
                json.dumps(filters_applied),
                top_k,
                total_searched,
                best_score,
                avg_top_k_score,
                num_above_threshold,
                weak_coverage,
                reason_codes,
            )
            match_run_id = row["id"]

            logger.info(
                "match_run_recorded",
                match_run_id=str(match_run_id),
                workspace_id=str(workspace_id),
                weak_coverage=weak_coverage,
                reason_codes=reason_codes,
            )

            return match_run_id

    async def list_coverage_gaps(
        self,
        workspace_id: UUID,
        weak_only: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        List match runs with coverage gap info.

        Args:
            workspace_id: Workspace to query
            weak_only: If True, only return weak coverage runs
            limit: Max results
            offset: Pagination offset

        Returns:
            (list of match runs, total count)
        """
        # Count query
        count_query = """
            SELECT COUNT(*) as total
            FROM match_runs
            WHERE workspace_id = $1
        """
        count_params = [workspace_id]

        if weak_only:
            count_query += " AND weak_coverage = true"

        # Data query
        data_query = """
            SELECT
                id, workspace_id, created_at, source_type, source_id, video_id,
                intent_signature, intent_json, query_used, filters_applied,
                top_k, total_searched, best_score, avg_top_k_score,
                num_above_threshold, weak_coverage, reason_codes
            FROM match_runs
            WHERE workspace_id = $1
        """
        data_params: list = [workspace_id]

        if weak_only:
            data_query += " AND weak_coverage = true"

        data_query += " ORDER BY created_at DESC LIMIT $2 OFFSET $3"
        data_params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            count_row = await conn.fetchrow(count_query, *count_params)
            total = count_row["total"] if count_row else 0

            rows = await conn.fetch(data_query, *data_params)

        return [dict(row) for row in rows], total

    async def get_intent_frequency(
        self,
        workspace_id: UUID,
        days: int = 30,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get most frequent intents with weak coverage.

        Useful for identifying recurring gaps.

        Args:
            workspace_id: Workspace to query
            days: Lookback period
            limit: Max results

        Returns:
            List of {intent_signature, count, weak_count, sample_query}
        """
        query = (
            """
            SELECT
                intent_signature,
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE weak_coverage = true) as weak_count,
                (array_agg(query_used ORDER BY created_at DESC))[1] as sample_query,
                (array_agg(intent_json ORDER BY created_at DESC))[1] as sample_intent
            FROM match_runs
            WHERE workspace_id = $1
              AND created_at > NOW() - INTERVAL '%s days'
            GROUP BY intent_signature
            HAVING COUNT(*) FILTER (WHERE weak_coverage = true) > 0
            ORDER BY weak_count DESC, total_count DESC
            LIMIT $2
        """
            % days
        )  # Safe: days is int, not user input

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, limit)

        return [dict(row) for row in rows]
