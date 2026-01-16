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
        candidate_strategy_ids: Optional[list[UUID]] = None,
        candidate_scores: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """
        Record a match run for analytics.

        Args:
            candidate_strategy_ids: Strategy IDs with tag overlap (point-in-time)
            candidate_scores: Detailed scores {strategy_id: {score, matched_tags}}

        Returns:
            Created match_run ID
        """
        query = """
            INSERT INTO match_runs (
                workspace_id, source_type, source_id, video_id,
                intent_signature, intent_json, query_used, filters_applied,
                top_k, total_searched, best_score, avg_top_k_score,
                num_above_threshold, weak_coverage, reason_codes,
                candidate_strategy_ids, candidate_scores
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17
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
                candidate_strategy_ids or [],
                json.dumps(candidate_scores) if candidate_scores else None,
            )
            match_run_id = row["id"]

            logger.info(
                "match_run_recorded",
                match_run_id=str(match_run_id),
                workspace_id=str(workspace_id),
                weak_coverage=weak_coverage,
                reason_codes=reason_codes,
                candidate_count=(
                    len(candidate_strategy_ids) if candidate_strategy_ids else 0
                ),
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

    async def list_weak_coverage_for_cockpit(
        self,
        workspace_id: UUID,
        limit: int = 50,
        since: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        """
        List weak coverage runs shaped for cockpit UI.

        Returns rows with:
            run_id, created_at, intent_signature, script_type,
            weak_reason_codes[], best_score, num_above_threshold,
            candidate_strategy_ids[], query_preview, source_ref,
            coverage_status, priority_score

        Args:
            workspace_id: Workspace to query
            limit: Max results (default 50)
            since: ISO timestamp filter (e.g., "2026-01-01T00:00:00Z")
            status: Filter by coverage_status (open, acknowledged, resolved, all)

        Returns:
            List of cockpit-ready match run dicts sorted by priority_score desc
        """
        query = """
            SELECT
                id as run_id,
                created_at,
                intent_signature,
                filters_applied->>'script_type' as script_type,
                reason_codes as weak_reason_codes,
                best_score,
                num_above_threshold,
                candidate_strategy_ids,
                candidate_scores,
                LEFT(query_used, 120) as query_preview,
                source_type,
                source_id,
                video_id,
                coverage_status
            FROM match_runs
            WHERE workspace_id = $1
              AND weak_coverage = true
        """
        params: list[Any] = [workspace_id]
        param_idx = 2

        # Status filter (default: open only)
        if status and status != "all":
            query += f" AND coverage_status = ${param_idx}::coverage_status"
            params.append(status)
            param_idx += 1
        elif not status:
            # Default to open only
            query += f" AND coverage_status = ${param_idx}::coverage_status"
            params.append("open")
            param_idx += 1

        if since:
            query += f" AND created_at >= ${param_idx}::timestamptz"
            params.append(since)
            param_idx += 1

        query += f" ORDER BY created_at DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for row in rows:
            d = dict(row)
            # Build source_ref for UI display
            source_ref = None
            if d.get("source_type") == "youtube" and d.get("video_id"):
                source_ref = f"youtube:{d['video_id']}"
            elif d.get("source_id"):
                source_ref = f"doc:{d['source_id']}"

            # Parse candidate_scores if it's a string
            candidate_scores = d.get("candidate_scores")
            if isinstance(candidate_scores, str):
                try:
                    candidate_scores = json.loads(candidate_scores)
                except json.JSONDecodeError:
                    candidate_scores = None

            # Compute priority_score
            priority_score = self._compute_priority_score(
                best_score=d["best_score"],
                num_above_threshold=d["num_above_threshold"],
                reason_codes=d["weak_reason_codes"] or [],
                created_at=d["created_at"],
            )

            results.append(
                {
                    "run_id": d["run_id"],
                    "created_at": (
                        d["created_at"].isoformat() if d["created_at"] else None
                    ),
                    "intent_signature": d["intent_signature"],
                    "script_type": d["script_type"],
                    "weak_reason_codes": d["weak_reason_codes"] or [],
                    "best_score": d["best_score"],
                    "num_above_threshold": d["num_above_threshold"],
                    "candidate_strategy_ids": d["candidate_strategy_ids"] or [],
                    "candidate_scores": candidate_scores,
                    "query_preview": d["query_preview"],
                    "source_ref": source_ref,
                    "coverage_status": d["coverage_status"],
                    "priority_score": priority_score,
                }
            )

        # Sort by priority_score descending (most actionable first)
        results.sort(key=lambda x: x["priority_score"], reverse=True)

        return results

    def _compute_priority_score(
        self,
        best_score: Optional[float],
        num_above_threshold: int,
        reason_codes: list[str],
        created_at,
    ) -> float:
        """
        Compute priority score for triage ranking.

        Higher = more urgent/actionable.

        Formula:
        - base: (0.5 - best_score) clamped to [0, 0.5] (worse scores = higher priority)
        - +0.2 if num_above_threshold == 0
        - +0.15 for NO_MATCHES reason code
        - +0.1 for NO_STRONG_MATCHES reason code
        - +0.05 recency bonus (last 24h)
        """
        from datetime import datetime, timezone, timedelta

        score = 0.0

        # Base score from best_score (lower best_score = higher priority)
        if best_score is not None:
            base = max(0.0, min(0.5, 0.5 - best_score))
            score += base
        else:
            score += 0.5  # No score = worst case

        # Bump if no results above threshold
        if num_above_threshold == 0:
            score += 0.2

        # Bump for severe reason codes
        if "NO_MATCHES" in reason_codes:
            score += 0.15
        if "NO_STRONG_MATCHES" in reason_codes:
            score += 0.1

        # Recency bonus (last 24h)
        if created_at:
            now = datetime.now(timezone.utc)
            if isinstance(created_at, datetime):
                age = now - created_at.replace(tzinfo=timezone.utc)
            else:
                age = timedelta(days=365)  # Old if can't parse

            if age < timedelta(hours=24):
                score += 0.05

        return round(score, 3)

    async def update_coverage_status(
        self,
        run_id: UUID,
        workspace_id: UUID,
        status: str,
        note: Optional[str] = None,
        updated_by: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Update coverage status for a match run.

        Args:
            run_id: Match run ID
            workspace_id: Workspace for scoping
            status: New status (open, acknowledged, resolved)
            note: Optional resolution note
            updated_by: Who made the update

        Returns:
            Updated match run dict or None if not found
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Build update based on status
        if status == "acknowledged":
            query = """
                UPDATE match_runs
                SET coverage_status = $3::coverage_status,
                    acknowledged_at = $4,
                    acknowledged_by = $5,
                    resolution_note = COALESCE($6, resolution_note)
                WHERE id = $1 AND workspace_id = $2
                RETURNING id, coverage_status, acknowledged_at, acknowledged_by,
                          resolved_at, resolved_by, resolution_note
            """
            params = [run_id, workspace_id, status, now, updated_by, note]
        elif status == "resolved":
            query = """
                UPDATE match_runs
                SET coverage_status = $3::coverage_status,
                    resolved_at = $4,
                    resolved_by = $5,
                    resolution_note = COALESCE($6, resolution_note)
                WHERE id = $1 AND workspace_id = $2
                RETURNING id, coverage_status, acknowledged_at, acknowledged_by,
                          resolved_at, resolved_by, resolution_note
            """
            params = [run_id, workspace_id, status, now, updated_by, note]
        elif status == "open":
            # Reopen: clear timestamps but keep note
            query = """
                UPDATE match_runs
                SET coverage_status = $3::coverage_status,
                    acknowledged_at = NULL,
                    acknowledged_by = NULL,
                    resolved_at = NULL,
                    resolved_by = NULL
                WHERE id = $1 AND workspace_id = $2
                RETURNING id, coverage_status, acknowledged_at, acknowledged_by,
                          resolved_at, resolved_by, resolution_note
            """
            params = [run_id, workspace_id, status]
        else:
            raise ValueError(f"Invalid status: {status}")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        if row:
            logger.info(
                "coverage_status_updated",
                run_id=str(run_id),
                status=status,
                updated_by=updated_by,
            )
            return dict(row)

        return None
