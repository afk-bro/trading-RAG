"""Repository for run_plans table operations."""

import json
from typing import Any, Optional
from uuid import UUID

import structlog

from app.repositories.utils import ensure_json, parse_jsonb_fields

logger = structlog.get_logger(__name__)


class RunPlansRepository:
    """Repository for run_plans table operations."""

    def __init__(self, pool):
        """Initialize with database connection pool."""
        self.pool = pool

    async def get_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[dict[str, Any]]:
        """Get a run plan by idempotency key."""
        query = """
            SELECT id, status, idempotency_key, request_hash,
                   workspace_id, strategy_entity_id, objective_name,
                   created_at
            FROM run_plans
            WHERE idempotency_key = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, idempotency_key)
            if not row:
                return None
            return dict(row)

    async def get_by_request_hash(self, request_hash: str) -> Optional[dict[str, Any]]:
        """Get a run plan by request hash."""
        query = """
            SELECT id, status, idempotency_key, request_hash,
                   workspace_id, strategy_entity_id, objective_name,
                   created_at
            FROM run_plans
            WHERE request_hash = $1
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, request_hash)
            if not row:
                return None
            return dict(row)

    async def create_run_plan(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID],
        objective_name: str,
        n_variants: int,
        plan: dict[str, Any],
        status: str = "pending",
        idempotency_key: Optional[str] = None,
        request_hash: Optional[str] = None,
    ) -> UUID:
        """
        Create a new run plan record.

        Args:
            workspace_id: Workspace ID
            strategy_entity_id: Optional strategy entity ID
            objective_name: Objective function name
            n_variants: Number of variants in plan
            plan: Full plan JSON (inputs, resolved, provenance)
            status: Initial status (default: pending)
            idempotency_key: Client-provided idempotency key
            request_hash: Hash of request for deduplication

        Returns:
            The new run plan ID
        """
        query = """
            INSERT INTO run_plans (
                workspace_id, strategy_entity_id, objective_name,
                n_variants, plan, status, idempotency_key, request_hash
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            plan_id = await conn.fetchval(
                query,
                workspace_id,
                strategy_entity_id,
                objective_name,
                n_variants,
                json.dumps(plan),
                status,
                idempotency_key,
                request_hash,
            )

        logger.info(
            "Created run plan",
            plan_id=str(plan_id),
            workspace_id=str(workspace_id),
            n_variants=n_variants,
            idempotency_key=idempotency_key,
        )

        return plan_id

    async def update_run_plan_status(
        self,
        plan_id: UUID,
        status: str,
    ) -> None:
        """
        Update run plan status.

        When transitioning to 'running', also sets started_at.
        """
        if status == "running":
            query = """
                UPDATE run_plans
                SET status = $2, started_at = NOW()
                WHERE id = $1
            """
        else:
            query = "UPDATE run_plans SET status = $2 WHERE id = $1"

        async with self.pool.acquire() as conn:
            await conn.execute(query, plan_id, status)

        logger.info(
            "Updated run plan status",
            plan_id=str(plan_id),
            status=status,
        )

    async def complete_run_plan(
        self,
        plan_id: UUID,
        status: str,
        n_completed: int,
        n_failed: int,
        n_skipped: int,
        best_backtest_run_id: Optional[UUID] = None,
        best_objective_score: Optional[float] = None,
        error_summary: Optional[str] = None,
    ) -> None:
        """
        Complete a run plan with final aggregates.

        Args:
            plan_id: Run plan ID
            status: Final status (completed, failed, cancelled)
            n_completed: Count of successful variants
            n_failed: Count of failed variants
            n_skipped: Count of skipped variants
            best_backtest_run_id: ID of best variant's backtest_runs row
            best_objective_score: Best variant's objective score
            error_summary: Error message if failed
        """
        query = """
            UPDATE run_plans
            SET status = $2,
                completed_at = NOW(),
                n_completed = $3,
                n_failed = $4,
                n_skipped = $5,
                best_backtest_run_id = $6,
                best_objective_score = $7,
                error_summary = $8
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                plan_id,
                status,
                n_completed,
                n_failed,
                n_skipped,
                best_backtest_run_id,
                best_objective_score,
                error_summary,
            )

        logger.info(
            "Completed run plan",
            plan_id=str(plan_id),
            status=status,
            n_completed=n_completed,
            n_failed=n_failed,
            n_skipped=n_skipped,
        )

    async def get_run_plan(self, plan_id: UUID) -> Optional[dict[str, Any]]:
        """Get a run plan by ID."""
        query = """
            SELECT rp.*,
                   e.name as strategy_name
            FROM run_plans rp
            LEFT JOIN kb_entities e ON rp.strategy_entity_id = e.id
            WHERE rp.id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, plan_id)
            if not row:
                return None

            result = dict(row)
            result["plan"] = ensure_json(result.get("plan"))
            return result

    async def list_run_plans(
        self,
        workspace_id: UUID,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List run plans with optional filtering.

        Note: Does NOT include full plan JSONB to keep response size small.

        Returns:
            Tuple of (plans list, total count)
        """
        conditions = ["rp.workspace_id = $1"]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if status:
            conditions.append(f"rp.status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        count_query = f"SELECT COUNT(*) FROM run_plans rp WHERE {where_clause}"

        # Don't include full plan JSONB in list query - it can be large
        list_query = f"""
            SELECT rp.id, rp.workspace_id, rp.strategy_entity_id,
                   rp.status, rp.objective_name, rp.created_at, rp.started_at, rp.completed_at,
                   rp.n_variants, rp.n_completed, rp.n_failed, rp.n_skipped,
                   rp.best_backtest_run_id, rp.best_objective_score, rp.error_summary,
                   e.name as strategy_name
            FROM run_plans rp
            LEFT JOIN kb_entities e ON rp.strategy_entity_id = e.id
            WHERE {where_clause}
            ORDER BY rp.created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(list_query, *params)

        plans = [dict(row) for row in rows]
        return plans, total

    async def list_runs_for_plan(
        self,
        plan_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List backtest_runs for a run plan (without large blobs).

        Does NOT include equity_curve or trades columns to keep response small.

        Returns:
            Tuple of (runs list, total count)
        """
        count_query = """
            SELECT COUNT(*) FROM backtest_runs WHERE run_plan_id = $1
        """

        # Explicitly list columns, excluding large blobs
        list_query = """
            SELECT id, workspace_id, strategy_entity_id, run_plan_id,
                   variant_index, variant_fingerprint, run_kind,
                   status, objective_score, params, summary,
                   has_equity_curve, has_trades, equity_points, trade_count,
                   skip_reason, error, started_at, completed_at, created_at
            FROM backtest_runs
            WHERE run_plan_id = $1
            ORDER BY variant_index ASC
            LIMIT $2 OFFSET $3
        """

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, plan_id)
            rows = await conn.fetch(list_query, plan_id, limit, offset)

        runs = [parse_jsonb_fields(dict(row), ["params", "summary"]) for row in rows]
        return runs, total
