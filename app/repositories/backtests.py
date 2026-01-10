"""Repository for backtest runs persistence."""

import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


class BacktestRepository:
    """Repository for backtest_runs table operations."""

    def __init__(self, pool):
        """Initialize with database connection pool."""
        self.pool = pool

    async def create_run(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        strategy_spec_id: UUID,
        spec_version: int,
        params: dict[str, Any],
        engine: str,
        dataset_meta: dict[str, Any],
    ) -> UUID:
        """
        Create a new backtest run record (status=running).

        Returns:
            The new run ID
        """
        query = """
            INSERT INTO backtest_runs (
                workspace_id, strategy_entity_id, strategy_spec_id, spec_version,
                params, engine, dataset_meta, status, started_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, 'running', NOW())
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            run_id = await conn.fetchval(
                query,
                workspace_id,
                strategy_entity_id,
                strategy_spec_id,
                spec_version,
                json.dumps(params),
                engine,
                json.dumps(dataset_meta),
            )

        logger.info(
            "Created backtest run",
            run_id=str(run_id),
            strategy_entity_id=str(strategy_entity_id),
        )

        return run_id

    async def update_run_completed(
        self,
        run_id: UUID,
        summary: dict[str, Any],
        equity_curve: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        warnings: list[str],
    ) -> None:
        """Update run with completed results."""
        query = """
            UPDATE backtest_runs
            SET status = 'completed',
                summary = $2,
                equity_curve = $3,
                trades = $4,
                warnings = $5,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run_id,
                json.dumps(summary),
                json.dumps(equity_curve),
                json.dumps(trades),
                json.dumps(warnings),
            )

        logger.info("Updated backtest run as completed", run_id=str(run_id))

    async def update_run_failed(
        self,
        run_id: UUID,
        error: str,
        warnings: Optional[list[str]] = None,
    ) -> None:
        """Update run with failure status."""
        query = """
            UPDATE backtest_runs
            SET status = 'failed',
                error = $2,
                warnings = $3,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run_id,
                error,
                json.dumps(warnings or []),
            )

        logger.info("Updated backtest run as failed", run_id=str(run_id), error=error)

    async def get_run(self, run_id: UUID) -> Optional[dict[str, Any]]:
        """Get a backtest run by ID."""
        query = """
            SELECT r.*,
                   e.name as strategy_name
            FROM backtest_runs r
            LEFT JOIN kb_entities e ON r.strategy_entity_id = e.id
            WHERE r.id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, run_id)
            if not row:
                return None

            result = dict(row)

            # Parse JSONB fields
            for field in [
                "params",
                "dataset_meta",
                "summary",
                "equity_curve",
                "trades",
                "warnings",
            ]:
                if result.get(field) and isinstance(result[field], str):
                    result[field] = json.loads(result[field])

            return result

    async def list_runs(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List backtest runs with optional filtering.

        Returns:
            Tuple of (runs list, total count)
        """
        conditions = ["r.workspace_id = $1"]
        params = [workspace_id]
        param_idx = 2

        if strategy_entity_id:
            conditions.append(f"r.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if status:
            conditions.append(f"r.status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Get total count
        count_query = f"""
            SELECT COUNT(*) FROM backtest_runs r WHERE {where_clause}
        """

        # Get runs (without full equity_curve/trades for list view)
        list_query = f"""
            SELECT r.id, r.created_at, r.strategy_entity_id, r.spec_version,
                   r.status, r.params, r.engine, r.dataset_meta, r.summary,
                   r.warnings, r.error, r.started_at, r.completed_at,
                   e.name as strategy_name
            FROM backtest_runs r
            LEFT JOIN kb_entities e ON r.strategy_entity_id = e.id
            WHERE {where_clause}
            ORDER BY r.created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(list_query, *params)

        runs = []
        for row in rows:
            run = dict(row)
            # Parse JSONB fields
            for field in ["params", "dataset_meta", "summary", "warnings"]:
                if run.get(field) and isinstance(run[field], str):
                    run[field] = json.loads(run[field])
            runs.append(run)

        return runs, total

    async def delete_run(self, run_id: UUID) -> bool:
        """Delete a backtest run."""
        query = "DELETE FROM backtest_runs WHERE id = $1 RETURNING id"

        async with self.pool.acquire() as conn:
            deleted = await conn.fetchval(query, run_id)

        return deleted is not None

    # =========================================================================
    # Variant Run Methods (for RunOrchestrator persistence)
    # =========================================================================

    async def create_variant_run(
        self,
        run_plan_id: UUID,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        variant_index: int,
        variant_fingerprint: str,
        params: dict[str, Any],
        dataset_meta: dict[str, Any],
        run_kind: str = "test_variant",
    ) -> UUID:
        """
        Create a new backtest run for a plan variant.

        Args:
            run_plan_id: Parent run plan ID
            workspace_id: Workspace ID
            strategy_entity_id: Strategy entity ID
            variant_index: 0-based index within run plan
            variant_fingerprint: Hash of canonical params
            params: Variant parameters
            dataset_meta: Dataset metadata
            run_kind: Type of run (default: test_variant)

        Returns:
            The new run ID
        """
        query = """
            INSERT INTO backtest_runs (
                run_plan_id, workspace_id, strategy_entity_id,
                variant_index, variant_fingerprint, run_kind,
                params, dataset_meta, status, started_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'running', NOW())
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            run_id = await conn.fetchval(
                query,
                run_plan_id,
                workspace_id,
                strategy_entity_id,
                variant_index,
                variant_fingerprint,
                run_kind,
                json.dumps(params),
                json.dumps(dataset_meta),
            )

        logger.info(
            "Created variant run",
            run_id=str(run_id),
            run_plan_id=str(run_plan_id),
            variant_index=variant_index,
        )

        return run_id

    async def update_variant_completed(
        self,
        run_id: UUID,
        summary: dict[str, Any],
        equity_curve: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        objective_score: Optional[float],
        has_equity_curve: bool,
        has_trades: bool,
        equity_points: Optional[int],
        trade_count: Optional[int],
        warnings: Optional[list[str]] = None,
    ) -> None:
        """Update variant run with completed results."""
        query = """
            UPDATE backtest_runs
            SET status = 'completed',
                summary = $2,
                equity_curve = $3,
                trades = $4,
                objective_score = $5,
                has_equity_curve = $6,
                has_trades = $7,
                equity_points = $8,
                trade_count = $9,
                warnings = $10,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                run_id,
                json.dumps(summary),
                json.dumps(equity_curve),
                json.dumps(trades),
                objective_score,
                has_equity_curve,
                has_trades,
                equity_points,
                trade_count,
                json.dumps(warnings or []),
            )

        logger.info("Updated variant run as completed", run_id=str(run_id))

    async def update_variant_skipped(
        self,
        run_id: UUID,
        skip_reason: str,
    ) -> None:
        """Update variant run as skipped."""
        query = """
            UPDATE backtest_runs
            SET status = 'skipped',
                skip_reason = $2,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, run_id, skip_reason)

        logger.info(
            "Updated variant run as skipped",
            run_id=str(run_id),
            skip_reason=skip_reason,
        )

    async def update_variant_failed(
        self,
        run_id: UUID,
        error: str,
    ) -> None:
        """Update variant run as failed."""
        query = """
            UPDATE backtest_runs
            SET status = 'failed',
                error = $2,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, run_id, error)

        logger.info(
            "Updated variant run as failed",
            run_id=str(run_id),
            error=error,
        )


class TuneRepository:
    """Repository for backtest_tunes and backtest_tune_runs table operations."""

    def __init__(self, pool):
        """Initialize with database connection pool."""
        self.pool = pool

    async def create_tune(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        strategy_spec_id: Optional[UUID],
        search_type: str,
        n_trials: int,
        seed: Optional[int],
        param_space: dict[str, Any],
        objective_metric: str,
        min_trades: int,
        oos_ratio: Optional[float] = None,
        objective_type: str = "sharpe",
        objective_params: Optional[dict[str, Any]] = None,
        gates: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """
        Create a new tune record (status=queued).

        Args:
            gates: Gate policy snapshot (e.g., {"max_drawdown_pct": 20, "min_trades": 10, "evaluated_on": "oos"}).  # noqa: E501
                   Persisted once at creation for audit/reproducibility.

        Returns:
            The new tune ID
        """
        query = """
            INSERT INTO backtest_tunes (
                workspace_id, strategy_entity_id, strategy_spec_id,
                search_type, n_trials, seed, param_space,
                objective_metric, min_trades, oos_ratio,
                objective_type, objective_params, gates, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, 'queued')
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            tune_id = await conn.fetchval(
                query,
                workspace_id,
                strategy_entity_id,
                strategy_spec_id,
                search_type,
                n_trials,
                seed,
                json.dumps(param_space),
                objective_metric,
                min_trades,
                oos_ratio,
                objective_type,
                json.dumps(objective_params) if objective_params else None,
                json.dumps(gates) if gates else None,
            )

        logger.info(
            "Created tune",
            tune_id=str(tune_id),
            strategy_entity_id=str(strategy_entity_id),
            search_type=search_type,
            n_trials=n_trials,
        )

        return tune_id

    async def update_tune_status(
        self,
        tune_id: UUID,
        status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update tune status."""
        updates = ["status = $2"]
        params = [tune_id, status]
        param_idx = 3

        if started_at:
            updates.append(f"started_at = ${param_idx}")
            params.append(started_at)
            param_idx += 1

        if completed_at:
            updates.append(f"completed_at = ${param_idx}")
            params.append(completed_at)
            param_idx += 1

        if error:
            updates.append(f"error = ${param_idx}")
            params.append(error)
            param_idx += 1

        query = f"UPDATE backtest_tunes SET {', '.join(updates)} WHERE id = $1"

        async with self.pool.acquire() as conn:
            await conn.execute(query, *params)

    async def update_tune_progress(self, tune_id: UUID, trials_completed: int) -> None:
        """Update tune progress count."""
        query = "UPDATE backtest_tunes SET trials_completed = $2 WHERE id = $1"

        async with self.pool.acquire() as conn:
            await conn.execute(query, tune_id, trials_completed)

    async def complete_tune(
        self,
        tune_id: UUID,
        best_run_id: Optional[UUID],
        best_score: Optional[float],
        best_params: Optional[dict[str, Any]],
        leaderboard: list[dict[str, Any]],
        trials_completed: int,
    ) -> None:
        """
        Mark tune as completed with results.

        IMPORTANT: Does NOT overwrite 'canceled' status.
        A canceled tune remains canceled even if all trials finish.
        Best results are still persisted for partial results visibility.
        """
        async with self.pool.acquire() as conn:
            # First, check if tune is canceled - if so, only update results, not status
            current = await conn.fetchrow(
                "SELECT status FROM backtest_tunes WHERE id = $1", tune_id
            )

            if current and current["status"] == "canceled":
                # Tune was canceled - preserve status but update best results
                query = """
                    UPDATE backtest_tunes
                    SET best_run_id = $2,
                        best_score = $3,
                        best_params = $4,
                        leaderboard = $5,
                        trials_completed = $6
                    WHERE id = $1
                """
                logger.info(
                    "Tune was canceled, preserving status but updating best results",
                    tune_id=str(tune_id),
                    best_run_id=str(best_run_id) if best_run_id else None,
                )
            else:
                # Normal completion - set status to completed
                query = """
                    UPDATE backtest_tunes
                    SET status = 'completed',
                        best_run_id = $2,
                        best_score = $3,
                        best_params = $4,
                        leaderboard = $5,
                        trials_completed = $6,
                        completed_at = NOW()
                    WHERE id = $1
                """

            await conn.execute(
                query,
                tune_id,
                best_run_id,
                best_score,
                json.dumps(best_params) if best_params else None,
                json.dumps(leaderboard),
                trials_completed,
            )

        if not (current and current["status"] == "canceled"):
            logger.info(
                "Completed tune",
                tune_id=str(tune_id),
                best_run_id=str(best_run_id) if best_run_id else None,
                trials_completed=trials_completed,
            )

    async def create_tune_run(
        self,
        tune_id: UUID,
        trial_index: int,
        params: dict[str, Any],
        status: str = "queued",
    ) -> None:
        """Create a tune_run record (placeholder before actual run)."""
        query = """
            INSERT INTO backtest_tune_runs (tune_id, trial_index, params, status)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (tune_id, trial_index) DO UPDATE SET params = $3, status = $4
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, tune_id, trial_index, json.dumps(params), status)

    async def update_tune_run_status(
        self,
        tune_id: UUID,
        trial_index: int,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update tune_run status by trial index."""
        # Note: We don't have run_id yet, so update by tune_id + trial_index
        query = """
            UPDATE backtest_tune_runs
            SET status = $3
            WHERE tune_id = $1 AND trial_index = $2
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, tune_id, trial_index, status)

    async def update_tune_run_result(
        self,
        tune_id: UUID,
        trial_index: int,
        run_id: Optional[UUID],
        score: Optional[float],
        status: str,
        skip_reason: Optional[str] = None,
        failed_reason: Optional[str] = None,
        score_is: Optional[float] = None,
        score_oos: Optional[float] = None,
        metrics_is: Optional[dict[str, Any]] = None,
        metrics_oos: Optional[dict[str, Any]] = None,
        objective_score: Optional[float] = None,
    ) -> None:
        """Update tune_run with result after backtest completes."""
        query = """
            UPDATE backtest_tune_runs
            SET run_id = $3, score = $4, status = $5, skip_reason = $6,
                failed_reason = $7, score_is = $8, score_oos = $9,
                metrics_is = $10, metrics_oos = $11, objective_score = $12,
                finished_at = NOW()
            WHERE tune_id = $1 AND trial_index = $2
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                tune_id,
                trial_index,
                run_id,
                score,
                status,
                skip_reason,
                failed_reason,
                score_is,
                score_oos,
                json.dumps(metrics_is) if metrics_is else None,
                json.dumps(metrics_oos) if metrics_oos else None,
                objective_score,
            )

    async def start_tune_run(
        self,
        tune_id: UUID,
        trial_index: int,
    ) -> None:
        """Mark a tune_run as started (running)."""
        query = """
            UPDATE backtest_tune_runs
            SET status = 'running', started_at = NOW()
            WHERE tune_id = $1 AND trial_index = $2
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, tune_id, trial_index)

    async def cancel_tune(self, tune_id: UUID) -> bool:
        """
        Cancel a tune - set status to canceled and skip remaining queued runs.

        Returns:
            True if tune was canceled, False if not found or already terminal
        """
        # Only cancel if queued or running
        query = """
            UPDATE backtest_tunes
            SET status = 'canceled', completed_at = NOW()
            WHERE id = $1 AND status IN ('queued', 'running')
            RETURNING id
        """

        async with self.pool.acquire() as conn:
            canceled = await conn.fetchval(query, tune_id)

            if canceled:
                # Mark all queued runs as skipped
                await conn.execute(
                    """
                    UPDATE backtest_tune_runs
                    SET status = 'skipped', skip_reason = 'canceled', finished_at = NOW()
                    WHERE tune_id = $1 AND status = 'queued'
                    """,
                    tune_id,
                )
                logger.info("Canceled tune", tune_id=str(tune_id))

        return canceled is not None

    async def get_queued_trial_indices(self, tune_id: UUID) -> list[int]:
        """Get list of queued trial indices for a tune."""
        query = """
            SELECT trial_index FROM backtest_tune_runs
            WHERE tune_id = $1 AND status = 'queued'
            ORDER BY trial_index
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, tune_id)

        return [row["trial_index"] for row in rows]

    async def count_running_trials(self, tune_id: UUID) -> int:
        """Count currently running trials for a tune."""
        query = """
            SELECT COUNT(*) FROM backtest_tune_runs
            WHERE tune_id = $1 AND status = 'running'
        """

        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, tune_id)

    async def get_tune(self, tune_id: UUID) -> Optional[dict[str, Any]]:
        """Get a tune by ID."""
        query = """
            SELECT t.*,
                   e.name as strategy_name
            FROM backtest_tunes t
            LEFT JOIN kb_entities e ON t.strategy_entity_id = e.id
            WHERE t.id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, tune_id)
            if not row:
                return None

            result = dict(row)

            # Parse JSONB fields
            for field in ["param_space", "leaderboard", "objective_params", "gates"]:
                if result.get(field) and isinstance(result[field], str):
                    result[field] = json.loads(result[field])

            return result

    async def get_tune_status_counts(self, tune_id: UUID) -> dict[str, int]:
        """Get status counts for tune runs."""
        query = """
            SELECT status, COUNT(*) as count
            FROM backtest_tune_runs
            WHERE tune_id = $1
            GROUP BY status
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, tune_id)

        counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "skipped": 0}
        for row in rows:
            if row["status"] in counts:
                counts[row["status"]] = row["count"]

        return counts

    async def list_tunes(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
        status: Optional[str] = None,
        valid_only: bool = False,
        objective_type: Optional[str] = None,
        oos_enabled: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List tunes with optional filtering.

        Args:
            valid_only: If True, only return tunes with best_run_id IS NOT NULL
            objective_type: Filter by objective type (e.g., 'sharpe', 'sharpe_dd_penalty')
            oos_enabled: If True, filter to tunes with OOS split; if False, filter to no split

        Returns:
            Tuple of (tunes list, total count)
        """
        conditions = ["t.workspace_id = $1"]
        params = [workspace_id]
        param_idx = 2

        if strategy_entity_id:
            conditions.append(f"t.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if status:
            conditions.append(f"t.status = ${param_idx}")
            params.append(status)
            param_idx += 1

        if valid_only:
            conditions.append("t.best_run_id IS NOT NULL")

        if objective_type:
            conditions.append(f"t.objective_type = ${param_idx}")
            params.append(objective_type)
            param_idx += 1

        if oos_enabled is not None:
            if oos_enabled:
                conditions.append("t.oos_ratio IS NOT NULL AND t.oos_ratio > 0")
            else:
                conditions.append("(t.oos_ratio IS NULL OR t.oos_ratio = 0)")

        where_clause = " AND ".join(conditions)

        count_query = f"SELECT COUNT(*) FROM backtest_tunes t WHERE {where_clause}"

        list_query = f"""
            SELECT t.id, t.created_at, t.strategy_entity_id, t.search_type,
                   t.n_trials, t.seed, t.objective_metric, t.min_trades,
                   t.status, t.trials_completed, t.best_run_id, t.best_score,
                   t.best_params, t.leaderboard, t.started_at, t.completed_at, t.error,
                   t.oos_ratio, t.objective_type, t.objective_params, t.gates,
                   e.name as strategy_name
            FROM backtest_tunes t
            LEFT JOIN kb_entities e ON t.strategy_entity_id = e.id
            WHERE {where_clause}
            ORDER BY t.created_at DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(list_query, *params)

        tunes = []
        for row in rows:
            tune = dict(row)
            # Parse JSONB fields
            for field in ["leaderboard", "objective_params", "gates"]:
                if tune.get(field) and isinstance(tune[field], str):
                    tune[field] = json.loads(tune[field])
            tunes.append(tune)

        return tunes, total

    async def get_leaderboard(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
        valid_only: bool = True,
        objective_type: Optional[str] = None,
        oos_enabled: Optional[bool] = None,
        include_canceled: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get leaderboard data: tunes with their best run metrics.

        Joins backtest_tunes with backtest_tune_runs via best_run_id to get
        the winning trial's objective_score, metrics_oos (return, sharpe, DD,
        trades), and computes overfit gap (score_is - score_oos).

        Args:
            valid_only: Only include tunes with best_run_id (default True for leaderboard)
            include_canceled: Include canceled tunes (default False). Note: canceled
                tunes CAN have best_run_id if trials completed before cancellation.
                With valid_only=True + include_canceled=True, you get "canceled but
                had valid results" tunes. With valid_only=False + include_canceled=True,
                you get everything including incomplete canceled tunes.

        Returns:
            Tuple of (leaderboard entries, total count)

        Ordering:
            Primary: objective_score DESC (or fallback: score_oos, best_score)
            Tie-breakers: created_at DESC, id ASC (deterministic)
        """
        conditions = ["t.workspace_id = $1"]
        params = [workspace_id]
        param_idx = 2

        if strategy_entity_id:
            conditions.append(f"t.strategy_entity_id = ${param_idx}")
            params.append(strategy_entity_id)
            param_idx += 1

        if valid_only:
            conditions.append("t.best_run_id IS NOT NULL")

        if not include_canceled:
            conditions.append("t.status != 'canceled'")

        if objective_type:
            conditions.append(f"t.objective_type = ${param_idx}")
            params.append(objective_type)
            param_idx += 1

        if oos_enabled is not None:
            if oos_enabled:
                conditions.append("t.oos_ratio IS NOT NULL AND t.oos_ratio > 0")
            else:
                conditions.append("(t.oos_ratio IS NULL OR t.oos_ratio = 0)")

        where_clause = " AND ".join(conditions)

        count_query = f"SELECT COUNT(*) FROM backtest_tunes t WHERE {where_clause}"

        # Join with best run to get metrics
        list_query = f"""
            SELECT t.id, t.created_at, t.strategy_entity_id, t.objective_metric,
                   t.objective_type, t.objective_params, t.oos_ratio, t.gates,
                   t.status, t.best_run_id, t.best_score,
                   e.name as strategy_name,
                   -- Best run metrics from tune_runs
                   tr.objective_score as best_objective_score,
                   tr.score_is, tr.score_oos,
                   tr.metrics_oos as best_metrics_oos
            FROM backtest_tunes t
            LEFT JOIN kb_entities e ON t.strategy_entity_id = e.id
            LEFT JOIN backtest_tune_runs tr ON t.id = tr.tune_id AND tr.run_id = t.best_run_id
            WHERE {where_clause}
            ORDER BY COALESCE(tr.objective_score, tr.score_oos, t.best_score) DESC NULLS LAST,
                     t.created_at DESC,
                     t.id ASC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(list_query, *params)

        entries = []
        for row in rows:
            entry = dict(row)

            # Parse JSONB fields
            for field in ["objective_params", "gates", "best_metrics_oos"]:
                if entry.get(field) and isinstance(entry[field], str):
                    entry[field] = json.loads(entry[field])

            # Compute overfit gap if both scores available
            score_is = entry.get("score_is")
            score_oos = entry.get("score_oos")
            if score_is is not None and score_oos is not None:
                entry["overfit_gap"] = round(score_is - score_oos, 4)
            else:
                entry["overfit_gap"] = None

            entries.append(entry)

        return entries, total

    async def list_tune_runs(
        self,
        tune_id: UUID,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List tune runs with optional status filter.

        Returns:
            Tuple of (runs list, total count)
        """
        conditions = ["tr.tune_id = $1"]
        params = [tune_id]
        param_idx = 2

        if status:
            conditions.append(f"tr.status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        count_query = f"SELECT COUNT(*) FROM backtest_tune_runs tr WHERE {where_clause}"

        # Order by: objective_score (composite) → score_oos (OOS) → score (raw)
        list_query = f"""
            SELECT tr.tune_id, tr.run_id, tr.trial_index, tr.params,
                   tr.score, tr.score_is, tr.score_oos, tr.objective_score,
                   tr.status, tr.skip_reason, tr.failed_reason,
                   tr.metrics_is, tr.metrics_oos,
                   tr.started_at, tr.finished_at, tr.created_at
            FROM backtest_tune_runs tr
            WHERE {where_clause}
            ORDER BY COALESCE(tr.objective_score, tr.score_oos, tr.score) DESC NULLS LAST, tr.trial_index ASC  # noqa: E501
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(list_query, *params)

        runs = []
        for row in rows:
            run = dict(row)
            # Parse JSONB fields
            for field in ["params", "metrics_is", "metrics_oos"]:
                if run.get(field) and isinstance(run[field], str):
                    run[field] = json.loads(run[field])
            runs.append(run)

        return runs, total

    async def delete_tune(self, tune_id: UUID) -> bool:
        """Delete a tune and its runs (cascade)."""
        query = "DELETE FROM backtest_tunes WHERE id = $1 RETURNING id"

        async with self.pool.acquire() as conn:
            deleted = await conn.fetchval(query, tune_id)

        return deleted is not None

    # =========================================================================
    # KB Ingestion Support
    # =========================================================================

    async def list_tune_runs_for_kb(
        self,
        workspace_id: UUID,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
        only_missing_kb: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List tune_runs for KB ingestion.

        Returns tune_runs with their parent tune metadata for TrialDoc construction.

        Args:
            workspace_id: Workspace ID
            since: Only fetch runs created after this time
            limit: Maximum number to fetch
            only_missing_kb: Only fetch runs without kb_ingested_at

        Returns:
            List of dicts with 'tune_run' and 'tune' keys
        """
        conditions = [
            "t.workspace_id = $1",
            "tr.status = 'completed'",
        ]
        params: list[Any] = [workspace_id]
        param_idx = 2

        if only_missing_kb:
            conditions.append("tr.kb_ingested_at IS NULL")

        if since:
            conditions.append(f"tr.created_at >= ${param_idx}")
            params.append(since)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                tr.tune_id, tr.run_id, tr.trial_index, tr.params,
                tr.score, tr.score_is, tr.score_oos, tr.objective_score,
                tr.status, tr.skip_reason, tr.failed_reason,
                tr.metrics_is, tr.metrics_oos,
                tr.started_at, tr.finished_at, tr.created_at,
                tr.kb_ingested_at, tr.kb_embedding_model_id, tr.kb_vector_dim,
                t.strategy_entity_id, t.objective_type, t.objective_metric,
                t.oos_ratio, t.gates,
                e.name as strategy_name
            FROM backtest_tune_runs tr
            JOIN backtest_tunes t ON tr.tune_id = t.id
            LEFT JOIN kb_entities e ON t.strategy_entity_id = e.id
            WHERE {where_clause}
            ORDER BY tr.created_at ASC
        """

        if limit:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for row in rows:
            row_dict = dict(row)

            # Parse JSONB fields
            for field in ["params", "metrics_is", "metrics_oos", "gates"]:
                if row_dict.get(field) and isinstance(row_dict[field], str):
                    row_dict[field] = json.loads(row_dict[field])

            # Structure as tune_run + tune
            tune_run = {
                "run_id": row_dict["run_id"],
                "tune_id": row_dict["tune_id"],
                "trial_index": row_dict["trial_index"],
                "params": row_dict["params"],
                "score": row_dict["score"],
                "score_is": row_dict["score_is"],
                "score_oos": row_dict["score_oos"],
                "objective_score": row_dict["objective_score"],
                "status": row_dict["status"],
                "skip_reason": row_dict["skip_reason"],
                "failed_reason": row_dict["failed_reason"],
                "metrics_is": row_dict["metrics_is"],
                "metrics_oos": row_dict["metrics_oos"],
                "started_at": row_dict["started_at"],
                "finished_at": row_dict["finished_at"],
                "created_at": row_dict["created_at"],
                "kb_ingested_at": row_dict.get("kb_ingested_at"),
                "kb_embedding_model_id": row_dict.get("kb_embedding_model_id"),
                "kb_vector_dim": row_dict.get("kb_vector_dim"),
            }

            tune = {
                "id": row_dict["tune_id"],
                "strategy_entity_id": row_dict["strategy_entity_id"],
                "strategy_name": row_dict["strategy_name"],
                "objective_type": row_dict["objective_type"],
                "objective_metric": row_dict["objective_metric"],
                "oos_ratio": row_dict["oos_ratio"],
                "gates": row_dict.get("gates"),
            }

            results.append({"tune_run": tune_run, "tune": tune})

        return results

    async def mark_kb_ingested(
        self,
        tune_run_id: UUID,
        kb_ingested_at: str,
        kb_embedding_model_id: str,
        kb_vector_dim: int,
        kb_text_hash: str | None = None,
    ) -> None:
        """
        Mark a tune_run as ingested into KB.

        Args:
            tune_run_id: The run ID (not tune_id + trial_index)
            kb_ingested_at: ISO timestamp of ingestion
            kb_embedding_model_id: Embedding model used
            kb_vector_dim: Vector dimension
            kb_text_hash: SHA256[:16] hash of trial text for drift detection
        """
        query = """
            UPDATE backtest_tune_runs
            SET kb_ingested_at = $2,
                kb_embedding_model_id = $3,
                kb_vector_dim = $4,
                kb_text_hash = $5
            WHERE run_id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                tune_run_id,
                kb_ingested_at,
                kb_embedding_model_id,
                kb_vector_dim,
                kb_text_hash,
            )

    async def try_advisory_lock(self, lock_id: int) -> bool:
        """
        Try to acquire a PostgreSQL advisory lock.

        Uses pg_try_advisory_lock for non-blocking acquisition.
        Lock is held until connection closes or explicit release.

        Args:
            lock_id: Unique lock identifier (32-bit integer)

        Returns:
            True if lock acquired, False if already held by another session
        """
        query = "SELECT pg_try_advisory_lock($1)"
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(query, lock_id)
            return bool(result)

    async def release_advisory_lock(self, lock_id: int) -> bool:
        """
        Release a PostgreSQL advisory lock.

        Args:
            lock_id: Unique lock identifier

        Returns:
            True if lock was released, False if not held
        """
        query = "SELECT pg_advisory_unlock($1)"
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(query, lock_id)
            return bool(result)
