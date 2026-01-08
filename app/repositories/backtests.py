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
            for field in ["params", "dataset_meta", "summary", "equity_curve", "trades", "warnings"]:
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
    ) -> UUID:
        """
        Create a new tune record (status=queued).

        Returns:
            The new tune ID
        """
        query = """
            INSERT INTO backtest_tunes (
                workspace_id, strategy_entity_id, strategy_spec_id,
                search_type, n_trials, seed, param_space,
                objective_metric, min_trades, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'queued')
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
        leaderboard: list[dict[str, Any]],
        trials_completed: int,
    ) -> None:
        """Mark tune as completed with results."""
        query = """
            UPDATE backtest_tunes
            SET status = 'completed',
                best_run_id = $2,
                leaderboard = $3,
                trials_completed = $4,
                completed_at = NOW()
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                tune_id,
                best_run_id,
                json.dumps(leaderboard),
                trials_completed,
            )

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
        run_id: UUID,
        score: Optional[float],
        status: str,
    ) -> None:
        """Update tune_run with result after backtest completes."""
        query = """
            UPDATE backtest_tune_runs
            SET run_id = $3, score = $4, status = $5
            WHERE tune_id = $1 AND trial_index = $2
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, tune_id, trial_index, run_id, score, status)

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
            for field in ["param_space", "leaderboard"]:
                if result.get(field) and isinstance(result[field], str):
                    result[field] = json.loads(result[field])

            return result

    async def list_tunes(
        self,
        workspace_id: UUID,
        strategy_entity_id: Optional[UUID] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List tunes with optional filtering.

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

        where_clause = " AND ".join(conditions)

        count_query = f"SELECT COUNT(*) FROM backtest_tunes t WHERE {where_clause}"

        list_query = f"""
            SELECT t.id, t.created_at, t.strategy_entity_id, t.search_type,
                   t.n_trials, t.seed, t.objective_metric, t.min_trades,
                   t.status, t.trials_completed, t.best_run_id, t.leaderboard,
                   t.started_at, t.completed_at, t.error,
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
            if tune.get("leaderboard") and isinstance(tune["leaderboard"], str):
                tune["leaderboard"] = json.loads(tune["leaderboard"])
            tunes.append(tune)

        return tunes, total

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

        list_query = f"""
            SELECT tr.tune_id, tr.run_id, tr.trial_index, tr.params,
                   tr.score, tr.status, tr.created_at
            FROM backtest_tune_runs tr
            WHERE {where_clause}
            ORDER BY tr.score DESC NULLS LAST, tr.trial_index ASC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(list_query, *params)

        runs = []
        for row in rows:
            run = dict(row)
            if run.get("params") and isinstance(run["params"], str):
                run["params"] = json.loads(run["params"])
            runs.append(run)

        return runs, total

    async def delete_tune(self, tune_id: UUID) -> bool:
        """Delete a tune and its runs (cascade)."""
        query = "DELETE FROM backtest_tunes WHERE id = $1 RETURNING id"

        async with self.pool.acquire() as conn:
            deleted = await conn.fetchval(query, tune_id)

        return deleted is not None
