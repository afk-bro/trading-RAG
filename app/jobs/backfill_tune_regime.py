"""
Tune regime attribution backfill job.

Populates regime_key, regime_fingerprint, trend_tag, vol_tag, efficiency_tag,
best_oos_score, best_oos_params, and best_oos_run_id for existing backtest_tunes
that have completed runs but are missing regime data.

Uses the same logic as tuner.py completion handler (lines 929-959).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.kb.regime import (
    compute_regime_key,
    compute_regime_fingerprint,
    extract_regime_tags_for_attribution,
    DEFAULT_RULESET_ID,
)
from app.services.kb.types import RegimeSnapshot

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BackfillResult:
    """Result of tune regime backfill job."""

    processed: int = 0
    skipped_no_runs: int = 0
    skipped_no_regime: int = 0
    updated: int = 0
    would_update: int = 0  # For dry run
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Main Backfill Job
# =============================================================================


class BackfillTuneRegimeJob:
    """
    Backfill regime attribution columns on existing backtest_tunes.

    For each tune with regime_key IS NULL:
    1. Get the best OOS run (ordered by objective_score DESC)
    2. Extract regime snapshot from metrics_oos.regime
    3. Compute regime_key, fingerprint, and tags
    4. Get best_oos_score from metrics_oos.sharpe
    5. Update the tune record
    """

    def __init__(self, pool):
        """Initialize with database connection pool."""
        self.pool = pool

    async def run(
        self,
        workspace_id: Optional[UUID] = None,
        dry_run: bool = False,
        limit: Optional[int] = None,
    ) -> BackfillResult:
        """
        Run the backfill job.

        Args:
            workspace_id: Optional workspace filter (backfills all if None)
            dry_run: If True, don't write to DB (just compute stats)
            limit: Maximum number of tunes to process (None for all)

        Returns:
            BackfillResult with counts and errors
        """
        result = BackfillResult(dry_run=dry_run)

        logger.info(
            "tune_regime_backfill_started",
            workspace_id=str(workspace_id) if workspace_id else None,
            dry_run=dry_run,
            limit=limit,
        )

        # Get tunes missing regime data
        tunes = await self._get_tunes_missing_regime(workspace_id, limit)

        if not tunes:
            logger.info("tune_regime_backfill_no_tunes_found")
            return result

        logger.info(
            "tune_regime_backfill_found_tunes",
            count=len(tunes),
        )

        for tune in tunes:
            tune_id = tune["id"]
            result.processed += 1

            try:
                # Get best OOS run for this tune
                best_run = await self._get_best_oos_run(tune_id)

                if not best_run:
                    result.skipped_no_runs += 1
                    logger.debug(
                        "tune_regime_backfill_no_runs",
                        tune_id=str(tune_id),
                    )
                    continue

                # Extract regime data from metrics_oos
                metrics_oos = best_run.get("metrics_oos")
                regime_data = metrics_oos.get("regime") if metrics_oos else None

                if not regime_data:
                    result.skipped_no_regime += 1
                    logger.warning(
                        "tune_regime_backfill_no_regime_data",
                        tune_id=str(tune_id),
                        run_id=(
                            str(best_run["run_id"]) if best_run.get("run_id") else None
                        ),
                    )
                    continue

                # Compute regime attribution
                regime_snapshot = RegimeSnapshot.from_dict(regime_data)
                regime_key = compute_regime_key(
                    regime_snapshot, ruleset_id=DEFAULT_RULESET_ID
                )
                regime_fingerprint = compute_regime_fingerprint(regime_key)
                trend_tag, vol_tag, eff_tag = extract_regime_tags_for_attribution(
                    regime_snapshot
                )

                # Get OOS score (sharpe from metrics_oos if available)
                best_oos_score = metrics_oos.get("sharpe") if metrics_oos else None

                # Get best params and run_id
                best_oos_params = best_run.get("params")
                best_oos_run_id = best_run.get("run_id")

                if dry_run:
                    result.would_update += 1
                    logger.debug(
                        "tune_regime_backfill_would_update",
                        tune_id=str(tune_id),
                        regime_key=regime_key,
                        best_oos_score=best_oos_score,
                    )
                else:
                    # Update the tune record
                    await self._update_tune_regime_attribution(
                        tune_id=tune_id,
                        regime_schema_version=regime_snapshot.schema_version,
                        tag_ruleset_id=DEFAULT_RULESET_ID,
                        regime_key=regime_key,
                        regime_fingerprint=regime_fingerprint,
                        trend_tag=trend_tag,
                        vol_tag=vol_tag,
                        efficiency_tag=eff_tag,
                        best_oos_score=best_oos_score,
                        best_oos_params=best_oos_params,
                        best_oos_run_id=best_oos_run_id,
                    )
                    result.updated += 1
                    logger.debug(
                        "tune_regime_backfill_updated",
                        tune_id=str(tune_id),
                        regime_key=regime_key,
                    )

            except Exception as e:
                error_msg = f"Failed to backfill tune {tune_id}: {str(e)}"
                logger.error(
                    "tune_regime_backfill_error",
                    tune_id=str(tune_id),
                    error=str(e),
                )
                result.errors.append(error_msg)

        if dry_run:
            logger.info(
                "tune_regime_backfill_dry_run_complete",
                processed=result.processed,
                would_update=result.would_update,
                skipped_no_runs=result.skipped_no_runs,
                skipped_no_regime=result.skipped_no_regime,
                errors=len(result.errors),
            )
        else:
            logger.info(
                "tune_regime_backfill_complete",
                processed=result.processed,
                updated=result.updated,
                skipped_no_runs=result.skipped_no_runs,
                skipped_no_regime=result.skipped_no_regime,
                errors=len(result.errors),
            )

        return result

    async def _get_tunes_missing_regime(
        self,
        workspace_id: Optional[UUID] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Get tunes where regime_key IS NULL and status = 'completed'.

        Only includes completed tunes that have at least one completed run.
        """
        conditions = [
            "t.regime_key IS NULL",
            "t.status = 'completed'",
        ]
        params: list[Any] = []
        param_idx = 1

        if workspace_id:
            conditions.append(f"t.workspace_id = ${param_idx}")
            params.append(workspace_id)
            param_idx += 1

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT t.id, t.workspace_id, t.strategy_entity_id
            FROM backtest_tunes t
            WHERE {where_clause}
            ORDER BY t.created_at ASC
        """

        if limit:
            query += f" LIMIT ${param_idx}"
            params.append(limit)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]

    async def _get_best_oos_run(
        self,
        tune_id: UUID,
    ) -> Optional[dict[str, Any]]:
        """
        Get the best OOS run for a tune, ordered by objective_score DESC.

        Only considers completed runs with metrics_oos.
        """
        query = """
            SELECT tr.run_id, tr.params, tr.objective_score,
                   tr.score_oos, tr.metrics_oos
            FROM backtest_tune_runs tr
            WHERE tr.tune_id = $1
              AND tr.status = 'completed'
              AND tr.metrics_oos IS NOT NULL
            ORDER BY COALESCE(tr.objective_score, tr.score_oos) DESC NULLS LAST
            LIMIT 1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, tune_id)

        if not row:
            return None

        run = dict(row)

        # Parse JSONB fields
        for json_field in ["params", "metrics_oos"]:
            if run.get(json_field) and isinstance(run[json_field], str):
                run[json_field] = json.loads(run[json_field])

        return run

    async def _update_tune_regime_attribution(
        self,
        tune_id: UUID,
        regime_schema_version: str,
        tag_ruleset_id: str,
        regime_key: str,
        regime_fingerprint: str,
        trend_tag: Optional[str],
        vol_tag: Optional[str],
        efficiency_tag: Optional[str],
        best_oos_score: Optional[float],
        best_oos_params: Optional[dict[str, Any]],
        best_oos_run_id: Optional[UUID],
    ) -> None:
        """
        Update tune with regime attribution.

        Matches the same logic as TuneRepository.populate_regime_attribution.
        """
        query = """
            UPDATE backtest_tunes SET
                regime_schema_version = $2,
                tag_ruleset_id = $3,
                regime_key = $4,
                regime_fingerprint = $5,
                trend_tag = $6,
                vol_tag = $7,
                efficiency_tag = $8,
                best_oos_score = $9,
                best_oos_params = $10,
                best_oos_run_id = $11
            WHERE id = $1
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                tune_id,
                regime_schema_version,
                tag_ruleset_id,
                regime_key,
                regime_fingerprint,
                trend_tag,
                vol_tag,
                efficiency_tag,
                best_oos_score,
                json.dumps(best_oos_params) if best_oos_params else None,
                best_oos_run_id,
            )


# =============================================================================
# Convenience Function
# =============================================================================


async def run_tune_regime_backfill(
    db_pool,
    workspace_id: Optional[UUID] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> BackfillResult:
    """
    Run tune regime backfill job.

    Convenience function that creates job instance and runs it.

    Args:
        db_pool: asyncpg connection pool
        workspace_id: Optional workspace filter (backfills all if None)
        dry_run: If True, don't write to DB (just compute stats)
        limit: Maximum number of tunes to process (None for all)

    Returns:
        BackfillResult with counts and errors
    """
    job = BackfillTuneRegimeJob(db_pool)
    return await job.run(workspace_id=workspace_id, dry_run=dry_run, limit=limit)
