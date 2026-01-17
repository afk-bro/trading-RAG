"""KB trial sample retrieval for debugging."""

from typing import Any, List, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


async def get_trial_samples(
    pool,
    workspace_id: Optional[UUID],
    warning: Optional[str],
    is_valid: Optional[bool],
    has_oos: Optional[bool],
    strategy_name: Optional[str],
    limit: int,
) -> dict:
    """
    Get sample trials for debugging quality issues.

    Returns safe fields only - no sensitive internals.
    """
    async with pool.acquire() as conn:
        # Build dynamic WHERE clause
        conditions = ["1=1"]
        params: List[Any] = []
        param_idx = 1

        if workspace_id:
            conditions.append(f"workspace_id = ${param_idx}")
            params.append(workspace_id)
            param_idx += 1

        if warning:
            conditions.append(f"${param_idx} = ANY(warnings)")
            params.append(warning)
            param_idx += 1

        if is_valid is not None:
            conditions.append(f"is_valid = ${param_idx}")
            params.append(is_valid)
            param_idx += 1

        if has_oos is not None:
            conditions.append(f"has_oos_metrics = ${param_idx}")
            params.append(has_oos)
            param_idx += 1

        if strategy_name:
            conditions.append(f"strategy_name = ${param_idx}")
            params.append(strategy_name)
            param_idx += 1

        # Add limit
        params.append(limit)
        limit_param = f"${param_idx}"

        where_clause = " AND ".join(conditions)

        query = f"""
            SELECT
                id,
                point_id,
                tune_run_id,
                strategy_name,
                objective_type,
                objective_score,
                is_valid,
                has_oos_metrics,
                overfit_gap,
                sharpe_is,
                sharpe_oos,
                trades_is,
                trades_oos,
                max_drawdown_is,
                max_drawdown_oos,
                regime_tags_is,
                regime_tags_oos,
                warnings,
                created_at
            FROM kb_trial_vectors
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit_param}
        """

        try:
            rows = await conn.fetch(query, *params)
            samples = [
                {
                    "id": str(row["id"]),
                    "point_id": row["point_id"],
                    "tune_run_id": (
                        str(row["tune_run_id"]) if row["tune_run_id"] else None
                    ),
                    "strategy_name": row["strategy_name"],
                    "objective_type": row["objective_type"],
                    "objective_score": row["objective_score"],
                    "is_valid": row["is_valid"],
                    "has_oos_metrics": row["has_oos_metrics"],
                    "overfit_gap": row["overfit_gap"],
                    "metrics": {
                        "sharpe_is": row["sharpe_is"],
                        "sharpe_oos": row["sharpe_oos"],
                        "trades_is": row["trades_is"],
                        "trades_oos": row["trades_oos"],
                        "max_drawdown_is": row["max_drawdown_is"],
                        "max_drawdown_oos": row["max_drawdown_oos"],
                    },
                    "regime_tags_is": row["regime_tags_is"],
                    "regime_tags_oos": row["regime_tags_oos"],
                    "warnings": row["warnings"],
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                }
                for row in rows
            ]
        except Exception as e:
            logger.error("Failed to fetch KB samples", error=str(e))
            samples = []

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "filters": {
            "warning": warning,
            "is_valid": is_valid,
            "has_oos": has_oos,
            "strategy_name": strategy_name,
        },
        "count": len(samples),
        "samples": samples,
    }
