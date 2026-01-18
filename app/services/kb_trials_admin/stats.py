"""KB Trials statistics and ingestion status."""

import json
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)


async def compute_kb_trials_stats(
    pool,
    workspace_id: Optional[UUID],
    since: Optional[datetime],
    window_days: Optional[int],
) -> dict:
    """
    Compute KB trials statistics for a workspace.

    Returns point-in-time stats plus trend deltas when window_days is specified.
    """
    now = datetime.utcnow()

    # Calculate time boundaries
    if window_days:
        since = now - timedelta(days=window_days)
        prev_since = since - timedelta(days=window_days)
    else:
        prev_since = None

    async with pool.acquire() as conn:
        # Build query conditions
        workspace_cond = "AND t.workspace_id = $1" if workspace_id else ""
        base_params = [workspace_id] if workspace_id else []

        async def get_stats(time_filter: str = "", time_params: list = []):
            """Get stats with optional time filter."""
            params = base_params + time_params

            # Total trials
            total = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE 1=1 {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            if total == 0:
                return {
                    "total": 0,
                    "with_oos": 0,
                    "valid": 0,
                    "stale": 0,
                    "with_regime_is": 0,
                    "with_regime_oos": 0,
                    "with_objective_score": 0,
                    "with_sharpe_oos": 0,
                }

            # Core metrics
            with_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            valid = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.is_valid = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            # Stale count
            try:
                stale = (
                    await conn.fetchval(
                        f"""
                    SELECT COUNT(*) FROM kb_trial_vectors t
                    WHERE t.needs_reembed = true {workspace_cond} {time_filter}
                """,
                        *params,
                    )
                    or 0
                )
            except Exception:
                stale = 0

            # Coverage metrics
            with_regime_is = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_is IS NOT NULL {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_regime_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.regime_snapshot_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_objective = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.objective_score IS NOT NULL {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            with_sharpe_oos = (
                await conn.fetchval(
                    f"""
                SELECT COUNT(*) FROM kb_trial_vectors t
                WHERE t.sharpe_oos IS NOT NULL
                  AND t.has_oos_metrics = true {workspace_cond} {time_filter}
            """,
                    *params,
                )
                or 0
            )

            return {
                "total": total,
                "with_oos": with_oos,
                "valid": valid,
                "stale": stale,
                "with_regime_is": with_regime_is,
                "with_regime_oos": with_regime_oos,
                "with_objective_score": with_objective,
                "with_sharpe_oos": with_sharpe_oos,
            }

        # Get current stats
        if since:
            param_idx = len(base_params) + 1
            time_filter = f"AND t.created_at >= ${param_idx}"
            current = await get_stats(time_filter, [since])
        else:
            current = await get_stats()

        # Get previous window stats for deltas
        deltas = None
        if prev_since and window_days:
            # When window_days is set, since is guaranteed to be set
            assert since is not None
            param_idx = len(base_params) + 1
            time_filter = (
                f"AND t.created_at >= ${param_idx} AND t.created_at < ${param_idx + 1}"
            )
            previous = await get_stats(time_filter, [prev_since, since])

            # Calculate deltas
            deltas = {
                "trials_added": current["total"] - previous["total"],
                "valid_added": current["valid"] - previous["valid"],
                "stale_added": current["stale"] - previous["stale"],
                "pct_valid_delta": round(
                    (
                        current["valid"] / current["total"] * 100
                        if current["total"] > 0
                        else 0
                    )
                    - (
                        previous["valid"] / previous["total"] * 100
                        if previous["total"] > 0
                        else 0
                    ),
                    1,
                ),
                "window_days": window_days,
                "prev_window_start": prev_since.isoformat(),
                "prev_window_end": since.isoformat(),
            }

        # Last ingestion timestamp
        last_ts = await conn.fetchval(
            f"""
            SELECT MAX(t.created_at) FROM kb_trial_vectors t
            WHERE 1=1 {workspace_cond}
        """,
            *base_params,
        )

        # Workspace config for embedding info
        embed_model = "nomic-embed-text"
        embed_dim = 768
        collection_name = "trading_kb_trials__nomic-embed-text__768"

        if workspace_id:
            config_row = await conn.fetchrow(
                """
                SELECT config FROM workspaces WHERE id = $1
            """,
                workspace_id,
            )
            if config_row and config_row["config"]:
                config = config_row["config"]
                if isinstance(config, str):
                    config = json.loads(config)
                kb_config = config.get("kb", {})
                embed_model = kb_config.get("embed_model", embed_model)
                embed_dim = kb_config.get("embed_dim", embed_dim)
                collection_name = kb_config.get("collection_name", collection_name)

    # Calculate percentages
    total = current["total"]
    oos_count = current["with_oos"]

    def pct(num, denom):
        return round(num / denom * 100, 1) if denom > 0 else 0

    result = {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "total_trials": total,
        "trials_with_oos": oos_count,
        "trials_valid": current["valid"],
        "pct_with_oos": pct(oos_count, total),
        "pct_valid": pct(current["valid"], total),
        # Coverage metrics
        "coverage": {
            "pct_with_regime_is": pct(current["with_regime_is"], total),
            "pct_with_regime_oos": pct(current["with_regime_oos"], oos_count),
            "pct_with_objective_score": pct(current["with_objective_score"], total),
            "pct_with_sharpe_oos": pct(current["with_sharpe_oos"], oos_count),
        },
        # Embedding config
        "embedding_model": embed_model,
        "embedding_dim": embed_dim,
        "collection_name": collection_name,
        "last_ingestion_ts": last_ts.isoformat() if last_ts else None,
        "stale_text_hash_count": current["stale"],
    }

    # Add time window info
    if since:
        result["since"] = since.isoformat()
    if deltas:
        result["deltas"] = deltas

    return result


async def compute_ingestion_status(
    pool,
    workspace_id: Optional[UUID],
) -> dict:
    """
    Compute KB ingestion health status.

    Returns missing vectors/regime counts, warning aggregations, and recent runs.
    """
    async with pool.acquire() as conn:
        workspace_filter = "AND workspace_id = $1" if workspace_id else ""
        params = [workspace_id] if workspace_id else []

        # Missing vectors
        missing_vectors_query = f"""
            SELECT COUNT(*) as count
            FROM kb_trial_vectors
            WHERE vector IS NULL {workspace_filter}
        """
        try:
            missing_vectors = await conn.fetchval(missing_vectors_query, *params)
        except Exception:
            missing_vectors = 0

        # Missing regime
        missing_regime_query = f"""
            SELECT COUNT(*) as count
            FROM kb_trial_vectors
            WHERE regime_snapshot IS NULL {workspace_filter}
        """
        try:
            missing_regime = await conn.fetchval(missing_regime_query, *params)
        except Exception:
            missing_regime = 0

        # Warning counts
        warning_counts_query = f"""
            SELECT
                warning,
                COUNT(*) as count
            FROM kb_trial_vectors,
                 LATERAL unnest(warnings) as warning
            WHERE 1=1 {workspace_filter}
            GROUP BY warning
            ORDER BY count DESC
            LIMIT 20
        """
        try:
            warning_rows = await conn.fetch(warning_counts_query, *params)
            warning_counts = {row["warning"]: row["count"] for row in warning_rows}
        except Exception:
            warning_counts = {}

        # Recent ingestion runs (if tracked)
        recent_runs: List[dict] = []
        try:
            runs_query = f"""
                SELECT
                    id,
                    created_at,
                    ingested_count,
                    skipped_count,
                    error_count,
                    duration_ms
                FROM kb_ingestion_runs
                WHERE 1=1 {workspace_filter}
                ORDER BY created_at DESC
                LIMIT 10
            """
            run_rows = await conn.fetch(runs_query, *params)
            recent_runs = [
                {
                    "id": str(row["id"]),
                    "created_at": row["created_at"].isoformat(),
                    "ingested_count": row["ingested_count"],
                    "skipped_count": row["skipped_count"],
                    "error_count": row["error_count"],
                    "duration_ms": row["duration_ms"],
                }
                for row in run_rows
            ]
        except Exception:
            # Table might not exist yet
            pass

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "trials_missing_vectors": missing_vectors or 0,
        "trials_missing_regime": missing_regime or 0,
        "warning_counts": warning_counts,
        "recent_ingestion_runs": recent_runs,
    }
