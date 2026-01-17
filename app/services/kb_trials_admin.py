"""KB Trials Admin Service - orchestration layer for admin endpoints.

Extracts business logic from kb_trials router for better testability.
All functions return dicts suitable for JSON responses.
"""

import json
from datetime import datetime, timedelta
from typing import Any, List, Optional, Union
from uuid import UUID

import structlog

from app.admin.kb_trials_schemas import (
    PromotionPreviewResponse,
    PromotionPreviewSummary,
    TrialPreviewItem,
)

logger = structlog.get_logger(__name__)


def _json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable form."""
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)
    return obj


# =============================================================================
# KB Trials Stats
# =============================================================================


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


# =============================================================================
# Ingestion Status
# =============================================================================


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
        recent_runs = []
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


# =============================================================================
# Qdrant Collections
# =============================================================================


async def get_qdrant_collections(host: str, port: int) -> dict:
    """
    Get list of KB collections in Qdrant with health info.

    Returns collection metadata, vector config, and optimizer status.
    """
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(host=host, port=port)

    try:
        collections_response = await client.get_collections()

        result = []
        for coll in collections_response.collections:
            try:
                info = await client.get_collection(coll.name)

                # Parse embedding model from collection name if encoded
                # Format: trading_kb_trials__{model}__{dim}
                parts = coll.name.split("__")
                embedding_model = parts[1] if len(parts) >= 2 else None
                embedding_dim = int(parts[2]) if len(parts) >= 3 else None

                # Vector config - handle both single and named vectors
                vec_cfg = info.config.params.vectors
                vector_size = None
                distance = None
                if vec_cfg:
                    # Check if it's a dict (named vectors) or single VectorParams
                    if isinstance(vec_cfg, dict):
                        # Get first vector config from dict
                        first_vec = next(iter(vec_cfg.values()), None)
                        if first_vec:
                            vector_size = getattr(first_vec, "size", None)
                            dist_attr = getattr(first_vec, "distance", None)
                            distance = dist_attr.value if dist_attr else None
                    else:
                        vector_size = getattr(vec_cfg, "size", None)
                        dist_attr = getattr(vec_cfg, "distance", None)
                        distance = dist_attr.value if dist_attr else None

                # Payload indexes count
                payload_indexes = 0
                if info.payload_schema:
                    payload_indexes = len(info.payload_schema)

                # Optimizer status
                optimizer_status = "unknown"
                if info.optimizer_status:
                    optimizer_status = (
                        info.optimizer_status.status.value
                        if hasattr(info.optimizer_status, "status")
                        else str(info.optimizer_status)
                    )

                result.append(
                    {
                        "name": coll.name,
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count,
                        "status": info.status.value if info.status else "unknown",
                        "vector_size": vector_size or embedding_dim,
                        "distance": distance,
                        "embedding_model_id": embedding_model,
                        "payload_indexes_count": payload_indexes,
                        "optimizer_status": optimizer_status,
                        "segments_count": (
                            len(info.segments or [])
                            if hasattr(info, "segments")
                            else None
                        ),
                    }
                )
            except Exception as e:
                result.append(
                    {
                        "name": coll.name,
                        "error": str(e),
                    }
                )

        return {
            "collections": result,
            "qdrant_host": host,
            "qdrant_port": port,
            "total_collections": len(result),
        }

    finally:
        await client.close()


# =============================================================================
# Top Warnings
# =============================================================================


async def get_top_warnings(
    pool,
    workspace_id: Optional[UUID],
    limit: int,
) -> dict:
    """
    Get top warning types across KB trials.

    Returns aggregated warning counts sorted by frequency.
    """
    async with pool.acquire() as conn:
        workspace_filter = "AND workspace_id = $1" if workspace_id else ""
        params = [workspace_id, limit] if workspace_id else [limit]
        limit_param = "$2" if workspace_id else "$1"

        query = f"""
            SELECT
                warning,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (
                    SELECT COUNT(*) FROM kb_trial_vectors
                    WHERE 1=1 {workspace_filter}
                ) as pct
            FROM kb_trial_vectors,
                 LATERAL unnest(warnings) as warning
            WHERE 1=1 {workspace_filter}
            GROUP BY warning
            ORDER BY count DESC
            LIMIT {limit_param}
        """

        try:
            rows = await conn.fetch(query, *params)
            warnings = [
                {
                    "warning": row["warning"],
                    "count": row["count"],
                    "pct": round(row["pct"], 2) if row["pct"] else 0,
                }
                for row in rows
            ]
        except Exception:
            warnings = []

    return {
        "workspace_id": str(workspace_id) if workspace_id else None,
        "warnings": warnings,
    }


# =============================================================================
# Trial Samples
# =============================================================================


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


# =============================================================================
# Promotion Preview
# =============================================================================


async def compute_promotion_preview(
    pool,
    workspace_id: UUID,
    source_type: Optional[str],
    group_id: Optional[UUID],
    limit: int,
    offset: int,
    sort: str,
    include_ineligible: bool,
) -> PromotionPreviewResponse:
    """
    Compute promotion preview for trials.

    Returns trials that could be promoted with eligibility analysis.
    Uses the same candidacy logic as auto-promotion for consistency.
    """
    from app.services.kb.candidacy import (
        CandidacyConfig,
        is_candidate,
        VariantMetricsForCandidacy,
        KNOWN_EXPERIMENT_TYPES,
    )
    from app.services.kb.types import RegimeSnapshot

    # Build query for eligible trials view
    query = """
        SELECT
            source_type,
            experiment_type,
            source_id,
            group_id,
            workspace_id,
            strategy_name,
            params,
            trial_status,
            regime_is,
            regime_oos,
            regime_schema_version,
            sharpe_oos,
            return_frac_oos,
            max_dd_frac_oos,
            n_trades_oos,
            sharpe_is,
            kb_status,
            kb_promoted_at,
            objective_type,
            objective_score,
            created_at
        FROM kb_eligible_trials
        WHERE workspace_id = $1
    """
    params: List[Any] = [workspace_id]
    param_idx = 2

    if source_type:
        query += f" AND source_type = ${param_idx}"
        params.append(source_type)
        param_idx += 1

    if group_id:
        query += f" AND group_id = ${param_idx}"
        params.append(group_id)
        param_idx += 1

    # Also include excluded/candidate trials if include_ineligible
    if include_ineligible:
        query = f"""
            WITH eligible AS ({query})
            SELECT * FROM eligible
            UNION ALL
            SELECT
                'test_variant'::TEXT AS source_type,
                COALESCE(r.summary->>'experiment_type', 'sweep')::TEXT AS experiment_type,
                r.id AS source_id,
                r.run_plan_id AS group_id,
                r.workspace_id,
                COALESCE(r.summary->>'strategy_name', e.name) AS strategy_name,
                r.params,
                r.status AS trial_status,
                r.regime_is,
                r.regime_oos,
                r.regime_schema_version,
                (r.summary->>'sharpe')::FLOAT AS sharpe_oos,
                (r.summary->>'return_pct')::FLOAT AS return_frac_oos,
                (r.summary->>'max_drawdown_pct')::FLOAT AS max_dd_frac_oos,
                (r.summary->>'trade_count')::INT AS n_trades_oos,
                NULL::FLOAT AS sharpe_is,
                r.kb_status,
                r.kb_promoted_at,
                COALESCE(rp.objective_name, 'sharpe') AS objective_type,
                r.objective_score,
                r.created_at
            FROM backtest_runs r
            LEFT JOIN kb_entities e ON r.strategy_entity_id = e.id
            LEFT JOIN run_plans rp ON r.run_plan_id = rp.id
            WHERE r.workspace_id = $1
              AND r.run_kind = 'test_variant'
              AND r.kb_status = 'excluded'
              AND r.status IN ('completed', 'success')
        """

    # Add sorting
    sort_column = "sharpe_oos"
    if sort in ("sharpe_oos", "return_frac_oos", "max_dd_frac_oos", "created_at"):
        sort_column = sort

    query += f" ORDER BY {sort_column} DESC NULLS LAST, created_at DESC, source_id"
    query += f" LIMIT ${param_idx} OFFSET ${param_idx + 1}"
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        # Get total count
        count_query = """
            SELECT COUNT(*) FROM kb_eligible_trials
            WHERE workspace_id = $1
        """
        count_params: List[Any] = [workspace_id]
        if source_type:
            count_query += " AND source_type = $2"
            count_params.append(source_type)
        if group_id:
            count_query += f" AND group_id = ${len(count_params) + 1}"
            count_params.append(group_id)

        total = await conn.fetchval(count_query, *count_params)

    # Process rows with candidacy check
    trials = []
    summary = PromotionPreviewSummary()
    config = CandidacyConfig()

    for row in rows:
        regime_is = None
        regime_oos = None
        if row["regime_is"]:
            regime_is = RegimeSnapshot.from_dict(row["regime_is"])
        if row["regime_oos"]:
            regime_oos = RegimeSnapshot.from_dict(row["regime_oos"])

        # Check candidacy gates
        ineligibility_reasons = []
        passes_gates = False

        experiment_type = row["experiment_type"] or "sweep"

        if experiment_type not in KNOWN_EXPERIMENT_TYPES:
            ineligibility_reasons.append("unknown_experiment_type")
        elif experiment_type == "manual":
            ineligibility_reasons.append("manual_experiment_excluded")
        else:
            # Build metrics for candidacy check
            metrics = VariantMetricsForCandidacy(
                sharpe_oos=row["sharpe_oos"],
                max_dd_frac_oos=row["max_dd_frac_oos"] or 0.0,
                n_trades_oos=row["n_trades_oos"] or 0,
                overfit_gap=(
                    max(0, (row["sharpe_is"] or 0) - (row["sharpe_oos"] or 0))
                    if row["sharpe_is"] is not None and row["sharpe_oos"] is not None
                    else None
                ),
            )

            decision = is_candidate(
                metrics=metrics,
                regime_oos=regime_oos,
                experiment_type=experiment_type,
                config=config,
            )

            passes_gates = decision.eligible
            if not decision.eligible:
                ineligibility_reasons.append(decision.reason)

        # Determine promotion eligibility
        kb_status = row["kb_status"]
        can_promote = kb_status not in ("promoted", "rejected")
        is_eligible = passes_gates and can_promote

        # Update summary counts
        if kb_status == "promoted":
            summary.already_promoted += 1
        elif not regime_oos and kb_status != "promoted":
            summary.missing_regime += 1
        elif is_eligible:
            summary.would_promote += 1
        else:
            summary.would_skip += 1

        trials.append(
            TrialPreviewItem(
                source_type=row["source_type"],
                source_id=row["source_id"],
                group_id=row["group_id"],
                experiment_type=experiment_type,
                strategy_name=row["strategy_name"],
                kb_status=kb_status,
                sharpe_oos=row["sharpe_oos"],
                return_frac_oos=row["return_frac_oos"],
                max_dd_frac_oos=row["max_dd_frac_oos"],
                n_trades_oos=row["n_trades_oos"],
                passes_auto_gates=passes_gates,
                can_promote=can_promote,
                is_eligible=is_eligible,
                ineligibility_reasons=ineligibility_reasons,
                has_regime_is=regime_is is not None,
                has_regime_oos=regime_oos is not None,
                regime_schema_version=row["regime_schema_version"],
                created_at=row["created_at"],
            )
        )

    return PromotionPreviewResponse(
        summary=summary,
        pagination={
            "limit": limit,
            "offset": offset,
            "total": total or 0,
        },
        trials=trials,
    )
