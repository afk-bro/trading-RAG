"""KB trial promotion preview and eligibility analysis."""

from typing import Any, List, Optional
from uuid import UUID

from app.admin.kb_trials_schemas import (
    PromotionPreviewResponse,
    PromotionPreviewSummary,
    TrialPreviewItem,
)


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
