"""Repository for dashboard read queries."""

from typing import Any, Optional
from uuid import UUID


class DashboardsRepository:

    def __init__(self, pool):
        self.pool = pool

    async def get_equity_curve(self, workspace_id: UUID, days: int) -> list[dict]:
        query = """
            WITH snapshots AS (
                SELECT
                    snapshot_ts,
                    computed_at,
                    equity,
                    cash,
                    positions_value,
                    realized_pnl,
                    -- Calculate rolling peak and drawdown
                    MAX(equity) OVER (
                        ORDER BY snapshot_ts
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS peak_equity
                FROM paper_equity_snapshots
                WHERE workspace_id = $1
                  AND snapshot_ts >= NOW() - make_interval(days => $2)
                ORDER BY snapshot_ts ASC
            )
            SELECT
                snapshot_ts,
                computed_at,
                equity,
                cash,
                positions_value,
                realized_pnl,
                peak_equity,
                CASE
                    WHEN peak_equity > 0 THEN (peak_equity - equity) / peak_equity
                    ELSE 0.0
                END AS drawdown_pct
            FROM snapshots
            ORDER BY snapshot_ts ASC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, days)

        return [dict(r) for r in rows]

    async def get_intel_timeline(
        self,
        workspace_id: UUID,
        days: int,
        version_id: Optional[UUID] = None,
    ) -> list[dict]:
        if version_id:
            query = """
                SELECT
                    sis.id,
                    sis.strategy_version_id,
                    sv.version_number,
                    sv.version_tag,
                    s.name AS strategy_name,
                    sis.as_of_ts,
                    sis.computed_at,
                    sis.regime,
                    sis.confidence_score,
                    sis.confidence_components,
                    sis.source_snapshot_id
                FROM strategy_intel_snapshots sis
                JOIN strategy_versions sv ON sis.strategy_version_id = sv.id
                JOIN strategies s ON sv.strategy_id = s.id
                WHERE s.workspace_id = $1
                  AND sis.strategy_version_id = $2
                  AND sis.as_of_ts >= NOW() - make_interval(days => $3)
                ORDER BY sis.as_of_ts DESC
            """
            params: list[Any] = [workspace_id, version_id, days]
        else:
            query = """
                SELECT
                    sis.id,
                    sis.strategy_version_id,
                    sv.version_number,
                    sv.version_tag,
                    s.name AS strategy_name,
                    sis.as_of_ts,
                    sis.computed_at,
                    sis.regime,
                    sis.confidence_score,
                    sis.confidence_components,
                    sis.source_snapshot_id
                FROM strategy_intel_snapshots sis
                JOIN strategy_versions sv ON sis.strategy_version_id = sv.id
                JOIN strategies s ON sv.strategy_id = s.id
                WHERE s.workspace_id = $1
                  AND sv.state = 'active'
                  AND sis.as_of_ts >= NOW() - make_interval(days => $2)
                ORDER BY sis.as_of_ts DESC
            """
            params = [workspace_id, days]

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(r) for r in rows]

    async def get_alerts(
        self,
        workspace_id: UUID,
        days: int,
        include_resolved: bool = False,
    ) -> list[dict]:
        if include_resolved:
            status_filter = "1=1"  # No filter
        else:
            status_filter = "status = 'active'"

        query = f"""
            SELECT
                id,
                workspace_id,
                rule_type,
                severity,
                status,
                dedupe_key,
                payload,
                source,
                rule_version,
                occurrence_count,
                first_triggered_at,
                last_triggered_at,
                acknowledged_at,
                acknowledged_by,
                resolved_at,
                resolved_by,
                resolution_note,
                created_at
            FROM ops_alerts
            WHERE workspace_id = $1
              AND {status_filter}
              AND created_at >= NOW() - make_interval(days => $2)
            ORDER BY
                CASE status
                    WHEN 'active' THEN 0
                    WHEN 'acknowledged' THEN 1
                    WHEN 'resolved' THEN 2
                END,
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                END,
                last_triggered_at DESC
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id, days)

        return [dict(r) for r in rows]

    async def get_summary_equity(self, workspace_id: UUID) -> Optional[dict]:
        query = """
            WITH latest AS (
                SELECT
                    equity,
                    cash,
                    positions_value,
                    snapshot_ts,
                    MAX(equity) OVER (
                        ORDER BY snapshot_ts
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS peak_equity
                FROM paper_equity_snapshots
                WHERE workspace_id = $1
                  AND snapshot_ts >= NOW() - INTERVAL '30 days'
                ORDER BY snapshot_ts DESC
                LIMIT 1
            )
            SELECT
                equity,
                cash,
                positions_value,
                snapshot_ts,
                peak_equity,
                CASE
                    WHEN peak_equity > 0 THEN (peak_equity - equity) / peak_equity
                    ELSE 0.0
                END AS drawdown_pct
            FROM latest
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id)

        return dict(row) if row else None

    async def get_summary_intel(self, workspace_id: UUID) -> list[dict]:
        query = """
            WITH latest_per_version AS (
                SELECT
                    sis.strategy_version_id,
                    sv.version_number,
                    s.name AS strategy_name,
                    sis.regime,
                    sis.confidence_score,
                    sis.as_of_ts,
                    ROW_NUMBER() OVER (
                        PARTITION BY sis.strategy_version_id
                        ORDER BY sis.as_of_ts DESC
                    ) AS rn
                FROM strategy_intel_snapshots sis
                JOIN strategy_versions sv ON sis.strategy_version_id = sv.id
                JOIN strategies s ON sv.strategy_id = s.id
                WHERE s.workspace_id = $1
                  AND sv.state = 'active'
            )
            SELECT *
            FROM latest_per_version
            WHERE rn = 1
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id)

        return [dict(r) for r in rows]

    async def get_summary_alert_counts(self, workspace_id: UUID) -> list[dict]:
        query = """
            SELECT
                severity,
                COUNT(*) AS count
            FROM ops_alerts
            WHERE workspace_id = $1
              AND status = 'active'
            GROUP BY severity
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id)

        return [dict(r) for r in rows]

    async def get_backtest_run(
        self, run_id: UUID, workspace_id: UUID
    ) -> Optional[dict]:
        query = """
            SELECT
                br.id,
                br.status,
                br.created_at,
                br.started_at,
                br.completed_at,
                br.params,
                br.summary,
                br.dataset_meta,
                br.equity_curve,
                br.trades,
                br.warnings,
                br.run_kind,
                br.regime_is,
                br.regime_oos,
                br.trade_count,
                br.strategy_entity_id,
                br.strategy_version_id,
                -- Strategy name via kb_entities
                ke.name AS strategy_name
            FROM backtest_runs br
            LEFT JOIN kb_entities ke ON br.strategy_entity_id = ke.id
            WHERE br.id = $1
              AND br.workspace_id = $2
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, run_id, workspace_id)

        return dict(row) if row else None

    async def get_backtest_run_strategy(
        self, run_id: UUID, workspace_id: UUID
    ) -> Optional[dict]:
        query = """
            SELECT strategy_entity_id, completed_at
            FROM backtest_runs
            WHERE id = $1 AND workspace_id = $2
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, run_id, workspace_id)

        return dict(row) if row else None
