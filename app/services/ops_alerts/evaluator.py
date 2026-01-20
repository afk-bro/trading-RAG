"""Ops alert evaluator - evaluates rules and manages alert lifecycle."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import structlog

from app.repositories.ops_alerts import OpsAlertsRepository, OpsAlert
from app.routers.metrics import (
    record_ops_alert_eval,
    record_ops_alert_resolved,
    record_ops_alert_triggered,
    set_workspace_drawdown,
)
from app.services.ops_alerts.models import (
    AlertCondition,
    EvalContext,
    EvalResult,
    OpsAlertRule,
    OpsRuleType,
    Severity,
    get_all_rules,
)

logger = structlog.get_logger(__name__)


class OpsAlertEvaluator:
    """
    Evaluator for operational alerts.

    Evaluates all rules for a workspace, upserts triggered alerts,
    resolves cleared conditions, and returns notification candidates.
    """

    # Thresholds for health rule
    HEALTH_ERROR_STATES = {"error", "halted"}
    HEALTH_DEGRADED_STATES = {"degraded"}

    # Thresholds for coverage rules
    P1_PRIORITY_THRESHOLD = 0.75
    P2_PRIORITY_THRESHOLD = 0.40

    # Thresholds for drift rule (using match quality)
    DRIFT_THRESHOLD = 2.0  # weak_rate_recent >= weak_rate_baseline * 2
    DRIFT_SCORE_DROP_THRESHOLD = 0.12  # avg_score drops by this much

    # Thresholds for confidence rule
    CONFIDENCE_FLOOR = 0.45  # Absolute floor for median best_score
    CONFIDENCE_DELTA_THRESHOLD = 0.10  # Relative drop from baseline

    # Thresholds for strategy confidence alerts (v1.5)
    STRATEGY_CONFIDENCE_WARN = 0.35  # Warn if score < this
    STRATEGY_CONFIDENCE_CRITICAL = 0.20  # Critical if score < this
    STRATEGY_CONFIDENCE_CLEAR_WARN = 0.40  # Clear warn when score >= this
    STRATEGY_CONFIDENCE_CLEAR_CRITICAL = 0.25  # Clear critical when score >= this

    # Thresholds for workspace drawdown alerts
    DRAWDOWN_WARN = 0.12  # Warn if DD > 12%
    DRAWDOWN_CRITICAL = 0.20  # Critical if DD > 20%
    DRAWDOWN_CLEAR_WARN = 0.10  # Clear warn when DD < 10%
    DRAWDOWN_CLEAR_CRITICAL = 0.16  # Clear critical when DD < 16%
    DRAWDOWN_WINDOW_DAYS = 30  # Rolling window for peak calculation

    def __init__(self, repo: OpsAlertsRepository, pool: Any):
        """Initialize evaluator with repository and DB pool."""
        self.repo = repo
        self.pool = pool

    async def evaluate(
        self,
        workspace_id: UUID,
        now: Optional[datetime] = None,
        job_run_id: Optional[UUID] = None,
    ) -> EvalResult:
        """
        Evaluate all rules for a workspace.

        Returns EvalResult with metrics and notification candidates.
        """
        now = now or datetime.now(timezone.utc)
        result = EvalResult(
            workspace_id=workspace_id,
            job_run_id=job_run_id,
            timestamp=now,
        )

        log = logger.bind(workspace_id=str(workspace_id), job_run_id=str(job_run_id))
        log.info("ops_alert_eval_started")

        # Build context with data sources
        ctx = await self._build_context(workspace_id, now, job_run_id)

        # Track triggered dedupe keys for resolution pass
        triggered_keys: set[str] = set()

        # Evaluate each rule
        for rule in get_all_rules():
            try:
                eval_result = await self._evaluate_rule(rule, ctx)

                # Normalize to list (some rules return multiple conditions)
                conditions = (
                    eval_result if isinstance(eval_result, list) else [eval_result]
                )

                rule_result: dict[str, Any] = {
                    "triggered": False,
                    "count": 0,
                }

                for condition in conditions:
                    result.conditions_evaluated += 1

                    if condition.skip_reason:
                        rule_result["skipped"] = condition.skip_reason
                        # Record metric: evaluation skipped
                        record_ops_alert_eval(
                            rule_type=rule.rule_type.value,
                            triggered=False,
                            skipped=True,
                        )
                        log.debug(
                            "ops_alert_rule_skipped",
                            rule_type=rule.rule_type.value,
                            reason=condition.skip_reason,
                        )
                    elif condition.triggered:
                        rule_result["triggered"] = True
                        rule_result["count"] = rule_result.get("count", 0) + 1
                        triggered_keys.add(condition.dedupe_key)

                        # Record metric: evaluation triggered
                        record_ops_alert_eval(
                            rule_type=rule.rule_type.value,
                            triggered=True,
                            skipped=False,
                        )

                        # Upsert the alert
                        upsert_result = await self.repo.upsert(
                            workspace_id=workspace_id,
                            rule_type=rule.rule_type.value,
                            severity=condition.severity.value,
                            dedupe_key=condition.dedupe_key,
                            payload=condition.payload,
                            source="alert_evaluator",
                            job_run_id=job_run_id,
                            rule_version=rule.version,
                        )

                        # Record metric: alert triggered
                        record_ops_alert_triggered(
                            rule_type=rule.rule_type.value,
                            severity=condition.severity.value,
                        )

                        result.alerts_triggered += 1
                        if upsert_result.is_new:
                            result.alerts_new += 1
                            rule_result["new"] = True
                        else:
                            result.alerts_updated += 1

                        if upsert_result.escalated:
                            result.alerts_escalated += 1
                            rule_result["escalated"] = True

                        rule_result["alert_id"] = str(upsert_result.id)
                        rule_result["severity"] = condition.severity.value
                    else:
                        # Record metric: evaluation did not trigger
                        record_ops_alert_eval(
                            rule_type=rule.rule_type.value,
                            triggered=False,
                            skipped=False,
                        )

                result.by_rule_type[rule.rule_type.value] = rule_result

            except Exception as e:
                log.error(
                    "ops_alert_rule_error",
                    rule_type=rule.rule_type.value,
                    error=str(e),
                )
                result.errors.append(f"{rule.rule_type.value}: {str(e)}")
                result.by_rule_type[rule.rule_type.value] = {"error": str(e)}

        # Resolution pass - clear alerts whose conditions no longer hold
        resolved = await self._resolution_pass(workspace_id, triggered_keys, now)
        result.alerts_resolved = len(resolved)

        for alert in resolved:
            rule_key = alert.rule_type
            if rule_key in result.by_rule_type:
                result.by_rule_type[rule_key]["resolved"] = True
            else:
                result.by_rule_type[rule_key] = {"resolved": True}

        log.info(
            "ops_alert_eval_completed",
            conditions_evaluated=result.conditions_evaluated,
            alerts_triggered=result.alerts_triggered,
            alerts_new=result.alerts_new,
            alerts_resolved=result.alerts_resolved,
            errors=len(result.errors),
        )

        return result

    async def _build_context(
        self,
        workspace_id: UUID,
        now: datetime,
        job_run_id: Optional[UUID],
    ) -> EvalContext:
        """Build evaluation context with data sources."""
        ctx = EvalContext(
            workspace_id=workspace_id,
            now=now,
            job_run_id=job_run_id,
        )

        # Load health snapshot
        try:
            ctx.health_snapshot = await self._get_health_snapshot()
        except Exception as e:
            logger.warning("ops_alert_health_fetch_failed", error=str(e))

        # Load coverage stats
        try:
            ctx.coverage_stats = await self._get_coverage_stats(workspace_id)
        except Exception as e:
            logger.warning("ops_alert_coverage_fetch_failed", error=str(e))

        # Load match run stats for drift/confidence
        try:
            ctx.match_run_stats = await self._get_match_run_stats(workspace_id, now)
        except Exception as e:
            logger.warning("ops_alert_match_runs_fetch_failed", error=str(e))

        # Load strategy intel for confidence alerts (v1.5)
        try:
            ctx.strategy_intel = await self._get_strategy_intel(workspace_id)
        except Exception as e:
            logger.warning("ops_alert_strategy_intel_fetch_failed", error=str(e))

        # Load equity data for drawdown alerts
        try:
            ctx.equity_data = await self._get_equity_data(workspace_id)
            # Record drawdown metric for Grafana dashboards
            if ctx.equity_data and ctx.equity_data.get("drawdown_pct") is not None:
                set_workspace_drawdown(
                    str(workspace_id), ctx.equity_data["drawdown_pct"]
                )
        except Exception as e:
            logger.warning("ops_alert_equity_fetch_failed", error=str(e))

        return ctx

    async def _get_health_snapshot(self) -> Optional[Any]:
        """Get current system health snapshot."""
        try:
            from app.admin.services.health_checks import collect_system_health
            from app.config import get_settings

            settings = get_settings()
            return await collect_system_health(settings, self.pool)
        except Exception as e:
            logger.warning("health_snapshot_unavailable", error=str(e))
            return None

    async def _get_coverage_stats(self, workspace_id: UUID) -> dict:
        """
        Get coverage gap counts by priority tier.

        Returns:
            {
                "p1_open": count of open gaps with priority >= 0.75,
                "p2_open": count of open gaps with priority >= 0.40 and < 0.75,
                "total_open": total open gaps,
                "worst_score": lowest best_score among open gaps,
                "worst_run_id": run_id of worst gap,
            }
        """
        # priority_score is computed, not stored. Compute inline using the formula:
        # - base: (0.5 - best_score) clamped to [0, 0.5], or 0.5 if NULL
        # - +0.2 if num_above_threshold == 0
        # - +0.15 for NO_MATCHES reason code
        # - +0.1 for NO_STRONG_MATCHES reason code
        # - +0.05 recency bonus (last 24h)
        query = """
            WITH scored AS (
                SELECT
                    id,
                    best_score,
                    (
                        CASE WHEN best_score IS NULL THEN 0.5
                             ELSE GREATEST(0.0, LEAST(0.5, 0.5 - best_score))
                        END
                        + CASE WHEN num_above_threshold = 0 THEN 0.2 ELSE 0.0 END
                        + CASE WHEN 'NO_MATCHES' = ANY(reason_codes) THEN 0.15 ELSE 0.0 END
                        + CASE WHEN 'NO_STRONG_MATCHES' = ANY(reason_codes) THEN 0.1 ELSE 0.0 END
                        + CASE WHEN created_at > NOW() - INTERVAL '24 hours' THEN 0.05 ELSE 0.0 END
                    ) AS priority_score
                FROM match_runs
                WHERE workspace_id = $1
                  AND weak_coverage = true
                  AND coverage_status = 'open'
            )
            SELECT
                COUNT(*) FILTER (WHERE priority_score >= $2) AS p1_open,
                COUNT(*) FILTER (WHERE priority_score >= $3 AND priority_score < $2) AS p2_open,
                COUNT(*) AS total_open,
                MIN(best_score) AS worst_score,
                (SELECT id FROM scored ORDER BY priority_score DESC LIMIT 1) AS worst_run_id
            FROM scored
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                workspace_id,
                self.P1_PRIORITY_THRESHOLD,
                self.P2_PRIORITY_THRESHOLD,
            )

        return {
            "p1_open": row["p1_open"] or 0,
            "p2_open": row["p2_open"] or 0,
            "total_open": row["total_open"] or 0,
            "worst_score": row["worst_score"],
            "worst_run_id": row["worst_run_id"],
        }

    async def _get_match_run_stats(self, workspace_id: UUID, now: datetime) -> dict:
        """
        Get match run statistics for drift/confidence detection.

        Compares recent window (15m) to baseline (24h).

        Returns:
            {
                "count_15m": count in last 15 minutes,
                "weak_rate_15m": weak_coverage rate in last 15m,
                "avg_score_15m": average best_score in last 15m,
                "count_24h": count in last 24 hours,
                "weak_rate_24h": weak_coverage rate in last 24h,
                "avg_score_24h": average best_score in last 24h,
            }
        """
        t_15m = now - timedelta(minutes=15)
        t_24h = now - timedelta(hours=24)

        query = """
            SELECT
                COUNT(*) FILTER (WHERE created_at >= $2) AS count_15m,
                AVG(CASE WHEN weak_coverage AND created_at >= $2
                    THEN 1 ELSE 0 END) AS weak_rate_15m,
                AVG(best_score) FILTER (WHERE created_at >= $2) AS avg_score_15m,
                COUNT(*) FILTER (WHERE created_at >= $3) AS count_24h,
                AVG(CASE WHEN weak_coverage AND created_at >= $3
                    THEN 1 ELSE 0 END) AS weak_rate_24h,
                AVG(best_score) FILTER (WHERE created_at >= $3) AS avg_score_24h
            FROM match_runs
            WHERE workspace_id = $1 AND created_at >= $3
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, workspace_id, t_15m, t_24h)

        return {
            "count_15m": row["count_15m"] or 0,
            "weak_rate_15m": (
                float(row["weak_rate_15m"]) if row["weak_rate_15m"] else 0.0
            ),
            "avg_score_15m": (
                float(row["avg_score_15m"]) if row["avg_score_15m"] else None
            ),
            "count_24h": row["count_24h"] or 0,
            "weak_rate_24h": (
                float(row["weak_rate_24h"]) if row["weak_rate_24h"] else 0.0
            ),
            "avg_score_24h": (
                float(row["avg_score_24h"]) if row["avg_score_24h"] else None
            ),
        }

    async def _get_strategy_intel(self, workspace_id: UUID) -> list[dict]:
        """
        Get active strategy versions with their latest intel snapshots.

        Returns list of dicts with:
        - strategy_id, strategy_name, version_id
        - latest N intel snapshots (for persistence gating)
        - weakest confidence components
        """
        # Get all active versions for this workspace with recent intel
        query = """
            WITH active_versions AS (
                SELECT
                    v.id AS version_id,
                    v.strategy_id,
                    s.name AS strategy_name,
                    v.version_number,
                    v.version_tag
                FROM strategy_versions v
                JOIN strategies s ON v.strategy_id = s.id
                WHERE s.workspace_id = $1
                  AND v.state = 'active'
            ),
            recent_snapshots AS (
                SELECT
                    sis.strategy_version_id,
                    sis.as_of_ts,
                    sis.computed_at,
                    sis.regime,
                    sis.confidence_score,
                    sis.confidence_components,
                    ROW_NUMBER() OVER (
                        PARTITION BY sis.strategy_version_id
                        ORDER BY sis.as_of_ts DESC
                    ) AS rn
                FROM strategy_intel_snapshots sis
                JOIN active_versions av ON sis.strategy_version_id = av.version_id
            )
            SELECT
                av.version_id,
                av.strategy_id,
                av.strategy_name,
                av.version_number,
                av.version_tag,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'as_of_ts', rs.as_of_ts,
                            'computed_at', rs.computed_at,
                            'regime', rs.regime,
                            'confidence_score', rs.confidence_score,
                            'confidence_components', rs.confidence_components
                        ) ORDER BY rs.rn
                    ) FILTER (WHERE rs.rn <= 5),
                    '[]'::json
                ) AS recent_snapshots
            FROM active_versions av
            LEFT JOIN recent_snapshots rs ON rs.strategy_version_id = av.version_id
            GROUP BY av.version_id, av.strategy_id, av.strategy_name,
                     av.version_number, av.version_tag
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, workspace_id)

        result = []
        for row in rows:
            snapshots = row["recent_snapshots"] if row["recent_snapshots"] else []
            # Parse JSON if needed
            if isinstance(snapshots, str):
                import json

                snapshots = json.loads(snapshots)

            result.append(
                {
                    "version_id": row["version_id"],
                    "strategy_id": row["strategy_id"],
                    "strategy_name": row["strategy_name"],
                    "version_number": row["version_number"],
                    "version_tag": row["version_tag"],
                    "snapshots": snapshots,
                }
            )

        return result

    async def _get_equity_data(self, workspace_id: UUID) -> Optional[dict]:
        """
        Get equity drawdown data for a workspace.

        Uses the PaperEquityRepository to compute drawdown from equity snapshots.

        Returns:
            {
                "drawdown_pct": current drawdown percentage,
                "peak_equity": highest equity in window,
                "current_equity": current equity,
                "peak_ts": timestamp of peak,
                "current_ts": timestamp of current,
                "window_days": window used,
                "snapshot_count": number of snapshots in window,
            }
            or None if no equity data available
        """
        try:
            from app.repositories.paper_equity import PaperEquityRepository

            repo = PaperEquityRepository(self.pool)
            result = await repo.compute_drawdown(
                workspace_id, window_days=self.DRAWDOWN_WINDOW_DAYS
            )

            if result is None:
                return None

            return {
                "drawdown_pct": result.drawdown_pct,
                "peak_equity": result.peak_equity,
                "current_equity": result.current_equity,
                "peak_ts": result.peak_ts,
                "current_ts": result.current_ts,
                "window_days": result.window_days,
                "snapshot_count": result.snapshot_count,
            }
        except Exception as e:
            logger.warning("equity_data_unavailable", error=str(e))
            return None

    async def _evaluate_rule(
        self, rule: OpsAlertRule, ctx: EvalContext
    ) -> AlertCondition:
        """Evaluate a single rule against context."""

        if rule.rule_type == OpsRuleType.HEALTH_DEGRADED:
            return self._eval_health_degraded(rule, ctx)

        elif rule.rule_type == OpsRuleType.WEAK_COVERAGE_P1:
            return self._eval_weak_coverage_p1(rule, ctx)

        elif rule.rule_type == OpsRuleType.WEAK_COVERAGE_P2:
            return self._eval_weak_coverage_p2(rule, ctx)

        elif rule.rule_type == OpsRuleType.DRIFT_SPIKE:
            return self._eval_drift_spike(rule, ctx)

        elif rule.rule_type == OpsRuleType.CONFIDENCE_DROP:
            return self._eval_confidence_drop(rule, ctx)

        elif rule.rule_type == OpsRuleType.STRATEGY_CONFIDENCE_LOW:
            # Returns list of conditions (one per version)
            return await self._eval_strategy_confidence_low(rule, ctx)

        elif rule.rule_type == OpsRuleType.WORKSPACE_DRAWDOWN_HIGH:
            return self._eval_workspace_drawdown_high(rule, ctx)

        return AlertCondition(
            triggered=False,
            severity=rule.default_severity,
            skip_reason="unknown_rule_type",
        )

    def _eval_health_degraded(
        self, rule: OpsAlertRule, ctx: EvalContext
    ) -> AlertCondition:
        """Evaluate health_degraded rule."""
        if not ctx.health_snapshot:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason="health_unavailable",
            )

        overall = ctx.health_snapshot.overall_status
        bucket_key = rule.get_bucket_key(ctx.now)

        # Determine if triggered and severity
        if overall in self.HEALTH_ERROR_STATES:
            return AlertCondition(
                triggered=True,
                severity=Severity.CRITICAL,
                dedupe_key=rule.build_dedupe_key(bucket_key),
                payload={
                    "overall_status": overall,
                    "issues": ctx.health_snapshot.issues[:10],  # Cap at 10
                    "components_error": ctx.health_snapshot.components_error,
                    "components_degraded": ctx.health_snapshot.components_degraded,
                },
            )
        elif overall in self.HEALTH_DEGRADED_STATES:
            return AlertCondition(
                triggered=True,
                severity=Severity.HIGH,
                dedupe_key=rule.build_dedupe_key(bucket_key),
                payload={
                    "overall_status": overall,
                    "issues": ctx.health_snapshot.issues[:10],
                    "components_error": ctx.health_snapshot.components_error,
                    "components_degraded": ctx.health_snapshot.components_degraded,
                },
            )

        return AlertCondition(triggered=False, severity=rule.default_severity)

    def _eval_weak_coverage_p1(
        self, rule: OpsAlertRule, ctx: EvalContext
    ) -> AlertCondition:
        """Evaluate weak_coverage:P1 rule."""
        if not ctx.coverage_stats:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason="coverage_unavailable",
            )

        bucket_key = rule.get_bucket_key(ctx.now)
        p1_count = ctx.coverage_stats.get("p1_open", 0)

        if p1_count > 0:
            return AlertCondition(
                triggered=True,
                severity=Severity.HIGH,
                dedupe_key=rule.build_dedupe_key(bucket_key),
                payload={
                    "count": p1_count,
                    "worst_score": ctx.coverage_stats.get("worst_score"),
                    "worst_run_id": (
                        str(ctx.coverage_stats.get("worst_run_id"))
                        if ctx.coverage_stats.get("worst_run_id")
                        else None
                    ),
                    "threshold": self.P1_PRIORITY_THRESHOLD,
                },
            )

        return AlertCondition(triggered=False, severity=rule.default_severity)

    def _eval_weak_coverage_p2(
        self, rule: OpsAlertRule, ctx: EvalContext
    ) -> AlertCondition:
        """Evaluate weak_coverage:P2 rule."""
        if not ctx.coverage_stats:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason="coverage_unavailable",
            )

        bucket_key = rule.get_bucket_key(ctx.now)
        p2_count = ctx.coverage_stats.get("p2_open", 0)

        if p2_count > 0:
            return AlertCondition(
                triggered=True,
                severity=Severity.MEDIUM,
                dedupe_key=rule.build_dedupe_key(bucket_key),
                payload={
                    "count": p2_count,
                    "total_open": ctx.coverage_stats.get("total_open", 0),
                    "threshold_min": self.P2_PRIORITY_THRESHOLD,
                    "threshold_max": self.P1_PRIORITY_THRESHOLD,
                },
            )

        return AlertCondition(triggered=False, severity=rule.default_severity)

    def _eval_drift_spike(self, rule: OpsAlertRule, ctx: EvalContext) -> AlertCondition:
        """
        Evaluate drift_spike rule.

        Triggers if:
        - count_15m >= min_sample_count AND
        - (weak_rate_15m >= weak_rate_24h * 2 OR avg_score_15m <= avg_score_24h - 0.12)
        """
        if not ctx.match_run_stats:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason="match_runs_unavailable",
            )

        stats = ctx.match_run_stats
        count_15m = stats.get("count_15m", 0)

        # Volume gate
        if count_15m < rule.min_sample_count:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason=f"insufficient_volume:{count_15m}<{rule.min_sample_count}",
            )

        bucket_key = rule.get_bucket_key(ctx.now)
        weak_rate_15m = stats.get("weak_rate_15m", 0.0)
        weak_rate_24h = stats.get("weak_rate_24h", 0.0)
        avg_score_15m = stats.get("avg_score_15m")
        avg_score_24h = stats.get("avg_score_24h")

        # Check weak rate spike
        weak_rate_trigger = (
            weak_rate_24h > 0 and weak_rate_15m >= weak_rate_24h * self.DRIFT_THRESHOLD
        )

        # Check score drop
        score_drop_trigger = (
            avg_score_15m is not None
            and avg_score_24h is not None
            and avg_score_15m <= avg_score_24h - self.DRIFT_SCORE_DROP_THRESHOLD
        )

        if weak_rate_trigger or score_drop_trigger:
            return AlertCondition(
                triggered=True,
                severity=Severity.MEDIUM,
                dedupe_key=rule.build_dedupe_key(bucket_key),
                payload={
                    "weak_rate_15m": weak_rate_15m,
                    "weak_rate_24h": weak_rate_24h,
                    "avg_score_15m": avg_score_15m,
                    "avg_score_24h": avg_score_24h,
                    "count_15m": count_15m,
                    "trigger_reason": (
                        "weak_rate" if weak_rate_trigger else "score_drop"
                    ),
                    "thresholds": {
                        "weak_rate_multiplier": self.DRIFT_THRESHOLD,
                        "score_drop": self.DRIFT_SCORE_DROP_THRESHOLD,
                    },
                },
            )

        return AlertCondition(triggered=False, severity=rule.default_severity)

    def _eval_confidence_drop(
        self, rule: OpsAlertRule, ctx: EvalContext
    ) -> AlertCondition:
        """
        Evaluate confidence_drop rule.

        Triggers if:
        - count_15m >= min_sample_count AND
        - (avg_score_15m < CONFIDENCE_FLOOR OR avg_score_15m <= avg_score_24h - DELTA)
        """
        if not ctx.match_run_stats:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason="match_runs_unavailable",
            )

        stats = ctx.match_run_stats
        count_15m = stats.get("count_15m", 0)

        # Volume gate
        if count_15m < rule.min_sample_count:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason=f"insufficient_volume:{count_15m}<{rule.min_sample_count}",
            )

        bucket_key = rule.get_bucket_key(ctx.now)
        avg_score_15m = stats.get("avg_score_15m")
        avg_score_24h = stats.get("avg_score_24h")

        if avg_score_15m is None:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason="no_recent_scores",
            )

        # Check absolute floor
        floor_trigger = avg_score_15m < self.CONFIDENCE_FLOOR

        # Check relative drop
        delta_trigger = (
            avg_score_24h is not None
            and avg_score_15m <= avg_score_24h - self.CONFIDENCE_DELTA_THRESHOLD
        )

        if floor_trigger or delta_trigger:
            return AlertCondition(
                triggered=True,
                severity=Severity.MEDIUM,
                dedupe_key=rule.build_dedupe_key(bucket_key),
                payload={
                    "avg_score_15m": avg_score_15m,
                    "avg_score_24h": avg_score_24h,
                    "count_15m": count_15m,
                    "trigger_reason": "floor" if floor_trigger else "delta",
                    "thresholds": {
                        "floor": self.CONFIDENCE_FLOOR,
                        "delta": self.CONFIDENCE_DELTA_THRESHOLD,
                    },
                },
            )

        return AlertCondition(triggered=False, severity=rule.default_severity)

    async def _eval_strategy_confidence_low(
        self, rule: OpsAlertRule, ctx: EvalContext
    ) -> list[AlertCondition]:
        """
        Evaluate strategy_confidence_low rule.

        Returns a list of AlertConditions (one per active version with low confidence).

        Triggers if:
        - Version has >= persistence_count consecutive snapshots below threshold
        - Warn: score < 0.35
        - Critical: score < 0.20

        Dedupe key format: strategy_confidence_low:{version_id}:{severity_bucket}:{date}
        """
        if ctx.strategy_intel is None:
            return [
                AlertCondition(
                    triggered=False,
                    severity=rule.default_severity,
                    skip_reason="strategy_intel_unavailable",
                )
            ]

        if not ctx.strategy_intel:
            return [
                AlertCondition(
                    triggered=False,
                    severity=rule.default_severity,
                    skip_reason="no_active_versions",
                )
            ]

        conditions: list[AlertCondition] = []
        bucket_key = rule.get_bucket_key(ctx.now)
        persistence_required = rule.persistence_count

        for version_data in ctx.strategy_intel:
            version_id = version_data["version_id"]
            snapshots = version_data.get("snapshots", [])

            # Skip if no snapshots
            if not snapshots:
                continue

            # Check consecutive low scores
            consecutive_warn = 0
            consecutive_critical = 0

            for snap in snapshots[: persistence_required + 1]:
                score = snap.get("confidence_score")
                if score is None:
                    break

                if score < self.STRATEGY_CONFIDENCE_CRITICAL:
                    consecutive_critical += 1
                    consecutive_warn += 1  # Critical also counts as warn
                elif score < self.STRATEGY_CONFIDENCE_WARN:
                    consecutive_warn += 1
                    consecutive_critical = 0  # Reset critical streak
                else:
                    break  # Score is acceptable, stop counting

            # Determine severity based on persistence
            severity = None
            severity_bucket = None

            if consecutive_critical >= persistence_required:
                severity = Severity.HIGH
                severity_bucket = "critical"
            elif consecutive_warn >= persistence_required:
                severity = Severity.MEDIUM
                severity_bucket = "warn"

            if severity:
                latest = snapshots[0]
                components = latest.get("confidence_components", {})

                # Find weakest components
                weak_components = []
                if components:
                    sorted_components = sorted(
                        components.items(), key=lambda x: x[1] if x[1] else 1.0
                    )
                    weak_components = [
                        {"name": k, "score": v}
                        for k, v in sorted_components[:3]
                        if v is not None
                    ]

                dedupe_key = f"strategy_confidence_low:{version_id}:{severity_bucket}:{bucket_key}"

                conditions.append(
                    AlertCondition(
                        triggered=True,
                        severity=severity,
                        dedupe_key=dedupe_key,
                        payload={
                            "strategy_id": str(version_data["strategy_id"]),
                            "strategy_name": version_data["strategy_name"],
                            "strategy_version_id": str(version_id),
                            "version_number": version_data["version_number"],
                            "version_tag": version_data.get("version_tag"),
                            "regime": latest.get("regime"),
                            "confidence_score": latest.get("confidence_score"),
                            "as_of_ts": (
                                latest["as_of_ts"].isoformat()
                                if hasattr(latest.get("as_of_ts"), "isoformat")
                                else str(latest.get("as_of_ts"))
                            ),
                            "computed_at": (
                                latest["computed_at"].isoformat()
                                if hasattr(latest.get("computed_at"), "isoformat")
                                else str(latest.get("computed_at"))
                            ),
                            "weak_components": weak_components,
                            "consecutive_low_count": (
                                consecutive_critical
                                if severity_bucket == "critical"
                                else consecutive_warn
                            ),
                            "thresholds": {
                                "warn": self.STRATEGY_CONFIDENCE_WARN,
                                "critical": self.STRATEGY_CONFIDENCE_CRITICAL,
                                "persistence_required": persistence_required,
                            },
                        },
                    )
                )

        # Return at least one condition (even if nothing triggered)
        if not conditions:
            return [
                AlertCondition(
                    triggered=False,
                    severity=rule.default_severity,
                )
            ]

        return conditions

    def _eval_workspace_drawdown_high(
        self, rule: OpsAlertRule, ctx: EvalContext
    ) -> AlertCondition:
        """
        Evaluate workspace_drawdown_high rule.

        Triggers if:
        - Drawdown exceeds DRAWDOWN_WARN (12%) for warn severity
        - Drawdown exceeds DRAWDOWN_CRITICAL (20%) for critical (HIGH) severity

        Hysteresis:
        - Clear warn when DD < DRAWDOWN_CLEAR_WARN (10%)
        - Clear critical when DD < DRAWDOWN_CLEAR_CRITICAL (16%)

        Dedupe key format: workspace_drawdown_high:{severity_bucket}:{date}
        """
        if ctx.equity_data is None:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason="equity_data_unavailable",
            )

        drawdown_pct = ctx.equity_data.get("drawdown_pct", 0.0)
        snapshot_count = ctx.equity_data.get("snapshot_count", 0)

        # Skip if insufficient data
        if snapshot_count < 2:
            return AlertCondition(
                triggered=False,
                severity=rule.default_severity,
                skip_reason=f"insufficient_snapshots:{snapshot_count}<2",
            )

        bucket_key = rule.get_bucket_key(ctx.now)

        # Determine severity based on drawdown level
        severity = None
        severity_bucket = None

        if drawdown_pct >= self.DRAWDOWN_CRITICAL:
            severity = Severity.HIGH
            severity_bucket = "critical"
        elif drawdown_pct >= self.DRAWDOWN_WARN:
            severity = Severity.MEDIUM
            severity_bucket = "warn"

        if severity:
            dedupe_key = f"workspace_drawdown_high:{severity_bucket}:{bucket_key}"

            return AlertCondition(
                triggered=True,
                severity=severity,
                dedupe_key=dedupe_key,
                payload={
                    "workspace_id": str(ctx.workspace_id),
                    "drawdown_pct": drawdown_pct,
                    "peak_equity": ctx.equity_data.get("peak_equity"),
                    "current_equity": ctx.equity_data.get("current_equity"),
                    "peak_ts": (
                        ctx.equity_data["peak_ts"].isoformat()
                        if hasattr(ctx.equity_data.get("peak_ts"), "isoformat")
                        else str(ctx.equity_data.get("peak_ts"))
                    ),
                    "current_ts": (
                        ctx.equity_data["current_ts"].isoformat()
                        if hasattr(ctx.equity_data.get("current_ts"), "isoformat")
                        else str(ctx.equity_data.get("current_ts"))
                    ),
                    "window_days": ctx.equity_data.get("window_days"),
                    "snapshot_count": snapshot_count,
                    "thresholds": {
                        "warn": self.DRAWDOWN_WARN,
                        "critical": self.DRAWDOWN_CRITICAL,
                        "clear_warn": self.DRAWDOWN_CLEAR_WARN,
                        "clear_critical": self.DRAWDOWN_CLEAR_CRITICAL,
                    },
                },
            )

        return AlertCondition(triggered=False, severity=rule.default_severity)

    async def _resolution_pass(
        self,
        workspace_id: UUID,
        triggered_keys: set[str],
        now: datetime,
    ) -> list[OpsAlert]:
        """
        Resolve alerts whose conditions cleared.

        For daily singleton rules (health, coverage), if the rule didn't trigger
        today, resolve any active alerts with today's dedupe key.

        For strategy confidence alerts, resolve when:
        - The dedupe key was not triggered this pass (score recovered)

        For workspace drawdown alerts, resolve when:
        - The dedupe key was not triggered (DD recovered below threshold)
        """
        resolved: list[OpsAlert] = []
        today = now.strftime("%Y-%m-%d")

        # Singleton rules that should auto-resolve when condition clears
        singleton_rules = [
            (OpsRuleType.HEALTH_DEGRADED, f"health_degraded:{today}"),
            (OpsRuleType.WEAK_COVERAGE_P1, f"weak_coverage:P1:{today}"),
            (OpsRuleType.WEAK_COVERAGE_P2, f"weak_coverage:P2:{today}"),
        ]

        for rule_type, expected_key in singleton_rules:
            if expected_key not in triggered_keys:
                # Condition cleared - resolve if active
                alert = await self.repo.resolve_by_dedupe_key(
                    workspace_id, expected_key
                )
                if alert:
                    # Record metric: alert resolved
                    record_ops_alert_resolved(rule_type=rule_type.value)
                    logger.info(
                        "ops_alert_auto_resolved",
                        workspace_id=str(workspace_id),
                        rule_type=rule_type.value,
                        dedupe_key=expected_key,
                    )
                    resolved.append(alert)

        # Strategy confidence alerts - resolve any active that weren't triggered
        # These have dedupe_key format: strategy_confidence_low:{version_id}:{severity}:{date}
        active_strategy_keys = await self.repo.get_active_dedupe_keys(
            workspace_id, rule_type_prefix="strategy_confidence_low"
        )

        for dedupe_key in active_strategy_keys:
            if dedupe_key not in triggered_keys:
                # This alert's condition cleared (score recovered above threshold)
                alert = await self.repo.resolve_by_dedupe_key(workspace_id, dedupe_key)
                if alert:
                    # Record metric: alert resolved
                    record_ops_alert_resolved(rule_type="strategy_confidence_low")
                    logger.info(
                        "ops_alert_auto_resolved",
                        workspace_id=str(workspace_id),
                        rule_type="strategy_confidence_low",
                        dedupe_key=dedupe_key,
                    )
                    resolved.append(alert)

        # Workspace drawdown alerts - resolve any active that weren't triggered
        # These have dedupe_key format: workspace_drawdown_high:{severity}:{date}
        active_drawdown_keys = await self.repo.get_active_dedupe_keys(
            workspace_id, rule_type_prefix="workspace_drawdown_high"
        )

        for dedupe_key in active_drawdown_keys:
            if dedupe_key not in triggered_keys:
                # This alert's condition cleared (DD recovered below threshold)
                alert = await self.repo.resolve_by_dedupe_key(workspace_id, dedupe_key)
                if alert:
                    # Record metric: alert resolved
                    record_ops_alert_resolved(rule_type="workspace_drawdown_high")
                    logger.info(
                        "ops_alert_auto_resolved",
                        workspace_id=str(workspace_id),
                        rule_type="workspace_drawdown_high",
                        dedupe_key=dedupe_key,
                    )
                    resolved.append(alert)

        return resolved
