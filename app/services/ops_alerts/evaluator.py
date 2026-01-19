"""Ops alert evaluator - evaluates rules and manages alert lifecycle."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

import structlog

from app.repositories.ops_alerts import OpsAlertsRepository, OpsAlert
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
                condition = await self._evaluate_rule(rule, ctx)
                result.conditions_evaluated += 1

                rule_result: dict[str, Any] = {
                    "triggered": condition.triggered,
                }

                if condition.skip_reason:
                    rule_result["skipped"] = condition.skip_reason
                    log.debug(
                        "ops_alert_rule_skipped",
                        rule_type=rule.rule_type.value,
                        reason=condition.skip_reason,
                    )
                elif condition.triggered:
                    triggered_keys.add(condition.dedupe_key)

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
                    logger.info(
                        "ops_alert_auto_resolved",
                        workspace_id=str(workspace_id),
                        rule_type=rule_type.value,
                        dedupe_key=expected_key,
                    )
                    resolved.append(alert)

        return resolved
