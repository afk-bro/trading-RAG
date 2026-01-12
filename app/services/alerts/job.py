"""Alert evaluator job - scheduled evaluation of alert rules."""

from typing import Any
from uuid import UUID

import structlog

from app.repositories.alerts import AlertsRepository
from app.services.alerts.evaluators import RuleEvaluator
from app.services.alerts.models import (
    ConfidenceDropConfig,
    DriftSpikeConfig,
    RuleType,
)
from app.services.alerts.transitions import AlertTransitionManager

logger = structlog.get_logger(__name__)


class AlertEvaluatorJob:
    """Job for evaluating alert rules on schedule."""

    def __init__(self, pool):
        """Initialize with database pool."""
        self.pool = pool

    async def run(
        self,
        workspace_id: UUID,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Run alert evaluation for workspace.

        Returns job result with metrics.
        """
        metrics = {
            "rules_loaded": 0,
            "tuples_evaluated": 0,
            "tuples_skipped_insufficient_data": 0,
            "activations_suppressed_cooldown": 0,
            "alerts_activated": 0,
            "alerts_resolved": 0,
            "db_upserts": 0,
            "db_updates": 0,
            "evaluation_errors": 0,
        }

        async with self.pool.acquire() as conn:
            # Acquire advisory lock
            lock_key = hash(f"evaluate_alerts:{workspace_id}") % (2**31)
            lock_acquired = await conn.fetchval(
                "SELECT pg_try_advisory_lock($1)", lock_key
            )

            if not lock_acquired:
                return {
                    "lock_acquired": False,
                    "status": "already_running",
                    "metrics": metrics,
                }

            try:
                # Load enabled rules
                repo = AlertsRepository(self.pool)
                rules = await repo.list_rules(workspace_id, enabled_only=True)
                metrics["rules_loaded"] = len(rules)

                if not rules:
                    return {
                        "lock_acquired": True,
                        "status": "completed",
                        "metrics": metrics,
                    }

                # Initialize components
                evaluator = RuleEvaluator()
                transition_mgr = AlertTransitionManager(repo)

                # Process each rule
                for rule in rules:
                    try:
                        await self._process_rule(
                            rule=rule,
                            evaluator=evaluator,
                            transition_mgr=transition_mgr,
                            metrics=metrics,
                            dry_run=dry_run,
                        )
                    except Exception as e:
                        logger.exception(
                            "Rule evaluation failed", rule_id=rule["id"], error=str(e)
                        )
                        metrics["evaluation_errors"] += 1

            finally:
                # Release lock
                try:
                    await conn.fetchval("SELECT pg_advisory_unlock($1)", lock_key)
                except Exception as e:
                    logger.exception(
                        "Failed to release advisory lock", lock_key=lock_key, error=str(e)
                    )

        return {
            "lock_acquired": True,
            "status": "completed",
            "metrics": metrics,
        }

    async def _process_rule(
        self,
        rule: dict,
        evaluator: RuleEvaluator,
        transition_mgr: AlertTransitionManager,
        metrics: dict,
        dry_run: bool,
    ) -> None:
        """Process a single alert rule."""
        rule_type = RuleType(rule["rule_type"])
        workspace_id = rule["workspace_id"]
        strategy_entity_id = rule.get("strategy_entity_id")
        regime_key = rule.get("regime_key")
        timeframe = rule.get("timeframe") or "1h"
        config = rule.get("config") or {}
        cooldown_minutes = rule.get("cooldown_minutes", 60)

        # For v1, require explicit strategy_entity_id and regime_key
        if not strategy_entity_id:
            logger.info(
                "Skipping rule with NULL strategy_entity_id", rule_id=rule["id"]
            )
            return

        if not regime_key:
            logger.info("Skipping rule with NULL regime_key", rule_id=rule["id"])
            return

        # Fetch bucket data
        buckets = await self._fetch_buckets(
            workspace_id=workspace_id,
            strategy_entity_id=strategy_entity_id,
            regime_key=regime_key,
            timeframe=timeframe,
        )

        metrics["tuples_evaluated"] += 1

        # Build fingerprint
        fingerprint = f"v1:{regime_key}:{timeframe}"

        # Evaluate based on rule type
        if rule_type == RuleType.DRIFT_SPIKE:
            drift_config = DriftSpikeConfig(**config)
            eval_result = evaluator.evaluate_drift_spike(buckets, drift_config)

        elif rule_type == RuleType.CONFIDENCE_DROP:
            confidence_config = ConfidenceDropConfig(**config)
            eval_result = evaluator.evaluate_confidence_drop(buckets, confidence_config)

        elif rule_type == RuleType.COMBO:
            drift_config = DriftSpikeConfig(**config.get("drift", {}))
            confidence_config = ConfidenceDropConfig(**config.get("confidence", {}))
            eval_result = evaluator.evaluate_combo(
                buckets, drift_config, confidence_config
            )
        else:
            return

        # Track insufficient data
        if eval_result.insufficient_data:
            metrics["tuples_skipped_insufficient_data"] += 1
            return

        if dry_run:
            return

        # Process transition
        result = await transition_mgr.process_evaluation(
            eval_result=eval_result,
            workspace_id=workspace_id,
            rule_id=rule["id"],
            strategy_entity_id=strategy_entity_id,
            regime_key=regime_key,
            timeframe=timeframe,
            rule_type=rule_type,
            fingerprint=fingerprint,
            cooldown_minutes=cooldown_minutes,
        )

        # Update metrics
        action = result.get("action", "no_change")
        if action == "activated":
            metrics["alerts_activated"] += 1
            metrics["db_upserts"] += 1
        elif action == "resolved":
            metrics["alerts_resolved"] += 1
            metrics["db_updates"] += 1
        elif action == "updated_last_seen":
            metrics["db_updates"] += 1
        elif action == "suppressed_cooldown":
            metrics["activations_suppressed_cooldown"] += 1

    async def _fetch_buckets(
        self,
        workspace_id: UUID,
        strategy_entity_id: UUID,
        regime_key: str,
        timeframe: str,
    ) -> list:
        """
        Fetch drift/confidence bucket data.

        TODO: Integrate with existing analytics queries.
        """
        # Placeholder - integrate with existing regime drift queries
        return []
