"""OpsAlertEval job handler - evaluates operational alert rules.

This handler evaluates health, coverage, drift, and confidence rules
for one or all workspaces, upserts triggered alerts, resolves cleared
conditions, and sends Telegram notifications.

Hybrid workspace selection:
- If workspace_id provided: evaluate just that workspace
- If not provided: iterate all active workspaces
"""

from typing import Any
from uuid import UUID

import structlog

from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType
from app.repositories.ops_alerts import OpsAlertsRepository, OpsAlert
from app.services.ops_alerts.evaluator import OpsAlertEvaluator
from app.services.ops_alerts.telegram import get_telegram_notifier

logger = structlog.get_logger(__name__)


@default_registry.handler(JobType.OPS_ALERT_EVAL)
async def handle_ops_alert_eval(job: Job, ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Handle an OPS_ALERT_EVAL job.

    Evaluates operational alert rules and sends notifications.

    Job Payload:
        workspace_id: UUID (optional) - If provided, evaluate just this workspace
        dry_run: bool (optional) - If True, evaluate but don't write alerts or notify

    Context:
        pool: Database connection pool

    Returns:
        dict with:
            workspaces_evaluated: int
            total_conditions: int
            total_triggered: int
            total_new: int
            total_resolved: int
            telegram_sent: int
            errors: list[str]
            by_workspace: dict[str, EvalResult summary]
    """
    pool = ctx["pool"]
    payload = job.payload

    # Parse payload
    workspace_id_str = payload.get("workspace_id")
    workspace_id = UUID(workspace_id_str) if workspace_id_str else None
    dry_run = payload.get("dry_run", False)

    log = logger.bind(
        job_id=str(job.id),
        workspace_id=str(workspace_id) if workspace_id else "all",
        dry_run=dry_run,
    )
    log.info("ops_alert_eval_started")

    # Initialize components
    repo = OpsAlertsRepository(pool)
    evaluator = OpsAlertEvaluator(repo, pool)
    notifier = get_telegram_notifier() if not dry_run else None

    # Get workspaces to evaluate
    if workspace_id:
        workspace_ids = [workspace_id]
    else:
        workspace_ids = await _get_active_workspaces(pool)

    if not workspace_ids:
        log.info("ops_alert_eval_no_workspaces")
        return {
            "workspaces_evaluated": 0,
            "total_conditions": 0,
            "total_triggered": 0,
            "total_new": 0,
            "total_resolved": 0,
            "telegram_sent": 0,
            "errors": [],
            "by_workspace": {},
        }

    # Aggregate results
    result: dict[str, Any] = {
        "workspaces_evaluated": 0,
        "total_conditions": 0,
        "total_triggered": 0,
        "total_new": 0,
        "total_resolved": 0,
        "total_escalated": 0,
        "telegram_sent": 0,
        "errors": [],
        "by_workspace": {},
    }

    # Evaluate each workspace
    for ws_id in workspace_ids:
        try:
            eval_result = await evaluator.evaluate(
                workspace_id=ws_id,
                job_run_id=job.id,
            )

            result["workspaces_evaluated"] += 1
            result["total_conditions"] += eval_result.conditions_evaluated
            result["total_triggered"] += eval_result.alerts_triggered
            result["total_new"] += eval_result.alerts_new
            result["total_resolved"] += eval_result.alerts_resolved
            result["total_escalated"] += eval_result.alerts_escalated

            if eval_result.errors:
                result["errors"].extend([f"{ws_id}:{e}" for e in eval_result.errors])

            # Collect alerts to notify
            if notifier and not dry_run:
                sent = await _send_notifications(
                    notifier=notifier,
                    repo=repo,
                    eval_result=eval_result,
                    workspace_id=ws_id,
                )
                result["telegram_sent"] += sent

            # Summarize per workspace
            result["by_workspace"][str(ws_id)] = {
                "conditions": eval_result.conditions_evaluated,
                "triggered": eval_result.alerts_triggered,
                "new": eval_result.alerts_new,
                "resolved": eval_result.alerts_resolved,
                "escalated": eval_result.alerts_escalated,
                "errors": len(eval_result.errors),
            }

        except Exception as e:
            log.error(
                "ops_alert_eval_workspace_error",
                workspace_id=str(ws_id),
                error=str(e),
            )
            result["errors"].append(f"{ws_id}:job_error:{str(e)}")

    log.info(
        "ops_alert_eval_completed",
        workspaces=result["workspaces_evaluated"],
        triggered=result["total_triggered"],
        new=result["total_new"],
        resolved=result["total_resolved"],
        telegram_sent=result["telegram_sent"],
        errors=len(result["errors"]),
    )

    return result


async def _get_active_workspaces(pool) -> list[UUID]:
    """Get list of active workspace IDs."""
    query = """
        SELECT id FROM workspaces
        WHERE is_active = true
        ORDER BY created_at
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    return [r["id"] for r in rows]


async def _send_notifications(
    notifier,
    repo: OpsAlertsRepository,
    eval_result,
    workspace_id: UUID,
) -> int:
    """
    Send Telegram notifications for new, resolved, and escalated alerts.

    Returns count of messages sent.
    """
    sent = 0

    # Collect alerts to notify
    alerts_to_notify: list[tuple[OpsAlert, bool, bool]] = (
        []
    )  # (alert, is_recovery, is_escalation)

    for rule_type, details in eval_result.by_rule_type.items():
        alert_id_str = details.get("alert_id")
        if not alert_id_str:
            continue

        alert_id = UUID(alert_id_str)

        # New alert
        if details.get("new"):
            alert = await repo.get(alert_id)
            if alert:
                alerts_to_notify.append((alert, False, False))

        # Escalated alert
        elif details.get("escalated"):
            alert = await repo.get(alert_id)
            if alert:
                alerts_to_notify.append((alert, False, True))

        # Resolved alert
        if details.get("resolved"):
            # For resolved, we need to find the alert that was resolved
            # The alert_id in details is from the upsert, not the resolved one
            # We need to look it up differently
            pass

    # Handle resolved alerts
    for rule_type, details in eval_result.by_rule_type.items():
        if details.get("resolved"):
            # Find the most recently resolved alert for this rule type
            alerts, _ = await repo.list_alerts(
                workspace_id=workspace_id,
                status=["resolved"],
                rule_type=[rule_type],
                limit=1,
            )
            if alerts:
                alerts_to_notify.append((alerts[0], True, False))

    # Send notifications
    if alerts_to_notify:
        sent = await notifier.send_batch(alerts_to_notify)

    return sent
