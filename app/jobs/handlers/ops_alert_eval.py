"""OpsAlertEval job handler - evaluates operational alert rules.

This handler evaluates health, coverage, drift, and confidence rules
for one or all workspaces, upserts triggered alerts, resolves cleared
conditions, and sends notifications via Telegram and Discord.

Hybrid workspace selection:
- If workspace_id provided: evaluate just that workspace
- If not provided: iterate all active workspaces
"""

from typing import Any, Optional
from uuid import UUID

import structlog

from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType
from app.repositories.ops_alerts import OpsAlertsRepository
from app.services.ops_alerts.evaluator import OpsAlertEvaluator
from app.services.ops_alerts.telegram import get_telegram_notifier
from app.services.ops_alerts.discord import get_discord_notifier

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
    telegram_notifier = get_telegram_notifier() if not dry_run else None
    discord_notifier = get_discord_notifier() if not dry_run else None

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
            "discord_sent": 0,
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
        "discord_sent": 0,
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

            # Send pending notifications (idempotent - queries DB state)
            if not dry_run:
                telegram_sent, discord_sent = await _send_notifications(
                    telegram_notifier=telegram_notifier,
                    discord_notifier=discord_notifier,
                    repo=repo,
                    workspace_id=ws_id,
                )
                result["telegram_sent"] += telegram_sent
                result["discord_sent"] += discord_sent

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
        discord_sent=result["discord_sent"],
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
    telegram_notifier: Optional[Any],
    discord_notifier: Optional[Any],
    repo: OpsAlertsRepository,
    workspace_id: UUID,
) -> tuple[int, int]:
    """
    Send pending notifications for a workspace via Telegram and Discord.

    Uses DB state as source of truth (not eval_result) for idempotency.
    Conditional mark prevents duplicate notifications from concurrent workers.

    Both notifiers receive the same alerts. DB marking happens once per alert
    (not per channel) - if either channel succeeds, we mark it notified.

    Returns tuple of (telegram_sent, discord_sent) counts.
    """
    telegram_sent = 0
    discord_sent = 0

    if not telegram_notifier and not discord_notifier:
        return (0, 0)

    # Query DB for all pending notifications
    pending = await repo.get_pending_notifications(workspace_id)

    # Process activations (new alerts)
    for alert in pending["activations"]:
        t_ok, d_ok = await _send_to_channels(
            alert,
            telegram_notifier,
            discord_notifier,
            is_recovery=False,
            is_escalation=False,
        )
        if t_ok:
            telegram_sent += 1
        if d_ok:
            discord_sent += 1

        # Mark notified if at least one channel succeeded
        if t_ok or d_ok:
            try:
                await repo.mark_notified(alert.id, "activation", None)
            except Exception as e:
                logger.warning(
                    "mark_notified_error",
                    alert_id=str(alert.id),
                    error=str(e),
                )

    # Process recoveries (resolved alerts)
    for alert in pending["recoveries"]:
        t_ok, d_ok = await _send_to_channels(
            alert,
            telegram_notifier,
            discord_notifier,
            is_recovery=True,
            is_escalation=False,
        )
        if t_ok:
            telegram_sent += 1
        if d_ok:
            discord_sent += 1

        if t_ok or d_ok:
            try:
                await repo.mark_notified(alert.id, "recovery", None)
            except Exception as e:
                logger.warning(
                    "mark_notified_error",
                    alert_id=str(alert.id),
                    error=str(e),
                )

    # Process escalations (severity bumped on already-activated alerts)
    for alert in pending["escalations"]:
        t_ok, d_ok = await _send_to_channels(
            alert,
            telegram_notifier,
            discord_notifier,
            is_recovery=False,
            is_escalation=True,
        )
        if t_ok:
            telegram_sent += 1
        if d_ok:
            discord_sent += 1

        if t_ok or d_ok:
            try:
                await repo.mark_notified(alert.id, "escalation", None)
            except Exception as e:
                logger.warning(
                    "mark_notified_error",
                    alert_id=str(alert.id),
                    error=str(e),
                )

    return (telegram_sent, discord_sent)


async def _send_to_channels(
    alert,
    telegram_notifier: Optional[Any],
    discord_notifier: Optional[Any],
    is_recovery: bool,
    is_escalation: bool,
) -> tuple[bool, bool]:
    """
    Send alert to both Telegram and Discord channels.

    Returns tuple of (telegram_ok, discord_ok) indicating success for each.
    Failures are logged but don't prevent the other channel from sending.
    """
    telegram_ok = False
    discord_ok = False

    # Send to Telegram
    if telegram_notifier:
        try:
            result = await telegram_notifier.send_alert(
                alert, is_recovery=is_recovery, is_escalation=is_escalation
            )
            telegram_ok = result.ok
        except Exception as e:
            logger.warning(
                "telegram_delivery_error",
                alert_id=str(alert.id),
                error=str(e),
            )

    # Send to Discord
    if discord_notifier:
        try:
            result = await discord_notifier.send_alert(
                alert, is_recovery=is_recovery, is_escalation=is_escalation
            )
            discord_ok = result.ok
        except Exception as e:
            logger.warning(
                "discord_delivery_error",
                alert_id=str(alert.id),
                error=str(e),
            )

    return (telegram_ok, discord_ok)
