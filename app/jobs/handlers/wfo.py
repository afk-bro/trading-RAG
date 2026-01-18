"""WFOJob handler - runs walk-forward optimization via job queue.

This handler orchestrates WFO by:
1. Parsing job payload (wfo config, strategy, data source, param space)
2. Querying available data range
3. Generating folds based on config and available data
4. Enqueuing child TuneJobs for each fold (with parent_job_id)
5. Polling until all children complete (respecting cancellation)
6. Aggregating results and generating WFO artifacts
7. Recording artifacts in artifact_index
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog

from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType, JobStatus
from app.repositories.backtests import TuneRepository
from app.repositories.jobs import JobRepository
from app.repositories.job_events import JobEventsRepository
from app.repositories.ohlcv import OHLCVRepository
from app.services.backtest.wfo import (
    WFOConfig,
    Fold,
    InsufficientDataError,
    generate_folds,
)

logger = structlog.get_logger(__name__)

# Polling interval for child job completion (seconds)
CHILD_POLL_INTERVAL = 5

# Max time to wait for child jobs (seconds) - 4 hours
CHILD_TIMEOUT = 4 * 60 * 60


def parse_iso_timestamp(value: str) -> datetime:
    """Parse ISO format timestamp string to datetime."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _build_child_tune_payload(
    fold: Fold,
    base_payload: dict[str, Any],
    wfo_id: UUID,
    fold_index: int,
) -> dict[str, Any]:
    """Build payload for child tune job.

    Args:
        fold: The fold being processed
        base_payload: Base payload with strategy/param config
        wfo_id: Parent WFO job ID
        fold_index: Fold index for identification

    Returns:
        Complete payload for enqueuing child tune job
    """
    # Copy base payload and override data source dates
    payload = dict(base_payload)

    # Update data_source with fold train dates
    data_source = dict(payload.get("data_source", {}))
    data_source["start_ts"] = fold.train_start.isoformat()
    data_source["end_ts"] = fold.train_end.isoformat()
    payload["data_source"] = data_source

    # Add fold metadata
    payload["wfo_fold_index"] = fold_index
    payload["wfo_parent_id"] = str(wfo_id)

    # Set OOS ratio based on fold test window
    # The tune will use the test period as OOS
    total_days = fold.train_days + fold.test_days
    payload["oos_ratio"] = fold.test_days / total_days if total_days > 0 else 0.0

    return payload


async def _wait_for_children(
    job_repo: JobRepository,
    parent_job_id: UUID,
    events_repo: JobEventsRepository,
    log: Any,
) -> tuple[list[Job], list[Job], list[Job]]:
    """Wait for all child jobs to complete.

    Args:
        job_repo: Job repository
        parent_job_id: Parent WFO job ID
        events_repo: Events repo for logging
        log: Structured logger

    Returns:
        Tuple of (succeeded, failed, canceled) job lists
    """
    start_time = datetime.now(timezone.utc)
    last_status_log = start_time

    while True:
        # Check for timeout
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        if elapsed > CHILD_TIMEOUT:
            log.warning("wfo_child_timeout", elapsed_seconds=elapsed)
            await events_repo.warning(
                parent_job_id,
                f"Child jobs timed out after {elapsed:.0f}s",
            )
            break

        # Get all child jobs
        children = await job_repo.list_children(parent_job_id)

        # Categorize by status
        pending = [j for j in children if j.status == JobStatus.PENDING]
        running = [j for j in children if j.status == JobStatus.RUNNING]
        succeeded = [j for j in children if j.status == JobStatus.SUCCEEDED]
        failed = [j for j in children if j.status == JobStatus.FAILED]
        canceled = [j for j in children if j.status == JobStatus.CANCELED]

        # Check if all done
        if not pending and not running:
            log.info(
                "wfo_children_complete",
                succeeded=len(succeeded),
                failed=len(failed),
                canceled=len(canceled),
            )
            return succeeded, failed, canceled

        # Log status periodically (every 30s)
        now = datetime.now(timezone.utc)
        if (now - last_status_log).total_seconds() >= 30:
            log.info(
                "wfo_children_progress",
                pending=len(pending),
                running=len(running),
                succeeded=len(succeeded),
                failed=len(failed),
            )
            await events_repo.info(
                parent_job_id,
                f"Progress: {len(succeeded)} succeeded, "
                f"{len(running)} running, {len(pending)} pending",
            )
            last_status_log = now

        # Check if parent was canceled
        parent = await job_repo.get(parent_job_id)
        if parent and parent.status == JobStatus.CANCELED:
            log.info("wfo_parent_canceled_stopping")

            # Cancel pending children
            for child in pending:
                await job_repo.update_status(child.id, JobStatus.CANCELED)

            return succeeded, failed, canceled + pending

        await asyncio.sleep(CHILD_POLL_INTERVAL)

    # Timeout case - return what we have
    children = await job_repo.list_children(parent_job_id)
    succeeded = [j for j in children if j.status == JobStatus.SUCCEEDED]
    failed = [j for j in children if j.status == JobStatus.FAILED]
    canceled = [j for j in children if j.status == JobStatus.CANCELED]
    # Treat pending/running as failed on timeout
    timed_out = [
        j for j in children
        if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
    ]
    return succeeded, failed + timed_out, canceled


@default_registry.handler(JobType.WFO)
async def handle_wfo(job: Job, ctx: dict[str, Any]) -> dict[str, Any]:
    """Handle a WFO job.

    Orchestrates walk-forward optimization by generating folds and
    enqueuing child tune jobs.

    Job Payload:
        workspace_id: str - Workspace UUID
        wfo_id: str - Pre-created WFO record UUID
        strategy_entity_id: str - Strategy entity UUID
        data_source: dict - OHLCV data source config
            exchange_id: str
            symbol: str
            timeframe: str
            start_ts: str (optional, uses available if omitted)
            end_ts: str (optional, uses available if omitted)
        param_space: dict - Parameter search space
        wfo_config: dict - WFO configuration
            train_days: int
            test_days: int
            step_days: int
            min_folds: int
            leaderboard_top_k: int (default 10)
            allow_partial: bool (default False)
        search_type: str - "grid" or "random"
        objective_type: str - Objective function type
        objective_metric: str - Metric to optimize
        gates: dict - Gate policy
        seed: int - Random seed
        n_trials: int - Trials per fold

    Context:
        pool: Database connection pool
        events_repo: JobEventsRepository for logging

    Returns:
        dict with:
            status: str - "completed", "partial", or "failed"
            wfo_id: str - WFO UUID
            n_folds: int - Total folds generated
            folds_completed: int - Folds that succeeded
            folds_failed: int - Folds that failed
            child_tune_ids: list[str] - Child tune job IDs
            artifacts: list[str] - Artifact paths (if any)
            warnings: list[str] - Any warnings

    Raises:
        ValueError: If required payload fields are missing
        InsufficientDataError: If cannot generate minimum folds
    """
    pool = ctx["pool"]
    events_repo: JobEventsRepository = ctx["events_repo"]

    # Parse payload
    payload = job.payload
    workspace_id_str = payload.get("workspace_id")
    wfo_id_str = payload.get("wfo_id")
    strategy_entity_id_str = payload.get("strategy_entity_id")
    data_source = payload.get("data_source")
    wfo_config_dict = payload.get("wfo_config")

    # Validate required fields
    if not workspace_id_str:
        raise ValueError("Missing required payload field: workspace_id")
    if not wfo_id_str:
        raise ValueError("Missing required payload field: wfo_id")
    if not strategy_entity_id_str:
        raise ValueError("Missing required payload field: strategy_entity_id")
    if not data_source:
        raise ValueError("Missing required payload field: data_source")
    if not wfo_config_dict:
        raise ValueError("Missing required payload field: wfo_config")

    workspace_id = UUID(workspace_id_str)
    wfo_id = UUID(wfo_id_str)

    # Parse WFO config
    wfo_config = WFOConfig(
        train_days=wfo_config_dict["train_days"],
        test_days=wfo_config_dict["test_days"],
        step_days=wfo_config_dict["step_days"],
        min_folds=wfo_config_dict.get("min_folds", 3),
        start_ts=(
            parse_iso_timestamp(wfo_config_dict["start_ts"])
            if wfo_config_dict.get("start_ts")
            else None
        ),
        end_ts=(
            parse_iso_timestamp(wfo_config_dict["end_ts"])
            if wfo_config_dict.get("end_ts")
            else None
        ),
        leaderboard_top_k=wfo_config_dict.get("leaderboard_top_k", 10),
        allow_partial=wfo_config_dict.get("allow_partial", False),
    )

    log = logger.bind(
        job_id=str(job.id),
        wfo_id=str(wfo_id),
        strategy_entity_id=strategy_entity_id_str,
    )
    log.info("wfo_job_started")

    await events_repo.info(
        job.id,
        f"Starting WFO job: {wfo_config.train_days}d train, "
        f"{wfo_config.test_days}d test, {wfo_config.step_days}d step",
    )

    # Initialize repositories
    job_repo = JobRepository(pool)
    ohlcv_repo = OHLCVRepository(pool)
    tune_repo = TuneRepository(pool)

    # Get data source info
    exchange_id = data_source["exchange_id"]
    symbol = data_source["symbol"]
    timeframe = data_source["timeframe"]

    # Query available data range
    data_range = await ohlcv_repo.get_available_range(
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=timeframe,
    )

    if not data_range or not data_range["min_ts"] or not data_range["max_ts"]:
        raise ValueError(
            f"No OHLCV data available for {symbol} {timeframe} on {exchange_id}"
        )

    available_start = data_range["min_ts"]
    available_end = data_range["max_ts"]

    log.info(
        "wfo_data_range",
        available_start=available_start.isoformat(),
        available_end=available_end.isoformat(),
        row_count=data_range.get("row_count"),
    )

    await events_repo.info(
        job.id,
        f"Data available: {available_start.date()} to {available_end.date()} "
        f"({data_range.get('row_count', 'unknown')} candles)",
    )

    # Generate folds
    try:
        folds = generate_folds(wfo_config, (available_start, available_end))
    except InsufficientDataError as e:
        log.error("wfo_insufficient_data", error=str(e))
        await events_repo.error(job.id, f"Cannot generate folds: {e}")
        raise

    log.info("wfo_folds_generated", n_folds=len(folds))
    await events_repo.info(
        job.id,
        f"Generated {len(folds)} folds from {folds[0].train_start.date()} "
        f"to {folds[-1].test_end.date()}",
    )

    # Build base payload for child tunes (shared config)
    base_child_payload = {
        "workspace_id": workspace_id_str,
        "strategy_entity_id": strategy_entity_id_str,
        "data_source": data_source,
        "param_space": payload.get("param_space", {}),
        "search_type": payload.get("search_type", "grid"),
        "objective_type": payload.get("objective_type", "sharpe"),
        "objective_metric": payload.get("objective_metric", "sharpe"),
        "gates": payload.get("gates", {}),
        "seed": payload.get("seed"),
        "n_trials": payload.get("n_trials"),
        "min_trades": payload.get("min_trades"),
        "initial_cash": payload.get("initial_cash"),
        "commission_bps": payload.get("commission_bps"),
        "slippage_bps": payload.get("slippage_bps"),
    }

    # Enqueue child tune jobs
    child_job_ids: list[UUID] = []
    for fold in folds:
        # Create tune record for this fold
        tune_record = await tune_repo.create_tune(
            workspace_id=workspace_id,
            strategy_entity_id=UUID(strategy_entity_id_str),
            param_space=payload.get("param_space", {}),
            search_type=payload.get("search_type", "grid"),
            n_trials=payload.get("n_trials", 100),
            seed=payload.get("seed"),
            objective_metric=payload.get("objective_metric", "sharpe"),
            objective_type=payload.get("objective_type", "sharpe"),
            objective_params=payload.get("objective_params"),
            oos_ratio=fold.test_days / (fold.train_days + fold.test_days),
            gates=payload.get("gates"),
        )

        # Build child payload
        child_payload = _build_child_tune_payload(
            fold=fold,
            base_payload=base_child_payload,
            wfo_id=wfo_id,
            fold_index=fold.index,
        )
        child_payload["tune_id"] = str(tune_record["id"])

        # Enqueue child job
        child_job = await job_repo.enqueue(
            job_type=JobType.TUNE,
            payload=child_payload,
            workspace_id=workspace_id,
            parent_job_id=job.id,
            priority=100 + fold.index,  # Lower folds run first
        )
        child_job_ids.append(child_job.id)

        log.info(
            "wfo_child_enqueued",
            fold_index=fold.index,
            child_job_id=str(child_job.id),
            tune_id=str(tune_record["id"]),
        )

    await events_repo.info(
        job.id,
        f"Enqueued {len(child_job_ids)} child tune jobs",
    )

    # Wait for all children to complete
    succeeded, failed, canceled = await _wait_for_children(
        job_repo=job_repo,
        parent_job_id=job.id,
        events_repo=events_repo,
        log=log,
    )

    folds_completed = len(succeeded)
    folds_failed = len(failed) + len(canceled)

    # Determine status
    warnings: list[str] = []
    if folds_failed > 0:
        if wfo_config.allow_partial and folds_completed > 0:
            status = "partial"
            warnings.append(f"{folds_failed} folds failed but allow_partial=True")
        elif folds_completed == 0:
            status = "failed"
            warnings.append("All folds failed")
        else:
            status = "failed"
            warnings.append(
                f"{folds_failed} folds failed and allow_partial=False"
            )
    else:
        status = "completed"

    log.info(
        "wfo_job_finished",
        status=status,
        folds_completed=folds_completed,
        folds_failed=folds_failed,
    )

    await events_repo.info(
        job.id,
        f"WFO {status}: {folds_completed}/{len(folds)} folds completed",
    )

    # Return result (aggregation happens in separate step for now)
    return {
        "status": status,
        "wfo_id": str(wfo_id),
        "n_folds": len(folds),
        "folds_completed": folds_completed,
        "folds_failed": folds_failed,
        "child_tune_ids": [str(jid) for jid in child_job_ids],
        "artifacts": [],  # Artifacts generated in aggregation step
        "warnings": warnings,
    }
