"""TuneJob handler - runs parameter tuning via job queue.

This handler:
1. Parses job payload (tune_id, strategy, data source, param space, etc.)
2. If data_source provided: calls ensure_ohlcv_range and loads from DB
3. Gets strategy config from KB repo
4. Runs ParamTuner.run() (reuses existing tuning logic)
5. Generates artifacts (tune.json, trials.csv, equity_best.csv)
6. Records artifacts in artifact_index
7. Returns result with status, counts, and artifact paths
"""

import base64
import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

import structlog

from app.config import get_settings
from app.jobs.models import Job
from app.jobs.registry import default_registry
from app.jobs.types import JobType
from app.repositories.artifacts import ArtifactRepository
from app.repositories.backtests import BacktestRepository, TuneRepository
from app.repositories.kb import KnowledgeBaseRepository
from app.repositories.ohlcv import OHLCVRepository
from app.repositories.job_events import JobEventsRepository
from app.services.backtest.tuner import ParamTuner, TuneResult
from app.services.market_data.ensure_range import ensure_ohlcv_range

logger = structlog.get_logger(__name__)


def parse_iso_timestamp(value: str) -> datetime:
    """Parse ISO format timestamp string to datetime.

    Args:
        value: ISO format string (e.g., '2024-01-01T00:00:00Z')

    Returns:
        datetime with UTC timezone
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def candles_to_csv_bytes(candles: list, symbol: str) -> bytes:
    """Convert list of Candle objects to CSV bytes.

    Args:
        candles: List of Candle dataclass instances
        symbol: Symbol for filename context

    Returns:
        UTF-8 encoded CSV bytes
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(["date", "open", "high", "low", "close", "volume"])

    # Write data rows
    for candle in candles:
        writer.writerow(
            [
                candle.ts.isoformat(),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ]
        )

    return output.getvalue().encode("utf-8")


def _write_file_atomic(path: Path, content: bytes) -> int:
    """Write file atomically and return file size.

    Creates parent directories if needed.
    Writes to temp file then renames for atomic write.

    Args:
        path: Target file path
        content: File content as bytes

    Returns:
        File size in bytes
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(content)
    temp_path.rename(path)
    return len(content)


def _generate_tune_json(
    tune: dict[str, Any],
    result: TuneResult,
    data_revision: Optional[dict],
) -> bytes:
    """Generate tune.json content.

    Args:
        tune: Tune record from database
        result: TuneResult from ParamTuner
        data_revision: Data source metadata

    Returns:
        JSON bytes
    """
    tune_json = {
        "identifiers": {
            "tune_id": str(tune["id"]),
            "workspace_id": str(tune["workspace_id"]),
            "strategy_entity_id": str(tune["strategy_entity_id"]),
            "strategy_name": tune.get("strategy_name"),
        },
        "data_revision": data_revision,
        "param_space": tune.get("param_space", {}),
        "search": {
            "type": tune.get("search_type"),
            "n_trials": tune.get("n_trials"),
            "seed": tune.get("seed"),
        },
        "objective": {
            "metric": tune.get("objective_metric"),
            "type": tune.get("objective_type"),
            "params": tune.get("objective_params"),
        },
        "split": {
            "oos_ratio": tune.get("oos_ratio"),
        },
        "gates": tune.get("gates"),
        "results": {
            "status": result.status,
            "n_trials": result.n_trials,
            "trials_completed": result.trials_completed,
            "best_run_id": str(result.best_run_id) if result.best_run_id else None,
            "best_params": result.best_params,
            "best_score": result.best_score,
        },
        "created_at": (
            tune["created_at"].isoformat() if tune.get("created_at") else None
        ),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    return json.dumps(tune_json, indent=2, default=str).encode("utf-8")


def _generate_trials_csv(runs: list[dict[str, Any]]) -> bytes:
    """Generate trials.csv content.

    Args:
        runs: List of tune runs from database

    Returns:
        CSV bytes with trial results
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow([
        "trial_index",
        "run_id",
        "params_json",
        "score",
        "score_is",
        "score_oos",
        "objective_score",
        "status",
        "skip_reason",
        "failed_reason",
        "return_pct_is",
        "sharpe_is",
        "max_drawdown_pct_is",
        "num_trades_is",
        "return_pct_oos",
        "sharpe_oos",
        "max_drawdown_pct_oos",
        "num_trades_oos",
    ])

    # Write data rows
    for run in runs:
        metrics_is = run.get("metrics_is") or {}
        metrics_oos = run.get("metrics_oos") or {}

        writer.writerow([
            run.get("trial_index"),
            str(run.get("run_id")) if run.get("run_id") else "",
            json.dumps(run.get("params", {})),
            run.get("score"),
            run.get("score_is"),
            run.get("score_oos"),
            run.get("objective_score"),
            run.get("status"),
            run.get("skip_reason", ""),
            run.get("failed_reason", ""),
            metrics_is.get("return_pct"),
            metrics_is.get("sharpe"),
            metrics_is.get("max_drawdown_pct"),
            metrics_is.get("num_trades"),
            metrics_oos.get("return_pct"),
            metrics_oos.get("sharpe"),
            metrics_oos.get("max_drawdown_pct"),
            metrics_oos.get("num_trades"),
        ])

    return output.getvalue().encode("utf-8")


def _generate_equity_csv(equity_curve: list[dict[str, Any]]) -> bytes:
    """Generate equity_best.csv content.

    Args:
        equity_curve: Equity curve from backtest run

    Returns:
        CSV bytes with equity curve
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(["ts", "equity", "drawdown_pct"])

    # Track peak for drawdown calculation
    peak = 0.0
    for point in equity_curve:
        ts = point.get("t", "")
        equity = float(point.get("equity", 0))

        # Update peak and calculate drawdown
        if equity > peak:
            peak = equity
        drawdown_pct = ((peak - equity) / peak * 100) if peak > 0 else 0

        writer.writerow([ts, equity, round(drawdown_pct, 4)])

    return output.getvalue().encode("utf-8")


async def generate_tune_artifacts(
    tune_id: UUID,
    result: TuneResult,
    workspace_id: UUID,
    data_revision: Optional[dict],
    artifact_repo: ArtifactRepository,
    tune_repo: TuneRepository,
    backtest_repo: BacktestRepository,
    job_id: Optional[UUID] = None,
) -> list[str]:
    """
    Generate and record tune artifacts.

    Creates files on filesystem and records in artifact_index.

    Artifacts generated:
    - tune.json: Tune configuration and results summary
    - trials.csv: All trial results with metrics
    - equity_best.csv: Equity curve for best trial (if available)

    Args:
        tune_id: The tune ID
        result: TuneResult from ParamTuner
        workspace_id: Workspace UUID
        data_revision: Optional data revision metadata
        artifact_repo: ArtifactRepository instance
        tune_repo: TuneRepository for fetching tune data
        backtest_repo: BacktestRepository for fetching equity curves
        job_id: Optional job ID for artifact linking

    Returns:
        List of artifact paths that were created
    """
    settings = get_settings()
    artifacts_dir = Path(settings.artifacts_dir)
    tune_dir = artifacts_dir / "tunes" / str(tune_id)
    artifacts_recorded = []

    log = logger.bind(tune_id=str(tune_id))

    # Fetch tune record for metadata
    tune = await tune_repo.get_tune(tune_id)
    if not tune:
        log.warning("tune_not_found_for_artifacts")
        return []

    # 1. Generate tune.json
    tune_json_path = tune_dir / "tune.json"
    tune_json_content = _generate_tune_json(tune, result, data_revision)
    tune_json_size = _write_file_atomic(tune_json_path, tune_json_content)

    await artifact_repo.create(
        workspace_id=workspace_id,
        run_id=tune_id,
        job_type="tune",
        artifact_kind="tune_json",
        artifact_path=f"tunes/{tune_id}/tune.json",
        file_size_bytes=tune_json_size,
        data_revision=data_revision,
        job_id=job_id,
    )
    artifacts_recorded.append(f"tunes/{tune_id}/tune.json")
    log.debug("tune_json_written", path=str(tune_json_path), size=tune_json_size)

    # 2. Generate trials.csv
    runs, _ = await tune_repo.list_tune_runs(tune_id, limit=10000)

    trials_csv_path = tune_dir / "trials.csv"
    trials_csv_content = _generate_trials_csv(runs)
    trials_csv_size = _write_file_atomic(trials_csv_path, trials_csv_content)

    await artifact_repo.create(
        workspace_id=workspace_id,
        run_id=tune_id,
        job_type="tune",
        artifact_kind="trials_csv",
        artifact_path=f"tunes/{tune_id}/trials.csv",
        file_size_bytes=trials_csv_size,
        data_revision=data_revision,
        job_id=job_id,
    )
    artifacts_recorded.append(f"tunes/{tune_id}/trials.csv")
    log.debug("trials_csv_written", path=str(trials_csv_path), size=trials_csv_size)

    # 3. Generate equity_best.csv (if best run exists)
    if result.best_run_id:
        best_run = await backtest_repo.get_run(result.best_run_id)
        if best_run and best_run.get("equity_curve"):
            equity_csv_path = tune_dir / "equity_best.csv"
            equity_csv_content = _generate_equity_csv(best_run["equity_curve"])
            equity_csv_size = _write_file_atomic(equity_csv_path, equity_csv_content)

            await artifact_repo.create(
                workspace_id=workspace_id,
                run_id=tune_id,
                job_type="tune",
                artifact_kind="equity_csv",
                artifact_path=f"tunes/{tune_id}/equity_best.csv",
                file_size_bytes=equity_csv_size,
                data_revision=data_revision,
                job_id=job_id,
            )
            artifacts_recorded.append(f"tunes/{tune_id}/equity_best.csv")
            log.debug(
                "equity_csv_written", path=str(equity_csv_path), size=equity_csv_size
            )
        else:
            log.debug("equity_curve_not_available", best_run_id=str(result.best_run_id))

    log.info(
        "tune_artifacts_generated",
        artifacts_count=len(artifacts_recorded),
        total_size=sum(
            os.path.getsize(tune_dir / Path(p).name)
            for p in artifacts_recorded
            if (tune_dir / Path(p).name).exists()
        ),
    )

    return artifacts_recorded


@default_registry.handler(JobType.TUNE)
async def handle_tune(job: Job, ctx: dict[str, Any]) -> dict[str, Any]:
    """Handle a TUNE job.

    Runs parameter tuning via ParamTuner and records artifacts.

    Job Payload:
        workspace_id: str - Workspace UUID
        tune_id: str - Pre-created tune record UUID
        strategy_entity_id: str - Strategy entity UUID
        ohlcv_file_content: str - Base64 encoded CSV (OR data_source)
        data_source: dict - Alternative: use stored OHLCV
            exchange_id: str
            symbol: str
            timeframe: str
            start_ts: str
            end_ts: str
        filename: str - Optional filename for the OHLCV data
        param_space: dict - Parameter search space
        search_type: str - "grid" or "random"
        objective_type: str - Objective function type
        gates: dict - Gate policy
        oos_ratio: float - Out-of-sample ratio
        seed: int - Random seed
        n_trials: int - Number of trials
        objective_metric: str - Metric to optimize
        min_trades: int - Minimum trades for valid trial
        initial_cash: float - Starting capital
        commission_bps: float - Commission in basis points
        slippage_bps: float - Slippage in basis points
        date_from: str - Optional start date filter
        date_to: str - Optional end date filter

    Context:
        pool: Database connection pool
        events_repo: JobEventsRepository for logging

    Returns:
        dict with:
            status: str - "completed" or "failed"
            tune_id: str - Tune UUID
            n_trials: int - Total trials planned
            trials_completed: int - Trials that completed
            best_run_id: str | None - Best trial run ID
            best_score: float | None - Best trial score
            artifacts: list[str] - Artifact paths recorded

    Raises:
        ValueError: If required payload fields are missing
    """
    pool = ctx["pool"]
    events_repo: JobEventsRepository = ctx["events_repo"]

    # Parse payload
    payload = job.payload
    workspace_id_str = payload.get("workspace_id")
    tune_id_str = payload.get("tune_id")
    strategy_entity_id_str = payload.get("strategy_entity_id")

    # Validate required fields
    if not workspace_id_str:
        raise ValueError("Missing required payload field: workspace_id")
    if not tune_id_str:
        raise ValueError("Missing required payload field: tune_id")
    if not strategy_entity_id_str:
        raise ValueError("Missing required payload field: strategy_entity_id")

    workspace_id = UUID(workspace_id_str)
    tune_id = UUID(tune_id_str)
    strategy_entity_id = UUID(strategy_entity_id_str)

    # Get OHLCV data (either inline or from data_source)
    ohlcv_file_content_b64 = payload.get("ohlcv_file_content")
    data_source = payload.get("data_source")

    if not ohlcv_file_content_b64 and not data_source:
        raise ValueError(
            "Missing OHLCV data: provide either ohlcv_file_content or data_source"
        )

    log = logger.bind(
        job_id=str(job.id),
        tune_id=str(tune_id),
        strategy_entity_id=str(strategy_entity_id),
    )
    log.info("tune_job_started")

    await events_repo.info(
        job.id,
        f"Starting tune job for tune_id={tune_id}",
    )

    # Initialize repositories
    tune_repo = TuneRepository(pool)
    kb_repo = KnowledgeBaseRepository(pool)
    backtest_repo = BacktestRepository(pool)
    artifact_repo = ArtifactRepository(pool)

    # Track data revision for artifacts
    data_revision: Optional[dict] = None
    file_content: bytes
    filename: str

    if ohlcv_file_content_b64:
        # Decode inline base64 content
        file_content = base64.b64decode(ohlcv_file_content_b64)
        filename = payload.get("filename", "data.csv")
        log.info(
            "tune_using_inline_data", filename=filename, size_bytes=len(file_content)
        )
        await events_repo.info(job.id, f"Using inline OHLCV data: {filename}")

    else:
        # Load from data_source via ensure_ohlcv_range
        # data_source is guaranteed to exist here due to validation above
        assert data_source is not None  # for mypy
        exchange_id = data_source["exchange_id"]
        symbol = data_source["symbol"]
        timeframe = data_source["timeframe"]
        start_ts = parse_iso_timestamp(data_source["start_ts"])
        end_ts = parse_iso_timestamp(data_source["end_ts"])

        log.info(
            "tune_ensuring_ohlcv_range",
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts.isoformat(),
            end_ts=end_ts.isoformat(),
        )
        await events_repo.info(
            job.id,
            f"Ensuring OHLCV data: {symbol} {timeframe} from {exchange_id}",
        )

        # Ensure data is available
        ensure_result = await ensure_ohlcv_range(
            pool=pool,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        log.info(
            "tune_ohlcv_range_ensured",
            total_candles=ensure_result.total_candles,
            fetched_candles=ensure_result.fetched_candles,
            was_cached=ensure_result.was_cached,
        )

        # Load candles from database as CSV bytes
        ohlcv_repo = OHLCVRepository(pool)
        candles = await ohlcv_repo.get_range(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        file_content = candles_to_csv_bytes(candles, symbol)
        filename = f"{symbol.replace('-', '')}_{timeframe}.csv"

        # Build data revision for artifact metadata
        data_revision = {
            "exchange_id": exchange_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "start_ts": start_ts.isoformat(),
            "end_ts": end_ts.isoformat(),
            "row_count": len(candles),
        }

        await events_repo.info(
            job.id,
            f"Loaded {len(candles)} candles from database",
        )

    # Extract tuning parameters from payload
    param_space = payload.get("param_space", {})
    search_type = payload.get("search_type", "grid")
    n_trials = payload.get("n_trials", 50)
    seed = payload.get("seed")
    objective_metric = payload.get("objective_metric", "sharpe")
    min_trades = payload.get("min_trades", 5)
    oos_ratio = payload.get("oos_ratio")
    objective_type = payload.get("objective_type", "sharpe")
    objective_params = payload.get("objective_params")
    initial_cash = payload.get("initial_cash", 10000.0)
    commission_bps = payload.get("commission_bps", 10)
    slippage_bps = payload.get("slippage_bps", 5)

    # Optional date filters
    date_from = None
    date_to = None
    if payload.get("date_from"):
        date_from = parse_iso_timestamp(payload["date_from"])
    if payload.get("date_to"):
        date_to = parse_iso_timestamp(payload["date_to"])

    # Create ParamTuner and run
    tuner = ParamTuner(kb_repo, backtest_repo, tune_repo)

    log.info(
        "tune_running_param_tuner",
        search_type=search_type,
        n_trials=n_trials,
        oos_ratio=oos_ratio,
    )
    await events_repo.info(
        job.id,
        f"Running ParamTuner: {search_type} search, {n_trials} trials",
    )

    result: TuneResult = await tuner.run(
        tune_id=tune_id,
        strategy_entity_id=strategy_entity_id,
        workspace_id=workspace_id,
        file_content=file_content,
        filename=filename,
        param_space=param_space,
        search_type=search_type,
        n_trials=n_trials,
        seed=seed,
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        objective_metric=objective_metric,
        min_trades=min_trades,
        oos_ratio=oos_ratio,
        objective_type=objective_type,
        objective_params=objective_params,
        date_from=date_from,
        date_to=date_to,
    )

    log.info(
        "tune_completed",
        status=result.status,
        trials_completed=result.trials_completed,
        best_score=result.best_score,
    )
    await events_repo.info(
        job.id,
        f"Tune completed: {result.trials_completed}/{result.n_trials} trials, "
        f"best_score={result.best_score}",
    )

    # Generate and record artifacts
    artifacts = await generate_tune_artifacts(
        tune_id=tune_id,
        result=result,
        workspace_id=workspace_id,
        data_revision=data_revision,
        artifact_repo=artifact_repo,
        tune_repo=tune_repo,
        backtest_repo=backtest_repo,
        job_id=job.id,
    )

    await events_repo.info(
        job.id,
        f"Recorded {len(artifacts)} artifacts",
    )

    return {
        "status": result.status,
        "tune_id": str(tune_id),
        "n_trials": result.n_trials,
        "trials_completed": result.trials_completed,
        "best_run_id": str(result.best_run_id) if result.best_run_id else None,
        "best_score": result.best_score,
        "artifacts": artifacts,
    }
