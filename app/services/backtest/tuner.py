"""Parameter tuning orchestration for backtests.

Coordinates grid/random search over strategy parameters,
running multiple backtests and tracking results.

Supports IS/OOS (in-sample / out-of-sample) splits for anti-overfit validation.
"""

import asyncio
import csv
import io
import itertools
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.backtest.runner import BacktestRunner, BacktestRunError
from app.services.backtest.scoring import compute_score, rank_trials
from app.services.backtest.regime_integration import (
    add_regime_to_metrics,
    detect_timeframe_from_ohlcv,
    extract_instrument_from_filename,
)

logger = structlog.get_logger(__name__)

# Configuration from environment
TUNER_MAX_CONCURRENCY = int(os.environ.get("TUNER_MAX_CONCURRENCY", "4"))
TUNER_TRIAL_TIMEOUT = int(os.environ.get("TUNER_TRIAL_TIMEOUT", "120"))  # seconds

# Minimum bars required for IS/OOS windows
MIN_BARS_IS = int(os.environ.get("TUNER_MIN_BARS_IS", "200"))
MIN_BARS_OOS = int(os.environ.get("TUNER_MIN_BARS_OOS", "100"))

# Gate thresholds (configurable via environment)
# Pass if max_drawdown_pct >= -GATE_MAX_DD_PCT (e.g., -20 means DD must be >= -20%)
GATE_MAX_DD_PCT = float(os.environ.get("TUNER_GATE_MAX_DD_PCT", "20"))
# Pass if trades >= GATE_MIN_TRADES
GATE_MIN_TRADES = int(os.environ.get("TUNER_GATE_MIN_TRADES", "10"))

# Canonical metrics keys to persist
METRICS_KEYS = [
    "return_pct",
    "sharpe",
    "max_drawdown_pct",
    "win_rate",
    "trades",
    "profit_factor",
]


def serialize_metrics(summary: dict) -> dict:
    """
    Serialize backtest summary to canonical metrics format.

    - Filters to allowed keys
    - Converts Decimal → float
    - Converts NaN/inf → None
    - Rounds appropriately
    - Rejects non-numeric values
    """
    import math
    from decimal import Decimal

    metrics = {}
    for key in METRICS_KEYS:
        val = summary.get(key)

        if val is None:
            metrics[key] = None
            continue

        # Reject non-numeric values (strings, etc.)
        if not isinstance(val, (int, float, Decimal)):
            metrics[key] = None
            continue

        # Convert Decimal to float
        if isinstance(val, Decimal):
            val = float(val)

        # Handle NaN/inf
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            metrics[key] = None
            continue

        # Round appropriately
        if key == "trades":
            metrics[key] = int(val)
        elif key in ("return_pct", "max_drawdown_pct"):
            metrics[key] = round(float(val), 2)
        else:
            metrics[key] = round(float(val), 4)

    return metrics


def evaluate_gates(metrics: dict) -> tuple[bool, list[str]]:
    """
    Evaluate gate constraints against metrics.

    Returns:
        (passed, failures) where:
        - passed: True if all gates pass
        - failures: List of failure strings like "gate:max_drawdown_pct (-27.4 < -20.0)"
    """
    if not metrics:
        return False, ["gate:missing_metrics"]

    failures = []

    # Gate: max_drawdown_pct >= -GATE_MAX_DD_PCT
    dd = metrics.get("max_drawdown_pct")
    if dd is not None:
        threshold = -GATE_MAX_DD_PCT
        if dd < threshold:
            failures.append(f"gate:max_drawdown_pct ({dd:.1f} < {threshold:.1f})")
    else:
        failures.append("gate:max_drawdown_pct (missing)")

    # Gate: trades >= GATE_MIN_TRADES
    trades = metrics.get("trades")
    if trades is not None:
        if trades < GATE_MIN_TRADES:
            failures.append(f"gate:trades ({trades} < {GATE_MIN_TRADES})")
    else:
        failures.append("gate:trades (missing)")

    passed = len(failures) == 0
    return passed, failures


# Default lambda for DD penalty objectives
DEFAULT_DD_LAMBDA = 0.02  # 10% DD → 0.2 Sharpe penalty


def compute_objective_score(
    metrics: dict,
    objective_type: str = "sharpe",
    objective_params: Optional[dict] = None,
) -> Optional[float]:
    """
    Compute composite objective score from metrics.

    Objective types:
    - "sharpe": Raw Sharpe ratio (default)
    - "sharpe_dd_penalty": sharpe - λ * abs(max_drawdown_pct)
    - "return": Raw return percentage
    - "return_dd_penalty": return_pct - λ * abs(max_drawdown_pct)
    - "calmar": Calmar ratio (return / max_drawdown)

    Args:
        metrics: Serialized metrics dict
        objective_type: Type of objective function
        objective_params: Parameters like {"dd_lambda": 0.02}

    Returns:
        Objective score or None if required metrics missing
    """
    if not metrics:
        return None

    params = objective_params or {}
    dd_lambda = params.get("dd_lambda", DEFAULT_DD_LAMBDA)

    if objective_type == "sharpe":
        return metrics.get("sharpe")

    elif objective_type == "sharpe_dd_penalty":
        sharpe = metrics.get("sharpe")
        dd = metrics.get("max_drawdown_pct")
        if sharpe is None:
            return None
        if dd is None:
            return sharpe  # No penalty if DD unknown
        # DD is negative (e.g., -15), abs gives 15, penalty = 15 * 0.02 = 0.3
        return round(sharpe - dd_lambda * abs(dd), 4)

    elif objective_type == "return":
        return metrics.get("return_pct")

    elif objective_type == "return_dd_penalty":
        ret = metrics.get("return_pct")
        dd = metrics.get("max_drawdown_pct")
        if ret is None:
            return None
        if dd is None:
            return ret
        return round(ret - dd_lambda * abs(dd), 4)

    elif objective_type == "calmar":
        ret = metrics.get("return_pct")
        dd = metrics.get("max_drawdown_pct")
        if ret is None or dd is None or dd == 0:
            return None
        # Calmar = return / abs(max_dd)
        return round(ret / abs(dd), 4)

    else:
        # Unknown type, fallback to sharpe
        logger.warning(
            "Unknown objective_type, using sharpe", objective_type=objective_type
        )
        return metrics.get("sharpe")


@dataclass
class TuneResult:
    """Result of a parameter tuning session."""

    tune_id: UUID
    status: str
    n_trials: int
    trials_completed: int
    best_run_id: Optional[UUID]
    best_params: Optional[dict[str, Any]]
    best_score: Optional[float]
    leaderboard: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


class ParamTuner:
    """
    Orchestrates parameter tuning over strategy backtests.

    Supports grid and random search with bounded concurrency.
    """

    MAX_GRID_SIZE = 200

    def __init__(self, kb_repo, backtest_repo, tune_repo):
        self.kb_repo = kb_repo
        self.backtest_repo = backtest_repo
        self.tune_repo = tune_repo
        self.runner = BacktestRunner(kb_repo, backtest_repo)
        self.max_concurrency = TUNER_MAX_CONCURRENCY
        self.trial_timeout = TUNER_TRIAL_TIMEOUT
        self._canceled = False

    def _compute_oos_split(
        self,
        file_content: bytes,
        oos_ratio: float,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> tuple[Optional[datetime], int, int]:
        """
        Compute the IS/OOS split timestamp from CSV data.

        Returns:
            (split_timestamp, n_bars_is, n_bars_oos) or (None, 0, 0) if insufficient bars
        """
        # Parse CSV to get date column
        try:
            text = file_content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))

            # Find date column (common aliases)
            dates = []
            for row in reader:
                date_val = (
                    row.get("date") or row.get("timestamp") or row.get("datetime")
                )
                if date_val:
                    # Parse date string
                    try:
                        dt = datetime.fromisoformat(date_val.replace("Z", "+00:00"))
                    except ValueError:
                        # Try other formats
                        for fmt in [
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%d",
                            "%Y/%m/%d %H:%M:%S",
                        ]:
                            try:
                                dt = datetime.strptime(date_val, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            continue  # Skip unparseable dates
                    dates.append(dt)

            if not dates:
                logger.warning("No dates found in CSV for OOS split")
                return None, 0, 0

            # Apply date filters
            if date_from:
                dates = [d for d in dates if d >= date_from]
            if date_to:
                dates = [d for d in dates if d <= date_to]

            # Sort by time
            dates.sort()
            n_total = len(dates)

            if n_total < MIN_BARS_IS + MIN_BARS_OOS:
                logger.warning(
                    "Insufficient bars for OOS split",
                    n_total=n_total,
                    min_required=MIN_BARS_IS + MIN_BARS_OOS,
                )
                return None, 0, 0

            # Compute split index
            n_oos = int(n_total * oos_ratio)
            n_is = n_total - n_oos

            # Guard: ensure minimum bars in each window
            if n_is < MIN_BARS_IS or n_oos < MIN_BARS_OOS:
                logger.warning(
                    "OOS split would create windows below minimum",
                    n_is=n_is,
                    n_oos=n_oos,
                    min_is=MIN_BARS_IS,
                    min_oos=MIN_BARS_OOS,
                )
                return None, 0, 0

            # Split timestamp is the date of the first OOS bar
            split_timestamp = dates[n_is]

            logger.debug(
                "Computed OOS split",
                n_total=n_total,
                n_is=n_is,
                n_oos=n_oos,
                split_timestamp=split_timestamp.isoformat(),
            )

            return split_timestamp, n_is, n_oos

        except Exception as e:
            logger.error("Failed to compute OOS split", error=str(e))
            return None, 0, 0

    async def run(
        self,
        tune_id: UUID,
        strategy_entity_id: UUID,
        workspace_id: UUID,
        file_content: bytes,
        filename: str,
        param_space: dict[str, Any],
        search_type: str,
        n_trials: int,
        seed: Optional[int],
        initial_cash: float,
        commission_bps: float,
        slippage_bps: float,
        objective_metric: str,
        min_trades: int,
        oos_ratio: Optional[float] = None,
        objective_type: str = "sharpe",
        objective_params: Optional[dict[str, Any]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> TuneResult:
        """
        Run parameter tuning session.

        Args:
            tune_id: ID of pre-created tune record
            strategy_entity_id: Strategy to tune
            workspace_id: Workspace scope
            file_content: OHLCV CSV bytes
            filename: Original filename
            param_space: Parameter search space
            search_type: "grid" or "random"
            n_trials: Number of trials to run
            seed: Random seed for reproducibility
            initial_cash: Starting capital
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
            objective_metric: Optimization objective
            min_trades: Minimum trades for valid trial
            oos_ratio: Out-of-sample split ratio (0-0.5). When set, score=score_oos.
            date_from: Optional date filter
            date_to: Optional date filter

        Returns:
            TuneResult with best params and leaderboard
        """
        warnings = []

        logger.info(
            "Starting parameter tuning",
            tune_id=str(tune_id),
            strategy_entity_id=str(strategy_entity_id),
            search_type=search_type,
            n_trials=n_trials,
            seed=seed,
            oos_ratio=oos_ratio,
        )

        # Compute IS/OOS split if requested
        oos_split_timestamp = None
        if oos_ratio:
            oos_split_timestamp, n_is, n_oos = self._compute_oos_split(
                file_content, oos_ratio, date_from, date_to
            )
            if oos_split_timestamp:
                logger.info(
                    "IS/OOS split enabled",
                    tune_id=str(tune_id),
                    oos_ratio=oos_ratio,
                    n_is=n_is,
                    n_oos=n_oos,
                    split_timestamp=oos_split_timestamp.isoformat(),
                )
            else:
                warnings.append(
                    f"OOS split requested (ratio={oos_ratio}) but insufficient bars. "
                    f"Running without split."
                )
                logger.warning(
                    "OOS split requested but could not be computed",
                    tune_id=str(tune_id),
                    oos_ratio=oos_ratio,
                )

        # Update tune status to running
        await self.tune_repo.update_tune_status(
            tune_id, "running", started_at=datetime.utcnow()
        )

        # Detect timeframe and instrument for regime computation
        detected_timeframe = detect_timeframe_from_ohlcv(file_content, filename)
        detected_instrument = extract_instrument_from_filename(filename)

        try:
            # Generate parameter combinations
            param_sets = self._generate_params(param_space, search_type, n_trials, seed)
            actual_trials = len(param_sets)

            if actual_trials < n_trials:
                warnings.append(
                    f"Grid search produced {actual_trials} combinations (requested {n_trials})"
                )

            logger.info(
                "Generated parameter sets",
                tune_id=str(tune_id),
                param_sets=actual_trials,
            )

            # Insert all tune_runs as queued
            for idx, params in enumerate(param_sets):
                await self.tune_repo.create_tune_run(
                    tune_id=tune_id,
                    trial_index=idx,
                    params=params,
                    status="queued",
                )

            # Run trials with bounded concurrency
            semaphore = asyncio.Semaphore(self.max_concurrency)
            trials_completed = 0
            gate_failures_count = 0

            async def run_trial(idx: int, params: dict) -> Optional[dict]:
                nonlocal trials_completed, gate_failures_count

                async with semaphore:
                    # Check for cancellation before starting
                    tune = await self.tune_repo.get_tune(tune_id)
                    if tune and tune.get("status") == "canceled":
                        logger.info(
                            "Tune canceled, skipping trial",
                            tune_id=str(tune_id),
                            trial_index=idx,
                        )
                        return None

                    # Mark as running with started_at
                    await self.tune_repo.start_tune_run(tune_id, idx)

                    try:
                        # IS/OOS split mode: run two backtests
                        if oos_split_timestamp:
                            # Run IS backtest (fitting window)
                            is_result = await asyncio.wait_for(
                                self.runner.run(
                                    strategy_entity_id=strategy_entity_id,
                                    file_content=file_content,
                                    filename=filename,
                                    params=params,
                                    workspace_id=workspace_id,
                                    initial_cash=initial_cash,
                                    commission_bps=commission_bps,
                                    slippage_bps=slippage_bps,
                                    date_from=date_from,
                                    date_to=oos_split_timestamp,
                                    allow_draft=True,
                                ),
                                timeout=self.trial_timeout,
                            )
                            is_summary = is_result["summary"]
                            score_is = compute_score(
                                is_summary, objective_metric, min_trades
                            )

                            # Run OOS backtest (validation window)
                            oos_result = await asyncio.wait_for(
                                self.runner.run(
                                    strategy_entity_id=strategy_entity_id,
                                    file_content=file_content,
                                    filename=filename,
                                    params=params,
                                    workspace_id=workspace_id,
                                    initial_cash=initial_cash,
                                    commission_bps=commission_bps,
                                    slippage_bps=slippage_bps,
                                    date_from=oos_split_timestamp,
                                    date_to=date_to,
                                    allow_draft=True,
                                ),
                                timeout=self.trial_timeout,
                            )
                            oos_summary = oos_result["summary"]
                            score_oos = compute_score(
                                oos_summary, objective_metric, min_trades
                            )

                            # Use OOS run_id as the canonical run (per design decision)
                            run_id = UUID(oos_result["run_id"])
                            summary = oos_summary

                            # Primary score is OOS score
                            score = score_oos

                            # Determine skip reason if OOS score invalid
                            if score_oos is None:
                                oos_trades = oos_summary.get("trades", 0)
                                if oos_trades < min_trades:
                                    skip_reason = f"oos_min_trades_not_met ({oos_trades}<{min_trades})"
                                elif oos_summary.get(objective_metric) is None:
                                    skip_reason = (
                                        f"oos_metric_unavailable ({objective_metric})"
                                    )
                                else:
                                    skip_reason = "oos_score_unavailable"

                                await self.tune_repo.update_tune_run_result(
                                    tune_id=tune_id,
                                    trial_index=idx,
                                    run_id=run_id,
                                    score=None,
                                    status="skipped",
                                    skip_reason=skip_reason,
                                    score_is=score_is,
                                    score_oos=None,
                                )
                                logger.debug(
                                    "Trial skipped (OOS)",
                                    tune_id=str(tune_id),
                                    trial_index=idx,
                                    skip_reason=skip_reason,
                                    score_is=score_is,
                                )
                                return None

                            # Serialize metrics
                            metrics_is_data = serialize_metrics(is_summary)
                            metrics_oos_data = serialize_metrics(oos_summary)

                            # Add regime snapshots to metrics
                            add_regime_to_metrics(
                                metrics_is_data,
                                file_content=file_content,
                                filename=filename,
                                date_from=date_from,
                                date_to=oos_split_timestamp,
                                source="is",
                                timeframe=detected_timeframe,
                                instrument=detected_instrument,
                            )
                            add_regime_to_metrics(
                                metrics_oos_data,
                                file_content=file_content,
                                filename=filename,
                                date_from=oos_split_timestamp,
                                date_to=date_to,
                                source="oos",
                                timeframe=detected_timeframe,
                                instrument=detected_instrument,
                            )

                            # Evaluate gates on OOS metrics
                            gates_passed, gate_failures = evaluate_gates(
                                metrics_oos_data
                            )

                            if not gates_passed:
                                # Gate failure → skipped (not failed)
                                gate_failures_count += 1
                                skip_reason = "; ".join(gate_failures)
                                await self.tune_repo.update_tune_run_result(
                                    tune_id=tune_id,
                                    trial_index=idx,
                                    run_id=run_id,  # Keep run_id for drill-through
                                    score=score_oos,
                                    status="skipped",
                                    skip_reason=skip_reason,
                                    score_is=score_is,
                                    score_oos=score_oos,
                                    metrics_is=metrics_is_data,
                                    metrics_oos=metrics_oos_data,
                                )
                                logger.debug(
                                    "Trial skipped (gate failure)",
                                    tune_id=str(tune_id),
                                    trial_index=idx,
                                    gate_failures=gate_failures,
                                )
                                return None

                            # Compute objective score from OOS metrics
                            obj_score = compute_objective_score(
                                metrics_oos_data, objective_type, objective_params
                            )

                            # Completed with IS/OOS scores (passed gates)
                            await self.tune_repo.update_tune_run_result(
                                tune_id=tune_id,
                                trial_index=idx,
                                run_id=run_id,
                                score=score_oos,
                                status="completed",
                                score_is=score_is,
                                score_oos=score_oos,
                                metrics_is=metrics_is_data,
                                metrics_oos=metrics_oos_data,
                                objective_score=obj_score,
                            )

                            trials_completed += 1

                            if trials_completed % 5 == 0:
                                await self.tune_repo.update_tune_progress(
                                    tune_id, trials_completed
                                )

                            return {
                                "trial_index": idx,
                                "run_id": str(run_id),
                                "params": params,
                                "score": score_oos,
                                "score_is": score_is,
                                "score_oos": score_oos,
                                "objective_score": obj_score,
                                "summary": summary,
                                "metrics_is": metrics_is_data,
                                "metrics_oos": metrics_oos_data,
                            }

                        # Standard mode (no IS/OOS split)
                        else:
                            result = await asyncio.wait_for(
                                self.runner.run(
                                    strategy_entity_id=strategy_entity_id,
                                    file_content=file_content,
                                    filename=filename,
                                    params=params,
                                    workspace_id=workspace_id,
                                    initial_cash=initial_cash,
                                    commission_bps=commission_bps,
                                    slippage_bps=slippage_bps,
                                    date_from=date_from,
                                    date_to=date_to,
                                    allow_draft=True,
                                ),
                                timeout=self.trial_timeout,
                            )

                            summary = result["summary"]
                            run_id = UUID(result["run_id"])
                            score = compute_score(summary, objective_metric, min_trades)

                            if score is None:
                                trades = summary.get("trades", 0)
                                if trades < min_trades:
                                    skip_reason = (
                                        f"min_trades_not_met ({trades}<{min_trades})"
                                    )
                                elif summary.get(objective_metric) is None:
                                    skip_reason = (
                                        f"metric_unavailable ({objective_metric})"
                                    )
                                else:
                                    skip_reason = "unknown"

                                await self.tune_repo.update_tune_run_result(
                                    tune_id=tune_id,
                                    trial_index=idx,
                                    run_id=run_id,
                                    score=None,
                                    status="skipped",
                                    skip_reason=skip_reason,
                                )
                                logger.debug(
                                    "Trial skipped",
                                    tune_id=str(tune_id),
                                    trial_index=idx,
                                    trades=trades,
                                    skip_reason=skip_reason,
                                )
                                return None

                            # Serialize metrics to metrics_oos (Option A: primary window)
                            metrics_data = serialize_metrics(summary)

                            # Add regime snapshot (full window = primary)
                            add_regime_to_metrics(
                                metrics_data,
                                file_content=file_content,
                                filename=filename,
                                date_from=date_from,
                                date_to=date_to,
                                source="live",
                                timeframe=detected_timeframe,
                                instrument=detected_instrument,
                            )

                            # Evaluate gates on primary metrics
                            gates_passed, gate_failures = evaluate_gates(metrics_data)

                            if not gates_passed:
                                # Gate failure → skipped (not failed)
                                gate_failures_count += 1
                                skip_reason = "; ".join(gate_failures)
                                await self.tune_repo.update_tune_run_result(
                                    tune_id=tune_id,
                                    trial_index=idx,
                                    run_id=run_id,  # Keep run_id for drill-through
                                    score=score,
                                    status="skipped",
                                    skip_reason=skip_reason,
                                    metrics_oos=metrics_data,
                                )
                                logger.debug(
                                    "Trial skipped (gate failure)",
                                    tune_id=str(tune_id),
                                    trial_index=idx,
                                    gate_failures=gate_failures,
                                )
                                return None

                            # Compute objective score from primary metrics
                            obj_score = compute_objective_score(
                                metrics_data, objective_type, objective_params
                            )

                            await self.tune_repo.update_tune_run_result(
                                tune_id=tune_id,
                                trial_index=idx,
                                run_id=run_id,
                                score=score,
                                status="completed",
                                metrics_oos=metrics_data,
                                objective_score=obj_score,
                            )

                            trials_completed += 1

                            if trials_completed % 5 == 0:
                                await self.tune_repo.update_tune_progress(
                                    tune_id, trials_completed
                                )

                            return {
                                "trial_index": idx,
                                "run_id": str(run_id),
                                "params": params,
                                "score": score,
                                "objective_score": obj_score,
                                "summary": summary,
                                "metrics_oos": metrics_data,
                            }

                    except asyncio.TimeoutError:
                        logger.warning(
                            "Trial timed out",
                            tune_id=str(tune_id),
                            trial_index=idx,
                            timeout_seconds=self.trial_timeout,
                        )
                        await self.tune_repo.update_tune_run_result(
                            tune_id=tune_id,
                            trial_index=idx,
                            run_id=None,
                            score=None,
                            status="failed",
                            failed_reason=f"timeout ({self.trial_timeout}s)",
                        )
                        return None

                    except BacktestRunError as e:
                        logger.warning(
                            "Trial failed",
                            tune_id=str(tune_id),
                            trial_index=idx,
                            error=e.message,
                        )
                        await self.tune_repo.update_tune_run_result(
                            tune_id=tune_id,
                            trial_index=idx,
                            run_id=None,
                            score=None,
                            status="failed",
                            failed_reason=f"error: {e.message}",
                        )
                        return None

                    except Exception as e:
                        logger.error(
                            "Trial error",
                            tune_id=str(tune_id),
                            trial_index=idx,
                            error=str(e),
                        )
                        await self.tune_repo.update_tune_run_result(
                            tune_id=tune_id,
                            trial_index=idx,
                            run_id=None,
                            score=None,
                            status="failed",
                            failed_reason=f"exception: {str(e)[:100]}",
                        )
                        return None

            # Execute all trials
            tasks = [run_trial(i, p) for i, p in enumerate(param_sets)]
            trial_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful results
            valid_results = [
                r for r in trial_results if isinstance(r, dict) and r is not None
            ]

            # Sort by objective_score (composite) → score_oos (OOS) → score (raw)
            def sort_key(r):
                obj = r.get("objective_score")
                if obj is not None:
                    return (1, obj)  # Has objective score, sort by it
                oos = r.get("score_oos")
                if oos is not None:
                    return (1, oos)  # Has OOS score, sort by it
                score = r.get("score")
                if score is not None:
                    return (1, score)  # Has raw score
                return (0, 0)  # No valid score

            sorted_results = sorted(valid_results, key=sort_key, reverse=True)

            # Build leaderboard from sorted results
            leaderboard = []
            for rank, trial in enumerate(sorted_results[:10], start=1):
                # Use objective_score if available, otherwise fallback
                display_score = (
                    trial.get("objective_score")
                    or trial.get("score_oos")
                    or trial.get("score")
                )
                leaderboard.append(
                    {
                        "rank": rank,
                        "run_id": trial["run_id"],
                        "params": trial["params"],
                        "score": display_score,
                        "objective_score": trial.get("objective_score"),
                        "summary": trial.get("summary"),
                    }
                )

            # Determine best
            best_run_id = None
            best_params = None
            best_score = None

            if leaderboard:
                best = leaderboard[0]
                best_run_id = UUID(best["run_id"])
                best_params = best["params"]
                best_score = best["score"]
            elif gate_failures_count > 0:
                # No valid trials, and some were skipped due to gates
                warnings.append(
                    f"No trials passed gates ({gate_failures_count} skipped due to gate violations)"
                )
                logger.warning(
                    "No trials passed gates",
                    tune_id=str(tune_id),
                    gate_failures_count=gate_failures_count,
                )

            # Update tune with final results
            await self.tune_repo.complete_tune(
                tune_id=tune_id,
                best_run_id=best_run_id,
                best_score=best_score,
                best_params=best_params,
                leaderboard=leaderboard,
                trials_completed=trials_completed,
            )

            logger.info(
                "Parameter tuning completed",
                tune_id=str(tune_id),
                trials_completed=trials_completed,
                best_score=best_score,
            )

            return TuneResult(
                tune_id=tune_id,
                status="completed",
                n_trials=actual_trials,
                trials_completed=trials_completed,
                best_run_id=best_run_id,
                best_params=best_params,
                best_score=best_score,
                leaderboard=leaderboard,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(
                "Parameter tuning failed",
                tune_id=str(tune_id),
                error=str(e),
            )
            await self.tune_repo.update_tune_status(
                tune_id, "failed", error=str(e), completed_at=datetime.utcnow()
            )
            raise

    def _generate_params(
        self,
        param_space: dict[str, Any],
        search_type: str,
        n_trials: int,
        seed: Optional[int],
    ) -> list[dict[str, Any]]:
        """
        Generate parameter combinations for search.

        Args:
            param_space: Search space definition
            search_type: "grid" or "random"
            n_trials: Desired number of trials
            seed: Random seed

        Returns:
            List of parameter dicts to try
        """
        if search_type == "grid":
            return self._generate_grid_params(param_space, n_trials)
        else:
            return self._generate_random_params(param_space, n_trials, seed)

    def _generate_grid_params(
        self,
        param_space: dict[str, Any],
        max_trials: int,
    ) -> list[dict[str, Any]]:
        """Generate grid search parameter combinations."""
        # Convert space to lists
        param_lists = {}
        for name, spec in param_space.items():
            if isinstance(spec, list):
                param_lists[name] = spec
            elif isinstance(spec, dict) and "min" in spec and "max" in spec:
                # Continuous range - discretize to 5 points
                min_val = spec["min"]
                max_val = spec["max"]
                step = (max_val - min_val) / 4
                param_lists[name] = [min_val + i * step for i in range(5)]
            else:
                # Single value
                param_lists[name] = [spec]

        # Check grid size
        grid_size = 1
        for values in param_lists.values():
            grid_size *= len(values)

        if grid_size > self.MAX_GRID_SIZE:
            logger.warning(
                "Grid size exceeds maximum, truncating",
                grid_size=grid_size,
                max_size=self.MAX_GRID_SIZE,
            )

        # Generate combinations
        names = list(param_lists.keys())
        value_lists = [param_lists[n] for n in names]

        combinations = []
        for values in itertools.product(*value_lists):
            if len(combinations) >= max_trials:
                break
            combinations.append(dict(zip(names, values)))

        return combinations

    def _generate_random_params(
        self,
        param_space: dict[str, Any],
        n_trials: int,
        seed: Optional[int],
    ) -> list[dict[str, Any]]:
        """Generate random search parameter combinations."""
        rng = random.Random(seed)

        combinations = []
        for _ in range(n_trials):
            params = {}
            for name, spec in param_space.items():
                if isinstance(spec, list):
                    params[name] = rng.choice(spec)
                elif isinstance(spec, dict) and "min" in spec and "max" in spec:
                    min_val = spec["min"]
                    max_val = spec["max"]
                    if spec.get("type") == "int" or spec.get("type") == "integer":
                        params[name] = rng.randint(int(min_val), int(max_val))
                    else:
                        params[name] = rng.uniform(min_val, max_val)
                else:
                    params[name] = spec
            combinations.append(params)

        return combinations


def derive_param_space(
    param_schema: dict[str, Any],
    search_type: str = "random",
) -> dict[str, Any]:
    """
    Auto-derive tunable param space from compiled param_schema.

    Conservative derivation - creates small discrete sets around defaults.

    Args:
        param_schema: JSON schema for strategy parameters
        search_type: "grid" or "random"

    Returns:
        Parameter space dict suitable for tuner
    """
    space = {}
    properties = param_schema.get("properties", {})

    for name, spec in properties.items():
        # Enum - use values directly
        if "enum" in spec:
            space[name] = spec["enum"]

        # Range with min/max
        elif "minimum" in spec and "maximum" in spec:
            minimum = spec["minimum"]
            maximum = spec["maximum"]
            default = spec.get("default", (minimum + maximum) / 2)

            if spec.get("type") == "integer":
                # 5 discrete integer points around default
                points = [
                    int(default * 0.7),
                    int(default * 0.85),
                    int(default),
                    int(default * 1.15),
                    int(default * 1.3),
                ]
                # Clamp and dedupe
                clamped = sorted(
                    set(max(int(minimum), min(int(maximum), p)) for p in points)
                )
                space[name] = clamped

            elif search_type == "random":
                # Continuous range for random search
                space[name] = {
                    "min": float(minimum),
                    "max": float(maximum),
                    "type": "float",
                }

            else:
                # Grid with floats - 5 discrete points
                points = [default * m for m in [0.7, 0.85, 1.0, 1.15, 1.3]]
                clamped = [round(max(minimum, min(maximum, p)), 4) for p in points]
                space[name] = sorted(set(clamped))

        # Default only - fixed value
        elif "default" in spec:
            space[name] = [spec["default"]]

    return space
