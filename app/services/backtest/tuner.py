"""Parameter tuning orchestration for backtests.

Coordinates grid/random search over strategy parameters,
running multiple backtests and tracking results.
"""

import asyncio
import heapq
import itertools
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.backtest.runner import BacktestRunner, BacktestRunError
from app.services.backtest.scoring import compute_score, rank_trials

logger = structlog.get_logger(__name__)


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

    MAX_CONCURRENCY = 4
    MAX_GRID_SIZE = 200

    def __init__(self, kb_repo, backtest_repo, tune_repo):
        self.kb_repo = kb_repo
        self.backtest_repo = backtest_repo
        self.tune_repo = tune_repo
        self.runner = BacktestRunner(kb_repo, backtest_repo)

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
        )

        # Update tune status to running
        await self.tune_repo.update_tune_status(tune_id, "running", started_at=datetime.utcnow())

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
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENCY)
            results = []
            trials_completed = 0

            async def run_trial(idx: int, params: dict) -> Optional[dict]:
                nonlocal trials_completed

                async with semaphore:
                    # Update status to running
                    await self.tune_repo.update_tune_run_status(tune_id, idx, "running")

                    try:
                        # Run backtest with these params
                        result = await self.runner.run(
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
                            allow_draft=True,  # Allow draft specs during tuning
                        )

                        summary = result["summary"]
                        run_id = UUID(result["run_id"])

                        # Compute score
                        score = compute_score(summary, objective_metric, min_trades)

                        if score is None:
                            # Skipped - insufficient trades or missing metric
                            await self.tune_repo.update_tune_run_result(
                                tune_id=tune_id,
                                trial_index=idx,
                                run_id=run_id,
                                score=None,
                                status="skipped",
                            )
                            logger.debug(
                                "Trial skipped",
                                tune_id=str(tune_id),
                                trial_index=idx,
                                trades=summary.get("trades", 0),
                            )
                            return None

                        # Completed successfully
                        await self.tune_repo.update_tune_run_result(
                            tune_id=tune_id,
                            trial_index=idx,
                            run_id=run_id,
                            score=score,
                            status="completed",
                        )

                        trials_completed += 1

                        # Update progress periodically
                        if trials_completed % 5 == 0:
                            await self.tune_repo.update_tune_progress(tune_id, trials_completed)

                        return {
                            "trial_index": idx,
                            "run_id": str(run_id),
                            "params": params,
                            "score": score,
                            "summary": summary,
                        }

                    except BacktestRunError as e:
                        logger.warning(
                            "Trial failed",
                            tune_id=str(tune_id),
                            trial_index=idx,
                            error=e.message,
                        )
                        await self.tune_repo.update_tune_run_status(
                            tune_id, idx, "failed", error=e.message
                        )
                        return None
                    except Exception as e:
                        logger.error(
                            "Trial error",
                            tune_id=str(tune_id),
                            trial_index=idx,
                            error=str(e),
                        )
                        await self.tune_repo.update_tune_run_status(
                            tune_id, idx, "failed", error=str(e)
                        )
                        return None

            # Execute all trials
            tasks = [run_trial(i, p) for i, p in enumerate(param_sets)]
            trial_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful results
            valid_results = [r for r in trial_results if isinstance(r, dict) and r is not None]

            # Build leaderboard
            leaderboard = rank_trials(valid_results, objective_metric, min_trades, top_n=10)

            # Determine best
            best_run_id = None
            best_params = None
            best_score = None

            if leaderboard:
                best = leaderboard[0]
                best_run_id = UUID(best["run_id"])
                best_params = best["params"]
                best_score = best["score"]

            # Update tune with final results
            await self.tune_repo.complete_tune(
                tune_id=tune_id,
                best_run_id=best_run_id,
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
                clamped = sorted(set(
                    max(int(minimum), min(int(maximum), p)) for p in points
                ))
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
                clamped = [
                    round(max(minimum, min(maximum, p)), 4)
                    for p in points
                ]
                space[name] = sorted(set(clamped))

        # Default only - fixed value
        elif "default" in spec:
            space[name] = [spec["default"]]

    return space
