"""RunOrchestrator - Executes RunPlan variants through StrategyRunner + PaperBroker."""

import csv
import io
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

import structlog

from app.schemas import (
    PaperState,
    PaperPosition,
    TradeEvent,
    TradeEventType,
    IntentAction,
)
from app.services.strategy.models import (
    OHLCVBar,
    ExecutionSpec,
    MarketSnapshot,
    EntryConfig,
    ExitConfig,
    RiskConfig,
)
from app.services.testing.models import (
    RunPlan,
    RunVariant,
    RunResult,
    RunResultStatus,
    VariantMetrics,
    apply_overrides,
    validate_variant_params,
    get_variant_namespace,
    TESTING_VARIANT_NAMESPACE,
)
from app.services.kb.candidacy import (
    CandidacyConfig,
    VariantMetricsForCandidacy,
    is_candidate,
)
from app.services.kb.circuit_breaker import (
    CandidacyCircuitBreaker,
    BacktestBreakerAdapter,
)

logger = structlog.get_logger(__name__)

# Re-export for backwards compatibility
VARIANT_NS = TESTING_VARIANT_NAMESPACE

# Lambda for drawdown penalty objectives
DD_PENALTY_LAMBDA = 0.5


class RunOrchestrator:
    """Orchestrates execution of RunPlan variants with persistence.

    Responsibilities:
    - Parse OHLCV CSV dataset
    - Execute each variant through StrategyRunner + PaperBroker
    - Compute metrics (sharpe, return, max drawdown, etc.)
    - Journal RUN_STARTED and RUN_COMPLETED events
    - Persist run_plans and backtest_runs to database
    - Maintain variant isolation via uuid5 namespaces
    """

    def __init__(
        self,
        events_repo,
        runner,
        run_plans_repo=None,
        backtest_repo=None,
        candidacy_config: Optional[CandidacyConfig] = None,
    ):
        """Initialize orchestrator.

        Args:
            events_repo: TradeEventsRepository for journaling events
            runner: StrategyRunner for evaluating strategies
            run_plans_repo: RunPlansRepository for plan persistence (optional)
            backtest_repo: BacktestRepository for variant persistence (optional)
            candidacy_config: Configuration for KB candidacy gates (optional)
        """
        self._events_repo = events_repo
        self._runner = runner
        self._run_plans_repo = run_plans_repo
        self._backtest_repo = backtest_repo
        self._candidacy_config = candidacy_config or CandidacyConfig(
            # Disable regime requirement for test variants (not computed yet)
            require_regime=False,
        )
        # Circuit breaker is initialized lazily when backtest_repo is available
        self._circuit_breaker: Optional[CandidacyCircuitBreaker] = None
        if backtest_repo:
            adapter = BacktestBreakerAdapter(backtest_repo)
            self._circuit_breaker = CandidacyCircuitBreaker(adapter)

    async def execute(
        self, run_plan: RunPlan, dataset_content: bytes
    ) -> list[RunResult]:
        """Execute all variants in the run plan.

        Args:
            run_plan: The RunPlan containing variants to execute
            dataset_content: OHLCV CSV data as bytes

        Returns:
            List of RunResult, one per variant
        """
        log = logger.bind(
            run_plan_id=str(run_plan.run_plan_id),
            workspace_id=str(run_plan.workspace_id),
            n_variants=run_plan.n_variants,
        )

        # Parse dataset
        bars = self._parse_ohlcv_csv(dataset_content)
        log.info("dataset_parsed", bar_count=len(bars))

        # Build plan JSONB for persistence
        plan_jsonb = self._build_plan_jsonb(run_plan, dataset_content)

        # Persist run_plan at start
        db_plan_id = None
        if self._run_plans_repo:
            db_plan_id = await self._run_plans_repo.create_run_plan(
                workspace_id=run_plan.workspace_id,
                strategy_entity_id=None,  # Not linked to kb_entity in v0
                objective_name=run_plan.objective,
                n_variants=run_plan.n_variants,
                plan=plan_jsonb,
                status="pending",
            )
            # Update status to running
            await self._run_plans_repo.update_run_plan_status(db_plan_id, "running")
            log.info("run_plan_persisted", db_plan_id=str(db_plan_id))

        # Journal RUN_STARTED
        await self._journal_run_event(
            run_plan,
            "RUN_STARTED",
            {
                "n_variants": run_plan.n_variants,
                "objective": run_plan.objective,
                "dataset_ref": run_plan.dataset_ref,
                "bar_count": len(bars),
            },
        )

        # Execute variants with try/finally to ensure plan reaches terminal state
        results: list[RunResult] = []
        variant_run_ids: dict[str, Optional[UUID]] = {}  # variant_id -> db run_id
        plan_failed_unexpectedly = False
        unexpected_error: Optional[str] = None

        try:
            for idx, variant in enumerate(run_plan.variants):
                # Create pending row for this variant
                run_id = None
                if self._backtest_repo and db_plan_id:
                    run_id = await self._backtest_repo.create_variant_run(
                        run_plan_id=db_plan_id,
                        workspace_id=run_plan.workspace_id,
                        strategy_entity_id=UUID("00000000-0000-0000-0000-000000000000"),
                        variant_index=idx,
                        variant_fingerprint=variant.variant_id,
                        params=variant.spec_overrides,
                        dataset_meta={
                            "ref": run_plan.dataset_ref,
                            "bar_count": len(bars),
                        },
                        run_kind="test_variant",
                    )
                    variant_run_ids[variant.variant_id] = run_id

                try:
                    result = await self._execute_variant(run_plan, variant, bars)
                    results.append(result)

                    # Persist variant result
                    if self._backtest_repo and run_id:
                        await self._persist_variant_result(run_id, result)
                        # Evaluate and persist KB status for successful runs
                        await self._evaluate_and_persist_kb_status(
                            run_id=run_id,
                            workspace_id=run_plan.workspace_id,
                            result=result,
                            variant=variant,
                        )

                    log.info(
                        "variant_completed",
                        variant_id=variant.variant_id,
                        status=result.status,
                        objective_score=result.objective_score,
                    )
                except Exception as e:
                    # Catch variant-level errors
                    log.error(
                        "variant_failed",
                        variant_id=variant.variant_id,
                        error=str(e),
                    )
                    failed_result = RunResult(
                        run_plan_id=run_plan.run_plan_id,
                        variant_id=variant.variant_id,
                        status=RunResultStatus.failed,
                        error=str(e),
                        started_at=datetime.now(timezone.utc),
                        completed_at=datetime.now(timezone.utc),
                    )
                    results.append(failed_result)

                    # Persist failed result
                    if self._backtest_repo and run_id:
                        await self._backtest_repo.update_variant_failed(run_id, str(e))

        except Exception as e:
            # Unexpected error during execution loop (not variant-level)
            log.error("run_plan_failed_unexpectedly", error=str(e))
            plan_failed_unexpectedly = True
            unexpected_error = str(e)

        finally:
            # ALWAYS finalize the plan - never leave stuck in "running"
            n_success = sum(1 for r in results if r.status == RunResultStatus.success)
            n_skipped = sum(1 for r in results if r.status == RunResultStatus.skipped)
            n_failed = sum(1 for r in results if r.status == RunResultStatus.failed)
            best_id, best_score = select_best_variant(results)

            # Find best variant's DB run ID
            best_run_id = variant_run_ids.get(best_id) if best_id else None

            # Determine final status
            if plan_failed_unexpectedly:
                final_status = "failed"
            else:
                final_status = "completed"

            # Complete run_plan with aggregates
            if self._run_plans_repo and db_plan_id:
                await self._run_plans_repo.complete_run_plan(
                    plan_id=db_plan_id,
                    status=final_status,
                    n_completed=n_success,
                    n_failed=n_failed,
                    n_skipped=n_skipped,
                    best_backtest_run_id=best_run_id,
                    best_objective_score=best_score,
                    error_summary=unexpected_error,
                )

            # Journal appropriate event
            if plan_failed_unexpectedly:
                await self._journal_run_event(
                    run_plan,
                    "RUN_FAILED",
                    {
                        "n_variants": run_plan.n_variants,
                        "n_success": n_success,
                        "n_skipped": n_skipped,
                        "n_failed": n_failed,
                        "error": unexpected_error,
                    },
                )
            else:
                await self._journal_run_event(
                    run_plan,
                    "RUN_COMPLETED",
                    {
                        "n_variants": run_plan.n_variants,
                        "n_success": n_success,
                        "n_skipped": n_skipped,
                        "n_failed": n_failed,
                        "best_variant_id": best_id,
                        "best_objective_score": best_score,
                    },
                )

            log.info(
                "run_finalized",
                status=final_status,
                n_success=n_success,
                n_skipped=n_skipped,
                n_failed=n_failed,
                best_variant_id=best_id,
                best_score=best_score,
            )

        # Re-raise if we had an unexpected failure
        if plan_failed_unexpectedly:
            raise RuntimeError(f"Run plan failed unexpectedly: {unexpected_error}")

        return results

    async def _persist_variant_result(self, run_id: UUID, result: RunResult) -> None:
        """Persist variant result to backtest_runs.

        Args:
            run_id: The backtest_runs row ID
            result: The RunResult to persist
        """
        if result.status == RunResultStatus.skipped:
            await self._backtest_repo.update_variant_skipped(
                run_id=run_id,
                skip_reason=result.skip_reason or "unknown",
            )
        elif result.status == RunResultStatus.failed:
            await self._backtest_repo.update_variant_failed(
                run_id=run_id,
                error=result.error or "unknown error",
            )
        elif result.status == RunResultStatus.success and result.metrics:
            # Build summary from metrics
            summary = {
                "sharpe": result.metrics.sharpe,
                "return_pct": result.metrics.return_pct,
                "max_drawdown_pct": result.metrics.max_drawdown_pct,
                "trade_count": result.metrics.trade_count,
                "win_rate": result.metrics.win_rate,
                "ending_equity": result.metrics.ending_equity,
                "profit_factor": result.metrics.profit_factor,
            }
            await self._backtest_repo.update_variant_completed(
                run_id=run_id,
                summary=summary,
                equity_curve=[],  # Empty for now - full curve not stored in v0
                trades=[],  # Empty for now
                objective_score=result.objective_score,
                has_equity_curve=False,
                has_trades=False,
                equity_points=0,
                trade_count=result.metrics.trade_count,
            )

    async def _evaluate_and_persist_kb_status(
        self,
        run_id: UUID,
        workspace_id: UUID,
        result: RunResult,
        variant: RunVariant,
    ) -> None:
        """Evaluate candidacy gates and persist KB status.

        Called after variant completion for successful runs.
        Determines if the variant should be auto-promoted to 'candidate' status.

        Args:
            run_id: The backtest_runs row ID
            workspace_id: Workspace ID for circuit breaker
            result: The completed RunResult
            variant: The variant definition (for experiment type)
        """
        if not self._backtest_repo:
            return

        # Only evaluate successful runs with metrics
        if result.status != RunResultStatus.success or not result.metrics:
            return

        # Determine experiment type from variant tags
        experiment_type = "sweep"  # Default for test variants
        if "ablation" in variant.tags:
            experiment_type = "ablation"
        elif "baseline" in variant.tags:
            experiment_type = "sweep"  # baseline is still part of sweep
        elif "manual" in variant.tags:
            experiment_type = "manual"

        # Build metrics for candidacy check
        # Note: Test variants don't have IS/OOS split, so we use full metrics
        # max_drawdown_pct is a percentage (e.g., 10.5), convert to fraction
        candidacy_metrics = VariantMetricsForCandidacy(
            n_trades_oos=result.metrics.trade_count,
            max_dd_frac_oos=result.metrics.max_drawdown_pct / 100.0,
            overfit_gap=None,  # No IS/OOS split in test variants
            sharpe_oos=result.metrics.sharpe,
        )

        # Evaluate candidacy gates
        decision = is_candidate(
            metrics=candidacy_metrics,
            regime_oos=None,  # Regime not computed for test variants yet
            experiment_type=experiment_type,
            config=self._candidacy_config,
        )

        kb_status = "excluded"
        auto_candidate_gate = decision.reason
        auto_candidate_breaker = None

        if decision.eligible:
            # Gates passed - check circuit breaker
            if self._circuit_breaker:
                allowed, trip_reason = await self._circuit_breaker.check(workspace_id)
                if allowed:
                    kb_status = "candidate"
                else:
                    kb_status = "excluded"
                    auto_candidate_breaker = trip_reason
            else:
                # No circuit breaker configured - allow candidacy
                kb_status = "candidate"

        # Persist KB status
        await self._backtest_repo.update_variant_kb_status(
            run_id=run_id,
            kb_status=kb_status,
            auto_candidate_gate=auto_candidate_gate,
            auto_candidate_breaker=auto_candidate_breaker,
        )

        logger.info(
            "kb_status_evaluated",
            run_id=str(run_id),
            kb_status=kb_status,
            gate=auto_candidate_gate,
            breaker=auto_candidate_breaker,
            experiment_type=experiment_type,
        )

    def _build_plan_jsonb(self, run_plan: RunPlan, dataset_content: bytes) -> dict:
        """Build the plan JSONB structure for persistence.

        Args:
            run_plan: The RunPlan to serialize
            dataset_content: Dataset bytes (for bar count)

        Returns:
            Dict with inputs, resolved, provenance sections
        """
        from app.services.testing.plan_builder import PlanBuilder

        builder = PlanBuilder(
            base_spec=run_plan.base_spec,
            objective=run_plan.objective,
            constraints={},  # Constraints not preserved in RunPlan currently
            dataset_ref=run_plan.dataset_ref,
            generator_name="run_orchestrator",
            generator_version="1.0.0",
        )

        for idx, variant in enumerate(run_plan.variants):
            # Determine param_source from tags
            param_source = "unknown"
            if "baseline" in variant.tags:
                param_source = "baseline"
            elif "grid" in variant.tags:
                param_source = "grid"
            elif "ablation" in variant.tags:
                param_source = "ablation"
            elif not variant.spec_overrides:
                param_source = "baseline"

            builder.add_variant(
                variant_index=idx,
                params=variant.spec_overrides,
                param_source=param_source,
            )

        return builder.build()

    async def _execute_variant(
        self, run_plan: RunPlan, variant: RunVariant, bars: list[OHLCVBar]
    ) -> RunResult:
        """Execute a single variant.

        Args:
            run_plan: Parent run plan
            variant: The variant to execute
            bars: OHLCV bars for simulation

        Returns:
            RunResult with metrics
        """
        started_at = datetime.now(timezone.utc)

        # Generate isolated namespace for this variant using helper
        variant_namespace = get_variant_namespace(
            run_plan.run_plan_id, variant.variant_id
        )

        log = logger.bind(
            run_plan_id=str(run_plan.run_plan_id),
            variant_id=variant.variant_id,
            variant_namespace=str(variant_namespace),
        )

        try:
            # Materialize spec with overrides
            materialized_spec = apply_overrides(
                run_plan.base_spec, variant.spec_overrides
            )

            # Validate params BEFORE execution - skip obviously broken variants
            is_valid, validation_error = validate_variant_params(materialized_spec)
            if not is_valid:
                log.info(
                    "variant_skipped_invalid_params",
                    reason=validation_error,
                )
                completed_at = datetime.now(timezone.utc)
                duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                return RunResult(
                    run_plan_id=run_plan.run_plan_id,
                    variant_id=variant.variant_id,
                    status=RunResultStatus.skipped,
                    skip_reason=validation_error,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                )

            # Create ExecutionSpec from materialized spec
            exec_spec = self._build_execution_spec(materialized_spec, variant_namespace)

            # Create isolated PaperState for this variant
            starting_equity = 10000.0
            paper_state = PaperState(
                workspace_id=variant_namespace,  # Isolated namespace
                starting_equity=starting_equity,
                cash=starting_equity,
                realized_pnl=0.0,
                last_event_id=None,
                last_event_at=None,
                reconciled_at=None,
            )

            # Run simulation through bars
            closed_trades, events_count = self._simulate_bars(
                exec_spec, paper_state, bars, log
            )

            # Calculate metrics
            metrics = self._calculate_metrics(
                starting_equity, paper_state, closed_trades
            )

            # Compute objective score
            objective_score = self._compute_objective_score(metrics, run_plan.objective)

            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RunResult(
                run_plan_id=run_plan.run_plan_id,
                variant_id=variant.variant_id,
                status=RunResultStatus.success,
                metrics=metrics,
                objective_score=objective_score,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                events_recorded=events_count,
            )

        except Exception as e:
            log.error("variant_execution_error", error=str(e))
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            return RunResult(
                run_plan_id=run_plan.run_plan_id,
                variant_id=variant.variant_id,
                status=RunResultStatus.failed,
                error=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

    def _build_execution_spec(
        self, spec_dict: dict, instance_id: UUID
    ) -> ExecutionSpec:
        """Build ExecutionSpec from materialized spec dict.

        Args:
            spec_dict: Materialized specification dictionary
            instance_id: Unique instance ID for this variant

        Returns:
            ExecutionSpec ready for evaluation
        """
        return ExecutionSpec(
            strategy_id=spec_dict["strategy_id"],
            instance_id=instance_id,
            name=spec_dict.get("name", "variant"),
            workspace_id=(
                UUID(spec_dict["workspace_id"])
                if isinstance(spec_dict["workspace_id"], str)
                else spec_dict["workspace_id"]
            ),
            symbols=spec_dict["symbols"],
            timeframe=spec_dict["timeframe"],
            entry=EntryConfig(**spec_dict["entry"]),
            exit=ExitConfig(**spec_dict["exit"]),
            risk=RiskConfig(**spec_dict["risk"]),
        )

    def _simulate_bars(
        self,
        spec: ExecutionSpec,
        paper_state: PaperState,
        bars: list[OHLCVBar],
        log,
    ) -> tuple[list[dict], int]:
        """Simulate strategy execution through OHLCV bars.

        In-memory simulation that:
        1. Builds MarketSnapshot for each bar
        2. Calls runner.evaluate() to get intents
        3. Executes intents (open/close positions)
        4. Tracks closed trades for metrics

        Args:
            spec: ExecutionSpec for this variant
            paper_state: Isolated paper trading state
            bars: OHLCV bars to simulate through
            log: Bound logger for this variant

        Returns:
            Tuple of (closed_trades list, event count)
        """
        closed_trades: list[dict] = []
        events_count = 0
        symbol = spec.symbols[0]  # Single symbol for v0

        # Track open position for this symbol
        open_position: Optional[dict] = None

        # Need at least lookback + 1 bars to start evaluating
        min_bars = max(spec.entry.lookback_days, 2)

        for i, bar in enumerate(bars):
            # Skip until we have enough history
            if i < min_bars:
                continue

            # Build MarketSnapshot with history up to current bar
            history = bars[: i + 1]
            is_last_bar = i == len(bars) - 1

            snapshot = MarketSnapshot(
                symbol=symbol,
                ts=bar.ts,
                timeframe=spec.timeframe,
                bars=history,
                is_eod=is_last_bar,  # Force exit on last bar
                last_price=None,
                high_52w=None,
                low_52w=None,
            )

            # Evaluate strategy
            evaluation = self._runner.evaluate(spec, snapshot, paper_state)
            events_count += 1

            # Process intents
            for intent in evaluation.intents:
                if intent.action == IntentAction.OPEN_LONG and open_position is None:
                    # Open new position
                    qty = spec.risk.dollars_per_trade / bar.close
                    cost = qty * bar.close
                    if cost <= paper_state.cash:
                        paper_state.cash -= cost
                        open_position = {
                            "symbol": symbol,
                            "qty": qty,
                            "entry_price": bar.close,
                            "entry_ts": bar.ts,
                        }
                        # Add to paper_state positions
                        paper_state.positions[symbol] = PaperPosition(
                            workspace_id=paper_state.workspace_id,
                            symbol=symbol,
                            side="long",
                            quantity=qty,
                            avg_price=bar.close,
                            unrealized_pnl=0.0,
                            opened_at=bar.ts,
                        )
                        log.debug(
                            "position_opened",
                            symbol=symbol,
                            qty=qty,
                            price=bar.close,
                        )

                elif intent.action == IntentAction.CLOSE_LONG and open_position:
                    # Close position
                    exit_price = bar.close
                    pnl = (exit_price - open_position["entry_price"]) * open_position[
                        "qty"
                    ]
                    proceeds = open_position["qty"] * exit_price
                    paper_state.cash += proceeds
                    paper_state.realized_pnl += pnl

                    # Track equity at close for drawdown calculation
                    exit_equity = paper_state.cash + sum(
                        p.quantity * p.avg_price for p in paper_state.positions.values()
                    )

                    closed_trades.append(
                        {
                            "symbol": symbol,
                            "entry_price": open_position["entry_price"],
                            "exit_price": exit_price,
                            "qty": open_position["qty"],
                            "pnl": pnl,
                            "exit_equity": exit_equity,
                            "entry_ts": open_position["entry_ts"],
                            "exit_ts": bar.ts,
                        }
                    )

                    # Remove from positions
                    if symbol in paper_state.positions:
                        del paper_state.positions[symbol]
                    open_position = None

                    log.debug(
                        "position_closed",
                        symbol=symbol,
                        pnl=pnl,
                        exit_price=exit_price,
                    )

        # Force close any remaining position at end
        if open_position and bars:
            last_bar = bars[-1]
            exit_price = last_bar.close
            pnl = (exit_price - open_position["entry_price"]) * open_position["qty"]
            proceeds = open_position["qty"] * exit_price
            paper_state.cash += proceeds
            paper_state.realized_pnl += pnl

            exit_equity = paper_state.cash

            closed_trades.append(
                {
                    "symbol": symbol,
                    "entry_price": open_position["entry_price"],
                    "exit_price": exit_price,
                    "qty": open_position["qty"],
                    "pnl": pnl,
                    "exit_equity": exit_equity,
                    "entry_ts": open_position["entry_ts"],
                    "exit_ts": last_bar.ts,
                }
            )

            if symbol in paper_state.positions:
                del paper_state.positions[symbol]

            log.debug(
                "position_force_closed",
                symbol=symbol,
                pnl=pnl,
                reason="end_of_data",
            )

        return closed_trades, events_count

    @staticmethod
    def _parse_ohlcv_csv(content: bytes) -> list[OHLCVBar]:
        """Parse OHLCV CSV content into bars.

        Strict parsing with minimal guardrails:
        - BOM stripping (UTF-8 BOM: EF BB BF)
        - Required column validation
        - Timestamp parsing (fail fast)
        - Monotonic increasing timestamps
        - ≥2 bars required

        Explicitly NOT implemented (deferred to v1+):
        - Delimiter guessing
        - Gap interpolation
        - Resampling / downsampling
        - Timezone inference

        Expected columns: ts, open, high, low, close, volume

        Args:
            content: CSV data as bytes

        Returns:
            List of OHLCVBar with monotonically increasing timestamps

        Raises:
            ValueError: If CSV is invalid, has < 2 bars, or non-monotonic timestamps
        """
        # Strip UTF-8 BOM if present (EF BB BF = "\ufeff")
        if content.startswith(b"\xef\xbb\xbf"):
            content = content[3:]

        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))

        # Validate required columns
        required_columns = {"ts", "open", "high", "low", "close", "volume"}
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        # Strip BOM from first column name if present (can happen with some encodings)
        fieldnames = [f.lstrip("\ufeff") for f in reader.fieldnames]
        missing = required_columns - set(fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        bars: list[OHLCVBar] = []
        prev_ts: datetime | None = None
        row_num = 1  # Start at 1 for human-readable error messages

        for row in reader:
            row_num += 1
            try:
                ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))

                # Validate monotonically increasing timestamps
                if prev_ts is not None and ts <= prev_ts:
                    raise ValueError(
                        f"Non-monotonic timestamp at row {row_num}: "
                        f"{ts.isoformat()} <= {prev_ts.isoformat()}"
                    )
                prev_ts = ts

                bar = OHLCVBar(
                    ts=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
                bars.append(bar)
            except (ValueError, KeyError) as e:
                raise ValueError(f"Invalid row {row_num} in CSV: {row}, error: {e}")

        if len(bars) < 2:
            raise ValueError("CSV must contain at least 2 bars")

        return bars

    @staticmethod
    def _calculate_metrics(
        starting_equity: float,
        paper_state: PaperState,
        closed_trades: list[dict],
    ) -> VariantMetrics:
        """Calculate metrics from closed trades and final state.

        Args:
            starting_equity: Initial equity value
            paper_state: Final paper trading state
            closed_trades: List of closed trade dicts with 'pnl' and 'exit_equity'

        Returns:
            VariantMetrics with all computed values
        """
        trade_count = len(closed_trades)

        if trade_count == 0:
            return VariantMetrics(
                sharpe=None,
                return_pct=0.0,
                max_drawdown_pct=0.0,
                trade_count=0,
                win_rate=0.0,
                ending_equity=starting_equity,
                gross_profit=0.0,
                gross_loss=0.0,
                profit_factor=None,
            )

        # Calculate win rate
        wins = sum(1 for t in closed_trades if t["pnl"] > 0)
        win_rate = wins / trade_count if trade_count > 0 else 0.0

        # Calculate gross profit and loss
        gross_profit = sum(t["pnl"] for t in closed_trades if t["pnl"] > 0)
        gross_loss = sum(t["pnl"] for t in closed_trades if t["pnl"] < 0)

        # Profit factor: None if no losses
        profit_factor = None
        if gross_loss < 0:
            profit_factor = gross_profit / abs(gross_loss)

        # Build equity curve: [starting_equity, exit_equity_1, exit_equity_2, ...]
        equity_curve = [starting_equity]
        for trade in closed_trades:
            equity_curve.append(trade["exit_equity"])

        # Ending equity
        ending_equity = equity_curve[-1] if len(equity_curve) > 1 else starting_equity

        # Return percentage
        return_pct = ((ending_equity - starting_equity) / starting_equity) * 100

        # Max drawdown
        max_drawdown_pct = RunOrchestrator._compute_max_drawdown(equity_curve)

        # Sharpe ratio
        sharpe = RunOrchestrator._compute_trade_sharpe(equity_curve)

        return VariantMetrics(
            sharpe=sharpe,
            return_pct=return_pct,
            max_drawdown_pct=max_drawdown_pct,
            trade_count=trade_count,
            win_rate=win_rate,
            ending_equity=ending_equity,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
        )

    @staticmethod
    def _compute_max_drawdown(equity_curve: list[float]) -> float:
        """Compute maximum drawdown percentage from equity curve.

        Args:
            equity_curve: List of equity values over time

        Returns:
            Maximum drawdown as a percentage (e.g., 13.6 for 13.6%)
        """
        if len(equity_curve) <= 1:
            return 0.0

        max_dd = 0.0
        peak = equity_curve[0]

        for equity in equity_curve[1:]:
            if equity > peak:
                peak = equity
            else:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        return max_dd

    @staticmethod
    def _compute_trade_sharpe(equity_curve: list[float]) -> Optional[float]:
        """Compute Sharpe ratio from equity curve.

        Uses trade-to-trade returns (percentage changes between equity points).
        Requires at least 3 points (2 trades) to compute.

        Args:
            equity_curve: List of equity values over time

        Returns:
            Sharpe ratio or None if insufficient data or zero std
        """
        if len(equity_curve) < 3:
            return None

        # Calculate percentage returns between consecutive points
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            curr = equity_curve[i]
            ret = (curr - prev) / prev * 100
            returns.append(ret)

        if len(returns) < 2:
            return None

        # Calculate mean and std
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        std_ret = variance**0.5

        if std_ret == 0:
            return None

        return mean_ret / std_ret

    @staticmethod
    def _compute_objective_score(metrics: VariantMetrics, objective: str) -> float:
        """Compute objective score from metrics.

        Supported objectives:
        - sharpe: Sharpe ratio (returns -999.0 if sharpe is None)
        - sharpe_dd_penalty: sharpe - 0.5 * (max_dd_pct / 100)
        - return: return_pct
        - return_dd_penalty: return_pct - 0.5 * max_dd_pct
        - calmar: return_pct / abs(max_dd_pct), 0 if max_dd is 0

        Args:
            metrics: VariantMetrics to compute score from
            objective: Objective type string

        Returns:
            Objective score (float)
        """
        if objective == "sharpe":
            if metrics.sharpe is None:
                return -999.0
            return metrics.sharpe

        elif objective == "sharpe_dd_penalty":
            if metrics.sharpe is None:
                return -999.0
            penalty = DD_PENALTY_LAMBDA * (metrics.max_drawdown_pct / 100.0)
            return metrics.sharpe - penalty

        elif objective == "return":
            return metrics.return_pct

        elif objective == "return_dd_penalty":
            penalty = DD_PENALTY_LAMBDA * metrics.max_drawdown_pct
            return metrics.return_pct - penalty

        elif objective == "calmar":
            if metrics.max_drawdown_pct == 0:
                return 0.0
            return metrics.return_pct / abs(metrics.max_drawdown_pct)

        else:
            # Unknown objective, default to sharpe
            if metrics.sharpe is None:
                return -999.0
            return metrics.sharpe

    async def _journal_run_event(
        self, run_plan: RunPlan, event_type: str, payload: dict
    ) -> None:
        """Journal a run-level event.

        Args:
            run_plan: The RunPlan this event belongs to
            event_type: Event type string (RUN_STARTED, RUN_COMPLETED, etc.)
            payload: Event payload dict
        """
        # Map event type string to TradeEventType enum
        event_type_map = {
            "RUN_STARTED": TradeEventType.RUN_STARTED,
            "RUN_COMPLETED": TradeEventType.RUN_COMPLETED,
            "RUN_FAILED": TradeEventType.RUN_FAILED,
            "RUN_CANCELLED": TradeEventType.RUN_CANCELLED,
        }

        mapped_type = event_type_map.get(event_type)
        if not mapped_type:
            logger.warning(
                "unknown_run_event_type",
                event_type=event_type,
                run_plan_id=str(run_plan.run_plan_id),
            )
            return

        event = TradeEvent(
            correlation_id=str(run_plan.run_plan_id),
            workspace_id=run_plan.workspace_id,
            event_type=mapped_type,
            payload=payload,
            strategy_entity_id=None,
            symbol=None,
            timeframe=None,
            intent_id=None,
            order_id=None,
            position_id=None,
        )

        try:
            await self._events_repo.insert(event)
        except Exception as e:
            logger.warning(
                "failed_to_journal_run_event",
                event_type=event_type,
                error=str(e),
            )


def select_best_variant(
    results: list[RunResult],
) -> tuple[Optional[str], Optional[float]]:
    """Select the best variant from results.

    Selection criteria with deterministic tie-breaking:
    1. Highest objective score
    2. Higher return_pct (tie-break)
    3. Lower max_drawdown_pct (tie-break)
    4. Smaller variant_id lexicographically (tie-break)

    Only considers successful results with objective_score.

    Args:
        results: List of RunResult from variant execution

    Returns:
        Tuple of (best_variant_id, best_objective_score) or (None, None)
    """
    if not results:
        return None, None

    # Filter to successful results with scores
    valid_results = [
        r
        for r in results
        if r.status == RunResultStatus.success
        and r.objective_score is not None
        and r.metrics is not None
    ]

    if not valid_results:
        return None, None

    # Sort by: -objective_score, -return_pct, +max_drawdown_pct, +variant_id
    def sort_key(r: RunResult) -> tuple[float, float, float, str]:
        # Type narrowing: we filtered to ensure these are not None
        assert r.objective_score is not None
        assert r.metrics is not None
        return (
            -r.objective_score,  # Higher is better
            -r.metrics.return_pct,  # Higher is better
            r.metrics.max_drawdown_pct,  # Lower is better
            r.variant_id,  # Smaller is better
        )

    sorted_results = sorted(valid_results, key=sort_key)
    best = sorted_results[0]

    return best.variant_id, best.objective_score
