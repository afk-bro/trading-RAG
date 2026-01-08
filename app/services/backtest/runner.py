"""Backtest runner - orchestrates strategy compilation, data loading, and execution."""

import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import structlog

from app.services.backtest.data import parse_ohlcv_csv, OHLCVParseResult, OHLCVParseError
from app.services.backtest.validate import validate_params, ParamValidationError
from app.services.backtest.engines.base import BacktestResult
from app.services.backtest.engines.backtestingpy import BacktestingPyEngine

logger = structlog.get_logger(__name__)


class BacktestRunError(Exception):
    """Error during backtest execution."""

    def __init__(self, message: str, code: str = "BACKTEST_ERROR", details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class BacktestRunner:
    """
    Orchestrates backtest execution from strategy spec to results.

    Flow:
    1. Load approved spec (from kb_repo)
    2. Compile spec (use cache if available)
    3. Validate params against param_schema
    4. Parse uploaded CSV data
    5. Run backtest engine
    6. Return results
    """

    def __init__(self, kb_repo, backtest_repo):
        """
        Initialize runner with repositories.

        Args:
            kb_repo: KnowledgeBaseRepository for strategy specs
            backtest_repo: BacktestRepository for persisting runs
        """
        self.kb_repo = kb_repo
        self.backtest_repo = backtest_repo
        self.engine = BacktestingPyEngine()

    async def run(
        self,
        strategy_entity_id: UUID,
        file_content: bytes,
        filename: str,
        params: dict[str, Any],
        workspace_id: UUID,
        initial_cash: float = 10000,
        commission_bps: float = 10,
        slippage_bps: float = 0,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        allow_draft: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a complete backtest run.

        Args:
            strategy_entity_id: The strategy entity UUID
            file_content: Raw CSV bytes
            filename: Original filename
            params: User-provided parameters
            workspace_id: Workspace for scoping
            initial_cash: Starting capital
            commission_bps: Commission in basis points
            slippage_bps: Slippage in basis points
            date_from: Optional date filter
            date_to: Optional date filter
            allow_draft: Allow running on draft specs

        Returns:
            Dict with run_id, status, summary, equity_curve, trades, warnings

        Raises:
            BacktestRunError: If backtest fails
        """
        warnings = []
        run_id = None

        try:
            # Step 1: Load and validate spec
            logger.info(
                "Loading strategy spec",
                strategy_entity_id=str(strategy_entity_id),
                allow_draft=allow_draft,
            )

            spec = await self.kb_repo.get_strategy_spec(strategy_entity_id)
            if not spec:
                raise BacktestRunError(
                    f"No strategy spec found for entity {strategy_entity_id}",
                    code="SPEC_NOT_FOUND",
                )

            spec_status = spec.get("status", "draft")
            if spec_status != "approved" and not allow_draft:
                raise BacktestRunError(
                    f"Strategy spec is '{spec_status}', not approved. Use allow_draft=true or approve the spec first.",
                    code="SPEC_NOT_APPROVED",
                    details={"status": spec_status},
                )

            spec_id = spec.get("id")
            spec_version = spec.get("version", 1)

            # Step 2: Compile spec (use cache)
            logger.info("Compiling strategy spec", spec_id=str(spec_id))
            compiled = await self.kb_repo.compile_strategy_spec(strategy_entity_id)
            if not compiled:
                raise BacktestRunError(
                    "Failed to compile strategy spec",
                    code="COMPILE_FAILED",
                )

            param_schema = compiled.get("param_schema", {})
            backtest_config = compiled.get("backtest_config", {})

            # Step 3: Validate params
            logger.info("Validating parameters", param_count=len(params))
            try:
                validated_params = validate_params(params, param_schema)
            except ParamValidationError as e:
                raise BacktestRunError(
                    e.message,
                    code="PARAM_VALIDATION_FAILED",
                    details={"errors": e.errors},
                )

            # Step 4: Parse CSV data
            logger.info("Parsing OHLCV data", filename=filename)
            try:
                parse_result = parse_ohlcv_csv(
                    file_content=file_content,
                    filename=filename,
                    date_from=date_from,
                    date_to=date_to,
                )
                warnings.extend(parse_result.warnings)
            except OHLCVParseError as e:
                raise BacktestRunError(
                    e.message,
                    code="DATA_PARSE_FAILED",
                    details=e.details,
                )

            # Create run record (status=running)
            dataset_meta = {
                "filename": filename,
                "row_count": parse_result.row_count,
                "date_min": parse_result.date_min.isoformat(),
                "date_max": parse_result.date_max.isoformat(),
            }

            run_id = await self.backtest_repo.create_run(
                workspace_id=workspace_id,
                strategy_entity_id=strategy_entity_id,
                strategy_spec_id=spec_id,
                spec_version=spec_version,
                params=validated_params,
                engine=self.engine.name,
                dataset_meta=dataset_meta,
            )

            logger.info("Created backtest run", run_id=str(run_id))

            # Step 5: Run backtest
            logger.info(
                "Executing backtest",
                run_id=str(run_id),
                data_rows=parse_result.row_count,
            )

            try:
                result = self.engine.run(
                    ohlcv_df=parse_result.df,
                    config=backtest_config,
                    params=validated_params,
                    initial_cash=initial_cash,
                    commission_bps=commission_bps,
                    slippage_bps=slippage_bps,
                )
                warnings.extend(result.warnings)
            except Exception as e:
                # Update run as failed
                await self.backtest_repo.update_run_failed(
                    run_id=run_id,
                    error=str(e),
                    warnings=warnings,
                )
                raise BacktestRunError(
                    f"Backtest execution failed: {e}",
                    code="EXECUTION_FAILED",
                    details={"engine_error": str(e)},
                )

            # Step 6: Persist results
            summary = {
                "return_pct": result.return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe": result.sharpe_ratio,
                "win_rate": result.win_rate,
                "trades": result.num_trades,
                "buy_hold_return_pct": result.buy_hold_return_pct,
                "avg_trade_pct": result.avg_trade_pct,
                "profit_factor": result.profit_factor,
            }

            await self.backtest_repo.update_run_completed(
                run_id=run_id,
                summary=summary,
                equity_curve=result.equity_curve,
                trades=result.trades,
                warnings=warnings,
            )

            logger.info(
                "Backtest completed",
                run_id=str(run_id),
                return_pct=result.return_pct,
                num_trades=result.num_trades,
            )

            return {
                "run_id": str(run_id),
                "status": "completed",
                "summary": summary,
                "equity_curve": result.equity_curve,
                "trades": result.trades,
                "warnings": warnings,
            }

        except BacktestRunError:
            raise
        except Exception as e:
            logger.error("Unexpected error in backtest", error=str(e))
            if run_id:
                await self.backtest_repo.update_run_failed(
                    run_id=run_id,
                    error=str(e),
                    warnings=warnings,
                )
            raise BacktestRunError(
                f"Unexpected error: {e}",
                code="INTERNAL_ERROR",
            )
