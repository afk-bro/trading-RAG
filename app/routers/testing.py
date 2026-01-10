"""Testing API endpoints for generating and executing run plans."""

import json
from typing import Any, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.repositories.trade_events import TradeEventsRepository
from app.services.strategy.models import ExecutionSpec
from app.services.strategy.runner import StrategyRunner
from app.services.testing import (
    GeneratorConstraints,
    RunOrchestrator,
    RunPlan,
    TestGenerator,
    VariantMetrics,
    select_best_variant,
)


router = APIRouter(prefix="/testing", tags=["testing"])
logger = structlog.get_logger(__name__)

# Global connection pool (set during app startup)
_db_pool = None


def set_db_pool(pool):
    """Set the database pool for this router."""
    global _db_pool
    _db_pool = pool


def _get_events_repo() -> TradeEventsRepository:
    """Get events repository."""
    if _db_pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not available",
        )
    return TradeEventsRepository(_db_pool)


# =============================================================================
# Request/Response Models
# =============================================================================


class GenerateRequest(BaseModel):
    """Request for generating a run plan."""

    workspace_id: UUID = Field(..., description="Workspace UUID")
    base_spec: dict[str, Any] = Field(
        ..., description="Base strategy specification (validated as ExecutionSpec)"
    )
    dataset_ref: str = Field(
        ..., description="Reference to the dataset for backtesting"
    )
    constraints: dict[str, Any] = Field(
        ..., description="Generator constraints (validated as GeneratorConstraints)"
    )
    objective: str = Field(
        default="sharpe_dd_penalty", description="Objective function for optimization"
    )


class VariantSummary(BaseModel):
    """Summary of a single variant."""

    variant_id: str = Field(..., description="Variant identifier (16-char hex)")
    label: str = Field(..., description="Human-readable label")
    tags: list[str] = Field(default_factory=list, description="Variant tags")


class RunPlanResponse(BaseModel):
    """Response from run plan generation."""

    run_plan_id: UUID = Field(..., description="Unique run plan identifier")
    workspace_id: UUID = Field(..., description="Workspace UUID")
    n_variants: int = Field(..., description="Number of variants in the plan")
    objective: str = Field(..., description="Objective function")
    variants: list[VariantSummary] = Field(..., description="List of variant summaries")


class ResultSummary(BaseModel):
    """Summary of a single variant execution result."""

    variant_id: str = Field(..., description="Variant identifier")
    status: str = Field(..., description="'success' or 'failed'")
    objective_score: Optional[float] = Field(
        None, description="Computed objective score"
    )
    metrics: Optional[VariantMetrics] = Field(None, description="Variant metrics")


class ExecuteRunPlanResponse(BaseModel):
    """Response from run plan generation and execution."""

    run_plan_id: UUID = Field(..., description="Unique run plan identifier")
    n_variants: int = Field(..., description="Number of variants executed")
    results: list[ResultSummary] = Field(..., description="Execution results")
    best_variant_id: Optional[str] = Field(None, description="Best variant ID")
    best_score: Optional[float] = Field(None, description="Best objective score")


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/run-plans/generate",
    response_model=RunPlanResponse,
    responses={
        200: {"description": "Run plan generated successfully"},
        422: {"description": "Invalid base_spec or constraints"},
        503: {"description": "Database unavailable"},
    },
    summary="Generate a run plan",
    description="Generate a RunPlan with variants from a base ExecutionSpec and constraints.",
)
async def generate_run_plan(request: GenerateRequest) -> RunPlanResponse:
    """
    Generate a run plan from base specification and constraints.

    The generator produces:
    1. Baseline variant (always first, empty overrides)
    2. Grid variants (cartesian product of sweep values)
    3. Ablation variants (one param reset to base default per variant)

    Deduplication ensures no duplicate variant_ids.
    """
    log = logger.bind(
        workspace_id=str(request.workspace_id),
        dataset_ref=request.dataset_ref,
        objective=request.objective,
    )
    log.info("generate_run_plan_request")

    # Validate base_spec as ExecutionSpec
    try:
        base_spec = ExecutionSpec(**request.base_spec)
    except Exception as e:
        log.warning("invalid_base_spec", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Invalid base_spec: {e}",
                "error_code": "INVALID_BASE_SPEC",
            },
        )

    # Validate constraints as GeneratorConstraints
    try:
        constraints = GeneratorConstraints(**request.constraints)
        # Override objective from request if provided
        constraints.objective = request.objective
    except Exception as e:
        log.warning("invalid_constraints", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Invalid constraints: {e}",
                "error_code": "INVALID_CONSTRAINTS",
            },
        )

    # Generate run plan
    generator = TestGenerator()
    run_plan = generator.generate(
        base_spec=base_spec,
        dataset_ref=request.dataset_ref,
        constraints=constraints,
    )

    log.info(
        "run_plan_generated",
        run_plan_id=str(run_plan.run_plan_id),
        n_variants=run_plan.n_variants,
    )

    # Build response
    variants = [
        VariantSummary(
            variant_id=v.variant_id,
            label=v.label,
            tags=v.tags,
        )
        for v in run_plan.variants
    ]

    return RunPlanResponse(
        run_plan_id=run_plan.run_plan_id,
        workspace_id=run_plan.workspace_id,
        n_variants=run_plan.n_variants,
        objective=run_plan.objective,
        variants=variants,
    )


@router.post(
    "/run-plans/generate-and-execute",
    response_model=ExecuteRunPlanResponse,
    responses={
        200: {"description": "Run plan generated and executed successfully"},
        422: {"description": "Invalid parameters or CSV data"},
        503: {"description": "Database unavailable"},
    },
    summary="Generate and execute a run plan",
    description="Generate a RunPlan and immediately execute all variants against uploaded OHLCV data.",
)
async def generate_and_execute_run_plan(
    file: UploadFile = File(
        ..., description="OHLCV CSV file (columns: ts, open, high, low, close, volume)"
    ),
    workspace_id: UUID = Form(..., description="Workspace UUID"),
    base_spec_json: str = Form(..., description="Base ExecutionSpec as JSON string"),
    constraints_json: str = Form(
        ..., description="GeneratorConstraints as JSON string"
    ),
    objective: str = Form(
        default="sharpe_dd_penalty", description="Objective function"
    ),
) -> ExecuteRunPlanResponse:
    """
    Generate and execute a run plan with uploaded OHLCV data.

    **CSV Requirements:**
    - Required columns: ts, open, high, low, close, volume
    - ts should be ISO format timestamps
    - Minimum 2 bars required

    **Example curl:**
    ```bash
    curl -X POST http://localhost:8000/testing/run-plans/generate-and-execute \\
      -F "file=@BTC_1h.csv" \\
      -F "workspace_id=<uuid>" \\
      -F 'base_spec_json={"strategy_id":"breakout_52w_high","name":"Test",...}' \\
      -F 'constraints_json={"lookback_days_values":[200,252],...}' \\
      -F "objective=sharpe_dd_penalty"
    ```
    """
    log = logger.bind(
        workspace_id=str(workspace_id),
        objective=objective,
        filename=file.filename,
    )
    log.info("generate_and_execute_request")

    # Parse base_spec JSON
    try:
        base_spec_dict = json.loads(base_spec_json)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Invalid base_spec_json: {e}",
                "error_code": "INVALID_JSON",
            },
        )

    # Parse constraints JSON
    try:
        constraints_dict = json.loads(constraints_json)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Invalid constraints_json: {e}",
                "error_code": "INVALID_JSON",
            },
        )

    # Validate base_spec as ExecutionSpec
    try:
        base_spec = ExecutionSpec(**base_spec_dict)
    except Exception as e:
        log.warning("invalid_base_spec", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Invalid base_spec: {e}",
                "error_code": "INVALID_BASE_SPEC",
            },
        )

    # Validate constraints as GeneratorConstraints
    try:
        constraints = GeneratorConstraints(**constraints_dict)
        constraints.objective = objective
    except Exception as e:
        log.warning("invalid_constraints", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Invalid constraints: {e}",
                "error_code": "INVALID_CONSTRAINTS",
            },
        )

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Failed to read file: {e}",
                "error_code": "FILE_READ_ERROR",
            },
        )

    dataset_ref = file.filename or "uploaded.csv"

    # Generate run plan
    generator = TestGenerator()
    run_plan = generator.generate(
        base_spec=base_spec,
        dataset_ref=dataset_ref,
        constraints=constraints,
    )

    log.info(
        "run_plan_generated",
        run_plan_id=str(run_plan.run_plan_id),
        n_variants=run_plan.n_variants,
    )

    # Execute run plan
    events_repo = _get_events_repo()
    runner = StrategyRunner()
    orchestrator = RunOrchestrator(events_repo=events_repo, runner=runner)

    try:
        results = await orchestrator.execute(run_plan, file_content)
    except ValueError as e:
        # CSV parsing errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": str(e),
                "error_code": "INVALID_CSV",
            },
        )
    except Exception as e:
        log.error("execution_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": f"Execution failed: {e}",
                "error_code": "EXECUTION_ERROR",
            },
        )

    # Select best variant
    best_variant_id, best_score = select_best_variant(results)

    log.info(
        "execution_completed",
        run_plan_id=str(run_plan.run_plan_id),
        n_variants=len(results),
        best_variant_id=best_variant_id,
        best_score=best_score,
    )

    # Build response
    result_summaries = [
        ResultSummary(
            variant_id=r.variant_id,
            status=r.status,
            objective_score=r.objective_score,
            metrics=r.metrics,
        )
        for r in results
    ]

    return ExecuteRunPlanResponse(
        run_plan_id=run_plan.run_plan_id,
        n_variants=len(results),
        results=result_summaries,
        best_variant_id=best_variant_id,
        best_score=best_score,
    )
