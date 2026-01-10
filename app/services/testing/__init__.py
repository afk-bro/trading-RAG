# Testing package - Test generation and orchestration for trading strategies
"""
Testing package for generating and orchestrating backtest runs.

This module provides:
- RunPlan: Plan for executing a batch of backtest variants
- RunVariant: Individual variant with parameter overrides
- RunResult: Results from executing a variant
- GeneratorConstraints: Constraints for test generation
- TestGenerator: Generates RunPlan variants from a base ExecutionSpec
- RunOrchestrator: Executes RunPlan variants through StrategyRunner + PaperBroker
- VARIANT_NS: UUID namespace for variant isolation
- select_best_variant: Selects best variant with deterministic tie-breaking
"""

from app.services.testing.models import (
    canonical_json,
    hash_variant,
    apply_overrides,
    validate_variant_params,
    RunPlanStatus,
    GeneratorConstraints,
    RunVariant,
    RunPlan,
    VariantMetrics,
    RunResult,
    TESTING_VARIANT_NAMESPACE,
    get_variant_namespace,
)
from app.services.testing.test_generator import TestGenerator
from app.services.testing.run_orchestrator import (
    RunOrchestrator,
    VARIANT_NS,  # Backwards compatibility alias
    select_best_variant,
)
from app.services.testing.plan_builder import PlanBuilder

__all__ = [
    "canonical_json",
    "hash_variant",
    "apply_overrides",
    "validate_variant_params",
    "RunPlanStatus",
    "GeneratorConstraints",
    "RunVariant",
    "RunPlan",
    "VariantMetrics",
    "RunResult",
    "TestGenerator",
    "RunOrchestrator",
    "TESTING_VARIANT_NAMESPACE",
    "get_variant_namespace",
    "VARIANT_NS",  # Backwards compatibility alias
    "select_best_variant",
    "PlanBuilder",
]
